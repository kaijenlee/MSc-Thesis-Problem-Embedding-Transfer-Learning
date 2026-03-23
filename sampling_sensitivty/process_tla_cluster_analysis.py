"""
Compute ARI/NMI clustering analysis for TLA features.

For each configuration and feature segment:
  - Load all 30 runs at once for the segment
  - For each run in parallel: standardize, PCA 99%, k-means (k=24, n_init=50)
  - Compute ARI and NMI against true BBOB function class labels

Parallelized across runs using joblib.

Usage:
  python compute_tla_clustering.py /path/to/tla/h5files
  python compute_tla_clustering.py /path/to/tla/h5files --output-dir /path/to/output
  python compute_tla_clustering.py /path/to/tla/h5files --n-jobs 4
"""

import argparse
import numpy as np
import h5py
import os
import gc
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FILES = [
    "ilhs_10_tla.h5", "ilhs_25_tla.h5", "ilhs_50_tla.h5", "ilhs_75_tla.h5", "ilhs_100_tla.h5",
    "lhs_10_tla.h5", "lhs_25_tla.h5", "lhs_50_tla.h5", "lhs_75_tla.h5", "lhs_100_tla.h5",
    "sobol_10_tla.h5", "sobol_25_tla.h5", "sobol_50_tla.h5", "sobol_75_tla.h5", "sobol_100_tla.h5",
    "uniform_10_tla.h5", "uniform_25_tla.h5", "uniform_50_tla.h5", "uniform_75_tla.h5", "uniform_100_tla.h5",
    "cma_random_10_tla.h5", "cma_random_25_tla.h5", "cma_random_50_tla.h5", "cma_random_75_tla.h5", "cma_random_100_tla.h5",
]

N_FUNCTIONS = 24
N_INSTANCES = 100
N_RUNS = 30
DIMENSION = 2
K_CLUSTERS = 24
N_INIT = 50
PCA_VARIANCE = 0.99
RANDOM_STATE = 42

PERSPECTIVES_LIST = ["volume", "axis"]
HOMOLOGIES = ["h0", "h1", "h2"]
FEATURE_LENGTHS = {"h0": 100, "h1": 10000, "h2": 10000}
TOTAL_FEATURES = 2 * (100 + 10000 + 10000)

SEGMENTS = [
    ("all", None, None),
    ("volume_all", "volume", None),
    ("axis_all", "axis", None),
    ("volume_h0", "volume", "h0"),
    ("volume_h1", "volume", "h1"),
    ("volume_h2", "volume", "h2"),
    ("axis_h0", "axis", "h0"),
    ("axis_h1", "axis", "h1"),
    ("axis_h2", "axis", "h2"),
]

TRUE_LABELS = np.repeat(np.arange(N_FUNCTIONS), N_INSTANCES)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_segment_spec(perspective_filter, homology_filter):
    specs = []
    for persp in PERSPECTIVES_LIST:
        if perspective_filter is not None and persp != perspective_filter:
            continue
        for hom in HOMOLOGIES:
            if homology_filter is not None and hom != homology_filter:
                continue
            specs.append((persp, hom))
    return specs


def get_segment_length(perspective_filter, homology_filter):
    total = 0
    for persp, hom in get_segment_spec(perspective_filter, homology_filter):
        total += FEATURE_LENGTHS[hom]
    return total


def load_segment_all_runs(h5_path, perspective_filter, homology_filter):
    """
    Load all 30 runs for one segment, all 2400 instances at once.

    Returns
    -------
    np.ndarray, shape (N_RUNS, 2400, segment_length)
    """
    specs = get_segment_spec(perspective_filter, homology_filter)
    seg_len = sum(FEATURE_LENGTHS[hom] for _, hom in specs)
    data = np.empty((N_RUNS, N_FUNCTIONS * N_INSTANCES, seg_len))

    with h5py.File(h5_path, "r") as f:
        for func_idx in range(N_FUNCTIONS):
            func_id = func_idx + 1
            start_row = func_idx * N_INSTANCES

            for inst_idx in range(N_INSTANCES):
                inst_id = inst_idx + 1
                key = f"{func_id}_{inst_id}_{DIMENSION}"
                group = f[key]

                offset = 0
                for persp, hom in specs:
                    feat_len = FEATURE_LENGTHS[hom]
                    arr = group[persp][hom][:]
                    arr = arr.reshape(arr.shape[0], -1)  # (N_RUNS, feat_len)
                    data[:, start_row + inst_idx, offset:offset + feat_len] = arr
                    offset += feat_len

    return data


# ---------------------------------------------------------------------------
# Clustering pipeline
# ---------------------------------------------------------------------------

def cluster_and_score(features, true_labels, random_state=RANDOM_STATE):
    """
    Standardize, PCA, k-means, compute ARI and NMI.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    nan_cols = np.any(~np.isfinite(features_scaled), axis=0)
    if np.any(nan_cols):
        features_scaled = features_scaled[:, ~nan_cols]

    if features_scaled.shape[1] == 0:
        return {
            "ari": np.nan,
            "nmi": np.nan,
            "n_pca_components": 0,
            "explained_variance": 0.0,
        }

    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    features_pca = pca.fit_transform(features_scaled)
    n_components = features_pca.shape[1]
    explained = pca.explained_variance_ratio_.sum()

    kmeans = KMeans(
        n_clusters=K_CLUSTERS,
        n_init=N_INIT,
        random_state=random_state,
        max_iter=300,
    )
    pred_labels = kmeans.fit_predict(features_pca)

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    return {
        "ari": ari,
        "nmi": nmi,
        "n_pca_components": n_components,
        "explained_variance": explained,
    }


def _process_single_run(run_data, true_labels):
    """Wrapper for parallel execution of a single run."""
    return cluster_and_score(run_data, true_labels)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_filename(filename):
    parts = filename.replace("_tla.h5", "").rsplit("_", 1)
    return parts[0], parts[1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir=None, n_jobs=-1):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "tla_cluster_analysis.h5")

    print("Segments to evaluate:")
    for seg_name, persp_f, hom_f in SEGMENTS:
        seg_len = get_segment_length(persp_f, hom_f)
        print(f"  {seg_name}: {seg_len} features")
    print(f"\nClustering: k={K_CLUSTERS}, n_init={N_INIT}, PCA={PCA_VARIANCE*100:.0f}%")
    print(f"Per-run matrix: {N_FUNCTIONS * N_INSTANCES} instances")
    print(f"Parallel jobs: {n_jobs}")
    print()

    with h5py.File(output_file, "w") as out:
        for filename in FILES:
            filepath = os.path.join(input_dir, filename)
            if not os.path.exists(filepath):
                print(f"WARNING: {filepath} not found, skipping.")
                continue

            sampling_strategy, sample_size = parse_filename(filename)
            config_key = f"{sampling_strategy}_{sample_size}"

            print(f"\n{'='*60}")
            print(f"Processing: {filename} ({config_key})")
            print(f"{'='*60}")

            config_grp = out.create_group(config_key)
            config_grp.attrs["sampling_strategy"] = sampling_strategy
            config_grp.attrs["sample_size"] = int(sample_size)

            for seg_name, persp_f, hom_f in SEGMENTS:
                seg_len = get_segment_length(persp_f, hom_f)
                mem_gb = N_RUNS * N_FUNCTIONS * N_INSTANCES * seg_len * 8 / (1024**3)
                print(f"\n  Segment: {seg_name} ({seg_len} features, ~{mem_gb:.1f} GB)")

                # Load all 30 runs at once
                print(f"    Loading all runs...")
                all_runs_data = load_segment_all_runs(filepath, persp_f, hom_f)
                print(f"    Data shape: {all_runs_data.shape}")

                # Process all 30 runs in parallel
                print(f"    Clustering {N_RUNS} runs in parallel (n_jobs={n_jobs})...")
                results = Parallel(n_jobs=n_jobs, verbose=1)(
                    delayed(_process_single_run)(all_runs_data[run_idx], TRUE_LABELS)
                    for run_idx in range(N_RUNS)
                )

                del all_runs_data
                gc.collect()

                # Collect results
                ari_values = np.array([r["ari"] for r in results])
                nmi_values = np.array([r["nmi"] for r in results])
                n_components_values = np.array([r["n_pca_components"] for r in results],
                                               dtype=int)

                # Save results
                seg_grp = config_grp.create_group(seg_name)
                seg_grp.create_dataset("ari", data=ari_values)
                seg_grp.create_dataset("nmi", data=nmi_values)
                seg_grp.create_dataset("n_pca_components", data=n_components_values)

                seg_grp.attrs["ari_mean"] = float(np.nanmean(ari_values))
                seg_grp.attrs["ari_std"] = float(np.nanstd(ari_values))
                seg_grp.attrs["ari_median"] = float(np.nanmedian(ari_values))
                seg_grp.attrs["nmi_mean"] = float(np.nanmean(nmi_values))
                seg_grp.attrs["nmi_std"] = float(np.nanstd(nmi_values))
                seg_grp.attrs["nmi_median"] = float(np.nanmedian(nmi_values))

                print(f"    Summary — ARI: {np.nanmean(ari_values):.4f} "
                      f"(±{np.nanstd(ari_values):.4f}), "
                      f"NMI: {np.nanmean(nmi_values):.4f} "
                      f"(±{np.nanstd(nmi_values):.4f})")

                # Print a few individual results
                for i in [0, 14, 29]:
                    print(f"      Run {i+1}: ARI={ari_values[i]:.4f}, "
                          f"NMI={nmi_values[i]:.4f}, "
                          f"PCA={n_components_values[i]}")

            gc.collect()

    print(f"\n\nResults saved to: {output_file}")
    print(f"\nOutput h5 structure:")
    print(f"  {{config_key}}/")
    print(f"    attrs: sampling_strategy, sample_size")
    for seg_name, _, _ in SEGMENTS:
        print(f"    {seg_name}/")
        print(f"      ari                (30,)")
        print(f"      nmi                (30,)")
        print(f"      n_pca_components   (30,)")
        print(f"      attrs: ari_mean, ari_std, ari_median, nmi_mean, nmi_std, nmi_median")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute ARI/NMI clustering analysis for TLA features."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing the TLA h5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output h5 file. Defaults to input_dir.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs. -1 = all cores (default).",
    )
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir, n_jobs=args.n_jobs)