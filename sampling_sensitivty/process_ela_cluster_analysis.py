"""
Compute ARI/NMI clustering analysis for ELA features (all groups concatenated).

For each configuration:
  - For each of 30 runs, build the 2400 x n_features matrix
  - Standardize (z-score)
  - PCA to 99% variance
  - K-means (k=24, n_init=50)
  - Compute ARI and NMI against true BBOB function class labels

Output: h5 file with 30 ARI/NMI values per configuration.

Usage:
  python compute_ela_clustering.py /path/to/dim2feat
  python compute_ela_clustering.py /path/to/dim2feat --output-dir /path/to/output
"""

import argparse
import pickle
import numpy as np
import h5py
import os
import gc
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_FUNCTIONS = 24
N_INSTANCES = 100
N_RUNS = 30
DIMENSION = 2
K_CLUSTERS = 24
N_INIT = 50
PCA_VARIANCE = 0.99
RANDOM_STATE = 42

OMIT_FEATURES = {
    "disp.diff_median_02", "disp.ratio_median_02",
    "ela_level.lda_qda_50", "ela_level.lda_qda_25",
    "ic.eps_ratio", "disp.ratio_mean_02", "disp.diff_mean_02",
    "ela_level.lda_qda_10", "ela_meta.quad_simple.cond",
}
OMIT_GROUPS = {"levelset"}

ELA_FILES = {
    "cma_random_25": "cma_random_25_ela.pkl",
    "cma_random_50": "cma_random_50_ela.pkl",
    "cma_random_75": "cma_random_75_ela.pkl",
    "cma_random_100": "cma_random_100_ela.pkl",
    "ilhs_25": "ilhs_25_ela.pkl",
    "ilhs_50": "ilhs_50_ela.pkl",
    "ilhs_75": "ilhs_75_ela.pkl",
    "ilhs_100": "ilhs_100_ela.pkl",
    "lhs_25": "lhs_25_ela.pkl",
    "lhs_50": "lhs_50_ela.pkl",
    "lhs_75": "lhs_75_ela.pkl",
    "lhs_100": "lhs_100_ela.pkl",
    "sobol_25": "sobol_25_ela.pkl",
    "sobol_50": "sobol_50_ela.pkl",
    "sobol_75": "sobol_75_ela.pkl",
    "sobol_100": "sobol_100_ela.pkl",
    "uniform_25": "uniform_25_ela.pkl",
    "uniform_50": "uniform_50_ela.pkl",
    "uniform_75": "uniform_75_ela.pkl",
    "uniform_100": "uniform_100_ela.pkl",
}

ELA_FEATURE_GROUPS = {
    "ela_dist": [
        "ela_distr.skewness", "ela_distr.kurtosis",
        "ela_distr.number_of_peaks", "ela_distr.costs_runtime",
    ],
    "meta": [
        "ela_meta.lin_simple.adj_r2", "ela_meta.lin_simple.intercept",
        "ela_meta.lin_simple.coef.min", "ela_meta.lin_simple.coef.max",
        "ela_meta.lin_simple.coef.max_by_min", "ela_meta.lin_w_interact.adj_r2",
        "ela_meta.quad_simple.adj_r2", "ela_meta.quad_simple.cond",
        "ela_meta.quad_w_interact.adj_r2", "ela_meta.costs_runtime",
    ],
    "disp": [
        "disp.ratio_mean_02", "disp.ratio_mean_05", "disp.ratio_mean_10",
        "disp.ratio_mean_25", "disp.ratio_median_02", "disp.ratio_median_05",
        "disp.ratio_median_10", "disp.ratio_median_25", "disp.diff_mean_02",
        "disp.diff_mean_05", "disp.diff_mean_10", "disp.diff_mean_25",
        "disp.diff_median_02", "disp.diff_median_05", "disp.diff_median_10",
        "disp.diff_median_25", "disp.costs_runtime",
    ],
    "nbc": [
        "nbc.nn_nb.sd_ratio", "nbc.nn_nb.mean_ratio", "nbc.nn_nb.cor",
        "nbc.dist_ratio.coeff_var", "nbc.nb_fitness.cor", "nbc.costs_runtime",
    ],
    "ic": [
        "ic.h_max", "ic.eps_s", "ic.eps_max", "ic.eps_ratio",
        "ic.m0", "ic.costs_runtime",
    ],
}

# Build filtered feature order
FILTERED_FEATURES = []  # list of (group_name, feature_name)
for grp_name, feat_list in ELA_FEATURE_GROUPS.items():
    if grp_name in OMIT_GROUPS:
        continue
    for feat_name in feat_list:
        if feat_name not in OMIT_FEATURES:
            FILTERED_FEATURES.append((grp_name, feat_name))

N_FEATURES = len(FILTERED_FEATURES)

# True labels
TRUE_LABELS = np.repeat(np.arange(N_FUNCTIONS), N_INSTANCES)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_single_run(data, run_idx):
    """
    Build feature matrix for one run from loaded pickle data.

    Parameters
    ----------
    data : dict, the loaded pickle file
    run_idx : int, which run (0-29)

    Returns
    -------
    np.ndarray, shape (2400, N_FEATURES)
    """
    features = np.empty((N_FUNCTIONS * N_INSTANCES, N_FEATURES))

    for func_idx in range(N_FUNCTIONS):
        func_id = func_idx + 1
        start_row = func_idx * N_INSTANCES

        for inst_idx in range(N_INSTANCES):
            inst_id = inst_idx + 1
            instance_key = (func_id, inst_id, DIMENSION)
            instance_data = data[instance_key]

            for feat_idx, (grp_name, feat_name) in enumerate(FILTERED_FEATURES):
                features[start_row + inst_idx, feat_idx] = (
                    instance_data[grp_name][run_idx][feat_name]
                )

    return features


# ---------------------------------------------------------------------------
# Clustering pipeline
# ---------------------------------------------------------------------------

def cluster_and_score(features, true_labels, random_state=RANDOM_STATE):
    """
    Standardize, PCA, k-means, compute ARI and NMI.
    """
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Handle NaN/Inf after scaling (constant features become NaN)
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

    # PCA to 99% variance
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    features_pca = pca.fit_transform(features_scaled)
    n_components = features_pca.shape[1]
    explained = pca.explained_variance_ratio_.sum()

    # K-means
    kmeans = KMeans(
        n_clusters=K_CLUSTERS,
        n_init=N_INIT,
        random_state=random_state,
        max_iter=300,
    )
    pred_labels = kmeans.fit_predict(features_pca)

    # Score
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    return {
        "ari": ari,
        "nmi": nmi,
        "n_pca_components": n_components,
        "explained_variance": explained,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    input_dir = Path(input_dir)
    output_file = Path(output_dir) / "ela_cluster_analysis.h5"

    print(f"Total features after filtering: {N_FEATURES}")
    print(f"Feature list:")
    for grp_name, feat_name in FILTERED_FEATURES:
        print(f"  {grp_name}/{feat_name}")
    print(f"\nClustering: k={K_CLUSTERS}, n_init={N_INIT}, PCA={PCA_VARIANCE*100:.0f}%")
    print()

    with h5py.File(output_file, "w") as out:
        for config_key, filename in ELA_FILES.items():
            filepath = input_dir / filename
            if not filepath.exists():
                print(f"WARNING: {filepath} not found, skipping.")
                continue

            print(f"\n{'='*60}")
            print(f"Processing: {filename} ({config_key})")
            print(f"{'='*60}")

            with open(filepath, "rb") as f:
                data = pickle.load(f)

            config_grp = out.create_group(config_key)

            ari_values = np.empty(N_RUNS)
            nmi_values = np.empty(N_RUNS)
            n_components_values = np.empty(N_RUNS, dtype=int)

            for run_idx in range(N_RUNS):
                features = load_single_run(data, run_idx)
                result = cluster_and_score(features, TRUE_LABELS)

                ari_values[run_idx] = result["ari"]
                nmi_values[run_idx] = result["nmi"]
                n_components_values[run_idx] = result["n_pca_components"]

                if run_idx % 10 == 0 or run_idx == N_RUNS - 1:
                    print(f"  Run {run_idx+1}/{N_RUNS}: "
                          f"ARI={result['ari']:.4f}, NMI={result['nmi']:.4f}, "
                          f"PCA components={result['n_pca_components']}")

                del features

            # Save results
            all_grp = config_grp.create_group("all")
            all_grp.create_dataset("ari", data=ari_values)
            all_grp.create_dataset("nmi", data=nmi_values)
            all_grp.create_dataset("n_pca_components", data=n_components_values)

            all_grp.attrs["ari_mean"] = float(np.nanmean(ari_values))
            all_grp.attrs["ari_std"] = float(np.nanstd(ari_values))
            all_grp.attrs["ari_median"] = float(np.nanmedian(ari_values))
            all_grp.attrs["nmi_mean"] = float(np.nanmean(nmi_values))
            all_grp.attrs["nmi_std"] = float(np.nanstd(nmi_values))
            all_grp.attrs["nmi_median"] = float(np.nanmedian(nmi_values))

            print(f"  Summary — ARI: {np.nanmean(ari_values):.4f} "
                  f"(±{np.nanstd(ari_values):.4f}), "
                  f"NMI: {np.nanmean(nmi_values):.4f} "
                  f"(±{np.nanstd(nmi_values):.4f})")

            del data
            gc.collect()

    print(f"\n\nResults saved to: {output_file}")
    print(f"\nOutput h5 structure:")
    print(f"  {{config_key}}/")
    print(f"    all/")
    print(f"      ari              (30,)")
    print(f"      nmi              (30,)")
    print(f"      n_pca_components (30,)")
    print(f"      attrs: ari_mean, ari_std, ari_median, nmi_mean, nmi_std, nmi_median")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute ARI/NMI clustering analysis for ELA features."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing the ELA pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output h5 file. Defaults to input_dir.",
    )
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir)