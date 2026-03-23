"""
Compute centroid-based distances (euclidean and cosine) for ELA features.

For each instance, across 30 runs:
  - Compute mean and median centroid vectors
  - Compute euclidean and cosine distance from each run to both centroids

Done for each feature group individually and for all features concatenated.

Output: h5 file matching the TLA centroid distance structure.

Usage:
  python compute_ela_centroid_dist.py /path/to/dim2feat --output-dir /path/to/output
"""

import argparse
import pickle
import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
import h5py
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_FUNCTIONS = 24
N_INSTANCES = 100
N_RUNS = 30
DIMENSION = 2

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

# Build filtered feature lists per group (applying omissions)
FILTERED_GROUPS = {}
for grp_name, feat_list in ELA_FEATURE_GROUPS.items():
    if grp_name in OMIT_GROUPS:
        continue
    filtered = [f for f in feat_list if f not in OMIT_FEATURES]
    if filtered:
        FILTERED_GROUPS[grp_name] = filtered


# ---------------------------------------------------------------------------
# Feature vector extraction
# ---------------------------------------------------------------------------

def extract_group_vectors(instance_data, group_name, feature_names):
    """
    Extract feature vectors for one group across 30 runs.

    Returns
    -------
    np.ndarray, shape (N_RUNS, len(feature_names))
    """
    vectors = np.empty((N_RUNS, len(feature_names)))
    for run_idx in range(N_RUNS):
        run_data = instance_data[group_name][run_idx]
        for feat_idx, feat_name in enumerate(feature_names):
            vectors[run_idx, feat_idx] = run_data[feat_name]
    return vectors


def extract_all_vectors(instance_data):
    """
    Extract concatenated feature vectors across all (filtered) groups.

    Returns
    -------
    np.ndarray, shape (N_RUNS, total_filtered_features)
    """
    parts = []
    for group_name, feature_names in FILTERED_GROUPS.items():
        parts.append(extract_group_vectors(instance_data, group_name, feature_names))
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Centroid distance computation
# ---------------------------------------------------------------------------

def compute_centroid_distances(vectors):
    """
    Compute centroid distances for a set of run vectors.

    Parameters
    ----------
    vectors : np.ndarray, shape (N_RUNS, n_features)

    Returns
    -------
    dict with centroid_mean, centroid_median, and distance arrays
    """
    centroid_mean = np.mean(vectors, axis=0)
    centroid_median = np.median(vectors, axis=0)

    euclidean_to_mean = np.array([
        np.linalg.norm(vectors[r] - centroid_mean) for r in range(N_RUNS)
    ])
    euclidean_to_median = np.array([
        np.linalg.norm(vectors[r] - centroid_median) for r in range(N_RUNS)
    ])

    cosine_to_mean = np.array([
        cosine_dist(vectors[r], centroid_mean)
        if np.any(vectors[r] != 0) and np.any(centroid_mean != 0)
        else np.nan
        for r in range(N_RUNS)
    ])
    cosine_to_median = np.array([
        cosine_dist(vectors[r], centroid_median)
        if np.any(vectors[r] != 0) and np.any(centroid_median != 0)
        else np.nan
        for r in range(N_RUNS)
    ])

    return {
        "centroid_mean": centroid_mean,
        "centroid_median": centroid_median,
        "euclidean_dist_to_mean": euclidean_to_mean,
        "euclidean_dist_to_median": euclidean_to_median,
        "cosine_dist_to_mean": cosine_to_mean,
        "cosine_dist_to_median": cosine_to_median,
    }


def write_distances_to_h5(h5_group, distances):
    """Write distance results to an h5 group."""
    for key, val in distances.items():
        h5_group.create_dataset(key, data=val)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    input_dir = Path(input_dir)
    output_file = Path(output_dir) / "ela_centroid_distances.h5"

    # Print feature summary
    total_features = sum(len(v) for v in FILTERED_GROUPS.values())
    print(f"Feature groups (after filtering):")
    for grp_name, feat_list in FILTERED_GROUPS.items():
        print(f"  {grp_name}: {len(feat_list)} features")
    print(f"  all: {total_features} features (concatenated)")
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

            for func_id in range(1, N_FUNCTIONS + 1):
                for inst_id in range(1, N_INSTANCES + 1):
                    instance_key = (func_id, inst_id, DIMENSION)

                    if instance_key not in data:
                        print(f"  WARNING: {instance_key} not found")
                        continue

                    instance_data = data[instance_key]
                    inst_grp = config_grp.create_group(
                        f"{func_id}_{inst_id}_{DIMENSION}"
                    )

                    # Per feature group
                    for group_name, feature_names in FILTERED_GROUPS.items():
                        vectors = extract_group_vectors(
                            instance_data, group_name, feature_names
                        )
                        distances = compute_centroid_distances(vectors)
                        grp = inst_grp.create_group(group_name)
                        write_distances_to_h5(grp, distances)

                    # All features concatenated
                    all_vectors = extract_all_vectors(instance_data)
                    all_distances = compute_centroid_distances(all_vectors)
                    all_grp = inst_grp.create_group("all")
                    write_distances_to_h5(all_grp, all_distances)

                print(f"  Function {func_id} done.")

            del data
            print(f"  Config {config_key} complete.")

    print(f"\n\nResults saved to: {output_file}")
    print(f"\nOutput h5 structure:")
    print(f"  {{config_key}}/")
    print(f"    {{func}}_{{inst}}_{{dim}}/")
    print(f"      all/")
    print(f"        centroid_mean, centroid_median")
    print(f"        cosine_dist_to_mean, cosine_dist_to_median    (30,)")
    print(f"        euclidean_dist_to_mean, euclidean_dist_to_median (30,)")
    for grp_name in FILTERED_GROUPS:
        print(f"      {grp_name}/")
        print(f"        (same datasets as all/)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute centroid-based distances for ELA features."
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