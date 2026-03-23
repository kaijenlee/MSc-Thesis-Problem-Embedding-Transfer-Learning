"""
Precompute t-SNE embeddings for ELA features.

Three modes per configuration:
  1) instance-level mean (2400 points)
  2) instance-level median (2400 points)
  3) all runs (72000 points)

Uses openTSNE for speed. Standardizes features then computes 2D embeddings.

Usage:
  python precompute_tsne_ela.py /path/to/ela/pkl/dir
  python precompute_tsne_ela.py /path/to/ela/pkl/dir --output-dir /path/to/output
  python precompute_tsne_ela.py /path/to/ela/pkl/dir --configs ilhs_50 sobol_50
  python precompute_tsne_ela.py /path/to/ela/pkl/dir --modes mean median
"""

import argparse
import pickle
import numpy as np
import h5py
import os
import gc
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from openTSNE import TSNE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_FUNCTIONS = 24
N_INSTANCES = 100
N_RUNS = 30
DIMENSION = 2

TSNE_PERPLEXITY = 40
TSNE_RANDOM_STATE = 42
TSNE_N_ITER = 1000

OMIT_FEATURES = {
    "disp.diff_median_02", "disp.ratio_median_02",
    "ela_level.lda_qda_50", "ela_level.lda_qda_25",
    "ic.eps_ratio", "disp.ratio_mean_02", "disp.diff_mean_02",
    "ela_level.lda_qda_10", "ela_meta.quad_simple.cond",
}
OMIT_GROUPS = {"levelset"}

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

FILTERED_FEATURES = []
for _grp, _feats in ELA_FEATURE_GROUPS.items():
    if _grp in OMIT_GROUPS:
        continue
    for _f in _feats:
        if _f not in OMIT_FEATURES:
            FILTERED_FEATURES.append((_grp, _f))
N_FEATURES = len(FILTERED_FEATURES)

ELA_FILES = {
    "cma_random_25": "cma_random_25_ela.pkl", "cma_random_50": "cma_random_50_ela.pkl",
    "cma_random_75": "cma_random_75_ela.pkl", "cma_random_100": "cma_random_100_ela.pkl",
    "ilhs_25": "ilhs_25_ela.pkl", "ilhs_50": "ilhs_50_ela.pkl",
    "ilhs_75": "ilhs_75_ela.pkl", "ilhs_100": "ilhs_100_ela.pkl",
    "lhs_25": "lhs_25_ela.pkl", "lhs_50": "lhs_50_ela.pkl",
    "lhs_75": "lhs_75_ela.pkl", "lhs_100": "lhs_100_ela.pkl",
    "sobol_25": "sobol_25_ela.pkl", "sobol_50": "sobol_50_ela.pkl",
    "sobol_75": "sobol_75_ela.pkl", "sobol_100": "sobol_100_ela.pkl",
    "uniform_25": "uniform_25_ela.pkl", "uniform_50": "uniform_50_ela.pkl",
    "uniform_75": "uniform_75_ela.pkl", "uniform_100": "uniform_100_ela.pkl",
}

ALL_MODES = ["mean", "median", "all_runs"]

# True labels
TRUE_LABELS = np.repeat(np.arange(N_FUNCTIONS), N_INSTANCES)
# For all_runs: each instance repeated N_RUNS times
TRUE_LABELS_ALL = np.repeat(np.arange(N_FUNCTIONS), N_INSTANCES * N_RUNS)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _build_all_runs_matrix(data):
    """
    Build (24*100*30, n_features) matrix — all runs, all instances.
    Row order: func0_inst0_run0, func0_inst0_run1, ..., func0_inst1_run0, ...
    """
    matrix = np.empty((N_FUNCTIONS * N_INSTANCES * N_RUNS, N_FEATURES))

    for func_idx in range(N_FUNCTIONS):
        func_id = func_idx + 1
        for inst_idx in range(N_INSTANCES):
            inst_id = inst_idx + 1
            instance_key = (func_id, inst_id, DIMENSION)
            instance_data = data[instance_key]

            base_row = (func_idx * N_INSTANCES + inst_idx) * N_RUNS

            for run_idx in range(N_RUNS):
                row = base_row + run_idx
                for feat_idx, (grp_name, feat_name) in enumerate(FILTERED_FEATURES):
                    matrix[row, feat_idx] = instance_data[grp_name][run_idx][feat_name]

    return matrix


def _build_collapsed_matrix(data, agg_func):
    """
    Build (2400, n_features) matrix with runs collapsed by agg_func (np.nanmean or np.nanmedian).
    """
    matrix = np.empty((N_FUNCTIONS * N_INSTANCES, N_FEATURES))

    for func_idx in range(N_FUNCTIONS):
        func_id = func_idx + 1
        for inst_idx in range(N_INSTANCES):
            inst_id = inst_idx + 1
            row = func_idx * N_INSTANCES + inst_idx
            instance_key = (func_id, inst_id, DIMENSION)
            instance_data = data[instance_key]

            for feat_idx, (grp_name, feat_name) in enumerate(FILTERED_FEATURES):
                values = [instance_data[grp_name][run][feat_name]
                          for run in range(N_RUNS)]
                matrix[row, feat_idx] = agg_func(values)

    return matrix


# ---------------------------------------------------------------------------
# t-SNE computation
# ---------------------------------------------------------------------------

def _compute_tsne(features, perplexity=TSNE_PERPLEXITY):
    """Standardize then t-SNE to 2D."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Remove constant/NaN columns
    nan_cols = np.any(~np.isfinite(X), axis=0)
    if np.any(nan_cols):
        n_removed = int(np.sum(nan_cols))
        print(f"    Removed {n_removed} non-finite columns")
        X = X[:, ~nan_cols]

    # Replace any remaining NaN with 0 (shouldn't happen after column removal)
    X = np.nan_to_num(X, nan=0.0)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=TSNE_RANDOM_STATE,
        n_iter=TSNE_N_ITER,
        initialization="pca",
        n_jobs=-1,
    )
    embedding = tsne.fit(X)
    return np.array(embedding)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir=None, configs=None, modes=None):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    input_dir = Path(input_dir)
    output_file = Path(output_dir) / "ela_tsne_embeddings.h5"
    modes = modes or ALL_MODES

    # Determine which configs to process
    if configs:
        config_keys = [c for c in configs if c in ELA_FILES]
    else:
        config_keys = list(ELA_FILES.keys())

    print(f"ELA t-SNE Preprocessing")
    print(f"Features: {N_FEATURES}")
    print(f"Modes: {modes}")
    print(f"Configs: {len(config_keys)}")
    print(f"Perplexity: {TSNE_PERPLEXITY}")
    print()

    with h5py.File(output_file, "a") as out:
        # Save labels once
        if "labels" not in out:
            out.create_dataset("labels", data=TRUE_LABELS)
        if "labels_all_runs" not in out:
            out.create_dataset("labels_all_runs", data=TRUE_LABELS_ALL)

        for config_key in config_keys:
            filepath = input_dir / ELA_FILES[config_key]
            if not filepath.exists():
                print(f"WARNING: {filepath} not found, skipping.")
                continue

            print(f"{'='*60}")
            print(f"Processing: {config_key}")
            print(f"{'='*60}")

            with open(filepath, "rb") as f:
                data = pickle.load(f)

            # Create or get config group
            if config_key in out:
                config_grp = out[config_key]
            else:
                config_grp = out.create_group(config_key)

            # Mean
            if "mean" in modes:
                if "mean" in config_grp:
                    print(f"  mean: already exists, skipping")
                else:
                    print(f"  Building mean matrix...")
                    matrix = _build_collapsed_matrix(data, np.nanmean)
                    print(f"    Shape: {matrix.shape}")
                    print(f"  Computing t-SNE (mean)...")
                    embedding = _compute_tsne(matrix)
                    config_grp.create_dataset("mean", data=embedding)
                    print(f"    Done. Embedding shape: {embedding.shape}")
                    del matrix, embedding

            # Median
            if "median" in modes:
                if "median" in config_grp:
                    print(f"  median: already exists, skipping")
                else:
                    print(f"  Building median matrix...")
                    matrix = _build_collapsed_matrix(data, np.nanmedian)
                    print(f"    Shape: {matrix.shape}")
                    print(f"  Computing t-SNE (median)...")
                    embedding = _compute_tsne(matrix)
                    config_grp.create_dataset("median", data=embedding)
                    print(f"    Done. Embedding shape: {embedding.shape}")
                    del matrix, embedding

            # All runs
            if "all_runs" in modes:
                if "all_runs" in config_grp:
                    print(f"  all_runs: already exists, skipping")
                else:
                    print(f"  Building all-runs matrix...")
                    matrix = _build_all_runs_matrix(data)
                    print(f"    Shape: {matrix.shape}")
                    print(f"  Computing t-SNE (all_runs) — this may take a while...")
                    embedding = _compute_tsne(matrix)
                    config_grp.create_dataset("all_runs", data=embedding)
                    print(f"    Done. Embedding shape: {embedding.shape}")
                    del matrix, embedding

            del data
            gc.collect()
            print()

    print(f"\nResults saved to: {output_file}")
    print(f"\nOutput h5 structure:")
    print(f"  labels              ({N_FUNCTIONS * N_INSTANCES},)")
    print(f"  labels_all_runs     ({N_FUNCTIONS * N_INSTANCES * N_RUNS},)")
    print(f"  {{config_key}}/")
    print(f"    mean              ({N_FUNCTIONS * N_INSTANCES}, 2)")
    print(f"    median            ({N_FUNCTIONS * N_INSTANCES}, 2)")
    print(f"    all_runs          ({N_FUNCTIONS * N_INSTANCES * N_RUNS}, 2)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute t-SNE embeddings for ELA features."
    )
    parser.add_argument("input_dir", type=str,
                        help="Directory containing ELA pkl files.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output h5. Defaults to input_dir.")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific configs to process (e.g. ilhs_50 sobol_50).")
    parser.add_argument("--modes", nargs="+", default=None,
                        choices=ALL_MODES,
                        help="Which modes to compute (default: all three).")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir,
         configs=args.configs, modes=args.modes)