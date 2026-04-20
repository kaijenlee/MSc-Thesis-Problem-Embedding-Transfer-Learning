"""
Compute per-feature CV from classification training sets — 2 variants.

Reads the raw ELA pkl files and the classification HDF5 results (which
record fold splits and sampled run indices) and produces two types of CV:

per_fold_per_function:
    For each (fold, function), pool its ~80 training instances × n_runs_train
    sampled runs → one CV per feature.
    Shape: results[...]["per_fold_per_function"][fold_idx][func_id] = {group: {feat: cv}}

across_fold_per_function (deduplicated):
    For each function, collect the union of (instance, sampled_runs) across
    all folds where the instance appeared in training. Each instance
    contributes once with the union of its sampled runs across folds.
    → one CV per feature per function.
    Shape: results[...]["across_fold_per_function"][func_id] = {group: {feat: cv}}

Output format (pkl):
    results[config_key][n_runs_train]["per_fold_per_function"][fold_idx][func_id][group][feat] = cv
    results[config_key][n_runs_train]["across_fold_per_function"][func_id][group][feat] = cv

Usage:
  python compute_ela_cv_training_set.py /path/to/ela/pkl/dir /path/to/results.h5
  python compute_ela_cv_training_set.py /path/to/ela/pkl/dir /path/to/results.h5 \
      --output-dir /path/to/output
  python compute_ela_cv_training_set.py /path/to/ela/pkl/dir /path/to/results.h5 \
      --configs ilhs_50 sobol_50
  python compute_ela_cv_training_set.py /path/to/ela/pkl/dir /path/to/results.h5 \
      --n-runs-train 1 5 10 15 30
"""

import argparse
import pickle
import numpy as np
import h5py
import os
from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Configuration (must match classify_ela_allruns.py)
# ---------------------------------------------------------------------------

N_FUNCTIONS = 24
N_INSTANCES = 100
N_RUNS = 30
DIMENSION = 2
N_FOLDS = 5
RANDOM_STATE = 42

DEFAULT_N_RUNS_TRAIN = [1, 5, 10, 15, 30]

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

# Build filtered feature list (same order as classification script)
FILTERED_FEATURES = []
for _grp, _feats in ELA_FEATURE_GROUPS.items():
    if _grp in OMIT_GROUPS:
        continue
    for _f in _feats:
        if _f not in OMIT_FEATURES:
            FILTERED_FEATURES.append((_grp, _f))
N_FEATURES = len(FILTERED_FEATURES)

ELA_FILES = {
    "cma_random_10": "cma_random_10_ela.pkl",
    "cma_random_25": "cma_random_25_ela.pkl", "cma_random_50": "cma_random_50_ela.pkl",
    "cma_random_75": "cma_random_75_ela.pkl", "cma_random_100": "cma_random_100_ela.pkl",
    "ilhs_10": "ilhs_10_ela.pkl",
    "ilhs_25": "ilhs_25_ela.pkl", "ilhs_50": "ilhs_50_ela.pkl",
    "ilhs_75": "ilhs_75_ela.pkl", "ilhs_100": "ilhs_100_ela.pkl",
    "lhs_10": "lhs_10_ela.pkl",
    "lhs_25": "lhs_25_ela.pkl", "lhs_50": "lhs_50_ela.pkl",
    "lhs_75": "lhs_75_ela.pkl", "lhs_100": "lhs_100_ela.pkl",
    "sobol_10": "sobol_10_ela.pkl",
    "sobol_25": "sobol_25_ela.pkl", "sobol_50": "sobol_50_ela.pkl",
    "sobol_75": "sobol_75_ela.pkl", "sobol_100": "sobol_100_ela.pkl",
    "uniform_10": "uniform_10_ela.pkl",
    "uniform_25": "uniform_25_ela.pkl", "uniform_50": "uniform_50_ela.pkl",
    "uniform_75": "uniform_75_ela.pkl", "uniform_100": "uniform_100_ela.pkl",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_cv(values):
    """CV = std / |mean|. Returns np.nan if mean==0 or any NaN present."""
    values = np.array(values, dtype=float)
    if np.any(np.isnan(values)):
        return np.nan
    mean = np.mean(values)
    if mean == 0:
        return np.nan
    return np.std(values, ddof=0) / abs(mean)


def cv_dict_from_pool(pool):
    """
    Compute CV per feature from a pool array of shape (n_samples, N_FEATURES).
    Returns nested dict {group: {feat: cv}}.
    """
    result = {}
    for feat_idx, (grp_name, feat_name) in enumerate(FILTERED_FEATURES):
        if grp_name not in result:
            result[grp_name] = {}
        result[grp_name][feat_name] = compute_cv(pool[:, feat_idx])
    return result


def row_to_func_id(row):
    """Convert flat row index (0-2399) to function id (1-24)."""
    return row // N_INSTANCES + 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_all_run_features(data):
    """
    Build per-instance, per-run feature matrix.

    Returns
    -------
    all_run_features : np.ndarray, shape (2400, N_RUNS, N_FEATURES)
    labels : np.ndarray, shape (2400,)
    """
    n_total = N_FUNCTIONS * N_INSTANCES
    all_run_features = np.empty((n_total, N_RUNS, N_FEATURES))
    labels = np.empty(n_total, dtype=int)

    for func_idx in range(N_FUNCTIONS):
        func_id = func_idx + 1
        for inst_idx in range(N_INSTANCES):
            inst_id = inst_idx + 1
            row = func_idx * N_INSTANCES + inst_idx
            instance_key = (func_id, inst_id, DIMENSION)
            instance_data = data[instance_key]

            labels[row] = func_idx

            for feat_idx, (grp_name, feat_name) in enumerate(
                    FILTERED_FEATURES):
                for run in range(N_RUNS):
                    all_run_features[row, run, feat_idx] = \
                        instance_data[grp_name][run][feat_name]

    return all_run_features, labels


def load_sampled_runs_from_h5(h5_file, config_key, n_runs_train, fold_idx):
    """
    Read the sampled run indices for a specific fold from the HDF5 results.

    Returns
    -------
    sampled_runs : np.ndarray, shape (n_train_instances, n_runs_train)
    """
    subkey = f"n_runs_{n_runs_train:02d}"
    grp = h5_file[config_key][subkey]

    if "sampled_runs" in grp:
        ds = grp["sampled_runs"]
        if isinstance(ds, h5py.Dataset):
            return ds[fold_idx]
        elif isinstance(ds, h5py.Group):
            return np.array(ds[f"fold_{fold_idx}"])
    raise KeyError(f"sampled_runs not found in {config_key}/{subkey}")


def get_fold_splits(labels):
    """Reproduce the exact fold splits from the classification script."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
    return list(skf.split(np.zeros(len(labels)), labels))


# ---------------------------------------------------------------------------
# Extract training data for one fold
# ---------------------------------------------------------------------------

def extract_fold_training_data(all_run_features, train_idx, sampled_runs):
    """
    Extract the training feature values for one fold.

    Parameters
    ----------
    all_run_features : (2400, N_RUNS, N_FEATURES)
    train_idx : array of row indices in training set
    sampled_runs : (n_train, n_runs_train) — run indices per instance

    Returns
    -------
    X_train : (n_train, n_runs_train, N_FEATURES)
    train_func_ids : (n_train,) — function id (1-24) for each training instance
    """
    n_train = len(train_idx)
    row_idx = np.arange(n_train)[:, None]
    X_train = all_run_features[train_idx][row_idx, sampled_runs, :]
    train_func_ids = np.array([row_to_func_id(r) for r in train_idx])
    return X_train, train_func_ids


# ---------------------------------------------------------------------------
# CV computation
# ---------------------------------------------------------------------------

def compute_per_fold_per_function(X_train, train_func_ids):
    """
    Per fold, per function.
    For each function, pool its training instances × n_runs_train.

    Returns {func_id: {group: {feat: cv}}}
    """
    result = {}
    for func_id in range(1, N_FUNCTIONS + 1):
        mask = train_func_ids == func_id
        if not np.any(mask):
            continue
        pool = X_train[mask].reshape(-1, N_FEATURES)
        result[func_id] = cv_dict_from_pool(pool)
    return result


def compute_across_fold_per_function(all_run_features, fold_splits, sampled_runs_all_folds):
    """
    Across folds, deduplicated, per function.

    For each instance (row), take the union of sampled run indices across
    all folds where it appeared in the training set. Each instance
    contributes once with its deduplicated runs.

    Returns
    -------
    type3 : {func_id: {group: {feat: cv}}}
    """
    # For each row, collect the union of sampled runs across folds
    instance_runs = defaultdict(set)

    for fold_idx, (train_idx, _) in enumerate(fold_splits):
        sampled_runs = sampled_runs_all_folds[fold_idx]
        for local_i, global_row in enumerate(train_idx):
            runs = sampled_runs[local_i]
            instance_runs[global_row].update(runs.tolist())

    # Build per-function pools
    func_pools = defaultdict(list)

    for row in sorted(instance_runs.keys()):
        func_id = row_to_func_id(row)
        run_indices = sorted(instance_runs[row])
        vectors = all_run_features[row, run_indices, :]
        func_pools[func_id].append(vectors)

    result = {}
    for func_id in range(1, N_FUNCTIONS + 1):
        if func_id not in func_pools:
            continue
        pool = np.vstack(func_pools[func_id])
        result[func_id] = cv_dict_from_pool(pool)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, h5_path, output_dir=None, configs=None,
         n_runs_train_list=None):
    if output_dir is None:
        output_dir = os.path.dirname(h5_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    if n_runs_train_list is None:
        n_runs_train_list = DEFAULT_N_RUNS_TRAIN

    input_dir = Path(input_dir)
    output_file = Path(output_dir) / "ela_cv_training_set.pkl"

    h5_file = h5py.File(h5_path, "r")

    if configs:
        config_keys = [c for c in configs if c in ELA_FILES]
    else:
        config_keys = [k for k in h5_file.keys() if k in ELA_FILES]

    print(f"Compute per-feature CV from classification training sets")
    print(f"Features: {N_FEATURES}")
    print(f"Folds: {N_FOLDS}")
    print(f"n_runs_train values: {n_runs_train_list}")
    print(f"Configs: {config_keys}")
    print(f"CV types:")
    print(f"  per_fold_per_function: per fold, per function")
    print(f"  across_fold_per_function: across folds (dedup), per function")
    print()

    results = {}

    for config_key in config_keys:
        filepath = input_dir / ELA_FILES[config_key]
        if not filepath.exists():
            print(f"WARNING: {filepath} not found, skipping.")
            continue

        if config_key not in h5_file:
            print(f"WARNING: {config_key} not in HDF5 results, skipping.")
            continue

        print(f"{'='*60}")
        print(f"Processing: {config_key}")
        print(f"{'='*60}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        print("  Building feature matrices...")
        all_run_features, labels = build_all_run_features(data)
        del data

        fold_splits = get_fold_splits(labels)

        config_results = {}

        for n_runs_train in n_runs_train_list:
            subkey = f"n_runs_{n_runs_train:02d}"
            if subkey not in h5_file[config_key]:
                print(f"  WARNING: {subkey} not in HDF5 for {config_key}, "
                      f"skipping.")
                continue

            print(f"\n  --- n_runs_train = {n_runs_train} ---")

            # Load sampled runs for all folds
            sampled_runs_all_folds = []
            for fold_idx in range(N_FOLDS):
                sr = load_sampled_runs_from_h5(
                    h5_file, config_key, n_runs_train, fold_idx)
                sampled_runs_all_folds.append(sr)

            # ── per_fold_per_function ─────────────────────────────────
            pfpf_results = {}

            for fold_idx, (train_idx, _) in enumerate(fold_splits):
                print(f"    Fold {fold_idx + 1}/{N_FOLDS}...")
                sampled_runs = sampled_runs_all_folds[fold_idx]

                X_train, train_func_ids = extract_fold_training_data(
                    all_run_features, train_idx, sampled_runs)

                pfpf = compute_per_fold_per_function(X_train, train_func_ids)
                pfpf_results[fold_idx] = pfpf

            # ── across_fold_per_function (deduplicated) ──────────────────
            print(f"    Computing across_fold_per_function (deduplicated)...")
            afpf_results = compute_across_fold_per_function(
                all_run_features, fold_splits, sampled_runs_all_folds)

            # Summary
            all_cvs = []
            for func_dict in afpf_results.values():
                for grp_dict in func_dict.values():
                    all_cvs.extend(v for v in grp_dict.values()
                                   if not np.isnan(v))
            if all_cvs:
                print(f"      across_fold_per_function — Median CV: "
                      f"{np.median(all_cvs):.4f}, "
                      f"Mean CV: {np.mean(all_cvs):.4f}")

            config_results[n_runs_train] = {
                "per_fold_per_function": pfpf_results,
                "across_fold_per_function": afpf_results,
            }

        results[config_key] = config_results
        del all_run_features
        print()

    h5_file.close()

    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    print(f"\nResults saved to: {output_file}")
    print(f"\nAccess format:")
    print(f"  per_fold_per_function:")
    print(f"    results[config][n_runs]['per_fold_per_function'][fold][func_id][group][feat]")
    print(f"  across_fold_per_function (deduplicated):")
    print(f"    results[config][n_runs]['across_fold_per_function'][func_id][group][feat]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute 2 types of per-feature CV from classification "
                    "training sets.",
    )
    parser.add_argument(
        "input_dir", type=str,
        help="Directory containing the ELA pkl files.",
    )
    parser.add_argument(
        "h5_path", type=str,
        help="Path to the classification results HDF5 file "
             "(ela_classification_results_allruns.h5).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output pkl. Defaults to directory of h5_path.",
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Specific configs to process.",
    )
    parser.add_argument(
        "--n-runs-train", nargs="+", type=int, default=None,
        help=f"Values of n_runs_train to process "
             f"(default: {DEFAULT_N_RUNS_TRAIN}).",
    )
    args = parser.parse_args()
    main(input_dir=args.input_dir, h5_path=args.h5_path,
         output_dir=args.output_dir, configs=args.configs,
         n_runs_train_list=args.n_runs_train)