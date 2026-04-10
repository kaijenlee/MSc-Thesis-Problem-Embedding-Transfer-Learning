"""
Learning curve experiment for ELA features.

Varies the amount of training data along two axes:
  - Number of training instances per class (subset of CV train fold)
  - Number of runs per training instance (subset of 30)

For each (n_instances, n_runs) cell:
  - 5-fold stratified CV
  - Repeat with several random subsamples to estimate variance
  - Train RF on (n_instances * n_runs) rows per class
  - Test on the full held-out fold (median features + all 30 runs)

Output: HDF5 with per-cell, per-fold, per-repeat accuracies.

Usage:
  python classify_ela_learning_curve.py /path/to/ela/pkl/dir
  python classify_ela_learning_curve.py /path/to/ela/pkl/dir --configs ilhs_50
  python classify_ela_learning_curve.py /path/to/ela/pkl/dir --n-repeats 5
  python classify_ela_learning_curve.py /path/to/ela/pkl/dir --n-jobs -1
"""

import argparse
import pickle
import numpy as np
import h5py
import os
import gc
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_FUNCTIONS = 24
N_INSTANCES = 100
N_RUNS = 30
DIMENSION = 2
N_FOLDS = 5
RANDOM_STATE = 42

# Learning curve grid
INSTANCE_COUNTS = [5, 10, 20, 40, 80]   # per-class training instances
RUN_COUNTS = [1, 5, 10, 20, 30]          # runs per training instance

# Random Forest
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = None
RF_MAX_FEATURES = "sqrt"
RF_MIN_SAMPLES_LEAF = 1

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_instance_data(data):
    """
    Returns
    -------
    median_features  : (2400, N_FEATURES)
    all_run_features : (2400, N_RUNS, N_FEATURES)
    labels           : (2400,)
    """
    n_total = N_FUNCTIONS * N_INSTANCES
    median_features = np.empty((n_total, N_FEATURES))
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

            for feat_idx, (grp_name, feat_name) in enumerate(FILTERED_FEATURES):
                values = [instance_data[grp_name][run][feat_name]
                          for run in range(N_RUNS)]
                median_features[row, feat_idx] = np.nanmedian(values)
                for run in range(N_RUNS):
                    all_run_features[row, run, feat_idx] = values[run]

    return median_features, all_run_features, labels


# ---------------------------------------------------------------------------
# Learning curve experiment
# ---------------------------------------------------------------------------

def subsample_train_indices(train_idx, train_labels, n_inst_per_class, rng):
    """
    From the train fold, pick `n_inst_per_class` instances per class.

    Returns
    -------
    selected_idx : np.ndarray of indices into the original 2400-row arrays
    """
    selected = []
    for c in range(N_FUNCTIONS):
        class_mask = train_labels == c
        class_indices = train_idx[class_mask]
        if len(class_indices) < n_inst_per_class:
            chosen = class_indices  # use all if fewer available
        else:
            chosen = rng.choice(class_indices, size=n_inst_per_class, replace=False)
        selected.extend(chosen)
    return np.array(selected)


def run_learning_curve(median_features, all_run_features, labels,
                        instance_counts, run_counts, n_repeats, n_jobs=1):
    """
    For each (n_inst, n_runs) cell, run 5-fold CV with `n_repeats`
    random subsamples per fold.

    Returns
    -------
    dict with arrays of shape (n_inst_grid, n_runs_grid, N_FOLDS, n_repeats)
    """
    n_grid_inst = len(instance_counts)
    n_grid_runs = len(run_counts)

    acc_median = np.full((n_grid_inst, n_grid_runs, N_FOLDS, n_repeats), np.nan)
    acc_allruns = np.full((n_grid_inst, n_grid_runs, N_FOLDS, n_repeats), np.nan)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(np.zeros(len(labels)), labels))

    total_cells = n_grid_inst * n_grid_runs
    cell_count = 0

    for i_inst, n_inst in enumerate(instance_counts):
        for i_runs, n_runs in enumerate(run_counts):
            cell_count += 1
            print(f"  Cell {cell_count}/{total_cells}: "
                  f"instances={n_inst}, runs={n_runs}")

            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                y_test = labels[test_idx]
                X_test_median = median_features[test_idx]
                X_test_runs = all_run_features[test_idx]  # (n_test, N_RUNS, N_FEATURES)

                for rep in range(n_repeats):
                    rng = np.random.default_rng(
                        RANDOM_STATE + 1000 * fold_idx + rep)

                    # Subsample instances from train fold
                    train_labels = labels[train_idx]
                    sel_inst = subsample_train_indices(
                        train_idx, train_labels, n_inst, rng)

                    # Subsample runs (same indices for all selected instances)
                    sel_runs = rng.choice(N_RUNS, size=n_runs, replace=False)

                    # Build training matrix: (n_classes * n_inst * n_runs, N_FEATURES)
                    X_train = all_run_features[np.ix_(sel_inst, sel_runs)] \
                        .reshape(len(sel_inst) * n_runs, N_FEATURES)
                    y_train = np.repeat(labels[sel_inst], n_runs)

                    # Standardize
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)

                    # Train RF
                    rf = RandomForestClassifier(
                        n_estimators=RF_N_ESTIMATORS,
                        max_depth=RF_MAX_DEPTH,
                        max_features=RF_MAX_FEATURES,
                        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                        random_state=RANDOM_STATE + rep,
                        n_jobs=n_jobs,
                    )
                    rf.fit(X_train_scaled, y_train)

                    # Test on median features
                    X_test_median_scaled = np.nan_to_num(
                        scaler.transform(X_test_median), nan=0.0)
                    y_pred_median = rf.predict(X_test_median_scaled)
                    acc_median[i_inst, i_runs, fold_idx, rep] = \
                        accuracy_score(y_test, y_pred_median)

                    # Test on all 30 runs
                    n_test = len(test_idx)
                    X_test_runs_2d = X_test_runs.reshape(n_test * N_RUNS, N_FEATURES)
                    X_test_runs_scaled = np.nan_to_num(
                        scaler.transform(X_test_runs_2d), nan=0.0)
                    y_pred_runs = rf.predict(X_test_runs_scaled)
                    y_test_repeated = np.repeat(y_test, N_RUNS)
                    acc_allruns[i_inst, i_runs, fold_idx, rep] = \
                        accuracy_score(y_test_repeated, y_pred_runs)

            # Cell summary
            cell_med = acc_median[i_inst, i_runs]
            cell_run = acc_allruns[i_inst, i_runs]
            print(f"    median test: {cell_med.mean():.4f} (±{cell_med.std():.4f})")
            print(f"    runs test:   {cell_run.mean():.4f} (±{cell_run.std():.4f})")

    return {
        "instance_counts": np.array(instance_counts),
        "run_counts": np.array(run_counts),
        "acc_median": acc_median,
        "acc_allruns": acc_allruns,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir=None, configs=None, n_repeats=3, n_jobs=1,
         instance_counts=None, run_counts=None):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    if instance_counts is None:
        instance_counts = INSTANCE_COUNTS
    if run_counts is None:
        run_counts = RUN_COUNTS

    input_dir = Path(input_dir)
    output_file = Path(output_dir) / "ela_learning_curve_results.h5"

    if configs:
        config_keys = [c for c in configs if c in ELA_FILES]
    else:
        config_keys = list(ELA_FILES.keys())

    print(f"ELA Learning Curve Experiment")
    print(f"Features: {N_FEATURES}")
    print(f"Folds: {N_FOLDS}")
    print(f"Instance grid: {instance_counts}")
    print(f"Run grid:      {run_counts}")
    print(f"Repeats per cell: {n_repeats}")
    print(f"Total fits per config: "
          f"{len(instance_counts) * len(run_counts) * N_FOLDS * n_repeats}")
    print(f"Configs: {len(config_keys)}")
    print(f"n_jobs: {n_jobs}")
    print()

    with h5py.File(output_file, "a") as out:
        for config_key in config_keys:
            filepath = input_dir / ELA_FILES[config_key]
            if not filepath.exists():
                print(f"WARNING: {filepath} not found, skipping.")
                continue

            if config_key in out:
                print(f"  {config_key}: already exists, skipping")
                continue

            print(f"{'='*60}")
            print(f"Processing: {config_key}")
            print(f"{'='*60}")

            with open(filepath, "rb") as f:
                data = pickle.load(f)

            print("  Building feature matrices...")
            median_features, all_run_features, labels = build_instance_data(data)
            del data
            gc.collect()

            print("  Running learning curve experiment...")
            results = run_learning_curve(
                median_features, all_run_features, labels,
                instance_counts=instance_counts,
                run_counts=run_counts,
                n_repeats=n_repeats,
                n_jobs=n_jobs,
            )

            # Save
            grp = out.create_group(config_key)
            grp.create_dataset("instance_counts", data=results["instance_counts"])
            grp.create_dataset("run_counts", data=results["run_counts"])
            grp.create_dataset("acc_median", data=results["acc_median"])
            grp.create_dataset("acc_allruns", data=results["acc_allruns"])
            grp.attrs["n_repeats"] = n_repeats
            grp.attrs["n_folds"] = N_FOLDS

            del median_features, all_run_features
            gc.collect()
            print()

    print(f"\nResults saved to: {output_file}")
    print(f"\nOutput h5 structure:")
    print(f"  {{config_key}}/")
    print(f"    instance_counts  (n_inst_grid,)")
    print(f"    run_counts       (n_runs_grid,)")
    print(f"    acc_median       (n_inst_grid, n_runs_grid, n_folds, n_repeats)")
    print(f"    acc_allruns      (n_inst_grid, n_runs_grid, n_folds, n_repeats)")
    print(f"    attrs: n_repeats, n_folds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learning curve experiment for ELA features."
    )
    parser.add_argument("input_dir", type=str,
                        help="Directory containing ELA pkl files.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output h5. Defaults to input_dir.")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific configs to process.")
    parser.add_argument("--n-repeats", type=int, default=3,
                        help="Random subsamples per cell (default: 3).")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel jobs for Random Forest (default: 1).")
    parser.add_argument("--instance-counts", nargs="+", type=int, default=None,
                        help=f"Override instance grid (default: {INSTANCE_COUNTS}).")
    parser.add_argument("--run-counts", nargs="+", type=int, default=None,
                        help=f"Override run grid (default: {RUN_COUNTS}).")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir,
         configs=args.configs, n_repeats=args.n_repeats, n_jobs=args.n_jobs,
         instance_counts=args.instance_counts, run_counts=args.run_counts)