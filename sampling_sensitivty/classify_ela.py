"""
Classification experiment for ELA features using Random Forest.

For each configuration:
  - 5-fold stratified CV over 100 instances
  - Train on median features (collapsed across 30 runs)
  - Test on individual runs (30 predictions per test instance)
  - Record per-instance consistency and overall accuracy

Usage:
  python classify_ela.py /path/to/ela/pkl/dir
  python classify_ela.py /path/to/ela/pkl/dir --output-dir /path/to/output
  python classify_ela.py /path/to/ela/pkl/dir --configs ilhs_50 sobol_50
  python classify_ela.py /path/to/ela/pkl/dir --n-jobs -1
"""

import argparse
import pickle
import numpy as np
import h5py
import os
import gc
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_FUNCTIONS = 24
N_INSTANCES = 100
N_RUNS = 30
DIMENSION = 2
N_FOLDS = 5
RANDOM_STATE = 42

# Random Forest defaults
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
    Build per-instance data structures.

    Returns
    -------
    mean_features : np.ndarray, shape (2400, N_FEATURES)
        Mean across 30 runs per instance.
    all_run_features : np.ndarray, shape (2400, N_RUNS, N_FEATURES)
        All 30 runs per instance.
    labels : np.ndarray, shape (2400,)
        Function class labels (0-23).
    instance_indices : np.ndarray, shape (2400,)
        Instance index within class (0-99).
    """
    n_total = N_FUNCTIONS * N_INSTANCES
    mean_features = np.empty((n_total, N_FEATURES))
    all_run_features = np.empty((n_total, N_RUNS, N_FEATURES))
    labels = np.empty(n_total, dtype=int)
    instance_indices = np.empty(n_total, dtype=int)

    for func_idx in range(N_FUNCTIONS):
        func_id = func_idx + 1
        for inst_idx in range(N_INSTANCES):
            inst_id = inst_idx + 1
            row = func_idx * N_INSTANCES + inst_idx
            instance_key = (func_id, inst_id, DIMENSION)
            instance_data = data[instance_key]

            labels[row] = func_idx
            instance_indices[row] = inst_idx

            for feat_idx, (grp_name, feat_name) in enumerate(FILTERED_FEATURES):
                values = [instance_data[grp_name][run][feat_name]
                          for run in range(N_RUNS)]
                mean_features[row, feat_idx] = np.nanmedian(values)
                for run in range(N_RUNS):
                    all_run_features[row, run, feat_idx] = values[run]

    return mean_features, all_run_features, labels, instance_indices


# ---------------------------------------------------------------------------
# Classification experiment
# ---------------------------------------------------------------------------

def run_classification(mean_features, all_run_features, labels, n_jobs=1):
    """
    Run 5-fold stratified CV.

    Train on median features, test on individual runs.

    Returns dict with results.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Per-fold results
    fold_accuracies_mean = []       # accuracy on mean test features
    fold_accuracies_all_runs = []   # accuracy on all individual runs
    fold_per_instance_consistency = []  # fraction of 30 runs correct per instance

    # Collect all predictions for overall metrics
    all_true_labels = []
    all_pred_mean = []
    all_pred_runs = []  # (n_test_instances, N_RUNS)
    all_consistency = []  # per-instance consistency scores

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(mean_features, labels)):
        print(f"    Fold {fold_idx + 1}/{N_FOLDS}...")

        # Train data: mean features
        X_train = mean_features[train_idx]
        y_train = labels[train_idx]

        # Standardize based on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Handle NaN
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)

        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            max_features=RF_MAX_FEATURES,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
        )
        rf.fit(X_train_scaled, y_train)

        # Test on mean features
        X_test_mean = mean_features[test_idx]
        X_test_mean_scaled = scaler.transform(X_test_mean)
        X_test_mean_scaled = np.nan_to_num(X_test_mean_scaled, nan=0.0)
        y_test = labels[test_idx]
        y_pred_mean = rf.predict(X_test_mean_scaled)

        acc_mean = accuracy_score(y_test, y_pred_mean)
        fold_accuracies_mean.append(acc_mean)
        all_true_labels.extend(y_test)
        all_pred_mean.extend(y_pred_mean)

        # Test on individual runs
        instance_consistencies = []
        instance_run_preds = []

        for i, test_row in enumerate(test_idx):
            run_preds = []
            for run_idx in range(N_RUNS):
                X_run = all_run_features[test_row, run_idx, :].reshape(1, -1)
                X_run_scaled = scaler.transform(X_run)
                X_run_scaled = np.nan_to_num(X_run_scaled, nan=0.0)
                pred = rf.predict(X_run_scaled)[0]
                run_preds.append(pred)

            run_preds = np.array(run_preds)
            consistency = np.mean(run_preds == y_test[i])
            instance_consistencies.append(consistency)
            instance_run_preds.append(run_preds)

        # Per-run accuracy (flatten all runs)
        all_run_preds_flat = np.array(instance_run_preds).flatten()
        y_test_repeated = np.repeat(y_test, N_RUNS)
        acc_all_runs = accuracy_score(y_test_repeated, all_run_preds_flat)

        fold_accuracies_all_runs.append(acc_all_runs)
        fold_per_instance_consistency.append(np.mean(instance_consistencies))
        all_consistency.extend(instance_consistencies)
        all_pred_runs.extend(instance_run_preds)

        print(f"      Mean-test accuracy: {acc_mean:.4f}")
        print(f"      All-runs accuracy:  {acc_all_runs:.4f}")
        print(f"      Mean consistency:   {np.mean(instance_consistencies):.4f}")

    results = {
        "fold_accuracies_mean": np.array(fold_accuracies_mean),
        "fold_accuracies_all_runs": np.array(fold_accuracies_all_runs),
        "fold_consistency": np.array(fold_per_instance_consistency),
        "overall_accuracy_mean": accuracy_score(all_true_labels, all_pred_mean),
        "overall_accuracy_all_runs": np.mean(fold_accuracies_all_runs),
        "overall_consistency_mean": np.mean(all_consistency),
        "overall_consistency_std": np.std(all_consistency),
        "per_instance_consistency": np.array(all_consistency),
    }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir=None, configs=None, n_jobs=1):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    input_dir = Path(input_dir)
    output_file = Path(output_dir) / "ela_classification_results.h5"

    if configs:
        config_keys = [c for c in configs if c in ELA_FILES]
    else:
        config_keys = list(ELA_FILES.keys())

    print(f"ELA Classification Experiment")
    print(f"Features: {N_FEATURES}")
    print(f"Folds: {N_FOLDS}")
    print(f"RF: n_estimators={RF_N_ESTIMATORS}, max_depth={RF_MAX_DEPTH}, "
          f"max_features={RF_MAX_FEATURES}")
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
            mean_features, all_run_features, labels, _ = build_instance_data(data)
            del data

            print("  Running classification...")
            results = run_classification(mean_features, all_run_features, labels,
                                         n_jobs=n_jobs)

            # Save results
            config_grp = out.create_group(config_key)

            config_grp.create_dataset("fold_accuracies_mean",
                                       data=results["fold_accuracies_mean"])
            config_grp.create_dataset("fold_accuracies_all_runs",
                                       data=results["fold_accuracies_all_runs"])
            config_grp.create_dataset("fold_consistency",
                                       data=results["fold_consistency"])
            config_grp.create_dataset("per_instance_consistency",
                                       data=results["per_instance_consistency"])

            config_grp.attrs["overall_accuracy_mean"] = results["overall_accuracy_mean"]
            config_grp.attrs["overall_accuracy_all_runs"] = results["overall_accuracy_all_runs"]
            config_grp.attrs["overall_consistency_mean"] = results["overall_consistency_mean"]
            config_grp.attrs["overall_consistency_std"] = results["overall_consistency_std"]

            print(f"\n  Results:")
            print(f"    Mean-test accuracy:  {results['overall_accuracy_mean']:.4f}")
            print(f"    All-runs accuracy:   {results['overall_accuracy_all_runs']:.4f}")
            print(f"    Consistency (mean):  {results['overall_consistency_mean']:.4f} "
                  f"(±{results['overall_consistency_std']:.4f})")
            print(f"    Fold accuracies (mean-test): {results['fold_accuracies_mean']}")
            print(f"    Fold accuracies (all-runs):  {results['fold_accuracies_all_runs']}")
            print()

            del mean_features, all_run_features
            gc.collect()

    print(f"\nResults saved to: {output_file}")
    print(f"\nOutput h5 structure:")
    print(f"  {{config_key}}/")
    print(f"    fold_accuracies_mean       ({N_FOLDS},)")
    print(f"    fold_accuracies_all_runs   ({N_FOLDS},)")
    print(f"    fold_consistency           ({N_FOLDS},)")
    print(f"    per_instance_consistency   (n_instances,)")
    print(f"    attrs: overall_accuracy_mean, overall_accuracy_all_runs,")
    print(f"           overall_consistency_mean, overall_consistency_std")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classification experiment for ELA features."
    )
    parser.add_argument("input_dir", type=str,
                        help="Directory containing ELA pkl files.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output h5. Defaults to input_dir.")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific configs to process.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel jobs for Random Forest (default: 1).")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir,
         configs=args.configs, n_jobs=args.n_jobs)