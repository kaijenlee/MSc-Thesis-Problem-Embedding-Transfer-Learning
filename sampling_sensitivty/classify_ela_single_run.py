"""
Classification experiment for ELA features using Random Forest.
Variant: train on SINGLE RUN per instance (30 separate models).

For each configuration:
  - 5-fold stratified CV over 100 instances (split by instance)
  - For each of the 30 runs, train a separate RF on that run's features
    (one row per instance)
  - Each model is tested on:
      (a) median features of test instances
      (b) all 30 individual runs of test instances
  - Record per-run-model accuracies, consistency, and predictions

Usage:
  python classify_ela_singlerun.py /path/to/ela/pkl/dir
  python classify_ela_singlerun.py /path/to/ela/pkl/dir --output-dir /path/to/output
  python classify_ela_singlerun.py /path/to/ela/pkl/dir --configs ilhs_50 sobol_50
  python classify_ela_singlerun.py /path/to/ela/pkl/dir --n-jobs -1
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
from scipy.stats import mode

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
    median_features : np.ndarray, shape (2400, N_FEATURES)
        Median across 30 runs per instance.
    all_run_features : np.ndarray, shape (2400, N_RUNS, N_FEATURES)
        All 30 runs per instance.
    labels : np.ndarray, shape (2400,)
        Function class labels (0-23).
    instance_indices : np.ndarray, shape (2400,)
        Instance index within class (0-99).
    """
    n_total = N_FUNCTIONS * N_INSTANCES
    median_features = np.empty((n_total, N_FEATURES))
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
                median_features[row, feat_idx] = np.nanmedian(values)
                for run in range(N_RUNS):
                    all_run_features[row, run, feat_idx] = values[run]

    return median_features, all_run_features, labels, instance_indices


# ---------------------------------------------------------------------------
# Classification experiment
# ---------------------------------------------------------------------------

def run_classification(median_features, all_run_features, labels, n_jobs=1):
    """
    Run 5-fold stratified CV with 30 models (one per run).

    For each fold and each training run index (0-29):
      - Train RF on that single run's features for train instances
      - Test on median features and on all 30 runs of test instances

    Returns dict with results.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_total = len(labels)

    # Prediction arrays across all folds (filled in fold order)
    # pred_median_by_model[i, r] = prediction for instance i by model trained on run r
    pred_median_by_model = np.empty((n_total, N_RUNS), dtype=int)
    # pred_runs_by_model[i, r_train, r_test] = prediction for instance i,
    #   model trained on run r_train, tested on run r_test
    pred_runs_by_model = np.empty((n_total, N_RUNS, N_RUNS), dtype=int)
    true_labels_ordered = np.empty(n_total, dtype=int)

    # Per-fold, per-run-model accuracies
    # fold_acc_median[fold, run_model] = accuracy on median test features
    fold_acc_median = np.empty((N_FOLDS, N_RUNS))
    # fold_acc_allruns[fold, run_model] = accuracy on all 30 test runs (flattened)
    fold_acc_allruns = np.empty((N_FOLDS, N_RUNS))

    fill_offset = 0  # tracks where to write in the n_total arrays

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(
            np.zeros(n_total), labels)):
        n_test = len(test_idx)
        print(f"    Fold {fold_idx + 1}/{N_FOLDS} "
              f"(train={len(train_idx)}, test={n_test})...")

        y_test = labels[test_idx]
        true_labels_ordered[fill_offset:fill_offset + n_test] = y_test

        for run_model in range(N_RUNS):
            # ----- Train on single run -----
            X_train = all_run_features[train_idx, run_model, :]  # (n_train, N_FEATURES)
            y_train = labels[train_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)

            rf = RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                max_features=RF_MAX_FEATURES,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                random_state=RANDOM_STATE,
                n_jobs=n_jobs,
            )
            rf.fit(X_train_scaled, y_train)

            # ----- Test on median features -----
            X_test_median = median_features[test_idx]
            X_test_median_scaled = scaler.transform(X_test_median)
            X_test_median_scaled = np.nan_to_num(X_test_median_scaled, nan=0.0)
            y_pred_median = rf.predict(X_test_median_scaled)

            pred_median_by_model[fill_offset:fill_offset + n_test, run_model] = y_pred_median
            fold_acc_median[fold_idx, run_model] = accuracy_score(y_test, y_pred_median)

            # ----- Test on all 30 runs -----
            # Batch all test instances × all runs
            X_test_runs = all_run_features[test_idx]  # (n_test, N_RUNS, N_FEATURES)
            X_test_runs_2d = X_test_runs.reshape(n_test * N_RUNS, N_FEATURES)
            X_test_runs_scaled = scaler.transform(X_test_runs_2d)
            X_test_runs_scaled = np.nan_to_num(X_test_runs_scaled, nan=0.0)
            y_pred_runs = rf.predict(X_test_runs_scaled).reshape(n_test, N_RUNS)

            pred_runs_by_model[fill_offset:fill_offset + n_test, run_model, :] = y_pred_runs
            y_test_repeated = np.repeat(y_test, N_RUNS)
            fold_acc_allruns[fold_idx, run_model] = accuracy_score(
                y_test_repeated, y_pred_runs.flatten())

        # Summary for this fold
        print(f"      Median-test accuracy (mean over 30 models): "
              f"{fold_acc_median[fold_idx].mean():.4f} "
              f"(±{fold_acc_median[fold_idx].std():.4f})")
        print(f"      All-runs accuracy    (mean over 30 models): "
              f"{fold_acc_allruns[fold_idx].mean():.4f} "
              f"(±{fold_acc_allruns[fold_idx].std():.4f})")

        fill_offset += n_test

    # --- Aggregate results ---
    # Per-model overall accuracy on median test (across all folds)
    per_model_acc_median = np.array([
        accuracy_score(true_labels_ordered, pred_median_by_model[:, r])
        for r in range(N_RUNS)
    ])
    # Per-model overall accuracy on all runs test
    per_model_acc_allruns = np.array([
        accuracy_score(
            np.repeat(true_labels_ordered, N_RUNS),
            pred_runs_by_model[:, r, :].flatten())
        for r in range(N_RUNS)
    ])

    # Majority vote across the 30 models for median-test predictions
    majority_vote_median = mode(pred_median_by_model, axis=1,
                                keepdims=False).mode  # (n_total,)

    # Per-instance consistency: for each model, fraction of 30 test runs
    # that agree with the true label, then averaged over 30 models
    # Shape: (n_total, N_RUNS) where [i, r_model] = consistency of model r_model
    #   on instance i across 30 test runs
    per_instance_consistency = np.mean(
        pred_runs_by_model == true_labels_ordered[:, None, None],
        axis=2)  # (n_total, N_RUNS)

    results = {
        # Per-fold, per-model accuracies
        "fold_acc_median": fold_acc_median,              # (N_FOLDS, N_RUNS)
        "fold_acc_allruns": fold_acc_allruns,             # (N_FOLDS, N_RUNS)
        # Per-model overall accuracies
        "per_model_acc_median": per_model_acc_median,    # (N_RUNS,)
        "per_model_acc_allruns": per_model_acc_allruns,   # (N_RUNS,)
        # Per-instance consistency per model
        "per_instance_consistency": per_instance_consistency,  # (n_total, N_RUNS)
        # For confusion matrices
        "true_labels": true_labels_ordered,               # (n_total,)
        "pred_median_by_model": pred_median_by_model,     # (n_total, N_RUNS)
        "pred_runs_by_model": pred_runs_by_model,         # (n_total, N_RUNS, N_RUNS)
        "pred_majority_vote_median": majority_vote_median, # (n_total,)
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
    output_file = Path(output_dir) / "ela_classification_results_singlerun.h5"

    if configs:
        config_keys = [c for c in configs if c in ELA_FILES]
    else:
        config_keys = list(ELA_FILES.keys())

    print(f"ELA Classification Experiment (Train on Single Run, 30 models)")
    print(f"Features: {N_FEATURES}")
    print(f"Folds: {N_FOLDS}")
    print(f"Models per fold: {N_RUNS}")
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
            median_features, all_run_features, labels, _ = build_instance_data(data)
            del data

            print("  Running classification (30 models × 5 folds)...")
            results = run_classification(median_features, all_run_features, labels,
                                         n_jobs=n_jobs)

            # Save results
            config_grp = out.create_group(config_key)

            config_grp.create_dataset("fold_acc_median",
                                       data=results["fold_acc_median"])
            config_grp.create_dataset("fold_acc_allruns",
                                       data=results["fold_acc_allruns"])
            config_grp.create_dataset("per_model_acc_median",
                                       data=results["per_model_acc_median"])
            config_grp.create_dataset("per_model_acc_allruns",
                                       data=results["per_model_acc_allruns"])
            config_grp.create_dataset("per_instance_consistency",
                                       data=results["per_instance_consistency"])

            # Predictions for confusion matrices
            config_grp.create_dataset("true_labels",
                                       data=results["true_labels"])
            config_grp.create_dataset("pred_median_by_model",
                                       data=results["pred_median_by_model"])
            config_grp.create_dataset("pred_runs_by_model",
                                       data=results["pred_runs_by_model"])
            config_grp.create_dataset("pred_majority_vote_median",
                                       data=results["pred_majority_vote_median"])

            # Summary attributes
            config_grp.attrs["overall_acc_median_mean"] = results["per_model_acc_median"].mean()
            config_grp.attrs["overall_acc_median_std"] = results["per_model_acc_median"].std()
            config_grp.attrs["overall_acc_allruns_mean"] = results["per_model_acc_allruns"].mean()
            config_grp.attrs["overall_acc_allruns_std"] = results["per_model_acc_allruns"].std()
            config_grp.attrs["overall_acc_majority_vote_median"] = accuracy_score(
                results["true_labels"], results["pred_majority_vote_median"])
            config_grp.attrs["overall_consistency_mean"] = results["per_instance_consistency"].mean()
            config_grp.attrs["overall_consistency_std"] = results["per_instance_consistency"].std()

            print(f"\n  Results (averaged over 30 models):")
            print(f"    Median-test accuracy:        "
                  f"{results['per_model_acc_median'].mean():.4f} "
                  f"(±{results['per_model_acc_median'].std():.4f})")
            print(f"    All-runs accuracy:           "
                  f"{results['per_model_acc_allruns'].mean():.4f} "
                  f"(±{results['per_model_acc_allruns'].std():.4f})")
            print(f"    Majority-vote median acc:    "
                  f"{accuracy_score(results['true_labels'], results['pred_majority_vote_median']):.4f}")
            print(f"    Consistency (mean):          "
                  f"{results['per_instance_consistency'].mean():.4f} "
                  f"(±{results['per_instance_consistency'].std():.4f})")
            print()

            del median_features, all_run_features
            gc.collect()

    print(f"\nResults saved to: {output_file}")
    print(f"\nOutput h5 structure:")
    print(f"  {{config_key}}/")
    print(f"    fold_acc_median            ({N_FOLDS}, {N_RUNS})")
    print(f"    fold_acc_allruns           ({N_FOLDS}, {N_RUNS})")
    print(f"    per_model_acc_median       ({N_RUNS},)")
    print(f"    per_model_acc_allruns      ({N_RUNS},)")
    print(f"    per_instance_consistency   (n_instances, {N_RUNS})")
    print(f"    true_labels                (n_instances,)")
    print(f"    pred_median_by_model       (n_instances, {N_RUNS})")
    print(f"    pred_runs_by_model         (n_instances, {N_RUNS}, {N_RUNS})")
    print(f"    pred_majority_vote_median  (n_instances,)")
    print(f"    attrs: overall_acc_median_mean/std, overall_acc_allruns_mean/std,")
    print(f"           overall_acc_majority_vote_median, overall_consistency_mean/std")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classification experiment for ELA features (train on single run)."
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