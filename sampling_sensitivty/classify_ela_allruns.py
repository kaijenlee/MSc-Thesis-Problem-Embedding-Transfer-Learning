"""
Classification experiment for ELA features using Random Forest.
Variant: train on a variable number of runs per instance.

For each configuration and each value of N_RUNS_TRAIN:
  - 5-fold stratified CV over 100 instances (split by instance, not run)
  - Train on N_RUNS_TRAIN sampled runs per training instance
  - Test on individual runs (30 predictions per test instance) and on median
  - Record per-instance consistency, overall accuracy, and which runs were
    selected for training.

Usage:
  python classify_ela_allruns.py /path/to/ela/pkl/dir
  python classify_ela_allruns.py /path/to/ela/pkl/dir --output-dir /path/to/output
  python classify_ela_allruns.py /path/to/ela/pkl/dir --configs ilhs_50 sobol_50
  python classify_ela_allruns.py /path/to/ela/pkl/dir --n-runs-train 1 5 10 15 30
  python classify_ela_allruns.py /path/to/ela/pkl/dir --n-jobs -1
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
DIMENSION = 5
N_FOLDS = 5
RANDOM_STATE = 42

# Default sweep over number of runs used per training instance
DEFAULT_N_RUNS_TRAIN = [1, 5, 10, 15, 30]

# Random Forest defaults
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = None
RF_MAX_FEATURES = "sqrt"
RF_MIN_SAMPLES_LEAF = 1

# OMIT_FEATURES = {
#     "disp.diff_median_02", "disp.ratio_median_02",
#     "ela_level.lda_qda_50", "ela_level.lda_qda_25",
#     "ic.eps_ratio", "disp.ratio_mean_02", "disp.diff_mean_02",
#     "ela_level.lda_qda_10", "ela_meta.quad_simple.cond",
# }

OMIT_FEATURES = {
    'ela_meta.quad_simple.cond', 'ela_level.lda_qda_50'
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
    "pca": [
        "pca.expl_var.cov_x", "pca.expl_var.cor_x", "pca.expl_var.cov_init",
        "pca.expl_var.cor_init", "pca.expl_var_PC1.cov_x",
        "pca.expl_var_PC1.cor_x", "pca.expl_var_PC1.cov_init",
        "pca.expl_var_PC1.cor_init", "pca.costs_runtime",
    ]
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
    # "cma_random_10": "cma_random_10_ela.pkl",
    "cma_random_25": "cma_random_25_ela.pkl", "cma_random_50": "cma_random_50_ela.pkl",
    "cma_random_75": "cma_random_75_ela.pkl", "cma_random_100": "cma_random_100_ela.pkl",
    # "ilhs_10": "ilhs_10_ela.pkl",
    "ilhs_25": "ilhs_25_ela.pkl", "ilhs_50": "ilhs_50_ela.pkl",
    "ilhs_75": "ilhs_75_ela.pkl", "ilhs_100": "ilhs_100_ela.pkl",
    # "lhs_10": "lhs_10_ela.pkl",
    "lhs_25": "lhs_25_ela.pkl", "lhs_50": "lhs_50_ela.pkl",
    "lhs_75": "lhs_75_ela.pkl", "lhs_100": "lhs_100_ela.pkl",
    # "sobol_10": "sobol_10_ela.pkl",
    "sobol_25": "sobol_25_ela.pkl", "sobol_50": "sobol_50_ela.pkl",
    "sobol_75": "sobol_75_ela.pkl", "sobol_100": "sobol_100_ela.pkl",
    # "uniform_10": "uniform_10_ela.pkl",
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

def run_classification(median_features, all_run_features, labels,
                       n_runs_train, n_jobs=1):
    """
    Run 5-fold stratified CV with a specified number of training runs per
    training instance.

    For each training instance, `n_runs_train` runs are randomly sampled
    (without replacement) from the 30 available runs. The same sampled-run
    indices are used across all training instances within a fold (one draw
    per fold) for reproducibility and simplicity; see `sampled_runs` in the
    returned dict for the exact indices used.

    Test behaviour is unchanged: we test on all 30 individual runs per test
    instance and on the median feature vector.

    CV split is at the instance level to prevent leakage.

    Returns dict with results.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    # Per-fold results
    fold_accuracies_median = []
    fold_accuracies_all_runs = []
    fold_per_instance_consistency = []

    # Per-fold record of which run indices were used during training.
    # Shape: (N_FOLDS, n_train_instances_in_fold, n_runs_train)
    # Stored as an object array because fold sizes could differ by 1.
    sampled_runs_per_fold = []

    # Collect all predictions for overall metrics
    all_true_labels = []
    all_pred_median = []
    all_pred_runs = []
    all_consistency = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(
            np.zeros(len(labels)), labels)):
        print(f"    Fold {fold_idx + 1}/{N_FOLDS} "
              f"(n_runs_train={n_runs_train})...")

        n_train = len(train_idx)

        # ----- Sample n_runs_train per training instance -----
        # One independent draw per training instance, reproducible via rng.
        if n_runs_train >= N_RUNS:
            sampled_runs = np.tile(np.arange(N_RUNS), (n_train, 1))
        else:
            sampled_runs = np.empty((n_train, n_runs_train), dtype=int)
            for i in range(n_train):
                sampled_runs[i] = rng.choice(N_RUNS, size=n_runs_train,
                                             replace=False)
        sampled_runs_per_fold.append(sampled_runs)

        # Gather sampled runs for each training instance
        # all_run_features[train_idx] has shape (n_train, N_RUNS, N_FEATURES).
        # We select along axis=1 per row.
        row_idx = np.arange(n_train)[:, None]
        X_train = all_run_features[train_idx][row_idx, sampled_runs, :]
        # Shape: (n_train, n_runs_train, N_FEATURES)
        X_train = X_train.reshape(n_train * n_runs_train, N_FEATURES)
        y_train = np.repeat(labels[train_idx], n_runs_train)

        # Standardize based on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
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

        # Free training data
        del X_train, X_train_scaled
        gc.collect()

        # ----- Test on median features -----
        y_test = labels[test_idx]
        X_test_median = median_features[test_idx]
        X_test_median_scaled = scaler.transform(X_test_median)
        X_test_median_scaled = np.nan_to_num(X_test_median_scaled, nan=0.0)
        y_pred_median = rf.predict(X_test_median_scaled)

        acc_median = accuracy_score(y_test, y_pred_median)
        fold_accuracies_median.append(acc_median)
        all_pred_median.extend(y_pred_median)

        # ----- Test on individual runs (always all 30) -----
        instance_consistencies = []
        instance_run_preds = []

        for i, test_row in enumerate(test_idx):
            X_runs = all_run_features[test_row]  # (N_RUNS, N_FEATURES)
            X_runs_scaled = scaler.transform(X_runs)
            X_runs_scaled = np.nan_to_num(X_runs_scaled, nan=0.0)
            run_preds = rf.predict(X_runs_scaled)

            consistency = np.mean(run_preds == y_test[i])
            instance_consistencies.append(consistency)
            instance_run_preds.append(run_preds)

        all_run_preds_flat = np.array(instance_run_preds).flatten()
        y_test_repeated = np.repeat(y_test, N_RUNS)
        acc_all_runs = accuracy_score(y_test_repeated, all_run_preds_flat)

        fold_accuracies_all_runs.append(acc_all_runs)
        fold_per_instance_consistency.append(np.mean(instance_consistencies))
        all_consistency.extend(instance_consistencies)
        all_pred_runs.extend(instance_run_preds)
        all_true_labels.extend(y_test)

        print(f"      Median-test accuracy: {acc_median:.4f}")
        print(f"      All-runs accuracy:    {acc_all_runs:.4f}")
        print(f"      Mean consistency:     {np.mean(instance_consistencies):.4f}")

    # Build prediction arrays for confusion matrices
    all_true_labels = np.array(all_true_labels)
    all_pred_median = np.array(all_pred_median)
    all_pred_runs = np.array(all_pred_runs)

    majority_vote = mode(all_pred_runs, axis=1, keepdims=False).mode

    # If all folds have the same training-set size (likely, since 24*100=2400
    # is evenly divisible by 5), we can stack sampled_runs into a single
    # ndarray. Otherwise, keep as object array.
    try:
        sampled_runs_arr = np.stack(sampled_runs_per_fold, axis=0)
    except ValueError:
        sampled_runs_arr = np.array(sampled_runs_per_fold, dtype=object)

    results = {
        "fold_accuracies_median": np.array(fold_accuracies_median),
        "fold_accuracies_all_runs": np.array(fold_accuracies_all_runs),
        "fold_consistency": np.array(fold_per_instance_consistency),
        "overall_accuracy_median": accuracy_score(all_true_labels, all_pred_median),
        "overall_accuracy_all_runs": np.mean(fold_accuracies_all_runs),
        "overall_consistency_mean": np.mean(all_consistency),
        "overall_consistency_std": np.std(all_consistency),
        "per_instance_consistency": np.array(all_consistency),
        "true_labels": all_true_labels,
        "pred_median": all_pred_median,
        "pred_runs": all_pred_runs,
        "pred_majority_vote": majority_vote,
        "sampled_runs": sampled_runs_arr,
        "n_runs_train": n_runs_train,
    }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir=None, configs=None, n_runs_train_list=None,
         n_jobs=1):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    if n_runs_train_list is None:
        n_runs_train_list = DEFAULT_N_RUNS_TRAIN

    # Validate
    for n in n_runs_train_list:
        if n < 1 or n > N_RUNS:
            raise ValueError(f"n_runs_train must be in [1, {N_RUNS}], got {n}")

    input_dir = Path(input_dir)
    output_file = Path(output_dir) / "ela_classification_results_allruns.h5"

    if configs:
        config_keys = [c for c in configs if c in ELA_FILES]
    else:
        config_keys = list(ELA_FILES.keys())

    print(f"ELA Classification Experiment (variable n_runs_train)")
    print(f"Features: {N_FEATURES}")
    print(f"Folds: {N_FOLDS}")
    print(f"n_runs_train sweep: {n_runs_train_list}")
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

            print(f"{'='*60}")
            print(f"Processing: {config_key}")
            print(f"{'='*60}")

            # Work out which n_runs_train values still need to be computed.
            existing_group = out.get(config_key)
            pending_n_runs = []
            for n in n_runs_train_list:
                subkey = f"n_runs_{n:02d}"
                if (existing_group is not None
                        and subkey in existing_group):
                    print(f"  {config_key}/{subkey}: already exists, skipping")
                else:
                    pending_n_runs.append(n)

            if not pending_n_runs:
                continue

            with open(filepath, "rb") as f:
                data = pickle.load(f)

            print("  Building feature matrices...")
            median_features, all_run_features, labels, _ = build_instance_data(data)
            del data

            config_grp = out.require_group(config_key)

            for n_runs_train in pending_n_runs:
                print(f"\n  --- n_runs_train = {n_runs_train} ---")
                results = run_classification(
                    median_features, all_run_features, labels,
                    n_runs_train=n_runs_train, n_jobs=n_jobs,
                )

                subkey = f"n_runs_{n_runs_train:02d}"
                sub_grp = config_grp.create_group(subkey)

                sub_grp.create_dataset("fold_accuracies_median",
                                       data=results["fold_accuracies_median"])
                sub_grp.create_dataset("fold_accuracies_all_runs",
                                       data=results["fold_accuracies_all_runs"])
                sub_grp.create_dataset("fold_consistency",
                                       data=results["fold_consistency"])
                sub_grp.create_dataset("per_instance_consistency",
                                       data=results["per_instance_consistency"])
                sub_grp.create_dataset("true_labels",
                                       data=results["true_labels"])
                sub_grp.create_dataset("pred_median",
                                       data=results["pred_median"])
                sub_grp.create_dataset("pred_runs",
                                       data=results["pred_runs"])
                sub_grp.create_dataset("pred_majority_vote",
                                       data=results["pred_majority_vote"])

                # Save the sampled run indices. If ragged, h5py can't store
                # an object array directly, so fall back to per-fold datasets.
                sampled = results["sampled_runs"]
                if sampled.dtype == object:
                    sr_grp = sub_grp.create_group("sampled_runs")
                    for f_i, arr in enumerate(sampled):
                        sr_grp.create_dataset(f"fold_{f_i}",
                                              data=np.asarray(arr))
                else:
                    sub_grp.create_dataset("sampled_runs", data=sampled)

                sub_grp.attrs["n_runs_train"] = n_runs_train
                sub_grp.attrs["overall_accuracy_median"] = \
                    results["overall_accuracy_median"]
                sub_grp.attrs["overall_accuracy_all_runs"] = \
                    results["overall_accuracy_all_runs"]
                sub_grp.attrs["overall_accuracy_majority_vote"] = accuracy_score(
                    results["true_labels"], results["pred_majority_vote"])
                sub_grp.attrs["overall_consistency_mean"] = \
                    results["overall_consistency_mean"]
                sub_grp.attrs["overall_consistency_std"] = \
                    results["overall_consistency_std"]

                print(f"  Results (n_runs_train={n_runs_train}):")
                print(f"    Median-test accuracy:   "
                      f"{results['overall_accuracy_median']:.4f}")
                print(f"    All-runs accuracy:      "
                      f"{results['overall_accuracy_all_runs']:.4f}")
                print(f"    Majority-vote accuracy: "
                      f"{accuracy_score(results['true_labels'], results['pred_majority_vote']):.4f}")
                print(f"    Consistency (mean):     "
                      f"{results['overall_consistency_mean']:.4f} "
                      f"(±{results['overall_consistency_std']:.4f})")
                out.flush()

            del median_features, all_run_features
            gc.collect()

    print(f"\nResults saved to: {output_file}")
    print(f"\nOutput h5 structure:")
    print(f"  {{config_key}}/")
    print(f"    n_runs_XX/  (one group per value in the sweep)")
    print(f"      fold_accuracies_median     ({N_FOLDS},)")
    print(f"      fold_accuracies_all_runs   ({N_FOLDS},)")
    print(f"      fold_consistency           ({N_FOLDS},)")
    print(f"      per_instance_consistency   (n_instances,)")
    print(f"      true_labels                (n_instances,)")
    print(f"      pred_median                (n_instances,)")
    print(f"      pred_runs                  (n_instances, {N_RUNS})")
    print(f"      pred_majority_vote         (n_instances,)")
    print(f"      sampled_runs               (N_FOLDS, n_train_per_fold, "
          f"n_runs_train)")
    print(f"      attrs: n_runs_train, overall_accuracy_median,")
    print(f"             overall_accuracy_all_runs, overall_accuracy_majority_vote,")
    print(f"             overall_consistency_mean, overall_consistency_std")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classification experiment for ELA features "
                    "(variable number of training runs per instance)."
    )
    parser.add_argument("input_dir", type=str,
                        help="Directory containing ELA pkl files.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output h5. Defaults to input_dir.")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific configs to process.")
    parser.add_argument("--n-runs-train", nargs="+", type=int,
                        default=None,
                        help=f"Values of n_runs_train to sweep over "
                             f"(default: {DEFAULT_N_RUNS_TRAIN}).")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel jobs for Random Forest (default: 1).")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir,
         configs=args.configs, n_runs_train_list=args.n_runs_train,
         n_jobs=args.n_jobs)