"""
Classification experiment for TLA features using Random Forest.
Variant: train on ALL individual runs (30 rows per instance).

For each configuration and segment:
  - 5-fold stratified CV over 100 instances (split by instance, not run)
  - Train on all 30 runs per instance (expanded training set), with PCA 99%
  - Test on median features and individual runs (30 predictions per test instance)
  - Record per-instance consistency and overall accuracy

Usage:
  python classify_tla_allruns.py /path/to/tla/h5/dir
  python classify_tla_allruns.py /path/to/tla/h5/dir --output-dir /path/to/output
  python classify_tla_allruns.py /path/to/tla/h5/dir --segment volume_h0
  python classify_tla_allruns.py /path/to/tla/h5/dir --configs ilhs_50 sobol_50
  python classify_tla_allruns.py /path/to/tla/h5/dir --n-jobs -1
"""

import argparse
import numpy as np
import h5py
import os
import gc
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
PCA_VARIANCE = 0.99

# Random Forest defaults
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = None
RF_MAX_FEATURES = "sqrt"
RF_MIN_SAMPLES_LEAF = 1

PERSPECTIVES_LIST = ["volume", "axis"]
HOMOLOGIES = ["h0", "h1", "h2"]
FEATURE_LENGTHS = {"h0": 100, "h1": 10000, "h2": 10000}

SEGMENTS = {
    "all": (None, None),
    "volume_all": ("volume", None),
    "axis_all": ("axis", None),
    "volume_h0": ("volume", "h0"),
    "volume_h1": ("volume", "h1"),
    "volume_h2": ("volume", "h2"),
    "axis_h0": ("axis", "h0"),
    "axis_h1": ("axis", "h1"),
    "axis_h2": ("axis", "h2"),
}

TLA_FILES = {
    "ilhs_10": "ilhs_10_tla.h5", "ilhs_25": "ilhs_25_tla.h5",
    "ilhs_50": "ilhs_50_tla.h5", "ilhs_75": "ilhs_75_tla.h5",
    "ilhs_100": "ilhs_100_tla.h5",
    "lhs_10": "lhs_10_tla.h5", "lhs_25": "lhs_25_tla.h5",
    "lhs_50": "lhs_50_tla.h5", "lhs_75": "lhs_75_tla.h5",
    "lhs_100": "lhs_100_tla.h5",
    "sobol_10": "sobol_10_tla.h5", "sobol_25": "sobol_25_tla.h5",
    "sobol_50": "sobol_50_tla.h5", "sobol_75": "sobol_75_tla.h5",
    "sobol_100": "sobol_100_tla.h5",
    "uniform_10": "uniform_10_tla.h5", "uniform_25": "uniform_25_tla.h5",
    "uniform_50": "uniform_50_tla.h5", "uniform_75": "uniform_75_tla.h5",
    "uniform_100": "uniform_100_tla.h5",
    "cma_random_10": "cma_random_10_tla.h5", "cma_random_25": "cma_random_25_tla.h5",
    "cma_random_50": "cma_random_50_tla.h5", "cma_random_75": "cma_random_75_tla.h5",
    "cma_random_100": "cma_random_100_tla.h5",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_segment_specs(perspective_filter, homology_filter):
    specs = []
    for persp in PERSPECTIVES_LIST:
        if perspective_filter is not None and persp != perspective_filter:
            continue
        for hom in HOMOLOGIES:
            if homology_filter is not None and hom != homology_filter:
                continue
            specs.append((persp, hom))
    return specs


def get_segment_length(specs):
    return sum(FEATURE_LENGTHS[hom] for _, hom in specs)


def load_all_data(h5_path, specs):
    """
    Load all data for a segment: all instances, all runs.

    Returns
    -------
    data : np.ndarray, shape (2400, N_RUNS, seg_len)
    """
    seg_len = get_segment_length(specs)
    data = np.empty((N_FUNCTIONS * N_INSTANCES, N_RUNS, seg_len))

    with h5py.File(h5_path, "r") as f:
        for func_idx in range(N_FUNCTIONS):
            func_id = func_idx + 1
            for inst_idx in range(N_INSTANCES):
                inst_id = inst_idx + 1
                row = func_idx * N_INSTANCES + inst_idx
                key = f"{func_id}_{inst_id}_{DIMENSION}"
                group = f[key]

                offset = 0
                for persp, hom in specs:
                    feat_len = FEATURE_LENGTHS[hom]
                    arr = group[persp][hom][:]
                    arr = arr.reshape(arr.shape[0], -1)  # (N_RUNS, feat_len)
                    data[row, :, offset:offset + feat_len] = arr
                    offset += feat_len

    return data


def _sanitize(X):
    """Clip extremes and replace non-finite values."""
    X = np.clip(X, -1e10, 1e10)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# ---------------------------------------------------------------------------
# Classification experiment
# ---------------------------------------------------------------------------

def run_classification(all_data, labels, n_jobs=1):
    """
    Run 5-fold stratified CV.

    Train on all individual runs (expanded), with PCA.
    Test on median features and individual runs.

    Parameters
    ----------
    all_data : np.ndarray, shape (2400, N_RUNS, seg_len)
    labels : np.ndarray, shape (2400,)

    Returns dict with results.
    """
    median_features = np.median(all_data, axis=1)  # (2400, seg_len)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_accuracies_median = []
    fold_accuracies_all_runs = []
    fold_per_instance_consistency = []
    fold_n_pca_components = []

    all_true_labels = []
    all_pred_median = []
    all_pred_runs = []
    all_consistency = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(
            np.zeros(len(labels)), labels)):
        print(f"    Fold {fold_idx + 1}/{N_FOLDS}...")

        y_train = labels[train_idx]
        y_test = labels[test_idx]

        # ----- Build expanded training set: all runs for train instances -----
        n_train = len(train_idx)
        X_train = all_data[train_idx].reshape(n_train * N_RUNS, -1)
        y_train_expanded = np.repeat(y_train, N_RUNS)

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Filter non-finite columns
        nan_cols = np.any(~np.isfinite(X_train_scaled), axis=0)
        valid_cols = ~nan_cols
        X_train_scaled = _sanitize(X_train_scaled[:, valid_cols])

        if X_train_scaled.shape[1] == 0:
            print(f"      WARNING: No valid features after filtering")
            fold_accuracies_median.append(0.0)
            fold_accuracies_all_runs.append(0.0)
            fold_per_instance_consistency.append(0.0)
            fold_n_pca_components.append(0)
            continue

        # PCA
        pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
        X_train_pca = pca.fit_transform(X_train_scaled)
        n_components = X_train_pca.shape[1]
        fold_n_pca_components.append(n_components)

        # Train RF
        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            max_features=RF_MAX_FEATURES,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
        )
        rf.fit(X_train_pca, y_train_expanded)

        # Free training data
        del X_train, X_train_scaled, X_train_pca
        gc.collect()

        # ----- Test on median features -----
        X_test_median = median_features[test_idx]
        X_test_median_scaled = _sanitize(scaler.transform(X_test_median)[:, valid_cols])
        X_test_median_pca = _sanitize(pca.transform(X_test_median_scaled))
        y_pred_median = rf.predict(X_test_median_pca)

        acc_median = accuracy_score(y_test, y_pred_median)
        fold_accuracies_median.append(acc_median)
        all_pred_median.extend(y_pred_median)

        # ----- Test on individual runs -----
        instance_consistencies = []
        instance_run_preds = []

        for i, test_row in enumerate(test_idx):
            X_runs = all_data[test_row]  # (N_RUNS, seg_len)
            X_runs_scaled = _sanitize(scaler.transform(X_runs)[:, valid_cols])
            X_runs_pca = _sanitize(pca.transform(X_runs_scaled))
            run_preds = rf.predict(X_runs_pca)

            consistency = np.mean(run_preds == y_test[i])
            instance_consistencies.append(consistency)
            instance_run_preds.append(run_preds)

        # Per-run accuracy
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
        print(f"      PCA components:       {n_components}")

    # Build prediction arrays for confusion matrices
    all_true_labels = np.array(all_true_labels)
    all_pred_median = np.array(all_pred_median)
    all_pred_runs = np.array(all_pred_runs)

    majority_vote = mode(all_pred_runs, axis=1, keepdims=False).mode

    results = {
        "fold_accuracies_median": np.array(fold_accuracies_median),
        "fold_accuracies_all_runs": np.array(fold_accuracies_all_runs),
        "fold_consistency": np.array(fold_per_instance_consistency),
        "fold_n_pca_components": np.array(fold_n_pca_components),
        "overall_accuracy_median": accuracy_score(all_true_labels, all_pred_median),
        "overall_accuracy_all_runs": np.mean(fold_accuracies_all_runs),
        "overall_consistency_mean": np.mean(all_consistency),
        "overall_consistency_std": np.std(all_consistency),
        "per_instance_consistency": np.array(all_consistency),
        # For confusion matrices
        "true_labels": all_true_labels,
        "pred_median": all_pred_median,
        "pred_runs": all_pred_runs,
        "pred_majority_vote": majority_vote,
    }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir=None, segment="all", configs=None, n_jobs=1):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    input_dir = Path(input_dir)
    output_file = Path(output_dir) / f"tla_classification_results_allruns_{segment}.h5"

    if segment not in SEGMENTS:
        print(f"Unknown segment: {segment}")
        print(f"Available: {list(SEGMENTS.keys())}")
        return

    perspective_filter, homology_filter = SEGMENTS[segment]
    specs = get_segment_specs(perspective_filter, homology_filter)
    seg_len = get_segment_length(specs)

    if configs:
        config_keys = [c for c in configs if c in TLA_FILES]
    else:
        config_keys = list(TLA_FILES.keys())

    labels = np.repeat(np.arange(N_FUNCTIONS), N_INSTANCES)

    mem_gb = N_FUNCTIONS * N_INSTANCES * N_RUNS * seg_len * 8 / (1024**3)

    print(f"TLA Classification Experiment (Train on All Runs)")
    print(f"Segment: {segment} ({seg_len} features, ~{mem_gb:.1f} GB)")
    print(f"PCA: {PCA_VARIANCE*100:.0f}% variance")
    print(f"Training rows per fold: ~{N_FUNCTIONS * N_INSTANCES * (N_FOLDS-1) // N_FOLDS * N_RUNS}")
    print(f"Folds: {N_FOLDS}")
    print(f"RF: n_estimators={RF_N_ESTIMATORS}, max_depth={RF_MAX_DEPTH}, "
          f"max_features={RF_MAX_FEATURES}")
    print(f"Configs: {len(config_keys)}")
    print(f"n_jobs: {n_jobs}")
    print()

    with h5py.File(output_file, "a") as out:
        for config_key in config_keys:
            filename = TLA_FILES[config_key]
            filepath = input_dir / filename
            if not filepath.exists():
                print(f"WARNING: {filepath} not found, skipping.")
                continue

            if config_key in out:
                print(f"  {config_key}: already exists, skipping")
                continue

            print(f"{'='*60}")
            print(f"Processing: {config_key} ({segment})")
            print(f"{'='*60}")

            print("  Loading data...")
            all_data = load_all_data(filepath, specs)
            print(f"  Data shape: {all_data.shape}")

            print("  Running classification...")
            results = run_classification(all_data, labels, n_jobs=n_jobs)

            # Save results
            config_grp = out.create_group(config_key)

            config_grp.create_dataset("fold_accuracies_median",
                                       data=results["fold_accuracies_median"])
            config_grp.create_dataset("fold_accuracies_all_runs",
                                       data=results["fold_accuracies_all_runs"])
            config_grp.create_dataset("fold_consistency",
                                       data=results["fold_consistency"])
            config_grp.create_dataset("fold_n_pca_components",
                                       data=results["fold_n_pca_components"])
            config_grp.create_dataset("per_instance_consistency",
                                       data=results["per_instance_consistency"])

            # Predictions for confusion matrices
            config_grp.create_dataset("true_labels",
                                       data=results["true_labels"])
            config_grp.create_dataset("pred_median",
                                       data=results["pred_median"])
            config_grp.create_dataset("pred_runs",
                                       data=results["pred_runs"])
            config_grp.create_dataset("pred_majority_vote",
                                       data=results["pred_majority_vote"])

            config_grp.attrs["overall_accuracy_median"] = results["overall_accuracy_median"]
            config_grp.attrs["overall_accuracy_all_runs"] = results["overall_accuracy_all_runs"]
            config_grp.attrs["overall_accuracy_majority_vote"] = accuracy_score(
                results["true_labels"], results["pred_majority_vote"])
            config_grp.attrs["overall_consistency_mean"] = results["overall_consistency_mean"]
            config_grp.attrs["overall_consistency_std"] = results["overall_consistency_std"]

            print(f"\n  Results:")
            print(f"    Median-test accuracy:   {results['overall_accuracy_median']:.4f}")
            print(f"    All-runs accuracy:      {results['overall_accuracy_all_runs']:.4f}")
            print(f"    Majority-vote accuracy: {accuracy_score(results['true_labels'], results['pred_majority_vote']):.4f}")
            print(f"    Consistency (mean):     {results['overall_consistency_mean']:.4f} "
                  f"(±{results['overall_consistency_std']:.4f})")
            print(f"    PCA components:         {results['fold_n_pca_components']}")
            print()

            del all_data
            gc.collect()

    print(f"\nResults saved to: {output_file}")
    print(f"\nOutput h5 structure:")
    print(f"  {{config_key}}/")
    print(f"    fold_accuracies_median     ({N_FOLDS},)")
    print(f"    fold_accuracies_all_runs   ({N_FOLDS},)")
    print(f"    fold_consistency           ({N_FOLDS},)")
    print(f"    fold_n_pca_components      ({N_FOLDS},)")
    print(f"    per_instance_consistency   (n_instances,)")
    print(f"    true_labels                (n_instances,)")
    print(f"    pred_median                (n_instances,)")
    print(f"    pred_runs                  (n_instances, {N_RUNS})")
    print(f"    pred_majority_vote         (n_instances,)")
    print(f"    attrs: overall_accuracy_median, overall_accuracy_all_runs,")
    print(f"           overall_accuracy_majority_vote,")
    print(f"           overall_consistency_mean, overall_consistency_std")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classification experiment for TLA features (train on all runs)."
    )
    parser.add_argument("input_dir", type=str,
                        help="Directory containing TLA h5 files.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output h5. Defaults to input_dir.")
    parser.add_argument("--segment", type=str, default="all",
                        choices=list(SEGMENTS.keys()),
                        help="TLA segment to use (default: all).")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific configs to process.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel jobs for Random Forest (default: 1).")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir,
         segment=args.segment, configs=args.configs, n_jobs=args.n_jobs)