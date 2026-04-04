"""
Classification experiment for TLA features using Random Forest.
Variant: train on SINGLE RUN per instance (30 separate models).

For each configuration and segment:
  - 5-fold stratified CV over 100 instances (split by instance)
  - For each of the 30 runs, train a separate RF (with PCA 99%)
    on that run's features (one row per instance)
  - Each model is tested on:
      (a) median features of test instances
      (b) all 30 individual runs of test instances
  - Record per-run-model accuracies, consistency, and predictions

Usage:
  python classify_tla_singlerun.py /path/to/tla/h5/dir
  python classify_tla_singlerun.py /path/to/tla/h5/dir --output-dir /path/to/output
  python classify_tla_singlerun.py /path/to/tla/h5/dir --segment volume_h0
  python classify_tla_singlerun.py /path/to/tla/h5/dir --configs ilhs_50 sobol_50
  python classify_tla_singlerun.py /path/to/tla/h5/dir --n-jobs -1
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
    Run 5-fold stratified CV with 30 models (one per run).

    For each fold and each training run index (0-29):
      - Train RF (with PCA) on that single run's features for train instances
      - Test on median features and on all 30 runs of test instances

    Parameters
    ----------
    all_data : np.ndarray, shape (2400, N_RUNS, seg_len)
    labels : np.ndarray, shape (2400,)

    Returns dict with results.
    """
    median_features = np.median(all_data, axis=1)  # (2400, seg_len)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_total = len(labels)

    # Prediction arrays across all folds
    pred_median_by_model = np.empty((n_total, N_RUNS), dtype=int)
    pred_runs_by_model = np.empty((n_total, N_RUNS, N_RUNS), dtype=int)
    true_labels_ordered = np.empty(n_total, dtype=int)

    # Per-fold, per-run-model accuracies
    fold_acc_median = np.empty((N_FOLDS, N_RUNS))
    fold_acc_allruns = np.empty((N_FOLDS, N_RUNS))
    fold_n_pca_components = np.empty((N_FOLDS, N_RUNS), dtype=int)

    fill_offset = 0

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(
            np.zeros(n_total), labels)):
        n_test = len(test_idx)
        print(f"    Fold {fold_idx + 1}/{N_FOLDS} "
              f"(train={len(train_idx)}, test={n_test})...")

        y_test = labels[test_idx]
        true_labels_ordered[fill_offset:fill_offset + n_test] = y_test

        for run_model in range(N_RUNS):
            # ----- Train on single run -----
            X_train = all_data[train_idx, run_model, :]  # (n_train, seg_len)
            y_train = labels[train_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # Filter non-finite columns
            nan_cols = np.any(~np.isfinite(X_train_scaled), axis=0)
            valid_cols = ~nan_cols
            X_train_scaled = _sanitize(X_train_scaled[:, valid_cols])

            if X_train_scaled.shape[1] == 0:
                # No valid features — fill with dummy predictions
                pred_median_by_model[fill_offset:fill_offset + n_test, run_model] = -1
                pred_runs_by_model[fill_offset:fill_offset + n_test, run_model, :] = -1
                fold_acc_median[fold_idx, run_model] = 0.0
                fold_acc_allruns[fold_idx, run_model] = 0.0
                fold_n_pca_components[fold_idx, run_model] = 0
                continue

            # PCA
            pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
            X_train_pca = pca.fit_transform(X_train_scaled)
            fold_n_pca_components[fold_idx, run_model] = X_train_pca.shape[1]

            # Train RF
            rf = RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                max_features=RF_MAX_FEATURES,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                random_state=RANDOM_STATE,
                n_jobs=n_jobs,
            )
            rf.fit(X_train_pca, y_train)

            # ----- Test on median features -----
            X_test_median = median_features[test_idx]
            X_test_median_scaled = _sanitize(scaler.transform(X_test_median)[:, valid_cols])
            X_test_median_pca = _sanitize(pca.transform(X_test_median_scaled))
            y_pred_median = rf.predict(X_test_median_pca)

            pred_median_by_model[fill_offset:fill_offset + n_test, run_model] = y_pred_median
            fold_acc_median[fold_idx, run_model] = accuracy_score(y_test, y_pred_median)

            # ----- Test on all 30 runs -----
            X_test_runs = all_data[test_idx]  # (n_test, N_RUNS, seg_len)
            X_test_runs_2d = X_test_runs.reshape(n_test * N_RUNS, -1)
            X_test_runs_scaled = _sanitize(scaler.transform(X_test_runs_2d)[:, valid_cols])
            X_test_runs_pca = _sanitize(pca.transform(X_test_runs_scaled))
            y_pred_runs = rf.predict(X_test_runs_pca).reshape(n_test, N_RUNS)

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
        print(f"      PCA components (min/max): "
              f"{fold_n_pca_components[fold_idx].min()}/{fold_n_pca_components[fold_idx].max()}")

        fill_offset += n_test

    # --- Aggregate results ---
    per_model_acc_median = np.array([
        accuracy_score(true_labels_ordered, pred_median_by_model[:, r])
        for r in range(N_RUNS)
    ])
    per_model_acc_allruns = np.array([
        accuracy_score(
            np.repeat(true_labels_ordered, N_RUNS),
            pred_runs_by_model[:, r, :].flatten())
        for r in range(N_RUNS)
    ])

    majority_vote_median = mode(pred_median_by_model, axis=1,
                                keepdims=False).mode

    per_instance_consistency = np.mean(
        pred_runs_by_model == true_labels_ordered[:, None, None],
        axis=2)  # (n_total, N_RUNS)

    results = {
        "fold_acc_median": fold_acc_median,
        "fold_acc_allruns": fold_acc_allruns,
        "fold_n_pca_components": fold_n_pca_components,
        "per_model_acc_median": per_model_acc_median,
        "per_model_acc_allruns": per_model_acc_allruns,
        "per_instance_consistency": per_instance_consistency,
        # For confusion matrices
        "true_labels": true_labels_ordered,
        "pred_median_by_model": pred_median_by_model,
        "pred_runs_by_model": pred_runs_by_model,
        "pred_majority_vote_median": majority_vote_median,
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
    output_file = Path(output_dir) / f"tla_classification_results_singlerun_{segment}.h5"

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

    print(f"TLA Classification Experiment (Train on Single Run, 30 models)")
    print(f"Segment: {segment} ({seg_len} features, ~{mem_gb:.1f} GB)")
    print(f"PCA: {PCA_VARIANCE*100:.0f}% variance")
    print(f"Folds: {N_FOLDS}")
    print(f"Models per fold: {N_RUNS}")
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

            print("  Running classification (30 models × 5 folds)...")
            results = run_classification(all_data, labels, n_jobs=n_jobs)

            # Save results
            config_grp = out.create_group(config_key)

            config_grp.create_dataset("fold_acc_median",
                                       data=results["fold_acc_median"])
            config_grp.create_dataset("fold_acc_allruns",
                                       data=results["fold_acc_allruns"])
            config_grp.create_dataset("fold_n_pca_components",
                                       data=results["fold_n_pca_components"])
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

            del all_data
            gc.collect()

    print(f"\nResults saved to: {output_file}")
    print(f"\nOutput h5 structure:")
    print(f"  {{config_key}}/")
    print(f"    fold_acc_median            ({N_FOLDS}, {N_RUNS})")
    print(f"    fold_acc_allruns           ({N_FOLDS}, {N_RUNS})")
    print(f"    fold_n_pca_components      ({N_FOLDS}, {N_RUNS})")
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
        description="Classification experiment for TLA features (train on single run)."
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