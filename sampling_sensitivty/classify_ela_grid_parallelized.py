"""
Classification experiment for ELA features using Random Forest.
Variant: subsample both instances and runs for training.

Parallelised version: worker processes handle classification, a dedicated
writer thread serialises HDF5 output.

For each configuration and each (n_instances_train, n_runs_train) pair:
  - 5-fold stratified CV over 100 instances with a 20/80 train/test split
    (20 instances per class for training, 80 for testing)
  - From the 20 training instances per fold, subsample n_instances_train
    instances *per function class*
  - From the 30 runs per training instance, subsample n_runs_train runs
  - Test on all 30 individual runs per test instance, on the median
    feature vector, and via majority vote
  - Record per-instance consistency, overall accuracy, and which
    instances/runs were selected for training

The total number of function evaluations used for training is:
  n_feval_train = N_FUNCTIONS * n_instances_train * n_runs_train
                  * sample_size_per_dim * dimension

This budget is stored as an attribute so that results across different
(config, n_instances_train, n_runs_train) settings can be compared on
a common x-axis.

Usage:
  python classify_ela_subsample_parallel.py /path/to/ela/pkl/dir
  python classify_ela_subsample_parallel.py /path/to/ela/pkl/dir --output-dir /out
  python classify_ela_subsample_parallel.py /path/to/ela/pkl/dir --configs ilhs_50
  python classify_ela_subsample_parallel.py /path/to/ela/pkl/dir --n-instances-train 1 5 10 20
  python classify_ela_subsample_parallel.py /path/to/ela/pkl/dir --n-runs-train 1 3 5
  python classify_ela_subsample_parallel.py /path/to/ela/pkl/dir --max-workers 8
"""

import argparse
import pickle
import re
import numpy as np
import h5py
import os
import gc
import queue
import threading
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
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
N_FOLDS = 5
N_TRAIN_PER_FOLD = 20   # instances per class in each training fold
RANDOM_STATE = 42

# Default sweeps
DEFAULT_N_INSTANCES_TRAIN = [1, 2, 3, 5, 7, 10, 15, 20]
DEFAULT_N_RUNS_TRAIN = [1, 2, 3, 5]

# Random Forest defaults
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = None
RF_MAX_FEATURES = "sqrt"
RF_MIN_SAMPLES_LEAF = 1

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

# Candidate features: everything from non-omitted groups, minus costs_runtime.
CANDIDATE_FEATURES = []
for _grp, _feats in ELA_FEATURE_GROUPS.items():
    if _grp in OMIT_GROUPS:
        continue
    for _f in _feats:
        if "costs_runtime" in _f:
            continue
        CANDIDATE_FEATURES.append((_grp, _f))
N_CANDIDATE_FEATURES = len(CANDIDATE_FEATURES)

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


def parse_sample_size(config_key):
    m = re.search(r'_(\d+)$', config_key)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse sample size from config key: {config_key}")


def get_omit_features(dimension):
    if dimension == 2:
        return {
            'disp.diff_median_02', 'disp.ratio_median_02',
            'disp.ratio_mean_02', 'ela_meta.quad_simple.cond',
            'disp.diff_mean_02', 'ic.eps_ratio'
        }
    elif dimension in (5, 10):
        return {'ela_meta.quad_simple.cond'}
    return set()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_instance_data(data, dimension, omit_features=None,
                        drop_nan_features=False):
    """
    Build per-instance data structures, optionally detecting and
    dropping features that contain any NaN values.

    Returns
    -------
    median_features : np.ndarray, shape (n_total, n_kept_features)
    all_run_features : np.ndarray, shape (n_total, N_RUNS, n_kept_features)
    labels : np.ndarray, shape (n_total,)
    instance_indices : np.ndarray, shape (n_total,)
    kept_features : list of (str, str)
    omitted_features : list of (str, str)
    """
    if omit_features is None:
        omit_features = set()

    n_total = N_FUNCTIONS * N_INSTANCES

    median_all = np.empty((n_total, N_CANDIDATE_FEATURES))
    runs_all = np.empty((n_total, N_RUNS, N_CANDIDATE_FEATURES))
    labels = np.empty(n_total, dtype=int)
    instance_indices = np.empty(n_total, dtype=int)

    for func_idx in range(N_FUNCTIONS):
        func_id = func_idx + 1
        for inst_idx in range(N_INSTANCES):
            inst_id = inst_idx + 1
            row = func_idx * N_INSTANCES + inst_idx
            instance_key = (func_id, inst_id, dimension)
            instance_data = data[instance_key]

            labels[row] = func_idx
            instance_indices[row] = inst_idx

            for feat_idx, (grp_name, feat_name) in enumerate(CANDIDATE_FEATURES):
                values = [instance_data[grp_name][run][feat_name]
                          for run in range(N_RUNS)]
                median_all[row, feat_idx] = np.nanmedian(values)
                for run in range(N_RUNS):
                    runs_all[row, run, feat_idx] = values[run]

    if drop_nan_features:
        flat = runs_all.reshape(-1, N_CANDIDATE_FEATURES)
        has_bad = np.any(~np.isfinite(flat), axis=0)
    else:
        has_bad = np.zeros(N_CANDIDATE_FEATURES, dtype=bool)

    kept_features = []
    omitted_features_list = []
    kept_indices = []
    for feat_idx, (grp, feat) in enumerate(CANDIDATE_FEATURES):
        if has_bad[feat_idx] or feat in omit_features:
            omitted_features_list.append((grp, feat))
        else:
            kept_features.append((grp, feat))
            kept_indices.append(feat_idx)

    kept_indices = np.array(kept_indices)
    median_features = median_all[:, kept_indices]
    all_run_features = runs_all[:, :, kept_indices]

    return (median_features, all_run_features, labels, instance_indices,
            kept_features, omitted_features_list)


# ---------------------------------------------------------------------------
# Classification (runs in worker processes)
# ---------------------------------------------------------------------------

def run_classification(median_features, all_run_features, labels,
                       n_instances_train, n_runs_train):
    """
    Run 5-fold stratified CV with subsampled training instances and runs.

    NOTE: n_jobs is always 1 inside the worker — parallelism comes from
    running multiple grid points concurrently via ProcessPoolExecutor.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    fold_accuracies_median = []
    fold_accuracies_all_runs = []
    fold_per_instance_consistency = []

    sampled_instances_per_fold = []
    sampled_runs_per_fold = []

    all_true_labels = []
    all_pred_median = []
    all_pred_runs = []
    all_consistency = []

    for fold_idx, (test_idx, train_idx) in enumerate(skf.split(
            np.zeros(len(labels)), labels)):
        # NOTE: swapped — skf yields (80%, 20%), we use 20% as train.
        train_labels = labels[train_idx]
        selected_train_idx = []

        for cls in range(N_FUNCTIONS):
            cls_mask = train_labels == cls
            cls_indices = train_idx[cls_mask]
            if n_instances_train >= len(cls_indices):
                chosen = cls_indices
            else:
                chosen = rng.choice(cls_indices, size=n_instances_train,
                                    replace=False)
            selected_train_idx.append(chosen)

        selected_train_idx = np.concatenate(selected_train_idx)
        sampled_instances_per_fold.append(selected_train_idx.copy())
        n_train = len(selected_train_idx)

        if n_runs_train >= N_RUNS:
            sampled_runs = np.tile(np.arange(N_RUNS), (n_train, 1))
        else:
            sampled_runs = np.empty((n_train, n_runs_train), dtype=int)
            for i in range(n_train):
                sampled_runs[i] = rng.choice(N_RUNS, size=n_runs_train,
                                             replace=False)
        sampled_runs_per_fold.append(sampled_runs)

        n_features = all_run_features.shape[2]
        row_idx = np.arange(n_train)[:, None]
        X_train = all_run_features[selected_train_idx][row_idx, sampled_runs, :]
        X_train = X_train.reshape(n_train * n_runs_train, n_features)
        y_train = np.repeat(labels[selected_train_idx], n_runs_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)

        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            max_features=RF_MAX_FEATURES,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=1,  # single-threaded inside worker
        )
        rf.fit(X_train_scaled, y_train)

        del X_train, X_train_scaled

        # Test on median features
        y_test = labels[test_idx]
        X_test_median = median_features[test_idx]
        X_test_median_scaled = scaler.transform(X_test_median)
        X_test_median_scaled = np.nan_to_num(X_test_median_scaled, nan=0.0)
        y_pred_median = rf.predict(X_test_median_scaled)

        acc_median = accuracy_score(y_test, y_pred_median)
        fold_accuracies_median.append(acc_median)
        all_pred_median.extend(y_pred_median)

        # Test on individual runs
        instance_consistencies = []
        instance_run_preds = []

        for i, test_row in enumerate(test_idx):
            X_runs = all_run_features[test_row]
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

    all_true_labels = np.array(all_true_labels)
    all_pred_median = np.array(all_pred_median)
    all_pred_runs = np.array(all_pred_runs)

    majority_vote = mode(all_pred_runs, axis=1, keepdims=False).mode

    results = {
        "fold_accuracies_median": np.array(fold_accuracies_median),
        "fold_accuracies_all_runs": np.array(fold_accuracies_all_runs),
        "fold_consistency": np.array(fold_per_instance_consistency),
        "overall_accuracy_median": accuracy_score(all_true_labels,
                                                  all_pred_median),
        "overall_accuracy_all_runs": np.mean(fold_accuracies_all_runs),
        "overall_consistency_mean": np.mean(all_consistency),
        "overall_consistency_std": np.std(all_consistency),
        "per_instance_consistency": np.array(all_consistency),
        "true_labels": all_true_labels,
        "pred_median": all_pred_median,
        "pred_runs": all_pred_runs,
        "pred_majority_vote": majority_vote,
        "sampled_instances": sampled_instances_per_fold,
        "sampled_runs": sampled_runs_per_fold,
        "n_instances_train": n_instances_train,
        "n_runs_train": n_runs_train,
    }

    return results


def _worker(median_features, all_run_features, labels,
            config_key, sample_size, dimension,
            n_instances_train, n_runs_train):
    """
    Top-level function executed by each worker process.

    Returns a dict with the classification results plus metadata needed
    by the writer thread.
    """
    n_feval_train = (N_FUNCTIONS * n_instances_train
                     * n_runs_train * sample_size * dimension)

    results = run_classification(
        median_features, all_run_features, labels,
        n_instances_train=n_instances_train,
        n_runs_train=n_runs_train,
    )

    results["config_key"] = config_key
    results["sample_size"] = sample_size
    results["dimension"] = dimension
    results["n_feval_train"] = n_feval_train

    return results


# ---------------------------------------------------------------------------
# Writer thread
# ---------------------------------------------------------------------------

_SENTINEL = None  # signals the writer thread to shut down


def _writer_loop(write_queue, output_file):
    """
    Runs in a dedicated thread.  Pulls results dicts from *write_queue*
    and appends them to the HDF5 file.  Stops when it receives _SENTINEL.
    """
    with h5py.File(output_file, "a") as out:
        while True:
            item = write_queue.get()
            if item is _SENTINEL:
                write_queue.task_done()
                break

            results = item
            config_key = results["config_key"]
            n_instances_train = results["n_instances_train"]
            n_runs_train = results["n_runs_train"]
            sample_size = results["sample_size"]
            dimension = results["dimension"]
            n_feval_train = results["n_feval_train"]

            config_grp = out.require_group(config_key)
            subkey = f"inst_{n_instances_train:02d}_runs_{n_runs_train:02d}"

            # Guard against duplicate writes (shouldn't happen, but safe)
            if subkey in config_grp:
                write_queue.task_done()
                continue

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

            si_grp = sub_grp.create_group("sampled_instances")
            for f_i, arr in enumerate(results["sampled_instances"]):
                si_grp.create_dataset(f"fold_{f_i}", data=arr)

            sr_grp = sub_grp.create_group("sampled_runs")
            for f_i, arr in enumerate(results["sampled_runs"]):
                sr_grp.create_dataset(f"fold_{f_i}", data=arr)

            acc_mv = accuracy_score(results["true_labels"],
                                    results["pred_majority_vote"])
            sub_grp.attrs["n_instances_train"] = n_instances_train
            sub_grp.attrs["n_runs_train"] = n_runs_train
            sub_grp.attrs["sample_size_per_dim"] = sample_size
            sub_grp.attrs["dimension"] = dimension
            sub_grp.attrs["n_feval_train"] = n_feval_train
            sub_grp.attrs["overall_accuracy_median"] = \
                results["overall_accuracy_median"]
            sub_grp.attrs["overall_accuracy_all_runs"] = \
                results["overall_accuracy_all_runs"]
            sub_grp.attrs["overall_accuracy_majority_vote"] = acc_mv
            sub_grp.attrs["overall_consistency_mean"] = \
                results["overall_consistency_mean"]
            sub_grp.attrs["overall_consistency_std"] = \
                results["overall_consistency_std"]

            out.flush()

            print(f"  [written] {config_key}/{subkey}  "
                  f"median={results['overall_accuracy_median']:.4f}  "
                  f"all_runs={results['overall_accuracy_all_runs']:.4f}  "
                  f"mv={acc_mv:.4f}  "
                  f"consist={results['overall_consistency_mean']:.4f}"
                  f"(+/-{results['overall_consistency_std']:.4f})  "
                  f"fevals={n_feval_train}")

            write_queue.task_done()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir=None, configs=None,
         n_instances_train_list=None, n_runs_train_list=None,
         dimension=5, max_workers=None, drop_nan_features=False):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    if n_instances_train_list is None:
        n_instances_train_list = DEFAULT_N_INSTANCES_TRAIN
    if n_runs_train_list is None:
        n_runs_train_list = DEFAULT_N_RUNS_TRAIN

    for n in n_instances_train_list:
        if n < 1 or n > N_TRAIN_PER_FOLD:
            raise ValueError(
                f"n_instances_train must be in [1, {N_TRAIN_PER_FOLD}], "
                f"got {n}")
    for n in n_runs_train_list:
        if n < 1 or n > N_RUNS:
            raise ValueError(
                f"n_runs_train must be in [1, {N_RUNS}], got {n}")

    omit_features = get_omit_features(dimension)

    input_dir = Path(input_dir)
    output_file = Path(output_dir) / "ela_classification_subsample.h5"

    if configs:
        config_keys = [c for c in configs if c in ELA_FILES]
    else:
        config_keys = list(ELA_FILES.keys())

    grid = [(ni, nr) for ni in n_instances_train_list
            for nr in n_runs_train_list]

    if max_workers is None:
        max_workers = min(len(grid), os.cpu_count() or 4)

    print(f"ELA Classification Experiment (parallel, subsampled instances & runs)")
    print(f"Dimension: {dimension}")
    print(f"Candidate features: {N_CANDIDATE_FEATURES}")
    print(f"Drop NaN/Inf features: {drop_nan_features}")
    if omit_features:
        print(f"Dimension-specific omissions: {omit_features}")
    print(f"Folds: {N_FOLDS} (20 train / 80 test per class per fold)")
    print(f"n_instances_train sweep: {n_instances_train_list}")
    print(f"n_runs_train sweep:      {n_runs_train_list}")
    print(f"Total grid points:       {len(grid)}")
    print(f"RF: n_estimators={RF_N_ESTIMATORS}, max_depth={RF_MAX_DEPTH}, "
          f"max_features={RF_MAX_FEATURES}")
    print(f"Configs: {len(config_keys)}")
    print(f"Max workers: {max_workers}")
    print()

    # --- Start writer thread ---
    write_queue = queue.Queue()
    writer_thread = threading.Thread(
        target=_writer_loop,
        args=(write_queue, str(output_file)),
        daemon=True,
    )
    writer_thread.start()

    # --- Determine already-completed grid points ---
    existing_subkeys = {}  # config_key -> set of subkeys
    if output_file.exists():
        with h5py.File(output_file, "r") as f:
            for ck in config_keys:
                if ck in f:
                    existing_subkeys[ck] = set(f[ck].keys()) - {
                        "kept_features", "kept_feature_groups",
                        "omitted_features"
                    }

    # --- Process each config ---
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for config_key in config_keys:
            filepath = input_dir / ELA_FILES[config_key]
            if not filepath.exists():
                print(f"WARNING: {filepath} not found, skipping.")
                continue

            sample_size = parse_sample_size(config_key)

            # Filter grid to pending items
            existing = existing_subkeys.get(config_key, set())
            pending_grid = []
            for ni, nr in grid:
                subkey = f"inst_{ni:02d}_runs_{nr:02d}"
                if subkey in existing:
                    print(f"  {config_key}/{subkey}: already exists, skipping")
                else:
                    pending_grid.append((ni, nr))

            if not pending_grid:
                continue

            print(f"\n{'='*60}")
            print(f"Processing: {config_key}  "
                  f"(sample_size_per_dim={sample_size})")
            print(f"{'='*60}")

            with open(filepath, "rb") as f:
                data = pickle.load(f)

            print("  Building feature matrices...")
            median_features, all_run_features, labels, _, \
                kept_features, omitted_features = \
                build_instance_data(data, dimension,
                                    omit_features=omit_features,
                                    drop_nan_features=drop_nan_features)
            del data

            n_features = len(kept_features)
            print(f"  Features retained: {n_features} / "
                  f"{N_CANDIDATE_FEATURES}")
            if omitted_features:
                omitted_names = [f for _, f in omitted_features]
                print(f"  Omitted (NaN/Inf/dim): {omitted_names}")

            # Write feature metadata (synchronous, before workers start)
            with h5py.File(str(output_file), "a") as out:
                config_grp = out.require_group(config_key)
                if "kept_features" not in config_grp:
                    kept_names = [f for _, f in kept_features]
                    kept_groups = [g for g, _ in kept_features]
                    config_grp.create_dataset(
                        "kept_features",
                        data=np.array(kept_names, dtype=h5py.string_dtype()))
                    config_grp.create_dataset(
                        "kept_feature_groups",
                        data=np.array(kept_groups, dtype=h5py.string_dtype()))
                    if omitted_features:
                        omitted_names = [f for _, f in omitted_features]
                        config_grp.create_dataset(
                            "omitted_features",
                            data=np.array(omitted_names,
                                          dtype=h5py.string_dtype()))
                    config_grp.attrs["n_features"] = n_features
                    config_grp.attrs["n_candidate_features"] = \
                        N_CANDIDATE_FEATURES
                out.flush()

            # Submit all pending grid points for this config
            print(f"  Submitting {len(pending_grid)} jobs to worker pool...")
            futures = {}
            for ni, nr in pending_grid:
                fut = pool.submit(
                    _worker,
                    median_features, all_run_features, labels,
                    config_key, sample_size, dimension,
                    ni, nr,
                )
                futures[fut] = (config_key, ni, nr)

            # Collect results and queue them for writing
            for fut in as_completed(futures):
                ck, ni, nr = futures[fut]
                try:
                    results = fut.result()
                    write_queue.put(results)
                except Exception as exc:
                    print(f"  ERROR {ck}/inst_{ni:02d}_runs_{nr:02d}: {exc}")

            del median_features, all_run_features
            gc.collect()

    # --- Shut down writer ---
    write_queue.put(_SENTINEL)
    write_queue.join()
    writer_thread.join()

    print(f"\nResults saved to: {output_file}")
    print(f"\nOutput h5 structure:")
    print(f"  {{config_key}}/")
    print(f"    inst_XX_runs_YY/  (one group per grid point)")
    print(f"      fold_accuracies_median     ({N_FOLDS},)")
    print(f"      fold_accuracies_all_runs   ({N_FOLDS},)")
    print(f"      fold_consistency           ({N_FOLDS},)")
    print(f"      per_instance_consistency   (n_test_instances,)")
    print(f"      true_labels                (n_test_instances,)")
    print(f"      pred_median                (n_test_instances,)")
    print(f"      pred_runs                  (n_test_instances, {N_RUNS})")
    print(f"      pred_majority_vote         (n_test_instances,)")
    print(f"      sampled_instances/fold_X   (n_selected_train,)")
    print(f"      sampled_runs/fold_X        (n_selected_train, n_runs_train)")
    print(f"      attrs: n_instances_train, n_runs_train, sample_size_per_dim,")
    print(f"             dimension, n_feval_train,")
    print(f"             overall_accuracy_median, overall_accuracy_all_runs,")
    print(f"             overall_accuracy_majority_vote,")
    print(f"             overall_consistency_mean, overall_consistency_std")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classification experiment for ELA features "
                    "(parallel, subsampled instances and runs)."
    )
    parser.add_argument("input_dir", type=str,
                        help="Directory containing ELA pkl files.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output h5. Defaults to input_dir.")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific configs to process.")
    parser.add_argument("--n-instances-train", nargs="+", type=int,
                        default=None,
                        help=f"Values of n_instances_train to sweep "
                             f"(default: {DEFAULT_N_INSTANCES_TRAIN}).")
    parser.add_argument("--n-runs-train", nargs="+", type=int,
                        default=None,
                        help=f"Values of n_runs_train to sweep "
                             f"(default: {DEFAULT_N_RUNS_TRAIN}).")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max parallel worker processes "
                             "(default: min(grid_size, cpu_count)).")
    parser.add_argument("--dimension", type=int, default=5,
                        help="Problem dimensionality (default: 5).")
    parser.add_argument("--drop-nan-features", action="store_true",
                        default=False,
                        help="Automatically drop features containing "
                             "NaN/Inf values (default: off).")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir,
         configs=args.configs,
         n_instances_train_list=args.n_instances_train,
         n_runs_train_list=args.n_runs_train,
         dimension=args.dimension,
         max_workers=args.max_workers,
         drop_nan_features=args.drop_nan_features)