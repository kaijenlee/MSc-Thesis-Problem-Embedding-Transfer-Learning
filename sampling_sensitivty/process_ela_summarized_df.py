"""
Tabulate ELA CV and classification results into a single DataFrame.

Reads:
  - ela_cv_results.pkl          (per-instance CV, from compute_ela_cv.py)
  - ela_cv_per_function.pkl     (per-function CV, from compute_ela_cv_per_function.py)
  - ela_cv_training_set.pkl     (per-fold/function CV, from compute_ela_cv_training_set.py)
  - ela_classification_results_allruns.h5 (classification, from classify_ela_allruns.py)

Produces a DataFrame with one row per (dimension, sampling_strategy, sample_size, n_runs_train).

Usage:
  python tabulate_results.py /path/to/dim2/results /path/to/dim5/results --output results_table.csv
  python tabulate_results.py /path/to/dim2/results --dimensions 2 --output results_table.csv
"""

import argparse
import pickle
import numpy as np
import h5py
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Feature filtering (must match the CV / classification scripts)
# ---------------------------------------------------------------------------

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

# Build the filtered feature list (group, feat_name) — same order as scripts
FILTERED_FEATURES = []
for _grp, _feats in ELA_FEATURE_GROUPS.items():
    if _grp in OMIT_GROUPS:
        continue
    for _f in _feats:
        if _f not in OMIT_FEATURES:
            FILTERED_FEATURES.append((_grp, _f))


def parse_config_key(config_key):
    """
    Parse e.g. 'ilhs_50' -> ('ilhs', 50).
    Handles multi-part strategies like 'cma_random_50' -> ('cma_random', 50).
    """
    parts = config_key.rsplit("_", 1)
    return parts[0], int(parts[1])


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def collect_all_feature_cvs_instance(instance_cv_data, config_key, dimension):
    """
    From per-instance CV data: ela_cv[config][(func,inst,dim)][group][feat]
    Returns dict {(group, feat): list of CV values} across all 24*100 instances.
    """
    config_data = instance_cv_data.get(config_key, {})
    feat_cvs = {ff: [] for ff in FILTERED_FEATURES}

    for (func, inst, dim), inst_cv in config_data.items():
        if dim != dimension:
            continue
        for grp, feat in FILTERED_FEATURES:
            if grp in inst_cv and feat in inst_cv[grp]:
                v = inst_cv[grp][feat]
                if not np.isnan(v):
                    feat_cvs[(grp, feat)].append(v)

    return feat_cvs


def collect_all_feature_cvs_function(function_cv_data, config_key, dimension):
    """
    From per-function CV data: ela_cv[config][(func,dim)][group][feat]
    Returns dict {(group, feat): list of CV values} across 24 functions.
    """
    config_data = function_cv_data.get(config_key, {})
    feat_cvs = {ff: [] for ff in FILTERED_FEATURES}

    for (func, dim), func_cv in config_data.items():
        if dim != dimension:
            continue
        for grp, feat in FILTERED_FEATURES:
            if grp in func_cv and feat in func_cv[grp]:
                v = func_cv[grp][feat]
                if not np.isnan(v):
                    feat_cvs[(grp, feat)].append(v)

    return feat_cvs


def collect_all_feature_cvs_training_folds(training_cv_data, config_key,
                                           n_runs_train):
    """
    From training-set CV data:
      results[config][n_runs]['per_fold_per_function'][fold][func_id][group][feat]
    Returns dict {(group, feat): list of CV values} across 24 functions × 5 folds.
    """
    config_data = training_cv_data.get(config_key, {})
    nrt_data = config_data.get(n_runs_train, {})
    pfpf = nrt_data.get("per_fold_per_function", {})

    feat_cvs = {ff: [] for ff in FILTERED_FEATURES}

    for fold_idx, fold_data in pfpf.items():
        for func_id, func_cv in fold_data.items():
            for grp, feat in FILTERED_FEATURES:
                if grp in func_cv and feat in func_cv[grp]:
                    v = func_cv[grp][feat]
                    if not np.isnan(v):
                        feat_cvs[(grp, feat)].append(v)

    return feat_cvs


def aggregate_median_then_stat(feat_cvs, stat_fn):
    """
    For each feature, take the median of its CV values across instances/functions,
    then apply stat_fn (np.mean or np.median) across all features.
    """
    medians = []
    for ff in FILTERED_FEATURES:
        vals = feat_cvs.get(ff, [])
        if vals:
            medians.append(np.median(vals))
    if not medians:
        return np.nan
    return stat_fn(medians)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_table(result_dirs, dimensions):
    """
    Build the combined results table.

    Parameters
    ----------
    result_dirs : list of (dimension, path) tuples
    dimensions : not used separately, encoded in result_dirs
    """
    rows = []

    for dimension, result_dir in result_dirs:
        result_dir = Path(result_dir)

        # Load available data files
        instance_cv_path = result_dir / "ela_cv_results.pkl"
        function_cv_path = result_dir / "ela_cv_per_function.pkl"
        training_cv_path = result_dir / "ela_cv_training_set.pkl"
        h5_path = result_dir / "ela_classification_results_allruns.h5"

        instance_cv = {}
        function_cv = {}
        training_cv = {}

        if instance_cv_path.exists():
            with open(instance_cv_path, "rb") as f:
                instance_cv = pickle.load(f)
            print(f"  Loaded {instance_cv_path}")
        else:
            print(f"  WARNING: {instance_cv_path} not found")

        if function_cv_path.exists():
            with open(function_cv_path, "rb") as f:
                function_cv = pickle.load(f)
            print(f"  Loaded {function_cv_path}")
        else:
            print(f"  WARNING: {function_cv_path} not found")

        if training_cv_path.exists():
            with open(training_cv_path, "rb") as f:
                training_cv = pickle.load(f)
            print(f"  Loaded {training_cv_path}")
        else:
            print(f"  WARNING: {training_cv_path} not found")

        h5_file = None
        if h5_path.exists():
            h5_file = h5py.File(h5_path, "r")
            print(f"  Loaded {h5_path}")
        else:
            print(f"  WARNING: {h5_path} not found")

        # Determine which config keys and n_runs_train values exist
        # from the HDF5 (classification results drive the row structure)
        if h5_file is not None:
            config_keys = list(h5_file.keys())
        else:
            # Fall back to union of keys from CV data
            config_keys = sorted(
                set(list(instance_cv.keys()) +
                    list(function_cv.keys()) +
                    list(training_cv.keys()))
            )

        for config_key in config_keys:
            strategy, base_sample_size = parse_config_key(config_key)
            sample_size = base_sample_size * dimension

            # Determine n_runs_train values from HDF5 or training CV
            n_runs_values = set()
            if h5_file is not None and config_key in h5_file:
                for subkey in h5_file[config_key].keys():
                    if subkey.startswith("n_runs_"):
                        n_runs_values.add(int(subkey.split("_")[-1]))
            if config_key in training_cv:
                n_runs_values.update(training_cv[config_key].keys())
            if not n_runs_values:
                # If no n_runs info, use default 30 (all runs)
                n_runs_values = {30}

            for n_runs_train in sorted(n_runs_values):
                row = {
                    "dimension": dimension,
                    "sampling_strategy": strategy,
                    "sample_size_per_dim": base_sample_size,
                    "runs_training_per_instance": n_runs_train,
                }

                # feval_training = (sample_size_per_dim * dim) * 24 functions * 80 instances * n_runs_train
                # Note: 80 training instances per fold (100 * 4/5 in 5-fold CV)
                n_training_instances = 24 * 80  # per fold
                row["feval_training"] = sample_size * n_training_instances * n_runs_train

                # --- Per-instance CV (columns 5-6) ---
                feat_cvs_inst = collect_all_feature_cvs_instance(
                    instance_cv, config_key, dimension)
                row["cv_instance_median_mean"] = aggregate_median_then_stat(
                    feat_cvs_inst, np.mean)
                row["cv_instance_median_median"] = aggregate_median_then_stat(
                    feat_cvs_inst, np.median)

                # --- Per-function CV (columns 7-8) ---
                feat_cvs_func = collect_all_feature_cvs_function(
                    function_cv, config_key, dimension)
                row["cv_function_median_mean"] = aggregate_median_then_stat(
                    feat_cvs_func, np.mean)
                row["cv_function_median_median"] = aggregate_median_then_stat(
                    feat_cvs_func, np.median)

                # --- Training folds per-function CV (columns 9-10) ---
                feat_cvs_train = collect_all_feature_cvs_training_folds(
                    training_cv, config_key, n_runs_train)
                row["cv_training_folds_function_median_mean"] = \
                    aggregate_median_then_stat(feat_cvs_train, np.mean)
                row["cv_training_folds_function_median_median"] = \
                    aggregate_median_then_stat(feat_cvs_train, np.median)

                # --- Classification accuracy ---
                if (h5_file is not None and config_key in h5_file):
                    subkey = f"n_runs_{n_runs_train:02d}"
                    if subkey in h5_file[config_key]:
                        grp = h5_file[config_key][subkey]
                        # Median-test accuracy
                        fold_accs_median = grp["fold_accuracies_median"][:]
                        row["accuracy_median_mean"] = np.mean(fold_accs_median)
                        row["accuracy_median_sd"] = np.std(fold_accs_median, ddof=1)
                        # All-runs accuracy
                        fold_accs_runs = grp["fold_accuracies_all_runs"][:]
                        row["accuracy_allruns_mean"] = np.mean(fold_accs_runs)
                        row["accuracy_allruns_sd"] = np.std(fold_accs_runs, ddof=1)
                    else:
                        row["accuracy_median_mean"] = np.nan
                        row["accuracy_median_sd"] = np.nan
                        row["accuracy_allruns_mean"] = np.nan
                        row["accuracy_allruns_sd"] = np.nan
                else:
                    row["accuracy_median_mean"] = np.nan
                    row["accuracy_median_sd"] = np.nan
                    row["accuracy_allruns_mean"] = np.nan
                    row["accuracy_allruns_sd"] = np.nan

                rows.append(row)

        if h5_file is not None:
            h5_file.close()

    df = pd.DataFrame(rows, columns=[
        "dimension",
        "sampling_strategy",
        "sample_size_per_dim",
        "feval_training",
        "runs_training_per_instance",
        "cv_instance_median_mean",
        "cv_instance_median_median",
        "cv_function_median_mean",
        "cv_function_median_median",
        "cv_training_folds_function_median_mean",
        "cv_training_folds_function_median_median",
        "accuracy_median_mean",
        "accuracy_median_sd",
        "accuracy_allruns_mean",
        "accuracy_allruns_sd",
    ])

    df.sort_values(
        by=["dimension", "sampling_strategy", "sample_size_per_dim",
            "runs_training_per_instance"],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Tabulate ELA CV and classification results into a "
                    "DataFrame.",
    )
    parser.add_argument(
        "result_dirs", nargs="+", type=str,
        help="One or two result directories. If two are given, use "
             "--dimensions to specify which dimension each corresponds to.",
    )
    parser.add_argument(
        "--dimensions", nargs="+", type=int, default=None,
        help="Dimension for each result directory (e.g. --dimensions 2 5). "
             "Must match number of result_dirs. If not given, defaults to "
             "[2] for one dir or [2, 5] for two dirs.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file base path (default: results_table in the first "
             "input directory).",
    )
    args = parser.parse_args()

    # Default output location: first input directory
    if args.output is None:
        args.output = str(Path(args.result_dirs[0]) / "results_table")

    if args.dimensions is None:
        if len(args.result_dirs) == 1:
            dims = [2]
        elif len(args.result_dirs) == 2:
            dims = [2, 5]
        else:
            raise ValueError(
                "Please specify --dimensions when passing more than 2 dirs.")
    else:
        dims = args.dimensions

    if len(dims) != len(args.result_dirs):
        raise ValueError(
            f"Number of dimensions ({len(dims)}) must match number of "
            f"result directories ({len(args.result_dirs)}).")

    result_dirs = list(zip(dims, args.result_dirs))
    print(f"Building results table...")
    for dim, rd in result_dirs:
        print(f"  Dimension {dim}: {rd}")

    df = build_table(result_dirs, dims)

    # Save CSV
    csv_path = Path(args.output).with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")

    # Save pkl
    pkl_path = Path(args.output).with_suffix(".pkl")
    df.to_pickle(pkl_path)
    print(f"Saved {len(df)} rows to {pkl_path}")

    print(f"\nColumns: {list(df.columns)}")
    print(f"\nPreview:")
    print(df.to_string(max_rows=20))


if __name__ == "__main__":
    main()