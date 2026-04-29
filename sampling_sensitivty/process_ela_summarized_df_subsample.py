"""
Tabulate ELA subsample classification results into a single DataFrame.

Reads:
  - ela_classification_subsample.h5  (from classify_ela_subsample.py)
  - ela_cv_results.pkl               (per-instance CV, optional)
  - ela_cv_per_function.pkl          (per-function CV, optional)
  - ela_cv_training_set.pkl          (per-fold/function CV, optional)

For each dimension directory, reads the HDF5 file and extracts per-config,
per-(n_instances_train, n_runs_train) results, and optionally joins CV
statistics from the pickle files.

Produces a DataFrame with one row per
  (dimension, sampling_strategy, sample_size_per_dim, n_instances_train,
   n_runs_train, fold)

with columns for function evaluation budget, CV statistics, accuracies,
and consistency.

Usage:
  python tabulate_subsample_results.py /path/to/dim2/results /path/to/dim5/results
  python tabulate_subsample_results.py /path/to/dim2/results --dimensions 2
  python tabulate_subsample_results.py /path/to/dim2 /path/to/dim5 --dimensions 2 5 --output results.csv
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
    ],
}

# Build the candidate feature list (group, feat_name) — same order as
# classify_ela_subsample.py
CANDIDATE_FEATURES = []
for _grp, _feats in ELA_FEATURE_GROUPS.items():
    if _grp in OMIT_GROUPS:
        continue
    for _f in _feats:
        CANDIDATE_FEATURES.append((_grp, _f))


# ---------------------------------------------------------------------------
# CV aggregation helpers
# ---------------------------------------------------------------------------

def collect_all_feature_cvs_instance(instance_cv_data, config_key, dimension):
    """
    From per-instance CV data: cv[config][(func,inst,dim)][group][feat]
    Returns dict {(group, feat): list of CV values} across all instances.
    """
    config_data = instance_cv_data.get(config_key, {})
    feat_cvs = {ff: [] for ff in CANDIDATE_FEATURES}

    for (func, inst, dim), inst_cv in config_data.items():
        if dim != dimension:
            continue
        for grp, feat in CANDIDATE_FEATURES:
            if grp in inst_cv and feat in inst_cv[grp]:
                v = inst_cv[grp][feat]
                if np.isfinite(v):
                    feat_cvs[(grp, feat)].append(v)

    return feat_cvs


def collect_all_feature_cvs_function(function_cv_data, config_key, dimension):
    """
    From per-function CV data: cv[config][(func,dim)][group][feat]
    Returns dict {(group, feat): list of CV values} across 24 functions.
    """
    config_data = function_cv_data.get(config_key, {})
    feat_cvs = {ff: [] for ff in CANDIDATE_FEATURES}

    for (func, dim), func_cv in config_data.items():
        if dim != dimension:
            continue
        for grp, feat in CANDIDATE_FEATURES:
            if grp in func_cv and feat in func_cv[grp]:
                v = func_cv[grp][feat]
                if np.isfinite(v):
                    feat_cvs[(grp, feat)].append(v)

    return feat_cvs


def collect_all_feature_cvs_training_folds(training_cv_data, config_key,
                                           n_runs_train):
    """
    From training-set CV data:
      results[config][n_runs]['per_fold_per_function'][fold][func_id][group][feat]
    Returns dict {(group, feat): list of CV values} across functions × folds.
    """
    config_data = training_cv_data.get(config_key, {})
    nrt_data = config_data.get(n_runs_train, {})
    pfpf = nrt_data.get("per_fold_per_function", {})

    feat_cvs = {ff: [] for ff in CANDIDATE_FEATURES}

    for fold_idx, fold_data in pfpf.items():
        for func_id, func_cv in fold_data.items():
            for grp, feat in CANDIDATE_FEATURES:
                if grp in func_cv and feat in func_cv[grp]:
                    v = func_cv[grp][feat]
                    if np.isfinite(v):
                        feat_cvs[(grp, feat)].append(v)

    return feat_cvs


def aggregate_median_then_stat(feat_cvs, stat_fn):
    """
    For each feature, take the median of its CV values across
    instances/functions, then apply stat_fn (np.mean or np.median)
    across all features.
    """
    medians = []
    for ff in CANDIDATE_FEATURES:
        vals = feat_cvs.get(ff, [])
        if vals:
            medians.append(np.median(vals))
    if not medians:
        return np.nan
    return stat_fn(medians)


def parse_config_key(config_key):
    """
    Parse e.g. 'ilhs_50' -> ('ilhs', 50).
    Handles multi-part strategies like 'cma_random_50' -> ('cma_random', 50).
    """
    parts = config_key.rsplit("_", 1)
    return parts[0], int(parts[1])


def parse_subkey(subkey):
    """
    Parse e.g. 'inst_05_runs_03' -> (5, 3).
    """
    # Format: inst_XX_runs_YY
    parts = subkey.split("_")
    n_inst = int(parts[1])
    n_runs = int(parts[3])
    return n_inst, n_runs


def build_table(result_dirs):
    """
    Build the combined results table from subsample classification HDF5
    files and optional CV pickle files.

    Each fold is a separate row in the output DataFrame.

    Parameters
    ----------
    result_dirs : list of (dimension, path) tuples

    Returns
    -------
    pd.DataFrame
    """
    rows = []

    for dimension, result_dir in result_dirs:
        result_dir = Path(result_dir)

        # --- Load classification HDF5 ---
        h5_path = result_dir / "ela_classification_subsample.h5"
        if not h5_path.exists():
            print(f"  WARNING: {h5_path} not found, skipping.")
            continue
        print(f"  Loading {h5_path}")

        # --- Load optional CV pickle files ---
        instance_cv = {}
        function_cv = {}
        training_cv = {}

        instance_cv_path = result_dir / "ela_cv_results.pkl"
        if instance_cv_path.exists():
            with open(instance_cv_path, "rb") as f:
                instance_cv = pickle.load(f)
            print(f"  Loaded {instance_cv_path}")
        else:
            print(f"  (no per-instance CV: {instance_cv_path.name})")

        function_cv_path = result_dir / "ela_cv_per_function.pkl"
        if function_cv_path.exists():
            with open(function_cv_path, "rb") as f:
                function_cv = pickle.load(f)
            print(f"  Loaded {function_cv_path}")
        else:
            print(f"  (no per-function CV: {function_cv_path.name})")

        training_cv_path = result_dir / "ela_cv_training_set.pkl"
        if training_cv_path.exists():
            with open(training_cv_path, "rb") as f:
                training_cv = pickle.load(f)
            print(f"  Loaded {training_cv_path}")
        else:
            print(f"  (no training-set CV: {training_cv_path.name})")

        with h5py.File(h5_path, "r") as h5:
            for config_key in h5.keys():
                config_grp = h5[config_key]
                strategy, sample_size_per_dim = parse_config_key(config_key)

                # Read feature metadata if available
                n_features = int(config_grp.attrs.get("n_features", -1))

                # Read omitted features (NaN/Inf) if stored
                if "omitted_features" in config_grp:
                    omitted_features = list(
                        config_grp["omitted_features"].asstr()[:])
                else:
                    omitted_features = []
                omitted_features_str = ", ".join(omitted_features) \
                    if omitted_features else ""

                # --- Per-instance CV (same for all grid points) ---
                feat_cvs_inst = collect_all_feature_cvs_instance(
                    instance_cv, config_key, dimension)
                cv_inst_median_mean = aggregate_median_then_stat(
                    feat_cvs_inst, np.mean)
                cv_inst_median_median = aggregate_median_then_stat(
                    feat_cvs_inst, np.median)

                # --- Per-function CV (same for all grid points) ---
                feat_cvs_func = collect_all_feature_cvs_function(
                    function_cv, config_key, dimension)
                cv_func_median_mean = aggregate_median_then_stat(
                    feat_cvs_func, np.mean)
                cv_func_median_median = aggregate_median_then_stat(
                    feat_cvs_func, np.median)

                # Iterate over inst_XX_runs_YY subgroups
                for subkey in sorted(config_grp.keys()):
                    # Skip metadata datasets (kept_features, etc.)
                    if not subkey.startswith("inst_"):
                        continue

                    sub_grp = config_grp[subkey]
                    n_inst, n_runs = parse_subkey(subkey)

                    # Read attributes
                    attrs = dict(sub_grp.attrs)
                    n_feval_train = int(attrs.get("n_feval_train", 0))
                    dim_stored = int(attrs.get("dimension", dimension))

                    # Read fold-level arrays
                    fold_acc_median = sub_grp["fold_accuracies_median"][:]
                    fold_acc_allruns = sub_grp["fold_accuracies_all_runs"][:]
                    fold_consistency = sub_grp["fold_consistency"][:]

                    # --- Training-set CV (depends on n_runs_train) ---
                    feat_cvs_train = collect_all_feature_cvs_training_folds(
                        training_cv, config_key, n_runs)
                    cv_train_median_mean = aggregate_median_then_stat(
                        feat_cvs_train, np.mean)
                    cv_train_median_median = aggregate_median_then_stat(
                        feat_cvs_train, np.median)

                    n_folds = len(fold_acc_median)
                    for fold_idx in range(n_folds):
                        row = {
                            "dimension": dim_stored,
                            "sampling_strategy": strategy,
                            "sample_size_per_dim": sample_size_per_dim,
                            "n_instances_train": n_inst,
                            "n_runs_train": n_runs,
                            "n_feval_train": n_feval_train,
                            "fold": fold_idx,
                            "n_features": n_features,
                            "n_omitted_features": len(omitted_features),
                            "omitted_features": omitted_features_str,
                            # CV columns
                            "cv_instance_median_mean": cv_inst_median_mean,
                            "cv_instance_median_median": cv_inst_median_median,
                            "cv_function_median_mean": cv_func_median_mean,
                            "cv_function_median_median": cv_func_median_median,
                            "cv_training_folds_median_mean":
                                cv_train_median_mean,
                            "cv_training_folds_median_median":
                                cv_train_median_median,
                            # Per-fold classification accuracy
                            "accuracy_median": float(
                                fold_acc_median[fold_idx]),
                            "accuracy_allruns": float(
                                fold_acc_allruns[fold_idx]),
                            # Per-fold consistency
                            "consistency": float(
                                fold_consistency[fold_idx]),
                        }
                        rows.append(row)

    df = pd.DataFrame(rows, columns=[
        "dimension",
        "sampling_strategy",
        "sample_size_per_dim",
        "n_instances_train",
        "n_runs_train",
        "n_feval_train",
        "fold",
        "n_features",
        "n_omitted_features",
        "omitted_features",
        "cv_instance_median_mean",
        "cv_instance_median_median",
        "cv_function_median_mean",
        "cv_function_median_median",
        "cv_training_folds_median_mean",
        "cv_training_folds_median_median",
        "accuracy_median",
        "accuracy_allruns",
        "consistency",
    ])

    df.sort_values(
        by=["dimension", "sampling_strategy", "sample_size_per_dim",
            "n_instances_train", "n_runs_train", "fold"],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Tabulate ELA subsample classification results into a "
                    "DataFrame.",
    )
    parser.add_argument(
        "result_dirs", nargs="+", type=str,
        help="One or more result directories containing "
             "ela_classification_subsample.h5.",
    )
    parser.add_argument(
        "--dimensions", nargs="+", type=int, default=None,
        help="Dimension for each result directory (e.g. --dimensions 2 5). "
             "Must match number of result_dirs. If not given, defaults to "
             "[2] for one dir or [2, 5] for two dirs.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file base path (without extension). Defaults to "
             "subsample_results_table in the first input directory.",
    )
    args = parser.parse_args()

    # Default output location
    if args.output is None:
        args.output = str(
            Path(args.result_dirs[0]) / "subsample_results_table")

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
    print(f"Building subsample results table...")
    for dim, rd in result_dirs:
        print(f"  Dimension {dim}: {rd}")

    df = build_table(result_dirs)

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
    print(df.to_string(max_rows=30))


if __name__ == "__main__":
    main()