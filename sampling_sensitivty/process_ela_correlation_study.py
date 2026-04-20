"""
Correlation studies between ELA CV metrics and classification results.

Studies:
  1) Per-instance consistency (classification) vs per-instance CV
  2) Per-instance CV vs mean classification accuracy across 5 folds
  3) Per-function CV vs mean classification accuracy across 5 folds
  4) Per-instance CV vs std of classification accuracy across 5 folds
  5) Training per-function CV vs mean classification accuracy across 5 folds
  6) Training per-function CV vs classification accuracy std across 5 folds

Reads:
  - ela_cv_results.pkl          (per-instance CV)
  - ela_cv_per_function.pkl     (per-function CV)
  - ela_cv_training_set.pkl     (training per-function CV)
  - ela_classification_results_allruns.h5  (classification results)

Usage:
  python correlation_studies.py /path/to/results --dimension 5
  python correlation_studies.py /path/to/dim2 /path/to/dim5 --dimensions 2 5
  python correlation_studies.py /path/to/results --dimension 5 --output correlations.csv
"""

import argparse
import pickle
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
from scipy import stats


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

FILTERED_FEATURES = []
for _grp, _feats in ELA_FEATURE_GROUPS.items():
    if _grp in OMIT_GROUPS:
        continue
    for _f in _feats:
        if _f not in OMIT_FEATURES:
            FILTERED_FEATURES.append((_grp, _f))


def parse_config_key(config_key):
    """Parse e.g. 'ilhs_50' -> ('ilhs', 50)."""
    parts = config_key.rsplit("_", 1)
    return parts[0], int(parts[1])


# ---------------------------------------------------------------------------
# CV aggregation helpers (reused from process_ela_summarized_df.py)
# ---------------------------------------------------------------------------

def collect_instance_cvs(instance_cv_data, config_key, dimension):
    """Per-instance CV: dict {(group, feat): list of CV values} across all instances."""
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


def collect_function_cvs(function_cv_data, config_key, dimension):
    """Per-function CV: dict {(group, feat): list of CV values} across 24 functions."""
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


def collect_training_fold_cvs(training_cv_data, config_key, n_runs_train):
    """Training per-function CV across folds and functions."""
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


def summarize_cv(feat_cvs, stat_fn=np.mean):
    """
    For each feature take the median CV across instances/functions,
    then apply stat_fn across all features.
    """
    medians = []
    for ff in FILTERED_FEATURES:
        vals = feat_cvs.get(ff, [])
        if vals:
            medians.append(np.median(vals))
    if not medians:
        return np.nan
    return stat_fn(medians)


def _detect_dimension(result_dir):
    """Auto-detect dimension from instance CV keys: (func, inst, dim)."""
    fpath = Path(result_dir) / "ela_cv_results.pkl"
    if fpath.exists():
        with open(fpath, "rb") as f:
            icv = pickle.load(f)
        for cfg_data in icv.values():
            for key in cfg_data.keys():
                if isinstance(key, tuple) and len(key) == 3:
                    return key[2]  # dim is third element
    # Fallback: try function CV
    fpath = Path(result_dir) / "ela_cv_per_function.pkl"
    if fpath.exists():
        with open(fpath, "rb") as f:
            fcv = pickle.load(f)
        for cfg_data in fcv.values():
            for key in cfg_data.keys():
                if isinstance(key, tuple) and len(key) == 2:
                    return key[1]  # dim is second element
    raise ValueError(f"Cannot auto-detect dimension from {result_dir}")


# ---------------------------------------------------------------------------
# Build the combined long-form table for correlations
# ---------------------------------------------------------------------------

def build_correlation_table(result_dirs):
    """
    Build a table with one row per (dimension, config_key, n_runs_train)
    containing all CV and classification metrics needed for the 6 studies.
    """
    rows = []

    for dimension, result_dir in result_dirs:
        result_dir = Path(result_dir)

        # Load data files
        instance_cv = {}
        function_cv = {}
        training_cv = {}

        fpath = result_dir / "ela_cv_results.pkl"
        if fpath.exists():
            with open(fpath, "rb") as f:
                instance_cv = pickle.load(f)
            print(f"  Loaded {fpath}")
        else:
            print(f"  WARNING: {fpath} not found")

        fpath = result_dir / "ela_cv_per_function.pkl"
        if fpath.exists():
            with open(fpath, "rb") as f:
                function_cv = pickle.load(f)
            print(f"  Loaded {fpath}")
        else:
            print(f"  WARNING: {fpath} not found")

        fpath = result_dir / "ela_cv_training_set.pkl"
        if fpath.exists():
            with open(fpath, "rb") as f:
                training_cv = pickle.load(f)
            print(f"  Loaded {fpath}")
        else:
            print(f"  WARNING: {fpath} not found")

        h5_path = result_dir / "ela_classification_results_allruns.h5"
        h5_file = None
        if h5_path.exists():
            h5_file = h5py.File(h5_path, "r")
            print(f"  Loaded {h5_path}")
        else:
            print(f"  WARNING: {h5_path} not found — skipping dimension {dimension}")
            continue

        config_keys = list(h5_file.keys())

        for config_key in config_keys:
            strategy, base_sample_size = parse_config_key(config_key)
            sample_size = base_sample_size * dimension

            # Enumerate n_runs_train values from HDF5
            n_runs_values = set()
            if config_key in h5_file:
                for subkey in h5_file[config_key].keys():
                    if subkey.startswith("n_runs_"):
                        n_runs_values.add(int(subkey.split("_")[-1]))
            if not n_runs_values:
                n_runs_values = {30}

            # Pre-compute per-instance and per-function CVs (independent of n_runs_train)
            feat_cvs_inst = collect_instance_cvs(instance_cv, config_key, dimension)
            cv_instance_mean = summarize_cv(feat_cvs_inst, np.mean)
            cv_instance_median = summarize_cv(feat_cvs_inst, np.median)

            feat_cvs_func = collect_function_cvs(function_cv, config_key, dimension)
            cv_function_mean = summarize_cv(feat_cvs_func, np.mean)
            cv_function_median = summarize_cv(feat_cvs_func, np.median)

            for n_runs_train in sorted(n_runs_values):
                subkey = f"n_runs_{n_runs_train:02d}"
                if config_key not in h5_file or subkey not in h5_file[config_key]:
                    continue

                grp = h5_file[config_key][subkey]

                # Classification metrics — median-test
                fold_accs = grp["fold_accuracies_median"][:]
                accuracy_mean = np.mean(fold_accs)
                accuracy_sd = np.std(fold_accs, ddof=1)

                # Classification metrics — all-runs-test
                fold_accs_allruns = grp["fold_accuracies_all_runs"][:]
                accuracy_allruns_mean = np.mean(fold_accs_allruns)
                accuracy_allruns_sd = np.std(fold_accs_allruns, ddof=1)

                # Per-instance consistency
                per_inst_consistency = grp["per_instance_consistency"][:]
                consistency_mean = float(grp.attrs["overall_consistency_mean"])
                consistency_std = float(grp.attrs["overall_consistency_std"])

                # Fold-level consistency
                fold_consistency = grp["fold_consistency"][:]

                # Training per-function CV
                feat_cvs_train = collect_training_fold_cvs(
                    training_cv, config_key, n_runs_train)
                cv_training_func_mean = summarize_cv(feat_cvs_train, np.mean)
                cv_training_func_median = summarize_cv(feat_cvs_train, np.median)

                row = {
                    "dimension": dimension,
                    "sampling_strategy": strategy,
                    "sample_size_per_dim": base_sample_size,
                    "sample_size": sample_size,
                    "n_runs_train": n_runs_train,
                    # CV metrics
                    "cv_instance_median_mean": cv_instance_mean,
                    "cv_instance_median_median": cv_instance_median,
                    "cv_function_median_mean": cv_function_mean,
                    "cv_function_median_median": cv_function_median,
                    "cv_training_func_median_mean": cv_training_func_mean,
                    "cv_training_func_median_median": cv_training_func_median,
                    # Classification metrics
                    "accuracy_mean": accuracy_mean,
                    "accuracy_sd": accuracy_sd,
                    "accuracy_allruns_mean": accuracy_allruns_mean,
                    "accuracy_allruns_sd": accuracy_allruns_sd,
                    "consistency_mean": consistency_mean,
                    "consistency_std": consistency_std,
                    "fold_consistency_mean": np.mean(fold_consistency),
                    # Per-fold arrays (stored for reference)
                    "fold_accuracies": fold_accs.tolist(),
                    "fold_consistency_values": fold_consistency.tolist(),
                }
                rows.append(row)

        h5_file.close()

    df = pd.DataFrame(rows)
    df.sort_values(
        by=["dimension", "sampling_strategy", "sample_size_per_dim", "n_runs_train"],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Correlation computation
# ---------------------------------------------------------------------------

def compute_correlation(x, y, label_x, label_y):
    """Compute Pearson and Spearman correlations, dropping NaN pairs."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean, y_clean = x[mask], y[mask]
    n = len(x_clean)
    if n < 3:
        return {
            "x_variable": label_x,
            "y_variable": label_y,
            "n": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }
    pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
    spearman_rho, spearman_p = stats.spearmanr(x_clean, y_clean)
    return {
        "x_variable": label_x,
        "y_variable": label_y,
        "n": n,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }


def run_correlation_studies(df):
    """
    Run the 6 correlation studies. Returns a summary DataFrame and a dict
    of per-study detail DataFrames for plotting.
    """
    results = []

    # Study 1: Per-instance consistency vs per-instance CV
    # Both are single scalars per config — consistency_mean vs cv_instance_median_mean
    results.append(compute_correlation(
        df["consistency_mean"].values,
        df["cv_instance_median_mean"].values,
        "per_instance_consistency_mean",
        "cv_instance_median_mean",
    ))
    results[-1]["study"] = "1: Instance consistency vs instance CV"

    # Also with median summary
    results.append(compute_correlation(
        df["consistency_mean"].values,
        df["cv_instance_median_median"].values,
        "per_instance_consistency_mean",
        "cv_instance_median_median",
    ))
    results[-1]["study"] = "1b: Instance consistency vs instance CV (median)"

    # Study 2: Per-instance CV vs mean classification accuracy
    results.append(compute_correlation(
        df["cv_instance_median_mean"].values,
        df["accuracy_mean"].values,
        "cv_instance_median_mean",
        "accuracy_mean",
    ))
    results[-1]["study"] = "2: Instance CV vs accuracy mean"

    results.append(compute_correlation(
        df["cv_instance_median_median"].values,
        df["accuracy_mean"].values,
        "cv_instance_median_median",
        "accuracy_mean",
    ))
    results[-1]["study"] = "2b: Instance CV (median) vs accuracy mean"

    # Study 3: Per-function CV vs mean classification accuracy
    results.append(compute_correlation(
        df["cv_function_median_mean"].values,
        df["accuracy_mean"].values,
        "cv_function_median_mean",
        "accuracy_mean",
    ))
    results[-1]["study"] = "3: Function CV vs accuracy mean"

    results.append(compute_correlation(
        df["cv_function_median_median"].values,
        df["accuracy_mean"].values,
        "cv_function_median_median",
        "accuracy_mean",
    ))
    results[-1]["study"] = "3b: Function CV (median) vs accuracy mean"

    # Study 4: Per-instance CV vs accuracy standard deviation
    results.append(compute_correlation(
        df["cv_instance_median_mean"].values,
        df["accuracy_sd"].values,
        "cv_instance_median_mean",
        "accuracy_sd",
    ))
    results[-1]["study"] = "4: Instance CV vs accuracy SD"

    results.append(compute_correlation(
        df["cv_instance_median_median"].values,
        df["accuracy_sd"].values,
        "cv_instance_median_median",
        "accuracy_sd",
    ))
    results[-1]["study"] = "4b: Instance CV (median) vs accuracy SD"

    # Study 5: Training per-function CV vs mean classification accuracy
    results.append(compute_correlation(
        df["cv_training_func_median_mean"].values,
        df["accuracy_mean"].values,
        "cv_training_func_median_mean",
        "accuracy_mean",
    ))
    results[-1]["study"] = "5: Training function CV vs accuracy mean"

    results.append(compute_correlation(
        df["cv_training_func_median_median"].values,
        df["accuracy_mean"].values,
        "cv_training_func_median_median",
        "accuracy_mean",
    ))
    results[-1]["study"] = "5b: Training function CV (median) vs accuracy mean"

    # Study 6: Training per-function CV vs accuracy standard deviation
    results.append(compute_correlation(
        df["cv_training_func_median_mean"].values,
        df["accuracy_sd"].values,
        "cv_training_func_median_mean",
        "accuracy_sd",
    ))
    results[-1]["study"] = "6: Training function CV vs accuracy SD"

    results.append(compute_correlation(
        df["cv_training_func_median_median"].values,
        df["accuracy_sd"].values,
        "cv_training_func_median_median",
        "accuracy_sd",
    ))
    results[-1]["study"] = "6b: Training function CV (median) vs accuracy SD"

    # ---- All-runs accuracy variants ----

    # Study 2ar: Per-instance CV vs all-runs accuracy mean
    results.append(compute_correlation(
        df["cv_instance_median_mean"].values,
        df["accuracy_allruns_mean"].values,
        "cv_instance_median_mean",
        "accuracy_allruns_mean",
    ))
    results[-1]["study"] = "2ar: Instance CV vs all-runs accuracy mean"

    results.append(compute_correlation(
        df["cv_instance_median_median"].values,
        df["accuracy_allruns_mean"].values,
        "cv_instance_median_median",
        "accuracy_allruns_mean",
    ))
    results[-1]["study"] = "2ar-b: Instance CV (median) vs all-runs accuracy mean"

    # Study 3ar: Per-function CV vs all-runs accuracy mean
    results.append(compute_correlation(
        df["cv_function_median_mean"].values,
        df["accuracy_allruns_mean"].values,
        "cv_function_median_mean",
        "accuracy_allruns_mean",
    ))
    results[-1]["study"] = "3ar: Function CV vs all-runs accuracy mean"

    results.append(compute_correlation(
        df["cv_function_median_median"].values,
        df["accuracy_allruns_mean"].values,
        "cv_function_median_median",
        "accuracy_allruns_mean",
    ))
    results[-1]["study"] = "3ar-b: Function CV (median) vs all-runs accuracy mean"

    # Study 4ar: Per-instance CV vs all-runs accuracy SD
    results.append(compute_correlation(
        df["cv_instance_median_mean"].values,
        df["accuracy_allruns_sd"].values,
        "cv_instance_median_mean",
        "accuracy_allruns_sd",
    ))
    results[-1]["study"] = "4ar: Instance CV vs all-runs accuracy SD"

    results.append(compute_correlation(
        df["cv_instance_median_median"].values,
        df["accuracy_allruns_sd"].values,
        "cv_instance_median_median",
        "accuracy_allruns_sd",
    ))
    results[-1]["study"] = "4ar-b: Instance CV (median) vs all-runs accuracy SD"

    # Study 5ar: Training function CV vs all-runs accuracy mean
    results.append(compute_correlation(
        df["cv_training_func_median_mean"].values,
        df["accuracy_allruns_mean"].values,
        "cv_training_func_median_mean",
        "accuracy_allruns_mean",
    ))
    results[-1]["study"] = "5ar: Training function CV vs all-runs accuracy mean"

    results.append(compute_correlation(
        df["cv_training_func_median_median"].values,
        df["accuracy_allruns_mean"].values,
        "cv_training_func_median_median",
        "accuracy_allruns_mean",
    ))
    results[-1]["study"] = "5ar-b: Training function CV (median) vs all-runs accuracy mean"

    # Study 6ar: Training function CV vs all-runs accuracy SD
    results.append(compute_correlation(
        df["cv_training_func_median_mean"].values,
        df["accuracy_allruns_sd"].values,
        "cv_training_func_median_mean",
        "accuracy_allruns_sd",
    ))
    results[-1]["study"] = "6ar: Training function CV vs all-runs accuracy SD"

    results.append(compute_correlation(
        df["cv_training_func_median_median"].values,
        df["accuracy_allruns_sd"].values,
        "cv_training_func_median_median",
        "accuracy_allruns_sd",
    ))
    results[-1]["study"] = "6ar-b: Training function CV (median) vs all-runs accuracy SD"

    summary_df = pd.DataFrame(results)
    # Reorder columns
    cols = ["study", "x_variable", "y_variable", "n",
            "pearson_r", "pearson_p", "spearman_rho", "spearman_p"]
    summary_df = summary_df[cols]
    return summary_df


def run_stratified_correlations(df):
    """
    Run the same 6 studies but stratified by dimension and by
    sampling_strategy, to see if correlations hold within subgroups.
    """
    study_specs = [
        ("1", "consistency_mean", "cv_instance_median_mean",
         "Instance consistency vs instance CV"),
        ("2", "cv_instance_median_mean", "accuracy_mean",
         "Instance CV vs accuracy mean"),
        ("3", "cv_function_median_mean", "accuracy_mean",
         "Function CV vs accuracy mean"),
        ("4", "cv_instance_median_mean", "accuracy_sd",
         "Instance CV vs accuracy SD"),
        ("5", "cv_training_func_median_mean", "accuracy_mean",
         "Training function CV vs accuracy mean"),
        ("6", "cv_training_func_median_mean", "accuracy_sd",
         "Training function CV vs accuracy SD"),
        ("2ar", "cv_instance_median_mean", "accuracy_allruns_mean",
         "Instance CV vs all-runs accuracy mean"),
        ("3ar", "cv_function_median_mean", "accuracy_allruns_mean",
         "Function CV vs all-runs accuracy mean"),
        ("4ar", "cv_instance_median_mean", "accuracy_allruns_sd",
         "Instance CV vs all-runs accuracy SD"),
        ("5ar", "cv_training_func_median_mean", "accuracy_allruns_mean",
         "Training function CV vs all-runs accuracy mean"),
        ("6ar", "cv_training_func_median_mean", "accuracy_allruns_sd",
         "Training function CV vs all-runs accuracy SD"),
    ]

    rows = []

    # By dimension
    for dim in sorted(df["dimension"].unique()):
        sub = df[df["dimension"] == dim]
        for study_id, x_col, y_col, desc in study_specs:
            r = compute_correlation(
                sub[x_col].values, sub[y_col].values, x_col, y_col)
            r["study"] = f"{study_id}: {desc}"
            r["stratification"] = f"dim={dim}"
            rows.append(r)

    # By sampling strategy
    for strat in sorted(df["sampling_strategy"].unique()):
        sub = df[df["sampling_strategy"] == strat]
        for study_id, x_col, y_col, desc in study_specs:
            r = compute_correlation(
                sub[x_col].values, sub[y_col].values, x_col, y_col)
            r["study"] = f"{study_id}: {desc}"
            r["stratification"] = f"strategy={strat}"
            rows.append(r)

    strat_df = pd.DataFrame(rows)
    cols = ["study", "stratification", "x_variable", "y_variable", "n",
            "pearson_r", "pearson_p", "spearman_rho", "spearman_p"]
    strat_df = strat_df[[c for c in cols if c in strat_df.columns]]
    return strat_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Correlation studies between ELA CV and classification.")
    parser.add_argument(
        "result_dirs", nargs="+", type=str,
        help="One or more result directories.")
    parser.add_argument(
        "--dimensions", "--dimension", nargs="+", type=int, default=None,
        help="Dimension for each directory (e.g. --dimensions 2 5).")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file base path (default: correlation_studies in first dir).")
    args = parser.parse_args()

    if args.dimensions is None:
        if len(args.result_dirs) == 1:
            # Auto-detect from instance CV keys
            dims = [_detect_dimension(args.result_dirs[0])]
            print(f"  Auto-detected dimension: {dims[0]}")
        elif len(args.result_dirs) == 2:
            dims = [_detect_dimension(d) for d in args.result_dirs]
            print(f"  Auto-detected dimensions: {dims}")
        else:
            raise ValueError("Specify --dimensions for >2 directories.")
    else:
        dims = args.dimensions

    if len(dims) != len(args.result_dirs):
        raise ValueError("Number of dimensions must match number of directories.")

    if args.output is None:
        args.output = str(Path(args.result_dirs[0]) / "correlation_studies")

    result_dirs = list(zip(dims, args.result_dirs))

    print("Building correlation data table...")
    for dim, rd in result_dirs:
        print(f"  Dimension {dim}: {rd}")

    df = build_correlation_table(result_dirs)

    if df.empty:
        print("ERROR: No data found. Check your input directories.")
        return

    print(f"\nData table: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # --- Overall correlations ---
    print("\n" + "=" * 70)
    print("OVERALL CORRELATION STUDIES")
    print("=" * 70)
    summary = run_correlation_studies(df)
    print(summary.to_string(index=False, float_format="%.4f"))

    # --- Stratified correlations ---
    print("\n" + "=" * 70)
    print("STRATIFIED CORRELATIONS (by dimension and sampling strategy)")
    print("=" * 70)
    strat = run_stratified_correlations(df)
    print(strat.to_string(index=False, float_format="%.4f"))

    # --- Save outputs ---
    base = Path(args.output)

    # Data table
    data_csv = base.with_name(base.stem + "_data.csv")
    # Drop list columns for CSV
    df_save = df.drop(columns=["fold_accuracies", "fold_consistency_values"],
                      errors="ignore")
    df_save.to_csv(data_csv, index=False)
    print(f"\nData table saved to {data_csv}")

    # Summary correlations
    summary_csv = base.with_name(base.stem + "_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Summary saved to {summary_csv}")

    # Stratified correlations
    strat_csv = base.with_name(base.stem + "_stratified.csv")
    strat.to_csv(strat_csv, index=False)
    print(f"Stratified results saved to {strat_csv}")

    # Also save as pickle for downstream use
    pkl_path = base.with_suffix(".pkl")
    output = {
        "data": df,
        "summary": summary,
        "stratified": strat,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(output, f)
    print(f"All results saved to {pkl_path}")


if __name__ == "__main__":
    main()