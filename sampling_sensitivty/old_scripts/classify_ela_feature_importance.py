"""
Feature importance analysis for ELA classification.

Re-trains the same Random Forest models as classify_ela_allruns.py and
extracts feature importances (MDI = Mean Decrease in Impurity) per fold.

Analyses:
  1) Overall feature ranking (mean importance across folds, configs)
  2) Per-feature-group importance aggregation
  3) Importance shifts across sampling strategies
  4) Importance shifts across sample sizes
  5) Importance shifts across dimensions (if multiple dirs provided)

Usage:
  python feature_importance_analysis.py /path/to/ela/pkl/dir
  python feature_importance_analysis.py /path/to/ela/pkl/dir --dimension 2
  python feature_importance_analysis.py /path/to/dim2 /path/to/dim5 --dimensions 2 5
  python feature_importance_analysis.py /path/to/ela/pkl/dir --n-runs-train 30
  python feature_importance_analysis.py /path/to/ela/pkl/dir --configs ilhs_50 sobol_50
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import gc
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from scipy import stats


# ---------------------------------------------------------------------------
# Configuration (must match classify_ela_allruns.py exactly)
# ---------------------------------------------------------------------------

N_FUNCTIONS = 24
N_INSTANCES = 100
N_RUNS = 30
N_FOLDS = 5
RANDOM_STATE = 42

DEFAULT_N_RUNS_TRAIN = [30]  # Default: just use all runs

RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = None
RF_MAX_FEATURES = "sqrt"
RF_MIN_SAMPLES_LEAF = 1

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
FEATURE_TO_GROUP = {}
for _grp, _feats in ELA_FEATURE_GROUPS.items():
    if _grp in OMIT_GROUPS:
        continue
    for _f in _feats:
        if _f not in OMIT_FEATURES:
            FILTERED_FEATURES.append((_grp, _f))
            FEATURE_TO_GROUP[_f] = _grp
N_FEATURES = len(FILTERED_FEATURES)

# Feature names list (flat) for DataFrame columns
FEATURE_NAMES = [f for _, f in FILTERED_FEATURES]
GROUP_NAMES = [g for g, _ in FILTERED_FEATURES]

# Unique groups in order
UNIQUE_GROUPS = list(dict.fromkeys(GROUP_NAMES))

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


def parse_config_key(config_key):
    parts = config_key.rsplit("_", 1)
    return parts[0], int(parts[1])


# ---------------------------------------------------------------------------
# Data loading (identical to classify_ela_allruns.py)
# ---------------------------------------------------------------------------

def build_instance_data(data, dimension):
    """Build per-instance feature matrices, identical to classification script."""
    n_total = N_FUNCTIONS * N_INSTANCES
    median_features = np.empty((n_total, N_FEATURES))
    all_run_features = np.empty((n_total, N_RUNS, N_FEATURES))
    labels = np.empty(n_total, dtype=int)

    for func_idx in range(N_FUNCTIONS):
        func_id = func_idx + 1
        for inst_idx in range(N_INSTANCES):
            inst_id = inst_idx + 1
            row = func_idx * N_INSTANCES + inst_idx
            instance_key = (func_id, inst_id, dimension)
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
# Feature importance extraction
# ---------------------------------------------------------------------------

def extract_importances(median_features, all_run_features, labels,
                        n_runs_train, n_jobs=1):
    """
    Re-run the exact same 5-fold CV as classify_ela_allruns.py and
    extract feature importances from each fold's RF model.

    Returns array of shape (N_FOLDS, N_FEATURES).
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    fold_importances = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(
            np.zeros(len(labels)), labels)):

        n_train = len(train_idx)

        # Sample runs (identical logic to classification script)
        if n_runs_train >= N_RUNS:
            sampled_runs = np.tile(np.arange(N_RUNS), (n_train, 1))
        else:
            sampled_runs = np.empty((n_train, n_runs_train), dtype=int)
            for i in range(n_train):
                sampled_runs[i] = rng.choice(N_RUNS, size=n_runs_train,
                                             replace=False)

        row_idx = np.arange(n_train)[:, None]
        X_train = all_run_features[train_idx][row_idx, sampled_runs, :]
        X_train = X_train.reshape(n_train * n_runs_train, N_FEATURES)
        y_train = np.repeat(labels[train_idx], n_runs_train)

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

        fold_importances.append(rf.feature_importances_.copy())

        del X_train, X_train_scaled, rf
        gc.collect()

        print(f"      Fold {fold_idx + 1}/{N_FOLDS} done")

    return np.array(fold_importances)  # (N_FOLDS, N_FEATURES)


# ---------------------------------------------------------------------------
# Auto-detect dimension
# ---------------------------------------------------------------------------

def detect_dimension(data):
    """Detect dimension from the first key in the ELA data dict."""
    for key in data.keys():
        if isinstance(key, tuple) and len(key) == 3:
            return key[2]
    raise ValueError("Cannot detect dimension from data keys")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(input_dirs, dimensions=None, output_dir=None, configs=None,
         n_runs_train_list=None, n_jobs=1):

    if n_runs_train_list is None:
        n_runs_train_list = DEFAULT_N_RUNS_TRAIN

    # Pair up directories and dimensions
    if dimensions is None:
        dimensions = [None] * len(input_dirs)  # will auto-detect

    all_importance_rows = []

    for input_dir, dimension in zip(input_dirs, dimensions):
        input_dir = Path(input_dir)

        if configs:
            config_keys = [c for c in configs if c in ELA_FILES]
        else:
            # Find which configs actually exist in the directory
            config_keys = [k for k, v in ELA_FILES.items()
                           if (input_dir / v).exists()]

        if not config_keys:
            print(f"WARNING: No ELA pkl files found in {input_dir}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing directory: {input_dir}")
        print(f"Configs found: {len(config_keys)}")
        print(f"n_runs_train sweep: {n_runs_train_list}")
        print(f"{'=' * 60}")

        for config_key in config_keys:
            filepath = input_dir / ELA_FILES[config_key]
            if not filepath.exists():
                print(f"  WARNING: {filepath} not found, skipping.")
                continue

            strategy, base_sample_size = parse_config_key(config_key)

            print(f"\n  --- {config_key} ---")
            print(f"  Loading {filepath}...")

            with open(filepath, "rb") as f:
                data = pickle.load(f)

            # Auto-detect dimension if not provided
            if dimension is None:
                dim = detect_dimension(data)
                print(f"  Auto-detected dimension: {dim}")
            else:
                dim = dimension

            sample_size = base_sample_size * dim

            print(f"  Building feature matrices...")
            median_features, all_run_features, labels = \
                build_instance_data(data, dim)
            del data

            for n_runs_train in n_runs_train_list:
                print(f"\n    n_runs_train = {n_runs_train}")
                fold_imps = extract_importances(
                    median_features, all_run_features, labels,
                    n_runs_train=n_runs_train, n_jobs=n_jobs,
                )
                # fold_imps shape: (N_FOLDS, N_FEATURES)

                # Store per-fold rows
                for fold_idx in range(N_FOLDS):
                    row = {
                        "dimension": dim,
                        "sampling_strategy": strategy,
                        "sample_size_per_dim": base_sample_size,
                        "sample_size": sample_size,
                        "n_runs_train": n_runs_train,
                        "fold": fold_idx,
                    }
                    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
                        row[feat_name] = fold_imps[fold_idx, feat_idx]
                    all_importance_rows.append(row)

                # Print top-10 features for this config (mean across folds)
                mean_imp = fold_imps.mean(axis=0)
                top_idx = np.argsort(mean_imp)[::-1][:10]
                print(f"    Top 10 features (mean importance across folds):")
                for rank, idx in enumerate(top_idx):
                    print(f"      {rank+1:2d}. {FEATURE_NAMES[idx]:40s} "
                          f"{mean_imp[idx]:.4f} ({GROUP_NAMES[idx]})")

            del median_features, all_run_features
            gc.collect()

    # Build the full DataFrame
    df = pd.DataFrame(all_importance_rows)
    meta_cols = ["dimension", "sampling_strategy", "sample_size_per_dim",
                 "sample_size", "n_runs_train", "fold"]
    df.sort_values(by=meta_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Compute analysis summaries ---
    print(f"\n\n{'=' * 60}")
    print("ANALYSIS SUMMARIES")
    print(f"{'=' * 60}")

    # 1) Overall feature ranking
    overall_ranking = compute_overall_ranking(df)
    print("\n--- 1) Overall Feature Ranking (mean importance) ---")
    print(overall_ranking.to_string(index=False, float_format="%.4f"))

    # 2) Per-group importance
    group_importance = compute_group_importance(df)
    print("\n--- 2) Feature Group Importance ---")
    print(group_importance.to_string(index=False, float_format="%.4f"))

    # 3) Importance by sampling strategy
    strategy_ranking = compute_ranking_by(df, "sampling_strategy")
    print("\n--- 3) Top Features by Sampling Strategy ---")
    print(strategy_ranking.to_string(index=False, float_format="%.4f"))

    # 4) Importance by sample size
    size_ranking = compute_ranking_by(df, "sample_size_per_dim")
    print("\n--- 4) Top Features by Sample Size ---")
    print(size_ranking.to_string(index=False, float_format="%.4f"))

    # 5) Group importance by strategy
    group_by_strategy = compute_group_by(df, "sampling_strategy")
    print("\n--- 5) Group Importance by Sampling Strategy ---")
    print(group_by_strategy.to_string(index=False, float_format="%.4f"))

    # 6) Group importance by sample size
    group_by_size = compute_group_by(df, "sample_size_per_dim")
    print("\n--- 6) Group Importance by Sample Size ---")
    print(group_by_size.to_string(index=False, float_format="%.4f"))

    # 7) If multiple dimensions, compare
    if df["dimension"].nunique() > 1:
        dim_ranking = compute_ranking_by(df, "dimension")
        print("\n--- 7) Top Features by Dimension ---")
        print(dim_ranking.to_string(index=False, float_format="%.4f"))

        group_by_dim = compute_group_by(df, "dimension")
        print("\n--- 8) Group Importance by Dimension ---")
        print(group_by_dim.to_string(index=False, float_format="%.4f"))

    # 8) Rank stability analysis
    rank_stability = compute_rank_stability(df)
    print("\n--- 9) Rank Stability Across Strategies ---")
    print(rank_stability.to_string(index=False, float_format="%.4f"))

    # --- Save outputs ---
    if output_dir is None:
        output_dir = input_dirs[0]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = output_dir / "feature_importance"

    # Raw per-fold importances
    csv_path = base.with_name(base.stem + "_raw.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nRaw importances saved to {csv_path}")

    # All analyses as pickle
    pkl_path = base.with_suffix(".pkl")
    output = {
        "raw": df,
        "overall_ranking": overall_ranking,
        "group_importance": group_importance,
        "ranking_by_strategy": strategy_ranking,
        "ranking_by_size": size_ranking,
        "group_by_strategy": group_by_strategy,
        "group_by_size": group_by_size,
        "rank_stability": rank_stability,
    }
    if df["dimension"].nunique() > 1:
        output["ranking_by_dimension"] = dim_ranking
        output["group_by_dimension"] = group_by_dim

    with open(pkl_path, "wb") as f:
        pickle.dump(output, f)
    print(f"All results saved to {pkl_path}")

    return output


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_overall_ranking(df):
    """Rank features by mean importance across all folds and configs."""
    feat_means = df[FEATURE_NAMES].mean()
    feat_stds = df[FEATURE_NAMES].std()

    ranking = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "group": GROUP_NAMES,
        "mean_importance": feat_means.values,
        "std_importance": feat_stds.values,
    })
    ranking.sort_values("mean_importance", ascending=False, inplace=True)
    ranking["rank"] = range(1, len(ranking) + 1)
    ranking = ranking[["rank", "feature", "group",
                        "mean_importance", "std_importance"]]
    ranking.reset_index(drop=True, inplace=True)
    return ranking


def compute_group_importance(df):
    """Sum importances by feature group."""
    group_rows = []
    for grp in UNIQUE_GROUPS:
        grp_feats = [f for f, g in zip(FEATURE_NAMES, GROUP_NAMES) if g == grp]
        grp_sum = df[grp_feats].sum(axis=1)
        group_rows.append({
            "group": grp,
            "n_features": len(grp_feats),
            "total_importance_mean": grp_sum.mean(),
            "total_importance_std": grp_sum.std(),
            "per_feature_importance_mean": df[grp_feats].mean().mean(),
        })
    result = pd.DataFrame(group_rows)
    result.sort_values("total_importance_mean", ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def compute_ranking_by(df, by_column, top_n=10):
    """Top-N features for each value of by_column."""
    rows = []
    for val in sorted(df[by_column].unique()):
        sub = df[df[by_column] == val]
        feat_means = sub[FEATURE_NAMES].mean()
        top_idx = feat_means.argsort()[::-1][:top_n]
        for rank, idx in enumerate(top_idx):
            rows.append({
                by_column: val,
                "rank": rank + 1,
                "feature": FEATURE_NAMES[idx],
                "group": GROUP_NAMES[idx],
                "mean_importance": feat_means.iloc[idx],
            })
    return pd.DataFrame(rows)


def compute_group_by(df, by_column):
    """Group importance for each value of by_column."""
    rows = []
    for val in sorted(df[by_column].unique()):
        sub = df[df[by_column] == val]
        for grp in UNIQUE_GROUPS:
            grp_feats = [f for f, g in zip(FEATURE_NAMES, GROUP_NAMES)
                         if g == grp]
            grp_sum = sub[grp_feats].sum(axis=1)
            rows.append({
                by_column: val,
                "group": grp,
                "total_importance_mean": grp_sum.mean(),
                "per_feature_importance_mean": sub[grp_feats].mean().mean(),
            })
    return pd.DataFrame(rows)


def compute_rank_stability(df):
    """
    For each feature, compute its rank within each strategy and report
    rank mean, std, and range to show how stable the ranking is.
    """
    strategy_ranks = {}
    for strat in sorted(df["sampling_strategy"].unique()):
        sub = df[df["sampling_strategy"] == strat]
        feat_means = sub[FEATURE_NAMES].mean()
        ranks = feat_means.rank(ascending=False).astype(int)
        strategy_ranks[strat] = ranks

    rank_df = pd.DataFrame(strategy_ranks, index=FEATURE_NAMES)

    result = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "group": GROUP_NAMES,
        "rank_mean": rank_df.mean(axis=1).values,
        "rank_std": rank_df.std(axis=1).values,
        "rank_min": rank_df.min(axis=1).values.astype(int),
        "rank_max": rank_df.max(axis=1).values.astype(int),
        "rank_range": (rank_df.max(axis=1) - rank_df.min(axis=1)).values.astype(int),
    })

    # Add per-strategy rank columns
    for strat in sorted(df["sampling_strategy"].unique()):
        result[f"rank_{strat}"] = strategy_ranks[strat].values.astype(int)

    result.sort_values("rank_mean", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and analyse RF feature importances for ELA "
                    "classification.")
    parser.add_argument(
        "input_dirs", nargs="+", type=str,
        help="One or more directories containing ELA pkl files.")
    parser.add_argument(
        "--dimensions", "--dimension", nargs="+", type=int, default=None,
        help="Dimension for each input directory.")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: first input dir).")
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Specific configs to process (e.g. ilhs_50 sobol_50).")
    parser.add_argument(
        "--n-runs-train", nargs="+", type=int, default=None,
        help=f"n_runs_train values (default: {DEFAULT_N_RUNS_TRAIN}).")
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel jobs for Random Forest (default: 1).")
    args = parser.parse_args()

    main(
        input_dirs=args.input_dirs,
        dimensions=args.dimensions,
        output_dir=args.output_dir,
        configs=args.configs,
        n_runs_train_list=args.n_runs_train,
        n_jobs=args.n_jobs,
    )