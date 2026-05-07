"""
RQ2 — Analysis 1b: Budget-Conditioned Factor Importance
========================================================

Helper functions for Jupyter notebook usage.

Usage in notebook:
    from rq2_factor_importance import *

    # df is your fold_agg_df (already filtered to one dimension)
    df, tier_labels = assign_budget_tiers(df, n_tiers=6)
    imp_df = compute_factor_importance(df, metric="acc_mean", tier_labels=tier_labels)

    # Individual plots
    plot_eta_squared(imp_df)
    plot_marginal_range(imp_df)
    plot_marginal_means_per_tier(df, tier_labels=tier_labels)
    plot_eta_squared_share(imp_df)

    # Or all at once
    imp_df = run_all(df, metric="acc_mean", n_tiers=6)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STRATEGY_COLORS = {
    "cma_random": "#9467bd", "uniform": "#d62728", "lhs": "#ff7f0e",
    "ilhs": "#1f77b4", "lhs_rcd": "#8c564b", "sobol": "#2ca02c",
}
STRATEGY_LABELS = {
    "cma_random": "CMA-ES", "uniform": "Uniform", "lhs": "LHS",
    "ilhs": "iLHS", "lhs_rcd": "LHS-RCD", "sobol": "Sobol",
}
STRATEGY_ORDER = ["cma_random", "uniform", "lhs", "lhs_rcd", "ilhs", "sobol"]

FACTOR_COLS = [
    "sampling_strategy", "sample_size_per_dim",
    "n_instances_train", "n_runs_train",
]
FACTOR_LABELS = {
    "sampling_strategy":   "Strategy",
    "sample_size_per_dim": "Sample size",
    "n_instances_train":   "Instances",
    "n_runs_train":        "Runs",
}
FACTOR_COLORS = {
    "sampling_strategy":   "#4C72B0",
    "sample_size_per_dim": "#DD8452",
    "n_instances_train":   "#55A868",
    "n_runs_train":        "#C44E52",
}

DEFAULT_METRIC = "acc_mean"


# =========================================================================
# Budget tiers
# =========================================================================

def assign_budget_tiers(df, n_tiers=4):
    """
    Add 'budget_tier' column to df using pd.qcut so that each tier
    contains roughly the same number of rows.

    Parameters
    ----------
    df : DataFrame with 'n_feval_train' column
    n_tiers : int  (default 4)

    Returns
    -------
    df : DataFrame (copy) with 'budget_tier' column (Interval dtype)
    tier_labels : list  ordered tier categories
    """
    df = df.copy()
    df["budget_tier"] = pd.qcut(df["n_feval_train"], q=n_tiers)
    tier_labels = df["budget_tier"].cat.categories.tolist()
    return df, tier_labels


# =========================================================================
# Eta-squared
# =========================================================================

def eta_squared_oneway(group_labels, values):
    """
    η² = SS_between / SS_total for a one-way layout.
    """
    vals = pd.Series(values, dtype=float)
    grand_mean = vals.mean()
    ss_total = ((vals - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0

    groups = pd.Series(group_labels)
    ss_between = 0.0
    for _, idx in groups.groupby(groups).groups.items():
        grp_vals = vals.iloc[idx]
        ss_between += len(grp_vals) * (grp_vals.mean() - grand_mean) ** 2

    return ss_between / ss_total


# =========================================================================
# Factor importance table
# =========================================================================

def compute_factor_importance(df, metric=DEFAULT_METRIC, tier_labels=None):
    """
    For each budget tier × factor, compute η² and marginal range.

    Parameters
    ----------
    df : DataFrame with 'budget_tier' column (from assign_budget_tiers)
    metric : str, column name for the accuracy metric
    tier_labels : list[str], ordered tier labels

    Returns
    -------
    DataFrame with columns:
        budget_tier, factor, factor_label, eta_sq, marginal_range,
        n_rows, n_levels
    """
    if tier_labels is None:
        tier_labels = df["budget_tier"].cat.categories.tolist()

    rows = []
    for tier in tier_labels:
        sub = df[df["budget_tier"] == tier]
        if len(sub) < 3:
            continue

        for col in FACTOR_COLS:
            levels = sub[col].nunique()
            if levels < 2:
                eta = 0.0
                marg_range = 0.0
            else:
                eta = eta_squared_oneway(sub[col].values, sub[metric].values)
                level_means = sub.groupby(col)[metric].mean()
                marg_range = level_means.max() - level_means.min()

            rows.append({
                "budget_tier": tier,
                "factor": col,
                "factor_label": FACTOR_LABELS[col],
                "eta_sq": eta,
                "marginal_range": marg_range,
                "n_rows": len(sub),
                "n_levels": levels,
            })

    return pd.DataFrame(rows)


# =========================================================================
# Plot 1 — η² grouped bar chart
# =========================================================================

def plot_eta_squared(imp_df, metric=DEFAULT_METRIC, ax=None):
    """
    Grouped bar chart: x = budget tier, bars = factors, height = η².
    """
    tiers = imp_df["budget_tier"].unique()
    n_tiers = len(tiers)
    n_factors = len(FACTOR_COLS)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(10, n_tiers * 1.8), 5))

    bar_width = 0.8 / n_factors
    x = np.arange(n_tiers)

    for i, col in enumerate(FACTOR_COLS):
        sub = imp_df[imp_df["factor"] == col].set_index("budget_tier")
        vals = [sub.loc[t, "eta_sq"] if t in sub.index else 0.0 for t in tiers]
        offset = (i - n_factors / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, width=bar_width,
               color=FACTOR_COLORS[col], label=FACTOR_LABELS[col],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(tiers, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Budget tier (function evaluations)")
    ax.set_ylabel("η² (variance explained)")
    ax.set_title(f"Factor importance by budget tier — {metric}")
    ax.legend(title="Design factor", loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.15))

    plt.tight_layout()
    return ax


# =========================================================================
# Plot 2 — Marginal range
# =========================================================================

def plot_marginal_range(imp_df, metric=DEFAULT_METRIC, ax=None):
    """
    Grouped bar chart: height = marginal accuracy range
    (best − worst level mean) per factor within each tier.
    """
    tiers = imp_df["budget_tier"].unique()
    n_tiers = len(tiers)
    n_factors = len(FACTOR_COLS)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(10, n_tiers * 1.8), 5))

    bar_width = 0.8 / n_factors
    x = np.arange(n_tiers)

    for i, col in enumerate(FACTOR_COLS):
        sub = imp_df[imp_df["factor"] == col].set_index("budget_tier")
        vals = [sub.loc[t, "marginal_range"] if t in sub.index else 0.0
                for t in tiers]
        offset = (i - n_factors / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, width=bar_width,
               color=FACTOR_COLORS[col], label=FACTOR_LABELS[col],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(tiers, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Budget tier (function evaluations)")
    ax.set_ylabel("Marginal accuracy range (best − worst level)")
    ax.set_title(f"Marginal effect size by budget tier — {metric}")
    ax.legend(title="Design factor", loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return ax


# =========================================================================
# Plot 3 — Per-tier marginal means (small multiples)
# =========================================================================

def plot_marginal_means_per_tier(df, metric=DEFAULT_METRIC, tier_labels=None,
                                  figsize=None):
    """
    Grid: rows = budget tiers, cols = factors.
    Each subplot shows mean accuracy at each level of that factor,
    marginalised over the others.
    """
    if tier_labels is None:
        tier_labels = df["budget_tier"].cat.categories.tolist()

    valid_tiers = [t for t in tier_labels if len(df[df["budget_tier"] == t]) >= 3]
    if not valid_tiers:
        print("Not enough data per tier for marginal means plot")
        return None

    n_tiers = len(valid_tiers)
    if figsize is None:
        figsize = (16, 3 * n_tiers)

    fig, axes = plt.subplots(n_tiers, 4, figsize=figsize,
                              squeeze=False, sharey=True)

    for row, tier in enumerate(valid_tiers):
        sub = df[df["budget_tier"] == tier]

        for col_idx, factor in enumerate(FACTOR_COLS):
            ax = axes[row, col_idx]
            agg = sub.groupby(factor)[metric].agg(["mean", "std"]).reset_index()
            agg = agg.sort_values(factor)

            x_labels = agg[factor].astype(str).values
            x_pos = np.arange(len(x_labels))

            if factor == "sampling_strategy":
                colors = [STRATEGY_COLORS.get(s, "gray") for s in agg[factor]]
                x_labels = [STRATEGY_LABELS.get(s, s) for s in agg[factor]]
            else:
                colors = [FACTOR_COLORS[factor]] * len(x_labels)

            ax.bar(x_pos, agg["mean"], yerr=agg["std"], color=colors,
                   capsize=2, alpha=0.8, edgecolor="white")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)

            if row == 0:
                ax.set_title(FACTOR_LABELS[factor], fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"{tier}\n", fontsize=7)

            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Marginal means per budget tier — {metric}\n"
        "Each bar = mean accuracy at that factor level, averaged over other factors",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


# =========================================================================
# Plot 4 — Relative η² share (stacked bars)
# =========================================================================

def plot_eta_squared_share(imp_df, metric=DEFAULT_METRIC, ax=None):
    """
    Stacked bar chart: each tier's η² values normalised to sum to 1,
    showing how factor dominance shifts across budget levels.
    """
    tiers = imp_df["budget_tier"].unique()
    n_tiers = len(tiers)

    mat = np.zeros((n_tiers, len(FACTOR_COLS)))
    for i, tier in enumerate(tiers):
        for j, col in enumerate(FACTOR_COLS):
            row = imp_df[(imp_df["budget_tier"] == tier) &
                         (imp_df["factor"] == col)]
            if len(row) > 0:
                mat[i, j] = row["eta_sq"].values[0]

    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat_norm = mat / row_sums

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(n_tiers)
    bottoms = np.zeros(n_tiers)
    for j, col in enumerate(FACTOR_COLS):
        ax.bar(x, mat_norm[:, j], bottom=bottoms, width=0.7,
               color=FACTOR_COLORS[col], label=FACTOR_LABELS[col],
               edgecolor="white", linewidth=0.5)
        bottoms += mat_norm[:, j]

    ax.set_xticks(x)
    ax.set_xticklabels(tiers, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Budget tier (function evaluations)")
    ax.set_ylabel("Share of explained variance (η²)")
    ax.set_title(f"Relative factor importance — {metric}")
    ax.legend(title="Design factor", loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    return ax


# =========================================================================
# Convenience: run everything
# =========================================================================

def filter_strategies(df, omit_strategies=None):
    """
    Drop rows whose sampling_strategy is in omit_strategies.

    Parameters
    ----------
    df : DataFrame
    omit_strategies : str or list[str], e.g. "cma_random" or ["cma_random", "uniform"]

    Returns
    -------
    DataFrame (copy) with rows removed
    """
    if omit_strategies is None:
        return df
    if isinstance(omit_strategies, str):
        omit_strategies = [omit_strategies]
    return df[~df["sampling_strategy"].isin(omit_strategies)].copy()


# =========================================================================
# Convenience: run everything
# =========================================================================

def run_all(df, metric=DEFAULT_METRIC, n_tiers=4, omit_strategies=None):
    """
    Assign budget tiers, compute importance, and produce all four plots.

    Parameters
    ----------
    df : fold-aggregated DataFrame (one row per design config)
    metric : accuracy column name
    n_tiers : number of quantile-based budget tiers
    omit_strategies : str or list[str], strategies to exclude
                      e.g. "cma_random" or ["cma_random", "uniform"]

    Returns
    -------
    imp_df : factor importance table (DataFrame)
    """
    df = filter_strategies(df, omit_strategies)
    if omit_strategies:
        print(f"Omitted strategies: {omit_strategies}")
        print(f"Remaining: {sorted(df['sampling_strategy'].unique())}")
        print()

    df, tier_labels = assign_budget_tiers(df, n_tiers=n_tiers)

    print(f"Budget tiers ({n_tiers}):")
    for t in tier_labels:
        n = len(df[df["budget_tier"] == t])
        print(f"  {t}: {n} configs")
    print()

    imp_df = compute_factor_importance(df, metric=metric, tier_labels=tier_labels)
    print(imp_df.to_string(index=False))
    print()

    plot_eta_squared(imp_df, metric=metric)
    plt.show()

    plot_marginal_range(imp_df, metric=metric)
    plt.show()

    fig = plot_marginal_means_per_tier(df, metric=metric, tier_labels=tier_labels)
    if fig is not None:
        plt.show()

    plot_eta_squared_share(imp_df, metric=metric)
    plt.show()

    return imp_df