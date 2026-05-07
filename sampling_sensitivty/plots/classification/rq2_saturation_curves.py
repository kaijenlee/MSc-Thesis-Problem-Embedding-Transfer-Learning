"""
RQ2 — Marginal Saturation Curves
==================================

For each design factor, plot accuracy as a function of that factor's level,
conditioned on the levels of each other factor. This directly answers:
  - At what level does each factor saturate?
  - Does the saturation point depend on other design choices?

Produces for each of the 4 factors:
  - 3 panels (one per conditioning factor)
  - Each panel has one line per level of the conditioning factor
  - x-axis = levels of the focal factor
  - y-axis = mean accuracy (averaged over remaining factors)

Usage in notebook:
    from rq2_saturation_curves import *

    # All factors
    run_all(fold_agg_df, metric="acc_mean", omit_strategies="cma_random")

    # Single factor
    plot_saturation("n_instances_train", fold_agg_df, metric="acc_mean")

    # Custom conditioning
    plot_saturation_single("n_instances_train", "sampling_strategy",
                            fold_agg_df, metric="acc_mean")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FACTOR_COLS = [
    "sampling_strategy", "sample_size_per_dim",
    "n_instances_train", "n_runs_train",
]
FACTOR_LABELS = {
    "sampling_strategy":   "Strategy",
    "sample_size_per_dim": "Sample size (×d)",
    "n_instances_train":   "Instances",
    "n_runs_train":        "Runs",
}
STRATEGY_COLORS = {
    "cma_random": "#9467bd", "uniform": "#d62728", "lhs": "#ff7f0e",
    "ilhs": "#1f77b4", "lhs_rcd": "#8c564b", "sobol": "#2ca02c",
}
STRATEGY_LABELS = {
    "cma_random": "CMA-ES", "uniform": "Uniform", "lhs": "LHS",
    "ilhs": "iLHS", "lhs_rcd": "LHS-RCD", "sobol": "Sobol",
}

DEFAULT_METRIC = "acc_allruns_mean"


def filter_strategies(df, omit_strategies=None):
    if omit_strategies is None:
        return df
    if isinstance(omit_strategies, str):
        omit_strategies = [omit_strategies]
    return df[~df["sampling_strategy"].isin(omit_strategies)].copy()


def _sort_levels(values):
    """Sort factor levels numerically if possible, otherwise alphabetically."""
    try:
        return sorted(values, key=float)
    except (ValueError, TypeError):
        return sorted(values)


def _get_colormap(cond_col, levels):
    """Get colours and labels for a conditioning factor."""
    if cond_col == "sampling_strategy":
        colors = {lv: STRATEGY_COLORS.get(lv, "gray") for lv in levels}
        labels = {lv: STRATEGY_LABELS.get(lv, lv) for lv in levels}
    else:
        cmap = cm.get_cmap("viridis", len(levels))
        colors = {lv: cmap(i) for i, lv in enumerate(levels)}
        labels = {lv: str(lv) for lv in levels}
    return colors, labels


# =========================================================================
# Core: single panel — one focal factor, one conditioning factor
# =========================================================================

def plot_saturation_single(focal_col, cond_col, df, metric=DEFAULT_METRIC,
                            ax=None, show_legend=True):
    """
    Plot accuracy vs levels of focal_col, with one line per level of cond_col.
    Accuracy is averaged over all other factors (those not focal or cond).

    Parameters
    ----------
    focal_col : str, the factor whose saturation we're examining
    cond_col : str, the factor we condition on (separate lines)
    df : DataFrame
    metric : str, accuracy column
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    focal_levels = _sort_levels(df[focal_col].unique())
    cond_levels = _sort_levels(df[cond_col].unique())
    colors, labels = _get_colormap(cond_col, cond_levels)

    for cond_lv in cond_levels:
        sub = df[df[cond_col] == cond_lv]
        means = []
        sds = []
        valid_focal = []

        for focal_lv in focal_levels:
            s = sub[sub[focal_col] == focal_lv]
            if len(s) > 0:
                means.append(s[metric].mean())
                sds.append(s[metric].std())
                valid_focal.append(focal_lv)

        if not means:
            continue

        # x positions: numeric if possible, otherwise ordinal
        try:
            x = [float(v) for v in valid_focal]
        except (ValueError, TypeError):
            x = list(range(len(valid_focal)))

        color = colors[cond_lv]
        label = labels[cond_lv]

        ax.plot(x, means, marker="o", markersize=5, color=color,
                label=label, linewidth=1.5, alpha=0.8)
        # ax.fill_between(x,
        #                  [m - s for m, s in zip(means, sds)],
        #                  [m + s for m, s in zip(means, sds)],
        #                  color=color, alpha=0.08)

    # x-axis formatting
    try:
        [float(v) for v in focal_levels]
        # numeric: keep as is
    except (ValueError, TypeError):
        # categorical: use tick labels
        ax.set_xticks(range(len(focal_levels)))
        if focal_col == "sampling_strategy":
            ax.set_xticklabels([STRATEGY_LABELS.get(v, v) for v in focal_levels],
                               rotation=45, ha="right", fontsize=8)
        else:
            ax.set_xticklabels([str(v) for v in focal_levels],
                               rotation=45, ha="right", fontsize=8)

    ax.set_xlabel(FACTOR_LABELS.get(focal_col, focal_col))
    ax.set_ylabel("Classification accuracy")
    ax.set_title(f"by {FACTOR_LABELS.get(cond_col, cond_col)}", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 1.02)

    if show_legend:
        ax.legend(title=FACTOR_LABELS.get(cond_col, cond_col),
                  fontsize=7, title_fontsize=8, loc="lower right")

    return ax


# =========================================================================
# Multi-panel: one focal factor, conditioned on each other factor
# =========================================================================

def plot_saturation(focal_col, df, metric=DEFAULT_METRIC, omit_strategies=None,
                     figsize=None):
    """
    For one focal factor, produce a row of panels — one per conditioning factor.
    Each panel shows how the focal factor's effect changes depending on the
    conditioning factor.

    Parameters
    ----------
    focal_col : str, the factor to examine
    """
    df = filter_strategies(df, omit_strategies)
    cond_cols = [c for c in FACTOR_COLS if c != focal_col]

    n = len(cond_cols)
    if figsize is None:
        figsize = (6 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, cond_col in zip(axes, cond_cols):
        plot_saturation_single(focal_col, cond_col, df, metric=metric,
                                ax=ax, show_legend=True)

    axes[0].set_ylabel("Classification accuracy")
    fig.suptitle(
        f"Saturation of {FACTOR_LABELS.get(focal_col, focal_col)} — {metric}\n"
        f"Each line = a level of the conditioning factor, "
        f"averaged over remaining factors",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    return fig


# =========================================================================
# Summary: marginal gain from each additional level
# =========================================================================

def compute_marginal_gains(df, metric=DEFAULT_METRIC):
    """
    For each ordinal factor, compute the accuracy gain from each step
    (level[i] → level[i+1]), averaged over all other factors.

    Returns DataFrame with columns:
        factor, from_level, to_level, mean_gain, pct_of_total_gain
    """
    rows = []
    ordinal_cols = ["sample_size_per_dim", "n_instances_train", "n_runs_train"]

    for col in ordinal_cols:
        levels = _sort_levels(df[col].unique())
        level_means = df.groupby(col)[metric].mean()

        total_gain = float(level_means[levels[-1]]) - float(level_means[levels[0]])

        for i in range(len(levels) - 1):
            lv_from = levels[i]
            lv_to = levels[i + 1]
            gain = float(level_means[lv_to]) - float(level_means[lv_from])
            pct = (gain / total_gain * 100) if total_gain != 0 else 0

            rows.append({
                "factor": FACTOR_LABELS.get(col, col),
                "from_level": lv_from,
                "to_level": lv_to,
                "mean_gain": gain,
                "total_gain": total_gain,
                "pct_of_total_gain": pct,
            })

    return pd.DataFrame(rows)


def plot_marginal_gains(gains_df, ax=None):
    """
    For each ordinal factor, bar chart of accuracy gain per step.
    Shows where the diminishing returns kick in.
    """
    factors = gains_df["factor"].unique()
    n = len(factors)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, factor in zip(axes, factors):
        sub = gains_df[gains_df["factor"] == factor]
        labels = [f"{r['from_level']}→{r['to_level']}" for _, r in sub.iterrows()]
        gains = sub["mean_gain"].values

        colors = ["#2ca02c" if g > 0 else "#d62728" for g in gains]
        x_pos = np.arange(len(labels))

        ax.bar(x_pos, gains, color=colors, edgecolor="white", alpha=0.8)

        for i, g in enumerate(gains):
            ax.text(i, g + 0.002 if g >= 0 else g - 0.008,
                    f"{g:.3f}", ha="center", va="bottom" if g >= 0 else "top",
                    fontsize=8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(factor)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("Accuracy gain per step")

    fig.suptitle("Accuracy gain per step in design factors",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# =========================================================================
# Convenience: run everything
# =========================================================================

def run_all(df, metric=DEFAULT_METRIC, omit_strategies=None):
    """
    Produce saturation curves for all four factors and the marginal gains summary.
    """
    df = filter_strategies(df, omit_strategies)
    if omit_strategies:
        print(f"Omitted strategies: {omit_strategies}")
        print(f"Remaining: {sorted(df['sampling_strategy'].unique())}")
        print()

    for focal_col in FACTOR_COLS:
        print(f"Saturation curves for {FACTOR_LABELS[focal_col]} ...")
        plot_saturation(focal_col, df, metric=metric)
        plt.show()

    # Marginal gains
    print("Marginal gains per step ...")
    gains_df = compute_marginal_gains(df, metric=metric)
    print(gains_df.to_string(index=False))
    print()

    plot_marginal_gains(gains_df)
    plt.show()

    return gains_df


# =========================================================================
# Budget-tier saturation: how saturation shifts across budget levels
# =========================================================================

def plot_saturation_by_tier(focal_col, df, metric=DEFAULT_METRIC,
                             omit_strategies=None, n_tiers=4, figsize=None):
    """
    One panel per budget tier, showing accuracy vs focal factor level.
    Each line within a panel is a level of the conditioning factor that
    varies most (n_instances for runs, n_runs for instances, etc.),
    or you can see the overall trend per tier.

    This answers: does the saturation shape change at different budgets?
    e.g. do instances saturate earlier at high budgets?
    """
    df = filter_strategies(df, omit_strategies)
    df = df.copy()
    df["budget_tier"] = pd.qcut(df["n_feval_train"], q=n_tiers)
    tier_labels = df["budget_tier"].cat.categories.tolist()

    # Pick the most relevant conditioning factor for each focal
    default_cond = {
        "n_instances_train": "sample_size_per_dim",
        "n_runs_train": "n_instances_train",
        "sample_size_per_dim": "n_instances_train",
        "sampling_strategy": "n_instances_train",
    }
    cond_col = default_cond.get(focal_col, "n_instances_train")

    n = len(tier_labels)
    if figsize is None:
        figsize = (6 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, tier in zip(axes, tier_labels):
        sub = df[df["budget_tier"] == tier]
        if len(sub) < 5:
            ax.text(0.5, 0.5, "too few\npoints", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
            ax.set_title(f"{tier}")
            continue

        plot_saturation_single(focal_col, cond_col, sub, metric=metric,
                                ax=ax, show_legend=(ax == axes[-1]))
        ax.set_title(f"Budget: {tier}", fontsize=9)

    axes[0].set_ylabel("Classification accuracy")
    fig.suptitle(
        f"Saturation of {FACTOR_LABELS.get(focal_col, focal_col)} "
        f"across budget tiers\n"
        f"Lines = {FACTOR_LABELS.get(cond_col, cond_col)}",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    return fig


def plot_saturation_overall_by_tier(focal_col, df, metric=DEFAULT_METRIC,
                                     omit_strategies=None, n_tiers=4, ax=None):
    """
    Single panel: accuracy vs focal factor level, with one line per budget tier.
    No conditioning on other factors — just the overall marginal trend
    within each tier. Directly shows how the saturation curve shifts.
    """
    df = filter_strategies(df, omit_strategies)
    df = df.copy()
    df["budget_tier"] = pd.qcut(df["n_feval_train"], q=n_tiers)
    tier_labels = df["budget_tier"].cat.categories.tolist()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    cmap = cm.get_cmap("coolwarm", len(tier_labels))

    for i, tier in enumerate(tier_labels):
        sub = df[df["budget_tier"] == tier]
        focal_levels = _sort_levels(sub[focal_col].unique())

        means = []
        valid_levels = []
        for lv in focal_levels:
            s = sub[sub[focal_col] == lv]
            if len(s) > 0:
                means.append(s[metric].mean())
                valid_levels.append(lv)

        if not means:
            continue

        try:
            x = [float(v) for v in valid_levels]
        except (ValueError, TypeError):
            x = list(range(len(valid_levels)))

        ax.plot(x, means, marker="o", markersize=5, color=cmap(i),
                label=f"{tier}", linewidth=2, alpha=0.8)

    try:
        [float(v) for v in _sort_levels(df[focal_col].unique())]
    except (ValueError, TypeError):
        all_levels = _sort_levels(df[focal_col].unique())
        ax.set_xticks(range(len(all_levels)))
        if focal_col == "sampling_strategy":
            ax.set_xticklabels([STRATEGY_LABELS.get(v, v) for v in all_levels],
                               rotation=45, ha="right", fontsize=8)
        else:
            ax.set_xticklabels([str(v) for v in all_levels],
                               rotation=45, ha="right", fontsize=8)

    ax.set_xlabel(FACTOR_LABELS.get(focal_col, focal_col))
    ax.set_ylabel("Classification accuracy")
    ax.set_title(f"Saturation of {FACTOR_LABELS.get(focal_col, focal_col)} "
                 f"by budget tier")
    ax.legend(title="Budget tier", fontsize=7, title_fontsize=8,
              loc="lower right")
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    return ax


def compute_marginal_gains_by_tier(df, metric=DEFAULT_METRIC, n_tiers=4):
    """
    Compute marginal gains per step for each ordinal factor, within each
    budget tier. Shows whether diminishing returns kick in earlier or
    later depending on budget.

    Returns DataFrame with columns:
        budget_tier, factor, from_level, to_level, mean_gain, pct_of_total_gain
    """
    df = df.copy()
    df["budget_tier"] = pd.qcut(df["n_feval_train"], q=n_tiers)
    tier_labels = df["budget_tier"].cat.categories.tolist()
    ordinal_cols = ["sample_size_per_dim", "n_instances_train", "n_runs_train"]

    rows = []
    for tier in tier_labels:
        sub = df[df["budget_tier"] == tier]

        for col in ordinal_cols:
            levels = _sort_levels(sub[col].unique())
            if len(levels) < 2:
                continue

            level_means = sub.groupby(col)[metric].mean()
            total_gain = (float(level_means[levels[-1]]) -
                          float(level_means[levels[0]]))

            for i in range(len(levels) - 1):
                lv_from = levels[i]
                lv_to = levels[i + 1]
                gain = float(level_means[lv_to]) - float(level_means[lv_from])
                pct = (gain / total_gain * 100) if total_gain != 0 else 0

                rows.append({
                    "budget_tier": str(tier),
                    "factor": FACTOR_LABELS.get(col, col),
                    "from_level": lv_from,
                    "to_level": lv_to,
                    "mean_gain": gain,
                    "total_gain": total_gain,
                    "pct_of_total_gain": pct,
                })

    return pd.DataFrame(rows)


def plot_marginal_gains_by_tier(gains_df):
    """
    For each ordinal factor, a grouped bar chart: x = step (from→to),
    bars grouped by budget tier.
    """
    factors = gains_df["factor"].unique()
    n_factors = len(factors)
    tiers = gains_df["budget_tier"].unique()
    n_tiers = len(tiers)

    cmap = cm.get_cmap("coolwarm", n_tiers)

    fig, axes = plt.subplots(1, n_factors, figsize=(7 * n_factors, 5))
    if n_factors == 1:
        axes = [axes]

    for ax, factor in zip(axes, factors):
        factor_data = gains_df[gains_df["factor"] == factor]
        # Get unique steps (in order)
        steps = []
        seen = set()
        for _, r in factor_data.iterrows():
            step = f"{r['from_level']}→{r['to_level']}"
            if step not in seen:
                steps.append(step)
                seen.add(step)

        n_steps = len(steps)
        x = np.arange(n_steps)
        bar_width = 0.8 / n_tiers

        for i, tier in enumerate(tiers):
            tier_data = factor_data[factor_data["budget_tier"] == tier]
            vals = []
            for step in steps:
                parts = step.split("→")
                match = tier_data[
                    (tier_data["from_level"].astype(str) == parts[0]) &
                    (tier_data["to_level"].astype(str) == parts[1])
                ]
                vals.append(match["mean_gain"].values[0] if len(match) > 0 else 0)

            offset = (i - n_tiers / 2 + 0.5) * bar_width
            ax.bar(x + offset, vals, width=bar_width, color=cmap(i),
                   label=tier if ax == axes[0] else None,
                   edgecolor="white", linewidth=0.3, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(steps, rotation=45, ha="right", fontsize=8)
        ax.set_title(factor)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("Accuracy gain")

    axes[0].legend(title="Budget tier", fontsize=6, title_fontsize=7,
                   loc="upper right")
    fig.suptitle("Accuracy gains per step by budget tier", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# =========================================================================
# Convenience: run budget-tier analysis
# =========================================================================

def run_by_tier(df, metric=DEFAULT_METRIC, omit_strategies=None, n_tiers=4):
    """
    Produce budget-tier saturation curves and marginal gains for all factors.

    Usage:
        gains_df = run_by_tier(fold_agg_df, metric="acc_mean",
                                omit_strategies="cma_random")
    """
    df = filter_strategies(df, omit_strategies)
    if omit_strategies:
        print(f"Omitted strategies: {omit_strategies}")
        print(f"Remaining: {sorted(df['sampling_strategy'].unique())}")
        print()

    # Overall saturation by tier (one line per tier)
    for focal_col in FACTOR_COLS:
        print(f"Overall saturation by tier: {FACTOR_LABELS[focal_col]} ...")
        plot_saturation_overall_by_tier(focal_col, df, metric=metric,
                                         n_tiers=n_tiers)
        plt.show()

    # Detailed: per-tier panels with conditioning
    for focal_col in FACTOR_COLS:
        print(f"Per-tier saturation: {FACTOR_LABELS[focal_col]} ...")
        plot_saturation_by_tier(focal_col, df, metric=metric,
                                 n_tiers=n_tiers)
        plt.show()

    # Marginal gains by tier
    print("Marginal gains by tier ...")
    gains_df = compute_marginal_gains_by_tier(df, metric=metric, n_tiers=n_tiers)
    print(gains_df.to_string(index=False))
    print()

    plot_marginal_gains_by_tier(gains_df)
    plt.show()

    return gains_df