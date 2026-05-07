"""
RQ2 — Analysis 2: CV–Accuracy Correlation
===========================================

Does feature stability (low CV from RQ1) predict classification accuracy (RQ2)?

Since CV is constant per (strategy, sample_size) pair — it doesn't depend on
n_instances_train or n_runs_train — there are 6 strategies × 4 sample sizes = 24
unique CV values. For each we can compute accuracy aggregated over the
(n_instances, n_runs) grid.

Produces:
  1. Scatter: CV vs accuracy, one point per (strategy, sample_size) pair
  2. Same scatter faceted by sample_size
  3. Scatter faceted by (n_instances, n_runs) to check if the correlation
     holds across different training set compositions
  4. Rank correlation summary table

Usage in notebook:
    from rq2_cv_accuracy import *

    # fold_agg_df is your fold-aggregated dataframe
    corr_df = run_all(fold_agg_df, cv_col="cv_instance_median_mean", acc_col="acc_mean")

    # Or step by step
    pair_df = build_pair_df(fold_agg_df)
    plot_cv_vs_accuracy(pair_df)
    plot_cv_vs_accuracy_by_size(pair_df)
    plot_cv_vs_accuracy_by_allocation(fold_agg_df)
    corr_df = compute_correlations(fold_agg_df)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

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

SIZE_MARKERS = {25: "o", 50: "s", 75: "D", 100: "^"}

DEFAULT_CV_COL = "cv_instance_median_mean"
DEFAULT_ACC_COL = "acc_mean"


def _get_strategies(df):
    return [s for s in STRATEGY_ORDER if s in df["sampling_strategy"].unique()]


def filter_strategies(df, omit_strategies=None):
    """Drop rows whose sampling_strategy is in omit_strategies."""
    if omit_strategies is None:
        return df
    if isinstance(omit_strategies, str):
        omit_strategies = [omit_strategies]
    return df[~df["sampling_strategy"].isin(omit_strategies)].copy()


# =========================================================================
# Build pair-level dataframe (one row per strategy × sample_size)
# =========================================================================

def build_pair_df(df, cv_col=DEFAULT_CV_COL, acc_col=DEFAULT_ACC_COL):
    """
    Aggregate accuracy over (n_instances_train, n_runs_train) for each
    (strategy, sample_size) pair. CV is already constant within each pair.

    Returns DataFrame with columns:
        sampling_strategy, sample_size_per_dim, cv, acc_mean, acc_std, n_configs
    """
    pair_df = (
        df.groupby(["sampling_strategy", "sample_size_per_dim"])
        .agg(
            cv=(cv_col, "first"),
            acc_mean=(acc_col, "mean"),
            acc_std=(acc_col, "std"),
            n_configs=(acc_col, "count"),
        )
        .reset_index()
    )
    return pair_df


# =========================================================================
# Plot 1 — Main scatter: CV vs accuracy
# =========================================================================

def plot_cv_vs_accuracy(pair_df, cv_col="cv", acc_col="acc_mean", ax=None):
    """
    Scatter plot with one point per (strategy, sample_size) pair.
    Colour = strategy, marker shape = sample size.
    Includes Spearman correlation annotation.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    strategies = _get_strategies(pair_df)

    for strat in strategies:
        sub = pair_df[pair_df["sampling_strategy"] == strat]
        color = STRATEGY_COLORS.get(strat, "gray")
        label = STRATEGY_LABELS.get(strat, strat)

        for _, row in sub.iterrows():
            sz = int(row["sample_size_per_dim"])
            marker = SIZE_MARKERS.get(sz, "o")
            ax.scatter(row[cv_col], row[acc_col],
                       c=color, marker=marker, s=100, edgecolors="white",
                       linewidth=0.5, zorder=3)

        # One label entry per strategy (use first row for legend)
        ax.scatter([], [], c=color, marker="o", s=80, label=label)

    # Size legend
    for sz, marker in SIZE_MARKERS.items():
        ax.scatter([], [], c="gray", marker=marker, s=60,
                   label=f"{sz}d", edgecolors="white")

    # Spearman correlation
    rho, pval = stats.spearmanr(pair_df[cv_col], pair_df[acc_col])
    ax.annotate(
        f"Spearman ρ = {rho:.3f}\np = {pval:.4f}",
        xy=(0.97, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
    )

    # Trend line
    slope, intercept, _, _, _ = stats.linregress(pair_df[cv_col], pair_df[acc_col])
    x_range = np.linspace(pair_df[cv_col].min(), pair_df[cv_col].max(), 100)
    ax.plot(x_range, slope * x_range + intercept,
            color="gray", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xlabel("Feature CV (RQ1 stability)")
    ax.set_ylabel("Classification accuracy (RQ2)")
    ax.set_title("Feature stability vs classification accuracy")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return ax


# =========================================================================
# Plot 2 — Faceted by sample_size
# =========================================================================

def plot_cv_vs_accuracy_by_size(pair_df, cv_col="cv", acc_col="acc_mean"):
    """
    One subplot per sample_size_per_dim. Within each, scatter of
    strategies with correlation annotation. Shows whether the
    CV–accuracy relationship holds at each sample size or only globally.
    """
    sizes = sorted(pair_df["sample_size_per_dim"].unique())
    n = len(sizes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    strategies = _get_strategies(pair_df)

    for ax, sz in zip(axes, sizes):
        sub = pair_df[pair_df["sample_size_per_dim"] == sz]

        for strat in strategies:
            s = sub[sub["sampling_strategy"] == strat]
            if len(s) == 0:
                continue
            color = STRATEGY_COLORS.get(strat, "gray")
            label = STRATEGY_LABELS.get(strat, strat)
            ax.scatter(s[cv_col], s[acc_col], c=color, s=100,
                       edgecolors="white", linewidth=0.5, label=label, zorder=3)

        # Correlation (only if enough points)
        if len(sub) >= 4:
            rho, pval = stats.spearmanr(sub[cv_col], sub[acc_col])
            ax.annotate(
                f"ρ = {rho:.2f}, p = {pval:.3f}",
                xy=(0.97, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.6),
            )

        ax.set_title(f"Sample size = {sz}d")
        ax.set_xlabel("Feature CV")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Classification accuracy")
    axes[-1].legend(loc="lower left", fontsize=7)

    fig.suptitle("CV vs accuracy — by sample size", y=1.02)
    fig.tight_layout()
    return fig


# =========================================================================
# Plot 3 — Correlation strength heatmap across allocations
# =========================================================================

def plot_correlation_heatmap(df, cv_col=DEFAULT_CV_COL, acc_col=DEFAULT_ACC_COL,
                              ax=None):
    """
    Heatmap: rows = n_instances_train, cols = n_runs_train.
    Cell colour = Spearman ρ between CV and accuracy at that allocation.

    Since CV is fixed per (strategy, sample_size), each cell has one point
    per (strategy, sample_size) pair. The question: does the CV–accuracy
    correlation weaken as training data grows (more instances/runs)?
    If so, stability only matters when data is scarce.
    """
    inst_vals = sorted(df["n_instances_train"].unique())
    run_vals = sorted(df["n_runs_train"].unique())

    mat = np.full((len(inst_vals), len(run_vals)), np.nan)
    pmat = np.full_like(mat, np.nan)

    for i, n_inst in enumerate(inst_vals):
        for j, n_run in enumerate(run_vals):
            sub = df[(df["n_instances_train"] == n_inst) &
                     (df["n_runs_train"] == n_run)]
            if len(sub) >= 4:
                rho, p = stats.spearmanr(sub[cv_col], sub[acc_col])
                mat[i, j] = rho
                pmat[i, j] = p

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 7))

    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", origin="lower",
                   vmin=-1, vmax=1)

    for i in range(len(inst_vals)):
        for j in range(len(run_vals)):
            if not np.isnan(mat[i, j]):
                sig = "*" if pmat[i, j] < 0.05 else ""
                color = "white" if abs(mat[i, j]) > 0.5 else "black"
                ax.text(j, i, f"{mat[i,j]:.2f}{sig}", ha="center", va="center",
                        fontsize=8, color=color)

    ax.set_xticks(range(len(run_vals)))
    ax.set_xticklabels(run_vals)
    ax.set_yticks(range(len(inst_vals)))
    ax.set_yticklabels(inst_vals)
    ax.set_xlabel("n_runs_train")
    ax.set_ylabel("n_instances_train")
    ax.set_title("Spearman ρ(CV, accuracy) per allocation\n* = p < 0.05")
    plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)

    plt.tight_layout()
    return ax


# =========================================================================
# Partial correlation: CV → accuracy controlling for sample size / budget
# =========================================================================

def _partial_spearman(x, y, covariate):
    """
    Partial Spearman correlation between x and y, controlling for covariate.
    Method: rank both x and y, regress each on the covariate ranks,
    then correlate the residuals.
    """
    from scipy.stats import rankdata

    rx = rankdata(x)
    ry = rankdata(y)
    rc = rankdata(covariate)

    # Regress ranks on covariate
    def _resid(r, c):
        slope, intercept, _, _, _ = stats.linregress(c, r)
        return r - (slope * c + intercept)

    rx_resid = _resid(rx, rc)
    ry_resid = _resid(ry, rc)

    rho, pval = stats.spearmanr(rx_resid, ry_resid)
    return rho, pval


def compute_partial_correlations(df, cv_col=DEFAULT_CV_COL, acc_col=DEFAULT_ACC_COL):
    """
    Compute partial Spearman correlations controlling for confounds.

    Returns DataFrame with columns:
        analysis, controlling_for, n, spearman_rho, p_value

    Analyses:
      1. Raw (no control)
      2. Controlling for sample_size_per_dim
      3. Controlling for n_feval_train (total budget)
      4. Within each sample_size level (stratified, no partial needed)
    """
    rows = []

    # 1. Raw correlation
    rho, p = stats.spearmanr(df[cv_col], df[acc_col])
    rows.append({
        "analysis": "Raw",
        "controlling_for": "—",
        "n": len(df),
        "spearman_rho": rho,
        "p_value": p,
    })

    # 2. Partial: controlling for sample_size_per_dim
    rho, p = _partial_spearman(
        df[cv_col].values, df[acc_col].values,
        df["sample_size_per_dim"].values,
    )
    rows.append({
        "analysis": "Partial",
        "controlling_for": "sample_size_per_dim",
        "n": len(df),
        "spearman_rho": rho,
        "p_value": p,
    })

    # 3. Partial: controlling for n_feval_train
    if "n_feval_train" in df.columns:
        rho, p = _partial_spearman(
            df[cv_col].values, df[acc_col].values,
            df["n_feval_train"].values,
        )
        rows.append({
            "analysis": "Partial",
            "controlling_for": "n_feval_train",
            "n": len(df),
            "spearman_rho": rho,
            "p_value": p,
        })

    # 4. Within each sample_size (stratified)
    for sz in sorted(df["sample_size_per_dim"].unique()):
        sub = df[df["sample_size_per_dim"] == sz]
        if len(sub) >= 4:
            rho, p = stats.spearmanr(sub[cv_col], sub[acc_col])
            rows.append({
                "analysis": f"Within size={int(sz)}d",
                "controlling_for": "(stratified)",
                "n": len(sub),
                "spearman_rho": rho,
                "p_value": p,
            })

    return pd.DataFrame(rows)


def plot_partial_correlation_comparison(partial_df, ax=None):
    """
    Bar chart comparing raw vs partial vs within-size correlations.
    Visually answers: does controlling for sample size / budget
    weaken the CV–accuracy link?
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    labels = []
    for _, row in partial_df.iterrows():
        if row["controlling_for"] == "—":
            labels.append("Raw")
        elif row["controlling_for"] == "(stratified)":
            labels.append(row["analysis"])
        else:
            labels.append(f"Partial\n(ctrl: {row['controlling_for']})")

    x_pos = np.arange(len(labels))
    rhos = partial_df["spearman_rho"].values
    pvals = partial_df["p_value"].values

    colors = []
    for p in pvals:
        if p < 0.01:
            colors.append("#2ca02c")   # green = significant
        elif p < 0.05:
            colors.append("#ff7f0e")   # orange = marginally significant
        else:
            colors.append("#d62728")   # red = not significant

    ax.bar(x_pos, rhos, color=colors, edgecolor="white", alpha=0.8)

    for i, (r, p) in enumerate(zip(rhos, pvals)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        y_offset = 0.02 if r >= 0 else -0.05
        ax.text(i, r + y_offset, f"ρ={r:.2f}\n{sig}", ha="center", va="bottom",
                fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("CV–accuracy correlation: raw vs controlling for confounds")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Legend for significance colours
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#2ca02c", label="p < 0.01"),
        Patch(color="#ff7f0e", label="p < 0.05"),
        Patch(color="#d62728", label="n.s."),
    ], loc="lower right", fontsize=7)

    plt.tight_layout()
    return ax


# =========================================================================
# Correlation summary table
# =========================================================================

def compute_correlations(df, cv_col=DEFAULT_CV_COL, acc_col=DEFAULT_ACC_COL):
    """
    Compute Spearman correlation between CV and accuracy at multiple levels:
      - Global (all rows)
      - Per sample_size_per_dim
      - Per (n_instances_train, n_runs_train)

    Returns DataFrame with columns:
        level, level_value, n, spearman_rho, p_value
    """
    rows = []

    # Global
    rho, p = stats.spearmanr(df[cv_col], df[acc_col])
    rows.append({"level": "global", "level_value": "all", "n": len(df),
                 "spearman_rho": rho, "p_value": p})

    # Per sample size
    for sz in sorted(df["sample_size_per_dim"].unique()):
        sub = df[df["sample_size_per_dim"] == sz]
        if len(sub) >= 4:
            rho, p = stats.spearmanr(sub[cv_col], sub[acc_col])
            rows.append({"level": "sample_size", "level_value": str(int(sz)),
                         "n": len(sub), "spearman_rho": rho, "p_value": p})

    # Per (n_instances, n_runs)
    for n_inst in sorted(df["n_instances_train"].unique()):
        for n_run in sorted(df["n_runs_train"].unique()):
            sub = df[(df["n_instances_train"] == n_inst) &
                     (df["n_runs_train"] == n_run)]
            if len(sub) >= 4:
                rho, p = stats.spearmanr(sub[cv_col], sub[acc_col])
                rows.append({
                    "level": "allocation",
                    "level_value": f"{int(n_inst)}inst_{int(n_run)}run",
                    "n": len(sub),
                    "spearman_rho": rho, "p_value": p,
                })

    return pd.DataFrame(rows)


# =========================================================================
# Convenience: run everything
# =========================================================================

def run_all(df, cv_col=DEFAULT_CV_COL, acc_col=DEFAULT_ACC_COL,
            omit_strategies=None):
    """
    Build pair-level data, produce all plots, and return correlation table.

    Parameters
    ----------
    df : fold-aggregated DataFrame
    cv_col : CV column from RQ1
    acc_col : accuracy column from RQ2
    omit_strategies : str or list[str], strategies to exclude

    Returns
    -------
    corr_df : correlation summary table (DataFrame)
    partial_df : partial correlation analysis table (DataFrame)
    """
    df = filter_strategies(df, omit_strategies)
    if omit_strategies:
        print(f"Omitted strategies: {omit_strategies}")
        print(f"Remaining: {sorted(df['sampling_strategy'].unique())}")
        print()

    # Plot 1: main scatter
    pair_df = build_pair_df(df, cv_col=cv_col, acc_col=acc_col)
    print(f"Pair-level data ({len(pair_df)} strategy×size combos):")
    print(pair_df.to_string(index=False))
    print()

    plot_cv_vs_accuracy(pair_df)
    plt.show()

    # Plot 2: faceted by sample size
    fig = plot_cv_vs_accuracy_by_size(pair_df)
    plt.show()

    # Plot 3: correlation strength heatmap across allocations
    plot_correlation_heatmap(df, cv_col=cv_col, acc_col=acc_col)
    plt.show()

    # Plot 4: partial correlation comparison
    partial_df = compute_partial_correlations(df, cv_col=cv_col, acc_col=acc_col)
    print("\nPartial correlation analysis:")
    print(partial_df.to_string(index=False))
    print()

    plot_partial_correlation_comparison(partial_df)
    plt.show()

    # Correlation table (raw, per-size, per-allocation)
    corr_df = compute_correlations(df, cv_col=cv_col, acc_col=acc_col)
    print("Full correlation summary:")
    print(corr_df.to_string(index=False))

    return corr_df, partial_df