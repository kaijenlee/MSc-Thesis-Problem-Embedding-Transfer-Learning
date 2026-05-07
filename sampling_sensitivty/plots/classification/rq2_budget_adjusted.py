"""
RQ2 — Factor Importance with Budget as Covariate
==================================================

Core idea: fit a regression predicting accuracy from all four design factors
PLUS log(n_feval_train) as a covariate. The budget covariate absorbs the
"more evaluations → better accuracy" effect. The remaining factor coefficients
tell you what matters BEYOND budget.

This separates:
  - "instances help because they increase the budget"
  - "instances help because they provide diverse training data"

Produces:
  1. OLS summary with log(budget) + all factors
  2. Type II ANOVA on the budget-adjusted model
  3. η² bar chart comparing raw vs budget-adjusted importance
  4. Partial regression plots (added-variable plots) for each factor
  5. Budget-adjusted marginal means

Usage in notebook:
    from rq2_budget_adjusted import *

    results = run_all(fold_agg_df, metric="acc_mean")
    results = run_all(fold_agg_df, metric="acc_mean", omit_strategies="cma_random")

    # Step by step
    model = fit_budget_adjusted_model(fold_agg_df, metric="acc_mean")
    anova_raw, anova_adj = compare_anova(fold_agg_df, metric="acc_mean")
    plot_importance_comparison(anova_raw, anova_adj)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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

STRATEGY_LABELS = {
    "cma_random": "CMA-ES", "uniform": "Uniform", "lhs": "LHS",
    "ilhs": "iLHS", "lhs_rcd": "LHS-RCD", "sobol": "Sobol",
}

DEFAULT_METRIC = "acc_mean"

# Clean ANOVA index names
_ANOVA_RENAME = {
    "C(sampling_strategy)":   "Strategy",
    "C(sample_size_per_dim)": "Sample size",
    "C(n_instances_train)":   "Instances",
    "C(n_runs_train)":        "Runs",
    "log_budget":             "log(budget)",
    "Residual":               "Residual",
}


def filter_strategies(df, omit_strategies=None):
    if omit_strategies is None:
        return df
    if isinstance(omit_strategies, str):
        omit_strategies = [omit_strategies]
    return df[~df["sampling_strategy"].isin(omit_strategies)].copy()


def _prep(df, metric):
    """Prepare dataframe: ensure string factors, add log budget."""
    df = df.copy()
    df["log_budget"] = np.log(df["n_feval_train"].astype(float))
    for col in FACTOR_COLS:
        df[col] = df[col].astype(str)
    return df


def _rename_anova(table):
    table = table.copy()
    table.index = [_ANOVA_RENAME.get(i, i) for i in table.index]
    return table


def _add_eta(table):
    ss_total = table["sum_sq"].sum()
    ss_resid = table.loc["Residual", "sum_sq"]
    table["eta_sq"] = table["sum_sq"] / ss_total
    table["eta_sq_partial"] = table["sum_sq"] / (table["sum_sq"] + ss_resid)
    return table


# =========================================================================
# Model fitting
# =========================================================================

def fit_raw_model(df, metric=DEFAULT_METRIC):
    """OLS with main effects only (no budget covariate)."""
    df = _prep(df, metric)
    formula = (
        f"{metric} ~ C(sampling_strategy) + C(sample_size_per_dim) + "
        f"C(n_instances_train) + C(n_runs_train)"
    )
    return ols(formula, data=df).fit()


def fit_budget_adjusted_model(df, metric=DEFAULT_METRIC):
    """OLS with log(budget) as covariate + all main effects."""
    df = _prep(df, metric)
    formula = (
        f"{metric} ~ log_budget + C(sampling_strategy) + C(sample_size_per_dim) + "
        f"C(n_instances_train) + C(n_runs_train)"
    )
    return ols(formula, data=df).fit()


# =========================================================================
# ANOVA comparison
# =========================================================================

def compare_anova(df, metric=DEFAULT_METRIC):
    """
    Compute Type II ANOVA for both the raw model and the budget-adjusted
    model. Returns both tables for comparison.

    The key comparison: if a factor's η² drops substantially when budget
    is included as a covariate, that factor's importance was driven by
    its correlation with budget rather than an independent effect.
    """
    df = _prep(df, metric)

    # Raw model
    raw_formula = (
        f"{metric} ~ C(sampling_strategy) + C(sample_size_per_dim) + "
        f"C(n_instances_train) + C(n_runs_train)"
    )
    raw_model = ols(raw_formula, data=df).fit()
    anova_raw = _rename_anova(_add_eta(anova_lm(raw_model, typ=2)))

    # Budget-adjusted model
    adj_formula = (
        f"{metric} ~ log_budget + C(sampling_strategy) + C(sample_size_per_dim) + "
        f"C(n_instances_train) + C(n_runs_train)"
    )
    adj_model = ols(adj_formula, data=df).fit()
    anova_adj = _rename_anova(_add_eta(anova_lm(adj_model, typ=2)))

    return anova_raw, anova_adj


# =========================================================================
# Plot 1 — Side-by-side η² comparison
# =========================================================================

def plot_importance_comparison(anova_raw, anova_adj, ax=None):
    """
    Grouped bar chart: for each factor, show η² from the raw model
    vs the budget-adjusted model. The drop (if any) reveals how much
    of that factor's importance was really just a budget effect.
    """
    factors = [f for f in ["Strategy", "Sample size", "Instances", "Runs"]
               if f in anova_raw.index and f in anova_adj.index]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(factors))
    width = 0.35

    raw_vals = [anova_raw.loc[f, "eta_sq"] for f in factors]
    adj_vals = [anova_adj.loc[f, "eta_sq"] for f in factors]

    bars1 = ax.bar(x - width/2, raw_vals, width, label="Raw (no budget control)",
                   color="#7fb3d8", edgecolor="white")
    bars2 = ax.bar(x + width/2, adj_vals, width, label="Budget-adjusted",
                   color="#2166ac", edgecolor="white")

    # Annotate with values and % change
    for i, (rv, av) in enumerate(zip(raw_vals, adj_vals)):
        ax.text(i - width/2, rv + 0.005, f"{rv:.3f}", ha="center",
                va="bottom", fontsize=8)
        ax.text(i + width/2, av + 0.005, f"{av:.3f}", ha="center",
                va="bottom", fontsize=8)
        if rv > 0.001:
            pct = (av - rv) / rv * 100
            color = "#d62728" if pct < -20 else "#2ca02c" if pct > 20 else "gray"
            ax.text(i, max(rv, av) + 0.025, f"{pct:+.0f}%", ha="center",
                    va="bottom", fontsize=8, fontweight="bold", color=color)

    # Show budget covariate η² as annotation
    if "log(budget)" in anova_adj.index:
        budget_eta = anova_adj.loc["log(budget)", "eta_sq"]
        ax.annotate(
            f"log(budget) alone: η² = {budget_eta:.3f}",
            xy=(0.97, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(factors, fontsize=10)
    ax.set_ylabel("η² (proportion of total variance)")
    ax.set_title("Factor importance: raw vs budget-adjusted")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return ax


# =========================================================================
# Plot 2 — Partial regression (added-variable) plots
# =========================================================================

def plot_partial_regression(df, metric=DEFAULT_METRIC, ax=None):
    """
    Added-variable plots for the budget-adjusted model.
    Each subplot shows the relationship between a factor and accuracy
    AFTER removing the effect of budget and all other factors.

    For categorical factors (strategy), shows residual box plots.
    For ordinal factors (instances, runs, sample_size), shows residual scatter.
    """
    df = _prep(df, metric)

    ordinal_factors = {
        "n_instances_train": "Instances",
        "n_runs_train": "Runs",
        "sample_size_per_dim": "Sample size",
    }

    n_plots = len(ordinal_factors) + 1  # +1 for strategy
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    # Get residuals from model without each factor (added-variable approach)
    all_cat_factors = " + ".join(f"C({c})" for c in FACTOR_COLS)

    # --- Strategy (categorical): box plot of residuals ---
    ax = axes[0]
    other_terms = "log_budget + " + " + ".join(
        f"C({c})" for c in FACTOR_COLS if c != "sampling_strategy"
    )
    resid_model = ols(f"{metric} ~ {other_terms}", data=df).fit()
    df["_resid_y"] = resid_model.resid

    strat_order = sorted(df["sampling_strategy"].unique())
    strat_labels = [STRATEGY_LABELS.get(s, s) for s in strat_order]
    box_data = [df[df["sampling_strategy"] == s]["_resid_y"].values
                for s in strat_order]

    bp = ax.boxplot(box_data, labels=strat_labels, patch_artist=True, widths=0.6)
    for patch, s in zip(bp["boxes"], strat_order):
        patch.set_facecolor(FACTOR_COLORS["sampling_strategy"])
        patch.set_alpha(0.6)

    ax.set_title("Strategy\n(budget-adjusted residuals)")
    ax.set_ylabel("Residual accuracy")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    # --- Ordinal factors: scatter of residuals ---
    for idx, (col, label) in enumerate(ordinal_factors.items(), 1):
        ax = axes[idx]
        other_terms = "log_budget + " + " + ".join(
            f"C({c})" for c in FACTOR_COLS if c != col
        )
        resid_y_model = ols(f"{metric} ~ {other_terms}", data=df).fit()
        resid_x_model = ols(f"log_budget ~ {other_terms}", data=df).fit()

        # residualise the factor against budget + other factors
        # For ordinal: use the numeric value
        df["_resid_y"] = resid_y_model.resid
        factor_numeric = df[col].astype(float)

        # Group by factor level, show mean residual
        grouped = df.groupby(col)["_resid_y"].agg(["mean", "std"]).reset_index()
        grouped[col] = grouped[col].astype(float)
        grouped = grouped.sort_values(col)

        ax.scatter(factor_numeric, df["_resid_y"], alpha=0.15, s=15,
                   color=FACTOR_COLORS[col], zorder=2)
        ax.errorbar(grouped[col], grouped["mean"], yerr=grouped["std"],
                    fmt="o-", color="black", markersize=6, linewidth=1.5,
                    capsize=3, zorder=3)

        ax.set_title(f"{label}\n(budget-adjusted residuals)")
        ax.set_xlabel(label)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Residual accuracy")
    fig.suptitle("Partial effects after controlling for budget", fontsize=12, y=1.02)
    fig.tight_layout()

    # Clean up temp columns
    df.drop(columns=["_resid_y"], inplace=True, errors="ignore")

    return fig


# =========================================================================
# Plot 3 — Budget-adjusted marginal means
# =========================================================================

def plot_adjusted_marginal_means(df, metric=DEFAULT_METRIC):
    """
    For each factor, show:
      - Raw marginal means (confounded with budget)
      - Adjusted marginal means (from the budget-adjusted model)

    The adjusted means answer: "if all configurations had the same budget,
    what would the mean accuracy be at each factor level?"
    """
    df = _prep(df, metric)

    model = fit_budget_adjusted_model(df, metric=metric)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax_idx, col in enumerate(FACTOR_COLS):
        ax = axes[ax_idx]
        label = FACTOR_LABELS[col]

        # Raw marginal means
        raw = df.groupby(col)[metric].agg(["mean", "std"]).reset_index()
        raw = raw.sort_values(col)

        # Adjusted marginal means: predict at each level, holding
        # other factors at their observed distribution and budget at
        # the grand mean log budget
        mean_log_budget = df["log_budget"].mean()
        adj_means = []

        for level in raw[col].values:
            # Create prediction data: all rows but with this factor
            # fixed and budget set to mean
            pred_df = df.copy()
            pred_df[col] = level
            pred_df["log_budget"] = mean_log_budget
            pred = model.predict(pred_df)
            adj_means.append(pred.mean())

        raw_vals = raw["mean"].values
        adj_vals = np.array(adj_means)

        x_labels = raw[col].values
        if col == "sampling_strategy":
            x_labels = [STRATEGY_LABELS.get(s, s) for s in x_labels]

        x_pos = np.arange(len(x_labels))
        width = 0.35

        ax.bar(x_pos - width/2, raw_vals, width, label="Raw",
               color="#7fb3d8", edgecolor="white", alpha=0.8)
        ax.bar(x_pos + width/2, adj_vals, width, label="Budget-adjusted",
               color="#2166ac", edgecolor="white", alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(label)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.3)

        if ax_idx == 0:
            ax.set_ylabel("Mean accuracy")
            ax.legend(fontsize=7)

        # Annotate range
        raw_range = raw_vals.max() - raw_vals.min()
        adj_range = adj_vals.max() - adj_vals.min()
        ax.text(0.95, 0.05,
                f"raw range: {raw_range:.3f}\nadj range: {adj_range:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.6))

    fig.suptitle("Raw vs budget-adjusted marginal means", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# =========================================================================
# Summary printer
# =========================================================================

def print_comparison(anova_raw, anova_adj):
    """Print side-by-side comparison of raw vs adjusted η²."""
    factors = [f for f in ["Strategy", "Sample size", "Instances", "Runs"]
               if f in anova_raw.index]

    print(f"\n{'Factor':<15} {'η² (raw)':>10} {'η² (adj)':>10} {'Change':>10}")
    print("-" * 50)
    for f in factors:
        rv = anova_raw.loc[f, "eta_sq"]
        av = anova_adj.loc[f, "eta_sq"]
        pct = (av - rv) / rv * 100 if rv > 0.001 else 0
        print(f"{f:<15} {rv:>10.4f} {av:>10.4f} {pct:>+9.1f}%")

    if "log(budget)" in anova_adj.index:
        bv = anova_adj.loc["log(budget)", "eta_sq"]
        print(f"{'log(budget)':<15} {'—':>10} {bv:>10.4f}")

    for label, table in [("Raw", anova_raw), ("Adjusted", anova_adj)]:
        resid = table.loc["Residual", "eta_sq"]
        print(f"{'Residual ('+label+')':<15} {resid:>10.4f}")
    print()


# =========================================================================
# Convenience: run everything
# =========================================================================

def run_all(df, metric=DEFAULT_METRIC, omit_strategies=None):
    """
    Full budget-adjusted factor importance analysis.

    Returns
    -------
    dict with keys: anova_raw, anova_adj, model_raw, model_adj
    """
    df = filter_strategies(df, omit_strategies)
    if omit_strategies:
        print(f"Omitted strategies: {omit_strategies}")
        print(f"Remaining: {sorted(df['sampling_strategy'].unique())}")
        print()

    # Compare ANOVA tables
    anova_raw, anova_adj = compare_anova(df, metric=metric)
    print_comparison(anova_raw, anova_adj)

    # Plot 1: η² comparison
    print("Plot 1: Raw vs budget-adjusted η² ...")
    plot_importance_comparison(anova_raw, anova_adj)
    plt.show()

    # Plot 2: Partial regression plots
    print("Plot 2: Partial regression (added-variable) plots ...")
    fig = plot_partial_regression(df, metric=metric)
    plt.show()

    # Plot 3: Adjusted marginal means
    print("Plot 3: Raw vs adjusted marginal means ...")
    fig = plot_adjusted_marginal_means(df, metric=metric)
    plt.show()

    return {
        "anova_raw": anova_raw,
        "anova_adj": anova_adj,
    }