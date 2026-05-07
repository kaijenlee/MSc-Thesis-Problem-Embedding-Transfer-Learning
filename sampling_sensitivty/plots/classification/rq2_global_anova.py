"""
RQ2 — Analysis 1a: Global ANOVA / Variance Decomposition
==========================================================

Formal decomposition of classification accuracy variance across design factors:
  - sampling_strategy
  - sample_size_per_dim
  - n_instances_train
  - n_runs_train

Produces:
  1. Type II ANOVA table (SS, df, F, p, η² per factor)
  2. η² bar chart — proportion of variance explained by each main effect
  3. Interaction analysis: two-way η² heatmap for all factor pairs
  4. Full regression summary with interactions (OLS)

Note on Type II vs Type III:
  Type II tests each factor after accounting for all other main effects
  (but not interactions). This is appropriate here because we have a
  balanced-ish design and care about main effects primarily. Type III
  would require sum-to-zero coding and is more relevant when interactions
  dominate.

Usage in notebook:
    from rq2_global_anova import *

    # Full analysis
    anova_df, interaction_df = run_all(fold_agg_df, metric="acc_mean")

    # Without CMA-ES
    anova_df, interaction_df = run_all(fold_agg_df, metric="acc_mean",
                                        omit_strategies="cma_random")

    # Step by step
    anova_df = compute_anova(fold_agg_df, metric="acc_mean")
    plot_eta_squared_bar(anova_df)
    interaction_df = compute_interaction_effects(fold_agg_df, metric="acc_mean")
    plot_interaction_heatmap(interaction_df)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

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

DEFAULT_METRIC = "acc_mean"


def filter_strategies(df, omit_strategies=None):
    """Drop rows whose sampling_strategy is in omit_strategies."""
    if omit_strategies is None:
        return df
    if isinstance(omit_strategies, str):
        omit_strategies = [omit_strategies]
    return df[~df["sampling_strategy"].isin(omit_strategies)].copy()


# =========================================================================
# Type II ANOVA (using statsmodels)
# =========================================================================

def compute_anova(df, metric=DEFAULT_METRIC):
    """
    Fit an OLS model with all four main effects and compute Type II ANOVA.

    Factors are treated as categorical (C()) so that strategies, sample sizes,
    etc. are properly dummy-coded.

    Returns
    -------
    anova_df : DataFrame with columns SS, df, F, p, eta_sq, eta_sq_partial
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    # Ensure proper types
    df = df.copy()
    for col in FACTOR_COLS:
        df[col] = df[col].astype(str)

    formula = (
        f"{metric} ~ "
        f"C(sampling_strategy) + C(sample_size_per_dim) + "
        f"C(n_instances_train) + C(n_runs_train)"
    )

    model = ols(formula, data=df).fit()
    anova_table = anova_lm(model, typ=2)

    # Add η² (SS_factor / SS_total)
    ss_total = anova_table["sum_sq"].sum()
    anova_table["eta_sq"] = anova_table["sum_sq"] / ss_total

    # Add partial η² (SS_factor / (SS_factor + SS_residual))
    ss_resid = anova_table.loc["Residual", "sum_sq"]
    anova_table["eta_sq_partial"] = (
        anova_table["sum_sq"] / (anova_table["sum_sq"] + ss_resid)
    )

    # Clean up index names
    rename = {
        "C(sampling_strategy)":   "Strategy",
        "C(sample_size_per_dim)": "Sample size",
        "C(n_instances_train)":   "Instances",
        "C(n_runs_train)":        "Runs",
        "Residual":               "Residual",
    }
    anova_table.index = [rename.get(i, i) for i in anova_table.index]

    return anova_table


# =========================================================================
# ANOVA with interactions
# =========================================================================

def compute_anova_with_interactions(df, metric=DEFAULT_METRIC):
    """
    Fit OLS with all main effects + all two-way interactions.
    Returns Type II ANOVA table.
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    df = df.copy()
    for col in FACTOR_COLS:
        df[col] = df[col].astype(str)

    main = " + ".join(f"C({c})" for c in FACTOR_COLS)
    interactions = " + ".join(
        f"C({a}):C({b})" for a, b in combinations(FACTOR_COLS, 2)
    )
    formula = f"{metric} ~ {main} + {interactions}"

    model = ols(formula, data=df).fit()
    anova_table = anova_lm(model, typ=2)

    ss_total = anova_table["sum_sq"].sum()
    anova_table["eta_sq"] = anova_table["sum_sq"] / ss_total

    ss_resid = anova_table.loc["Residual", "sum_sq"]
    anova_table["eta_sq_partial"] = (
        anova_table["sum_sq"] / (anova_table["sum_sq"] + ss_resid)
    )

    # Clean up names
    rename = {
        "C(sampling_strategy)":   "Strategy",
        "C(sample_size_per_dim)": "Sample size",
        "C(n_instances_train)":   "Instances",
        "C(n_runs_train)":        "Runs",
        "Residual":               "Residual",
    }
    for a, b in combinations(FACTOR_COLS, 2):
        key = f"C({a}):C({b})"
        la = FACTOR_LABELS.get(a, a)
        lb = FACTOR_LABELS.get(b, b)
        rename[key] = f"{la} × {lb}"

    anova_table.index = [rename.get(i, i) for i in anova_table.index]

    return anova_table


# =========================================================================
# Interaction effects (manual η² for each pair)
# =========================================================================

def compute_interaction_effects(df, metric=DEFAULT_METRIC):
    """
    For each pair of factors, compute η² of the interaction term
    (the joint grouping minus the sum of individual main effects).

    Returns DataFrame: factor_a, factor_b, eta_sq_joint, eta_sq_a,
                       eta_sq_b, eta_sq_interaction
    """
    df = df.reset_index(drop=True)
    y = df[metric].values
    grand_mean = y.mean()
    ss_total = ((y - grand_mean) ** 2).sum()

    def _eta_sq(groups):
        """η² for a grouping variable."""
        ss = 0.0
        for _, idx in groups.items():
            grp = y[idx]
            ss += len(grp) * (grp.mean() - grand_mean) ** 2
        return ss / ss_total

    rows = []
    for a, b in combinations(FACTOR_COLS, 2):
        # Individual effects
        groups_a = df.groupby(a).groups
        groups_b = df.groupby(b).groups
        eta_a = _eta_sq({k: v.values for k, v in groups_a.items()})
        eta_b = _eta_sq({k: v.values for k, v in groups_b.items()})

        # Joint effect (grouping by the pair)
        groups_ab = df.groupby([a, b]).groups
        eta_joint = _eta_sq({k: v.values for k, v in groups_ab.items()})

        # Interaction = joint - sum of main effects
        eta_interaction = max(0, eta_joint - eta_a - eta_b)

        rows.append({
            "factor_a": FACTOR_LABELS[a],
            "factor_b": FACTOR_LABELS[b],
            "eta_sq_a": eta_a,
            "eta_sq_b": eta_b,
            "eta_sq_joint": eta_joint,
            "eta_sq_interaction": eta_interaction,
        })

    return pd.DataFrame(rows)


# =========================================================================
# Plot 1 — η² bar chart (main effects)
# =========================================================================

def plot_eta_squared_bar(anova_df, ax=None):
    """
    Horizontal bar chart of η² for each main effect from the ANOVA table.
    """
    # Exclude Residual row
    factors = [f for f in anova_df.index if f != "Residual"]
    eta_vals = [anova_df.loc[f, "eta_sq"] for f in factors]

    # Map back to colours
    label_to_col = {v: k for k, v in FACTOR_LABELS.items()}
    colors = [FACTOR_COLORS.get(label_to_col.get(f, ""), "#999999") for f in factors]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    y_pos = np.arange(len(factors))
    ax.barh(y_pos, eta_vals, color=colors, edgecolor="white", height=0.6)

    # Annotate values
    for i, (f, v) in enumerate(zip(factors, eta_vals)):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(factors)
    ax.set_xlabel("η² (proportion of total variance)")
    ax.set_title("Global factor importance — main effects only")
    ax.set_xlim(0, max(eta_vals) * 1.25)
    ax.grid(axis="x", alpha=0.3)

    # Show residual as annotation
    resid = anova_df.loc["Residual", "eta_sq"]
    ax.annotate(f"Residual (unexplained): {resid:.3f}",
                xy=(0.97, 0.05), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8, color="gray")

    plt.tight_layout()
    return ax


# =========================================================================
# Plot 2 — η² bar chart (main effects + interactions)
# =========================================================================

def plot_eta_squared_full(anova_int_df, ax=None):
    """
    Horizontal bar chart from the ANOVA-with-interactions table,
    showing both main effects and interaction terms.
    """
    factors = [f for f in anova_int_df.index if f != "Residual"]
    eta_vals = [anova_int_df.loc[f, "eta_sq"] for f in factors]

    # Colour: main effects get their colour, interactions get gray
    label_to_col = {v: k for k, v in FACTOR_LABELS.items()}
    colors = []
    for f in factors:
        if f in label_to_col:
            colors.append(FACTOR_COLORS.get(label_to_col[f], "#999999"))
        else:
            colors.append("#BBBBBB")

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, max(5, len(factors) * 0.45)))

    y_pos = np.arange(len(factors))
    ax.barh(y_pos, eta_vals, color=colors, edgecolor="white", height=0.6)

    for i, v in enumerate(eta_vals):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(factors, fontsize=8)
    ax.set_xlabel("η² (proportion of total variance)")
    ax.set_title("Global factor importance — main effects + two-way interactions")
    ax.set_xlim(0, max(eta_vals) * 1.25)
    ax.grid(axis="x", alpha=0.3)

    resid = anova_int_df.loc["Residual", "eta_sq"]
    ax.annotate(f"Residual: {resid:.3f}",
                xy=(0.97, 0.05), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8, color="gray")

    plt.tight_layout()
    return ax


# =========================================================================
# Plot 3 — Interaction heatmap
# =========================================================================

def plot_interaction_heatmap(interaction_df, ax=None):
    """
    Symmetric heatmap of interaction η² for each factor pair.
    """
    labels = sorted(set(interaction_df["factor_a"]) | set(interaction_df["factor_b"]))
    n = len(labels)
    mat = np.zeros((n, n))

    for _, row in interaction_df.iterrows():
        i = labels.index(row["factor_a"])
        j = labels.index(row["factor_b"])
        mat[i, j] = row["eta_sq_interaction"]
        mat[j, i] = row["eta_sq_interaction"]

    # Diagonal = main effect η²
    for _, row in interaction_df.iterrows():
        i = labels.index(row["factor_a"])
        j = labels.index(row["factor_b"])
        mat[i, i] = max(mat[i, i], row["eta_sq_a"])
        mat[j, j] = max(mat[j, j], row["eta_sq_b"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(mat, cmap="YlOrRd", aspect="equal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate
    for i in range(n):
        for j in range(n):
            label = "main" if i == j else "inter."
            ax.text(j, i, f"{mat[i,j]:.3f}\n({label})", ha="center", va="center",
                    fontsize=7, color="white" if mat[i, j] > mat.max() * 0.5 else "black")

    ax.set_title("Main effects (diagonal) and interaction η² (off-diagonal)")
    plt.colorbar(im, ax=ax, label="η²", shrink=0.8)

    plt.tight_layout()
    return ax


# =========================================================================
# Summary printer
# =========================================================================

def print_anova_summary(anova_df, title="ANOVA Table"):
    """Pretty-print an ANOVA table with key columns."""
    display_cols = ["sum_sq", "df", "F", "PR(>F)", "eta_sq"]
    available = [c for c in display_cols if c in anova_df.columns]

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    formatted = anova_df[available].copy()
    formatted.columns = ["SS", "df", "F", "p-value", "η²"][:len(available)]
    print(formatted.to_string(float_format=lambda x: f"{x:.4f}"))

    # Highlight the dominant factor
    factors = [f for f in anova_df.index if f != "Residual"]
    if factors:
        dominant = max(factors, key=lambda f: anova_df.loc[f, "eta_sq"])
        print(f"\n  → Dominant factor: {dominant} "
              f"(η² = {anova_df.loc[dominant, 'eta_sq']:.3f})")
    print()


# =========================================================================
# Convenience: run everything
# =========================================================================

def run_all(df, metric=DEFAULT_METRIC, omit_strategies=None):
    """
    Compute ANOVA tables and produce all plots.

    Parameters
    ----------
    df : fold-aggregated DataFrame
    metric : accuracy column name
    omit_strategies : str or list[str], strategies to exclude

    Returns
    -------
    anova_df : main-effects ANOVA table
    interaction_df : pairwise interaction η² table
    """
    df = filter_strategies(df, omit_strategies)
    if omit_strategies:
        print(f"Omitted strategies: {omit_strategies}")
        print(f"Remaining: {sorted(df['sampling_strategy'].unique())}")
        print()

    # Main effects ANOVA
    anova_df = compute_anova(df, metric=metric)
    print_anova_summary(anova_df, title="Type II ANOVA — Main Effects")

    plot_eta_squared_bar(anova_df)
    plt.show()

    # ANOVA with interactions
    anova_int_df = compute_anova_with_interactions(df, metric=metric)
    print_anova_summary(anova_int_df,
                        title="Type II ANOVA — Main Effects + Two-Way Interactions")

    plot_eta_squared_full(anova_int_df)
    plt.show()

    # Interaction heatmap
    interaction_df = compute_interaction_effects(df, metric=metric)
    print("\nPairwise interaction η²:")
    print(interaction_df.to_string(index=False))
    print()

    plot_interaction_heatmap(interaction_df)
    plt.show()

    return anova_df, interaction_df