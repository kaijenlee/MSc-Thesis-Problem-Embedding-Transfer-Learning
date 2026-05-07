"""
RQ2 — Functional ANOVA (fANOVA) via Random Forest
====================================================

Implements the functional ANOVA decomposition (Hutter et al., 2014) using
scikit-learn's RandomForestRegressor instead of the `fanova` package
(which has difficult C++ dependencies).

The idea: fit a random forest predicting accuracy from the design factors,
then decompose the predicted variance into contributions from each factor
(and optionally pairwise interactions) by marginalising over subsets of
features.

For a factor X_j, its importance is:
  V_j = Var_{X_j}[ E_{X_{-j}}[ f(X) | X_j ] ]

i.e., how much does the prediction vary when we change X_j alone,
averaging over all other factors? This is computed by:
  1. For each unique value of X_j, predict accuracy for all combinations
     of the other factors, then average → gives E[f | X_j = x_j]
  2. Compute the variance of these conditional means across values of X_j

This properly handles interactions and non-linear effects, and the
contributions sum to the total predicted variance (by construction
of the ANOVA decomposition).

Usage in notebook:
    from rq2_fanova import *

    results = run_all(fold_agg_df, metric="acc_mean")
    results = run_all(fold_agg_df, metric="acc_mean", omit_strategies="cma_random")

    # Step by step
    rf, enc = fit_forest(fold_agg_df, metric="acc_mean")
    importance = compute_fanova(rf, enc, fold_agg_df)
    plot_fanova_importance(importance)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from itertools import combinations, product
import warnings

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
    "log_budget":          "log(Budget)",
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


def filter_strategies(df, omit_strategies=None):
    if omit_strategies is None:
        return df
    if isinstance(omit_strategies, str):
        omit_strategies = [omit_strategies]
    return df[~df["sampling_strategy"].isin(omit_strategies)].copy()


# =========================================================================
# Encoding and model fitting
# =========================================================================

def _get_factors(include_budget=False):
    """Return the list of factor columns, optionally including log budget."""
    cols = list(FACTOR_COLS)
    if include_budget:
        cols.append("log_budget")
    return cols


def fit_forest(df, metric=DEFAULT_METRIC, n_estimators=500, random_state=42,
               include_budget=False):
    """
    Fit a random forest predicting accuracy from the design factors,
    optionally including log(n_feval_train) as a fifth factor.

    Parameters
    ----------
    include_budget : bool
        If True, add log(n_feval_train) as a continuous factor. The fANOVA
        decomposition will then separate budget effects from design effects,
        and budget×factor interactions reveal whether factor rankings shift
        with budget.

    Returns
    -------
    rf : fitted RandomForestRegressor
    enc : fitted OrdinalEncoder (maps factor values to integers)
    factor_levels : dict mapping each factor to its unique values
    factors_used : list of factor column names used
    """
    df = df.copy()
    factors = _get_factors(include_budget)

    if include_budget:
        df["log_budget"] = np.log(df["n_feval_train"].astype(float))

    for col in FACTOR_COLS:
        df[col] = df[col].astype(str)

    # Encode categorical factors; log_budget stays numeric
    cat_cols = [c for c in factors if c != "log_budget"]
    enc = OrdinalEncoder()
    X_cat = enc.fit_transform(df[cat_cols])

    if include_budget:
        X_budget = df["log_budget"].values.reshape(-1, 1)
        X = np.hstack([X_cat, X_budget])
    else:
        X = X_cat

    y = df[metric].values.astype(float)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)

    # Store factor levels for marginalisation
    factor_levels = {}
    for i, col in enumerate(cat_cols):
        factor_levels[col] = enc.categories_[i]

    if include_budget:
        # Discretise log_budget into ~10 levels for marginalisation
        budget_vals = df["log_budget"].values
        percentiles = np.percentile(budget_vals, np.linspace(0, 100, 11))
        budget_levels = np.unique(np.round(percentiles, 4))
        factor_levels["log_budget"] = budget_levels.astype(str)

    print(f"Random forest R² = {rf.score(X, y):.4f}")
    if hasattr(rf, 'oob_score_'):
        print(f"OOB R² = {rf.oob_score_:.4f}")
    print(f"Factors: {factors}")

    return rf, enc, factor_levels, factors


def _build_grid(factor_levels, factors):
    """Build the full grid of all factor combinations."""
    level_lists = [factor_levels[col] for col in factors]
    grid = np.array(list(product(*level_lists)))
    return grid


def _encode_grid(grid, enc, factors):
    """Encode a grid for prediction. Handles mixed categorical + continuous."""
    cat_cols = [c for c in factors if c != "log_budget"]
    cat_indices = [factors.index(c) for c in cat_cols]
    grid_cat = grid[:, cat_indices]
    encoded_cat = enc.transform(grid_cat)

    if "log_budget" in factors:
        budget_idx = factors.index("log_budget")
        budget_vals = grid[:, budget_idx].astype(float).reshape(-1, 1)
        return np.hstack([encoded_cat, budget_vals])
    else:
        return encoded_cat


# =========================================================================
# fANOVA decomposition
# =========================================================================

def compute_fanova(rf, enc, factor_levels, factors, include_interactions=True):
    """
    Compute functional ANOVA decomposition.

    For each factor X_j:
      V_j = Var_{X_j}[ E_{X_{-j}}[ f(X) | X_j ] ]

    For each pair (X_i, X_j):
      V_{ij} = Var_{X_i,X_j}[ E_{X_{-ij}}[ f(X) | X_i, X_j ] ] - V_i - V_j

    Parameters
    ----------
    factors : list of factor column names used in the model

    Returns
    -------
    dict with keys:
      main_effects : DataFrame (factor, importance, importance_pct)
      interactions : DataFrame (factor_a, factor_b, importance, importance_pct)
                     (only if include_interactions=True)
      total_variance : float
    """
    grid = _build_grid(factor_levels, factors)
    grid_encoded = _encode_grid(grid, enc, factors)
    predictions = rf.predict(grid_encoded)

    # Total predicted variance
    total_var = np.var(predictions)
    grand_mean = np.mean(predictions)

    # Create a DataFrame for easier grouping
    grid_df = pd.DataFrame(grid, columns=factors)
    grid_df["pred"] = predictions

    # --- Main effects ---
    main_effects = []
    conditional_means = {}

    for col in factors:
        cond_means = grid_df.groupby(col)["pred"].mean()
        conditional_means[col] = cond_means
        v_j = np.var(cond_means.values)

        main_effects.append({
            "factor": col,
            "factor_label": FACTOR_LABELS.get(col, col),
            "importance": v_j,
        })

    main_df = pd.DataFrame(main_effects)
    main_df["importance_pct"] = main_df["importance"] / total_var * 100

    result = {
        "main_effects": main_df,
        "total_variance": total_var,
        "grand_mean": grand_mean,
        "factors": factors,
    }

    # --- Pairwise interactions ---
    if include_interactions:
        interactions = []
        for col_a, col_b in combinations(factors, 2):
            cond_means_ij = grid_df.groupby([col_a, col_b])["pred"].mean()

            v_joint = np.var(cond_means_ij.values)
            v_i = main_df.loc[main_df["factor"] == col_a, "importance"].values[0]
            v_j = main_df.loc[main_df["factor"] == col_b, "importance"].values[0]
            v_interaction = max(0, v_joint - v_i - v_j)

            interactions.append({
                "factor_a": col_a,
                "factor_b": col_b,
                "label": f"{FACTOR_LABELS.get(col_a, col_a)} × {FACTOR_LABELS.get(col_b, col_b)}",
                "importance": v_interaction,
            })

        int_df = pd.DataFrame(interactions)
        int_df["importance_pct"] = int_df["importance"] / total_var * 100
        result["interactions"] = int_df

    return result


# =========================================================================
# Plot 1 — Main effect importance bar chart
# =========================================================================

def plot_fanova_importance(result, ax=None):
    """
    Horizontal bar chart of fANOVA main effect importance (% of total
    predicted variance).
    """
    main_df = result["main_effects"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    y_pos = np.arange(len(main_df))
    colors = [FACTOR_COLORS.get(f, "gray") for f in main_df["factor"]]

    ax.barh(y_pos, main_df["importance_pct"], color=colors,
            edgecolor="white", height=0.6)

    for i, (_, row) in enumerate(main_df.iterrows()):
        ax.text(row["importance_pct"] + 0.5, i,
                f"{row['importance_pct']:.1f}%", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(main_df["factor_label"])
    ax.set_xlabel("% of predicted variance explained")
    ax.set_title("fANOVA — Main effects")
    ax.grid(axis="x", alpha=0.3)

    # Annotate residual (higher-order effects)
    main_total = main_df["importance_pct"].sum()
    if "interactions" in result:
        int_total = result["interactions"]["importance_pct"].sum()
        residual = 100 - main_total - int_total
        note = (f"Main effects: {main_total:.1f}%\n"
                f"Interactions: {int_total:.1f}%\n"
                f"Higher-order: {residual:.1f}%")
    else:
        note = f"Main effects: {main_total:.1f}%"

    ax.annotate(note, xy=(0.97, 0.05), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    return ax


# =========================================================================
# Plot 2 — Main effects + interactions combined
# =========================================================================

def plot_fanova_full(result, ax=None):
    """
    Horizontal bar chart showing both main effects and pairwise interactions,
    sorted by importance.
    """
    main_df = result["main_effects"][["factor_label", "importance_pct"]].copy()
    main_df.columns = ["label", "pct"]
    main_df["type"] = "main"

    rows = [main_df]
    if "interactions" in result:
        int_df = result["interactions"][["label", "importance_pct"]].copy()
        int_df.columns = ["label", "pct"]
        int_df["type"] = "interaction"
        rows.append(int_df)

    combined = pd.concat(rows, ignore_index=True)
    combined = combined.sort_values("pct", ascending=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, max(4, len(combined) * 0.45)))

    y_pos = np.arange(len(combined))
    colors = ["#2166ac" if t == "main" else "#BBBBBB"
              for t in combined["type"]]

    ax.barh(y_pos, combined["pct"], color=colors, edgecolor="white", height=0.6)

    for i, (_, row) in enumerate(combined.iterrows()):
        ax.text(row["pct"] + 0.3, i, f"{row['pct']:.1f}%",
                va="center", fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(combined["label"], fontsize=9)
    ax.set_xlabel("% of predicted variance explained")
    ax.set_title("fANOVA — Main effects + interactions")
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#2166ac", label="Main effect"),
        Patch(color="#BBBBBB", label="Interaction"),
    ], loc="lower right", fontsize=8)

    plt.tight_layout()
    return ax


# =========================================================================
# Plot 3 — Marginal effect curves (E[f | X_j])
# =========================================================================

def plot_marginal_curves(rf, enc, factor_levels, result, metric=DEFAULT_METRIC):
    """
    For each factor, plot E[f | X_j = x_j] — the expected accuracy at each
    level of that factor, averaging over all other factors.

    This is the fANOVA marginal effect: it shows the shape of each
    factor's influence (linear, saturating, etc.).
    """
    factors = result["factors"]
    grid = _build_grid(factor_levels, factors)
    grid_encoded = _encode_grid(grid, enc, factors)
    predictions = rf.predict(grid_encoded)

    grid_df = pd.DataFrame(grid, columns=factors)
    grid_df["pred"] = predictions

    fig, axes = plt.subplots(1, len(factors), figsize=(5 * len(factors), 5))
    if len(factors) == 1:
        axes = [axes]

    for ax, col in zip(axes, factors):
        label = FACTOR_LABELS.get(col, col)
        cond_means = grid_df.groupby(col)["pred"].agg(["mean", "std"]).reset_index()

        # Sort numerically where possible
        try:
            cond_means["_sort"] = cond_means[col].astype(float)
            cond_means = cond_means.sort_values("_sort")
        except (ValueError, TypeError):
            pass

        x_labels = cond_means[col].values
        if col == "sampling_strategy":
            x_labels = [STRATEGY_LABELS.get(s, s) for s in x_labels]
        elif col == "log_budget":
            x_labels = [f"{np.exp(float(v)):.0f}" for v in x_labels]

        x_pos = np.arange(len(x_labels))

        color = FACTOR_COLORS.get(col, "#666666")
        ax.bar(x_pos, cond_means["mean"], yerr=cond_means["std"],
               color=color, alpha=0.7, capsize=3, edgecolor="white")

        # Grand mean line
        ax.axhline(result["grand_mean"], color="gray", linestyle="--",
                   linewidth=1, alpha=0.6)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        pct = result['main_effects'].loc[
            result['main_effects']['factor'] == col, 'importance_pct'
        ].values[0]
        ax.set_title(f"{label}\n({pct:.1f}% of variance)", fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.3)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("E[accuracy | factor level]")
    fig.suptitle("fANOVA marginal effects — E[f | X_j]", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# =========================================================================
# Plot 4 — Interaction heatmap
# =========================================================================

def plot_fanova_interaction_heatmap(result, ax=None):
    """
    Symmetric heatmap: diagonal = main effect %, off-diagonal = interaction %.
    """
    if "interactions" not in result:
        print("No interactions computed")
        return None

    main_df = result["main_effects"]
    int_df = result["interactions"]
    factors = result["factors"]

    labels = [FACTOR_LABELS.get(col, col) for col in factors]
    n = len(labels)
    mat = np.zeros((n, n))

    # Diagonal: main effects
    for _, row in main_df.iterrows():
        if row["factor"] in factors:
            i = factors.index(row["factor"])
            mat[i, i] = row["importance_pct"]

    # Off-diagonal: interactions
    for _, row in int_df.iterrows():
        if row["factor_a"] in factors and row["factor_b"] in factors:
            i = factors.index(row["factor_a"])
            j = factors.index(row["factor_b"])
            mat[i, j] = row["importance_pct"]
            mat[j, i] = row["importance_pct"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(mat, cmap="YlOrRd", aspect="equal")

    for i in range(n):
        for j in range(n):
            label_text = "main" if i == j else "inter."
            color = "white" if mat[i, j] > mat.max() * 0.5 else "black"
            ax.text(j, i, f"{mat[i,j]:.1f}%\n({label_text})",
                    ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("fANOVA — Main effects (diagonal) + interactions (off-diagonal)")
    plt.colorbar(im, ax=ax, label="% of variance", shrink=0.8)

    plt.tight_layout()
    return ax


# =========================================================================
# Convenience: run everything
# =========================================================================

def run_all(df, metric=DEFAULT_METRIC, omit_strategies=None,
            include_budget=False, n_estimators=500, random_state=42):
    """
    Full fANOVA analysis.

    Parameters
    ----------
    include_budget : bool
        If True, include log(n_feval_train) as a fifth factor. This separates
        budget effects from design effects, and budget×factor interactions
        reveal whether factor rankings shift with budget.

    Returns
    -------
    result : dict with main_effects, interactions, total_variance, etc.
    """
    df = filter_strategies(df, omit_strategies)
    if omit_strategies:
        print(f"Omitted strategies: {omit_strategies}")
        print(f"Remaining: {sorted(df['sampling_strategy'].unique())}")
        print()

    # Fit forest
    print("Fitting random forest ...")
    rf, enc, factor_levels, factors = fit_forest(
        df, metric=metric, n_estimators=n_estimators,
        random_state=random_state, include_budget=include_budget,
    )
    print()

    # Compute fANOVA
    print("Computing fANOVA decomposition ...")
    result = compute_fanova(rf, enc, factor_levels, factors,
                            include_interactions=True)

    print("\nMain effects:")
    print(result["main_effects"].to_string(index=False))
    if "interactions" in result:
        print("\nInteractions:")
        print(result["interactions"].to_string(index=False))
    print(f"\nTotal predicted variance: {result['total_variance']:.6f}")
    print()

    # Plots
    print("Plot 1: Main effect importance ...")
    plot_fanova_importance(result)
    plt.show()

    print("Plot 2: Main effects + interactions ...")
    plot_fanova_full(result)
    plt.show()

    print("Plot 3: Marginal effect curves ...")
    fig = plot_marginal_curves(rf, enc, factor_levels, result, metric=metric)
    plt.show()

    print("Plot 4: Interaction heatmap ...")
    plot_fanova_interaction_heatmap(result)
    plt.show()

    return result


# =========================================================================
# Budget-tier fANOVA: run decomposition within each budget tier
# =========================================================================

def compute_fanova_by_tier(df, metric=DEFAULT_METRIC, n_tiers=4,
                            n_estimators=500, random_state=42):
    """
    Run fANOVA separately within each budget tier.

    Returns
    -------
    tier_results : list of (tier_label, result_dict) tuples
    summary_df : DataFrame with factor importance per tier (for plotting)
    """
    df = df.copy()
    df["budget_tier"] = pd.qcut(df["n_feval_train"], q=n_tiers)
    tier_labels = df["budget_tier"].cat.categories.tolist()

    tier_results = []
    summary_rows = []

    for tier in tier_labels:
        sub = df[df["budget_tier"] == tier].copy().reset_index(drop=True)
        n_rows = len(sub)

        if n_rows < 20:
            print(f"  Tier {tier}: skipping ({n_rows} rows)")
            continue

        skip = False
        for col in FACTOR_COLS:
            if sub[col].nunique() < 2:
                print(f"  Tier {tier}: skipping (factor {col} has <2 levels)")
                skip = True
                break
        if skip:
            continue

        print(f"  Tier {tier}: {n_rows} rows, fitting RF ...")

        for col in FACTOR_COLS:
            sub[col] = sub[col].astype(str)

        enc = OrdinalEncoder()
        X = enc.fit_transform(sub[FACTOR_COLS])
        y = sub[metric].values.astype(float)

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=max(2, n_rows // 20),
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        r2 = rf.score(X, y)

        factor_levels = {}
        for i, col in enumerate(FACTOR_COLS):
            factor_levels[col] = enc.categories_[i]

        factors = list(FACTOR_COLS)
        result = compute_fanova(rf, enc, factor_levels, factors,
                                include_interactions=False)
        result["r2"] = r2
        result["n_rows"] = n_rows

        tier_results.append((tier, result))

        for _, row in result["main_effects"].iterrows():
            summary_rows.append({
                "budget_tier": str(tier),
                "factor": row["factor"],
                "factor_label": row["factor_label"],
                "importance_pct": row["importance_pct"],
                "importance": row["importance"],
                "n_rows": n_rows,
                "r2": r2,
            })

    summary_df = pd.DataFrame(summary_rows)
    return tier_results, summary_df


def plot_fanova_by_tier(summary_df, ax=None):
    """
    Grouped bar chart: x = budget tier, bars = factors,
    height = fANOVA importance %.
    """
    tiers = summary_df["budget_tier"].unique()
    n_tiers = len(tiers)
    n_factors = len(FACTOR_COLS)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(10, n_tiers * 2.5), 5))

    bar_width = 0.8 / n_factors
    x = np.arange(n_tiers)

    for i, col in enumerate(FACTOR_COLS):
        sub = summary_df[summary_df["factor"] == col]
        vals = []
        for t in tiers:
            match = sub[sub["budget_tier"] == t]
            vals.append(match["importance_pct"].values[0] if len(match) > 0 else 0)

        offset = (i - n_factors / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, width=bar_width,
               color=FACTOR_COLORS.get(col, "gray"),
               label=FACTOR_LABELS.get(col, col),
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(tiers, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Budget tier (function evaluations)")
    ax.set_ylabel("fANOVA importance (% of predicted variance)")
    ax.set_title("fANOVA factor importance by budget tier")
    ax.legend(title="Design factor", loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return ax


def plot_fanova_share_by_tier(summary_df, ax=None):
    """
    Stacked bar chart: relative fANOVA importance per budget tier.
    Unlike the η²-based version, these shares come from a proper variance
    decomposition — main effects don't double-count.
    """
    tiers = summary_df["budget_tier"].unique()
    n_tiers = len(tiers)

    mat = np.zeros((n_tiers, len(FACTOR_COLS)))
    for i, tier in enumerate(tiers):
        for j, col in enumerate(FACTOR_COLS):
            match = summary_df[(summary_df["budget_tier"] == tier) &
                               (summary_df["factor"] == col)]
            if len(match) > 0:
                mat[i, j] = match["importance_pct"].values[0]

    # Normalise rows to sum to 100
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat_norm = mat / row_sums * 100

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(n_tiers)
    bottoms = np.zeros(n_tiers)
    for j, col in enumerate(FACTOR_COLS):
        ax.bar(x, mat_norm[:, j], bottom=bottoms, width=0.7,
               color=FACTOR_COLORS.get(col, "gray"),
               label=FACTOR_LABELS.get(col, col),
               edgecolor="white", linewidth=0.5)
        bottoms += mat_norm[:, j]

    ax.set_xticks(x)
    ax.set_xticklabels(tiers, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Budget tier (function evaluations)")
    ax.set_ylabel("Relative share (%)")
    ax.set_title("Relative fANOVA importance by budget tier")
    ax.legend(title="Design factor", loc="upper right", fontsize=8)
    ax.set_ylim(0, 105)

    # Annotate raw main-effect totals and R²
    for i, tier in enumerate(tiers):
        total = mat[i].sum()
        match = summary_df[summary_df["budget_tier"] == tier]
        r2 = match["r2"].values[0] if len(match) > 0 else 0
        ax.text(i, 102, f"Σ={total:.0f}%\nR²={r2:.2f}",
                ha="center", va="bottom", fontsize=7, color="gray")

    plt.tight_layout()
    return ax


def run_by_tier(df, metric=DEFAULT_METRIC, omit_strategies=None,
                n_tiers=4, n_estimators=500, random_state=42):
    """
    Run fANOVA within each budget tier.

    Usage:
        tier_results, summary_df = run_by_tier(fold_agg_df, metric="acc_mean",
                                                omit_strategies="cma_random")
    """
    df = filter_strategies(df, omit_strategies)
    if omit_strategies:
        print(f"Omitted strategies: {omit_strategies}")
        print(f"Remaining: {sorted(df['sampling_strategy'].unique())}")
        print()

    print("Computing fANOVA per budget tier ...")
    tier_results, summary_df = compute_fanova_by_tier(
        df, metric=metric, n_tiers=n_tiers,
        n_estimators=n_estimators, random_state=random_state,
    )
    print()

    print("Summary:")
    print(summary_df.to_string(index=False))
    print()

    print("Plot 1: fANOVA importance by tier (grouped bars) ...")
    plot_fanova_by_tier(summary_df)
    plt.show()

    print("Plot 2: Relative fANOVA share by tier (stacked bars) ...")
    plot_fanova_share_by_tier(summary_df)
    plt.show()

    return tier_results, summary_df