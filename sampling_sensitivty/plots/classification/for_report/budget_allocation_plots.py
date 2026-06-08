"""
Budget Allocation Visualization Module for ELA Classification Study.

Usage in notebook:
    import pandas as pd
    from budget_allocation_plots import *

    df = pd.read_pickle("subsample_results_table.pkl")

    # --- Viz 1: Pareto frontier ---
    fig = plot_pareto_frontier(df)

    # --- Viz 2: RF feature importances by budget tier ---
    fig = plot_feature_importances_by_budget_tier(df)

    # --- Viz 3: One-way PDPs with ICE ---
    fig = plot_pdp_ice(df)

    # --- Viz 4: Two-way PDPs ---
    fig = plot_pdp_2way(df)

    # --- Viz 5: CV–accuracy bridge ---
    fig = plot_cv_accuracy_bridge(df)

    # --- Viz 6: Optimal allocation recipe ---
    fig = plot_optimal_allocation(df)

    # --- Cross-dimension bridging (when multi-dim data available) ---
    # fig = plot_cross_dimension_pareto(df)
    # fig = plot_cross_dimension_importance_ranks(df)
    # fig = plot_cross_dimension_pdp_overlay(df)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.preprocessing import LabelEncoder
from adjustText import adjust_text

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGY_COLORS = {
    "cma_random":    "#e63946",
    "lhs":           "#457b9d",
    "ilhs":          "#2a9d8f",
    "lhs_random_cd": "#e9c46a",
    "sobol":         "#f4a261",
    "uniform":       "#6a4c93",
}

STRATEGY_LABELS = {
    "cma_random":    "CMA-Random",
    "lhs":           "LHS",
    "ilhs":          "iLHS",
    "lhs_random_cd": "LHS-Random-CD",
    "sobol":         "Sobol",
    "uniform":       "Uniform",
}

STRATEGY_ORDER = ["uniform", "lhs", "lhs_random_cd", "ilhs", "sobol", "cma_random"]

FACTOR_COLS = ["sampling_strategy", "sample_size_per_dim", "n_instances_train", "n_runs_train"]

BUDGET_TIER_LABELS = ["Low", "Medium", "High"]

EXCLUDED_STRATEGIES = {"lhs_random_cd"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pdp_grid_values(pdp_result):
    """Get grid values from partial_dependence result, compatible across sklearn versions.
    Older versions use 'grid_values', newer versions use 'values'."""
    if hasattr(pdp_result, "grid_values"):
        return pdp_result["grid_values"]
    return pdp_result["values"]

def _aggregate_folds(df, acc_col="accuracy_allruns"):
    """Average accuracy over CV folds to get one row per config."""
    df = df[~df["sampling_strategy"].isin(EXCLUDED_STRATEGIES)]
    group_cols = [
        "dimension", "sampling_strategy", "sample_size_per_dim",
        "n_instances_train", "n_runs_train", "n_feval_train",
    ]
    # Include CV columns (they're constant within group anyway)
    cv_cols = [c for c in df.columns if c.startswith("cv_")]
    agg_dict = {
        acc_col: "mean",
        "accuracy_median": "mean",
        "consistency": "mean",
    }
    # Ensure both accuracy columns are aggregated
    for extra in ["accuracy_allruns", "accuracy_median"]:
        if extra not in agg_dict:
            agg_dict[extra] = "mean"
    for c in cv_cols:
        agg_dict[c] = "first"

    return df.groupby(group_cols, as_index=False).agg(agg_dict)


def _assign_budget_tiers(df, n_tiers=3, col="n_feval_train"):
    """Assign budget tier labels based on quantile bins."""
    df = df.copy()
    df["budget_tier"] = pd.qcut(
        df[col].rank(method="first"), n_tiers,
        labels=BUDGET_TIER_LABELS[:n_tiers],
    )
    return df


def _get_strategy_color(s):
    return STRATEGY_COLORS.get(s, "#999999")


def _get_strategy_label(s):
    return STRATEGY_LABELS.get(s, s)


def _prepare_rf_data(agg_df):
    """Encode categoricals and prepare X, y for the RF regressor."""
    le_strategy = LabelEncoder()
    X = agg_df[FACTOR_COLS].copy()
    X["sampling_strategy"] = le_strategy.fit_transform(X["sampling_strategy"])
    y = agg_df["accuracy_allruns"].values
    feature_names = FACTOR_COLS.copy()
    return X, y, le_strategy, feature_names


def _fit_rf(X, y, random_state=42):
    """Fit a RandomForest regressor."""
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return rf


def _filter_dim(df, dim=None):
    """Filter to a single dimension. If None, use the only one or raise."""
    dims = df["dimension"].unique()
    if dim is None:
        if len(dims) == 1:
            dim = dims[0]
        else:
            raise ValueError(f"Multiple dimensions found: {dims}. Specify dim=.")
    return df[df["dimension"] == dim].copy(), dim


# ---------------------------------------------------------------------------
# Viz 1: Budget–Accuracy Efficiency Frontier
# ---------------------------------------------------------------------------

SAMPLE_SIZE_COLORS = {
    25:  "#264653",
    50:  "#2a9d8f",
    75:  "#e9c46a",
    100: "#e76f51",
}

INSTANCES_MARKERS = {
    1:  "o",
    2:  "s",
    3:  "^",
    5:  "D",
    7:  "v",
    10: "P",
    15: "X",
    20: "*",
}

STRATEGY_ABBREV = {
    "cma_random":    "CMA",
    "lhs":           "LHS",
    "ilhs":          "iLHS",
    "lhs_random_cd": "LCD",
    "sobol":         "Sob",
    "uniform":       "Uni",
}


def _best_per_budget(agg, acc_col):
    """Return the single best configuration at each unique budget level."""
    idx = agg.groupby("n_feval_train")[acc_col].idxmax()
    return agg.loc[idx].sort_values("n_feval_train").reset_index(drop=True)


def plot_budget_efficiency_frontier(df, dim=None, acc_col="accuracy_allruns",
                                     figsize=(12, 7), ax=None):
    """
    Best configuration at each budget level.
    - Color = sample_size_per_dim
    - Marker shape = n_instances_train
    - Annotation = (strategy abbreviation, n_runs)
    """
    df, dim = _filter_dim(df, dim)
    agg = _aggregate_folds(df, acc_col)
    best = _best_per_budget(agg, acc_col)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Connecting line
    # ax.plot(best["n_feval_train"], best[acc_col],
    #         color="black", linewidth=0.8, linestyle="-",
    #         alpha=0.2, zorder=1)

    # Plot each point
    annotations = []
    for _, row in best.iterrows():
        ss = int(row["sample_size_per_dim"])
        ni = int(row["n_instances_train"])
        nr = int(row["n_runs_train"])
        strat = row["sampling_strategy"]

        color = SAMPLE_SIZE_COLORS.get(ss, "#999999")
        marker = INSTANCES_MARKERS.get(ni, "o")
        abbrev = STRATEGY_ABBREV.get(strat, strat[:3])

        ax.scatter(
            row["n_feval_train"], row[acc_col],
            c=color, marker=marker, s=100,
            edgecolors="black", linewidths=0.5, zorder=3,
        )
        annotations.append((row["n_feval_train"], row[acc_col],
                             f"{abbrev}, r={nr}"))

    # Set log scale before text placement so adjustText uses the correct transform
    ax.set_xscale("log")

    # Use adjustText for automatic label placement with leader lines
    texts = []
    for x, y, label in annotations:
        texts.append(ax.text(x, y, label, fontsize=6, alpha=0.85))

    ax.set_xlabel("Training function evaluations")
    ax.set_ylabel(f"Classification accuracy ({acc_col})")
    ax.set_title(f"Best Configuration per Budget (D={dim})")
    ax.grid(True, alpha=0.15)

    # Run adjust_text after axis limits are set
    adjust_text(
        texts, ax=ax,
        arrowprops=dict(
            arrowstyle="-", color="gray", alpha=0.5, lw=0.5,
            shrinkA=5, shrinkB=5,
        ),
        force_points=(1.5, 1.5),
        force_text=(1.5, 1.5),
        expand_points=(2, 2),
        expand_text=(1.2, 1.2),
    )

    # --- Legend ---
    color_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=8, markeredgecolor="black", markeredgewidth=0.5,
               label=f"{ss}×D")
        for ss, c in sorted(SAMPLE_SIZE_COLORS.items())
    ]
    shape_handles = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor="gray",
               markersize=8, markeredgecolor="black", markeredgewidth=0.5,
               label=f"n_inst={ni}")
        for ni, m in sorted(INSTANCES_MARKERS.items())
    ]

    leg1 = ax.legend(
        handles=color_handles, title="Sample size", fontsize=7,
        title_fontsize=8, loc="lower right",
        bbox_to_anchor=(1.0, 0.0), framealpha=0.9,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=shape_handles, title="N instances", fontsize=7,
        title_fontsize=8, loc="lower right",
        bbox_to_anchor=(1.0, 0.35), framealpha=0.9,
    )

    fig.tight_layout()
    return fig


# Keep old name as alias for backward compatibility
plot_pareto_frontier = plot_budget_efficiency_frontier


# ---------------------------------------------------------------------------
# Viz 2: RF Feature Importances by Budget Tier
# ---------------------------------------------------------------------------

def plot_feature_importances_by_budget_tier(df, dim=None, n_tiers=3,
                                             figsize=(9, 5), ax=None):
    """
    Fit separate RFs per budget tier, show grouped bar chart of importances.
    """
    df, dim = _filter_dim(df, dim)
    agg = _aggregate_folds(df)
    agg = _assign_budget_tiers(agg, n_tiers)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    tier_labels = BUDGET_TIER_LABELS[:n_tiers]
    importances = {}

    for tier in tier_labels:
        sub = agg[agg["budget_tier"] == tier]
        X, y, _, feat_names = _prepare_rf_data(sub)
        rf = _fit_rf(X, y)
        importances[tier] = rf.feature_importances_

    x = np.arange(len(FACTOR_COLS))
    width = 0.8 / n_tiers
    tier_colors = ["#264653", "#2a9d8f", "#e9c46a"]

    for i, tier in enumerate(tier_labels):
        offset = (i - n_tiers / 2 + 0.5) * width
        bars = ax.bar(x + offset, importances[tier], width,
                       label=f"{tier} budget", color=tier_colors[i],
                       edgecolor="white", linewidth=0.5)

    display_names = ["Strategy", "Sample size", "N instances", "N runs"]
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10)
    ax.set_ylabel("Feature importance (MDI)")
    ax.set_title(f"Factor Importance by Budget Tier (D={dim})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15, axis="y")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viz 3: One-way PDPs with ICE Curves
# ---------------------------------------------------------------------------

def plot_pdp_ice(df, dim=None, acc_col="accuracy_allruns",
                 figsize=(16, 4)):
    """
    One-way PDP + ICE for each factor. Uses sklearn's PDP machinery.
    Strategy is shown as a bar chart (categorical), numeric factors as lines.
    """
    df, dim = _filter_dim(df, dim)
    agg = _aggregate_folds(df, acc_col)
    X, y, le_strategy, feat_names = _prepare_rf_data(agg)
    rf = _fit_rf(X, y)

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    numeric_factors = ["sample_size_per_dim", "n_instances_train", "n_runs_train"]
    display_names = {
        "sampling_strategy": "Strategy",
        "sample_size_per_dim": "Sample size (×D)",
        "n_instances_train": "N instances",
        "n_runs_train": "N runs",
    }

    # --- Strategy (categorical) as bar chart ---
    ax = axes[0]
    strat_encoded = sorted(X["sampling_strategy"].unique())
    strat_names = le_strategy.inverse_transform(strat_encoded)

    pdp_result = partial_dependence(
        rf, X, features=["sampling_strategy"],
        kind="both", grid_resolution=len(strat_encoded),
    )

    # ICE curves
    ice_values = pdp_result["individual"][0]  # (n_samples, n_grid_points)
    for i in range(ice_values.shape[0]):
        ax.plot(strat_encoded, ice_values[i], color="#aaaaaa",
                alpha=0.05, linewidth=0.5)

    # PDP bars
    pdp_values = pdp_result["average"][0]
    colors = [_get_strategy_color(s) for s in strat_names]
    ax.bar(strat_encoded, pdp_values, color=colors, edgecolor="white",
           linewidth=0.5, alpha=0.85, zorder=3)
    ax.set_xticks(strat_encoded)
    ax.set_xticklabels([_get_strategy_label(s) for s in strat_names],
                        rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Partial dependence")
    ax.set_title(display_names["sampling_strategy"])
    ax.grid(True, alpha=0.15, axis="y")

    # --- Numeric factors as line + ICE ---
    for idx, feat in enumerate(numeric_factors):
        ax = axes[idx + 1]
        feat_idx = feat_names.index(feat)

        pdp_result = partial_dependence(
            rf, X, features=[feat_idx],
            kind="both",
        )
        grid_vals = _pdp_grid_values(pdp_result)[0]
        ice_values = pdp_result["individual"][0]

        # ICE curves
        for i in range(ice_values.shape[0]):
            ax.plot(grid_vals, ice_values[i], color="#aaaaaa",
                    alpha=0.03, linewidth=0.5)

        # PDP line
        pdp_avg = pdp_result["average"][0]
        ax.plot(grid_vals, pdp_avg, color="#e63946", linewidth=2.5, zorder=5)

        ax.set_xlabel(display_names[feat])
        ax.set_title(display_names[feat])
        ax.grid(True, alpha=0.15)
        if idx == 0:
            pass  # ylabel already set on first axes
        ax.set_ylabel("")

    axes[0].set_ylabel("Partial dependence")
    fig.suptitle(f"Partial Dependence + ICE Curves (D={dim})", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viz 4: Two-way PDPs
# ---------------------------------------------------------------------------

def plot_pdp_2way(df, dim=None, acc_col="accuracy_allruns",
                  figsize=(15, 4)):
    """
    Two-way PDP heatmaps for key factor pairs.
    """
    df, dim = _filter_dim(df, dim)
    agg = _aggregate_folds(df, acc_col)
    X, y, le_strategy, feat_names = _prepare_rf_data(agg)
    rf = _fit_rf(X, y)

    pairs = [
        ("n_instances_train", "n_runs_train"),
        ("sample_size_per_dim", "n_instances_train"),
        ("sample_size_per_dim", "n_runs_train"),
    ]
    display_names = {
        "sampling_strategy": "Strategy",
        "sample_size_per_dim": "Sample size (×D)",
        "n_instances_train": "N instances",
        "n_runs_train": "N runs",
    }

    fig, axes = plt.subplots(1, len(pairs), figsize=figsize)

    for idx, (f1, f2) in enumerate(pairs):
        ax = axes[idx]
        i1 = feat_names.index(f1)
        i2 = feat_names.index(f2)

        pdp_result = partial_dependence(rf, X, features=[(i1, i2)])
        grid_0 = _pdp_grid_values(pdp_result)[0]
        grid_1 = _pdp_grid_values(pdp_result)[1]
        pdp_values = pdp_result["average"][0]

        im = ax.contourf(
            grid_1, grid_0, pdp_values,
            levels=20, cmap="RdYlGn",
        )
        ax.set_xlabel(display_names[f2])
        ax.set_ylabel(display_names[f1])
        ax.set_title(f"{display_names[f1]} × {display_names[f2]}")
        fig.colorbar(im, ax=ax, label="Partial dependence", shrink=0.8)

    fig.suptitle(f"Two-way Partial Dependence (D={dim})", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viz 5: CV–Accuracy Bridge
# ---------------------------------------------------------------------------

def plot_cv_accuracy_bridge(df, dim=None, cv_col="cv_instance_median_mean",
                             acc_col="accuracy_allruns", figsize=(8, 6),
                             ax=None):
    """
    Scatter: CV (per strategy×sample_size) vs best achievable accuracy
    at that (strategy, sample_size) across all (n_instances, n_runs).

    Also shows the accuracy at the maximum allocation (n_instances=20, n_runs=5)
    to separate the CV effect from the budget effect.
    """
    df, dim = _filter_dim(df, dim)
    agg = _aggregate_folds(df, acc_col)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Best accuracy per (strategy, sample_size) across allocations
    grouped_best = agg.groupby(["sampling_strategy", "sample_size_per_dim"]).agg(
        cv=pd.NamedAgg(column=cv_col, aggfunc="first"),
        acc_best=pd.NamedAgg(column=acc_col, aggfunc="max"),
        acc_mean=pd.NamedAgg(column=acc_col, aggfunc="mean"),
    ).reset_index()

    # Max allocation only (n_inst=20, n_runs=5)
    max_alloc = agg[
        (agg["n_instances_train"] == agg["n_instances_train"].max()) &
        (agg["n_runs_train"] == agg["n_runs_train"].max())
    ].copy()

    for strat in STRATEGY_ORDER:
        sub = grouped_best[grouped_best["sampling_strategy"] == strat]
        if sub.empty:
            continue
        color = _get_strategy_color(strat)
        label = _get_strategy_label(strat)

        # Best across all allocations (filled)
        ax.scatter(sub["cv"], sub["acc_best"], c=color, s=80,
                   label=f"{label} (best alloc)", edgecolors="white",
                   linewidths=0.5, zorder=5)

        # Mean across allocations (open)
        ax.scatter(sub["cv"], sub["acc_mean"], facecolors="none",
                   edgecolors=color, s=60, linewidths=1.5,
                   zorder=4)

        # Connect them
        for _, row in sub.iterrows():
            mean_row = sub[sub["sample_size_per_dim"] == row["sample_size_per_dim"]]
            if not mean_row.empty:
                ax.plot(
                    [row["cv"], row["cv"]],
                    [mean_row["acc_mean"].values[0], row["acc_best"]],
                    color=color, alpha=0.3, linewidth=1, zorder=3,
                )

    # Annotate sample sizes
    for _, row in grouped_best.iterrows():
        ax.annotate(
            f"{int(row['sample_size_per_dim'])}d",
            (row["cv"], row["acc_best"]),
            fontsize=6, alpha=0.7,
            xytext=(4, 4), textcoords="offset points",
        )

    ax.set_xlabel(f"Feature CV ({cv_col})")
    ax.set_ylabel(f"Classification accuracy ({acc_col})")
    ax.set_title(f"Feature Stability vs. Downstream Accuracy (D={dim})")

    # Legend
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=8, label="Best allocation"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="gray", markersize=8, markeredgewidth=1.5,
               label="Mean across allocations"),
    ]
    strat_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=_get_strategy_color(s), markersize=8,
               label=_get_strategy_label(s))
        for s in STRATEGY_ORDER if s in grouped_best["sampling_strategy"].values
    ]
    ax.legend(handles=strat_handles + handles, fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viz 6: Optimal Allocation Recipe
# ---------------------------------------------------------------------------

def plot_optimal_allocation(df, dim=None, n_tiers=3, acc_col="accuracy_allruns",
                             figsize=(10, 5)):
    """
    Heatmap: for each budget tier, show the empirical best setting per factor
    and the RF-predicted optimal.
    """
    df, dim = _filter_dim(df, dim)
    agg = _aggregate_folds(df, acc_col)
    agg = _assign_budget_tiers(agg, n_tiers)

    tier_labels = BUDGET_TIER_LABELS[:n_tiers]

    # Empirical best per tier
    records = []
    for tier in tier_labels:
        sub = agg[agg["budget_tier"] == tier]
        best_idx = sub[acc_col].idxmax()
        best_row = sub.loc[best_idx]
        records.append({
            "Budget tier": tier,
            "Strategy": _get_strategy_label(best_row["sampling_strategy"]),
            "Sample size (×D)": int(best_row["sample_size_per_dim"]),
            "N instances": int(best_row["n_instances_train"]),
            "N runs": int(best_row["n_runs_train"]),
            "Best accuracy": f"{best_row[acc_col]:.3f}",
            "Budget": int(best_row["n_feval_train"]),
        })
    recipe_df = pd.DataFrame(records)

    # Also find the most frequent best config per tier (top-3)
    top_records = []
    for tier in tier_labels:
        sub = agg[agg["budget_tier"] == tier].nlargest(10, acc_col)
        top_records.append({
            "Budget tier": tier,
            "Top strategies": ", ".join(
                sub["sampling_strategy"].map(_get_strategy_label).value_counts().head(3).index
            ),
            "Top sample sizes": ", ".join(
                sub["sample_size_per_dim"].astype(str).unique()[:3]
            ),
            "Top n_instances": ", ".join(
                sub["n_instances_train"].astype(str).value_counts().head(3).index
            ),
            "Top n_runs": ", ".join(
                sub["n_runs_train"].astype(str).value_counts().head(3).index
            ),
        })
    top_df = pd.DataFrame(top_records)

    # Plot as a table
    fig, axes = plt.subplots(2, 1, figsize=figsize,
                              gridspec_kw={"height_ratios": [1, 1]})

    # Table 1: Single best
    ax = axes[0]
    ax.axis("off")
    table1 = ax.table(
        cellText=recipe_df.values,
        colLabels=recipe_df.columns,
        loc="center",
        cellLoc="center",
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1, 1.5)
    ax.set_title(f"Best Configuration per Budget Tier (D={dim})", fontsize=12,
                  pad=20)

    # Table 2: Top-10 patterns
    ax = axes[1]
    ax.axis("off")
    table2 = ax.table(
        cellText=top_df.values,
        colLabels=top_df.columns,
        loc="center",
        cellLoc="center",
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.5)
    ax.set_title("Most Frequent Factors in Top-10 Configs per Tier", fontsize=12,
                  pad=20)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cross-dimension helpers (for when multi-dim data is available)
# ---------------------------------------------------------------------------

def plot_cross_dimension_pareto(df, acc_col="accuracy_allruns", figsize=(12, 5)):
    """
    Overlay Pareto frontiers from different dimensions on normalized budget axis.
    Budget is normalized as budget / (D * min_budget_per_dim).
    """
    dims = sorted(df["dimension"].unique())
    if len(dims) < 2:
        print(f"Only dimension(s) {dims} found. Need ≥2 for cross-dimension plot.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: raw budget
    ax = axes[0]
    for dim in dims:
        sub = _aggregate_folds(df[df["dimension"] == dim], acc_col)
        sorted_sub = sub.sort_values("n_feval_train")
        pareto_x, pareto_y = [], []
        best = -np.inf
        for _, row in sorted_sub.iterrows():
            if row[acc_col] > best:
                best = row[acc_col]
                pareto_x.append(row["n_feval_train"])
                pareto_y.append(row[acc_col])
        ax.step(pareto_x, pareto_y, where="post", linewidth=2,
                label=f"D={dim}")
    ax.set_xscale("log")
    ax.set_xlabel("Training function evaluations")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pareto Frontiers (raw budget)")
    ax.legend()
    ax.grid(True, alpha=0.15)

    # Right: normalized budget (budget / D)
    ax = axes[1]
    for dim in dims:
        sub = _aggregate_folds(df[df["dimension"] == dim], acc_col)
        sub["budget_norm"] = sub["n_feval_train"] / dim
        sorted_sub = sub.sort_values("budget_norm")
        pareto_x, pareto_y = [], []
        best = -np.inf
        for _, row in sorted_sub.iterrows():
            if row[acc_col] > best:
                best = row[acc_col]
                pareto_x.append(row["budget_norm"])
                pareto_y.append(row[acc_col])
        ax.step(pareto_x, pareto_y, where="post", linewidth=2,
                label=f"D={dim}")
    ax.set_xscale("log")
    ax.set_xlabel("Training FEs / D")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pareto Frontiers (budget normalized by D)")
    ax.legend()
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    return fig


def plot_cross_dimension_importance_ranks(df, figsize=(8, 4)):
    """
    Heatmap: rows = factors, columns = dimensions.
    Cell = feature importance rank (1 = most important).
    """
    dims = sorted(df["dimension"].unique())
    if len(dims) < 2:
        print(f"Only dimension(s) {dims} found. Need ≥2 for cross-dimension plot.")
        return None

    rank_data = {}
    for dim in dims:
        sub = _aggregate_folds(df[df["dimension"] == dim])
        X, y, _, feat_names = _prepare_rf_data(sub)
        rf = _fit_rf(X, y)
        imp = rf.feature_importances_
        # Rank: 1 = highest importance
        ranks = len(imp) - np.argsort(np.argsort(imp))
        rank_data[dim] = ranks

    rank_df = pd.DataFrame(rank_data, index=["Strategy", "Sample size", "N instances", "N runs"])
    rank_df.columns = [f"D={d}" for d in dims]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(rank_df.values, cmap="RdYlGn_r", aspect="auto",
                    vmin=1, vmax=len(FACTOR_COLS))
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(rank_df.columns)
    ax.set_yticks(range(len(rank_df)))
    ax.set_yticklabels(rank_df.index)

    for i in range(rank_df.shape[0]):
        for j in range(rank_df.shape[1]):
            ax.text(j, i, str(rank_df.values[i, j]),
                    ha="center", va="center", fontsize=14, fontweight="bold")

    ax.set_title("Factor Importance Rank by Dimension (1 = most important)")
    fig.colorbar(im, ax=ax, label="Rank", shrink=0.8)
    fig.tight_layout()
    return fig


def plot_cross_dimension_pdp_overlay(df, acc_col="accuracy_allruns",
                                      figsize=(15, 4)):
    """
    Overlay centered PDP curves from different dimensions.
    Each curve is shifted to mean=0 so the shapes can be compared.
    """
    dims = sorted(df["dimension"].unique())
    if len(dims) < 2:
        print(f"Only dimension(s) {dims} found. Need ≥2 for cross-dimension plot.")
        return None

    numeric_factors = ["sample_size_per_dim", "n_instances_train", "n_runs_train"]
    display_names = {
        "sample_size_per_dim": "Sample size (×D)",
        "n_instances_train": "N instances",
        "n_runs_train": "N runs",
    }
    dim_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(dims)))

    fig, axes = plt.subplots(1, len(numeric_factors), figsize=figsize)

    for idx, feat in enumerate(numeric_factors):
        ax = axes[idx]
        for di, dim in enumerate(dims):
            sub = _aggregate_folds(df[df["dimension"] == dim], acc_col)
            X, y, _, feat_names = _prepare_rf_data(sub)
            rf = _fit_rf(X, y)
            feat_idx = feat_names.index(feat)

            pdp_result = partial_dependence(rf, X, features=[feat_idx])
            grid_vals = _pdp_grid_values(pdp_result)[0]
            pdp_avg = pdp_result["average"][0]

            # Center
            pdp_centered = pdp_avg - pdp_avg.mean()
            ax.plot(grid_vals, pdp_centered, color=dim_colors[di],
                    linewidth=2, label=f"D={dim}")

        ax.set_xlabel(display_names[feat])
        ax.set_ylabel("Centered partial dependence" if idx == 0 else "")
        ax.set_title(display_names[feat])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    fig.suptitle("PDP Shape Comparison Across Dimensions (centered)", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def plot_cross_dimension_accuracy_ceiling(df, acc_col="accuracy_allruns",
                                           figsize=(10, 6)):
    """
    Pareto frontiers with accuracy expressed as fraction of per-dimension ceiling.
    Ceiling = accuracy at max config (n_instances=max, n_runs=max, best strategy+size).
    """
    dims = sorted(df["dimension"].unique())
    if len(dims) < 2:
        print(f"Only dimension(s) {dims} found. Need ≥2 for cross-dimension plot.")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    dim_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(dims)))

    for di, dim in enumerate(dims):
        sub = _aggregate_folds(df[df["dimension"] == dim], acc_col)
        ceiling = sub[acc_col].max()
        sub["acc_frac"] = sub[acc_col] / ceiling
        sub["budget_norm"] = sub["n_feval_train"] / dim

        sorted_sub = sub.sort_values("budget_norm")
        pareto_x, pareto_y = [], []
        best = -np.inf
        for _, row in sorted_sub.iterrows():
            if row["acc_frac"] > best:
                best = row["acc_frac"]
                pareto_x.append(row["budget_norm"])
                pareto_y.append(row["acc_frac"])

        ax.step(pareto_x, pareto_y, where="post", linewidth=2,
                color=dim_colors[di], label=f"D={dim} (ceiling={ceiling:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("Training FEs / D")
    ax.set_ylabel("Accuracy / ceiling")
    ax.set_title("Budget Efficiency: Fraction of Ceiling Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.15)
    ax.axhline(0.9, color="gray", linestyle=":", alpha=0.5, label="90% ceiling")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viz 7: Best Configuration per Budget for a Given Strategy
# ---------------------------------------------------------------------------

def plot_best_config_by_strategy(df, strategy, dim=None,
                                  acc_col="accuracy_allruns",
                                  figsize=(12, 7), ax=None):
    """
    Best configuration at each budget level for a single sampling strategy.
    - Color = sample_size_per_dim
    - Marker shape = n_instances_train
    - Annotation = n_runs
    """
    df, dim = _filter_dim(df, dim)
    agg = _aggregate_folds(df, acc_col)
    agg = agg[agg["sampling_strategy"] == strategy]

    if agg.empty:
        raise ValueError(f"No data for strategy '{strategy}'. "
                         f"Available: {sorted(df['sampling_strategy'].unique())}")

    best = _best_per_budget(agg, acc_col)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Connecting line
    ax.plot(best["n_feval_train"], best[acc_col],
            color="black", linewidth=0.8, linestyle="-",
            alpha=0.2, zorder=1)

    # Plot each point
    annotations = []
    for _, row in best.iterrows():
        ss = int(row["sample_size_per_dim"])
        ni = int(row["n_instances_train"])
        nr = int(row["n_runs_train"])

        color = SAMPLE_SIZE_COLORS.get(ss, "#999999")
        marker = INSTANCES_MARKERS.get(ni, "o")

        ax.scatter(
            row["n_feval_train"], row[acc_col],
            c=color, marker=marker, s=100,
            edgecolors="black", linewidths=0.5, zorder=3,
        )
        annotations.append((row["n_feval_train"], row[acc_col],
                             f"r={nr}"))

    # Set log scale before adjustText
    ax.set_xscale("log")

    # Use adjustText for label placement
    texts = []
    for x, y, label in annotations:
        texts.append(ax.text(x, y, label, fontsize=7, alpha=0.85))

    strat_label = _get_strategy_label(strategy)
    ax.set_xlabel("Training function evaluations")
    ax.set_ylabel(f"Classification accuracy ({acc_col})")
    ax.set_title(f"Best Configuration per Budget — {strat_label} (D={dim})")
    ax.grid(True, alpha=0.15)

    adjust_text(
        texts, ax=ax,
        arrowprops=dict(
            arrowstyle="-", color="gray", alpha=0.5, lw=0.5,
            shrinkA=5, shrinkB=5,
        ),
        force_points=(1.5, 1.5),
        force_text=(1.5, 1.5),
        expand_points=(2, 2),
        expand_text=(1.2, 1.2),
    )

    # Legend
    color_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=8, markeredgecolor="black", markeredgewidth=0.5,
               label=f"{ss}×D")
        for ss, c in sorted(SAMPLE_SIZE_COLORS.items())
    ]
    shape_handles = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor="gray",
               markersize=8, markeredgecolor="black", markeredgewidth=0.5,
               label=f"n_inst={ni}")
        for ni, m in sorted(INSTANCES_MARKERS.items())
    ]

    leg1 = ax.legend(
        handles=color_handles, title="Sample size", fontsize=7,
        title_fontsize=8, loc="lower right",
        bbox_to_anchor=(1.0, 0.0), framealpha=0.9,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=shape_handles, title="N instances", fontsize=7,
        title_fontsize=8, loc="lower right",
        bbox_to_anchor=(1.0, 0.35), framealpha=0.9,
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viz 8: N_instances vs Sample Size Tradeoff Heatmap (fixed strategy, n_runs)
# ---------------------------------------------------------------------------

def plot_instances_vs_samplesize(df, strategy="ilhs", n_runs=1, dim=None,
                                  acc_col="accuracy_allruns",
                                  figsize=(8, 6), ax=None):
    """
    Heatmap of accuracy for (sample_size × n_instances) at fixed strategy and n_runs.
    Budget isolines overlaid to show equal-cost configurations.
    """
    df, dim = _filter_dim(df, dim)
    agg = _aggregate_folds(df, acc_col)
    sub = agg[(agg["sampling_strategy"] == strategy) &
              (agg["n_runs_train"] == n_runs)]

    if sub.empty:
        raise ValueError(f"No data for strategy='{strategy}', n_runs={n_runs}")

    pivot = sub.pivot_table(
        index="n_instances_train", columns="sample_size_per_dim",
        values=acc_col, aggfunc="mean",
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(
        pivot.values, cmap="RdYlGn", aspect="auto",
        origin="lower",
        vmin=sub[acc_col].min(), vmax=sub[acc_col].max(),
    )

    # Axis labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c)}×D" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([int(i) for i in pivot.index])
    ax.set_xlabel("Sample size per dimension")
    ax.set_ylabel("N instances (training)")

    # Annotate cells with accuracy values
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                # Compute budget for this cell
                ss = int(pivot.columns[j])
                ni = int(pivot.index[i])
                budget = ss * dim * 24 * ni * n_runs
                ax.text(j, i, f"{val:.3f}\n({budget:,})",
                        ha="center", va="center", fontsize=7,
                        color="white" if val < pivot.values.mean() else "black")

    strat_label = _get_strategy_label(strategy)
    ax.set_title(f"N Instances × Sample Size — {strat_label}, r={n_runs} (D={dim})")
    fig.colorbar(im, ax=ax, label=f"Accuracy ({acc_col})", shrink=0.8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viz 9: Marginal Return Curves — N_instances vs Sample Size
# ---------------------------------------------------------------------------

def plot_marginal_returns(df, strategy="ilhs", n_runs=1, dim=None,
                           acc_col="accuracy_allruns",
                           figsize=(14, 5)):
    """
    Two panels showing marginal returns for a fixed strategy and n_runs:
      Left:  accuracy vs n_instances, one line per sample_size
      Right: accuracy vs sample_size, one line per n_instances
    """
    df, dim = _filter_dim(df, dim)
    agg = _aggregate_folds(df, acc_col)
    sub = agg[(agg["sampling_strategy"] == strategy) &
              (agg["n_runs_train"] == n_runs)]

    if sub.empty:
        raise ValueError(f"No data for strategy='{strategy}', n_runs={n_runs}")

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    strat_label = _get_strategy_label(strategy)

    # Left: accuracy vs n_instances, grouped by sample_size
    ax = axes[0]
    for ss in sorted(sub["sample_size_per_dim"].unique()):
        s = sub[sub["sample_size_per_dim"] == ss].sort_values("n_instances_train")
        color = SAMPLE_SIZE_COLORS.get(int(ss), "#999999")
        ax.plot(s["n_instances_train"], s[acc_col],
                marker="o", color=color, linewidth=2, markersize=6,
                label=f"{int(ss)}×D")
    ax.set_xlabel("N instances (training)")
    ax.set_ylabel(f"Accuracy ({acc_col})")
    ax.set_title(f"Marginal return of instances")
    ax.legend(title="Sample size", fontsize=8, title_fontsize=9)
    ax.grid(True, alpha=0.15)

    # Right: accuracy vs sample_size, grouped by n_instances
    ax = axes[1]
    n_inst_values = sorted(sub["n_instances_train"].unique())
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(n_inst_values)))
    for idx, ni in enumerate(n_inst_values):
        s = sub[sub["n_instances_train"] == ni].sort_values("sample_size_per_dim")
        ax.plot(s["sample_size_per_dim"], s[acc_col],
                marker="o", color=cmap[idx], linewidth=2, markersize=6,
                label=f"n={int(ni)}")
    ax.set_xlabel("Sample size (×D)")
    ax.set_ylabel(f"Accuracy ({acc_col})")
    ax.set_title(f"Marginal return of sample size")
    ax.legend(title="N instances", fontsize=7, title_fontsize=9,
              ncol=2)
    ax.grid(True, alpha=0.15)

    fig.suptitle(f"Marginal Returns — {strat_label}, r={n_runs} (D={dim})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viz 10: Minimum Budget to Reach Accuracy Thresholds
# ---------------------------------------------------------------------------

def compute_minimum_budget_table(df, dim=None, acc_col="accuracy_allruns",
                                  thresholds=(0.80, 0.85, 0.90),
                                  require_robust=False):
    """
    For each strategy, find the cheapest configuration that reaches each
    accuracy threshold.

    If require_robust=True, requires mean - 1*std across folds to exceed
    the threshold (i.e. the threshold is reliably reached).

    Returns a DataFrame with columns:
        strategy, threshold, sample_size, n_instances, n_runs, budget, accuracy
    """
    df_orig, dim = _filter_dim(df, dim)
    df_orig = df_orig[~df_orig["sampling_strategy"].isin(EXCLUDED_STRATEGIES)]

    if require_robust:
        # Compute mean and std per config across folds
        group_cols = [
            "dimension", "sampling_strategy", "sample_size_per_dim",
            "n_instances_train", "n_runs_train", "n_feval_train",
        ]
        stats = df_orig.groupby(group_cols, as_index=False).agg(
            acc_mean=pd.NamedAgg(column=acc_col, aggfunc="mean"),
            acc_std=pd.NamedAgg(column=acc_col, aggfunc="std"),
        )
        stats["acc_robust"] = stats["acc_mean"] - stats["acc_std"]
        agg = stats.rename(columns={"acc_mean": acc_col})
    else:
        agg = _aggregate_folds(df_orig, acc_col)
        agg["acc_robust"] = agg[acc_col]  # just use mean

    acc_check = "acc_robust" if require_robust else acc_col

    records = []
    for strat in STRATEGY_ORDER:
        sub = agg[agg["sampling_strategy"] == strat].sort_values("n_feval_train")
        for thr in thresholds:
            hits = sub[sub[acc_check] >= thr]
            if hits.empty:
                records.append({
                    "Strategy": _get_strategy_label(strat),
                    "Threshold": f"{thr:.0%}",
                    "Sample size": "—",
                    "N instances": "—",
                    "N runs": "—",
                    "Budget": "—",
                    "Accuracy": "—",
                })
            else:
                # Cheapest config that meets threshold
                best = hits.iloc[0]
                records.append({
                    "Strategy": _get_strategy_label(strat),
                    "Threshold": f"{thr:.0%}",
                    "Sample size": f"{int(best['sample_size_per_dim'])}×D",
                    "N instances": int(best["n_instances_train"]),
                    "N runs": int(best["n_runs_train"]),
                    "Budget": f"{int(best['n_feval_train']):,}",
                    "Accuracy": f"{best[acc_col]:.3f}",
                })

    return pd.DataFrame(records)


def plot_minimum_budget_thresholds(df, dim=None, acc_col="accuracy_allruns",
                                    thresholds=(0.80, 0.85, 0.90),
                                    figsize=(10, 5)):
    """
    Bar chart: for each strategy and threshold, show the minimum budget needed.
    Immediately visualizes the 'cost of instability'.
    """
    df, dim = _filter_dim(df, dim)
    agg = _aggregate_folds(df, acc_col)

    fig, ax = plt.subplots(figsize=figsize)

    strategies = [s for s in STRATEGY_ORDER if s in agg["sampling_strategy"].unique()]
    n_strat = len(strategies)
    n_thr = len(thresholds)
    x = np.arange(n_strat)
    width = 0.8 / n_thr
    threshold_colors = ["#2a9d8f", "#e9c46a", "#e76f51"]

    for i, thr in enumerate(thresholds):
        budgets = []
        for strat in strategies:
            sub = agg[agg["sampling_strategy"] == strat].sort_values("n_feval_train")
            hits = sub[sub[acc_col] >= thr]
            if hits.empty:
                budgets.append(0)
            else:
                budgets.append(hits.iloc[0]["n_feval_train"])

        offset = (i - n_thr / 2 + 0.5) * width
        bars = ax.bar(x + offset, budgets, width,
                       label=f"{thr:.0%}", color=threshold_colors[i],
                       edgecolor="white", linewidth=0.5)

        # Annotate budget values on bars
        for j, (bar, bud) in enumerate(zip(bars, budgets)):
            if bud > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{int(bud):,}", ha="center", va="bottom",
                        fontsize=6, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([_get_strategy_label(s) for s in strategies], fontsize=10)
    ax.set_ylabel("Minimum training function evaluations")
    ax.set_title(f"Minimum Budget to Reach Accuracy Thresholds (D={dim})")
    ax.legend(title="Threshold", fontsize=9, title_fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.15, axis="y")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viz 11: Minimum Viable Configuration — Finding the Knee
# ---------------------------------------------------------------------------

def plot_accuracy_vs_budget_knee(df, dim=None, acc_col="accuracy_allruns",
                                  n_runs=1, figsize=(12, 5)):
    """
    For each strategy at fixed n_runs, plot accuracy vs budget along the
    optimal allocation path (smallest sample size, increasing n_instances).
    Shows where diminishing returns set in.

    Left panel: accuracy vs budget curves per strategy
    Right panel: marginal accuracy gain per additional 1000 FEs
    """
    df, dim = _filter_dim(df, dim)
    agg = _aggregate_folds(df, acc_col)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    strategies = [s for s in STRATEGY_ORDER if s in agg["sampling_strategy"].unique()]

    min_ss = agg["sample_size_per_dim"].min()

    # Left: accuracy curves along optimal path
    ax = axes[0]
    for strat in strategies:
        sub = agg[(agg["sampling_strategy"] == strat) &
                   (agg["n_runs_train"] == n_runs) &
                   (agg["sample_size_per_dim"] == min_ss)]
        sub = sub.sort_values("n_instances_train")
        if sub.empty:
            continue
        color = _get_strategy_color(strat)
        ax.plot(sub["n_feval_train"], sub[acc_col],
                marker="o", color=color, linewidth=2, markersize=6,
                label=_get_strategy_label(strat))

        # Annotate n_instances on points
        for _, row in sub.iterrows():
            ax.annotate(f"n={int(row['n_instances_train'])}",
                        (row["n_feval_train"], row[acc_col]),
                        fontsize=5, alpha=0.6,
                        xytext=(3, 4), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Training function evaluations")
    ax.set_ylabel(f"Accuracy ({acc_col})")
    ax.set_title(f"Optimal path: {int(min_ss)}×D, r={n_runs}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    # Right: marginal gain (delta accuracy / delta budget)
    ax = axes[1]
    for strat in strategies:
        sub = agg[(agg["sampling_strategy"] == strat) &
                   (agg["n_runs_train"] == n_runs) &
                   (agg["sample_size_per_dim"] == min_ss)]
        sub = sub.sort_values("n_instances_train")
        if len(sub) < 2:
            continue

        budgets = sub["n_feval_train"].values
        accs = sub[acc_col].values
        delta_acc = np.diff(accs)
        delta_budget = np.diff(budgets)
        marginal = delta_acc / (delta_budget / 1000)  # per 1000 FEs
        mid_budget = (budgets[:-1] + budgets[1:]) / 2

        color = _get_strategy_color(strat)
        ax.plot(mid_budget, marginal, marker="o", color=color,
                linewidth=2, markersize=5,
                label=_get_strategy_label(strat))

    ax.set_xscale("log")
    ax.set_xlabel("Training function evaluations")
    ax.set_ylabel("Marginal accuracy gain per 1000 FEs")
    ax.set_title("Diminishing returns")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    fig.suptitle(f"Minimum Viable Configuration Analysis (D={dim})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def compute_knee_table(df, dim=None, acc_col="accuracy_allruns",
                        n_runs=1, marginal_threshold=0.005):
    """
    Find the 'knee point' for each strategy: the n_instances value beyond
    which the marginal accuracy gain per additional instance drops below
    the threshold.

    marginal_threshold: minimum accuracy gain per additional instance
    to be considered 'worth it' (default 0.005 = 0.5 percentage points).

    Returns a DataFrame summarizing the knee per strategy.
    """
    df, dim = _filter_dim(df, dim)
    agg = _aggregate_folds(df, acc_col)
    min_ss = agg["sample_size_per_dim"].min()

    records = []
    for strat in STRATEGY_ORDER:
        sub = agg[(agg["sampling_strategy"] == strat) &
                   (agg["n_runs_train"] == n_runs) &
                   (agg["sample_size_per_dim"] == min_ss)]
        sub = sub.sort_values("n_instances_train")
        if len(sub) < 2:
            continue

        instances = sub["n_instances_train"].values
        accs = sub[acc_col].values
        budgets = sub["n_feval_train"].values

        # Find first instance count where marginal gain drops below threshold
        knee_idx = None
        for i in range(1, len(instances)):
            marginal = accs[i] - accs[i - 1]
            if marginal < marginal_threshold:
                knee_idx = i - 1
                break

        if knee_idx is None:
            knee_idx = len(instances) - 1  # never drops below threshold

        records.append({
            "Strategy": _get_strategy_label(strat),
            "Knee n_instances": int(instances[knee_idx]),
            "Accuracy at knee": f"{accs[knee_idx]:.3f}",
            "Budget at knee": f"{int(budgets[knee_idx]):,}",
            "Max accuracy (n=20)": f"{accs[-1]:.3f}",
            "Max budget (n=20)": f"{int(budgets[-1]):,}",
            "Accuracy gap": f"{accs[-1] - accs[knee_idx]:.3f}",
            "Budget ratio": f"{budgets[-1] / budgets[knee_idx]:.1f}×",
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Viz 12: RF Feature Importance with CV Metrics
# ---------------------------------------------------------------------------

def plot_cv_importance_analysis(df, dim=None, acc_col="accuracy_allruns",
                                cv_cols=("cv_instance_median_mean", "cv_function_median_mean"),
                                figsize=(14, 5)):
    """
    Two RF models to assess whether CV metrics predict accuracy:

    Left panel (Model A): All 4 allocation factors + 2 CV metrics.
        Shows whether CV adds predictive power alongside strategy.

    Right panel (Model B): 3 numeric factors + 2 CV metrics, NO strategy.
        Shows whether CV is a useful proxy for strategy identity.

    If CV importance is high in Model B but low in Model A, it means CV
    and strategy carry the same information (CV proxies for strategy).
    If CV importance is high in both, CV predicts accuracy beyond strategy.
    """
    df, dim = _filter_dim(df, dim)
    agg = _aggregate_folds(df, acc_col)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Model A: all factors + CV ---
    cols_a = FACTOR_COLS + list(cv_cols)
    X_a = agg[cols_a].copy()
    le_a = LabelEncoder()
    X_a["sampling_strategy"] = le_a.fit_transform(X_a["sampling_strategy"])
    y = agg[acc_col].values

    rf_a = _fit_rf(X_a, y)

    display_names_a = ["Strategy", "Sample size", "N instances", "N runs"] + \
                      [c.replace("cv_", "CV: ").replace("_median_mean", "") for c in cv_cols]

    ax = axes[0]
    imp_a = rf_a.feature_importances_
    colors_a = ["#264653"] * 4 + ["#e76f51"] * len(cv_cols)
    bars = ax.barh(range(len(imp_a)), imp_a, color=colors_a,
                    edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(display_names_a)))
    ax.set_yticklabels(display_names_a, fontsize=9)
    ax.set_xlabel("Feature importance (MDI)")
    ax.set_title("Model A: All factors + CV")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.15, axis="x")

    # Annotate values
    for i, v in enumerate(imp_a):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

    # --- Model B: numeric factors + CV only (no strategy) ---
    numeric_factors = ["sample_size_per_dim", "n_instances_train", "n_runs_train"]
    cols_b = numeric_factors + list(cv_cols)
    X_b = agg[cols_b].copy()

    rf_b = _fit_rf(X_b, y)

    display_names_b = ["Sample size", "N instances", "N runs"] + \
                      [c.replace("cv_", "CV: ").replace("_median_mean", "") for c in cv_cols]

    ax = axes[1]
    imp_b = rf_b.feature_importances_
    colors_b = ["#264653"] * 3 + ["#e76f51"] * len(cv_cols)
    bars = ax.barh(range(len(imp_b)), imp_b, color=colors_b,
                    edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(display_names_b)))
    ax.set_yticklabels(display_names_b, fontsize=9)
    ax.set_xlabel("Feature importance (MDI)")
    ax.set_title("Model B: Numeric factors + CV (no strategy)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.15, axis="x")

    for i, v in enumerate(imp_b):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#264653", label="Allocation factors"),
        Patch(facecolor="#e76f51", label="CV metrics"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Does Feature Stability (CV) Predict Accuracy? (D={dim})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def compute_cv_importance_table(df, acc_col="accuracy_allruns",
                                 cv_cols=("cv_instance_median_mean", "cv_function_median_mean")):
    """
    Compute RF feature importances for Models A and B across all dimensions.
    Returns a DataFrame for easy comparison.
    """
    records = []

    for dim in sorted(df["dimension"].unique()):
        sub = df[df["dimension"] == dim]
        sub = sub[~sub["sampling_strategy"].isin(EXCLUDED_STRATEGIES)]
        agg = _aggregate_folds(sub, acc_col)
        y = agg[acc_col].values

        # Model A
        cols_a = FACTOR_COLS + list(cv_cols)
        X_a = agg[cols_a].copy()
        le = LabelEncoder()
        X_a["sampling_strategy"] = le.fit_transform(X_a["sampling_strategy"])
        rf_a = _fit_rf(X_a, y)

        names_a = ["Strategy", "Sample size", "N instances", "N runs"] + list(cv_cols)
        for name, imp in zip(names_a, rf_a.feature_importances_):
            records.append({
                "Dimension": dim,
                "Model": "A (with strategy)",
                "Feature": name,
                "Importance": imp,
            })

        # Model B
        numeric_factors = ["sample_size_per_dim", "n_instances_train", "n_runs_train"]
        cols_b = numeric_factors + list(cv_cols)
        X_b = agg[cols_b].copy()
        rf_b = _fit_rf(X_b, y)

        names_b = ["Sample size", "N instances", "N runs"] + list(cv_cols)
        for name, imp in zip(names_b, rf_b.feature_importances_):
            records.append({
                "Dimension": dim,
                "Model": "B (no strategy)",
                "Feature": name,
                "Importance": imp,
            })

    result = pd.DataFrame(records)
    return result