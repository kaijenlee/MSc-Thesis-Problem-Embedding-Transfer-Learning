"""
fanova_utils.py — Stratified hyperparameter importance analysis via
random-forest permutation importance (sklearn-based fANOVA surrogate).

Usage
-----
    from fanova_utils import run_stratified_fanova, plot_importance_heatmap, plot_importance_bars

    results, summary = run_stratified_fanova(
        df,
        hyperparams=["sampling_strategy", "sample_size_per_dim",
                      "n_instances_train", "n_runs_train"],
        target="accuracy_median",
        budget_col="n_feval_train",
        n_budget_tiers=4,
    )
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _encode_features(
    df: pd.DataFrame,
    hyperparams: list[str],
) -> tuple[np.ndarray, list[str], dict[str, LabelEncoder]]:
    """Encode categoricals to integers; pass numerics through."""
    X = df[hyperparams].copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in hyperparams:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        else:
            X[col] = X[col].astype(float)
    return X.values.astype(float), hyperparams, encoders


def _compute_importance(
    X: np.ndarray,
    y: np.ndarray,
    hp_names: list[str],
    n_estimators: int = 128,
    n_repeats: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit RF and compute permutation importance."""
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
    )
    rf.fit(X, y)

    perm = permutation_importance(
        rf, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="r2",
    )

    records = []
    for i, name in enumerate(hp_names):
        records.append({
            "hyperparameter": name,
            "importance_mean": perm.importances_mean[i],
            "importance_std": perm.importances_std[i],
            "importance_raw": perm.importances[i],  # full distribution
        })

    result = pd.DataFrame(records).sort_values(
        "importance_mean", ascending=False
    ).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    result["rf_oob_r2"] = rf.oob_score_ if hasattr(rf, "oob_score_") else np.nan

    return result


def _compute_pairwise_importance(
    X: np.ndarray,
    y: np.ndarray,
    hp_names: list[str],
    n_estimators: int = 128,
    n_repeats: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute pairwise interaction importance by jointly permuting pairs."""
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
    )
    rf.fit(X, y)

    # Get individual importances for computing interaction strength
    single_perm = permutation_importance(
        rf, X, y, n_repeats=n_repeats, random_state=random_state, scoring="r2"
    )

    records = []
    for i, j in combinations(range(len(hp_names)), 2):
        # Jointly permute both features
        rng = np.random.RandomState(random_state)
        baseline_score = rf.score(X, y)
        joint_scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            perm_idx = rng.permutation(len(X))
            X_perm[:, i] = X_perm[perm_idx, i]
            X_perm[:, j] = X_perm[perm_idx, j]
            joint_scores.append(rf.score(X_perm, y))

        joint_drop = baseline_score - np.array(joint_scores)
        individual_sum = single_perm.importances_mean[i] + single_perm.importances_mean[j]
        interaction = joint_drop.mean() - individual_sum

        records.append({
            "pair": f"{hp_names[i]} × {hp_names[j]}",
            "hp_i": hp_names[i],
            "hp_j": hp_names[j],
            "joint_importance_mean": joint_drop.mean(),
            "joint_importance_std": joint_drop.std(),
            "individual_sum": individual_sum,
            "interaction_strength": interaction,
        })

    return pd.DataFrame(records).sort_values(
        "interaction_strength", ascending=False, key=abs
    ).reset_index(drop=True)


def run_stratified_fanova(
    df: pd.DataFrame,
    hyperparams: list[str],
    target: str = "accuracy_median",
    budget_col: str = "n_feval_train",
    n_budget_tiers: int = 4,
    n_estimators: int = 128,
    n_repeats: int = 30,
    random_state: int = 42,
    compute_interactions: bool = True,
    min_samples_per_tier: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """
    Run permutation-importance analysis stratified by budget tier.

    Parameters
    ----------
    df : DataFrame with columns for hyperparams, target, and budget_col.
    hyperparams : Column names of the design-choice hyperparameters.
    target : Column name of the response variable (accuracy).
    budget_col : Column name of the total function evaluation budget.
    n_budget_tiers : Number of quantile-based budget tiers (passed to pd.qcut).
    compute_interactions : Whether to compute pairwise interaction importance.
    min_samples_per_tier : Skip tiers with fewer samples than this.

    Returns
    -------
    results : DataFrame of per-tier individual importances.
    summary : Pivot table (tiers × hyperparams) of importance_mean.
    interactions : DataFrame of per-tier pairwise interactions (or None).
    """
    work = df[hyperparams + [target, budget_col]].dropna().copy()

    # Create budget tiers
    tier_labels = [f"Q{i+1}" for i in range(n_budget_tiers)]
    work["budget_tier"], bins = pd.qcut(
        work[budget_col], q=n_budget_tiers, labels=tier_labels, retbins=True
    )

    # Store tier ranges for reporting
    tier_ranges = {}
    for i, label in enumerate(tier_labels):
        tier_ranges[label] = (bins[i], bins[i + 1])

    all_results = []
    all_interactions = []

    # Also run on full data (no stratification)
    tiers_to_run = ["ALL"] + tier_labels

    for tier in tiers_to_run:
        if tier == "ALL":
            subset = work
        else:
            subset = work[work["budget_tier"] == tier]

        if len(subset) < min_samples_per_tier:
            warnings.warn(
                f"Tier {tier} has only {len(subset)} samples "
                f"(< {min_samples_per_tier}), skipping."
            )
            continue

        X, hp_names, _ = _encode_features(subset, hyperparams)
        y = subset[target].values

        imp = _compute_importance(
            X, y, hp_names,
            n_estimators=n_estimators,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        imp["budget_tier"] = tier
        imp["n_samples"] = len(subset)
        if tier in tier_ranges:
            imp["budget_range_low"] = tier_ranges[tier][0]
            imp["budget_range_high"] = tier_ranges[tier][1]
        else:
            imp["budget_range_low"] = work[budget_col].min()
            imp["budget_range_high"] = work[budget_col].max()

        all_results.append(imp)

        if compute_interactions:
            inter = _compute_pairwise_importance(
                X, y, hp_names,
                n_estimators=n_estimators,
                n_repeats=n_repeats,
                random_state=random_state,
            )
            inter["budget_tier"] = tier
            all_interactions.append(inter)

    results = pd.concat(all_results, ignore_index=True)

    # Summary pivot
    summary = results.pivot_table(
        index="budget_tier",
        columns="hyperparameter",
        values="importance_mean",
    )
    # Reorder columns by overall importance
    overall_order = (
        results[results["budget_tier"] == "ALL"]
        .sort_values("importance_mean", ascending=False)["hyperparameter"]
        .tolist()
    )
    if overall_order:
        summary = summary[overall_order]

    interactions = (
        pd.concat(all_interactions, ignore_index=True) if all_interactions else None
    )

    return results, summary, interactions


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_importance_heatmap(
    summary: pd.DataFrame,
    title: str = "Hyperparameter importance by budget tier",
    figsize: tuple[int, int] = (10, 5),
    cmap: str = "YlOrRd",
    annotate: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Heatmap of importance_mean (rows=tiers, cols=hyperparams)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    data = summary.copy()
    im = ax.imshow(data.values, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)

    if annotate:
        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                val = data.values[i, j]
                color = "white" if val > data.values.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color=color)

    fig.colorbar(im, ax=ax, label="Permutation importance (R² drop)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_importance_bars(
    results: pd.DataFrame,
    figsize: tuple[int, int] | None = None,
    title: str = "Hyperparameter importance by budget tier",
) -> plt.Figure:
    """Grouped bar chart: one group per budget tier, bars per hyperparameter."""
    tiers = [t for t in results["budget_tier"].unique() if t != "ALL"]
    hps = (
        results[results["budget_tier"] == "ALL"]
        .sort_values("importance_mean", ascending=False)["hyperparameter"]
        .tolist()
    )
    if not hps:
        hps = results["hyperparameter"].unique().tolist()

    n_tiers = len(tiers)
    n_hps = len(hps)
    if figsize is None:
        figsize = (max(8, n_tiers * 2.5), 5)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_tiers)
    width = 0.8 / n_hps

    for i, hp in enumerate(hps):
        means, stds = [], []
        for tier in tiers:
            row = results[
                (results["budget_tier"] == tier) & (results["hyperparameter"] == hp)
            ]
            if len(row) == 1:
                means.append(row["importance_mean"].values[0])
                stds.append(row["importance_std"].values[0])
            else:
                means.append(0)
                stds.append(0)
        ax.bar(
            x + i * width - 0.4 + width / 2,
            means, width, yerr=stds,
            label=hp, capsize=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_xlabel("Budget tier")
    ax.set_ylabel("Permutation importance (R² drop)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_rank_table(
    results: pd.DataFrame,
    figsize: tuple[int, int] = (10, 4),
    title: str = "Importance ranking by budget tier",
) -> plt.Figure:
    """Table showing rank of each hyperparameter per budget tier."""
    rank_pivot = results.pivot_table(
        index="budget_tier",
        columns="hyperparameter",
        values="rank",
    )

    # Reorder by overall importance
    overall_order = (
        results[results["budget_tier"] == "ALL"]
        .sort_values("importance_mean", ascending=False)["hyperparameter"]
        .tolist()
    )
    if overall_order:
        rank_pivot = rank_pivot[overall_order]

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(
        cellText=rank_pivot.values.astype(int),
        rowLabels=rank_pivot.index,
        colLabels=rank_pivot.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color cells by rank
    n_hps = len(rank_pivot.columns)
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, n_hps))
    for i in range(len(rank_pivot.index)):
        for j in range(len(rank_pivot.columns)):
            rank = int(rank_pivot.values[i, j])
            table[i + 1, j].set_facecolor(colors[rank - 1])
            table[i + 1, j].set_alpha(0.7)

    ax.set_title(title, fontsize=12, pad=20)
    fig.tight_layout()
    return fig