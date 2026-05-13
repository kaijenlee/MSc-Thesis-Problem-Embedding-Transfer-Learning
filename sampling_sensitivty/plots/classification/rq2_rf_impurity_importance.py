"""
impurity_importance_utils.py — Stratified hyperparameter importance analysis via
random-forest impurity-based (mean decrease in variance) importance.

The impurity-based importance measures how much each hyperparameter contributes
to reducing prediction variance across all splits in all trees. Values are
normalised by sklearn to sum to 1.0 within each budget tier.

Caveat: biased toward hyperparameters with more unique values (cardinality bias).
Use alongside permutation importance (fanova_utils.py) as a robustness check.

Usage
-----
    from impurity_importance_utils import (
        run_stratified_impurity_importance,
        plot_importance_heatmap,
        plot_importance_bars,
        plot_rank_table,
    )

    results, summary, metrics = run_stratified_impurity_importance(
        df,
        hyperparams=["sampling_strategy", "sample_size_per_dim",
                      "n_instances_train", "n_runs_train"],
        target="acc_mean",
        budget_col="n_feval_train",
        n_budget_tiers=4,
    )
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Internal helpers
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


def _compute_impurity_importance(
    X: np.ndarray,
    y: np.ndarray,
    hp_names: list[str],
    n_estimators: int = 128,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Fit RF and extract impurity-based importance + model performance metrics.

    Returns
    -------
    result : DataFrame with importance per hyperparameter.
    metrics : dict with RF performance metrics (OOB R², MSE, MAE).
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        oob_score=True,
    )
    rf.fit(X, y)

    # --- Model performance metrics ---
    y_pred_train = rf.predict(X)
    y_pred_oob = rf.oob_prediction_

    metrics = {
        "oob_r2": rf.oob_score_,
        "mse_train": mean_squared_error(y, y_pred_train),
        "mae_train": mean_absolute_error(y, y_pred_train),
        "mse_oob": mean_squared_error(y, y_pred_oob),
        "mae_oob": mean_absolute_error(y, y_pred_oob),
        "n_samples": len(y),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
    }

    # --- Impurity importance (normalised to sum to 1 by sklearn) ---
    importances = rf.feature_importances_

    # Per-tree importances for std estimation
    tree_importances = np.array([
        tree.feature_importances_ for tree in rf.estimators_
    ])

    records = []
    for i, name in enumerate(hp_names):
        records.append({
            "hyperparameter": name,
            "importance_mean": importances[i],
            "importance_std": tree_importances[:, i].std(),
        })

    result = pd.DataFrame(records).sort_values(
        "importance_mean", ascending=False
    ).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    return result, metrics


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_stratified_impurity_importance(
    df: pd.DataFrame,
    hyperparams: list[str],
    target: str = "acc_allruns_mean",
    budget_col: str = "n_feval_train",
    n_budget_tiers: int = 4,
    n_estimators: int = 128,
    random_state: int = 42,
    min_samples_per_tier: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run impurity-based importance analysis stratified by budget tier.

    Parameters
    ----------
    df : DataFrame with columns for hyperparams, target, and budget_col.
    hyperparams : Column names of the design-choice hyperparameters.
    target : Column name of the response variable (accuracy).
    budget_col : Column name of the total function evaluation budget.
    n_budget_tiers : Number of quantile-based budget tiers (pd.qcut).
    min_samples_per_tier : Skip tiers with fewer samples than this.

    Returns
    -------
    results : DataFrame of per-tier individual importances.
    summary : Pivot table (tiers × hyperparams) of importance_mean.
    metrics_df : DataFrame of RF performance metrics per tier.
    """
    work = df[hyperparams + [target, budget_col]].dropna().copy()

    # Create budget tiers
    tier_labels = [f"Q{i+1}" for i in range(n_budget_tiers)]
    work["budget_tier"], bins = pd.qcut(
        work[budget_col], q=n_budget_tiers, labels=tier_labels, retbins=True
    )

    tier_ranges = {}
    for i, label in enumerate(tier_labels):
        tier_ranges[label] = (bins[i], bins[i + 1])

    all_results = []
    all_metrics = []
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

        imp, metrics = _compute_impurity_importance(
            X, y, hp_names,
            n_estimators=n_estimators,
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

        metrics["budget_tier"] = tier
        if tier in tier_ranges:
            metrics["budget_range_low"] = tier_ranges[tier][0]
            metrics["budget_range_high"] = tier_ranges[tier][1]
        all_metrics.append(metrics)

    results = pd.concat(all_results, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)

    # Summary pivot
    summary = results.pivot_table(
        index="budget_tier",
        columns="hyperparameter",
        values="importance_mean",
    )
    overall_order = (
        results[results["budget_tier"] == "ALL"]
        .sort_values("importance_mean", ascending=False)["hyperparameter"]
        .tolist()
    )
    if overall_order:
        summary = summary[overall_order]

    return results, summary, metrics_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_importance_heatmap(
    summary: pd.DataFrame,
    title: str = "Impurity-based importance by budget tier",
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

    fig.colorbar(im, ax=ax, label="Impurity-based importance (fraction)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_importance_bars(
    results: pd.DataFrame,
    figsize: tuple[int, int] | None = None,
    title: str = "Impurity-based importance by budget tier",
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
    ax.set_ylabel("Impurity-based importance (fraction)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_rank_table(
    results: pd.DataFrame,
    figsize: tuple[int, int] = (10, 4),
    title: str = "Impurity-based importance ranking by budget tier",
) -> plt.Figure:
    """Table showing rank of each hyperparameter per budget tier."""
    rank_pivot = results.pivot_table(
        index="budget_tier",
        columns="hyperparameter",
        values="rank",
    )

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


def plot_metrics_table(
    metrics_df: pd.DataFrame,
    figsize: tuple[int, int] = (12, 4),
    title: str = "Random Forest surrogate model performance",
) -> plt.Figure:
    """Table showing RF performance metrics per budget tier."""
    display_cols = [
        "budget_tier", "n_samples", "y_mean", "y_std",
        "oob_r2", "mse_oob", "mae_oob", "mse_train", "mae_train",
    ]
    display = metrics_df[display_cols].copy()

    # Round for readability
    for col in display.columns:
        if col not in ("budget_tier", "n_samples"):
            display[col] = display[col].apply(lambda v: f"{v:.4f}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Header styling
    for j in range(len(display.columns)):
        table[0, j].set_facecolor("#d9e2f3")
        table[0, j].set_text_props(fontweight="bold")

    ax.set_title(title, fontsize=12, pad=20)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Partial Dependence Plots
# ---------------------------------------------------------------------------

def _fit_surrogate(
    df: pd.DataFrame,
    hyperparams: list[str],
    target: str = "acc_allruns_mean",
    n_estimators: int = 128,
    random_state: int = 42,
) -> tuple[RandomForestRegressor, np.ndarray, np.ndarray, list[str], dict[str, LabelEncoder]]:
    """Fit RF surrogate and return model + encoded data + encoders."""
    X, hp_names, encoders = _encode_features(df, hyperparams)
    y = df[target].values
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        oob_score=True,
    )
    rf.fit(X, y)
    return rf, X, y, hp_names, encoders


def plot_pdp_individual(
    df: pd.DataFrame,
    hyperparams: list[str],
    target: str = "acc_allruns_mean",
    n_estimators: int = 128,
    random_state: int = 42,
    figsize: tuple[int, int] | None = None,
    title: str = "Partial Dependence Plots",
) -> plt.Figure:
    """
    1D partial dependence plot for each hyperparameter.

    Shows the marginal effect of each hyperparameter on predicted accuracy,
    averaged over all other hyperparameters. ICE lines show per-sample variation.
    """
    from sklearn.inspection import PartialDependenceDisplay

    rf, X, y, hp_names, encoders = _fit_surrogate(
        df, hyperparams, target, n_estimators, random_state
    )

    n_hps = len(hp_names)
    if figsize is None:
        figsize = (5 * n_hps, 4)

    fig, axes = plt.subplots(1, n_hps, figsize=figsize)
    if n_hps == 1:
        axes = [axes]

    for i, (hp, ax) in enumerate(zip(hp_names, axes)):
        PartialDependenceDisplay.from_estimator(
            rf, X, features=[i],
            feature_names=hp_names,
            ax=ax, kind="both",  # average + ICE lines
            ice_lines_kw={"alpha": 0.1, "linewidth": 0.5},
            pd_line_kw={"linewidth": 2},
        )
        # Fix x-axis labels for categorical features
        if hp in encoders:
            le = encoders[hp]
            tick_vals = sorted(set(X[:, i].astype(int)))
            ax.set_xticks(tick_vals)
            ax.set_xticklabels(
                [le.inverse_transform([v])[0] for v in tick_vals],
                rotation=30, ha="right",
            )
        ax.set_xlabel(hp)
        ax.set_ylabel("Partial dependence" if i == 0 else "")
        ax.set_title(hp)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def plot_pdp_2d(
    df: pd.DataFrame,
    hyperparams: list[str],
    feature_pair: tuple[str, str],
    target: str = "acc_allruns_mean",
    n_estimators: int = 128,
    random_state: int = 42,
    figsize: tuple[int, int] = (8, 6),
    title: str | None = None,
) -> plt.Figure:
    """
    2D partial dependence contour plot for a pair of hyperparameters.

    Useful for visualising the n_instances × n_runs tradeoff surface.
    """
    from sklearn.inspection import PartialDependenceDisplay

    rf, X, y, hp_names, encoders = _fit_surrogate(
        df, hyperparams, target, n_estimators, random_state
    )

    idx_i = hp_names.index(feature_pair[0])
    idx_j = hp_names.index(feature_pair[1])

    fig, ax = plt.subplots(figsize=figsize)
    PartialDependenceDisplay.from_estimator(
        rf, X, features=[(idx_i, idx_j)],
        feature_names=hp_names,
        ax=ax, kind="average",
    )

    # Fix axis labels for categoricals
    for idx, feat, set_ticks, set_labels in [
        (idx_i, feature_pair[0], ax.set_xticks, ax.set_xticklabels),
        (idx_j, feature_pair[1], ax.set_yticks, ax.set_yticklabels),
    ]:
        if feat in encoders:
            le = encoders[feat]
            tick_vals = sorted(set(X[:, idx].astype(int)))
            set_ticks(tick_vals)
            set_labels([le.inverse_transform([v])[0] for v in tick_vals])

    ax.set_xlabel(feature_pair[0])
    ax.set_ylabel(feature_pair[1])
    if title is None:
        title = f"2D Partial Dependence: {feature_pair[0]} × {feature_pair[1]}"
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pdp_by_strategy(
    df: pd.DataFrame,
    hyperparams: list[str],
    strategy_col: str = "sampling_strategy",
    feature: str = "n_instances_train",
    target: str = "acc_allruns_mean",
    n_estimators: int = 128,
    random_state: int = 42,
    figsize: tuple[int, int] = (8, 5),
    title: str | None = None,
) -> plt.Figure:
    """
    Overlay 1D partial dependence curves for a single hyperparameter,
    one curve per sampling strategy.

    This directly visualises the interaction between sampling strategy
    and the chosen hyperparameter — e.g., showing that stable strategies
    benefit more from additional instances while unstable strategies
    benefit more from additional runs.
    """
    strategies = sorted(df[strategy_col].unique())

    fig, ax = plt.subplots(figsize=figsize)

    for strategy in strategies:
        subset = df[df[strategy_col] == strategy].copy()
        if len(subset) < 20:
            continue

        rf, X, y, hp_names, encoders = _fit_surrogate(
            subset, hyperparams, target, n_estimators, random_state
        )
        feat_idx = hp_names.index(feature)

        from sklearn.inspection import partial_dependence
        pd_result = partial_dependence(
            rf, X, features=[feat_idx], kind="average"
        )
        pd_values = pd_result["average"][0]
        # sklearn >= 1.3 uses "grid_values", some versions use "values"
        feat_values = pd_result.get("grid_values", pd_result.get("values"))[0]

        ax.plot(feat_values, pd_values, marker="o", markersize=4, label=strategy)

    ax.set_xlabel(feature)
    ax.set_ylabel(f"Partial dependence on {target}")
    ax.legend(title=strategy_col)
    if title is None:
        title = f"Partial dependence of {feature} by {strategy_col}"
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pdp_2d_heatmap(
        df: pd.DataFrame,
        hyperparams: list[str],
        feature_pair: tuple[str, str],
        target: str = "acc_mean",
        n_estimators: int = 128,
        random_state: int = 42,
        figsize: tuple[int, int] = (8, 6),
        title: str | None = None,
        cmap: str = "YlOrRd",
        annotate: bool = True,
) -> plt.Figure:
    """
    2D partial dependence heatmap for a pair of hyperparameters.

    Useful for visualising the n_instances × n_runs tradeoff surface.
    """
    from sklearn.inspection import partial_dependence

    rf, X, y, hp_names, encoders = _fit_surrogate(
        df, hyperparams, target, n_estimators, random_state
    )

    idx_i = hp_names.index(feature_pair[0])
    idx_j = hp_names.index(feature_pair[1])

    # Use actual unique values from the data as the grid
    grid_i = np.sort(df[feature_pair[0]].unique())
    grid_j = np.sort(df[feature_pair[1]].unique())

    # Encode grid values if categorical
    grid_i_enc = grid_i.copy()
    grid_j_enc = grid_j.copy()
    if feature_pair[0] in encoders:
        grid_i_enc = encoders[feature_pair[0]].transform(grid_i.astype(str))
    else:
        grid_i_enc = grid_i.astype(float)
    if feature_pair[1] in encoders:
        grid_j_enc = encoders[feature_pair[1]].transform(grid_j.astype(str))
    else:
        grid_j_enc = grid_j.astype(float)

    # Compute PD manually over the actual data grid
    pd_values = np.zeros((len(grid_j), len(grid_i)))
    for ii, vi in enumerate(grid_i_enc):
        for jj, vj in enumerate(grid_j_enc):
            X_mod = X.copy()
            X_mod[:, idx_i] = vi
            X_mod[:, idx_j] = vj
            pd_values[jj, ii] = rf.predict(X_mod).mean()

    # Plot as heatmap
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        pd_values, cmap=cmap, aspect="auto",
        origin="lower",
    )

    # Axis ticks using actual values
    ax.set_xticks(range(len(grid_i)))
    ax.set_xticklabels(grid_i)
    ax.set_yticks(range(len(grid_j)))
    ax.set_yticklabels(grid_j)

    # Annotate cells
    if annotate:
        for ii in range(len(grid_i)):
            for jj in range(len(grid_j)):
                val = pd_values[jj, ii]
                color = "white" if val > pd_values.max() * 0.75 else "black"
                ax.text(ii, jj, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label=f"Predicted {target}")
    ax.set_xlabel(feature_pair[0])
    ax.set_ylabel(feature_pair[1])
    if title is None:
        title = f"2D Partial Dependence: {feature_pair[0]} × {feature_pair[1]}"
    ax.set_title(title)
    fig.tight_layout()
    return fig