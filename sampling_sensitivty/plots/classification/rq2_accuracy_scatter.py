"""
RQ2 — Accuracy vs Budget Scatter Plots
========================================

Clean scatter plots showing classification accuracy vs function evaluation
budget, with visual encoding of design choices.

Usage in notebook:
    from rq2_accuracy_scatter import *

    # Main plot: faceted by strategy, colour=sample_size, size=n_instances
    plot_accuracy_scatter(fold_agg_df, metric="acc_mean")

    # All strategies on one panel
    plot_accuracy_scatter_combined(fold_agg_df, metric="acc_mean")

    # With Pareto frontier
    plot_accuracy_scatter_combined(fold_agg_df, metric="acc_mean", show_pareto=True)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

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

SAMPLE_SIZE_COLORS = {
    25:  "#e41a1c",
    50:  "#377eb8",
    75:  "#4daf4a",
    100: "#984ea3",
}

INSTANCE_SIZES = {
    1: 15, 2: 25, 3: 35, 5: 50, 7: 65, 10: 85, 15: 110, 20: 140,
}

RUN_MARKERS = {
    1: "o", 2: "s", 3: "D", 5: "^",
}

DEFAULT_METRIC = "acc_mean"


def _get_strategies(df):
    return [s for s in STRATEGY_ORDER if s in df["sampling_strategy"].unique()]


def filter_strategies(df, omit_strategies=None):
    if omit_strategies is None:
        return df
    if isinstance(omit_strategies, str):
        omit_strategies = [omit_strategies]
    return df[~df["sampling_strategy"].isin(omit_strategies)].copy()


def _format_budget_axis(ax):
    """Format x-axis for budget (log scale with readable labels)."""
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"
    ))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


# =========================================================================
# Plot 1 — Faceted by strategy
# =========================================================================

def plot_accuracy_scatter(df, metric=DEFAULT_METRIC, omit_strategies=None,
                           ncols=3, figsize=None):
    """
    One subplot per strategy. Within each:
      - x = n_feval_train (log scale)
      - y = accuracy
      - colour = sample_size_per_dim
      - marker size = n_instances_train
      - marker shape = n_runs_train

    This separates strategy effects and lets you see within-strategy patterns.
    """
    df = filter_strategies(df, omit_strategies)
    strategies = _get_strategies(df)
    n = len(strategies)
    nrows = int(np.ceil(n / ncols))

    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes = np.atleast_2d(axes)

    for idx, strat in enumerate(strategies):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        sub = df[df["sampling_strategy"] == strat]

        for _, r in sub.iterrows():
            sz = int(r["sample_size_per_dim"])
            n_inst = int(r["n_instances_train"])
            n_run = int(r["n_runs_train"])

            color = SAMPLE_SIZE_COLORS.get(sz, "gray")
            size = INSTANCE_SIZES.get(n_inst, 50)
            marker = RUN_MARKERS.get(n_run, "o")

            ax.scatter(r["n_feval_train"], r[metric],
                       c=color, s=size, marker=marker,
                       alpha=0.6, edgecolors="white", linewidth=0.3,
                       zorder=3)

        ax.set_title(STRATEGY_LABELS.get(strat, strat), fontsize=12,
                     fontweight="bold")
        _format_budget_axis(ax)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 1.02)

        if col == 0:
            ax.set_ylabel("Classification accuracy")
        if row == nrows - 1:
            ax.set_xlabel("Function evaluations (training budget)")

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Legends
    _add_legends(fig)

    fig.suptitle(f"Accuracy vs budget by strategy — {metric}", fontsize=13, y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =========================================================================
# Plot 2 — All strategies combined
# =========================================================================

def plot_accuracy_scatter_combined(df, metric=DEFAULT_METRIC,
                                    omit_strategies=None,
                                    show_pareto=False, ax=None):
    """
    Single panel with all strategies. Here colour = strategy (since that's
    the most important factor), marker size = n_instances, marker shape =
    sample_size. Provides the overview.
    """
    df = filter_strategies(df, omit_strategies)
    strategies = _get_strategies(df)

    size_markers = {25: "o", 50: "s", 75: "D", 100: "^"}

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))

    for strat in strategies:
        sub = df[df["sampling_strategy"] == strat]
        color = STRATEGY_COLORS.get(strat, "gray")

        for _, r in sub.iterrows():
            sz = int(r["sample_size_per_dim"])
            n_inst = int(r["n_instances_train"])

            marker = size_markers.get(sz, "o")
            size = INSTANCE_SIZES.get(n_inst, 50)

            ax.scatter(r["n_feval_train"], r[metric],
                       c=color, s=size, marker=marker,
                       alpha=0.45, edgecolors="white", linewidth=0.3,
                       zorder=3)

    # Pareto frontier
    if show_pareto:
        pareto = (
            df.groupby("n_feval_train")[metric]
            .max()
            .reset_index()
            .sort_values("n_feval_train")
        )
        best_so_far = pareto[metric].cummax()
        pareto = pareto[pareto[metric] == best_so_far]
        ax.step(pareto["n_feval_train"], pareto[metric],
                where="post", color="black", linewidth=2, linestyle="--",
                label="Pareto frontier", zorder=5)

    _format_budget_axis(ax)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Function evaluations (training budget)")
    ax.set_ylabel("Classification accuracy")
    ax.set_title(f"All configurations — {metric}")

    # Strategy legend
    strat_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STRATEGY_COLORS[s],
               markersize=8, label=STRATEGY_LABELS[s])
        for s in strategies
    ]
    # Size legend (sample_size via marker shape)
    shape_handles = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor="gray",
               markersize=7, label=f"{sz}d")
        for sz, m in size_markers.items()
    ]
    # Instance legend (marker area)
    inst_show = [1, 5, 10, 20]
    inst_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=np.sqrt(INSTANCE_SIZES.get(i, 50)) * 0.8,
               label=f"{i} inst")
        for i in inst_show if i in INSTANCE_SIZES
    ]

    if show_pareto:
        strat_handles.append(
            Line2D([0], [0], color="black", linestyle="--", linewidth=2,
                   label="Pareto frontier")
        )

    leg1 = ax.legend(handles=strat_handles, title="Strategy",
                     loc="lower right", fontsize=7, title_fontsize=8)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=shape_handles + inst_handles,
                     title="Sample size / Instances",
                     loc="center right", fontsize=7, title_fontsize=8)

    plt.tight_layout()
    return ax


# =========================================================================
# Plot 3 — Faceted by sample_size, coloured by strategy, sized by instances
# =========================================================================

def plot_accuracy_scatter_by_size(df, metric=DEFAULT_METRIC,
                                   omit_strategies=None):
    """
    One subplot per sample_size_per_dim. Within each, the budget varies
    only via n_instances × n_runs (the sample size is fixed).
    Colour = strategy, size = n_instances, shape = n_runs.

    This directly answers: at a fixed sample quality, does adding more
    instances help more than adding more runs?
    """
    df = filter_strategies(df, omit_strategies)
    strategies = _get_strategies(df)
    sizes = sorted(df["sample_size_per_dim"].unique())
    n = len(sizes)

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, sz in zip(axes, sizes):
        sub = df[df["sample_size_per_dim"] == sz]

        for strat in strategies:
            s = sub[sub["sampling_strategy"] == strat]
            color = STRATEGY_COLORS.get(strat, "gray")

            for _, r in s.iterrows():
                n_inst = int(r["n_instances_train"])
                n_run = int(r["n_runs_train"])
                size = INSTANCE_SIZES.get(n_inst, 50)
                marker = RUN_MARKERS.get(n_run, "o")

                ax.scatter(r["n_feval_train"], r[metric],
                           c=color, s=size, marker=marker,
                           alpha=0.6, edgecolors="white", linewidth=0.3,
                           zorder=3)

        ax.set_title(f"Sample size = {sz}d", fontsize=11)
        _format_budget_axis(ax)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Function evaluations")

    axes[0].set_ylabel("Classification accuracy")

    _add_legends(fig, include_sample_size=False)

    fig.suptitle(f"Accuracy vs budget — faceted by sample size — {metric}",
                 fontsize=12, y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =========================================================================
# Plot 4 — Fixed instances, showing effect of sample_size × strategy
# =========================================================================

def plot_accuracy_scatter_by_instances(df, metric=DEFAULT_METRIC,
                                        omit_strategies=None,
                                        instance_levels=None):
    """
    One subplot per n_instances_train level. Within each, colour = strategy,
    shape = sample_size. Shows how sample size matters at each instance count.
    """
    df = filter_strategies(df, omit_strategies)
    strategies = _get_strategies(df)

    if instance_levels is None:
        instance_levels = [1, 3, 7, 20]
    instance_levels = [i for i in instance_levels
                       if i in df["n_instances_train"].unique()]

    n = len(instance_levels)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    size_markers = {25: "o", 50: "s", 75: "D", 100: "^"}

    for ax, n_inst in zip(axes, instance_levels):
        sub = df[df["n_instances_train"] == n_inst]

        for strat in strategies:
            s = sub[sub["sampling_strategy"] == strat]
            color = STRATEGY_COLORS.get(strat, "gray")

            for _, r in s.iterrows():
                sz = int(r["sample_size_per_dim"])
                n_run = int(r["n_runs_train"])
                marker = size_markers.get(sz, "o")
                # Use n_runs for size here
                size = 30 + 25 * n_run

                ax.scatter(r["n_feval_train"], r[metric],
                           c=color, s=size, marker=marker,
                           alpha=0.6, edgecolors="white", linewidth=0.3,
                           zorder=3)

        ax.set_title(f"n_instances = {n_inst}", fontsize=11)
        _format_budget_axis(ax)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Function evaluations")

    axes[0].set_ylabel("Classification accuracy")

    # Legend
    strat_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STRATEGY_COLORS[s],
               markersize=8, label=STRATEGY_LABELS[s])
        for s in strategies
    ]
    shape_handles = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor="gray",
               markersize=7, label=f"{sz}d")
        for sz, m in size_markers.items()
    ]
    fig.legend(handles=strat_handles + shape_handles,
               loc="upper center", ncol=len(strategies) + len(size_markers),
               fontsize=7, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle(f"Accuracy vs budget — faceted by n_instances — {metric}",
                 fontsize=12, y=1.1)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =========================================================================
# Shared legend helper
# =========================================================================

def _add_legends(fig, include_sample_size=True):
    """Add shared legends to figure top."""
    handles = []

    # Sample size (colour)
    if include_sample_size:
        for sz, color in SAMPLE_SIZE_COLORS.items():
            handles.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                       markersize=8, label=f"size={sz}d")
            )
        handles.append(Line2D([0], [0], color="w", label="  "))  # spacer

    # n_instances (size) — show a few
    for n_inst in [1, 5, 10, 20]:
        if n_inst in INSTANCE_SIZES:
            handles.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                       markersize=np.sqrt(INSTANCE_SIZES[n_inst]) * 0.7,
                       label=f"{n_inst} inst")
            )
    handles.append(Line2D([0], [0], color="w", label="  "))  # spacer

    # n_runs (shape)
    for n_run, marker in RUN_MARKERS.items():
        handles.append(
            Line2D([0], [0], marker=marker, color="w", markerfacecolor="gray",
                   markersize=7, label=f"{n_run} runs")
        )

    fig.legend(handles=handles, loc="upper center",
               ncol=len(handles), fontsize=7,
               bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.0, handletextpad=0.3)


# =========================================================================
# Convenience: run all scatter plots
# =========================================================================

def run_all(df, metric=DEFAULT_METRIC, omit_strategies=None):
    """Produce all four scatter plot variants."""
    print("Plot 1: Faceted by strategy ...")
    fig1 = plot_accuracy_scatter(df, metric=metric, omit_strategies=omit_strategies)
    plt.show()

    print("Plot 2: All strategies combined ...")
    plot_accuracy_scatter_combined(df, metric=metric,
                                   omit_strategies=omit_strategies,
                                   show_pareto=True)
    plt.show()

    print("Plot 3: Faceted by sample size ...")
    fig3 = plot_accuracy_scatter_by_size(df, metric=metric,
                                          omit_strategies=omit_strategies)
    plt.show()

    print("Plot 4: Faceted by n_instances ...")
    fig4 = plot_accuracy_scatter_by_instances(df, metric=metric,
                                               omit_strategies=omit_strategies)
    plt.show()