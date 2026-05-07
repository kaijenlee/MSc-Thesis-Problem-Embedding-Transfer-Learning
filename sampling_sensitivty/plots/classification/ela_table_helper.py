import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from itertools import product
from matplotlib.lines import Line2D

STRATEGY_COLORS = {
    "cma_random": "#9467bd",
    "uniform": "#d62728",
    "lhs": "#ff7f0e",
    "ilhs": "#1f77b4",
    "lhs_rcd": "#8c564b",
    "sobol": "#2ca02c",
}

STRATEGY_LABELS = {
    "cma_random": "CMA-ES",
    "uniform": "Uniform",
    "lhs": "LHS",
    "ilhs": "iLHS",
    "lhs_rcd": "LHS-RCD",
    "sobol": "Sobol",
}

STRATEGY_ORDER = ["cma_random", "uniform", "lhs", "lhs_rcd", "ilhs", "sobol"]

SAMPLE_SIZES = [25, 50, 75, 100]



def load_data(path, dim=None):
    """Load the fold-aggregated dataframe."""
    p = Path(path)
    if p.suffix == ".pkl":
        df = pd.read_pickle(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")

    if dim is not None:
        df = df[df["dimension"] == dim].copy()

    # Ensure numeric
    for col in ["n_feval_train", "acc_mean", "acc_sd", "acc_allruns_mean",
                "consistency_mean", "sample_size_per_dim", "n_instances_train",
                "n_runs_train"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort strategies into canonical order
    cat_type = pd.CategoricalDtype(
        categories=[s for s in STRATEGY_ORDER if s in df["sampling_strategy"].unique()],
        ordered=True,
    )
    df["sampling_strategy"] = df["sampling_strategy"].astype(cat_type)

    return df

def get_bins(df, n_bin):
    bins = pd.qcut(df['n_feval_train'], q=n_bin)
    bin_edges = [interval.left for interval in bins.cat.categories] + [bins.cat.categories[-1].right]
    return bin_edges


# =========================================================================
# Plot 1 — Accuracy vs budget, one line per strategy
# =========================================================================
def plot_accuracy_vs_budget(df, metric="acc_allruns_mean", outdir=None, ylim=(0,1.02), log_scale=False, agg_mode="max", plot_bins=None):
    """
    For each strategy, at each budget level, compute the mean (and band)
    of accuracy across all (sample_size, n_instances, n_runs) combos
    that land on that budget.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = [s for s in STRATEGY_ORDER if s in df["sampling_strategy"].unique()]

    for strat in strategies:
        sub = df[df["sampling_strategy"] == strat]
        agg = (
            sub.groupby("n_feval_train")[metric]
            .agg(["mean", "std", "count", "max"])
            .reset_index()
            .sort_values("n_feval_train")
        )

        color = STRATEGY_COLORS.get(strat, "gray")
        label = STRATEGY_LABELS.get(strat, strat)

        ax.plot(agg["n_feval_train"], agg[agg_mode],
                marker="o", markersize=3, color=color, label=label, linewidth=1.5)

        # ±1 SD band where we have >1 observation
        # mask = agg["count"] > 1
        # if mask.any():
        #     ax.fill_between(
        #         agg.loc[mask, "n_feval_train"],
        #         agg.loc[mask, "mean"] - agg.loc[mask, "std"],
        #         agg.loc[mask, "mean"] + agg.loc[mask, "std"],
        #         color=color, alpha=0.12,
        #     )

    ax.set_xlabel("Training function evaluations")
    ax.set_ylabel("Classification accuracy (5-fold CV)")
    ax.set_title(f"Accuracy vs budget — {agg_mode} over design choices {'(log scale on feval)' if log_scale else ''}")
    ax.legend(title="Sampling strategy", loc="lower right")
    if log_scale:
        ax.set_xscale("log")
    if plot_bins is not None:
        bin_edges = get_bins(df, plot_bins)
        for edge in bin_edges:
            ax.axvline(x=edge, color='black', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(ylim)

    fig.tight_layout()
    if outdir is not None:
        fig.savefig(Path(outdir) / "accuracy_vs_budget.pdf", dpi=150)
        fig.savefig(Path(outdir) / "accuracy_vs_budget.png", dpi=150)
        print("  Saved accuracy_vs_budget.pdf/png")
    plt.show()


# =========================================================================
# Plot 2 — Faceted by sample_size_per_dim
# =========================================================================

def plot_accuracy_vs_budget_faceted(df, metric="acc_allruns_mean", outdir=None, ylim=(0,1.02)):
    """
    One subplot per sample_size_per_dim. Within each, lines per strategy.
    x-axis is n_feval_train (which now varies only via n_instances × n_runs).
    """
    sizes = sorted(df["sample_size_per_dim"].unique())
    n = len(sizes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    strategies = [s for s in STRATEGY_ORDER if s in df["sampling_strategy"].unique()]

    for ax, sz in zip(axes, sizes):
        sub = df[df["sample_size_per_dim"] == sz]

        for strat in strategies:
            s = sub[sub["sampling_strategy"] == strat]
            agg = (
                s.groupby("n_feval_train")[metric]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("n_feval_train")
            )
            color = STRATEGY_COLORS.get(strat, "gray")
            label = STRATEGY_LABELS.get(strat, strat)
            ax.plot(agg["n_feval_train"], agg["mean"],
                    marker="o", markersize=3, color=color, label=label, linewidth=1.5)

        ax.set_title(f"Sample size = {sz}d")
        ax.set_xlabel("Training fevals")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(ylim)

    axes[0].set_ylabel("Classification accuracy")
    axes[-1].legend(title="Strategy", loc="lower right", fontsize=8)

    fig.suptitle("Accuracy vs budget — by sample size", y=1.02)
    fig.tight_layout()
    if outdir is not None:
        fig.savefig(Path(outdir) / "accuracy_vs_budget_faceted.pdf", dpi=150)
        fig.savefig(Path(outdir) / "accuracy_vs_budget_faceted.png", dpi=150)
    plt.show()


# =========================================================================
# Plot 3 — Scatter of all design points with Pareto frontier
# =========================================================================

def plot_scatter_with_pareto(df, metric="acc_allruns_mean", outdir=None, ylim=(0,1.02)):
    """
    Each point = one (strategy, sample_size, n_instances, n_runs) config.
    x = budget, y = accuracy. Colour = strategy, marker size = n_instances_train.
    Overlay Pareto frontier (max accuracy at each budget level).
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    strategies = [s for s in STRATEGY_ORDER if s in df["sampling_strategy"].unique()]

    # Size mapping: scale n_instances_train to marker area
    inst_vals = sorted(df["n_instances_train"].unique())
    size_map = {v: 15 + 120 * (i / max(1, len(inst_vals) - 1))
                for i, v in enumerate(inst_vals)}

    for strat in strategies:
        sub = df[df["sampling_strategy"] == strat]
        sizes = sub["n_instances_train"].map(size_map)
        color = STRATEGY_COLORS.get(strat, "gray")
        label = STRATEGY_LABELS.get(strat, strat)
        ax.scatter(sub["n_feval_train"], sub[metric],
                   s=sizes, c=color, alpha=0.5, label=label, edgecolors="white",
                   linewidth=0.3)

    # Pareto frontier: for each budget, best accuracy
    pareto = (
        df.groupby("n_feval_train")[metric]
        .max()
        .reset_index()
        .sort_values("n_feval_train")
    )
    # Keep only non-dominated points (monotonically non-decreasing)
    best_so_far = pareto[metric].cummax()
    pareto = pareto[pareto[metric] == best_so_far]

    ax.step(pareto["n_feval_train"], pareto[metric],
            where="post", color="black", linewidth=2, linestyle="--",
            label="Pareto frontier", zorder=5)

    ax.set_xlabel("Training function evaluations")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("All design configurations — accuracy vs budget")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(ylim)

    # Legend for strategies
    ax.legend(title="Sampling strategy", loc="lower right", fontsize=8)

    # Size legend
    from matplotlib.lines import Line2D
    size_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="",
               markersize=np.sqrt(size_map[v]), label=f"{v} inst",
               markeredgecolor="white", markeredgewidth=0.3, alpha=0.6)
        for v in inst_vals[::2]  # show every other to avoid clutter
    ]
    ax2 = ax.legend(handles=size_handles, title="n_instances",
                    loc="upper left", fontsize=7)
    ax.add_artist(ax.get_legend())  # keep the first legend

    fig.tight_layout()
    if outdir is not None:
        fig.savefig(Path(outdir) / "scatter_pareto.pdf", dpi=150)
        fig.savefig(Path(outdir) / "scatter_pareto.png", dpi=150)
        print("  Saved scatter_pareto.pdf/png")
    plt.show()



# =========================================================================
# Plot 4 — Heatmaps: (n_instances, n_runs) accuracy per strategy × size
# =========================================================================

def plot_allocation_heatmaps(df, metric="acc_allruns_mean", outdir=None):
    """
    Grid of heatmaps: rows = strategies, cols = sample_sizes.
    Each heatmap: x = n_runs_train, y = n_instances_train, colour = accuracy.
    This directly answers: given a strategy and sample size, is it better
    to invest in more instances or more repeat runs?
    """
    strategies = [s for s in STRATEGY_ORDER if s in df["sampling_strategy"].unique()]
    sizes = sorted(df["sample_size_per_dim"].unique())

    n_strat = len(strategies)
    n_size = len(sizes)

    fig, axes = plt.subplots(n_strat, n_size,
                             figsize=(3.5 * n_size, 2.5 * n_strat),
                             squeeze=False)

    # Global colour scale
    vmin = df[metric].quantile(0.05)
    vmax = df[metric].quantile(0.95)

    inst_vals = sorted(df["n_instances_train"].unique())
    run_vals = sorted(df["n_runs_train"].unique())

    for i, strat in enumerate(strategies):
        for j, sz in enumerate(sizes):
            ax = axes[i, j]
            sub = df[(df["sampling_strategy"] == strat) &
                     (df["sample_size_per_dim"] == sz)]

            # Pivot to matrix
            piv = sub.pivot_table(
                index="n_instances_train", columns="n_runs_train",
                values=metric, aggfunc="mean",
            )
            # Reindex to ensure consistent axes
            piv = piv.reindex(index=inst_vals, columns=run_vals)

            im = ax.imshow(piv.values, aspect="auto",
                           vmin=vmin, vmax=vmax, cmap="YlOrRd",
                           origin="lower")

            # Annotate cells
            for yi, inst in enumerate(inst_vals):
                for xi, run in enumerate(run_vals):
                    val = piv.iloc[yi, xi] if not np.isnan(piv.iloc[yi, xi]) else np.nan
                    if not np.isnan(val):
                        ax.text(xi, yi, f"{val:.3f}", ha="center", va="center",
                                fontsize=6,
                                color="white" if val > (vmin + vmax) / 2 else "black")

            ax.set_xticks(range(len(run_vals)))
            ax.set_xticklabels(run_vals, fontsize=7)
            ax.set_yticks(range(len(inst_vals)))
            ax.set_yticklabels(inst_vals, fontsize=7)

            if i == n_strat - 1:
                ax.set_xlabel("n_runs", fontsize=8)
            if j == 0:
                ax.set_ylabel(STRATEGY_LABELS.get(strat, strat), fontsize=9)
            if i == 0:
                ax.set_title(f"{sz}d", fontsize=9)

    fig.suptitle(
        "Accuracy by allocation — rows: strategy, cols: sample size\n"
        "y-axis: n_instances_train, x-axis: n_runs_train",
        fontsize=11, y=1.01,
    )
    fig.colorbar(im, ax=axes, shrink=0.6, label="Accuracy", pad=0.02)
    # fig.tight_layout()
    if outdir is not None:
        fig.savefig(Path(outdir) / "allocation_heatmaps.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(Path(outdir) / "allocation_heatmaps.png", dpi=150, bbox_inches="tight")
        print("  Saved allocation_heatmaps.pdf/png")
    plt.show()


# =========================================================================
# Plot 5 — Marginal effects: accuracy gain per factor
# =========================================================================

def plot_marginal_effects(df, metric="acc_allruns_mean", outdir=None):
    """
    For each design factor, plot the mean accuracy at each level of that factor,
    marginalised over all other factors. Shows which factor has the biggest
    swing in accuracy.
    """
    factors = {
        "sampling_strategy": ("Strategy", None),
        "sample_size_per_dim": ("Sample size (×d)", SAMPLE_SIZES),
        "n_instances_train": ("Instances", None),
        "n_runs_train": ("Runs", None),
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, (col, (label, _)) in zip(axes, factors.items()):
        agg = df.groupby(col)[metric].agg(["mean", "std"]).reset_index()
        agg = agg.sort_values(col)

        x_labels = agg[col].astype(str).values
        x_pos = np.arange(len(x_labels))

        if col == "sampling_strategy":
            colors = [STRATEGY_COLORS.get(s, "gray") for s in agg[col]]
            x_labels = [STRATEGY_LABELS.get(s, s) for s in agg[col]]
        else:
            colors = ["#4C72B0"] * len(x_labels)

        ax.bar(x_pos, agg["mean"], yerr=agg["std"], color=colors,
               capsize=3, alpha=0.8, edgecolor="white")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean accuracy" if ax == axes[0] else "")
        ax.set_title(label)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.3)

        # Annotate the range
        rng = agg["mean"].max() - agg["mean"].min()
        ax.text(0.95, 0.05, f"range: {rng:.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    fig.suptitle("Marginal effect of each design factor on accuracy", fontsize=12)
    if outdir is not None:
        fig.savefig(Path(outdir) / "marginal_effects.pdf", dpi=150)
        fig.savefig(Path(outdir) / "marginal_effects.png", dpi=150)
        print("  Saved marginal_effects.pdf/png")
    plt.show()

def plot_cv_vs_accuracy(df, output_path=None, figsize=(16, 18), dpi=180,
                        jitter_seed=42, ylim=(0,1)):
    """
    Plot feature CV vs classification accuracy by training budget.

    One subplot per sampling strategy, with jittered individual points.
    Colors encode sample_size_per_dim, markers encode n_instances_train.
    Filled markers = accuracy, open markers = CV.

    Accepts both per-fold DataFrames (with 'accuracy_allruns' column)
    and aggregated DataFrames (with 'accuracy_allruns_mean' column).

    Parameters
    ----------
    df : pd.DataFrame
        Results table.
    output_path : str or Path, optional
        Base path (without extension) to save .png and .pdf.
    figsize : tuple
        Figure size (width, height).
    dpi : int
        Resolution for the .png output.
    jitter_seed : int
        Seed for the horizontal jitter RNG.

    Returns
    -------
    fig, axes
    """
    # Support both column naming conventions
    acc_col = ('accuracy_allruns' if 'accuracy_allruns' in df.columns
               else 'accuracy_allruns_mean')
    cv_col = 'cv_instance_median_median'

    strategies = sorted(df['sampling_strategy'].unique().tolist())
    fevals_sorted = sorted(df['n_feval_train'].unique())
    n_fevals = len(fevals_sorted)
    feval_to_x = {f: i for i, f in enumerate(fevals_sorted)}

    # Color map: sample_size_per_dim
    sample_sizes = sorted(df['sample_size_per_dim'].unique())
    size_colors = {
        25: '#e41a1c',
        50: '#377eb8',
        75: '#4daf4a',
        100: '#984ea3',
    }
    _fallback_colors = ['#ff7f00', '#a65628', '#f781bf', '#999999']
    for i, s in enumerate(sample_sizes):
        if s not in size_colors:
            size_colors[s] = _fallback_colors[i % len(_fallback_colors)]

    # Marker map: n_instances_train
    instance_counts = sorted(df['n_instances_train'].unique())
    instance_markers = {
        1: 'o',
        5: 's',
        10: 'D',
        20: '^',
    }
    _fallback_markers = ['v', 'P', 'X', '*']
    for i, n in enumerate(instance_counts):
        if n not in instance_markers:
            instance_markers[n] = _fallback_markers[i % len(_fallback_markers)]

    n_strategies = len(strategies)
    fig, axes = plt.subplots(n_strategies, 1, figsize=figsize,
                             sharex=True, sharey=True)
    if n_strategies == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0.12, left=0.07, right=0.95,
                        top=0.91, bottom=0.08)

    rng = np.random.default_rng(jitter_seed)

    for ax_idx, strat in enumerate(strategies):
        ax = axes[ax_idx]
        sub = df[df['sampling_strategy'] == strat]

        for _, row in sub.iterrows():
            x_base = feval_to_x[row['n_feval_train']]
            color = size_colors[row['sample_size_per_dim']]
            marker = instance_markers[row['n_instances_train']]

            jx_cv = x_base - 0.15 + rng.uniform(-0.06, 0.06)
            jx_acc = x_base + 0.15 + rng.uniform(-0.06, 0.06)

            # CV: open marker
            # ax.scatter(jx_cv, row[cv_col],
            #            facecolors='none', edgecolors=color,
            #            marker=marker, s=30, linewidths=1,
            #            alpha=0.7, zorder=3)
            # Accuracy: filled marker
            ax.scatter(jx_acc, row[acc_col],
                       facecolors=color, edgecolors=color,
                       marker=marker, s=30, linewidths=0.5,
                       alpha=0.7, zorder=3)

        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_ylim(ylim)

        ax.set_ylabel('')
        ax.set_title(strat, fontsize=11, fontweight='bold', loc='left',
                     pad=4)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.grid(axis='x', alpha=0.15, linewidth=0.3)

    # X-axis labels
    axes[-1].set_xticks(range(n_fevals))
    feval_labels = [f'{f // 1000}k' if f >= 1000 else str(f)
                    for f in fevals_sorted]
    axes[-1].set_xticklabels(feval_labels, rotation=45, ha='right',
                             fontsize=8)
    axes[-1].set_xlabel('Function evaluations (training budget)',
                        fontsize=11)

    fig.text(0.02, 0.5, 'Value [0, 1]', va='center',
             rotation='vertical', fontsize=11)

    # --- Legend ---
    legend_elements = []

    # Metric type (filled vs open)
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
               markersize=7, label='Accuracy (filled)', alpha=0.8))
    # legend_elements.append(
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
    #            markeredgecolor='grey', markeredgewidth=1,
    #            markersize=7, label='CV (open)', alpha=0.8))

    # Spacer
    legend_elements.append(
        Line2D([0], [0], color='w', label=''))

    # Colors: sample_size_per_dim
    for s in sample_sizes:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=size_colors[s], markersize=7,
                   label=f'sample_size={s}'))

    # Spacer
    legend_elements.append(
        Line2D([0], [0], color='w', label=''))

    # Markers: n_instances_train
    for n in instance_counts:
        legend_elements.append(
            Line2D([0], [0], marker=instance_markers[n], color='w',
                   markerfacecolor='grey', markersize=7,
                   label=f'n_instances={n}'))

    fig.legend(handles=legend_elements, loc='upper center',
               ncol=len(legend_elements), fontsize=8,
               frameon=True, fancybox=False, edgecolor='#cccccc',
               bbox_to_anchor=(0.5, 0.98), columnspacing=1.2,
               handletextpad=0.4)

    fig.suptitle('Classification Accuracy against Training Budget',
                 fontsize=14, fontweight='bold', y=1.0)

    if output_path is not None:
        from pathlib import Path
        output_path = Path(output_path)
        fig.savefig(output_path.with_suffix('.png'), dpi=dpi,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(output_path.with_suffix('.pdf'),
                    bbox_inches='tight', facecolor='white')

    plt.show()
    return fig, axes
