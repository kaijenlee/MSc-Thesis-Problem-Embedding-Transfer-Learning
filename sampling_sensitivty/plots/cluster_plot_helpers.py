"""
Plot ARI/NMI clustering analysis results for ELA and TLA features.

All plot functions accept an optional `omit_strategies` parameter.

Usage in notebook:
    from plot_clustering import *

    ela_h5 = "path/to/ela_cluster_analysis.h5"
    tla_h5 = "path/to/tla_cluster_analysis.h5"

    run_all(ela_h5, tla_h5, omit_strategies={"cma_random"})
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_FUNCTIONS = 24
N_INSTANCES = 100
DIMENSION = 2

SAMPLING_STRATEGIES = ["ilhs", "lhs", "sobol", "uniform", "cma_random"]
SAMPLE_SIZES_ELA = [25, 50, 75, 100]
SAMPLE_SIZES_TLA = [10, 25, 50, 75, 100]

STRATEGY_COLORS = {
    "ilhs": "#1f77b4", "lhs": "#ff7f0e", "sobol": "#2ca02c",
    "uniform": "#d62728", "cma_random": "#9467bd",
}
STRATEGY_LABELS = {
    "ilhs": "iLHS", "lhs": "LHS", "sobol": "Sobol",
    "uniform": "Uniform", "cma_random": "CMA-Random",
}

FUNCTION_GROUPS = {
    "Separable": [1, 2, 3, 4, 5],
    "Low/Moderate Cond.": [6, 7, 8, 9],
    "High Conditioning": [10, 11, 12, 13, 14],
    "Multimodal (adequate)": [15, 16, 17, 18, 19],
    "Multimodal (weak)": [20, 21, 22, 23, 24],
}

CLUSTER_METRICS = [("ari", "ARI"), ("nmi", "NMI")]

TLA_SEGMENTS = [
    "all", "volume_all", "axis_all",
    "volume_h0", "volume_h1", "volume_h2",
    "axis_h0", "axis_h1", "axis_h2",
]


def _active_strategies(omit_strategies=None):
    omit = omit_strategies or set()
    return [s for s in SAMPLING_STRATEGIES if s not in omit]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_cluster_values(h5_path, config_key, segment, metric):
    """
    Load 30 ARI or NMI values for a config/segment.
    Returns (array_of_30, is_missing).
    """
    try:
        with h5py.File(h5_path, "r") as f:
            if config_key not in f:
                return np.array([]), True
            config_grp = f[config_key]
            if segment not in config_grp:
                return np.array([]), True
            seg_grp = config_grp[segment]
            if metric not in seg_grp:
                return np.array([]), True
            return seg_grp[metric][:], False
    except Exception as e:
        print(f"Error loading {config_key}/{segment}/{metric}: {e}")
        return np.array([]), True


# ===========================================================================
# Line plots
# ===========================================================================

def _plot_line(h5_path, sample_sizes, segment, ft_label, omit_strategies=None):
    """Line plot: mean ARI/NMI vs sample size, with error bars (±1 std)."""
    strategies = _active_strategies(omit_strategies)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    missing = []

    for ax_idx, (metric, metric_label) in enumerate(CLUSTER_METRICS):
        ax = axes[ax_idx]

        for strategy in strategies:
            means, stds, sizes = [], [], []
            for size in sample_sizes:
                config_key = f"{strategy}_{size}"
                values, is_missing = _load_cluster_values(
                    h5_path, config_key, segment, metric)
                if is_missing:
                    if config_key not in missing:
                        missing.append(config_key)
                    continue
                finite = values[np.isfinite(values)]
                if len(finite) > 0:
                    means.append(np.mean(finite))
                    stds.append(np.std(finite))
                    sizes.append(size)

            if sizes:
                ax.errorbar(sizes, means, yerr=stds, marker="o", linewidth=2,
                            markersize=6, capsize=3,
                            color=STRATEGY_COLORS[strategy],
                            label=STRATEGY_LABELS[strategy])

        ax.set_xlabel("Sample Size (×d)", fontsize=12)
        ax.set_ylabel(f"Mean {metric_label}", fontsize=12)
        ax.set_title(metric_label, fontsize=13)
        ax.set_xticks(sample_sizes)
        ax.set_xticklabels([f"{s}d" for s in sample_sizes])
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    seg_label = segment.replace("_", " ").title()
    fig.suptitle(f"{ft_label} Clustering Quality ({seg_label})",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    return missing


def plot_ela_line(h5_path, segment="all", omit_strategies=None):
    """ELA line plot: mean ARI/NMI with error bars."""
    return _plot_line(h5_path, SAMPLE_SIZES_ELA, segment, "ELA", omit_strategies)


def plot_tla_line(h5_path, segment="all", omit_strategies=None):
    """TLA line plot: mean ARI/NMI with error bars."""
    return _plot_line(h5_path, SAMPLE_SIZES_TLA, segment, "TLA", omit_strategies)


def plot_tla_line_all_segments(h5_path, omit_strategies=None):
    """TLA line plots: all segments in a grid, one metric at a time."""
    strategies = _active_strategies(omit_strategies)
    missing = []

    for metric, metric_label in CLUSTER_METRICS:
        ncols, nrows = 3, (len(TLA_SEGMENTS) + 2) // 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
        axes = axes.flatten()

        for idx, segment in enumerate(TLA_SEGMENTS):
            ax = axes[idx]
            for strategy in strategies:
                means, sizes = [], []
                for size in SAMPLE_SIZES_TLA:
                    config_key = f"{strategy}_{size}"
                    values, is_missing = _load_cluster_values(
                        h5_path, config_key, segment, metric)
                    if is_missing:
                        if config_key not in missing:
                            missing.append(config_key)
                        continue
                    finite = values[np.isfinite(values)]
                    if len(finite) > 0:
                        means.append(np.mean(finite))
                        sizes.append(size)
                if sizes:
                    ax.plot(sizes, means, marker="o", linewidth=1.5, markersize=4,
                            color=STRATEGY_COLORS[strategy],
                            label=STRATEGY_LABELS[strategy])

            ax.set_title(segment.replace("_", " ").title(), fontsize=11)
            ax.set_xticks(SAMPLE_SIZES_TLA)
            ax.set_xticklabels([f"{s}d" for s in SAMPLE_SIZES_TLA], fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
            if idx % ncols == 0:
                ax.set_ylabel(f"Mean {metric_label}", fontsize=10)
            if idx >= (nrows - 1) * ncols:
                ax.set_xlabel("Sample Size (×d)", fontsize=10)

        for idx in range(len(TLA_SEGMENTS), len(axes)):
            axes[idx].set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(strategies),
                   fontsize=10, bbox_to_anchor=(0.5, 1.02), frameon=True)
        fig.suptitle(f"TLA Clustering Quality — All Segments ({metric_label})",
                     fontsize=14, y=1.06)
        plt.tight_layout()
        plt.show()

    return missing


# ===========================================================================
# Boxplots
# ===========================================================================

def _plot_box(h5_path, sample_sizes, segment, ft_label, omit_strategies=None):
    """Boxplots: distribution of 30 ARI/NMI values per config."""
    strategies = _active_strategies(omit_strategies)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    missing = []
    n_strategies = len(strategies)
    width = 0.8 / n_strategies

    for ax_idx, (metric, metric_label) in enumerate(CLUSTER_METRICS):
        ax = axes[ax_idx]
        positions, box_data, colors = [], [], []
        tick_positions, tick_labels = [], []

        for size_idx, size in enumerate(sample_sizes):
            center = size_idx * (n_strategies + 1)
            tick_positions.append(center + (n_strategies - 1) / 2 * width)
            tick_labels.append(f"{size}d")

            for strat_idx, strategy in enumerate(strategies):
                config_key = f"{strategy}_{size}"
                values, is_missing = _load_cluster_values(
                    h5_path, config_key, segment, metric)
                if is_missing:
                    if config_key not in missing:
                        missing.append(config_key)
                    continue
                finite = values[np.isfinite(values)]
                if len(finite) > 0:
                    positions.append(center + strat_idx * width)
                    box_data.append(finite.tolist())
                    colors.append(STRATEGY_COLORS[strategy])

        if box_data:
            bp = ax.boxplot(box_data, positions=positions, widths=width * 0.8,
                           patch_artist=True, showfliers=True,
                           medianprops=dict(color="black", linewidth=1.5),
                           flierprops=dict(marker=".", markersize=3, alpha=0.5))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_title(metric_label, fontsize=13)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_xlabel("Sample Size (×d)", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(bottom=0)

    legend_elements = [Patch(facecolor=STRATEGY_COLORS[s], alpha=0.7,
                             label=STRATEGY_LABELS[s]) for s in strategies]
    fig.legend(handles=legend_elements, loc="upper center", ncol=len(strategies),
               fontsize=10, bbox_to_anchor=(0.5, 1.02), frameon=True)

    seg_label = segment.replace("_", " ").title()
    fig.suptitle(f"{ft_label} Clustering Quality — Boxplots ({seg_label})",
                 fontsize=14, y=1.06)
    plt.tight_layout()
    plt.show()
    return missing


def plot_ela_box(h5_path, segment="all", omit_strategies=None):
    """ELA boxplots: 30 ARI/NMI values per config."""
    return _plot_box(h5_path, SAMPLE_SIZES_ELA, segment, "ELA", omit_strategies)


def plot_tla_box(h5_path, segment="all", omit_strategies=None):
    """TLA boxplots: 30 ARI/NMI values per config."""
    return _plot_box(h5_path, SAMPLE_SIZES_TLA, segment, "TLA", omit_strategies)


def plot_tla_box_all_segments(h5_path, omit_strategies=None):
    """TLA boxplots: all segments in a grid, one metric at a time."""
    strategies = _active_strategies(omit_strategies)
    missing = []
    n_strategies = len(strategies)
    width = 0.8 / n_strategies

    for metric, metric_label in CLUSTER_METRICS:
        ncols, nrows = 3, (len(TLA_SEGMENTS) + 2) // 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), sharey=True)
        axes = axes.flatten()

        for idx, segment in enumerate(TLA_SEGMENTS):
            ax = axes[idx]
            positions, box_data, colors = [], [], []
            tick_positions, tick_labels = [], []

            for size_idx, size in enumerate(SAMPLE_SIZES_TLA):
                center = size_idx * (n_strategies + 1)
                tick_positions.append(center + (n_strategies - 1) / 2 * width)
                tick_labels.append(f"{size}d")

                for strat_idx, strategy in enumerate(strategies):
                    config_key = f"{strategy}_{size}"
                    values, is_missing = _load_cluster_values(
                        h5_path, config_key, segment, metric)
                    if is_missing:
                        if config_key not in missing:
                            missing.append(config_key)
                        continue
                    finite = values[np.isfinite(values)]
                    if len(finite) > 0:
                        positions.append(center + strat_idx * width)
                        box_data.append(finite.tolist())
                        colors.append(STRATEGY_COLORS[strategy])

            if box_data:
                bp = ax.boxplot(box_data, positions=positions, widths=width * 0.8,
                               patch_artist=True, showfliers=False,
                               medianprops=dict(color="black", linewidth=1.5))
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.set_title(segment.replace("_", " ").title(), fontsize=11)
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylim(bottom=0)
            if idx % ncols == 0:
                ax.set_ylabel(metric_label, fontsize=10)
            if idx >= (nrows - 1) * ncols:
                ax.set_xlabel("Sample Size (×d)", fontsize=10)

        for idx in range(len(TLA_SEGMENTS), len(axes)):
            axes[idx].set_visible(False)

        legend_elements = [Patch(facecolor=STRATEGY_COLORS[s], alpha=0.7,
                                 label=STRATEGY_LABELS[s]) for s in strategies]
        fig.legend(handles=legend_elements, loc="upper center", ncol=len(strategies),
                   fontsize=10, bbox_to_anchor=(0.5, 1.02), frameon=True)
        fig.suptitle(f"TLA Clustering Quality — All Segments ({metric_label})",
                     fontsize=14, y=1.06)
        plt.tight_layout()
        plt.show()

    return missing


# ===========================================================================
# Combined ELA vs TLA
# ===========================================================================

def plot_combined_line(ela_h5_path, tla_h5_path,
                       ela_segment="all", tla_segment="all",
                       omit_strategies=None):
    """Side-by-side line plots: ELA (left) vs TLA (right), ARI and NMI as rows."""
    strategies = _active_strategies(omit_strategies)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for met_idx, (metric, metric_label) in enumerate(CLUSTER_METRICS):
        # ELA (left column)
        ax_ela = axes[met_idx, 0]
        for strategy in strategies:
            means, stds, sizes = [], [], []
            for size in SAMPLE_SIZES_ELA:
                values, ism = _load_cluster_values(
                    ela_h5_path, f"{strategy}_{size}", ela_segment, metric)
                if not ism:
                    finite = values[np.isfinite(values)]
                    if len(finite) > 0:
                        means.append(np.mean(finite))
                        stds.append(np.std(finite))
                        sizes.append(size)
            if sizes:
                ax_ela.errorbar(sizes, means, yerr=stds, marker="o", linewidth=2,
                                markersize=5, capsize=3,
                                color=STRATEGY_COLORS[strategy],
                                label=STRATEGY_LABELS[strategy])
        ax_ela.set_ylabel(metric_label, fontsize=12)
        ax_ela.set_xticks(SAMPLE_SIZES_ELA)
        ax_ela.set_xticklabels([f"{s}d" for s in SAMPLE_SIZES_ELA])
        ax_ela.grid(True, alpha=0.3)
        ax_ela.set_ylim(bottom=0)
        ax_ela.legend(frameon=True, fontsize=9)
        if met_idx == 0:
            ax_ela.set_title("ELA", fontsize=13)
        if met_idx == 1:
            ax_ela.set_xlabel("Sample Size (×d)", fontsize=12)

        # TLA (right column)
        ax_tla = axes[met_idx, 1]
        for strategy in strategies:
            means, stds, sizes = [], [], []
            for size in SAMPLE_SIZES_TLA:
                values, ism = _load_cluster_values(
                    tla_h5_path, f"{strategy}_{size}", tla_segment, metric)
                if not ism:
                    finite = values[np.isfinite(values)]
                    if len(finite) > 0:
                        means.append(np.mean(finite))
                        stds.append(np.std(finite))
                        sizes.append(size)
            if sizes:
                ax_tla.errorbar(sizes, means, yerr=stds, marker="o", linewidth=2,
                                markersize=5, capsize=3,
                                color=STRATEGY_COLORS[strategy],
                                label=STRATEGY_LABELS[strategy])
        seg_label = tla_segment.replace("_", " ").title()
        ax_tla.set_xticks(SAMPLE_SIZES_TLA)
        ax_tla.set_xticklabels([f"{s}d" for s in SAMPLE_SIZES_TLA])
        ax_tla.grid(True, alpha=0.3)
        ax_tla.set_ylim(bottom=0)
        ax_tla.legend(frameon=True, fontsize=9)
        if met_idx == 0:
            ax_tla.set_title(f"TLA ({seg_label})", fontsize=13)
        if met_idx == 1:
            ax_tla.set_xlabel("Sample Size (×d)", fontsize=12)

    fig.suptitle("Clustering Quality Comparison — ELA vs TLA", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_combined_box(ela_h5_path, tla_h5_path,
                      ela_segment="all", tla_segment="all",
                      omit_strategies=None):
    """Side-by-side boxplots: ELA (left) vs TLA (right), ARI and NMI as rows."""
    strategies = _active_strategies(omit_strategies)
    n_strategies = len(strategies)
    width = 0.8 / n_strategies
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for met_idx, (metric, metric_label) in enumerate(CLUSTER_METRICS):
        for col_idx, (h5_path, sample_sizes, ft_label, segment) in enumerate([
            (ela_h5_path, SAMPLE_SIZES_ELA, "ELA", ela_segment),
            (tla_h5_path, SAMPLE_SIZES_TLA, "TLA", tla_segment),
        ]):
            ax = axes[met_idx, col_idx]
            positions, box_data, colors = [], [], []
            tick_positions, tick_labels = [], []

            for size_idx, size in enumerate(sample_sizes):
                center = size_idx * (n_strategies + 1)
                tick_positions.append(center + (n_strategies - 1) / 2 * width)
                tick_labels.append(f"{size}d")

                for strat_idx, strategy in enumerate(strategies):
                    config_key = f"{strategy}_{size}"
                    values, is_missing = _load_cluster_values(
                        h5_path, config_key, segment, metric)
                    if is_missing:
                        continue
                    finite = values[np.isfinite(values)]
                    if len(finite) > 0:
                        positions.append(center + strat_idx * width)
                        box_data.append(finite.tolist())
                        colors.append(STRATEGY_COLORS[strategy])

            if box_data:
                bp = ax.boxplot(box_data, positions=positions, widths=width * 0.8,
                               patch_artist=True, showfliers=True,
                               medianprops=dict(color="black", linewidth=1.5),
                               flierprops=dict(marker=".", markersize=3, alpha=0.5))
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylim(bottom=0)
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=12)
            if met_idx == 0:
                seg_lbl = segment.replace("_", " ").title() if ft_label == "TLA" else ""
                title = f"{ft_label} ({seg_lbl})" if seg_lbl else ft_label
                ax.set_title(title, fontsize=13)
            if met_idx == 1:
                ax.set_xlabel("Sample Size (×d)", fontsize=12)

    legend_elements = [Patch(facecolor=STRATEGY_COLORS[s], alpha=0.7,
                             label=STRATEGY_LABELS[s]) for s in strategies]
    fig.legend(handles=legend_elements, loc="upper center", ncol=len(strategies),
               fontsize=10, bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle("Clustering Quality Comparison — ELA vs TLA", fontsize=14, y=1.06)
    plt.tight_layout()
    plt.show()


# ===========================================================================
# Missing summaries
# ===========================================================================

def _print_missing(h5_path, sample_sizes, segments, ft_label, omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    print(f"\n{'='*60}\n{ft_label} — CLUSTERING MISSING DATA SUMMARY\n{'='*60}")
    missing_configs = []
    configs_with_nans = []

    with h5py.File(h5_path, "r") as f:
        for strategy in strategies:
            for size in sample_sizes:
                config_key = f"{strategy}_{size}"
                if config_key not in f:
                    missing_configs.append(config_key)
                    continue
                config_grp = f[config_key]
                nan_count = 0
                for seg in segments:
                    if seg in config_grp:
                        for metric, _ in CLUSTER_METRICS:
                            if metric in config_grp[seg]:
                                vals = config_grp[seg][metric][:]
                                nan_count += int(np.sum(~np.isfinite(vals)))
                if nan_count > 0:
                    configs_with_nans.append((config_key, nan_count))

    if missing_configs:
        print(f"\nMissing configurations ({len(missing_configs)}):")
        for cfg in missing_configs:
            print(f"  - {cfg}")
    else:
        print("\nNo missing configurations.")

    if configs_with_nans:
        print(f"\nConfigurations with NaN/Inf values ({len(configs_with_nans)}):")
        for cfg, count in configs_with_nans:
            print(f"  - {cfg}: {count} NaN/Inf values")
    else:
        print("\nNo NaN/Inf values found.")
    print("=" * 60)


def print_ela_missing_summary(h5_path, omit_strategies=None):
    _print_missing(h5_path, SAMPLE_SIZES_ELA, ["all"], "ELA", omit_strategies)


def print_tla_missing_summary(h5_path, omit_strategies=None):
    _print_missing(h5_path, SAMPLE_SIZES_TLA, TLA_SEGMENTS, "TLA", omit_strategies)


# ===========================================================================
# Run-all
# ===========================================================================

def run_all_ela(h5_path, segment="all", omit_strategies=None):
    print("=" * 60 + "\nELA CLUSTERING PLOTS\n" + "=" * 60)
    print("\nLine plots (mean ± std)")
    plot_ela_line(h5_path, segment, omit_strategies)
    print("\nBoxplots")
    plot_ela_box(h5_path, segment, omit_strategies)
    print_ela_missing_summary(h5_path, omit_strategies)


def run_all_tla(h5_path, segment="all", omit_strategies=None):
    print("=" * 60 + "\nTLA CLUSTERING PLOTS\n" + "=" * 60)
    print("\nLine plots (mean ± std)")
    plot_tla_line(h5_path, segment, omit_strategies)
    print("\nBoxplots")
    plot_tla_box(h5_path, segment, omit_strategies)
    print("\nAll segments — line plots")
    plot_tla_line_all_segments(h5_path, omit_strategies)
    print("\nAll segments — boxplots")
    plot_tla_box_all_segments(h5_path, omit_strategies)
    print_tla_missing_summary(h5_path, omit_strategies)


def run_all(ela_h5, tla_h5, ela_segment="all", tla_segment="all", omit_strategies=None):
    run_all_ela(ela_h5, ela_segment, omit_strategies)
    print("\n\n")
    run_all_tla(tla_h5, tla_segment, omit_strategies)
    print("\n\nCombined comparison — line plots:")
    plot_combined_line(ela_h5, tla_h5, ela_segment, tla_segment, omit_strategies)
    print("\nCombined comparison — boxplots:")
    plot_combined_box(ela_h5, tla_h5, ela_segment, tla_segment, omit_strategies)