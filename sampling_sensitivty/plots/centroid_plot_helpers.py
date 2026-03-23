"""
Plot centroid-based distance results for ELA and TLA features.

All plot functions accept an optional `omit_strategies` parameter (set of strategy
names to exclude, e.g. {"cma_random"}).

Usage in notebook:
    from plot_centroid_dist import *
    ela_h5 = "path/to/ela_centroid_distances.h5"
    tla_h5 = "path/to/tla_centroids_distances.h5"
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

DISTANCE_METRICS = [
    ("euclidean_dist_to_mean", "Euclidean to Mean"),
    ("euclidean_dist_to_median", "Euclidean to Median"),
    ("cosine_dist_to_mean", "Cosine to Mean"),
    ("cosine_dist_to_median", "Cosine to Median"),
]


def _active_strategies(omit_strategies=None):
    omit = omit_strategies or set()
    return [s for s in SAMPLING_STRATEGIES if s not in omit]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_distances_from_h5(h5_path, config_key, segment, metric, func_ids=None):
    """Load per-instance median distances. Returns (instance_medians, nan_count, is_missing)."""
    try:
        with h5py.File(h5_path, "r") as f:
            if config_key not in f:
                return [], 0, True
            config_grp = f[config_key]
            instance_medians, nan_count = [], 0
            for func_id in range(1, N_FUNCTIONS + 1):
                if func_ids is not None and func_id not in func_ids:
                    continue
                for inst_id in range(1, N_INSTANCES + 1):
                    inst_key = f"{func_id}_{inst_id}_{DIMENSION}"
                    if inst_key not in config_grp: continue
                    inst_grp = config_grp[inst_key]
                    if segment not in inst_grp: continue
                    seg_grp = inst_grp[segment]
                    if metric not in seg_grp: continue
                    dists = seg_grp[metric][:]
                    finite_mask = np.isfinite(dists)
                    nan_count += int(np.sum(~finite_mask))
                    finite_dists = dists[finite_mask]
                    if len(finite_dists) > 0:
                        instance_medians.append(np.median(finite_dists))
            return instance_medians, nan_count, False
    except Exception as e:
        print(f"Error loading {config_key}/{segment}/{metric}: {e}")
        return [], 0, True


def _load_all_run_distances(h5_path, config_key, segment, metric, func_ids=None):
    """Load per-instance median distances for boxplots. Returns (list, is_missing)."""
    try:
        with h5py.File(h5_path, "r") as f:
            if config_key not in f:
                return [], True
            config_grp = f[config_key]
            all_dists = []
            for func_id in range(1, N_FUNCTIONS + 1):
                if func_ids is not None and func_id not in func_ids:
                    continue
                for inst_id in range(1, N_INSTANCES + 1):
                    inst_key = f"{func_id}_{inst_id}_{DIMENSION}"
                    if inst_key not in config_grp: continue
                    if segment not in config_grp[inst_key]: continue
                    if metric not in config_grp[inst_key][segment]: continue
                    dists = config_grp[inst_key][segment][metric][:]
                    finite_dists = dists[np.isfinite(dists)]
                    if len(finite_dists) > 0:
                        all_dists.append(np.median(finite_dists))
            return all_dists, False
    except Exception as e:
        print(f"Error: {e}")
        return [], True


def _aggregate_median_distance(h5_path, config_key, segment, metric, func_ids=None):
    """Returns (overall_median, nan_count, is_missing)."""
    instance_medians, nan_count, is_missing = _load_distances_from_h5(
        h5_path, config_key, segment, metric, func_ids)
    if is_missing or not instance_medians:
        return np.nan, nan_count, is_missing
    return np.median(instance_medians), nan_count, False


# ===========================================================================
# Line plots — Perspective 1
# ===========================================================================

def _plot_line_p1(h5_path, sample_sizes, segment, ft_label, omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    missing = []
    for ax_idx, (metric, metric_label) in enumerate(DISTANCE_METRICS):
        ax = axes[ax_idx]
        for strategy in strategies:
            medians, sizes = [], []
            for size in sample_sizes:
                config_key = f"{strategy}_{size}"
                med, _, is_missing = _aggregate_median_distance(h5_path, config_key, segment, metric)
                if is_missing:
                    if config_key not in missing: missing.append(config_key)
                    continue
                if not np.isnan(med): medians.append(med); sizes.append(size)
            if sizes:
                ax.plot(sizes, medians, marker="o", linewidth=2, markersize=6,
                        color=STRATEGY_COLORS[strategy], label=STRATEGY_LABELS[strategy])
        ax.set_title(metric_label, fontsize=11)
        ax.set_xticks(sample_sizes); ax.set_xticklabels([f"{s}d" for s in sample_sizes])
        ax.grid(True, alpha=0.3)
        if ax_idx >= 2: ax.set_xlabel("Sample Size (×d)", fontsize=10)
        if ax_idx % 2 == 0: ax.set_ylabel("Median Distance", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(strategies),
               fontsize=10, bbox_to_anchor=(0.5, 1.02), frameon=True)
    seg_label = segment.replace("_", " ").title()
    fig.suptitle(f"{ft_label} Centroid Distance — Overall ({seg_label})", fontsize=14, y=1.06)
    plt.tight_layout(); plt.show()
    return missing


def plot_ela_line_perspective1(h5_path, segment="all", omit_strategies=None):
    return _plot_line_p1(h5_path, SAMPLE_SIZES_ELA, segment, "ELA", omit_strategies)

def plot_tla_line_perspective1(h5_path, segment="all", omit_strategies=None):
    return _plot_line_p1(h5_path, SAMPLE_SIZES_TLA, segment, "TLA", omit_strategies)


# ===========================================================================
# Boxplots — Perspective 1
# ===========================================================================

def _plot_box_p1(h5_path, sample_sizes, segment, ft_label, omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    missing = []
    n_strategies = len(strategies)
    width = 0.8 / n_strategies
    for ax_idx, (metric, metric_label) in enumerate(DISTANCE_METRICS):
        ax = axes[ax_idx]
        positions, box_data, colors = [], [], []
        tick_positions, tick_labels = [], []
        for size_idx, size in enumerate(sample_sizes):
            center = size_idx * (n_strategies + 1)
            tick_positions.append(center + (n_strategies - 1) / 2 * width)
            tick_labels.append(f"{size}d")
            for strat_idx, strategy in enumerate(strategies):
                config_key = f"{strategy}_{size}"
                dists, is_missing = _load_all_run_distances(h5_path, config_key, segment, metric)
                if is_missing:
                    if config_key not in missing: missing.append(config_key)
                    continue
                if dists:
                    positions.append(center + strat_idx * width)
                    box_data.append(dists)
                    colors.append(STRATEGY_COLORS[strategy])
        if box_data:
            bp = ax.boxplot(box_data, positions=positions, widths=width * 0.8,
                           patch_artist=True, showfliers=False,
                           medianprops=dict(color="black", linewidth=1.5))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels)
        ax.set_title(metric_label, fontsize=11); ax.grid(True, alpha=0.3, axis="y")
        if ax_idx >= 2: ax.set_xlabel("Sample Size (×d)", fontsize=10)
        if ax_idx % 2 == 0: ax.set_ylabel("Per-Instance Median Distance", fontsize=10)
    legend_elements = [Patch(facecolor=STRATEGY_COLORS[s], alpha=0.7,
                             label=STRATEGY_LABELS[s]) for s in strategies]
    fig.legend(handles=legend_elements, loc="upper center", ncol=len(strategies),
               fontsize=10, bbox_to_anchor=(0.5, 1.02), frameon=True)
    seg_label = segment.replace("_", " ").title()
    fig.suptitle(f"{ft_label} Centroid Distance — Overall Boxplots ({seg_label})",
                 fontsize=14, y=1.06)
    plt.tight_layout(); plt.show()
    return missing


def plot_ela_box_perspective1(h5_path, segment="all", omit_strategies=None):
    return _plot_box_p1(h5_path, SAMPLE_SIZES_ELA, segment, "ELA", omit_strategies)

def plot_tla_box_perspective1(h5_path, segment="all", omit_strategies=None):
    return _plot_box_p1(h5_path, SAMPLE_SIZES_TLA, segment, "TLA", omit_strategies)


# ===========================================================================
# Line plots — Perspective 2
# ===========================================================================

def _plot_line_p2(h5_path, sample_sizes, segment, ft_label,
                  metric_key, metric_label, omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    n_groups = len(FUNCTION_GROUPS)
    fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 4.5), sharey=True)
    missing = []
    for idx, (group_name, func_ids) in enumerate(FUNCTION_GROUPS.items()):
        ax = axes[idx]
        for strategy in strategies:
            medians, sizes = [], []
            for size in sample_sizes:
                config_key = f"{strategy}_{size}"
                med, _, is_missing = _aggregate_median_distance(
                    h5_path, config_key, segment, metric_key, func_ids)
                if is_missing and config_key not in missing: missing.append(config_key); continue
                if not np.isnan(med): medians.append(med); sizes.append(size)
            if sizes:
                ax.plot(sizes, medians, marker="o", linewidth=2, markersize=5,
                        color=STRATEGY_COLORS[strategy], label=STRATEGY_LABELS[strategy])
        ax.set_xlabel("Sample Size (×d)", fontsize=10)
        if idx == 0: ax.set_ylabel("Median Distance", fontsize=10)
        ax.set_title(group_name, fontsize=11)
        ax.set_xticks(sample_sizes)
        ax.set_xticklabels([f"{s}d" for s in sample_sizes], fontsize=9)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(strategies),
               fontsize=10, bbox_to_anchor=(0.5, 1.02), frameon=True)
    seg_label = segment.replace("_", " ").title()
    fig.suptitle(f"{ft_label} — Per Function Group ({metric_label}, {seg_label})",
                 fontsize=14, y=1.08)
    plt.tight_layout(); plt.show()
    return missing


def plot_ela_line_perspective2(h5_path, segment="all", omit_strategies=None):
    missing = []
    for mk, ml in DISTANCE_METRICS:
        m = _plot_line_p2(h5_path, SAMPLE_SIZES_ELA, segment, "ELA", mk, ml, omit_strategies)
        missing.extend([x for x in m if x not in missing])
    return missing

def plot_tla_line_perspective2(h5_path, segment="all", omit_strategies=None):
    missing = []
    for mk, ml in DISTANCE_METRICS:
        m = _plot_line_p2(h5_path, SAMPLE_SIZES_TLA, segment, "TLA", mk, ml, omit_strategies)
        missing.extend([x for x in m if x not in missing])
    return missing


# ===========================================================================
# Boxplots — Perspective 2
# ===========================================================================

def _plot_box_p2(h5_path, sample_sizes, segment, ft_label,
                 metric_key, metric_label, omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    n_groups = len(FUNCTION_GROUPS)
    fig, axes = plt.subplots(1, n_groups, figsize=(4.5 * n_groups, 5), sharey=True)
    missing = []
    n_strategies = len(strategies)
    width = 0.8 / n_strategies
    for idx, (group_name, func_ids) in enumerate(FUNCTION_GROUPS.items()):
        ax = axes[idx]
        positions, box_data, colors = [], [], []
        tick_positions, tick_labels = [], []
        for size_idx, size in enumerate(sample_sizes):
            center = size_idx * (n_strategies + 1)
            tick_positions.append(center + (n_strategies - 1) / 2 * width)
            tick_labels.append(f"{size}d")
            for strat_idx, strategy in enumerate(strategies):
                config_key = f"{strategy}_{size}"
                dists, is_missing = _load_all_run_distances(
                    h5_path, config_key, segment, metric_key, func_ids)
                if is_missing and config_key not in missing: missing.append(config_key); continue
                if dists:
                    positions.append(center + strat_idx * width)
                    box_data.append(dists); colors.append(STRATEGY_COLORS[strategy])
        if box_data:
            bp = ax.boxplot(box_data, positions=positions, widths=width * 0.8,
                           patch_artist=True, showfliers=False,
                           medianprops=dict(color="black", linewidth=1.5))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_title(group_name, fontsize=11); ax.grid(True, alpha=0.3, axis="y")
        ax.set_xlabel("Sample Size (×d)", fontsize=10)
        if idx == 0: ax.set_ylabel("Per-Instance Median Distance", fontsize=10)
    legend_elements = [Patch(facecolor=STRATEGY_COLORS[s], alpha=0.7,
                             label=STRATEGY_LABELS[s]) for s in strategies]
    fig.legend(handles=legend_elements, loc="upper center", ncol=len(strategies),
               fontsize=10, bbox_to_anchor=(0.5, 1.02), frameon=True)
    seg_label = segment.replace("_", " ").title()
    fig.suptitle(f"{ft_label} — Per Function Group Boxplots ({metric_label}, {seg_label})",
                 fontsize=14, y=1.08)
    plt.tight_layout(); plt.show()
    return missing


def plot_ela_box_perspective2(h5_path, segment="all", omit_strategies=None):
    missing = []
    for mk, ml in DISTANCE_METRICS:
        m = _plot_box_p2(h5_path, SAMPLE_SIZES_ELA, segment, "ELA", mk, ml, omit_strategies)
        missing.extend([x for x in m if x not in missing])
    return missing

def plot_tla_box_perspective2(h5_path, segment="all", omit_strategies=None):
    missing = []
    for mk, ml in DISTANCE_METRICS:
        m = _plot_box_p2(h5_path, SAMPLE_SIZES_TLA, segment, "TLA", mk, ml, omit_strategies)
        missing.extend([x for x in m if x not in missing])
    return missing


# ===========================================================================
# Heatmap — Perspective 3
# ===========================================================================

def _plot_heatmap_p3(h5_path, sample_sizes, segment, ft_label,
                     metric_key, metric_label, omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    configs, config_labels = [], []
    for strategy in strategies:
        for size in sample_sizes:
            configs.append(f"{strategy}_{size}")
            config_labels.append(f"{STRATEGY_LABELS[strategy]}\n{size}d")
    heatmap = np.full((N_FUNCTIONS, len(configs)), np.nan)
    missing = []
    for col_idx, config_key in enumerate(configs):
        for func_idx in range(N_FUNCTIONS):
            med, _, is_missing = _aggregate_median_distance(
                h5_path, config_key, segment, metric_key, [func_idx + 1])
            if is_missing and config_key not in missing: missing.append(config_key)
            heatmap[func_idx, col_idx] = med

    fig, ax = plt.subplots(figsize=(max(16, len(config_labels) * 0.8), 8))
    masked = np.ma.masked_invalid(heatmap)
    cmap = plt.cm.RdYlGn_r; cmap.set_bad(color="lightgray")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(config_labels)))
    ax.set_xticklabels(config_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(N_FUNCTIONS))
    ax.set_yticklabels([f"F{i+1}" for i in range(N_FUNCTIONS)], fontsize=9)
    ax.set_xlabel("Configuration", fontsize=12); ax.set_ylabel("Function Class", fontsize=12)
    seg_label = segment.replace("_", " ").title()
    ax.set_title(f"{ft_label} Centroid Distance — Per Function Class ({metric_label}, {seg_label})",
                 fontsize=13)
    group_boundaries = [0]
    for group_name, func_ids in FUNCTION_GROUPS.items():
        group_boundaries.append(group_boundaries[-1] + len(func_ids))
    for boundary in group_boundaries[1:-1]:
        ax.axhline(y=boundary - 0.5, color="black", linewidth=1.5)
    for group_name, func_ids in FUNCTION_GROUPS.items():
        mid = (min(func_ids) - 1 + max(func_ids) - 1) / 2
        ax.text(len(config_labels) + 0.5, mid, group_name, fontsize=8, va="center", ha="left")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.15)
    cbar.set_label("Median Distance", fontsize=11)
    plt.tight_layout(); plt.show()
    return missing


def plot_ela_heatmap_perspective3(h5_path, segment="all", omit_strategies=None):
    missing = []
    for mk, ml in DISTANCE_METRICS:
        m = _plot_heatmap_p3(h5_path, SAMPLE_SIZES_ELA, segment, "ELA", mk, ml, omit_strategies)
        missing.extend([x for x in m if x not in missing])
    return missing

def plot_tla_heatmap_perspective3(h5_path, segment="all", omit_strategies=None):
    missing = []
    for mk, ml in DISTANCE_METRICS:
        m = _plot_heatmap_p3(h5_path, SAMPLE_SIZES_TLA, segment, "TLA", mk, ml, omit_strategies)
        missing.extend([x for x in m if x not in missing])
    return missing


# ===========================================================================
# Combined — Perspective 1
# ===========================================================================

def plot_perspective1_combined(ela_h5_path, tla_h5_path,
                                ela_segment="all", tla_segment="all",
                                omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for met_idx, (metric_key, metric_label) in enumerate(DISTANCE_METRICS):
        # ELA top row
        ax_ela = axes[0, met_idx]
        for strategy in strategies:
            medians, sizes = [], []
            for size in SAMPLE_SIZES_ELA:
                med, _, ism = _aggregate_median_distance(
                    ela_h5_path, f"{strategy}_{size}", ela_segment, metric_key)
                if not ism and not np.isnan(med): medians.append(med); sizes.append(size)
            if sizes:
                ax_ela.plot(sizes, medians, marker="o", linewidth=2, markersize=5,
                            color=STRATEGY_COLORS[strategy], label=STRATEGY_LABELS[strategy])
        ax_ela.set_title(metric_label, fontsize=10)
        ax_ela.set_xticks(SAMPLE_SIZES_ELA)
        ax_ela.set_xticklabels([f"{s}d" for s in SAMPLE_SIZES_ELA], fontsize=8)
        ax_ela.grid(True, alpha=0.3)
        if met_idx == 0: ax_ela.set_ylabel("ELA\nMedian Distance", fontsize=10)

        # TLA bottom row
        ax_tla = axes[1, met_idx]
        for strategy in strategies:
            medians, sizes = [], []
            for size in SAMPLE_SIZES_TLA:
                med, _, ism = _aggregate_median_distance(
                    tla_h5_path, f"{strategy}_{size}", tla_segment, metric_key)
                if not ism and not np.isnan(med): medians.append(med); sizes.append(size)
            if sizes:
                ax_tla.plot(sizes, medians, marker="o", linewidth=2, markersize=5,
                            color=STRATEGY_COLORS[strategy], label=STRATEGY_LABELS[strategy])
        ax_tla.set_xlabel("Sample Size (×d)", fontsize=10)
        ax_tla.set_xticks(SAMPLE_SIZES_TLA)
        ax_tla.set_xticklabels([f"{s}d" for s in SAMPLE_SIZES_TLA], fontsize=8)
        ax_tla.grid(True, alpha=0.3)
        if met_idx == 0: ax_tla.set_ylabel("TLA\nMedian Distance", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(strategies),
               fontsize=10, bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle("Centroid Distance Comparison — ELA vs TLA", fontsize=14, y=1.06)
    plt.tight_layout(); plt.show()


# ===========================================================================
# Missing summaries
# ===========================================================================

def _print_missing(h5_path, sample_sizes, ft_label, omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    print(f"\n{'='*60}\n{ft_label} — CENTROID DISTANCE MISSING DATA SUMMARY\n{'='*60}")
    missing_configs, configs_with_nans = [], []
    with h5py.File(h5_path, "r") as f:
        for strategy in strategies:
            for size in sample_sizes:
                config_key = f"{strategy}_{size}"
                if config_key not in f: missing_configs.append(config_key); continue
                nan_count = 0
                config_grp = f[config_key]
                for inst_key in list(config_grp.keys())[:10]:
                    if "all" in config_grp[inst_key]:
                        for mk, _ in DISTANCE_METRICS:
                            if mk in config_grp[inst_key]["all"]:
                                dists = config_grp[inst_key]["all"][mk][:]
                                nan_count += int(np.sum(~np.isfinite(dists)))
                if nan_count > 0: configs_with_nans.append((config_key, nan_count))
    if missing_configs:
        print(f"\nMissing configurations ({len(missing_configs)}):")
        for cfg in missing_configs: print(f"  - {cfg}")
    else: print("\nNo missing configurations.")
    if configs_with_nans:
        print(f"\nConfigurations with NaN/Inf values (sampled) ({len(configs_with_nans)}):")
        for cfg, count in configs_with_nans: print(f"  - {cfg}: ~{count} NaN/Inf (first 10 instances)")
    else: print("\nNo NaN/Inf values found.")
    print("=" * 60)


def print_ela_missing_summary(h5_path, omit_strategies=None):
    _print_missing(h5_path, SAMPLE_SIZES_ELA, "ELA", omit_strategies)

def print_tla_missing_summary(h5_path, omit_strategies=None):
    _print_missing(h5_path, SAMPLE_SIZES_TLA, "TLA", omit_strategies)


# ===========================================================================
# Run-all
# ===========================================================================

def run_all_ela(h5_path, segment="all", omit_strategies=None):
    print("=" * 60 + "\nELA CENTROID DISTANCE PLOTS\n" + "=" * 60)
    print("\nPerspective 1 — Line plots")
    plot_ela_line_perspective1(h5_path, segment, omit_strategies)
    print("\nPerspective 1 — Boxplots")
    plot_ela_box_perspective1(h5_path, segment, omit_strategies)
    print("\nPerspective 2 — Line plots per function group")
    plot_ela_line_perspective2(h5_path, segment, omit_strategies)
    print("\nPerspective 2 — Boxplots per function group")
    plot_ela_box_perspective2(h5_path, segment, omit_strategies)
    print("\nPerspective 3 — Heatmaps per function class")
    plot_ela_heatmap_perspective3(h5_path, segment, omit_strategies)
    print_ela_missing_summary(h5_path, omit_strategies)


def run_all_tla(h5_path, segment="all", omit_strategies=None):
    print("=" * 60 + "\nTLA CENTROID DISTANCE PLOTS\n" + "=" * 60)
    print("\nPerspective 1 — Line plots")
    plot_tla_line_perspective1(h5_path, segment, omit_strategies)
    print("\nPerspective 1 — Boxplots")
    plot_tla_box_perspective1(h5_path, segment, omit_strategies)
    print("\nPerspective 2 — Line plots per function group")
    plot_tla_line_perspective2(h5_path, segment, omit_strategies)
    print("\nPerspective 2 — Boxplots per function group")
    plot_tla_box_perspective2(h5_path, segment, omit_strategies)
    print("\nPerspective 3 — Heatmaps per function class")
    plot_tla_heatmap_perspective3(h5_path, segment, omit_strategies)
    print_tla_missing_summary(h5_path, omit_strategies)


def run_all(ela_h5, tla_h5, ela_segment="all", tla_segment="all", omit_strategies=None):
    run_all_ela(ela_h5, ela_segment, omit_strategies)
    print("\n\n")
    run_all_tla(tla_h5, tla_segment, omit_strategies)
    print("\n\nCombined comparison:")
    plot_perspective1_combined(ela_h5, tla_h5, ela_segment, tla_segment, omit_strategies)