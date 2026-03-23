"""
Plot CV results for ELA and TLA features.

All plot functions accept an optional `omit_strategies` parameter (set of strategy
names to exclude, e.g. {"cma_random"}).

Usage in notebook:
    from plot_cv import *
    ela_cv = load_cv_data("path/to/ela_cv_results.pkl")
    tla_cv = load_cv_data("path/to/tla_cv.pkl")
    run_all(ela_cv, tla_cv, omit_strategies={"cma_random"})
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
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

ELA_GROUPS = ["ela_dist", "meta", "disp", "nbc", "ic"]
TLA_TRANSFORMS = ["volume", "axis"]
TLA_HOMOLOGIES = ["h0", "h1", "h2"]
TLA_SEGMENTS = {
    "all": None, "volume_all": ("volume", None), "axis_all": ("axis", None),
    "volume_h0": ("volume", "h0"), "volume_h1": ("volume", "h1"),
    "volume_h2": ("volume", "h2"), "axis_h0": ("axis", "h0"),
    "axis_h1": ("axis", "h1"), "axis_h2": ("axis", "h2"),
}


def _active_strategies(omit_strategies=None):
    omit = omit_strategies or set()
    return [s for s in SAMPLING_STRATEGIES if s not in omit]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cv_data(cv_path):
    with open(cv_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# ELA CV helpers
# ---------------------------------------------------------------------------

def _ela_collect_per_feature_cvs(ela_cv, config_key, func_ids=None):
    if config_key not in ela_cv:
        return {}
    config_data = ela_cv[config_key]
    feature_cvs = defaultdict(list)
    for func_id in range(1, N_FUNCTIONS + 1):
        if func_ids is not None and func_id not in func_ids:
            continue
        for inst_id in range(1, N_INSTANCES + 1):
            key = (func_id, inst_id, DIMENSION)
            if key not in config_data:
                continue
            for grp_name in ELA_GROUPS:
                if grp_name not in config_data[key]:
                    continue
                for feat_name, cv_val in config_data[key][grp_name].items():
                    if not np.isnan(cv_val):
                        feature_cvs[feat_name].append(cv_val)
    return feature_cvs


def _ela_collect_all_cvs(ela_cv, config_key, func_ids=None):
    if config_key not in ela_cv:
        return [], 0, True
    config_data = ela_cv[config_key]
    cvs, nan_count = [], 0
    for func_id in range(1, N_FUNCTIONS + 1):
        if func_ids is not None and func_id not in func_ids:
            continue
        for inst_id in range(1, N_INSTANCES + 1):
            key = (func_id, inst_id, DIMENSION)
            if key not in config_data:
                continue
            for grp_name in ELA_GROUPS:
                if grp_name not in config_data[key]:
                    continue
                for feat_name, cv_val in config_data[key][grp_name].items():
                    if np.isnan(cv_val):
                        nan_count += 1
                    else:
                        cvs.append(cv_val)
    return cvs, nan_count, False


def _ela_aggregate_median_cv(ela_cv, config_key, func_ids=None):
    feature_cvs = _ela_collect_per_feature_cvs(ela_cv, config_key, func_ids)
    if not feature_cvs:
        return np.nan, 0, config_key not in ela_cv
    feature_medians = [np.median(cvs) for cvs in feature_cvs.values() if cvs]
    if not feature_medians:
        return np.nan, 0, False
    _, nan_count, _ = _ela_collect_all_cvs(ela_cv, config_key, func_ids)
    return np.median(feature_medians), nan_count, False


# ---------------------------------------------------------------------------
# TLA CV helpers
# ---------------------------------------------------------------------------

def _tla_get_instance_cvs(tla_cv, config_key, func_id, inst_id,
                           transform=None, homology=None):
    key = (func_id, inst_id, DIMENSION)
    if key not in tla_cv.get(config_key, {}):
        return [], 0
    instance_data = tla_cv[config_key][key]
    cvs, nan_count = [], 0
    transforms = [transform] if transform else TLA_TRANSFORMS
    homologies_list = [homology] if homology else TLA_HOMOLOGIES
    for t in transforms:
        if t not in instance_data:
            continue
        for h in homologies_list:
            if h not in instance_data[t] or instance_data[t][h] is None:
                continue
            arr = instance_data[t][h].flatten()
            finite_mask = np.isfinite(arr)
            nan_count += int(np.sum(~finite_mask))
            cvs.extend(arr[finite_mask].tolist())
    return cvs, nan_count


def _tla_aggregate_median_cv(tla_cv, config_key, func_ids=None,
                              transform=None, homology=None):
    if config_key not in tla_cv:
        return np.nan, 0, True
    instance_medians, total_nans = [], 0
    for func_id in range(1, N_FUNCTIONS + 1):
        if func_ids is not None and func_id not in func_ids:
            continue
        for inst_id in range(1, N_INSTANCES + 1):
            cvs, nc = _tla_get_instance_cvs(tla_cv, config_key, func_id, inst_id,
                                             transform, homology)
            total_nans += nc
            if cvs:
                instance_medians.append(np.median(cvs))
    if not instance_medians:
        return np.nan, total_nans, False
    return np.median(instance_medians), total_nans, False


def _tla_collect_all_cvs(tla_cv, config_key, func_ids=None,
                          transform=None, homology=None):
    if config_key not in tla_cv:
        return [], 0, True
    all_cvs, total_nans = [], 0
    for func_id in range(1, N_FUNCTIONS + 1):
        if func_ids is not None and func_id not in func_ids:
            continue
        for inst_id in range(1, N_INSTANCES + 1):
            cvs, nc = _tla_get_instance_cvs(tla_cv, config_key, func_id, inst_id,
                                             transform, homology)
            total_nans += nc
            all_cvs.extend(cvs)
    return all_cvs, total_nans, False


# ===========================================================================
# ELA plots
# ===========================================================================

def plot_ela_perspective1(ela_cv, omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    fig, ax = plt.subplots(figsize=(8, 5))
    missing = []
    for strategy in strategies:
        medians, sizes = [], []
        for size in SAMPLE_SIZES_ELA:
            config_key = f"{strategy}_{size}"
            med, _, is_missing = _ela_aggregate_median_cv(ela_cv, config_key)
            if is_missing:
                missing.append(config_key)
                continue
            if not np.isnan(med):
                medians.append(med); sizes.append(size)
        if sizes:
            ax.plot(sizes, medians, marker="o", linewidth=2, markersize=6,
                    color=STRATEGY_COLORS[strategy], label=STRATEGY_LABELS[strategy])
    ax.set_xlabel("Sample Size (×d)", fontsize=12)
    ax.set_ylabel("Median CV", fontsize=12)
    ax.set_title("ELA Feature Stability — Overall", fontsize=14)
    ax.set_xticks(SAMPLE_SIZES_ELA)
    ax.set_xticklabels([f"{s}d" for s in SAMPLE_SIZES_ELA])
    ax.legend(frameon=True, fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
    return missing


def plot_ela_perspective2(ela_cv, omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    n_groups = len(FUNCTION_GROUPS)
    fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 4.5), sharey=True)
    missing = []
    for idx, (group_name, func_ids) in enumerate(FUNCTION_GROUPS.items()):
        ax = axes[idx]
        for strategy in strategies:
            medians, sizes = [], []
            for size in SAMPLE_SIZES_ELA:
                config_key = f"{strategy}_{size}"
                med, _, is_missing = _ela_aggregate_median_cv(ela_cv, config_key, func_ids)
                if is_missing:
                    if config_key not in missing: missing.append(config_key)
                    continue
                if not np.isnan(med):
                    medians.append(med); sizes.append(size)
            if sizes:
                ax.plot(sizes, medians, marker="o", linewidth=2, markersize=5,
                        color=STRATEGY_COLORS[strategy], label=STRATEGY_LABELS[strategy])
        ax.set_xlabel("Sample Size (×d)", fontsize=10)
        if idx == 0: ax.set_ylabel("Median CV", fontsize=10)
        ax.set_title(group_name, fontsize=11)
        ax.set_xticks(SAMPLE_SIZES_ELA)
        ax.set_xticklabels([f"{s}d" for s in SAMPLE_SIZES_ELA], fontsize=9)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(strategies),
               fontsize=10, bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle("ELA Feature Stability — Per Function Group", fontsize=14, y=1.08)
    plt.tight_layout(); plt.show()
    return missing


def plot_ela_perspective3(ela_cv, omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    configs, config_labels = [], []
    for strategy in strategies:
        for size in SAMPLE_SIZES_ELA:
            configs.append(f"{strategy}_{size}")
            config_labels.append(f"{STRATEGY_LABELS[strategy]}\n{size}d")
    heatmap = np.full((N_FUNCTIONS, len(configs)), np.nan)
    missing = []
    for col_idx, config_key in enumerate(configs):
        for func_idx in range(N_FUNCTIONS):
            med, _, is_missing = _ela_aggregate_median_cv(ela_cv, config_key, [func_idx + 1])
            if is_missing and config_key not in missing: missing.append(config_key)
            heatmap[func_idx, col_idx] = med
    _plot_heatmap(heatmap, config_labels, "ELA Feature Stability — Per Function Class (Median CV)")
    return missing


def plot_ela_per_feature(ela_cv, sample_size=50, strategy="ilhs"):
    config_key = f"{strategy}_{sample_size}"
    feature_cvs = _ela_collect_per_feature_cvs(ela_cv, config_key)
    if not feature_cvs:
        print(f"No data for {config_key}"); return
    group_color_map = {"ela_dist": "#1f77b4", "meta": "#ff7f0e", "disp": "#2ca02c",
                       "nbc": "#d62728", "ic": "#9467bd"}
    prefix_to_group = {"ela_distr": "ela_dist", "ela_meta": "meta",
                       "disp.": "disp", "nbc.": "nbc", "ic.": "ic"}
    all_names, all_medians, all_colors = [], [], []
    for grp_name in ELA_GROUPS:
        for feat_name in sorted(feature_cvs.keys()):
            for prefix, grp in prefix_to_group.items():
                if feat_name.startswith(prefix) and grp == grp_name:
                    all_names.append(feat_name)
                    all_medians.append(np.median(feature_cvs[feat_name]))
                    all_colors.append(group_color_map.get(grp_name, "gray"))
                    break
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(all_names)), all_medians, color=all_colors, edgecolor="white")
    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels(all_names, rotation=90, fontsize=8, ha="center")
    ax.set_ylabel("Median CV", fontsize=12)
    ax.set_title(f"ELA Per-Feature Stability — {STRATEGY_LABELS[strategy]} {sample_size}d", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(handles=[Patch(facecolor=group_color_map[g], label=g) for g in ELA_GROUPS],
              loc="upper right", fontsize=9)
    plt.tight_layout(); plt.show()


# ===========================================================================
# TLA plots
# ===========================================================================

def plot_tla_perspective1(tla_cv, segment="all", omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    if segment == "all":
        transform, homology = None, None
    else:
        spec = TLA_SEGMENTS.get(segment)
        if spec is None: print(f"Unknown segment: {segment}"); return []
        transform, homology = spec
    fig, ax = plt.subplots(figsize=(8, 5))
    missing = []
    for strategy in strategies:
        medians, sizes = [], []
        for size in SAMPLE_SIZES_TLA:
            config_key = f"{strategy}_{size}"
            med, _, is_missing = _tla_aggregate_median_cv(tla_cv, config_key,
                                                           transform=transform, homology=homology)
            if is_missing: missing.append(config_key); continue
            if not np.isnan(med): medians.append(med); sizes.append(size)
        if sizes:
            ax.plot(sizes, medians, marker="o", linewidth=2, markersize=6,
                    color=STRATEGY_COLORS[strategy], label=STRATEGY_LABELS[strategy])
    seg_label = segment.replace("_", " ").title()
    ax.set_xlabel("Sample Size (×d)", fontsize=12)
    ax.set_ylabel("Median CV", fontsize=12)
    ax.set_title(f"TLA Feature Stability — Overall ({seg_label})", fontsize=14)
    ax.set_xticks(SAMPLE_SIZES_TLA)
    ax.set_xticklabels([f"{s}d" for s in SAMPLE_SIZES_TLA])
    ax.legend(frameon=True, fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
    return missing


def plot_tla_perspective1_all_segments(tla_cv, omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    segments = list(TLA_SEGMENTS.keys())
    ncols, nrows = 3, (len(segments) + 2) // 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes = axes.flatten()
    missing = []
    for idx, segment in enumerate(segments):
        ax = axes[idx]
        if segment == "all": transform, homology = None, None
        else: transform, homology = TLA_SEGMENTS[segment]
        for strategy in strategies:
            medians, sizes = [], []
            for size in SAMPLE_SIZES_TLA:
                config_key = f"{strategy}_{size}"
                med, _, is_missing = _tla_aggregate_median_cv(
                    tla_cv, config_key, transform=transform, homology=homology)
                if is_missing and config_key not in missing: missing.append(config_key); continue
                if not np.isnan(med): medians.append(med); sizes.append(size)
            if sizes:
                ax.plot(sizes, medians, marker="o", linewidth=1.5, markersize=4,
                        color=STRATEGY_COLORS[strategy], label=STRATEGY_LABELS[strategy])
        ax.set_title(segment.replace("_", " ").title(), fontsize=11)
        ax.set_xticks(SAMPLE_SIZES_TLA)
        ax.set_xticklabels([f"{s}d" for s in SAMPLE_SIZES_TLA], fontsize=8)
        ax.grid(True, alpha=0.3)
        if idx % ncols == 0: ax.set_ylabel("Median CV", fontsize=10)
        if idx >= (nrows - 1) * ncols: ax.set_xlabel("Sample Size (×d)", fontsize=10)
    for idx in range(len(segments), len(axes)): axes[idx].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(strategies),
               fontsize=10, bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle("TLA Feature Stability — All Segments", fontsize=14, y=1.06)
    plt.tight_layout(); plt.show()
    return missing


def plot_tla_perspective2(tla_cv, segment="all", omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    if segment == "all": transform, homology = None, None
    else:
        spec = TLA_SEGMENTS.get(segment)
        if spec is None: print(f"Unknown segment: {segment}"); return []
        transform, homology = spec
    n_groups = len(FUNCTION_GROUPS)
    fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 4.5), sharey=True)
    missing = []
    for idx, (group_name, func_ids) in enumerate(FUNCTION_GROUPS.items()):
        ax = axes[idx]
        for strategy in strategies:
            medians, sizes = [], []
            for size in SAMPLE_SIZES_TLA:
                config_key = f"{strategy}_{size}"
                med, _, is_missing = _tla_aggregate_median_cv(
                    tla_cv, config_key, func_ids, transform=transform, homology=homology)
                if is_missing and config_key not in missing: missing.append(config_key); continue
                if not np.isnan(med): medians.append(med); sizes.append(size)
            if sizes:
                ax.plot(sizes, medians, marker="o", linewidth=2, markersize=5,
                        color=STRATEGY_COLORS[strategy], label=STRATEGY_LABELS[strategy])
        ax.set_xlabel("Sample Size (×d)", fontsize=10)
        if idx == 0: ax.set_ylabel("Median CV", fontsize=10)
        ax.set_title(group_name, fontsize=11)
        ax.set_xticks(SAMPLE_SIZES_TLA)
        ax.set_xticklabels([f"{s}d" for s in SAMPLE_SIZES_TLA], fontsize=9)
        ax.grid(True, alpha=0.3)
    seg_label = segment.replace("_", " ").title()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(strategies),
               fontsize=10, bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle(f"TLA Feature Stability — Per Function Group ({seg_label})", fontsize=14, y=1.08)
    plt.tight_layout(); plt.show()
    return missing


def plot_tla_perspective3(tla_cv, segment="all", omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    if segment == "all": transform, homology = None, None
    else:
        spec = TLA_SEGMENTS.get(segment)
        if spec is None: print(f"Unknown segment: {segment}"); return []
        transform, homology = spec
    configs, config_labels = [], []
    for strategy in strategies:
        for size in SAMPLE_SIZES_TLA:
            configs.append(f"{strategy}_{size}")
            config_labels.append(f"{STRATEGY_LABELS[strategy]}\n{size}d")
    heatmap = np.full((N_FUNCTIONS, len(configs)), np.nan)
    missing = []
    for col_idx, config_key in enumerate(configs):
        for func_idx in range(N_FUNCTIONS):
            med, _, is_missing = _tla_aggregate_median_cv(
                tla_cv, config_key, [func_idx + 1], transform=transform, homology=homology)
            if is_missing and config_key not in missing: missing.append(config_key)
            heatmap[func_idx, col_idx] = med
    seg_label = segment.replace("_", " ").title()
    _plot_heatmap(heatmap, config_labels, f"TLA Feature Stability — Per Function Class ({seg_label})")
    return missing


# ===========================================================================
# Combined
# ===========================================================================

def plot_perspective1_combined(ela_cv, tla_cv, tla_segments=None, omit_strategies=None):
    """
    Side-by-side: ELA (first panel) + one panel per TLA segment.

    tla_segments: str, list of str, or None.
        - None or "all": single TLA panel with all features
        - str: single TLA panel for that segment
        - list: one TLA panel per segment

    Examples:
        plot_perspective1_combined(ela_cv, tla_cv)
        plot_perspective1_combined(ela_cv, tla_cv, tla_segments="volume_h0")
        plot_perspective1_combined(ela_cv, tla_cv, tla_segments=["all", "volume_h0", "axis_h0"])
    """
    strategies = _active_strategies(omit_strategies)

    # Normalize tla_segments to a list
    if tla_segments is None:
        tla_segments = ["all"]
    elif isinstance(tla_segments, str):
        tla_segments = [tla_segments]

    n_panels = 1 + len(tla_segments)  # ELA + TLA segments
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(5 * n_panels, 5),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    # ELA panel (always first)
    ax_ela = axes[0]
    for strategy in strategies:
        medians, sizes = [], []
        for size in SAMPLE_SIZES_ELA:
            med, _, is_missing = _ela_aggregate_median_cv(ela_cv, f"{strategy}_{size}")
            if not is_missing and not np.isnan(med):
                medians.append(med); sizes.append(size)
        if sizes:
            ax_ela.plot(sizes, medians, marker="o", linewidth=2, markersize=6,
                        color=STRATEGY_COLORS[strategy], label=STRATEGY_LABELS[strategy])
    ax_ela.set_xlabel("Sample Size (×d)", fontsize=12)
    ax_ela.set_ylabel("Median CV", fontsize=12)
    ax_ela.set_title("ELA", fontsize=13)
    ax_ela.set_xticks(SAMPLE_SIZES_ELA)
    ax_ela.set_xticklabels([f"{s}d" for s in SAMPLE_SIZES_ELA])
    ax_ela.legend(frameon=True, fontsize=10)
    ax_ela.grid(True, alpha=0.3)

    # TLA panels
    for panel_idx, segment in enumerate(tla_segments):
        ax = axes[1 + panel_idx]

        if segment == "all":
            transform, homology = None, None
        else:
            spec = TLA_SEGMENTS.get(segment)
            if spec is None:
                print(f"Unknown segment: {segment}")
                continue
            transform, homology = spec

        for strategy in strategies:
            medians, sizes = [], []
            for size in SAMPLE_SIZES_TLA:
                med, _, is_missing = _tla_aggregate_median_cv(
                    tla_cv, f"{strategy}_{size}",
                    transform=transform, homology=homology)
                if not is_missing and not np.isnan(med):
                    medians.append(med); sizes.append(size)
            if sizes:
                ax.plot(sizes, medians, marker="o", linewidth=2, markersize=6,
                        color=STRATEGY_COLORS[strategy],
                        label=STRATEGY_LABELS[strategy])

        seg_label = segment.replace("_", " ").title()
        ax.set_xlabel("Sample Size (×d)", fontsize=12)
        ax.set_title(f"TLA ({seg_label})", fontsize=13)
        ax.set_xticks(SAMPLE_SIZES_TLA)
        ax.set_xticklabels([f"{s}d" for s in SAMPLE_SIZES_TLA])
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Feature Stability Comparison — ELA vs TLA", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# ===========================================================================
# Shared heatmap
# ===========================================================================

def _plot_heatmap(heatmap, config_labels, title):
    fig, ax = plt.subplots(figsize=(max(16, len(config_labels) * 0.8), 8))
    masked = np.ma.masked_invalid(heatmap)
    cmap = plt.cm.RdYlGn_r; cmap.set_bad(color="lightgray")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(config_labels)))
    ax.set_xticklabels(config_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(N_FUNCTIONS))
    ax.set_yticklabels([f"F{i+1}" for i in range(N_FUNCTIONS)], fontsize=9)
    ax.set_xlabel("Configuration", fontsize=12); ax.set_ylabel("Function Class", fontsize=12)
    ax.set_title(title, fontsize=14)
    group_boundaries = [0]
    for group_name, func_ids in FUNCTION_GROUPS.items():
        group_boundaries.append(group_boundaries[-1] + len(func_ids))
    for boundary in group_boundaries[1:-1]:
        ax.axhline(y=boundary - 0.5, color="black", linewidth=1.5)
    for group_name, func_ids in FUNCTION_GROUPS.items():
        mid = (min(func_ids) - 1 + max(func_ids) - 1) / 2
        ax.text(len(config_labels) + 0.5, mid, group_name, fontsize=8, va="center", ha="left")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.15)
    cbar.set_label("Median CV", fontsize=11)
    plt.tight_layout(); plt.show()


# ===========================================================================
# Missing summaries
# ===========================================================================

def print_ela_missing_summary(ela_cv, omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    print(f"\n{'='*60}\nELA — MISSING DATA SUMMARY\n{'='*60}")
    missing_configs, configs_with_nans = [], []
    for strategy in strategies:
        for size in SAMPLE_SIZES_ELA:
            config_key = f"{strategy}_{size}"
            if config_key not in ela_cv: missing_configs.append(config_key); continue
            _, nan_count, _ = _ela_collect_all_cvs(ela_cv, config_key)
            if nan_count > 0: configs_with_nans.append((config_key, nan_count))
    if missing_configs:
        print(f"\nMissing configurations ({len(missing_configs)}):")
        for cfg in missing_configs: print(f"  - {cfg}")
    else: print("\nNo missing configurations.")
    if configs_with_nans:
        print(f"\nConfigurations with NaN CV values ({len(configs_with_nans)}):")
        for cfg, count in configs_with_nans: print(f"  - {cfg}: {count} NaN values")
    else: print("\nNo NaN CV values found.")
    print("=" * 60)


def print_tla_missing_summary(tla_cv, omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    print(f"\n{'='*60}\nTLA — MISSING DATA SUMMARY\n{'='*60}")
    missing_configs, configs_with_nans = [], []
    for strategy in strategies:
        for size in SAMPLE_SIZES_TLA:
            config_key = f"{strategy}_{size}"
            if config_key not in tla_cv: missing_configs.append(config_key); continue
            _, nan_count, _ = _tla_collect_all_cvs(tla_cv, config_key)
            if nan_count > 0: configs_with_nans.append((config_key, nan_count))
    if missing_configs:
        print(f"\nMissing configurations ({len(missing_configs)}):")
        for cfg in missing_configs: print(f"  - {cfg}")
    else: print("\nNo missing configurations.")
    if configs_with_nans:
        print(f"\nConfigurations with NaN CV values ({len(configs_with_nans)}):")
        for cfg, count in configs_with_nans: print(f"  - {cfg}: {count} NaN values")
    else: print("\nNo NaN CV values found.")
    print("=" * 60)


# ===========================================================================
# Run-all
# ===========================================================================

def run_all_ela(ela_cv, omit_strategies=None):
    print("=" * 60 + "\nELA CV PLOTS\n" + "=" * 60)
    plot_ela_perspective1(ela_cv, omit_strategies)
    plot_ela_perspective2(ela_cv, omit_strategies)
    plot_ela_perspective3(ela_cv, omit_strategies)
    plot_ela_per_feature(ela_cv)
    print_ela_missing_summary(ela_cv, omit_strategies)


def run_all_tla(tla_cv, omit_strategies=None):
    print("=" * 60 + "\nTLA CV PLOTS\n" + "=" * 60)
    plot_tla_perspective1(tla_cv, omit_strategies=omit_strategies)
    plot_tla_perspective1_all_segments(tla_cv, omit_strategies)
    plot_tla_perspective2(tla_cv, omit_strategies=omit_strategies)
    plot_tla_perspective3(tla_cv, omit_strategies=omit_strategies)
    print_tla_missing_summary(tla_cv, omit_strategies)


def run_all(ela_cv, tla_cv, omit_strategies=None):
    run_all_ela(ela_cv, omit_strategies)
    print("\n\n")
    run_all_tla(tla_cv, omit_strategies)
    print("\n\nCombined comparison:")
    plot_perspective1_combined(ela_cv, tla_cv, omit_strategies=omit_strategies)