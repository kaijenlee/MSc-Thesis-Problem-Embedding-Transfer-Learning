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

ELA_GROUPS = ["ela_dist", "meta", "nbc", "ic"]
TLA_TRANSFORMS = ["volume", "axis"]
TLA_HOMOLOGIES = ["h0", "h1", "h2"]
TLA_SEGMENTS = {
    "all": None, "volume_all": ("volume", None), "axis_all": ("axis", None),
    "volume_h0": ("volume", "h0"), "volume_h1": ("volume", "h1"),
    "volume_h2": ("volume", "h2"), "axis_h0": ("axis", "h0"),
    "axis_h1": ("axis", "h1"), "axis_h2": ("axis", "h2"),
}

# Mapping from feature name prefix to ELA group
_PREFIX_TO_GROUP = {
    "ela_distr": "ela_dist", "ela_meta": "meta",
    "disp.": "disp", "nbc.": "nbc", "ic.": "ic",
}

ELA_GROUP_COLORS = {
    "ela_dist": "#1f77b4", "meta": "#ff7f0e", "disp": "#2ca02c",
    "nbc": "#d62728", "ic": "#9467bd",
}


def _active_strategies(omit_strategies=None):
    omit = omit_strategies or set()
    return [s for s in SAMPLING_STRATEGIES if s not in omit]


def _feature_to_group(feat_name):
    """Map an ELA feature name to its group."""
    for prefix, grp in _PREFIX_TO_GROUP.items():
        if feat_name.startswith(prefix):
            return grp
    return None


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
# ELA CV per feature (broken down by F1–F24)
# ---------------------------------------------------------------------------

def ela_cv_by_feature(ela_cv, config_key, fn_range=range(1, N_FUNCTIONS+1), inst_range=range(1, N_INSTANCES + 1)):
    """
    Collect CV values separated by individual ELA feature, per function.

    Returns
    -------
    dict : {feature_name: {func_id: [cv_values]}}
        Nested dict. Outer key is the feature name (e.g. "ela_distr.skewness"),
        inner key is the function id (1–24), value is a list of CV values
        across the 100 instances.
    """
    if config_key not in ela_cv:
        return {}
    config_data = ela_cv[config_key]
    result = defaultdict(lambda: defaultdict(list))
    for func_id in range(1, N_FUNCTIONS + 1):
        for inst_id in range(1, N_INSTANCES + 1):
            key = (func_id, inst_id, DIMENSION)
            if key not in config_data:
                continue
            for grp_name in ELA_GROUPS:
                if grp_name not in config_data[key]:
                    continue
                for feat_name, cv_val in config_data[key][grp_name].items():
                    if not np.isnan(cv_val):
                        result[feat_name][func_id].append(cv_val)
    return dict(result)


def ela_cv_by_feature_group(ela_cv, config_key):
    """
    Collect CV values separated by ELA feature group, per function.

    Returns
    -------
    dict : {group_name: {func_id: [cv_values]}}
        Outer key is the feature group ("ela_dist", "meta", "disp", "nbc", "ic"),
        inner key is the function id (1–24), value is a list of CV values
        across all features in that group and 100 instances.
    """
    if config_key not in ela_cv:
        return {}
    config_data = ela_cv[config_key]
    result = defaultdict(lambda: defaultdict(list))
    for func_id in range(1, N_FUNCTIONS + 1):
        for inst_id in range(1, N_INSTANCES + 1):
            key = (func_id, inst_id, DIMENSION)
            if key not in config_data:
                continue
            for grp_name in ELA_GROUPS:
                if grp_name not in config_data[key]:
                    continue
                for feat_name, cv_val in config_data[key][grp_name].items():
                    if not np.isnan(cv_val):
                        result[grp_name][func_id].append(cv_val)
    return dict(result)


# ---------------------------------------------------------------------------
# Plot: per-feature boxplots across F1–F24
# ---------------------------------------------------------------------------

def plot_ela_cv_by_feature(ela_cv, config_key, features=None, figsize=None):
    """
    Boxplot of CV distributions per function (F1–F24) for each ELA feature.

    One subplot per feature. Each subplot has 24 boxes (one per function).

    Parameters
    ----------
    ela_cv : dict – loaded CV data
    config_key : str – e.g. "ilhs_50"
    features : list[str] or None – subset of features to plot; None = all
    figsize : tuple or None – override figure size
    """
    data = ela_cv_by_feature(ela_cv, config_key)
    if not data:
        print(f"No data for {config_key}"); return

    if features is not None:
        data = {k: v for k, v in data.items() if k in features}
    feat_names = sorted(data.keys())
    if not feat_names:
        print("No matching features found."); return

    n_feats = len(feat_names)
    ncols = min(3, n_feats)
    nrows = (n_feats + ncols - 1) // ncols
    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, feat_name in enumerate(feat_names):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        func_data = data[feat_name]
        positions = list(range(1, N_FUNCTIONS + 1))
        box_data = [func_data.get(fid, []) for fid in positions]

        grp = _feature_to_group(feat_name)
        color = ELA_GROUP_COLORS.get(grp, "gray")

        bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", linewidth=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(feat_name, fontsize=10)
        ax.set_xlabel("Function", fontsize=9)
        ax.set_ylabel("CV", fontsize=9)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(f) for f in positions], fontsize=7, rotation=90)
        ax.grid(True, alpha=0.3, axis="y")

        # Add function group separators
        boundaries = []
        cumulative = 0
        for grp_name, fids in FUNCTION_GROUPS.items():
            cumulative += len(fids)
            boundaries.append(cumulative + 0.5)
        for b in boundaries[:-1]:
            ax.axvline(x=b, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # Hide unused subplots
    for idx in range(n_feats, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"ELA Per-Feature CV by Function — {config_key}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Plot: per-feature-group boxplots across F1–F24
# ---------------------------------------------------------------------------

def plot_ela_cv_by_feature_group(ela_cv, config_key, groups=None, figsize=None):
    """
    Boxplot of CV distributions per function (F1–F24) for each ELA feature group.

    One subplot per group. Each subplot has 24 boxes (one per function).

    Parameters
    ----------
    ela_cv : dict – loaded CV data
    config_key : str – e.g. "ilhs_50"
    groups : list[str] or None – subset of groups to plot; None = all ELA_GROUPS
    figsize : tuple or None – override figure size
    """
    data = ela_cv_by_feature_group(ela_cv, config_key)
    if not data:
        print(f"No data for {config_key}"); return

    grp_list = groups if groups is not None else ELA_GROUPS
    grp_list = [g for g in grp_list if g in data]
    if not grp_list:
        print("No matching feature groups found."); return

    n_groups = len(grp_list)
    ncols = min(3, n_groups)
    nrows = (n_groups + ncols - 1) // ncols
    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, grp_name in enumerate(grp_list):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        func_data = data[grp_name]
        positions = list(range(1, N_FUNCTIONS + 1))
        box_data = [func_data.get(fid, []) for fid in positions]

        color = ELA_GROUP_COLORS.get(grp_name, "gray")

        bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", linewidth=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(grp_name, fontsize=11)
        ax.set_xlabel("Function", fontsize=9)
        ax.set_ylabel("CV", fontsize=9)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(f) for f in positions], fontsize=7, rotation=90)
        ax.grid(True, alpha=0.3, axis="y")

        # Function group separators
        boundaries = []
        cumulative = 0
        for gn, fids in FUNCTION_GROUPS.items():
            cumulative += len(fids)
            boundaries.append(cumulative + 0.5)
        for b in boundaries[:-1]:
            ax.axvline(x=b, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    for idx in range(n_groups, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"ELA Per-Feature-Group CV by Function — {config_key}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Plot: heatmap of median CV — features × functions
# ---------------------------------------------------------------------------

def plot_ela_feature_function_heatmap(ela_cv, config_key, figsize=None):
    """
    Heatmap: rows = ELA features, columns = F1–F24.
    Cell value = median CV across 100 instances.
    Features are grouped by ELA group with separators.
    """
    data = ela_cv_by_feature(ela_cv, config_key)
    if not data:
        print(f"No data for {config_key}"); return

    # Sort features by group, then alphabetically within group
    ordered_features = []
    group_boundaries = []
    for grp_name in ELA_GROUPS:
        grp_feats = sorted([f for f in data if _feature_to_group(f) == grp_name])
        if grp_feats:
            ordered_features.extend(grp_feats)
            group_boundaries.append(len(ordered_features))

    if not ordered_features:
        print("No features found."); return

    n_feats = len(ordered_features)
    heatmap = np.full((n_feats, N_FUNCTIONS), np.nan)
    for row_idx, feat_name in enumerate(ordered_features):
        func_data = data[feat_name]
        for func_id in range(1, N_FUNCTIONS + 1):
            if func_id in func_data and func_data[func_id]:
                heatmap[row_idx, func_id - 1] = np.median(func_data[func_id])

    if figsize is None:
        figsize = (14, max(8, n_feats * 0.35))

    fig, ax = plt.subplots(figsize=figsize)
    masked = np.ma.masked_invalid(heatmap)
    cmap = plt.cm.RdYlGn_r
    cmap.set_bad(color="lightgray")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_xticks(range(N_FUNCTIONS))
    ax.set_xticklabels([f"F{i+1}" for i in range(N_FUNCTIONS)], fontsize=8)
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(ordered_features, fontsize=7)
    ax.set_xlabel("Function", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(f"ELA Feature × Function Median CV — {config_key}", fontsize=14)

    # Group separators (horizontal)
    for b in group_boundaries[:-1]:
        ax.axhline(y=b - 0.5, color="black", linewidth=1.5)

    # Function group separators (vertical)
    cumulative = 0
    for gn, fids in FUNCTION_GROUPS.items():
        cumulative += len(fids)
        if cumulative < N_FUNCTIONS:
            ax.axvline(x=cumulative - 0.5, color="black", linewidth=1)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Median CV", fontsize=11)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Plot: heatmap of median CV — feature groups × functions
# ---------------------------------------------------------------------------

def plot_ela_feature_group_function_heatmap(ela_cv, config_key, figsize=None):
    """
    Heatmap: rows = ELA feature groups, columns = F1–F24.
    Cell value = median CV across all features in the group and 100 instances.
    """
    data = ela_cv_by_feature_group(ela_cv, config_key)
    if not data:
        print(f"No data for {config_key}"); return

    grp_list = [g for g in ELA_GROUPS if g in data]
    if not grp_list:
        print("No feature groups found."); return

    heatmap = np.full((len(grp_list), N_FUNCTIONS), np.nan)
    for row_idx, grp_name in enumerate(grp_list):
        func_data = data[grp_name]
        for func_id in range(1, N_FUNCTIONS + 1):
            if func_id in func_data and func_data[func_id]:
                heatmap[row_idx, func_id - 1] = np.median(func_data[func_id])

    if figsize is None:
        figsize = (14, max(4, len(grp_list) * 1.2))

    fig, ax = plt.subplots(figsize=figsize)
    masked = np.ma.masked_invalid(heatmap)
    cmap = plt.cm.RdYlGn_r
    cmap.set_bad(color="lightgray")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_xticks(range(N_FUNCTIONS))
    ax.set_xticklabels([f"F{i+1}" for i in range(N_FUNCTIONS)], fontsize=9)
    ax.set_yticks(range(len(grp_list)))
    ax.set_yticklabels(grp_list, fontsize=10)
    ax.set_xlabel("Function", fontsize=12)
    ax.set_ylabel("Feature Group", fontsize=12)
    ax.set_title(f"ELA Feature Group × Function Median CV — {config_key}", fontsize=14)

    # Function group separators
    cumulative = 0
    for gn, fids in FUNCTION_GROUPS.items():
        cumulative += len(fids)
        if cumulative < N_FUNCTIONS:
            ax.axvline(x=cumulative - 0.5, color="black", linewidth=1)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Median CV", fontsize=11)
    plt.tight_layout()
    plt.show()


# ===========================================================================
# ELA plots (original)
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


def plot_ela_per_feature(ela_cv, sample_size=50, strategy="ilhs",
                        config_keys=None, figsize=None):
    """
    Bar chart of median CV per ELA feature, with support for comparing
    multiple configurations side by side.

    Single configuration (original behaviour):
        plot_ela_per_feature(ela_cv, sample_size=50, strategy="ilhs")

    Compare multiple configurations:
        plot_ela_per_feature(ela_cv, config_keys=["cma_random_25", "cma_random_50",
                                                   "cma_random_75", "cma_random_100"])

    Parameters
    ----------
    ela_cv : dict – loaded CV data
    sample_size : int – used when config_keys is None
    strategy : str – used when config_keys is None
    config_keys : list[str] or None – explicit list of config keys to compare.
        When provided, sample_size and strategy are ignored.
    figsize : tuple or None – override figure size
    """
    # Build list of configs to plot
    if config_keys is None:
        config_keys = [f"{strategy}_{sample_size}"]

    # Collect per-feature medians for each config
    config_data = {}
    for ck in config_keys:
        feature_cvs = _ela_collect_per_feature_cvs(ela_cv, ck)
        if not feature_cvs:
            print(f"No data for {ck}")
            continue
        config_data[ck] = {feat: np.median(vals)
                           for feat, vals in feature_cvs.items() if vals}

    if not config_data:
        print("No data for any configuration."); return

    # Determine feature order: sorted by group, then alphabetically
    all_feat_names = sorted(
        set().union(*(d.keys() for d in config_data.values()))
    )
    ordered_features = []
    for grp_name in ELA_GROUPS:
        for feat_name in all_feat_names:
            if _feature_to_group(feat_name) == grp_name:
                ordered_features.append(feat_name)
    if not ordered_features:
        print("No features matched any group."); return

    n_configs = len(config_data)
    n_features = len(ordered_features)
    configs_list = list(config_data.keys())

    # Colour logic
    if n_configs == 1:
        # Single config: colour by feature group (original behaviour)
        use_group_colors = True
    else:
        # Multiple configs: each config gets a distinct colour
        use_group_colors = False
        base_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                        "#bcbd22", "#17becf"]
        config_colors = {ck: base_palette[i % len(base_palette)]
                         for i, ck in enumerate(configs_list)}

    # Bar geometry
    total_bar_width = 0.8
    bar_width = total_bar_width / n_configs
    x = np.arange(n_features)

    if figsize is None:
        figsize = (max(14, n_features * 0.6), 6)
    fig, ax = plt.subplots(figsize=figsize)

    for ci, ck in enumerate(configs_list):
        medians = [config_data[ck].get(f, np.nan) for f in ordered_features]
        offset = (ci - (n_configs - 1) / 2) * bar_width
        if use_group_colors:
            colors = [ELA_GROUP_COLORS.get(_feature_to_group(f), "gray")
                      for f in ordered_features]
            ax.bar(x + offset, medians, width=bar_width, color=colors,
                   edgecolor="white", linewidth=0.5)
        else:
            ax.bar(x + offset, medians, width=bar_width,
                   color=config_colors[ck], edgecolor="white",
                   linewidth=0.5, label=ck)

    ax.set_xticks(x)
    ax.set_xticklabels(ordered_features, rotation=90, fontsize=8, ha="center")
    ax.set_ylabel("Median CV", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Add feature-group separator lines
    group_boundaries = []
    count = 0
    for grp_name in ELA_GROUPS:
        grp_feats = [f for f in ordered_features if _feature_to_group(f) == grp_name]
        count += len(grp_feats)
        group_boundaries.append(count)
    for b in group_boundaries[:-1]:
        ax.axvline(x=b - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # Legend
    if use_group_colors:
        ax.legend(handles=[Patch(facecolor=ELA_GROUP_COLORS[g], label=g)
                           for g in ELA_GROUPS],
                  loc="upper right", fontsize=9)
        ck = configs_list[0]
        parts = ck.split("_")
        strat = "_".join(parts[:-1])
        sz = parts[-1]
        label = STRATEGY_LABELS.get(strat, strat)
        ax.set_title(f"ELA Per-Feature Stability — {label} {sz}d", fontsize=14)
    else:
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title("ELA Per-Feature Stability — Configuration Comparison", fontsize=14)

    plt.tight_layout(); plt.show()


# ===========================================================================
# TLA plots
# ===========================================================================

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


# ---------------------------------------------------------------------------
# TLA per-segment / per-transform / per-homology bar charts
# ---------------------------------------------------------------------------

# Segment display config
_TLA_SEGMENT_LABELS = {
    "volume_h0": "Vol H0", "volume_h1": "Vol H1", "volume_h2": "Vol H2",
    "axis_h0": "Axis H0", "axis_h1": "Axis H1", "axis_h2": "Axis H2",
    "volume_all": "Volume (all)", "axis_all": "Axis (all)",
    "all": "All",
}

TLA_TRANSFORM_COLORS = {"volume": "#1f77b4", "axis": "#ff7f0e"}
TLA_HOMOLOGY_COLORS = {"h0": "#2ca02c", "h1": "#d62728", "h2": "#9467bd"}

# The 6 atomic segments (transform × homology)
_TLA_ATOMIC_SEGMENTS = [
    "volume_h0", "volume_h1", "volume_h2",
    "axis_h0", "axis_h1", "axis_h2",
]


def _tla_segment_median_cv(tla_cv, config_key, transform=None, homology=None):
    """Median CV for a single config+segment, across all functions/instances."""
    med, _, is_missing = _tla_aggregate_median_cv(
        tla_cv, config_key, transform=transform, homology=homology)
    return med


def plot_tla_per_segment(tla_cv, sample_size=50, strategy="ilhs",
                         config_keys=None, segments=None, figsize=None):
    """
    Bar chart of median CV per TLA segment, with multi-config comparison.

    Single config (coloured by transform):
        plot_tla_per_segment(tla_cv, sample_size=50, strategy="cma_random")

    Compare multiple configs:
        plot_tla_per_segment(tla_cv, config_keys=[
            "cma_random_25", "cma_random_50", "cma_random_75", "cma_random_100"])

    Parameters
    ----------
    tla_cv : dict – loaded TLA CV data
    sample_size : int – used when config_keys is None
    strategy : str – used when config_keys is None
    config_keys : list[str] or None – explicit configs to compare
    segments : list[str] or None – which segments to show (default: 6 atomic)
    figsize : tuple or None
    """
    if config_keys is None:
        config_keys = [f"{strategy}_{sample_size}"]

    if segments is None:
        segments = _TLA_ATOMIC_SEGMENTS

    # Collect median CV per segment per config
    config_data = {}
    for ck in config_keys:
        if ck not in tla_cv:
            print(f"No data for {ck}"); continue
        seg_medians = {}
        for seg in segments:
            if seg == "all":
                t, h = None, None
            else:
                spec = TLA_SEGMENTS.get(seg)
                if spec is None:
                    continue
                t, h = spec
            med = _tla_segment_median_cv(tla_cv, ck, transform=t, homology=h)
            if not np.isnan(med):
                seg_medians[seg] = med
        if seg_medians:
            config_data[ck] = seg_medians

    if not config_data:
        print("No data for any configuration."); return

    configs_list = list(config_data.keys())
    n_configs = len(configs_list)
    seg_labels = [_TLA_SEGMENT_LABELS.get(s, s) for s in segments]
    n_segs = len(segments)

    # Colour logic
    if n_configs == 1:
        # Single config: colour by transform
        use_segment_colors = True
    else:
        use_segment_colors = False
        base_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                        "#bcbd22", "#17becf"]
        config_colors = {ck: base_palette[i % len(base_palette)]
                         for i, ck in enumerate(configs_list)}

    total_bar_width = 0.8
    bar_width = total_bar_width / n_configs
    x = np.arange(n_segs)

    if figsize is None:
        figsize = (max(10, n_segs * 1.5), 5)
    fig, ax = plt.subplots(figsize=figsize)

    for ci, ck in enumerate(configs_list):
        medians = [config_data[ck].get(s, np.nan) for s in segments]
        offset = (ci - (n_configs - 1) / 2) * bar_width
        if use_segment_colors:
            colors = []
            for s in segments:
                spec = TLA_SEGMENTS.get(s)
                if spec and spec[0]:
                    colors.append(TLA_TRANSFORM_COLORS.get(spec[0], "gray"))
                else:
                    colors.append("gray")
            ax.bar(x + offset, medians, width=bar_width, color=colors,
                   edgecolor="white", linewidth=0.5)
        else:
            ax.bar(x + offset, medians, width=bar_width,
                   color=config_colors[ck], edgecolor="white",
                   linewidth=0.5, label=ck)

    ax.set_xticks(x)
    ax.set_xticklabels(seg_labels, fontsize=10)
    ax.set_ylabel("Median CV", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Separator between volume and axis segments
    vol_count = sum(1 for s in segments if s.startswith("volume"))
    if 0 < vol_count < n_segs:
        ax.axvline(x=vol_count - 0.5, color="gray", linewidth=0.8,
                   linestyle="--", alpha=0.5)

    if use_segment_colors:
        ax.legend(handles=[
            Patch(facecolor=TLA_TRANSFORM_COLORS["volume"], label="Volume"),
            Patch(facecolor=TLA_TRANSFORM_COLORS["axis"], label="Axis"),
        ], loc="upper right", fontsize=9)
        ck = configs_list[0]
        parts = ck.split("_")
        strat = "_".join(parts[:-1])
        sz = parts[-1]
        label = STRATEGY_LABELS.get(strat, strat)
        ax.set_title(f"TLA Per-Segment Stability — {label} {sz}d", fontsize=14)
    else:
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title("TLA Per-Segment Stability — Configuration Comparison", fontsize=14)

    plt.tight_layout(); plt.show()


def plot_tla_per_transform(tla_cv, sample_size=50, strategy="ilhs",
                           config_keys=None, figsize=None):
    """
    Bar chart of median CV per transform (volume, axis), with multi-config
    comparison support.

    Single config:
        plot_tla_per_transform(tla_cv, sample_size=50, strategy="cma_random")

    Compare:
        plot_tla_per_transform(tla_cv, config_keys=["cma_random_25", "cma_random_50"])
    """
    if config_keys is None:
        config_keys = [f"{strategy}_{sample_size}"]

    config_data = {}
    for ck in config_keys:
        if ck not in tla_cv:
            print(f"No data for {ck}"); continue
        t_medians = {}
        for t in TLA_TRANSFORMS:
            med = _tla_segment_median_cv(tla_cv, ck, transform=t, homology=None)
            if not np.isnan(med):
                t_medians[t] = med
        if t_medians:
            config_data[ck] = t_medians

    if not config_data:
        print("No data for any configuration."); return

    configs_list = list(config_data.keys())
    n_configs = len(configs_list)
    transforms = TLA_TRANSFORMS
    n_t = len(transforms)

    if n_configs == 1:
        use_transform_colors = True
    else:
        use_transform_colors = False
        base_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        config_colors = {ck: base_palette[i % len(base_palette)]
                         for i, ck in enumerate(configs_list)}

    total_bar_width = 0.6
    bar_width = total_bar_width / n_configs
    x = np.arange(n_t)

    if figsize is None:
        figsize = (max(6, n_t * 2), 5)
    fig, ax = plt.subplots(figsize=figsize)

    for ci, ck in enumerate(configs_list):
        medians = [config_data[ck].get(t, np.nan) for t in transforms]
        offset = (ci - (n_configs - 1) / 2) * bar_width
        if use_transform_colors:
            colors = [TLA_TRANSFORM_COLORS.get(t, "gray") for t in transforms]
            ax.bar(x + offset, medians, width=bar_width, color=colors,
                   edgecolor="white", linewidth=0.5)
        else:
            ax.bar(x + offset, medians, width=bar_width,
                   color=config_colors[ck], edgecolor="white",
                   linewidth=0.5, label=ck)

    ax.set_xticks(x)
    ax.set_xticklabels([t.title() for t in transforms], fontsize=11)
    ax.set_ylabel("Median CV", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    if use_transform_colors:
        ck = configs_list[0]
        parts = ck.split("_"); strat = "_".join(parts[:-1]); sz = parts[-1]
        label = STRATEGY_LABELS.get(strat, strat)
        ax.set_title(f"TLA Per-Transform Stability — {label} {sz}d", fontsize=14)
    else:
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title("TLA Per-Transform Stability — Configuration Comparison", fontsize=14)

    plt.tight_layout(); plt.show()


def plot_tla_per_homology(tla_cv, sample_size=50, strategy="ilhs",
                          config_keys=None, figsize=None):
    """
    Bar chart of median CV per homology dimension (h0, h1, h2), with
    multi-config comparison support.

    Single config:
        plot_tla_per_homology(tla_cv, sample_size=50, strategy="cma_random")

    Compare:
        plot_tla_per_homology(tla_cv, config_keys=["cma_random_25", "cma_random_50"])
    """
    if config_keys is None:
        config_keys = [f"{strategy}_{sample_size}"]

    config_data = {}
    for ck in config_keys:
        if ck not in tla_cv:
            print(f"No data for {ck}"); continue
        h_medians = {}
        for h in TLA_HOMOLOGIES:
            med = _tla_segment_median_cv(tla_cv, ck, transform=None, homology=h)
            if not np.isnan(med):
                h_medians[h] = med
        if h_medians:
            config_data[ck] = h_medians

    if not config_data:
        print("No data for any configuration."); return

    configs_list = list(config_data.keys())
    n_configs = len(configs_list)
    homologies = TLA_HOMOLOGIES
    n_h = len(homologies)

    if n_configs == 1:
        use_homology_colors = True
    else:
        use_homology_colors = False
        base_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        config_colors = {ck: base_palette[i % len(base_palette)]
                         for i, ck in enumerate(configs_list)}

    total_bar_width = 0.6
    bar_width = total_bar_width / n_configs
    x = np.arange(n_h)

    if figsize is None:
        figsize = (max(6, n_h * 2), 5)
    fig, ax = plt.subplots(figsize=figsize)

    for ci, ck in enumerate(configs_list):
        medians = [config_data[ck].get(h, np.nan) for h in homologies]
        offset = (ci - (n_configs - 1) / 2) * bar_width
        if use_homology_colors:
            colors = [TLA_HOMOLOGY_COLORS.get(h, "gray") for h in homologies]
            ax.bar(x + offset, medians, width=bar_width, color=colors,
                   edgecolor="white", linewidth=0.5)
        else:
            ax.bar(x + offset, medians, width=bar_width,
                   color=config_colors[ck], edgecolor="white",
                   linewidth=0.5, label=ck)

    ax.set_xticks(x)
    ax.set_xticklabels([h.upper() for h in homologies], fontsize=11)
    ax.set_ylabel("Median CV", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    if use_homology_colors:
        ck = configs_list[0]
        parts = ck.split("_"); strat = "_".join(parts[:-1]); sz = parts[-1]
        label = STRATEGY_LABELS.get(strat, strat)
        ax.set_title(f"TLA Per-Homology Stability — {label} {sz}d", fontsize=14)
    else:
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title("TLA Per-Homology Stability — Configuration Comparison", fontsize=14)

    plt.tight_layout(); plt.show()


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