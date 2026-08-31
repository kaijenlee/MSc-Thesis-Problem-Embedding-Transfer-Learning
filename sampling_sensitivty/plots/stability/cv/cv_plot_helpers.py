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

SAMPLING_STRATEGIES = ["ilhs", "lhs", "sobol", "uniform", "cma_random", "lhs_random_cd"]
SAMPLE_SIZES_ELA = [25, 50, 75, 100]
SAMPLE_SIZES_TLA = [10, 25, 50, 75, 100]

STRATEGY_COLORS = {
    "ilhs": "#1f77b4", "lhs": "#ff7f0e", "sobol": "#2ca02c",
    "uniform": "#d62728", "cma_random": "#9467bd",
    "lhs_random_cd": "#8c564b",
}

STRATEGY_LABELS = {
    "ilhs": "iLHS", "lhs": "LHS", "sobol": "Sobol",
    "uniform": "Uniform", "cma_random": "CMA-Random",
    "lhs_random_cd": "LHS-Random-CD",
}

FUNCTION_GROUPS = {
    "Separable": [1, 2, 3, 4, 5],
    "Low/Moderate Cond.": [6, 7, 8, 9],
    "High Conditioning": [10, 11, 12, 13, 14],
    "Multimodal (adequate)": [15, 16, 17, 18, 19],
    "Multimodal (weak)": [20, 21, 22, 23, 24],
}

ELA_GROUPS = ["ela_dist", "meta", "nbc", "ic", "disp"]
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

def ela_cv_by_feature(ela_cv, config_key, fn_range=range(1, N_FUNCTIONS + 1), inst_range=range(1, N_INSTANCES + 1)):
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
        print(f"No data for {config_key}");
        return

    if features is not None:
        data = {k: v for k, v in data.items() if k in features}
    feat_names = sorted(data.keys())
    if not feat_names:
        print("No matching features found.");
        return

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
        print(f"No data for {config_key}");
        return

    grp_list = groups if groups is not None else ELA_GROUPS
    grp_list = [g for g in grp_list if g in data]
    if not grp_list:
        print("No matching feature groups found.");
        return

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
        print(f"No data for {config_key}");
        return

    # Sort features by group, then alphabetically within group
    ordered_features = []
    group_boundaries = []
    for grp_name in ELA_GROUPS:
        grp_feats = sorted([f for f in data if _feature_to_group(f) == grp_name])
        if grp_feats:
            ordered_features.extend(grp_feats)
            group_boundaries.append(len(ordered_features))

    if not ordered_features:
        print("No features found.");
        return

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
    ax.set_xticklabels([f"F{i + 1}" for i in range(N_FUNCTIONS)], fontsize=8)
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
        print(f"No data for {config_key}");
        return

    grp_list = [g for g in ELA_GROUPS if g in data]
    if not grp_list:
        print("No feature groups found.");
        return

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
    ax.set_xticklabels([f"F{i + 1}" for i in range(N_FUNCTIONS)], fontsize=9)
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
                medians.append(med);
                sizes.append(size)
        if sizes:
            ax.plot(sizes, medians, marker="o", linewidth=2, markersize=6,
                    color=STRATEGY_COLORS[strategy], label=STRATEGY_LABELS[strategy])
    ax.set_xlabel("Sample Size (×d)", fontsize=12)
    ax.set_ylabel("Median CV", fontsize=12)
    ax.set_title("ELA Feature Stability — Overall", fontsize=14)
    ax.set_xticks(SAMPLE_SIZES_ELA)
    ax.set_xticklabels([f"{s}d" for s in SAMPLE_SIZES_ELA])
    ax.legend(frameon=True, fontsize=10);
    ax.grid(True, alpha=0.3)
    plt.tight_layout();
    plt.show()
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
                    medians.append(med);
                    sizes.append(size)
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
    plt.tight_layout();
    plt.show()
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
        print("No data for any configuration.");
        return

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
        print("No features matched any group.");
        return

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

    plt.tight_layout();
    plt.show()


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
    ax.legend(frameon=True, fontsize=10);
    ax.grid(True, alpha=0.3)
    plt.tight_layout();
    plt.show()
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
        if segment == "all":
            transform, homology = None, None
        else:
            transform, homology = TLA_SEGMENTS[segment]
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
    plt.tight_layout();
    plt.show()
    return missing


def plot_tla_perspective2(tla_cv, segment="all", omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    if segment == "all":
        transform, homology = None, None
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
    plt.tight_layout();
    plt.show()
    return missing


def plot_tla_perspective3(tla_cv, segment="all", omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    if segment == "all":
        transform, homology = None, None
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
            print(f"No data for {ck}");
            continue
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
        print("No data for any configuration.");
        return

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

    plt.tight_layout();
    plt.show()


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
            print(f"No data for {ck}");
            continue
        t_medians = {}
        for t in TLA_TRANSFORMS:
            med = _tla_segment_median_cv(tla_cv, ck, transform=t, homology=None)
            if not np.isnan(med):
                t_medians[t] = med
        if t_medians:
            config_data[ck] = t_medians

    if not config_data:
        print("No data for any configuration.");
        return

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
        parts = ck.split("_");
        strat = "_".join(parts[:-1]);
        sz = parts[-1]
        label = STRATEGY_LABELS.get(strat, strat)
        ax.set_title(f"TLA Per-Transform Stability — {label} {sz}d", fontsize=14)
    else:
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title("TLA Per-Transform Stability — Configuration Comparison", fontsize=14)

    plt.tight_layout();
    plt.show()


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
            print(f"No data for {ck}");
            continue
        h_medians = {}
        for h in TLA_HOMOLOGIES:
            med = _tla_segment_median_cv(tla_cv, ck, transform=None, homology=h)
            if not np.isnan(med):
                h_medians[h] = med
        if h_medians:
            config_data[ck] = h_medians

    if not config_data:
        print("No data for any configuration.");
        return

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
        parts = ck.split("_");
        strat = "_".join(parts[:-1]);
        sz = parts[-1]
        label = STRATEGY_LABELS.get(strat, strat)
        ax.set_title(f"TLA Per-Homology Stability — {label} {sz}d", fontsize=14)
    else:
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title("TLA Per-Homology Stability — Configuration Comparison", fontsize=14)

    plt.tight_layout();
    plt.show()


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
                medians.append(med);
                sizes.append(size)
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
                    medians.append(med);
                    sizes.append(size)
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
    cmap = plt.cm.RdYlGn_r;
    cmap.set_bad(color="lightgray")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(config_labels)))
    ax.set_xticklabels(config_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(N_FUNCTIONS))
    ax.set_yticklabels([f"F{i + 1}" for i in range(N_FUNCTIONS)], fontsize=9)
    ax.set_xlabel("Configuration", fontsize=12);
    ax.set_ylabel("Function Class", fontsize=12)
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
    plt.tight_layout();
    plt.show()


# ===========================================================================
# Missing summaries
# ===========================================================================

def print_ela_missing_summary(ela_cv, omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    print(f"\n{'=' * 60}\nELA — MISSING DATA SUMMARY\n{'=' * 60}")
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
    else:
        print("\nNo missing configurations.")
    if configs_with_nans:
        print(f"\nConfigurations with NaN CV values ({len(configs_with_nans)}):")
        for cfg, count in configs_with_nans: print(f"  - {cfg}: {count} NaN values")
    else:
        print("\nNo NaN CV values found.")
    print("=" * 60)


def print_tla_missing_summary(tla_cv, omit_strategies=None):
    strategies = _active_strategies(omit_strategies)
    print(f"\n{'=' * 60}\nTLA — MISSING DATA SUMMARY\n{'=' * 60}")
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
    else:
        print("\nNo missing configurations.")
    if configs_with_nans:
        print(f"\nConfigurations with NaN CV values ({len(configs_with_nans)}):")
        for cfg, count in configs_with_nans: print(f"  - {cfg}: {count} NaN values")
    else:
        print("\nNo NaN CV values found.")
    print("=" * 60)


# ============================================================================
# 1.  Enhanced perspective-1 for ELA
# ============================================================================

def _ela_aggregate_cv_extended(ela_cv, config_key, func_ids=None,
                               agg_func="median",
                               inner_agg=None,
                               outer_agg=None,
                               exclude_features=None,
                               exclude_groups=None):
    """
    Aggregate CV values for a single config key.

    Returns (aggregated_value, nan_count, is_missing).

    Parameters
    ----------
    agg_func : "median" | "mean"
        Shorthand: sets both inner and outer aggregation to the same function.
        Ignored for whichever of ``inner_agg`` / ``outer_agg`` are set.
    inner_agg : "median" | "mean" | None
        How to collapse raw CV values *within* each feature (step 2).
        Overrides ``agg_func`` for this step when provided.
    outer_agg : "median" | "mean" | None
        How to collapse the per-feature summaries *across* features (step 3).
        Overrides ``agg_func`` for this step when provided.
    exclude_features : set[str] | None
        Individual feature names (e.g. "ela_distr.skewness") to skip.
    exclude_groups : set[str] | None
        ELA group names (e.g. "meta", "ic") whose features are all skipped.
    """
    exclude_features = exclude_features or set()
    exclude_groups = exclude_groups or set()
    _inner = inner_agg or agg_func
    _outer = outer_agg or agg_func
    inner_fn = np.median if _inner == "median" else np.mean
    outer_fn = np.median if _outer == "median" else np.mean

    if config_key not in ela_cv:
        return np.nan, 0, True

    config_data = ela_cv[config_key]
    feature_values = defaultdict(list)  # feat_name -> [cv_values]
    nan_count = 0

    for func_id in range(1, N_FUNCTIONS + 1):
        if func_ids is not None and func_id not in func_ids:
            continue
        for inst_id in range(1, N_INSTANCES + 1):
            key = (func_id, inst_id, DIMENSION)
            if key not in config_data:
                continue
            for grp_name in ELA_GROUPS:
                if grp_name in exclude_groups:
                    continue
                if grp_name not in config_data[key]:
                    continue
                for feat_name, cv_val in config_data[key][grp_name].items():
                    if feat_name in exclude_features:
                        continue
                    if np.isnan(cv_val):
                        nan_count += 1
                    else:
                        feature_values[feat_name].append(cv_val)

    if not feature_values:
        return np.nan, nan_count, False

    # Step 2: collapse within each feature using inner_fn
    per_feature_summary = [inner_fn(vals) for vals in feature_values.values() if vals]
    # Step 3: collapse across features using outer_fn
    return outer_fn(per_feature_summary), nan_count, False


def plot_ela_perspective1_extended(ela_cv,
                                   omit_strategies=None,
                                   agg_func="median",
                                   inner_agg=None,
                                   outer_agg=None,
                                   exclude_features=None,
                                   exclude_groups=None,
                                   func_ids=None,
                                   figsize=(8, 5)):
    """
    Overall ELA feature-stability line plot with configurable aggregation
    and optional feature / group exclusions.

    Parameters
    ----------
    ela_cv : dict
        Loaded CV data.
    omit_strategies : set[str] | None
        Sampling strategies to hide (e.g. {"cma_random"}).
    agg_func : "median" | "mean"
        Shorthand that sets both inner and outer aggregation to the same
        function.  Ignored for whichever of ``inner_agg`` / ``outer_agg``
        are explicitly provided.
    inner_agg : "median" | "mean" | None
        How to collapse raw CV values *within* each feature (step 2).
        Overrides ``agg_func`` for this step.
    outer_agg : "median" | "mean" | None
        How to collapse the per-feature summaries *across* features (step 3).
        Overrides ``agg_func`` for this step.
        Example: ``inner_agg="median", outer_agg="mean"`` → mean of
        per-feature medians (what the original line plot would show if you
        want mixed aggregation).
    exclude_features : set[str] | None
        Feature names to exclude from aggregation entirely
        (e.g. {"ela_distr.skewness", "ic.eps.s"}).
    exclude_groups : set[str] | None
        ELA group names to exclude entirely
        (e.g. {"meta", "ic"}).
        Valid group names: "ela_dist", "meta", "disp", "nbc", "ic".
    func_ids : list[int] | None
        Restrict to a subset of function IDs (1-24). None = all.
    figsize : tuple
        Figure size.

    Returns
    -------
    list[str]
        Config keys that were not found in ela_cv.

    Examples
    --------
    # default — identical to original plot_ela_perspective1
    plot_ela_perspective1_extended(ela_cv)

    # use mean instead of median
    plot_ela_perspective1_extended(ela_cv, agg_func="mean")

    # exclude the "meta" and "ic" groups
    plot_ela_perspective1_extended(ela_cv, exclude_groups={"meta", "ic"})

    # exclude specific features
    plot_ela_perspective1_extended(
        ela_cv,
        exclude_features={"ela_distr.skewness", "nbc.nn_nb.sd_ratio"}
    )

    # combine: mean, no meta group, omit cma_random strategy
    plot_ela_perspective1_extended(
        ela_cv,
        agg_func="mean",
        exclude_groups={"meta"},
        omit_strategies={"cma_random"},
    )
    """
    strategies = _active_strategies(omit_strategies)
    fig, ax = plt.subplots(figsize=figsize)
    missing = []

    for strategy in strategies:
        medians, sizes = [], []
        for size in SAMPLE_SIZES_ELA:
            config_key = f"{strategy}_{size}"
            val, _, is_missing = _ela_aggregate_cv_extended(
                ela_cv, config_key,
                func_ids=func_ids,
                agg_func=agg_func,
                inner_agg=inner_agg,
                outer_agg=outer_agg,
                exclude_features=exclude_features,
                exclude_groups=exclude_groups,
            )
            if is_missing:
                missing.append(config_key)
                continue
            if not np.isnan(val):
                medians.append(val)
                sizes.append(size)
        if sizes:
            ax.plot(sizes, medians, marker="o", linewidth=2, markersize=6,
                    color=STRATEGY_COLORS[strategy],
                    label=STRATEGY_LABELS[strategy])

    # ── axis labels / title ──────────────────────────────────────────────────
    _inner = inner_agg or agg_func
    _outer = outer_agg or agg_func
    if _inner == _outer:
        ylabel = f"{_inner.capitalize()} CV"
        title_agg = _inner.capitalize()
    else:
        ylabel = "CV"
        title_agg = f"{_outer.capitalize()} of per-feature {_inner}s"
    ax.set_xlabel("Sample Size (×d)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    title_parts = [f"ELA Feature Stability — Overall ({title_agg})"]
    if exclude_groups:
        title_parts.append(f"excl. groups: {', '.join(sorted(exclude_groups))}")
    if exclude_features:
        n = len(exclude_features)
        title_parts.append(f"excl. {n} feature{'s' if n > 1 else ''}")
    ax.set_title("\n".join(title_parts), fontsize=13)

    ax.set_xticks(SAMPLE_SIZES_ELA)
    ax.set_xticklabels([f"{s}d" for s in SAMPLE_SIZES_ELA])
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return missing


# ============================================================================
# 2.  Multi-config version of plot_ela_cv_by_feature
# ============================================================================

def plot_ela_cv_by_feature_multi(ela_cv,
                                 config_keys,
                                 features=None,
                                 ncols_outer=2,
                                 figsize_per_feature=(5, 3.5)):
    """
    Boxplot of CV distributions per function (F1–F24) for each ELA feature,
    comparing up to 4 configurations side-by-side within each feature subplot.

    Layout
    ------
    • One *row* of sub-axes per feature (features are stacked vertically).
    • Within each feature row, one panel per config_key (≤ 4).
    • All panels in the same feature row share the **same y-axis**.

    Parameters
    ----------
    ela_cv : dict
        Loaded CV data.
    config_keys : list[str]
        Between 1 and 4 config keys to compare, e.g.
        ["ilhs_50", "lhs_50", "sobol_50", "uniform_50"]
        or ["cma_random_25", "cma_random_50", "cma_random_75", "cma_random_100"].
    features : list[str] | None
        Subset of feature names to include.  None = all features present in
        *any* of the supplied configs.
    ncols_outer : int
        How many feature-rows to place side-by-side on the page (default 2).
        Each "column" is itself a group of len(config_keys) sub-axes.
        Set to 1 for a single-column layout.
    figsize_per_feature : tuple
        (width, height) allocated per individual config panel within a feature.
        Total figure width  ≈ figsize_per_feature[0] × len(config_keys) × ncols_outer
        Total figure height ≈ figsize_per_feature[1] × ceil(n_features / ncols_outer)

    Examples
    --------
    # Compare 4 sample sizes for the same strategy
    plot_ela_cv_by_feature_multi(
        ela_cv,
        config_keys=["cma_random_25", "cma_random_50",
                     "cma_random_75", "cma_random_100"],
    )

    # Compare 4 strategies at the same sample size, only specific features
    plot_ela_cv_by_feature_multi(
        ela_cv,
        config_keys=["ilhs_50", "lhs_50", "sobol_50", "uniform_50"],
        features=["ela_distr.skewness", "nbc.nn_nb.sd_ratio", "ic.eps.s"],
    )
    """
    if not config_keys:
        print("No config_keys provided.")
        return
    if len(config_keys) > 4:
        print("At most 4 config_keys are supported; using the first 4.")
        config_keys = config_keys[:4]

    n_configs = len(config_keys)

    # ── collect data ─────────────────────────────────────────────────────────
    # all_data[ck][feat_name][func_id] = [cv_values]
    all_data = {}
    all_feat_names = set()
    for ck in config_keys:
        d = ela_cv_by_feature(ela_cv, ck)
        all_data[ck] = d
        all_feat_names.update(d.keys())

    if features is not None:
        all_feat_names = {f for f in all_feat_names if f in features}
    if not all_feat_names:
        print("No features found.")
        return

    # Sort features: by ELA group order, then alphabetically within group
    ordered_features = []
    for grp_name in ELA_GROUPS:
        grp_feats = sorted(f for f in all_feat_names
                           if _feature_to_group(f) == grp_name)
        ordered_features.extend(grp_feats)
    # Any features that didn't match a known group go at the end
    ordered_features.extend(sorted(f for f in all_feat_names
                                   if _feature_to_group(f) is None))

    n_features = len(ordered_features)

    # ── figure geometry ───────────────────────────────────────────────────────
    # Each "cell" in the outer grid holds n_configs sub-axes (one per config).
    # ncols_outer controls how many cells sit side-by-side horizontally.
    nrows_outer = (n_features + ncols_outer - 1) // ncols_outer

    # Each outer cell is n_configs panels wide × 1 panel tall.
    fig_width = figsize_per_feature[0] * n_configs * ncols_outer
    fig_height = figsize_per_feature[1] * nrows_outer

    # Build a grid of (nrows_outer, ncols_outer × n_configs) sub-axes.
    total_cols = ncols_outer * n_configs
    fig, axes_grid = plt.subplots(
        nrows_outer, total_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    # Config colours (consistent palette regardless of n_configs)
    base_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    config_colors = {ck: base_palette[i] for i, ck in enumerate(config_keys)}

    func_positions = list(range(1, N_FUNCTIONS + 1))

    # Precompute function-group separator positions (same for every subplot)
    separators = []
    cumulative = 0
    for _, fids in FUNCTION_GROUPS.items():
        cumulative += len(fids)
        separators.append(cumulative + 0.5)
    separators = separators[:-1]  # drop the last (after F24)

    for feat_idx, feat_name in enumerate(ordered_features):
        outer_row = feat_idx // ncols_outer
        outer_col = feat_idx % ncols_outer
        grp = _feature_to_group(feat_name)
        feat_color = ELA_GROUP_COLORS.get(grp, "gray")

        # ── gather y-range for shared axis ───────────────────────────────────
        all_vals_this_feature = []
        per_config_boxdata = {}
        for ck in config_keys:
            func_data = all_data[ck].get(feat_name, {})
            box_data = [func_data.get(fid, []) for fid in func_positions]
            per_config_boxdata[ck] = box_data
            for vals in box_data:
                all_vals_this_feature.extend(vals)

        if all_vals_this_feature:
            y_min = np.percentile(all_vals_this_feature, 1)
            y_max = np.percentile(all_vals_this_feature, 99)
            y_pad = max((y_max - y_min) * 0.08, 1e-6)
            ylim = (y_min - y_pad, y_max + y_pad)
        else:
            ylim = (0, 1)

        for ci, ck in enumerate(config_keys):
            col_in_grid = outer_col * n_configs + ci
            ax = axes_grid[outer_row][col_in_grid]

            box_data = per_config_boxdata[ck]
            bp = ax.boxplot(
                box_data,
                positions=func_positions,
                widths=0.6,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=1.2),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(config_colors[ck])
                patch.set_alpha(0.7)

            ax.set_ylim(ylim)
            ax.set_xlim(0.5, N_FUNCTIONS + 0.5)
            ax.set_xticks(func_positions)
            ax.set_xticklabels(
                [str(f) for f in func_positions], fontsize=6, rotation=90
            )
            ax.grid(True, alpha=0.25, axis="y")

            for sep in separators:
                ax.axvline(x=sep, color="gray", linewidth=0.7,
                           linestyle="--", alpha=0.5)

            # ── titles and axis labels ────────────────────────────────────────
            # Top subtitle = config key (shown only on top panel of the outer row)
            ax.set_title(ck, fontsize=8, pad=3, color=config_colors[ck],
                         fontweight="bold")

            # y-label only on the leftmost config panel of each feature group
            if ci == 0:
                ax.set_ylabel("CV", fontsize=8)
            else:
                ax.set_yticklabels([])

            # Feature name as a shared row label on the very first config panel
            if ci == 0:
                # Use a text annotation on the left edge as a row header
                ax.annotate(
                    feat_name,
                    xy=(-0.18, 0.5),
                    xycoords="axes fraction",
                    fontsize=8,
                    ha="right",
                    va="center",
                    rotation=0,
                    color=feat_color,
                    fontweight="bold",
                    annotation_clip=False,
                )

            ax.set_xlabel("Function", fontsize=7)

    # ── hide unused cells ─────────────────────────────────────────────────────
    for feat_idx in range(n_features, nrows_outer * ncols_outer):
        outer_row = feat_idx // ncols_outer
        outer_col = feat_idx % ncols_outer
        for ci in range(n_configs):
            col_in_grid = outer_col * n_configs + ci
            axes_grid[outer_row][col_in_grid].set_visible(False)

    # ── figure-level legend ───────────────────────────────────────────────────
    legend_patches = [
        Patch(facecolor=config_colors[ck], alpha=0.8, label=ck)
        for ck in config_keys
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper center",
        ncol=n_configs,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
    )

    fig.suptitle(
        "ELA Per-Feature CV by Function — Multi-Config Comparison",
        fontsize=13,
        y=1.04,
    )
    plt.tight_layout()
    plt.show()


# ============================================================================
# 3.  CV distribution histogram — stacked by ELA feature group
# ============================================================================

def _collect_cv_by_group(ela_cv, config_key, func_ids=None,
                         exclude_features=None, exclude_groups=None):
    """
    Collect raw (non-NaN) CV values separated by ELA feature group.

    Returns
    -------
    dict : {group_name: np.ndarray of cv values}
    """
    exclude_features = exclude_features or set()
    exclude_groups = exclude_groups or set()

    if config_key not in ela_cv:
        return {}

    config_data = ela_cv[config_key]
    group_vals = defaultdict(list)

    for func_id in range(1, N_FUNCTIONS + 1):
        if func_ids is not None and func_id not in func_ids:
            continue
        for inst_id in range(1, N_INSTANCES + 1):
            key = (func_id, inst_id, DIMENSION)
            if key not in config_data:
                continue
            for grp_name in ELA_GROUPS:
                if grp_name in exclude_groups:
                    continue
                if grp_name not in config_data[key]:
                    continue
                for feat_name, cv_val in config_data[key][grp_name].items():
                    if feat_name in exclude_features:
                        continue
                    if not np.isnan(cv_val):
                        group_vals[grp_name].append(cv_val)

    return {grp: np.array(vals) for grp, vals in group_vals.items()}


def plot_ela_cv_histogram(ela_cv,
                          config_keys,
                          n_bins=50,
                          log_x=False,
                          log_y=False,
                          x_range=None,
                          clip_percentile=99,
                          func_ids=None,
                          func_group_facets=False,
                          exclude_features=None,
                          exclude_groups=None,
                          show_group_medians=True,
                          figsize=None):
    """
    Stacked histogram of raw ELA CV values, colour-coded by feature group.

    Each bar is split proportionally among the ELA groups that contribute
    values to that bin, so the group composition of each CV range is
    immediately visible.

    Without log_x, the automatic x-range is derived from ``clip_percentile``
    rather than the global max, so extreme outliers do not collapse the bulk
    of the distribution into a single bin.  Values beyond the clipped range
    are still counted — they accumulate in the last bin (right-side overflow),
    which is marked with a hatch pattern and annotated with the overflow count.

    Parameters
    ----------
    ela_cv : dict
        Loaded CV data.
    config_keys : str | list[str]
        One or more config keys (e.g. "ilhs_50" or
        ["ilhs_50", "lhs_50", "sobol_50"]).
        When multiple keys are given, one subplot column is created per key
        so distributions can be compared directly.
    n_bins : int
        Number of histogram bins (default 50).
    log_x : bool
        Use a log-scale x-axis.  Bins are equally spaced in log space.
        When True, ``clip_percentile`` is ignored and the full data range
        is used (log scale already handles skew visually).
    log_y : bool
        Use a log-scale y-axis (useful when one group dominates).
    x_range : (float, float) | None
        Explicit CV range to display.  Overrides ``clip_percentile``.
        Values outside the range accumulate in the overflow bin.
    clip_percentile : float
        Percentile (0-100) used to set the upper x-axis limit when
        ``log_x=False`` and ``x_range`` is not given.  Default 99 means
        the top 1% of values fold into an overflow bin rather than
        stretching the axis.  Set to 100 to disable clipping.
    func_ids : list[int] | None
        Restrict to a subset of BBOB function IDs (1-24).  None = all.
    func_group_facets : bool
        If True, add one *row* per FUNCTION_GROUP (Separable, etc.) so you
        can see how the group composition differs across function classes.
        When False (default), all functions are pooled into one row.
    exclude_features : set[str] | None
        Individual ELA feature names to omit from the histogram.
    exclude_groups : set[str] | None
        ELA feature groups to omit entirely (e.g. {"meta", "ic"}).
    show_group_medians : bool
        Overlay a vertical dashed line at the median CV of each feature group
        (default True).
    figsize : tuple | None
        Override the automatic figure size.

    Examples
    --------
    # Single config — percentile clipping keeps the bulk visible
    plot_ela_cv_histogram(ela_cv, "ilhs_50")

    # Show full range on a log x-axis
    plot_ela_cv_histogram(ela_cv, "ilhs_50", log_x=True)

    # Tighter clipping to focus on the core distribution
    plot_ela_cv_histogram(ela_cv, "ilhs_50", clip_percentile=95)

    # Explicit range
    plot_ela_cv_histogram(ela_cv, "ilhs_50", x_range=(0, 2))

    # Compare four strategies
    plot_ela_cv_histogram(
        ela_cv,
        config_keys=["ilhs_50", "lhs_50", "sobol_50", "uniform_50"],
        n_bins=60,
    )

    # One row per BBOB function group
    plot_ela_cv_histogram(
        ela_cv, "cma_random_50",
        func_group_facets=True,
        exclude_groups={"meta", "disp"},
    )
    """
    # Normalise config_keys to list
    if isinstance(config_keys, str):
        config_keys = [config_keys]

    n_configs = len(config_keys)

    # Determine row structure
    if func_group_facets:
        row_specs = list(FUNCTION_GROUPS.items())  # [(name, [func_ids]), ...]
    else:
        row_specs = [("All Functions", None)]

    n_rows = len(row_specs)
    n_cols = n_configs

    # Active ELA groups (respecting exclusions)
    active_groups = [g for g in ELA_GROUPS
                     if g not in (exclude_groups or set())]

    # ── collect data upfront ─────────────────────────────────────────────────
    # all_data[config_key][row_label] = {group: array}
    all_data = {}
    all_vals_global = []  # flat pool used only for percentile computation

    for ck in config_keys:
        all_data[ck] = {}
        for row_label, fids in row_specs:
            gdata = _collect_cv_by_group(
                ela_cv, ck,
                func_ids=fids,
                exclude_features=exclude_features,
                exclude_groups=exclude_groups,
            )
            all_data[ck][row_label] = gdata
            for vals in gdata.values():
                all_vals_global.extend(vals.tolist())

    if not all_vals_global:
        print("No CV data found for the supplied configuration(s).")
        return

    all_vals_global = np.array(all_vals_global)

    # ── determine display range ───────────────────────────────────────────────
    if x_range is not None:
        x_lo, x_hi = x_range
        has_overflow = bool(np.any(all_vals_global > x_hi))
    elif log_x:
        # Log scale handles skew well — use full range
        x_lo = max(all_vals_global.min(), 1e-6)
        x_hi = all_vals_global.max()
        has_overflow = False
    else:
        # Percentile clipping: prevents outliers from collapsing the bulk
        x_lo = max(all_vals_global.min(), 0.0)
        x_hi = float(np.percentile(all_vals_global, clip_percentile))
        has_overflow = bool(np.any(all_vals_global > x_hi))

    # ── build shared bin edges ────────────────────────────────────────────────
    # Leave the last bin slot for overflow so its width matches the others.
    if log_x:
        bin_edges = np.logspace(np.log10(max(x_lo, 1e-6)),
                                np.log10(x_hi), n_bins + 1)
    else:
        bin_edges = np.linspace(x_lo, x_hi, n_bins + 1)

    # ── figure layout ────────────────────────────────────────────────────────
    if figsize is None:
        col_w = max(5, 10 / n_cols)
        row_h = 3.5 if n_rows == 1 else 2.8
        figsize = (col_w * n_cols, row_h * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        squeeze=False,
        sharex=True,  # same CV range across all panels
        sharey="row",  # same count scale within each function-group row
    )

    # ── plot ─────────────────────────────────────────────────────────────────
    for row_idx, (row_label, _) in enumerate(row_specs):
        for col_idx, ck in enumerate(config_keys):
            ax = axes[row_idx][col_idx]
            gdata = all_data[ck][row_label]

            # Build per-group histograms on the shared bin edges.
            # Values above x_hi are counted into a synthetic overflow bin
            # appended as the rightmost bar.
            group_counts = {}  # counts within [x_lo, x_hi]
            group_overflow = {}  # counts above x_hi

            for grp in active_groups:
                vals = gdata.get(grp, np.array([]))
                in_range = vals[(vals >= x_lo) & (vals <= x_hi)]
                overflow = vals[vals > x_hi]
                counts, _ = np.histogram(in_range, bins=bin_edges)
                group_counts[grp] = counts
                group_overflow[grp] = len(overflow)

            total_overflow = sum(group_overflow.values())

            # ── stacked bars for in-range values ─────────────────────────
            bottoms = np.zeros(n_bins)
            for grp in active_groups:
                counts = group_counts[grp]
                color = ELA_GROUP_COLORS.get(grp, "gray")
                ax.bar(
                    bin_edges[:-1],
                    counts,
                    width=np.diff(bin_edges),
                    align="edge",
                    bottom=bottoms,
                    color=color,
                    alpha=0.85,
                    label=grp,
                    linewidth=0,
                )
                bottoms += counts

            # ── overflow bar (hatched, rightmost) ─────────────────────────
            if has_overflow and total_overflow > 0:
                overflow_bin_width = (bin_edges[-1] - bin_edges[-2])
                overflow_x = bin_edges[-1]
                ov_bottoms = 0.0
                for grp in active_groups:
                    ov = group_overflow[grp]
                    if ov == 0:
                        continue
                    color = ELA_GROUP_COLORS.get(grp, "gray")
                    ax.bar(
                        overflow_x, ov,
                        width=overflow_bin_width,
                        align="edge",
                        bottom=ov_bottoms,
                        color=color,
                        alpha=0.85,
                        hatch="////",
                        edgecolor="white",
                        linewidth=0,
                    )
                    ov_bottoms += ov
                # Label the overflow bar
                ax.text(
                    overflow_x + overflow_bin_width * 0.5,
                    ov_bottoms * 1.02,
                    f"overflow\nn={total_overflow:,}",
                    ha="center", va="bottom",
                    fontsize=7, color="dimgray",
                )
                # Separator line between in-range and overflow
                ax.axvline(bin_edges[-1], color="dimgray",
                           linewidth=1.0, linestyle=":", alpha=0.7)

            # Median lines per group
            if show_group_medians:
                for grp in active_groups:
                    vals = gdata.get(grp, np.array([]))
                    # Median computed on full data (not clipped)
                    if len(vals):
                        med = np.median(vals)
                        if x_lo <= med <= x_hi:  # only draw if visible
                            color = ELA_GROUP_COLORS.get(grp, "gray")
                            ax.axvline(med, color=color, linewidth=1.4,
                                       linestyle="--", alpha=0.9, zorder=5)

            # Axis formatting
            if log_x:
                ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")

            ax.grid(True, alpha=0.25, axis="both")

            # Titles
            if row_idx == 0:
                ax.set_title(ck, fontsize=10, fontweight="bold", pad=6)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{row_label}\nCount", fontsize=9, labelpad=4
                )
            if row_idx == n_rows - 1:
                ax.set_xlabel("CV", fontsize=10)

    # ── shared legend ─────────────────────────────────────────────────────────
    legend_handles = [
        Patch(facecolor=ELA_GROUP_COLORS.get(g, "gray"), alpha=0.85, label=g)
        for g in active_groups
    ]
    if show_group_medians:
        legend_handles.append(
            plt.Line2D([0], [0], color="black", linewidth=1.4,
                       linestyle="--", label="group median")
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        fontsize=9,
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
    )

    title = "ELA CV Distribution — by Feature Group"
    if func_group_facets:
        title += " × Function Group"
    if has_overflow and not log_x:
        pct_label = f"x_range={x_range}" if x_range else f"clip p{clip_percentile}"
        title += f"  [{pct_label}; hatched = overflow]"
    fig.suptitle(title, fontsize=13, y=1.06)

    plt.tight_layout()
    plt.show()


# ============================================================================
# 4.  Median CV across sample sizes — per feature group (line plot)
# ============================================================================

def _median_cv_per_group_per_config(ela_cv, config_keys,
                                    exclude_features=None,
                                    exclude_groups=None,
                                    func_ids=None):
    """
    Returns
    -------
    dict : {group_name: {config_key: median_cv}}
    """
    exclude_features = exclude_features or set()
    exclude_groups = exclude_groups or set()
    active_groups = [g for g in ELA_GROUPS if g not in exclude_groups]

    result = {g: {} for g in active_groups}
    for ck in config_keys:
        gdata = _collect_cv_by_group(
            ela_cv, ck,
            func_ids=func_ids,
            exclude_features=exclude_features,
            exclude_groups=exclude_groups,
        )
        for grp in active_groups:
            vals = gdata.get(grp, np.array([]))
            result[grp][ck] = np.median(vals) if len(vals) else np.nan
    return result


def _median_cv_per_feature_per_config(ela_cv, config_keys,
                                      features=None,
                                      exclude_features=None,
                                      exclude_groups=None,
                                      func_ids=None):
    """
    Returns
    -------
    dict : {feature_name: {config_key: median_cv}}
    """
    exclude_features = exclude_features or set()
    exclude_groups = exclude_groups or set()

    # collect raw values per feature per config
    feat_data = {}  # {ck: {feat: [vals]}}
    all_feat_names = set()
    for ck in config_keys:
        d = ela_cv_by_feature(ela_cv, ck)  # {feat: {func_id: [vals]}}
        feat_data[ck] = {}
        for feat, func_dict in d.items():
            if feat in exclude_features:
                continue
            if _feature_to_group(feat) in exclude_groups:
                continue
            flat = [v for vs in func_dict.values() for v in vs]
            if func_ids is not None:
                flat = [v for fid, vs in func_dict.items()
                        if fid in func_ids for v in vs]
            feat_data[ck][feat] = flat
            all_feat_names.add(feat)

    if features is not None:
        all_feat_names = {f for f in all_feat_names if f in features}

    # Order: by group then alphabetically
    ordered = []
    for grp in ELA_GROUPS:
        ordered.extend(sorted(f for f in all_feat_names
                              if _feature_to_group(f) == grp))
    ordered.extend(sorted(f for f in all_feat_names
                          if _feature_to_group(f) is None))

    result = {feat: {} for feat in ordered}
    for ck in config_keys:
        for feat in ordered:
            vals = feat_data.get(ck, {}).get(feat, [])
            result[feat][ck] = np.median(vals) if vals else np.nan
    return result


def plot_median_cv_per_group(ela_cv,
                             config_keys,
                             exclude_features=None,
                             exclude_groups=None,
                             func_ids=None,
                             annotate_values=True,
                             annotation_fmt=".3f",
                             annotation_offset=(4, 4),
                             figsize=None):
    """
    Line plot of median CV across config_keys, one line per ELA feature group.

    Each point is annotated with its exact median CV value, making it easy
    to see which groups drive changes across sample sizes or strategies.

    Parameters
    ----------
    ela_cv : dict
        Loaded CV data.
    config_keys : list[str]
        Ordered list of config keys forming the x-axis
        (e.g. ["cma_random_25", "cma_random_50", "cma_random_75", "cma_random_100"]).
    exclude_features : set[str] | None
        Individual feature names to omit.
    exclude_groups : set[str] | None
        ELA group names to omit entirely.
    func_ids : list[int] | None
        Restrict to a subset of BBOB functions.
    annotate_values : bool
        Print the median CV value next to each point (default True).
    annotation_fmt : str
        Format string for the annotation, e.g. ".3f", ".2f", ".4g".
    annotation_offset : (dx, dy)
        Pixel offset of the annotation text relative to the point.
    figsize : tuple | None
        Override figure size.

    Examples
    --------
    # All groups — CMA-ES across sample sizes
    plot_median_cv_per_group(
        ela_cv,
        config_keys=["cma_random_25", "cma_random_50",
                     "cma_random_75", "cma_random_100"],
    )

    # Only disp group, all strategies at 50 samples
    plot_median_cv_per_group(
        ela_cv,
        config_keys=["ilhs_50", "lhs_50", "sobol_50",
                     "uniform_50", "cma_random_50"],
        exclude_groups={"ela_dist", "meta", "nbc", "ic"},
    )
    """
    active_groups = [g for g in ELA_GROUPS
                     if g not in (exclude_groups or set())]
    if not active_groups:
        print("No active groups after exclusions.");
        return

    group_medians = _median_cv_per_group_per_config(
        ela_cv, config_keys,
        exclude_features=exclude_features,
        exclude_groups=exclude_groups,
        func_ids=func_ids,
    )

    x = np.arange(len(config_keys))
    if figsize is None:
        figsize = (max(7, len(config_keys) * 1.6), 4.5)

    fig, ax = plt.subplots(figsize=figsize)

    for grp in active_groups:
        color = ELA_GROUP_COLORS.get(grp, "gray")
        y_vals = [group_medians[grp].get(ck, np.nan) for ck in config_keys]
        ax.plot(x, y_vals, marker="o", linewidth=2, markersize=7,
                color=color, label=grp)

        if annotate_values:
            for xi, yi in zip(x, y_vals):
                if not np.isnan(yi):
                    ax.annotate(
                        f"{yi:{annotation_fmt}}",
                        xy=(xi, yi),
                        xytext=annotation_offset,
                        textcoords="offset points",
                        fontsize=8,
                        color=color,
                        fontweight="bold",
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(config_keys, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Median CV", fontsize=11)
    ax.set_xlabel("Configuration", fontsize=11)
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)

    title = "Median CV per Feature Group"
    if exclude_groups:
        title += f"  (excl. {', '.join(sorted(exclude_groups))})"
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 5.  Median CV across sample sizes — per individual feature (line plot)
# ============================================================================

def plot_median_cv_per_feature(ela_cv,
                               config_keys,
                               features=None,
                               exclude_features=None,
                               exclude_groups=None,
                               func_ids=None,
                               annotate_values=True,
                               annotation_fmt=".3f",
                               ncols=2,
                               sharey=False,
                               legend_loc="bottom",
                               legend_ncols=None,
                               figsize_per_panel=(6, 3.8)):
    """
    One subplot per ELA feature group, each containing one line per feature.

    The legend is placed **outside** the axes (below or to the right) so it
    never overlaps the lines.  End-of-line feature labels are added on the
    right margin for quick identification without needing to look up the
    legend.  Median-CV annotations are shown only at the first and last
    config point to keep the plot tidy.

    Parameters
    ----------
    ela_cv : dict
        Loaded CV data.
    config_keys : list[str]
        Ordered config keys forming the x-axis.
    features : list[str] | None
        Restrict to a specific subset of features.  None = all.
    exclude_features : set[str] | None
        Individual feature names to drop.
    exclude_groups : set[str] | None
        Drop all features belonging to these ELA groups.
    func_ids : list[int] | None
        Restrict to a subset of BBOB functions.
    annotate_values : bool
        Print the median CV value at the first and last config point only
        (default True).  Showing all points is intentionally avoided to
        reduce clutter — use ``annotation_fmt`` to control precision.
    annotation_fmt : str
        Format string for annotated values, e.g. ".3f", ".2g".
    ncols : int
        Number of panel columns (default 2).
    sharey : bool
        Share the y-axis across all panels (default False).
    legend_loc : "bottom" | "right" | "none"
        Where to place the figure-level legend.
        "bottom" — one row of entries below all panels (good for many features).
        "right"  — vertical list to the right of all panels.
        "none"   — suppress the legend entirely (rely on end-of-line labels).
    legend_ncols : int | None
        Number of columns in the legend.  None = auto (all features in one row
        for "bottom", 1 column for "right").
    figsize_per_panel : (w, h)
        Size allocated per group panel *before* legend space is added.

    Examples
    --------
    # CMA-ES across sample sizes — what drives the increase?
    plot_median_cv_per_feature(
        ela_cv,
        config_keys=["cma_random_25", "cma_random_50",
                     "cma_random_75", "cma_random_100"],
    )

    # Zoom into the disp group only
    plot_median_cv_per_feature(
        ela_cv,
        config_keys=["cma_random_25", "cma_random_50",
                     "cma_random_75", "cma_random_100"],
        exclude_groups={"ela_dist", "meta", "nbc", "ic"},
        ncols=1,
    )

    # Cross-strategy comparison, legend on the right
    plot_median_cv_per_feature(
        ela_cv,
        config_keys=["ilhs_50", "lhs_50", "sobol_50", "cma_random_50"],
        features=["nbc.nn_nb.sd_ratio", "ic.eps.s", "ic.eps.max"],
        ncols=1,
        legend_loc="right",
    )
    """
    feat_medians = _median_cv_per_feature_per_config(
        ela_cv, config_keys,
        features=features,
        exclude_features=exclude_features,
        exclude_groups=exclude_groups,
        func_ids=func_ids,
    )
    if not feat_medians:
        print("No features found.");
        return

    # ── build panels (one per active ELA group) ───────────────────────────────
    active_groups = [g for g in ELA_GROUPS
                     if g not in (exclude_groups or set())]
    panels = []
    for grp in active_groups:
        grp_feats = [f for f in feat_medians if _feature_to_group(f) == grp]
        if grp_feats:
            panels.append((grp, grp_feats))
    ungrouped = [f for f in feat_medians if _feature_to_group(f) is None]
    if ungrouped:
        panels.append(("other", ungrouped))

    n_panels = len(panels)
    if n_panels == 0:
        print("No panels to draw.");
        return

    nrows = (n_panels + ncols - 1) // ncols

    # ── figure size: add right margin for end-of-line labels ─────────────────
    right_margin = 1.8  # inches reserved for inline feature labels
    fig_w = figsize_per_panel[0] * ncols + right_margin
    fig_h = figsize_per_panel[1] * nrows

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        sharey=sharey,
    )

    x = np.arange(len(config_keys))
    x_last = len(config_keys) - 1

    _PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]

    all_handles, all_labels = [], []  # collected for figure legend

    for panel_idx, (grp_name, grp_feats) in enumerate(panels):
        row = panel_idx // ncols
        col = panel_idx % ncols
        ax = axes[row][col]
        group_color = ELA_GROUP_COLORS.get(grp_name, "gray")

        # Collect (y_last, feat, color) for staggered end-of-line labels
        endpoints = []

        for fi, feat in enumerate(grp_feats):
            color = _PALETTE[fi % len(_PALETTE)]
            y_vals = [feat_medians[feat].get(ck, np.nan) for ck in config_keys]

            line, = ax.plot(x, y_vals, marker="o", linewidth=1.8,
                            markersize=6, color=color, label=feat)

            if panel_idx == 0:  # collect once for figure legend
                all_handles.append(line)
                all_labels.append(feat)

            # Annotate only first and last point to avoid clutter
            if annotate_values:
                for xi in [0, x_last]:
                    yi = y_vals[xi] if xi < len(y_vals) else np.nan
                    if not np.isnan(yi):
                        # First point: label to the left; last: to the right
                        ha = "right" if xi == 0 else "left"
                        dx = -5 if xi == 0 else 5
                        ax.annotate(
                            f"{yi:{annotation_fmt}}",
                            xy=(xi, yi),
                            xytext=(dx, 0),
                            textcoords="offset points",
                            fontsize=7,
                            color=color,
                            ha=ha, va="center",
                        )

            y_end = y_vals[x_last] if x_last < len(y_vals) else np.nan
            endpoints.append((y_end, feat, color))

        # ── staggered end-of-line feature labels ─────────────────────────────
        # Sort by y value, then space them out vertically so they don't collide
        endpoints.sort(key=lambda t: t[0] if not np.isnan(t[0]) else -np.inf)
        ax_ymin, ax_ymax = ax.get_ylim()
        y_range = ax_ymax - ax_ymin if ax_ymax != ax_ymin else 1.0

        min_step = y_range * 0.05  # minimum vertical gap between labels
        placed_y = []
        for y_end, feat, color in endpoints:
            if np.isnan(y_end):
                continue
            # Push label up if too close to the previous one
            y_label = y_end
            for prev_y in reversed(placed_y):
                if abs(y_label - prev_y) < min_step:
                    y_label = prev_y + min_step
            placed_y.append(y_label)
            # Strip common prefix (group name) for brevity
            short = feat.split(".")[-1] if "." in feat else feat
            ax.annotate(
                short,
                xy=(x_last, y_end),
                xytext=(10, 0),
                textcoords="offset points",
                fontsize=7,
                color=color,
                va="center",
                annotation_clip=False,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(config_keys, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Median CV", fontsize=9)
        ax.set_xlabel("Configuration", fontsize=9)
        ax.set_title(grp_name, fontsize=11,
                     color=group_color, fontweight="bold")
        ax.grid(True, alpha=0.3)
        # Extend x-axis slightly right to make room for inline labels
        ax.set_xlim(left=-0.3, right=x_last + 0.3)

    # Collect handles/labels from ALL panels for the figure legend
    all_handles, all_labels = [], []
    for panel_idx, (_, grp_feats) in enumerate(panels):
        row = panel_idx // ncols
        col = panel_idx % ncols
        ax = axes[row][col]
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)

    # ── hide unused panels ────────────────────────────────────────────────────
    for panel_idx in range(n_panels, nrows * ncols):
        axes[panel_idx // ncols][panel_idx % ncols].set_visible(False)

    # ── figure-level legend ───────────────────────────────────────────────────
    if legend_loc != "none" and all_handles:
        n_leg = len(all_handles)
        if legend_loc == "bottom":
            ncols_leg = legend_ncols or min(n_leg, 4)
            fig.legend(
                all_handles, all_labels,
                loc="lower center",
                ncol=ncols_leg,
                fontsize=7,
                bbox_to_anchor=(0.5, -0.02),
                frameon=True,
                title="Feature",
                title_fontsize=8,
            )
            plt.subplots_adjust(bottom=0.05 + 0.028 * ((n_leg // ncols_leg) + 1))
        elif legend_loc == "right":
            ncols_leg = legend_ncols or 1
            fig.legend(
                all_handles, all_labels,
                loc="center left",
                ncol=ncols_leg,
                fontsize=7,
                bbox_to_anchor=(1.0, 0.5),
                frameon=True,
                title="Feature",
                title_fontsize=8,
            )

    fig.suptitle(
        "Median CV per Feature — across Configurations",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    plt.show()


# ============================================================================
# 6.  Perspective-1 boxplot — distribution of per-feature medians
# ============================================================================

def _collect_per_feature_medians(ela_cv, config_key,
                                 func_ids=None,
                                 exclude_features=None,
                                 exclude_groups=None,
                                 agg_func="median"):
    """
    Step 1+2 of the aggregation pipeline:
      - pool all (func, instance) CV values per feature  (flat pool)
      - collapse that pool with ``agg_func`` ("median" or "mean")

    Returns
    -------
    np.ndarray
        One scalar per feature (the distribution the boxplot shows).
    list[str]
        Corresponding feature names (same order).
    """
    exclude_features = exclude_features or set()
    exclude_groups = exclude_groups or set()
    agg = np.median if agg_func == "median" else np.mean

    if config_key not in ela_cv:
        return np.array([]), []

    config_data = ela_cv[config_key]
    feature_pool = defaultdict(list)  # feat_name -> flat list of raw CVs

    for func_id in range(1, N_FUNCTIONS + 1):
        if func_ids is not None and func_id not in func_ids:
            continue
        for inst_id in range(1, N_INSTANCES + 1):
            key = (func_id, inst_id, DIMENSION)
            if key not in config_data:
                continue
            for grp_name in ELA_GROUPS:
                if grp_name in exclude_groups:
                    continue
                if grp_name not in config_data[key]:
                    continue
                for feat_name, cv_val in config_data[key][grp_name].items():
                    if feat_name in exclude_features:
                        continue
                    if not np.isnan(cv_val):
                        feature_pool[feat_name].append(cv_val)

    feat_names = sorted(feature_pool.keys())
    feat_aggs = np.array([agg(feature_pool[f]) for f in feat_names])
    return feat_aggs, feat_names


def plot_ela_perspective1_boxplot(ela_cv,
                                  omit_strategies=None,
                                  sample_sizes=None,
                                  exclude_features=None,
                                  exclude_groups=None,
                                  func_ids=None,
                                  agg_func="median",
                                  group_by="sample_size",
                                  show_median_line=True,
                                  show_swarm=False,
                                  swarm_alpha=0.35,
                                  swarm_size=3,
                                  ylim=None,
                                  figsize=None):
    """
    Perspective-1 as a boxplot.

    Each box shows the **distribution of per-feature aggregates** across all
    active ELA features for one (strategy, sample_size) configuration.
    ``agg_func`` controls how raw CV values are collapsed within each feature
    (step 2 of the pipeline) before the boxplot is drawn:

    * ``"median"`` (default) — each feature contributes its median CV across
      the full flat pool of (func × instance) observations.
    * ``"mean"`` — each feature contributes its mean CV instead.

    The central line inside each box is therefore the median *of those
    per-feature aggregates* — identical to the value plotted as a point in
    ``plot_ela_perspective1_extended`` with the same ``agg_func``.

    Two layout modes via ``group_by``:

    ``"sample_size"`` (default)
        X-axis = sample sizes; one group of boxes per sample size, one box
        per strategy inside each group.  Mirrors the original line-plot x-axis
        and makes it easy to see how spread changes with sample size.

    ``"strategy"``
        X-axis = strategies; one group of boxes per strategy, one box per
        sample size.  Better for comparing strategies head-to-head.

    Parameters
    ----------
    ela_cv : dict
        Loaded CV data.
    omit_strategies : set[str] | None
        Strategies to exclude (e.g. {"cma_random"}).
    sample_sizes : list[int] | None
        Sample sizes to include.  None = SAMPLE_SIZES_ELA = [25, 50, 75, 100].
    exclude_features : set[str] | None
        Individual feature names to omit from the per-feature median pool.
    exclude_groups : set[str] | None
        ELA groups to omit entirely.
    func_ids : list[int] | None
        Restrict to a subset of BBOB functions.
    agg_func : "median" | "mean"
        How to collapse the flat pool of CV values within each feature.
        "median" matches the default behaviour of ``plot_ela_perspective1_extended``.
        "mean" is more sensitive to outlier CV values within a feature.
    group_by : "sample_size" | "strategy"
        Primary grouping on the x-axis (see above).
    show_median_line : bool
        Draw a horizontal dashed line at the overall median of each box
        (i.e. the value that would appear in the original line plot).
        Default True.
    show_swarm : bool
        Overlay individual feature-median dots on each box (default False).
        Useful for seeing how many features drive the tails.
    swarm_alpha : float
        Opacity of swarm dots (default 0.35).
    swarm_size : float
        Radius of swarm dots in points (default 3).
    ylim : (float, float) | None
        Y-axis limits, e.g. ``(0, 0.5)`` to zoom into the low-CV region.
        None = automatic (default).
    figsize : tuple | None
        Override automatic figure size.

    Examples
    --------
    # Default — grouped by sample size, all strategies
    plot_ela_perspective1_boxplot(ela_cv)

    # Grouped by strategy, exclude CMA-random, show individual feature dots
    plot_ela_perspective1_boxplot(
        ela_cv,
        omit_strategies={"cma_random"},
        group_by="strategy",
        show_swarm=True,
    )

    # Only disp features, CMA-random across sample sizes
    plot_ela_perspective1_boxplot(
        ela_cv,
        omit_strategies={"ilhs", "lhs", "sobol", "uniform"},
        exclude_groups={"ela_dist", "meta", "nbc", "ic"},
        show_swarm=True,
    )
    """
    strategies = _active_strategies(omit_strategies)
    sample_sizes = sample_sizes or SAMPLE_SIZES_ELA

    # ── collect per-feature medians for every config ──────────────────────────
    # data[strategy][size] = np.ndarray of per-feature medians
    data = {}
    for strategy in strategies:
        data[strategy] = {}
        for size in sample_sizes:
            ck = f"{strategy}_{size}"
            feat_meds, _ = _collect_per_feature_medians(
                ela_cv, ck,
                func_ids=func_ids,
                exclude_features=exclude_features,
                exclude_groups=exclude_groups,
                agg_func=agg_func,
            )
            data[strategy][size] = feat_meds

    # ── build plot structure ──────────────────────────────────────────────────
    if group_by == "sample_size":
        groups = sample_sizes
        inner_items = strategies
        group_labels = [f"{s}d" for s in sample_sizes]
        inner_colors = [STRATEGY_COLORS[s] for s in strategies]
        inner_labels = [STRATEGY_LABELS[s] for s in strategies]

        def get_vals(grp, inner):
            return data[inner][grp]
    else:  # group_by == "strategy"
        groups = strategies
        inner_items = sample_sizes
        group_labels = [STRATEGY_LABELS[s] for s in strategies]
        inner_colors = [plt.cm.viridis(i / max(len(sample_sizes) - 1, 1))
                        for i in range(len(sample_sizes))]
        inner_labels = [f"{s}d" for s in sample_sizes]

        def get_vals(grp, inner):
            return data[grp][inner]

    n_groups = len(groups)
    n_inner = len(inner_items)

    # Spacing: groups separated by 1 unit, boxes within a group packed tightly
    box_width = 0.8 / n_inner
    group_gap = 1.2  # centre-to-centre distance between groups
    offsets = np.linspace(-(n_inner - 1) / 2 * box_width,
                          (n_inner - 1) / 2 * box_width,
                          n_inner)
    group_centres = np.arange(n_groups) * group_gap

    if figsize is None:
        figsize = (max(8, n_groups * group_gap * 1.1 + 2), 5)

    fig, ax = plt.subplots(figsize=figsize)

    for ii, (inner, color, label) in enumerate(
            zip(inner_items, inner_colors, inner_labels)):

        positions = group_centres + offsets[ii]

        box_data = [get_vals(grp, inner) for grp in groups]
        # Filter out empty arrays
        plot_positions = [p for p, d in zip(positions, box_data) if len(d) > 0]
        plot_data = [d for d in box_data if len(d) > 0]

        if not plot_data:
            continue

        bp = ax.boxplot(
            plot_data,
            positions=plot_positions,
            widths=box_width * 0.85,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker=".", markersize=3,
                            alpha=0.4, color=color),
            medianprops=dict(color="black", linewidth=1.8),
            whiskerprops=dict(color=color, linewidth=1.2),
            capprops=dict(color=color, linewidth=1.2),
            boxprops=dict(facecolor=color, alpha=0.55, linewidth=0),
            label=label,
        )

        # Overall-median marker (= the value in the original line plot)
        if show_median_line:
            for pos, d in zip(plot_positions, plot_data):
                overall_med = np.median(d)
                ax.plot(
                    [pos - box_width * 0.4, pos + box_width * 0.4],
                    [overall_med, overall_med],
                    color=color, linewidth=2.0, linestyle="--",
                    zorder=5,
                )

        # Optional swarm overlay
        if show_swarm:
            rng = np.random.default_rng(seed=42)
            for pos, d in zip(plot_positions, plot_data):
                jitter = rng.uniform(-box_width * 0.3, box_width * 0.3, size=len(d))
                ax.scatter(
                    pos + jitter, d,
                    s=swarm_size ** 2,
                    color=color,
                    alpha=swarm_alpha,
                    zorder=4,
                    linewidths=0,
                )

    # ── axes decoration ───────────────────────────────────────────────────────
    ax.set_xticks(group_centres)
    ax.set_xticklabels(group_labels, fontsize=10)
    ax.set_xlabel("Sample Size (×d)" if group_by == "sample_size"
                  else "Strategy", fontsize=12)
    ax.set_ylabel("Per-Feature Median CV", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    if ylim is not None:
        ax.set_ylim(ylim)

    # Vertical separators between groups
    for i in range(1, n_groups):
        ax.axvline(group_centres[i] - group_gap / 2,
                   color="lightgray", linewidth=0.8, linestyle="-")

    # Legend
    legend_handles = [
        Patch(facecolor=c, alpha=0.7, label=l)
        for c, l in zip(inner_colors, inner_labels)
    ]
    if show_median_line:
        legend_handles.append(
            plt.Line2D([0], [0], color="gray", linewidth=2,
                       linestyle="--", label="overall median")
        )
    ax.legend(handles=legend_handles, fontsize=9,
              loc="upper right", frameon=True)

    agg_label = agg_func.capitalize()
    title = f"ELA Feature Stability — Perspective 1 (Boxplot of Per-Feature {agg_label}s)"
    if exclude_groups:
        title += f"\nexcl. groups: {', '.join(sorted(exclude_groups))}"
    ax.set_title(title, fontsize=13)

    plt.tight_layout()
    plt.show()


# ============================================================================
# 7.  Perspective-1 boxplot — per feature group or per individual feature
# ============================================================================

def plot_ela_perspective1_boxplot_by_group(ela_cv,
                                           omit_strategies=None,
                                           sample_sizes=None,
                                           feature_groups=None,
                                           exclude_features=None,
                                           func_ids=None,
                                           agg_func="median",
                                           group_by="sample_size",
                                           show_median_line=True,
                                           show_swarm=False,
                                           swarm_alpha=0.35,
                                           swarm_size=3,
                                           ylim=None,
                                           ncols=2,
                                           figsize_per_panel=(6, 4)):
    """
    Perspective-1 boxplot faceted by ELA feature group.

    One panel per feature group, each showing the distribution of
    per-feature aggregates (median or mean of the flat CV pool) for
    every (strategy, sample_size) configuration — identical logic to
    ``plot_ela_perspective1_boxplot`` but restricted to the features
    belonging to each group, so inter-group differences are visible
    at a glance.

    Parameters
    ----------
    ela_cv : dict
        Loaded CV data.
    omit_strategies : set[str] | None
        Strategies to exclude (e.g. {"cma_random"}).
    sample_sizes : list[int] | None
        Sample sizes to include.  None = SAMPLE_SIZES_ELA.
    feature_groups : list[str] | None
        Which ELA groups to plot.  None = all groups that have data.
        Valid names: "ela_dist", "meta", "disp", "nbc", "ic".
    exclude_features : set[str] | None
        Individual feature names to omit.
    func_ids : list[int] | None
        Restrict to a subset of BBOB functions.
    agg_func : "median" | "mean"
        How to collapse the flat CV pool within each feature before boxing.
    group_by : "sample_size" | "strategy"
        Primary grouping on the x-axis within each panel.
    show_median_line : bool
        Draw a dashed line at the overall aggregate of each box (default True).
    show_swarm : bool
        Overlay individual feature dots on each box (default False).
    swarm_alpha : float
        Opacity of swarm dots.
    swarm_size : float
        Radius of swarm dots in points.
    ylim : (float, float) | None
        Shared y-axis limits across all panels, e.g. (0, 0.5).
        None = each panel scales independently.
    ncols : int
        Number of panel columns (default 2).
    figsize_per_panel : (w, h)
        Size allocated per panel.

    Examples
    --------
    # All groups, grouped by sample size
    plot_ela_perspective1_boxplot_by_group(ela_cv)

    # Only disp and nbc, mean aggregation, zoom in
    plot_ela_perspective1_boxplot_by_group(
        ela_cv,
        feature_groups=["disp", "nbc"],
        agg_func="mean",
        ylim=(0, 1),
        show_swarm=True,
    )

    # CMA-random only, all groups, grouped by sample size
    plot_ela_perspective1_boxplot_by_group(
        ela_cv,
        omit_strategies={"ilhs", "lhs", "sobol", "uniform"},
        show_swarm=True,
    )
    """
    strategies = _active_strategies(omit_strategies)
    sample_sizes = sample_sizes or SAMPLE_SIZES_ELA
    agg = np.median if agg_func == "median" else np.mean

    # Determine which groups to plot
    active_groups = feature_groups if feature_groups is not None else ELA_GROUPS

    # ── collect per-feature aggregates split by group ─────────────────────────
    # pool[strategy][size][group] = np.ndarray of per-feature aggregates
    pool = {}
    for strategy in strategies:
        pool[strategy] = {}
        for size in sample_sizes:
            ck = f"{strategy}_{size}"
            feat_aggs, feat_names = _collect_per_feature_medians(
                ela_cv, ck,
                func_ids=func_ids,
                exclude_features=exclude_features,
                exclude_groups=set(ELA_GROUPS) - set(active_groups),
                agg_func=agg_func,
            )
            # Split by group
            pool[strategy][size] = {}
            for grp in active_groups:
                mask = np.array([_feature_to_group(f) == grp
                                 for f in feat_names])
                pool[strategy][size][grp] = feat_aggs[mask] if mask.any() else np.array([])

    # Drop groups that have no data at all
    active_groups = [g for g in active_groups
                     if any(len(pool[s][sz][g]) > 0
                            for s in strategies for sz in sample_sizes)]
    if not active_groups:
        print("No data found for the specified groups.");
        return

    # ── layout ────────────────────────────────────────────────────────────────
    n_panels = len(active_groups)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    # Box geometry — same logic as the main boxplot function
    if group_by == "sample_size":
        groups = sample_sizes
        inner_items = strategies
        group_labels = [f"{s}d" for s in sample_sizes]
        inner_colors = [STRATEGY_COLORS[s] for s in strategies]
        inner_labels = [STRATEGY_LABELS[s] for s in strategies]

        def get_vals(grp_name, grp, inner):
            return pool[inner][grp][grp_name]
    else:
        groups = strategies
        inner_items = sample_sizes
        group_labels = [STRATEGY_LABELS[s] for s in strategies]
        inner_colors = [plt.cm.viridis(i / max(len(sample_sizes) - 1, 1))
                        for i in range(len(sample_sizes))]
        inner_labels = [f"{s}d" for s in sample_sizes]

        def get_vals(grp_name, grp, inner):
            return pool[grp][inner][grp_name]

    n_groups = len(groups)
    n_inner = len(inner_items)
    box_width = 0.8 / n_inner
    group_gap = 1.2
    offsets = np.linspace(-(n_inner - 1) / 2 * box_width,
                          (n_inner - 1) / 2 * box_width,
                          n_inner)
    group_centres = np.arange(n_groups) * group_gap

    rng = np.random.default_rng(seed=42)

    for panel_idx, grp_name in enumerate(active_groups):
        row = panel_idx // ncols
        col = panel_idx % ncols
        ax = axes[row][col]
        group_color = ELA_GROUP_COLORS.get(grp_name, "gray")

        for ii, (inner, color, label) in enumerate(
                zip(inner_items, inner_colors, inner_labels)):

            positions = group_centres + offsets[ii]
            box_data = [get_vals(grp_name, grp, inner) for grp in groups]
            plot_pos = [p for p, d in zip(positions, box_data) if len(d) > 0]
            plot_data = [d for d in box_data if len(d) > 0]

            if not plot_data:
                continue

            bp = ax.boxplot(
                plot_data,
                positions=plot_pos,
                widths=box_width * 0.85,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker=".", markersize=3,
                                alpha=0.4, color=color),
                medianprops=dict(color="black", linewidth=1.8),
                whiskerprops=dict(color=color, linewidth=1.2),
                capprops=dict(color=color, linewidth=1.2),
                boxprops=dict(facecolor=color, alpha=0.55, linewidth=0),
                label=label,
            )

            if show_median_line:
                for pos, d in zip(plot_pos, plot_data):
                    overall = agg(d)
                    ax.plot(
                        [pos - box_width * 0.4, pos + box_width * 0.4],
                        [overall, overall],
                        color=color, linewidth=2.0, linestyle="--", zorder=5,
                    )

            if show_swarm:
                for pos, d in zip(plot_pos, plot_data):
                    jitter = rng.uniform(-box_width * 0.3, box_width * 0.3,
                                         size=len(d))
                    ax.scatter(pos + jitter, d,
                               s=swarm_size ** 2, color=color,
                               alpha=swarm_alpha, zorder=4, linewidths=0)

        ax.set_xticks(group_centres)
        ax.set_xticklabels(group_labels, fontsize=9,
                           rotation=20 if group_by == "strategy" else 0,
                           ha="right" if group_by == "strategy" else "center")
        ax.set_xlabel("Sample Size (×d)" if group_by == "sample_size"
                      else "Strategy", fontsize=10)
        ax.set_ylabel(f"Per-Feature {agg_func.capitalize()} CV", fontsize=10)
        ax.set_title(grp_name, fontsize=11,
                     color=group_color, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        if ylim is not None:
            ax.set_ylim(ylim)

        # Group separators
        for i in range(1, n_groups):
            ax.axvline(group_centres[i] - group_gap / 2,
                       color="lightgray", linewidth=0.8)

    # Hide unused panels
    for panel_idx in range(n_panels, nrows * ncols):
        axes[panel_idx // ncols][panel_idx % ncols].set_visible(False)

    # Shared legend on the last visible panel
    legend_handles = [
        Patch(facecolor=c, alpha=0.7, label=l)
        for c, l in zip(inner_colors, inner_labels)
    ]
    if show_median_line:
        legend_handles.append(
            plt.Line2D([0], [0], color="gray", linewidth=2,
                       linestyle="--", label=f"overall {agg_func}")
        )
    fig.legend(handles=legend_handles, fontsize=9,
               loc="upper center", ncol=len(legend_handles),
               bbox_to_anchor=(0.5, 1.02), frameon=True)

    agg_label = agg_func.capitalize()
    fig.suptitle(
        f"ELA Feature Stability — Per Group Boxplot of Per-Feature {agg_label}s",
        fontsize=13, y=1.05,
    )
    plt.tight_layout()
    plt.show()


def plot_ela_perspective1_boxplot_by_feature(ela_cv,
                                             omit_strategies=None,
                                             sample_sizes=None,
                                             features=None,
                                             feature_groups=None,
                                             exclude_features=None,
                                             func_ids=None,
                                             agg_func="median",
                                             group_by="sample_size",
                                             show_mean_line=True,
                                             show_both_agg=False,
                                             show_fliers=False,
                                             flier_alpha=0.2,
                                             show_swarm=False,
                                             swarm_alpha=0.35,
                                             swarm_size=3,
                                             ylim=None,
                                             ncols=3,
                                             figsize_per_panel=(5, 3.5)):
    """
    Perspective-1 boxplot faceted by individual ELA feature.

    One panel per feature.  Each panel's box shows the distribution of
    raw CV values from the flat pool for that feature across all
    (func × instance) pairs — so the box represents within-feature
    variability, not across-feature spread as in the other boxplot functions.

    This makes it easy to see which specific features are stable vs noisy,
    and how that changes with sample size or strategy.

    Parameters
    ----------
    ela_cv : dict
        Loaded CV data.
    omit_strategies : set[str] | None
        Strategies to exclude.
    sample_sizes : list[int] | None
        Sample sizes to include.  None = SAMPLE_SIZES_ELA.
    features : list[str] | None
        Explicit list of feature names to plot.  None = all features
        (subject to ``feature_groups`` and ``exclude_features``).
    feature_groups : list[str] | None
        Restrict to features belonging to these ELA groups.
        None = all groups.
    exclude_features : set[str] | None
        Individual feature names to omit.
    func_ids : list[int] | None
        Restrict to a subset of BBOB functions.
    agg_func : "median" | "mean"
        Shown as a horizontal dashed line on each box when
        ``show_mean_line=True`` — the value from perspective-1.
    group_by : "sample_size" | "strategy"
        Primary grouping on the x-axis within each panel.
    show_mean_line : bool
        Overlay a dashed line at the per-feature aggregate (median or mean
        of the raw pool) — this is the step-2 value from the pipeline,
        i.e. what feeds into the cross-feature boxplot.  Default True.
    show_both_agg : bool
        When True, always draw **both** the median (dotted) and the mean
        (dashed) on every box, regardless of ``agg_func``.  The gap between
        them directly visualises outlier pull: a large mean–median gap means
        a small number of extreme CV values are dragging the mean up.
        Default False.
    show_fliers : bool
        Show individual outlier points beyond the whiskers (default False).
        The raw pool can have up to 2400 points per box so enabling this
        together with ``show_swarm`` is not recommended.
    flier_alpha : float
        Opacity of flier markers when ``show_fliers=True`` (default 0.2).
    show_swarm : bool
        Overlay individual (func × instance) CV dots (default False).
        Note: with 2400 points per box this can be slow — consider
        combining with ``func_ids`` to reduce the pool size.
    swarm_alpha : float
        Opacity of swarm dots.
    swarm_size : float
        Radius of swarm dots in points.
    ylim : (float, float) | None
        Shared y-axis limits across all panels.
    ncols : int
        Number of panel columns (default 3).
    figsize_per_panel : (w, h)
        Size allocated per panel.

    Examples
    --------
    # All disp features, CMA-random across sample sizes
    plot_ela_perspective1_boxplot_by_feature(
        ela_cv,
        feature_groups=["disp"],
        omit_strategies={"ilhs", "lhs", "sobol", "uniform"},
    )

    # Specific features, all strategies at 50 samples, zoom in
    plot_ela_perspective1_boxplot_by_feature(
        ela_cv,
        features=["nbc.nn_nb.sd_ratio", "ic.eps.s", "ic.eps.max"],
        sample_sizes=[50],
        group_by="strategy",
        ylim=(0, 2),
        show_swarm=True,
    )
    """
    strategies = _active_strategies(omit_strategies)
    sample_sizes = sample_sizes or SAMPLE_SIZES_ELA
    agg = np.median if agg_func == "median" else np.mean
    exclude_features = exclude_features or set()

    # Determine which groups to allow
    allowed_groups = set(feature_groups) if feature_groups is not None \
        else set(ELA_GROUPS)

    # ── collect raw CV pools per feature per config ───────────────────────────
    # raw_pool[strategy][size][feat] = np.ndarray of raw CV values (flat pool)
    raw_pool = {}
    all_feats = set()

    for strategy in strategies:
        raw_pool[strategy] = {}
        for size in sample_sizes:
            ck = f"{strategy}_{size}"
            if ck not in ela_cv:
                raw_pool[strategy][size] = {}
                continue
            config_data = ela_cv[ck]
            feat_pool = defaultdict(list)

            for func_id in range(1, N_FUNCTIONS + 1):
                if func_ids is not None and func_id not in func_ids:
                    continue
                for inst_id in range(1, N_INSTANCES + 1):
                    key = (func_id, inst_id, DIMENSION)
                    if key not in config_data:
                        continue
                    for grp_name in ELA_GROUPS:
                        if grp_name not in allowed_groups:
                            continue
                        if grp_name not in config_data[key]:
                            continue
                        for feat_name, cv_val in config_data[key][grp_name].items():
                            if feat_name in exclude_features:
                                continue
                            if not np.isnan(cv_val):
                                feat_pool[feat_name].append(cv_val)

            raw_pool[strategy][size] = {f: np.array(v)
                                        for f, v in feat_pool.items()}
            all_feats.update(feat_pool.keys())

    # Filter to requested features
    if features is not None:
        all_feats = {f for f in all_feats if f in features}
    if not all_feats:
        print("No features found.");
        return

    # Order: by group then alphabetically
    ordered_feats = []
    for grp in ELA_GROUPS:
        if grp not in allowed_groups:
            continue
        ordered_feats.extend(
            sorted(f for f in all_feats if _feature_to_group(f) == grp)
        )
    ordered_feats.extend(sorted(f for f in all_feats
                                if _feature_to_group(f) is None))

    # ── layout ────────────────────────────────────────────────────────────────
    n_panels = len(ordered_feats)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols,
                 figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    # Box geometry
    if group_by == "sample_size":
        groups = sample_sizes
        inner_items = strategies
        group_labels = [f"{s}d" for s in sample_sizes]
        inner_colors = [STRATEGY_COLORS[s] for s in strategies]
        inner_labels = [STRATEGY_LABELS[s] for s in strategies]

        def get_pool(feat, grp, inner):
            return raw_pool[inner][grp].get(feat, np.array([]))
    else:
        groups = strategies
        inner_items = sample_sizes
        group_labels = [STRATEGY_LABELS[s] for s in strategies]
        inner_colors = [plt.cm.viridis(i / max(len(sample_sizes) - 1, 1))
                        for i in range(len(sample_sizes))]
        inner_labels = [f"{s}d" for s in sample_sizes]

        def get_pool(feat, grp, inner):
            return raw_pool[grp][inner].get(feat, np.array([]))

    n_groups = len(groups)
    n_inner = len(inner_items)
    box_width = 0.8 / n_inner
    group_gap = 1.2
    offsets = np.linspace(-(n_inner - 1) / 2 * box_width,
                          (n_inner - 1) / 2 * box_width,
                          n_inner)
    group_centres = np.arange(n_groups) * group_gap

    rng = np.random.default_rng(seed=42)

    for panel_idx, feat in enumerate(ordered_feats):
        row = panel_idx // ncols
        col = panel_idx % ncols
        ax = axes[row][col]
        grp_name = _feature_to_group(feat)
        group_color = ELA_GROUP_COLORS.get(grp_name, "gray")

        for ii, (inner, color, label) in enumerate(
                zip(inner_items, inner_colors, inner_labels)):

            positions = group_centres + offsets[ii]
            box_data = [get_pool(feat, grp, inner) for grp in groups]
            plot_pos = [p for p, d in zip(positions, box_data) if len(d) > 0]
            plot_data = [d for d in box_data if len(d) > 0]

            if not plot_data:
                continue

            ax.boxplot(
                plot_data,
                positions=plot_pos,
                widths=box_width * 0.85,
                patch_artist=True,
                showfliers=show_fliers,
                flierprops=dict(marker=".", markersize=3,
                                alpha=flier_alpha, color=color),
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(color=color, linewidth=1.1),
                capprops=dict(color=color, linewidth=1.1),
                boxprops=dict(facecolor=color, alpha=0.5, linewidth=0),
                label=label,
            )

            # Aggregate overlay lines (step-2 value from the pipeline)
            if show_both_agg:
                # Always draw both median (dotted) and mean (dashed)
                # so the gap — i.e. outlier pull — is directly visible
                for pos, d in zip(plot_pos, plot_data):
                    med_val = np.median(d)
                    mean_val = np.mean(d)
                    ax.plot(
                        [pos - box_width * 0.4, pos + box_width * 0.4],
                        [med_val, med_val],
                        color=color, linewidth=2.0, linestyle=":", zorder=5,
                    )
                    ax.plot(
                        [pos - box_width * 0.4, pos + box_width * 0.4],
                        [mean_val, mean_val],
                        color=color, linewidth=2.0, linestyle="--", zorder=5,
                    )
            elif show_mean_line:
                for pos, d in zip(plot_pos, plot_data):
                    agg_val = agg(d)
                    ax.plot(
                        [pos - box_width * 0.4, pos + box_width * 0.4],
                        [agg_val, agg_val],
                        color=color, linewidth=2.0, linestyle="--", zorder=5,
                    )

            if show_swarm:
                for pos, d in zip(plot_pos, plot_data):
                    jitter = rng.uniform(-box_width * 0.3, box_width * 0.3,
                                         size=len(d))
                    ax.scatter(pos + jitter, d,
                               s=swarm_size ** 2, color=color,
                               alpha=swarm_alpha, zorder=4, linewidths=0)

        # Strip group prefix for brevity in title
        short = feat.split(".")[-1] if "." in feat else feat
        ax.set_title(short, fontsize=9, color=group_color, fontweight="bold")
        ax.set_xticks(group_centres)
        ax.set_xticklabels(group_labels, fontsize=8,
                           rotation=20 if group_by == "strategy" else 0,
                           ha="right" if group_by == "strategy" else "center")
        ax.set_ylabel("CV", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        if ylim is not None:
            ax.set_ylim(ylim)

        for i in range(1, n_groups):
            ax.axvline(group_centres[i] - group_gap / 2,
                       color="lightgray", linewidth=0.8)

    # Hide unused panels
    for panel_idx in range(n_panels, nrows * ncols):
        axes[panel_idx // ncols][panel_idx % ncols].set_visible(False)

    # Shared legend
    legend_handles = [
        Patch(facecolor=c, alpha=0.7, label=l)
        for c, l in zip(inner_colors, inner_labels)
    ]
    if show_both_agg:
        legend_handles.append(
            plt.Line2D([0], [0], color="gray", linewidth=2,
                       linestyle=":", label="per-feature median")
        )
        legend_handles.append(
            plt.Line2D([0], [0], color="gray", linewidth=2,
                       linestyle="--", label="per-feature mean")
        )
    elif show_mean_line:
        legend_handles.append(
            plt.Line2D([0], [0], color="gray", linewidth=2,
                       linestyle="--", label=f"per-feature {agg_func}")
        )
    fig.legend(handles=legend_handles, fontsize=9,
               loc="upper center", ncol=len(legend_handles),
               bbox_to_anchor=(0.5, 1.02), frameon=True)

    if show_both_agg:
        agg_subtitle = "median (dotted) and mean (dashed) shown — gap = outlier pull"
    else:
        agg_subtitle = f"{agg_func.capitalize()} shown as dashed line"
    fig.suptitle(
        f"ELA Raw CV Distribution per Feature  [{agg_subtitle}]",
        fontsize=13, y=1.05,
    )
    plt.tight_layout()
    plt.show()


# ============================================================================
# 8.  Per-feature boxplot sliced by function group or individual function
# ============================================================================

def _build_raw_pool_by_slice(ela_cv, config_keys,
                             features,
                             slice_ids,  # list of func_ids per slice
                             slice_labels,  # display label per slice
                             exclude_features=None,
                             exclude_groups=None):
    """
    Shared data-collection backend for the two sliced boxplot functions.

    Returns
    -------
    dict : {config_key: {feature_name: {slice_label: np.ndarray of raw CVs}}}
    """
    exclude_features = exclude_features or set()
    exclude_groups = exclude_groups or set()

    result = {}
    for ck in config_keys:
        if ck not in ela_cv:
            result[ck] = {}
            continue
        config_data = ela_cv[ck]
        feat_slice = {f: {lbl: [] for lbl in slice_labels} for f in features}

        for slice_label, func_id_list in zip(slice_labels, slice_ids):
            for func_id in func_id_list:
                for inst_id in range(1, N_INSTANCES + 1):
                    key = (func_id, inst_id, DIMENSION)
                    if key not in config_data:
                        continue
                    for grp_name in ELA_GROUPS:
                        if grp_name in exclude_groups:
                            continue
                        if grp_name not in config_data[key]:
                            continue
                        for feat_name, cv_val in config_data[key][grp_name].items():
                            if feat_name not in feat_slice:
                                continue
                            if feat_name in exclude_features:
                                continue
                            if not np.isnan(cv_val):
                                feat_slice[feat_name][slice_label].append(cv_val)

        result[ck] = {f: {lbl: np.array(v)
                          for lbl, v in slices.items()}
                      for f, slices in feat_slice.items()}
    return result


def _draw_sliced_boxplots(ax, plot_data_by_inner, slice_labels,
                          inner_items, inner_colors, inner_labels,
                          agg_func, show_mean_line, show_both_agg,
                          show_fliers, flier_alpha,
                          show_swarm, swarm_alpha, swarm_size,
                          group_gap=1.2):
    """
    Draw grouped boxplots on ``ax``.

    X-axis groups = slice_labels (function groups or individual functions).
    Within each group, one box per inner_item (strategy or sample size).
    """
    agg = np.median if agg_func == "median" else np.mean
    n_groups = len(slice_labels)
    n_inner = len(inner_items)
    box_width = 0.8 / max(n_inner, 1)
    offsets = np.linspace(-(n_inner - 1) / 2 * box_width,
                          (n_inner - 1) / 2 * box_width,
                          n_inner)
    group_centres = np.arange(n_groups) * group_gap
    rng = np.random.default_rng(seed=42)

    for ii, (inner, color, label) in enumerate(
            zip(inner_items, inner_colors, inner_labels)):

        positions = group_centres + offsets[ii]
        box_data = [plot_data_by_inner[inner].get(sl, np.array([]))
                    for sl in slice_labels]
        plot_pos = [p for p, d in zip(positions, box_data) if len(d) > 0]
        plot_d = [d for d in box_data if len(d) > 0]

        if not plot_d:
            continue

        ax.boxplot(
            plot_d,
            positions=plot_pos,
            widths=box_width * 0.85,
            patch_artist=True,
            showfliers=show_fliers,
            flierprops=dict(marker=".", markersize=3,
                            alpha=flier_alpha, color=color),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color=color, linewidth=1.1),
            capprops=dict(color=color, linewidth=1.1),
            boxprops=dict(facecolor=color, alpha=0.5, linewidth=0),
            label=label,
        )

        if show_both_agg:
            for pos, d in zip(plot_pos, plot_d):
                for val, ls in [(np.median(d), ":"), (np.mean(d), "--")]:
                    ax.plot([pos - box_width * 0.4, pos + box_width * 0.4],
                            [val, val],
                            color=color, linewidth=2.0,
                            linestyle=ls, zorder=5)
        elif show_mean_line:
            for pos, d in zip(plot_pos, plot_d):
                val = agg(d)
                ax.plot([pos - box_width * 0.4, pos + box_width * 0.4],
                        [val, val],
                        color=color, linewidth=2.0,
                        linestyle="--", zorder=5)

        if show_swarm:
            for pos, d in zip(plot_pos, plot_d):
                jitter = rng.uniform(-box_width * 0.3, box_width * 0.3,
                                     size=len(d))
                ax.scatter(pos + jitter, d,
                           s=swarm_size ** 2, color=color,
                           alpha=swarm_alpha, zorder=4, linewidths=0)

    ax.set_xticks(group_centres)
    ax.set_xticklabels(slice_labels, fontsize=8, rotation=45, ha="right")
    for i in range(1, n_groups):
        ax.axvline(group_centres[i] - group_gap / 2,
                   color="lightgray", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    return group_centres, box_width, group_gap


def _sliced_figure_legend(fig, inner_colors, inner_labels,
                          show_both_agg, show_mean_line, agg_func):
    handles = [Patch(facecolor=c, alpha=0.7, label=l)
               for c, l in zip(inner_colors, inner_labels)]
    if show_both_agg:
        handles += [
            plt.Line2D([0], [0], color="gray", lw=2, ls=":", label="median"),
            plt.Line2D([0], [0], color="gray", lw=2, ls="--", label="mean"),
        ]
    elif show_mean_line:
        handles.append(
            plt.Line2D([0], [0], color="gray", lw=2, ls="--",
                       label=f"per-feature {agg_func}")
        )
    fig.legend(handles=handles, fontsize=9, loc="upper center",
               ncol=len(handles), bbox_to_anchor=(0.5, 1.02), frameon=True)


def plot_ela_boxplot_by_feature_and_func_group(
        ela_cv,
        omit_strategies=None,
        sample_sizes=None,
        features=None,
        feature_groups=None,
        exclude_features=None,
        func_groups=None,
        func_ids=None,
        agg_func="median",
        group_by="sample_size",
        show_mean_line=True,
        show_both_agg=False,
        show_fliers=False,
        flier_alpha=0.2,
        show_swarm=False,
        swarm_alpha=0.35,
        swarm_size=3,
        ylim=None,
        ncols=3,
        figsize_per_panel=(6, 3.8)):
    """
    Per-feature boxplot where the x-axis groups are **BBOB function groups**.

    One panel per ELA feature. X-axis = function groups (≤5), one box per
    config (strategy × sample_size) within each group.

    Examples
    --------
    plot_ela_boxplot_by_feature_and_func_group(
        ela_cv,
        feature_groups=["disp"],
        omit_strategies={"ilhs", "lhs", "sobol", "uniform"},
        show_both_agg=True,
    )
    """
    strategies = _active_strategies(omit_strategies)
    sample_sizes = sample_sizes or SAMPLE_SIZES_ELA
    exclude_features = exclude_features or set()
    allowed_ela_groups = set(feature_groups) if feature_groups else set(ELA_GROUPS)

    fg_items = [(name, ids) for name, ids in FUNCTION_GROUPS.items()
                if func_groups is None or name in func_groups]
    if not fg_items:
        print("No function groups matched.");
        return

    slice_labels, slice_ids = [], []
    for name, ids in fg_items:
        filtered = [f for f in ids if func_ids is None or f in func_ids]
        if filtered:
            slice_labels.append(name)
            slice_ids.append(filtered)
    if not slice_labels:
        print("No functions remaining after func_ids filter.");
        return

    # Gather features
    all_feats = set()
    for ck in [f"{s}_{sz}" for s in strategies for sz in sample_sizes]:
        if ck not in ela_cv: continue
        for grp in ELA_GROUPS:
            if grp not in allowed_ela_groups: continue
            for func_id in range(1, N_FUNCTIONS + 1):
                for inst_id in range(1, N_INSTANCES + 1):
                    key = (func_id, inst_id, DIMENSION)
                    if key not in ela_cv[ck]: continue
                    if grp not in ela_cv[ck][key]: continue
                    all_feats.update(ela_cv[ck][key][grp].keys())
    all_feats -= exclude_features
    if features is not None:
        all_feats = {f for f in all_feats if f in features}
    if not all_feats:
        print("No features found.");
        return

    ordered_feats = []
    for grp in ELA_GROUPS:
        if grp not in allowed_ela_groups: continue
        ordered_feats.extend(
            sorted(f for f in all_feats if _feature_to_group(f) == grp))
    ordered_feats.extend(sorted(f for f in all_feats
                                if _feature_to_group(f) is None))

    config_keys = [f"{s}_{sz}" for s in strategies for sz in sample_sizes]
    pool = _build_raw_pool_by_slice(
        ela_cv, config_keys, ordered_feats,
        slice_ids, slice_labels,
        exclude_features=exclude_features,
        exclude_groups=set(ELA_GROUPS) - allowed_ela_groups,
    )

    inner_items, inner_colors, inner_labels = [], [], []
    for s in strategies:
        for sz in sample_sizes:
            inner_items.append(f"{s}_{sz}")
            inner_colors.append(STRATEGY_COLORS[s])
            inner_labels.append(f"{STRATEGY_LABELS[s]} {sz}d")

    def get_sliced_pool(inner, sl, feat):
        return pool.get(inner, {}).get(feat, {}).get(sl, np.array([]))

    n_panels = len(ordered_feats)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols,
                 figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    for panel_idx, feat in enumerate(ordered_feats):
        row = panel_idx // ncols
        col = panel_idx % ncols
        ax = axes[row][col]
        grp_color = ELA_GROUP_COLORS.get(_feature_to_group(feat), "gray")

        inner_pool = {inner: {sl: get_sliced_pool(inner, sl, feat)
                              for sl in slice_labels}
                      for inner in inner_items}

        _draw_sliced_boxplots(
            ax, inner_pool, slice_labels,
            inner_items, inner_colors, inner_labels,
            agg_func, show_mean_line, show_both_agg,
            show_fliers, flier_alpha,
            show_swarm, swarm_alpha, swarm_size,
        )

        short = feat.split(".")[-1] if "." in feat else feat
        ax.set_title(short, fontsize=9, color=grp_color, fontweight="bold")
        ax.set_ylabel("CV", fontsize=8)
        if ylim is not None:
            ax.set_ylim(ylim)

    for panel_idx in range(n_panels, nrows * ncols):
        axes[panel_idx // ncols][panel_idx % ncols].set_visible(False)

    _sliced_figure_legend(fig, inner_colors, inner_labels,
                          show_both_agg, show_mean_line, agg_func)
    fig.suptitle(
        "ELA CV Distribution per Feature — sliced by BBOB Function Group",
        fontsize=13, y=1.05,
    )
    plt.tight_layout()
    plt.show()


def plot_ela_boxplot_by_feature_and_func(
        ela_cv,
        omit_strategies=None,
        sample_sizes=None,
        features=None,
        feature_groups=None,
        exclude_features=None,
        func_ids=None,
        agg_func="median",
        group_by="sample_size",
        show_mean_line=True,
        show_both_agg=False,
        show_fliers=False,
        flier_alpha=0.2,
        show_swarm=False,
        swarm_alpha=0.35,
        swarm_size=3,
        ylim=None,
        ncols=2,
        figsize_per_panel=None):
    """
    Per-feature boxplot where the x-axis positions are **individual BBOB
    functions** (F1–F24 or a subset).

    One panel per ELA feature. X-axis = function IDs, one box per config.
    Each box is built from the 100 instance CV values for that function.
    Vertical dashed lines separate the 5 BBOB function groups.

    The figure width is computed automatically from the number of functions
    and configs so boxes never get crushed — override with ``figsize_per_panel``
    if needed.

    Examples
    --------
    # All disp features, CMA-random, two sample sizes
    plot_ela_boxplot_by_feature_and_func(
        ela_cv,
        feature_groups=["disp"],
        omit_strategies={"ilhs", "lhs", "sobol", "uniform"},
        sample_sizes=[25, 100],
        show_both_agg=True,
    )

    # Single feature, zoom into multimodal functions
    plot_ela_boxplot_by_feature_and_func(
        ela_cv,
        features=["nbc.nn_nb.sd_ratio"],
        func_ids=list(range(15, 25)),
        group_by="strategy",
        sample_sizes=[50],
        ylim=(0, 3),
        show_fliers=True,
    )
    """
    strategies = _active_strategies(omit_strategies)
    sample_sizes = sample_sizes or SAMPLE_SIZES_ELA
    exclude_features = exclude_features or set()
    allowed_ela_groups = set(feature_groups) if feature_groups else set(ELA_GROUPS)

    all_func_ids = list(range(1, N_FUNCTIONS + 1))
    if func_ids is not None:
        all_func_ids = [f for f in all_func_ids if f in func_ids]
    if not all_func_ids:
        print("No function IDs remaining.");
        return

    slice_labels = [f"F{f}" for f in all_func_ids]
    slice_ids = [[f] for f in all_func_ids]

    # Gather features
    all_feats = set()
    for ck in [f"{s}_{sz}" for s in strategies for sz in sample_sizes]:
        if ck not in ela_cv: continue
        for grp in ELA_GROUPS:
            if grp not in allowed_ela_groups: continue
            for func_id in all_func_ids:
                for inst_id in range(1, N_INSTANCES + 1):
                    key = (func_id, inst_id, DIMENSION)
                    if key not in ela_cv[ck]: continue
                    if grp not in ela_cv[ck][key]: continue
                    all_feats.update(ela_cv[ck][key][grp].keys())
    all_feats -= exclude_features
    if features is not None:
        all_feats = {f for f in all_feats if f in features}
    if not all_feats:
        print("No features found.");
        return

    ordered_feats = []
    for grp in ELA_GROUPS:
        if grp not in allowed_ela_groups: continue
        ordered_feats.extend(
            sorted(f for f in all_feats if _feature_to_group(f) == grp))
    ordered_feats.extend(sorted(f for f in all_feats
                                if _feature_to_group(f) is None))

    config_keys = [f"{s}_{sz}" for s in strategies for sz in sample_sizes]
    pool = _build_raw_pool_by_slice(
        ela_cv, config_keys, ordered_feats,
        slice_ids, slice_labels,
        exclude_features=exclude_features,
        exclude_groups=set(ELA_GROUPS) - allowed_ela_groups,
    )

    inner_items, inner_colors, inner_labels = [], [], []
    for s in strategies:
        for sz in sample_sizes:
            inner_items.append(f"{s}_{sz}")
            inner_colors.append(STRATEGY_COLORS[s])
            inner_labels.append(f"{STRATEGY_LABELS[s]} {sz}d")

    n_inner = len(inner_items)
    n_funcs = len(all_func_ids)
    group_gap = 1.2
    box_width = 0.8 / max(n_inner, 1)

    # Auto figsize: allocate enough width so each function's group of boxes
    # is at least 0.55 inches wide, with extra margin for y-labels
    if figsize_per_panel is None:
        panel_w = max(8, n_funcs * (n_inner * box_width + 0.3) * group_gap)
        panel_h = 3.8
    else:
        panel_w, panel_h = figsize_per_panel

    n_panels = len(ordered_feats)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_w * ncols, panel_h * nrows),
        squeeze=False,
    )

    def get_sliced_pool(inner, sl, feat):
        return pool.get(inner, {}).get(feat, {}).get(sl, np.array([]))

    for panel_idx, feat in enumerate(ordered_feats):
        row = panel_idx // ncols
        col = panel_idx % ncols
        ax = axes[row][col]
        grp_color = ELA_GROUP_COLORS.get(_feature_to_group(feat), "gray")

        inner_pool = {inner: {sl: get_sliced_pool(inner, sl, feat)
                              for sl in slice_labels}
                      for inner in inner_items}

        group_centres, box_width_used, _ = _draw_sliced_boxplots(
            ax, inner_pool, slice_labels,
            inner_items, inner_colors, inner_labels,
            agg_func, show_mean_line, show_both_agg,
            show_fliers, flier_alpha,
            show_swarm, swarm_alpha, swarm_size,
            group_gap=group_gap,
        )

        # ── BBOB function-group separators and labels ─────────────────────
        # Map func_id → index in all_func_ids so we can find positions
        func_to_idx = {f: i for i, f in enumerate(all_func_ids)}
        prev_end = 0
        for fg_name, fg_fids in FUNCTION_GROUPS.items():
            visible = [f for f in fg_fids if f in func_to_idx]
            if not visible:
                continue
            start_idx = func_to_idx[visible[0]]
            end_idx = func_to_idx[visible[-1]]

            # Separator after this group (skip the last)
            if end_idx < len(all_func_ids) - 1:
                sep_x = group_centres[end_idx] + group_gap / 2
                ax.axvline(sep_x, color="dimgray", linewidth=1.2,
                           linestyle="--", alpha=0.6, zorder=3)

            # Group label centred over its functions, just above the x-axis
            mid_x = (group_centres[start_idx] + group_centres[end_idx]) / 2
            ax.text(mid_x, -0.13, fg_name,
                    transform=ax.get_xaxis_transform(),
                    fontsize=6.5, ha="center", va="top",
                    color="dimgray", style="italic")

        short = feat.split(".")[-1] if "." in feat else feat
        ax.set_title(short, fontsize=9, color=grp_color, fontweight="bold")
        ax.set_ylabel("CV", fontsize=8)
        ax.set_xlabel("")  # group labels replace x-label
        if ylim is not None:
            ax.set_ylim(ylim)

        # Extend x-axis margins so first/last boxes aren't clipped
        ax.set_xlim(group_centres[0] - group_gap * 0.6,
                    group_centres[-1] + group_gap * 0.6)

    for panel_idx in range(n_panels, nrows * ncols):
        axes[panel_idx // ncols][panel_idx % ncols].set_visible(False)

    _sliced_figure_legend(fig, inner_colors, inner_labels,
                          show_both_agg, show_mean_line, agg_func)
    fig.suptitle(
        "ELA CV Distribution per Feature — sliced by Individual BBOB Function",
        fontsize=13, y=1.05,
    )
    plt.tight_layout()
    plt.show()

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
