"""
Boxplots of per-(func, feature) ICC values across sample sizes, for one
(dimension, sampling strategy). Two groupings are provided:

  * boxplot_icc_by_feature_group  -- one box per ELA feature group
  * boxplot_icc_by_function_group -- one box per COCO BBOB function group

Consumes the pickles written by compute_ela_icc.py:
    ela_icc[config_key][(func, dim)][feature_group][feature_name] = icc_value

For the chosen strategy, every config "<strategy>_<size>" becomes one cluster
on the x-axis (the sample size); within a cluster there is one boxplot per
category, summarising the distribution of all finite ICCs belonging to it.
"""

import re
import pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Canonical ELA feature-group order (matches compute_ela_icc.py).
GROUP_ORDER = ["ela_dist", "meta", "disp", "nbc", "ic", "pca"]

# COCO BBOB noiseless suite: the 5 standard function groups (f1..f24).
BBOB_FUNCTION_GROUPS = {
    "separable":                 range(1, 6),    # f1-f5
    "low/mod. conditioning":     range(6, 10),   # f6-f9
    "high cond. unimodal":       range(10, 15),  # f10-f14
    "multimodal w/ structure":   range(15, 20),  # f15-f19
    "multimodal weak structure": range(20, 25),  # f20-f24
}
BBOB_GROUP_ORDER = list(BBOB_FUNCTION_GROUPS)
# func_id -> group name
_FUNC_TO_GROUP = {f: g for g, fs in BBOB_FUNCTION_GROUPS.items() for f in fs}
# legend labels that show the f-range, e.g. "separable (f1-f5)"
def _bbob_label(g):
    fs = list(BBOB_FUNCTION_GROUPS[g])
    return f"{g} (f{fs[0]}-f{fs[-1]})"

# Koo & Li reliability bands, drawn as faint horizontal reference regions.
KOO_LI_BANDS = [
    (0.90, 1.00, "excellent"),
    (0.75, 0.90, "good"),
    (0.50, 0.75, "moderate"),
    (0.00, 0.50, "poor"),
]


def split_config_key(config_key):
    """'lhs_random_cd_75' -> ('lhs_random_cd', 75)."""
    m = re.match(r"^(.*)_(\d+)$", config_key)
    if not m:
        raise ValueError(f"Cannot parse strategy/size from config key {config_key!r}")
    return m.group(1), int(m.group(2))


def load_icc_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_source(source, dimension):
    """Turn source ({dim: path} | path | loaded dict) into an ela_icc dict."""
    if isinstance(source, dict):
        # already an ela_icc dict if its keys look like config strings
        if any(isinstance(k, str) for k in source):
            return source
        if dimension in source:
            return load_icc_pkl(source[dimension])
        raise KeyError(f"dimension {dimension} not in source keys {sorted(source)}")
    return load_icc_pkl(source)


def _collect_by_feature_group(ela_icc, strategy, dimension=None):
    """vals[size][feature_group] = list of finite ICCs (all funcs)."""
    vals = defaultdict(lambda: defaultdict(list))
    for config_key, by_func in ela_icc.items():
        st, size = split_config_key(config_key)
        if st != strategy:
            continue
        for func_key, by_group in by_func.items():
            if dimension is not None and func_key[1] != dimension:
                continue
            for group_name, feats in by_group.items():
                for v in feats.values():
                    if v is not None and np.isfinite(v):
                        vals[size][group_name].append(v)
    return vals


def _collect_by_function_group(ela_icc, strategy, dimension=None):
    """vals[size][bbob_function_group] = list of finite ICCs (all features)."""
    vals = defaultdict(lambda: defaultdict(list))
    for config_key, by_func in ela_icc.items():
        st, size = split_config_key(config_key)
        if st != strategy:
            continue
        for func_key, by_group in by_func.items():
            func_id = func_key[0]
            if dimension is not None and func_key[1] != dimension:
                continue
            fgroup = _FUNC_TO_GROUP.get(func_id)
            if fgroup is None:          # func id outside 1..24
                continue
            for feats in by_group.values():
                for v in feats.values():
                    if v is not None and np.isfinite(v):
                        vals[size][fgroup].append(v)
    return vals


def _draw_grouped_boxplots(vals, categories, color_of, ax, ylim,
                           show_bands, show_counts, showfliers, label_of=None):
    """Shared rendering: clusters = sample sizes, boxes = `categories`."""
    sizes = sorted(vals)
    label_of = label_of or {c: c for c in categories}
    G = len(categories)
    slot = 0.82                       # fraction of unit cluster used by boxes
    width = slot / max(G, 1)
    centers = np.arange(len(sizes))

    if show_bands and ylim is not None:
        for lo, hi, name in KOO_LI_BANDS:
            ax.axhspan(lo, hi, color="grey", alpha=0.06, zorder=0)
            ax.text(len(sizes) - 0.45, (lo + hi) / 2, name, va="center",
                    ha="left", fontsize=8, color="grey", alpha=0.8, zorder=0)

    for gi, c in enumerate(categories):
        offset = (gi - (G - 1) / 2) * width
        positions = centers + offset
        box_data = [np.asarray(vals[s].get(c, []), dtype=float) for s in sizes]
        bp = ax.boxplot(
            box_data, positions=positions, widths=width * 0.9,
            patch_artist=True, showfliers=showfliers, manage_ticks=False,
            medianprops=dict(color="black", linewidth=1.2),
            flierprops=dict(marker="o", markersize=2.5, alpha=0.4,
                            markerfacecolor=color_of[c], markeredgecolor="none"),
            whiskerprops=dict(color=color_of[c]),
            capprops=dict(color=color_of[c]),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color_of[c])
            patch.set_alpha(0.65)
            patch.set_edgecolor(color_of[c])
        if show_counts:
            y0 = ylim[0] if ylim else 0
            for x, d in zip(positions, box_data):
                if len(d):
                    ax.text(x, y0 + 0.02, str(len(d)), ha="center", va="bottom",
                            fontsize=6, rotation=90, color=color_of[c])

    ax.set_xticks(centers)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("sample size")
    ax.set_ylabel("ICC(1,1)")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(-0.5, len(sizes) - 0.5)
    ax.grid(True, axis="y", alpha=0.3)
    handles = [Patch(facecolor=color_of[c], alpha=0.65, label=label_of[c])
               for c in categories]
    return handles


def boxplot_icc_by_feature_group(
    source, dimension, strategy, groups=None, ax=None, ylim=(0, 1),
    show_bands=True, show_counts=False, showfliers=True,
    cmap_name="tab10", title=None,
):
    """ICC vs sample size, one box per ELA feature group. See module docstring."""
    ela_icc = _resolve_source(source, dimension)
    vals = _collect_by_feature_group(ela_icc, strategy, dimension=dimension)
    if not vals:
        raise ValueError(f"No data for strategy={strategy!r} at dimension={dimension}. "
                         f"Configs available: {sorted(ela_icc)}")

    present = {g for s in vals for g in vals[s]}
    palette_order = GROUP_ORDER + [g for g in sorted(present) if g not in GROUP_ORDER]
    cmap = plt.get_cmap(cmap_name)
    color_of = {g: cmap(palette_order.index(g) % 10) for g in palette_order}
    categories = [g for g in (groups or palette_order) if g in present]

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.6 * len(vals) + 3, 5.2))
    else:
        fig = ax.figure
    handles = _draw_grouped_boxplots(vals, categories, color_of, ax, ylim,
                                     show_bands, show_counts, showfliers)
    ax.set_title(title or f"ICC by feature group - {strategy}, dim {dimension}")
    ax.legend(handles=handles, title="feature group", loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    return fig, ax


def boxplot_icc_by_function_group(
    source, dimension, strategy, groups=None, ax=None, ylim=(0, 1),
    show_bands=True, show_counts=False, showfliers=True,
    cmap_name="tab10", title=None,
):
    """
    ICC vs sample size, one box per COCO BBOB function group.

    Categories are the 5 standard BBOB groups (separable, low/moderate
    conditioning, high-conditioning unimodal, multimodal with global structure,
    multimodal with weak structure). Each box pools every finite per-feature
    ICC across all functions in that group at that config.

    Same parameters/returns as boxplot_icc_by_feature_group; `groups` here
    selects/orders BBOB function groups by their names (see BBOB_GROUP_ORDER).
    """
    ela_icc = _resolve_source(source, dimension)
    vals = _collect_by_function_group(ela_icc, strategy, dimension=dimension)
    if not vals:
        raise ValueError(f"No data for strategy={strategy!r} at dimension={dimension}. "
                         f"Configs available: {sorted(ela_icc)}")

    present = {g for s in vals for g in vals[s]}
    cmap = plt.get_cmap(cmap_name)
    color_of = {g: cmap(BBOB_GROUP_ORDER.index(g) % 10) for g in BBOB_GROUP_ORDER}
    categories = [g for g in (groups or BBOB_GROUP_ORDER) if g in present]
    label_of = {g: _bbob_label(g) for g in BBOB_GROUP_ORDER}

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.6 * len(vals) + 3.5, 5.2))
    else:
        fig = ax.figure
    handles = _draw_grouped_boxplots(vals, categories, color_of, ax, ylim,
                                     show_bands, show_counts, showfliers,
                                     label_of=label_of)
    ax.set_title(title or f"ICC by BBOB function group - {strategy}, dim {dimension}")
    ax.legend(handles=handles, title="BBOB function group", loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    return fig, ax


def _collect_features_in_group(ela_icc, strategy, feature_group, dimension=None):
    """vals[feature][size] = list of finite ICCs across functions."""
    vals = defaultdict(lambda: defaultdict(list))
    for config_key, by_func in ela_icc.items():
        st, size = split_config_key(config_key)
        if st != strategy:
            continue
        for func_key, by_group in by_func.items():
            if dimension is not None and func_key[1] != dimension:
                continue
            feats = by_group.get(feature_group)
            if not feats:
                continue
            for fname, v in feats.items():
                if v is not None and np.isfinite(v):
                    vals[fname][size].append(v)
    return vals


def boxplot_features_in_group(
    source, dimension, strategy, feature_group, ncols=4, ylim=(0, 1),
    show_bands=True, showfliers=True, sort_by="median", cmap_name="tab10",
    title=None, width_per=3.1, height_per=2.7,
):
    """
    Small-multiples view of every feature in one ELA feature group: one panel
    per feature, each showing ICC vs sample size as boxplots (distribution
    across the BBOB functions) for one (dimension, strategy).

    sort_by : "median" -> panels ordered by descending overall median ICC
              (most reliable features first); anything else keeps the group's
              declared order. The overall median is annotated in each panel.

    Returns (fig, axes).
    """
    ela_icc = _resolve_source(source, dimension)
    vals = _collect_features_in_group(ela_icc, strategy, feature_group, dimension=dimension)
    if not vals:
        raise ValueError(
            f"No data for group={feature_group!r}, strategy={strategy!r}, "
            f"dim={dimension}. Configs available: {sorted(ela_icc)}")

    sizes = sorted({s for f in vals for s in vals[f]})

    def overall_median(f):
        allv = [v for s in vals[f] for v in vals[f][s]]
        return float(np.median(allv)) if allv else np.nan

    features = list(vals)
    if sort_by == "median":
        features.sort(key=lambda f: (np.nan_to_num(overall_median(f), nan=-1.0)),
                      reverse=True)

    # one consistent color = the group's color in the other plots
    palette_order = GROUP_ORDER + [feature_group] if feature_group not in GROUP_ORDER else GROUP_ORDER
    color = plt.get_cmap(cmap_name)(palette_order.index(feature_group) % 10)

    n = len(features)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width_per * ncols, height_per * nrows),
                             squeeze=False, sharey=True)
    axes_flat = axes.flatten()

    centers = np.arange(len(sizes))
    for ax, feat in zip(axes_flat, features):
        if show_bands and ylim is not None:
            for lo, hi, _ in KOO_LI_BANDS:
                ax.axhspan(lo, hi, color="grey", alpha=0.06, zorder=0)
        box_data = [np.asarray(vals[feat].get(s, []), dtype=float) for s in sizes]
        bp = ax.boxplot(
            box_data, positions=centers, widths=0.6, patch_artist=True,
            showfliers=showfliers, manage_ticks=False,
            medianprops=dict(color="black", linewidth=1.2),
            flierprops=dict(marker="o", markersize=2, alpha=0.35,
                            markerfacecolor=color, markeredgecolor="none"),
            whiskerprops=dict(color=color), capprops=dict(color=color),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
            patch.set_edgecolor(color)
        ax.set_title(feat, fontsize=8.5)
        ax.set_xticks(centers)
        ax.set_xticklabels(sizes, fontsize=8)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlim(-0.5, len(sizes) - 0.5)
        ax.grid(True, axis="y", alpha=0.3)
        m = overall_median(feat)
        if np.isfinite(m):
            ax.text(0.04, 0.95, f"med {m:.2f}", transform=ax.transAxes,
                    fontsize=7.5, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    for ax in axes_flat[n:]:
        ax.set_visible(False)
    for r in range(nrows):
        axes[r][0].set_ylabel("ICC(1,1)")
    for c in range(ncols):
        axes[nrows - 1][c].set_xlabel("sample size")

    fig.suptitle(title or f"'{feature_group}' features - {strategy}, dim {dimension}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig, axes


# BBOB function-group boundaries (last function id of each group) for separators.
_BBOB_BOUNDARIES = [max(rng) for rng in list(BBOB_FUNCTION_GROUPS.values())[:-1]]


def _collect_feature_func_matrix(ela_icc, strategy, dimension, feature_group=None):
    """data[size][feature][func] = single ICC value (np.nan if missing)."""
    from collections import defaultdict as _dd
    data = _dd(lambda: _dd(dict))
    feats_seen, funcs_seen = [], set()
    for config_key, by_func in ela_icc.items():
        st, size = split_config_key(config_key)
        if st != strategy:
            continue
        for func_key, by_group in by_func.items():
            if func_key[1] != dimension:
                continue
            func_id = func_key[0]
            funcs_seen.add(func_id)
            groups = ([feature_group] if feature_group else by_group.keys())
            for g in groups:
                feats = by_group.get(g)
                if not feats:
                    continue
                for fname, v in feats.items():
                    if fname not in feats_seen:
                        feats_seen.append(fname)
                    data[size][fname][func_id] = (
                        float(v) if (v is not None and np.isfinite(v)) else np.nan)
    return data, feats_seen, sorted(funcs_seen)


def heatmap_icc_per_size(
    source, dimension, strategy, feature_group=None, sort_by="median",
    cmap_name="RdYlGn", show_function_groups=True, ncols=None,
    width_per=4.6, row_height=0.34, title=None,
):
    """
    One ICC heatmap per sample size: rows = features, cols = BBOB functions
    (f1-f24), colour = the raw ICC for that (feature, function) at that size
    (no aggregation). NaN/degenerate cells are shown light grey.

    feature_group : restrict to one group's features (default: all features).
    sort_by : "median" -> rows ordered by descending overall-median ICC
              (consistent across all panels); else features keep first-seen order.
    """
    ela_icc = _resolve_source(source, dimension)
    data, features, funcs = _collect_feature_func_matrix(
        ela_icc, strategy, dimension, feature_group=feature_group)
    if not features:
        raise ValueError(f"No data for strategy={strategy!r}, dim={dimension}"
                         + (f", group={feature_group!r}" if feature_group else "")
                         + f". Configs: {sorted(ela_icc)}")

    sizes = sorted(data)

    def overall_median(f):
        vals = [data[s][f][fn] for s in sizes for fn in data[s].get(f, {})]
        vals = [v for v in vals if np.isfinite(v)]
        return np.median(vals) if vals else np.nan

    if sort_by == "median":
        features = sorted(features,
                          key=lambda f: np.nan_to_num(overall_median(f), nan=-1.0),
                          reverse=True)

    n = len(sizes)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(width_per * ncols, max(2.5, row_height * len(features) + 1.6) * nrows),
        squeeze=False, sharey=True, layout="constrained")
    axes_flat = axes.flatten()

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("lightgrey")

    fpos = {f: i for i, f in enumerate(funcs)}
    sep_x = [sum(1 for f in funcs if f <= b) - 0.5 for b in _BBOB_BOUNDARIES
             if any(f > b for f in funcs)]

    im = None
    for ax, size in zip(axes_flat, sizes):
        M = np.full((len(features), len(funcs)), np.nan)
        for r, feat in enumerate(features):
            row = data[size].get(feat, {})
            for fn, v in row.items():
                M[r, fpos[fn]] = v
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_title(f"n = {size}", fontsize=11)
        ax.set_xticks(range(len(funcs)))
        ax.set_xticklabels(funcs, fontsize=6)
        ax.set_xlabel("BBOB function")
        if show_function_groups:
            for x in sep_x:
                ax.axvline(x, color="black", lw=0.8, alpha=0.5)
        ax.set_yticks(range(len(features)))

    for r in range(nrows):
        axes[r][0].set_yticklabels(features, fontsize=7)
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    cbar = fig.colorbar(im, ax=axes_flat[:n].tolist() if hasattr(axes_flat[:n], "tolist")
                        else list(axes_flat[:n]), fraction=0.025, pad=0.01,
                        ticks=[0, 0.5, 0.75, 0.9, 1.0])
    cbar.set_label("ICC(1,1)   (grey = NaN/degenerate)")
    grp = feature_group or "all"
    fig.suptitle(title or f"ICC heatmap - {grp} features, {strategy}, dim {dimension}",
                 fontsize=13)
    return fig, axes