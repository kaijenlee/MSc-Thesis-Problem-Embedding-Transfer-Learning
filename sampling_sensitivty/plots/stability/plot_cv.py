"""
Plot within-instance CV for ELA features, mirroring the ICC plots.

Consumes the pickle from compute_ela_cv.py:
    ela_cv[config_key][(func, inst, dim)][feature_group][feature_name] = cv

AGGREGATION (your rule: "median between the same features, mean across the
median of features"):
  * Instances are ALWAYS collapsed by MEDIAN -> each feature's median CV over
    its repeated measurements. This is the atomic value in every plot.
  * Features are combined by MEAN, only where a single number/marker is needed
    (the overall vs-size line, and the mean-marker on the group boxplots).
  * Boxplots show the distribution of the per-feature medians (so the box uses
    only the median step; the mean step is drawn as a diamond marker).
  * Heatmap cells are the instance-median CV (features are an axis, not averaged).

`costs_runtime` features are filtered out here (compute_ela_cv.py does not drop
them). Lower CV = more stable, so colours/orderings put low CV = "good".
"""

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from plot_icc_boxplots import (
    split_config_key, _resolve_source, GROUP_ORDER,
    BBOB_FUNCTION_GROUPS, BBOB_GROUP_ORDER, _FUNC_TO_GROUP, _bbob_label,
    _BBOB_BOUNDARIES,
)


def _is_runtime(name):
    return "costs_runtime" in name


def _finite(v):
    return v is not None and np.isfinite(v)


# --------------------------------------------------------------------------- #
# Collectors                                                                  #
# --------------------------------------------------------------------------- #

def _collect_group_raw(ela_cv, strategy, dimension, by="feature", cap=None):
    """raw[size][category][feature][func] = list of per-instance CV.
    by="feature"  -> category = ELA feature group (scope = all functions)
    by="function" -> category = BBOB function group (scope = that group's funcs;
                     all features included)."""
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for cfg, by_inst in ela_cv.items():
        st, size = split_config_key(cfg)
        if st != strategy:
            continue
        for key, by_group in by_inst.items():
            func_id, _, dim = key
            if dim != dimension:
                continue
            cat_fn = None
            if by == "function":
                cat_fn = _FUNC_TO_GROUP.get(func_id)
                if cat_fn is None:
                    continue
            for g, feats in by_group.items():
                cat = g if by == "feature" else cat_fn
                for fn, v in feats.items():
                    if _is_runtime(fn) or not _finite(v) or (cap is not None and v > cap):
                        continue
                    raw[size][cat][fn][func_id].append(v)
    return raw


def _box_and_diamond(cat_raw, box_over):
    """From {feature: {func: [per-instance cv]}} produce (box_values, diamond).

    diamond is ALWAYS mean-across-feature-medians (your rule), independent of
    box_over. box_over selects what the box distribution is built from:
      "feature_median" : one point per feature  = median over (func x instance)
      "func_feature"   : one point per (feature, func) = median over instances
      "raw"            : every per-instance cv value (no aggregation)
    """
    feat_medians = []
    for by_func in cat_raw.values():
        allv = [v for lst in by_func.values() for v in lst]
        if allv:
            feat_medians.append(float(np.median(allv)))
    diamond = float(np.mean(feat_medians)) if feat_medians else np.nan
    if box_over == "raw":
        box = [v for by_func in cat_raw.values() for lst in by_func.values() for v in lst]
    elif box_over == "feature_median":
        box = feat_medians
    elif box_over == "func_feature":
        box = [float(np.median(lst)) for by_func in cat_raw.values()
               for lst in by_func.values() if lst]
    else:
        raise ValueError("box_over must be 'feature_median', 'func_feature', or 'raw'")
    return box, diamond


def _collect_features_in_group_raw(ela_cv, strategy, dimension, feature_group, cap=None):
    """tmp[feature][size][func] = list of per-instance CV."""
    tmp = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for cfg, by_inst in ela_cv.items():
        st, size = split_config_key(cfg)
        if st != strategy:
            continue
        for key, by_group in by_inst.items():
            func_id, _, dim = key
            if dim != dimension:
                continue
            feats = by_group.get(feature_group)
            if not feats:
                continue
            for fn, v in feats.items():
                if _is_runtime(fn) or not _finite(v) or (cap is not None and v > cap):
                    continue
                tmp[fn][size][func_id].append(v)
    return tmp


def _collect_matrix(ela_cv, strategy, dimension, feature_group=None, cap=None):
    """instance-median CV per (size, feature, func) for the heatmap."""
    tmp = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    feats_seen, funcs_seen = [], set()
    for cfg, by_inst in ela_cv.items():
        st, size = split_config_key(cfg)
        if st != strategy:
            continue
        for key, by_group in by_inst.items():
            func_id, _, dim = key
            if dim != dimension:
                continue
            funcs_seen.add(func_id)
            groups = [feature_group] if feature_group else by_group.keys()
            for g in groups:
                feats = by_group.get(g)
                if not feats:
                    continue
                for fn, v in feats.items():
                    if _is_runtime(fn):
                        continue
                    if fn not in feats_seen:
                        feats_seen.append(fn)
                    if _finite(v) and (cap is None or v <= cap):
                        tmp[size][fn][func_id].append(v)
    data = {s: {fn: {fc: float(np.median(lst)) for fc, lst in by_fc.items() if lst}
                for fn, by_fc in by_fn.items()}
            for s, by_fn in tmp.items()}
    return data, feats_seen, sorted(funcs_seen)


def _collect_overall(ela_cv, strategy, dimension, cap=None):
    """overall CV per size = mean across features of (median over (func,inst))."""
    raw = defaultdict(lambda: defaultdict(list))   # raw[size][feature] = all CV
    for cfg, by_inst in ela_cv.items():
        st, size = split_config_key(cfg)
        if st != strategy:
            continue
        for key, by_group in by_inst.items():
            if key[2] != dimension:
                continue
            for feats in by_group.values():
                for fn, v in feats.items():
                    if _is_runtime(fn) or not _finite(v) or (cap is not None and v > cap):
                        continue
                    raw[size][fn].append(v)
    out = {}
    for size, by_feat in raw.items():
        per_feat_median = [np.median(lst) for lst in by_feat.values() if lst]
        if per_feat_median:
            out[size] = float(np.mean(per_feat_median))
    return out


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _robust_top(arrays, q=98):
    allv = np.concatenate([np.asarray(a, float) for a in arrays if len(a)]) \
        if arrays else np.array([])
    allv = allv[np.isfinite(allv)]
    return float(np.percentile(allv, q)) if allv.size else 1.0


def _draw_grouped_cv(ax, box_by, dia_by, sizes, categories, color_of, ylim, logy):
    """Clusters = sizes, one box per category; white diamond = mean-across-feature-medians."""
    G = len(categories)
    slot = 0.82
    width = slot / max(G, 1)
    centers = np.arange(len(sizes))
    for gi, c in enumerate(categories):
        offset = (gi - (G - 1) / 2) * width
        positions = centers + offset
        data = [np.asarray(box_by[s].get(c, []), dtype=float) for s in sizes]
        bp = ax.boxplot(
            data, positions=positions, widths=width * 0.9, patch_artist=True,
            manage_ticks=False,
            medianprops=dict(color="black", linewidth=1.1),
            flierprops=dict(marker="o", markersize=2, alpha=0.3,
                            markerfacecolor=color_of[c], markeredgecolor="none"),
            whiskerprops=dict(color=color_of[c]), capprops=dict(color=color_of[c]),
        )
        for p in bp["boxes"]:
            p.set_facecolor(color_of[c]); p.set_alpha(0.6); p.set_edgecolor(color_of[c])
        ax.scatter(positions, [dia_by[s].get(c, np.nan) for s in sizes],
                   marker="D", facecolor="white", edgecolor="black", s=28, zorder=5)
    ax.set_xticks(centers); ax.set_xticklabels(sizes)
    ax.set_xlabel("sample size"); ax.set_ylabel("within-instance CV")
    if logy:
        ax.set_yscale("log")
    elif ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(-0.5, len(sizes) - 0.5)
    ax.grid(True, axis="y", alpha=0.3)


# --------------------------------------------------------------------------- #
# Public plots                                                                #
# --------------------------------------------------------------------------- #

def boxplot_cv_by_feature_group(source, dimension, strategy, groups=None, ax=None,
                                box_over="func_feature", cap=None, ylim="auto", logy=False,
                                cmap_name="tab10", title=None):
    """Box per ELA feature group. `box_over` controls the box contents
    ("feature_median" | "func_feature" | "raw"); white diamond is always the
    mean across feature medians."""
    ela_cv = _resolve_source(source, dimension)
    raw = _collect_group_raw(ela_cv, strategy, dimension, by="feature", cap=cap)
    if not raw:
        raise ValueError(f"No CV data for strategy={strategy!r}, dim={dimension}.")
    sizes = sorted(raw)
    present = [g for g in GROUP_ORDER if any(g in raw[s] for s in sizes)]
    present += [g for g in sorted({g for s in sizes for g in raw[s]}) if g not in present]
    cats = [g for g in (groups or present) if g in present]

    box_by = {s: {} for s in sizes}
    dia_by = {s: {} for s in sizes}
    for s in sizes:
        for c in cats:
            if c in raw[s]:
                b, d = _box_and_diamond(raw[s][c], box_over)
                box_by[s][c] = b; dia_by[s][c] = d

    cmap = plt.get_cmap(cmap_name)
    palette = GROUP_ORDER + [g for g in present if g not in GROUP_ORDER]
    color_of = {g: cmap(palette.index(g) % 10) for g in palette}
    if ylim == "auto":
        ylim = (0, _robust_top([box_by[s].get(c, []) for s in sizes for c in cats]) * 1.08)
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.6 * len(sizes) + 3, 5.2))
    else:
        fig = ax.figure
    _draw_grouped_cv(ax, box_by, dia_by, sizes, cats, color_of, ylim, logy)
    ax.set_title(title or f"CV by feature group [{box_over}] - {strategy}, dim {dimension}")
    handles = [Patch(facecolor=color_of[g], alpha=0.6, label=g) for g in cats]
    handles.append(Line2D([0], [0], marker="D", color="black", markerfacecolor="white",
                          linestyle="none", label="mean across feature medians"))
    ax.legend(handles=handles, title="feature group", loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    return fig, ax


def boxplot_cv_by_function_group(source, dimension, strategy, groups=None, ax=None,
                                 box_over="func_feature", cap=None, ylim="auto", logy=False,
                                 cmap_name="tab10", title=None):
    """Box per BBOB function group. `box_over` as in boxplot_cv_by_feature_group;
    per-feature medians here are taken over that function group's functions."""
    ela_cv = _resolve_source(source, dimension)
    raw = _collect_group_raw(ela_cv, strategy, dimension, by="function", cap=cap)
    if not raw:
        raise ValueError(f"No CV data for strategy={strategy!r}, dim={dimension}.")
    sizes = sorted(raw)
    present = [g for g in BBOB_GROUP_ORDER if any(g in raw[s] for s in sizes)]
    cats = [g for g in (groups or present) if g in present]

    box_by = {s: {} for s in sizes}
    dia_by = {s: {} for s in sizes}
    for s in sizes:
        for c in cats:
            if c in raw[s]:
                b, d = _box_and_diamond(raw[s][c], box_over)
                box_by[s][c] = b; dia_by[s][c] = d

    cmap = plt.get_cmap(cmap_name)
    color_of = {g: cmap(BBOB_GROUP_ORDER.index(g) % 10) for g in BBOB_GROUP_ORDER}
    if ylim == "auto":
        ylim = (0, _robust_top([box_by[s].get(c, []) for s in sizes for c in cats]) * 1.08)
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.6 * len(sizes) + 3.5, 5.2))
    else:
        fig = ax.figure
    _draw_grouped_cv(ax, box_by, dia_by, sizes, cats, color_of, ylim, logy)
    ax.set_title(title or f"CV by BBOB function group [{box_over}] - {strategy}, dim {dimension}")
    handles = [Patch(facecolor=color_of[g], alpha=0.6, label=_bbob_label(g)) for g in cats]
    handles.append(Line2D([0], [0], marker="D", color="black", markerfacecolor="white",
                          linestyle="none", label="mean across feature medians"))
    ax.legend(handles=handles, title="BBOB function group", loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    return fig, ax


def boxplot_cv_features_in_group(source, dimension, strategy, feature_group,
                                 ncols=4, box_over="func_feature", cap=None, ylim="auto",
                                 logy=False, sort_by="median", cmap_name="tab10",
                                 title=None, width_per=3.1, height_per=2.7):
    """Small multiples: one panel per feature, CV vs sample size.
    box_over="func_feature" -> each box is the instance-median CV across the 24
    functions; box_over="raw" -> each box is every per-instance CV (across
    functions x instances). ("feature_median" is not meaningful per feature and
    is treated as "func_feature".)"""
    if box_over == "feature_median":
        box_over = "func_feature"
    ela_cv = _resolve_source(source, dimension)
    tmp = _collect_features_in_group_raw(ela_cv, strategy, dimension, feature_group, cap=cap)
    if not tmp:
        raise ValueError(f"No CV data for group={feature_group!r}, strategy={strategy!r}, dim={dimension}.")
    sizes = sorted({s for f in tmp for s in tmp[f]})

    def box_values(feat, size):
        by_func = tmp[feat].get(size, {})
        if box_over == "raw":
            return [v for lst in by_func.values() for v in lst]
        return [float(np.median(lst)) for lst in by_func.values() if lst]

    def overall_median(feat):
        allv = [v for size in tmp[feat] for lst in tmp[feat][size].values() for v in lst]
        return float(np.median(allv)) if allv else np.nan

    feats = list(tmp)
    if sort_by == "median":   # most stable (lowest CV) first
        feats.sort(key=lambda f: np.nan_to_num(overall_median(f), nan=np.inf))

    palette = GROUP_ORDER + ([feature_group] if feature_group not in GROUP_ORDER else [])
    color = plt.get_cmap(cmap_name)(palette.index(feature_group) % 10)

    if ylim == "auto":
        ylim = (0, _robust_top([box_values(f, s) for f in feats for s in sizes]) * 1.08)

    n = len(feats); ncols = min(ncols, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width_per * ncols, height_per * nrows),
                             squeeze=False, sharey=True)
    axf = axes.flatten()
    centers = np.arange(len(sizes))
    for ax, feat in zip(axf, feats):
        data = [np.asarray(box_values(feat, s), float) for s in sizes]
        bp = ax.boxplot(data, positions=centers, widths=0.6, patch_artist=True,
                        manage_ticks=False,
                        medianprops=dict(color="black", linewidth=1.1),
                        flierprops=dict(marker="o", markersize=2, alpha=0.3,
                                        markerfacecolor=color, markeredgecolor="none"),
                        whiskerprops=dict(color=color), capprops=dict(color=color))
        for p in bp["boxes"]:
            p.set_facecolor(color); p.set_alpha(0.6); p.set_edgecolor(color)
        ax.set_title(feat, fontsize=8.5)
        ax.set_xticks(centers); ax.set_xticklabels(sizes, fontsize=8)
        if logy:
            ax.set_yscale("log")
        elif ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlim(-0.5, len(sizes) - 0.5)
        ax.grid(True, axis="y", alpha=0.3)
        m = overall_median(feat)
        if np.isfinite(m):
            ax.text(0.04, 0.95, f"med {m:.3f}", transform=ax.transAxes, fontsize=7.5,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    for ax in axf[n:]:
        ax.set_visible(False)
    for r in range(nrows):
        axes[r][0].set_ylabel("within-instance CV")
    for c in range(ncols):
        axes[nrows - 1][c].set_xlabel("sample size")
    fig.suptitle(title or f"'{feature_group}' CV [{box_over}] - {strategy}, dim {dimension}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig, axes


def heatmap_cv_per_size(source, dimension, strategy, feature_group=None,
                        sort_by="median", cmap_name="RdYlGn_r", cap=None,
                        show_function_groups=True, vmin=0.0, vmax=1.0,
                        width_per=4.6, row_height=0.34, title=None):
    """One CV heatmap per sample size: rows=features, cols=BBOB functions, cell =
    instance-median CV. Low CV = green (good). NaN = grey. Colour scale is fixed
    to [vmin, vmax] (default 0..1); any cell above vmax is clipped to the top
    colour, so the scale is comparable across strategies/dimensions."""
    ela_cv = _resolve_source(source, dimension)
    data, features, funcs = _collect_matrix(ela_cv, strategy, dimension, feature_group, cap=cap)
    if not features:
        raise ValueError(f"No CV data for strategy={strategy!r}, dim={dimension}"
                         + (f", group={feature_group!r}" if feature_group else "") + ".")
    sizes = sorted(data)

    def overall_median(f):
        vv = [data[s][f][fn] for s in sizes for fn in data[s].get(f, {})]
        vv = [v for v in vv if np.isfinite(v)]
        return np.median(vv) if vv else np.inf
    if sort_by == "median":
        features = sorted(features, key=overall_median)   # low CV first

    n = len(sizes)
    fig, axes = plt.subplots(1, n, figsize=(width_per * n,
                             max(2.5, row_height * len(features) + 1.6)),
                             squeeze=False, sharey=True, layout="constrained")
    axf = axes.flatten()
    cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad("lightgrey")
    fpos = {f: i for i, f in enumerate(funcs)}
    sep = [sum(1 for f in funcs if f <= b) - 0.5 for b in _BBOB_BOUNDARIES
           if any(f > b for f in funcs)]
    im = None
    for ax, size in zip(axf, sizes):
        M = np.full((len(features), len(funcs)), np.nan)
        for r, feat in enumerate(features):
            for fn, v in data[size].get(feat, {}).items():
                M[r, fpos[fn]] = v
        M = np.clip(M, vmin, vmax)   # values > vmax shown as top colour; NaN preserved
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"n = {size}", fontsize=11)
        ax.set_xticks(range(len(funcs))); ax.set_xticklabels(funcs, fontsize=6)
        ax.set_xlabel("BBOB function")
        if show_function_groups:
            for x in sep:
                ax.axvline(x, color="black", lw=0.8, alpha=0.5)
        ax.set_yticks(range(len(features)))
    axes[0][0].set_yticklabels(features, fontsize=7)
    cbar = fig.colorbar(im, ax=list(axf), fraction=0.025, pad=0.01)
    cbar.set_label(f"within-instance CV  (green=stable; grey=NaN; clipped at {vmax:g})")
    grp = feature_group or "all"
    fig.suptitle(title or f"CV heatmap - {grp} features, {strategy}, dim {dimension}", fontsize=13)
    return fig, axes


def plot_cv_vs_size(sources, strategies=None, cap=None, ncols=None, ylim=None,
                    width_per_plot=5.5, height_per_plot=4.2, title=None, sharey=True):
    """One panel per dimension, one line per strategy. y = overall CV =
    mean across features of (median over (func, inst)). `cap` drops per-instance
    CV above the cap before aggregating (e.g. cap=1.0 removes blown-out cells)."""
    dims = sorted(sources)
    n = len(dims); ncols = ncols or n; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_per_plot * ncols, height_per_plot * nrows),
                             squeeze=False, sharey=sharey)
    axf = axes.flatten()
    loaded = {d: _resolve_source(sources, d) for d in dims}
    all_strats = sorted({split_config_key(c)[0] for d in dims for c in loaded[d]})
    # all_strats.remove("lhs_random_cd")
    strategies = strategies or all_strats
    cmap = plt.get_cmap("tab10"); color_of = {s: cmap(i % 10) for i, s in enumerate(all_strats)}
    for ax, dim in zip(axf, dims):
        for st in strategies:
            ov = _collect_overall(loaded[dim], st, dim, cap=cap)
            if not ov:
                continue
            xs = sorted(ov); ax.plot(xs, [ov[x] for x in xs], marker="o", markersize=5,
                                     linewidth=1.8, color=color_of[st], label=st)
        ax.set_title(f"dimension {dim}"); ax.set_xlabel("sample size")
        ax.set_ylabel("overall CV (mean of feature medians)")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
    for ax in axf[n:]:
        ax.set_visible(False)
    handles = [Line2D([0], [0], color=color_of[s], marker="o", label=s) for s in strategies]
    fig.legend(handles=handles, loc="upper right", title="sampling strategy")
    fig.suptitle(title or "Overall ELA within-instance CV vs sample size", fontsize=14)
    fig.tight_layout()
    return fig


def cv_blowout_report(source, dimension, strategy=None, cap=1.0,
                      pathological_frac=0.5, top=12):
    """Report which CV cells a `cap` filter would remove, per feature.

    A cell (one per-instance CV) is "blown out" if CV > cap. With cap=1.0 that
    means the run-to-run std exceeds the mean's magnitude, so CV is no longer a
    meaningful stability value (typically a near-zero mean on a zero-crossing /
    bounded feature). Pooled over the chosen strategy (or all), all sizes,
    functions and instances at this dimension.

    Returns a per-feature pandas DataFrame (feature, n_cells, n_blown,
    frac_blown, median_cv, max_cv), sorted by frac_blown, and prints a summary
    splitting features into "pathological throughout" (frac_blown >=
    pathological_frac -> candidates to drop entirely) vs "sporadic tail".
    """
    import pandas as pd
    ela_cv = _resolve_source(source, dimension)
    stats = defaultdict(lambda: {"n": 0, "blown": 0, "max": 0.0, "vals": []})
    for cfg, by_inst in ela_cv.items():
        st, _ = split_config_key(cfg)
        if strategy is not None and st != strategy:
            continue
        for key, by_group in by_inst.items():
            if key[2] != dimension:
                continue
            for feats in by_group.values():
                for fn, v in feats.items():
                    if _is_runtime(fn) or not _finite(v):
                        continue
                    s = stats[fn]
                    s["n"] += 1
                    s["vals"].append(v)
                    if v > s["max"]:
                        s["max"] = v
                    if v > cap:
                        s["blown"] += 1
    rows = [{"feature": fn, "n_cells": s["n"], "n_blown": s["blown"],
             "frac_blown": (s["blown"] / s["n"]) if s["n"] else float("nan"),
             "median_cv": float(np.median(s["vals"])) if s["vals"] else float("nan"),
             "mean_cv": float(np.mean(s["vals"])) if s["vals"] else float("nan"),
             "max_cv": s["max"]}
            for fn, s in stats.items()]
    df = (pd.DataFrame(rows).sort_values("frac_blown", ascending=False)
          .reset_index(drop=True))
    if df.empty:
        print("No CV cells found.")
        return df

    tot = int(df["n_cells"].sum())
    blown = int(df["n_blown"].sum())
    patho = df[df["frac_blown"] >= pathological_frac]
    spor = df[(df["n_blown"] > 0) & (df["frac_blown"] < pathological_frac)]
    tag = f", strategy={strategy}" if strategy else " (all strategies)"
    print(f"dim {dimension}{tag}: {blown}/{tot} CV cells ({blown / tot:.2%}) "
          f"exceed cap={cap} and would be filtered.")
    if len(patho):
        print(f"\n  Pathological throughout (>= {pathological_frac:.0%} of cells blown) "
              f"-- consider excluding these features entirely:")
        for _, r in patho.head(top).iterrows():
            print(f"    {r.feature:<32s} {r.frac_blown:6.1%} blown "
                  f"(median CV {r.median_cv:.3g}, max {r.max_cv:.3g})")
    if len(spor):
        print(f"\n  Sporadic blow-ups (small unrepresentative tail) -- "
              f"top {min(top, len(spor))} by fraction:")
        for _, r in spor.head(top).iterrows():
            print(f"    {r.feature:<32s} {r.frac_blown:6.2%} blown "
                  f"({r.n_blown}/{r.n_cells} cells, median CV {r.median_cv:.3g})")
    clean = df[df["n_blown"] == 0]
    print(f"\n  {len(clean)}/{len(df)} features have no blown cells at cap={cap}.")
    return df