"""
plot_ela_cv_tiers.py

Overall within-instance CV vs sample size, split by feature-selection TIER
(safe / caveat / all), using the tier definitions in ela_cv_features.py.

The aggregation is IDENTICAL to plot_ela_cv.py::_collect_overall:

    overall_CV(size) = mean_over_features( median_over(func, inst)( CV ) )

i.e. every feature is first collapsed to ONE median CV (over its func x inst
measurements), then those per-feature medians are averaged. The ONLY thing that
changes between tiers is which features are allowed into that mean:

    safe    : CV_SAFE                      -> ratio-scaled, strictly positive
    caveat  : CV_SAFE + CV_CAVEATED        -> adds bounded / log / scale-dep.
    all     : everything except *costs_runtime*
              -> also pulls in CV_EXCLUDED (sign-changing correlations,
                 differences, adj_r2, intercept, skew/kurt ...). This tier is
                 expected to be higher and noisier; that contrast is the point.

`costs_runtime` is ALWAYS dropped in every tier (environment-dependent cost of
computation, not a landscape property). Lower CV = more stable.

Layout: rows = sampling strategies, cols = dimensions, one line per tier.
Pass a single-element `strategies=[...]` to get the clean one-row comparison.
"""

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from plot_icc_boxplots import (
    split_config_key, _resolve_source,
    _FUNC_TO_GROUP, BBOB_GROUP_ORDER, _bbob_label,
)
from ela_cv_features import CV_SAFE, CV_CAVEATED, CV_EXCLUDED


# --------------------------------------------------------------------------- #
# Tier definitions                                                            #
# --------------------------------------------------------------------------- #

_SAFE = set(CV_SAFE)
_CAVEAT = set(CV_CAVEATED)          # keys only (dict maps feature -> caveat)
_EXCLUDED = set(CV_EXCLUDED)

TIER_ORDER = ("safe", "caveat", "all")
TIER_COLOR = {"safe": "#2ca02c", "caveat": "#ff7f0e", "all": "#7f7f7f"}
TIER_STYLE = {"safe": "-", "caveat": "-", "all": "--"}


def _is_runtime(name):
    return "costs_runtime" in name


def _finite(v):
    return v is not None and np.isfinite(v)


def _tier_allows(tier):
    """Return a predicate fn(feature_name) -> bool for the given tier.
    Runtime features are filtered separately, upstream of this predicate."""
    if tier == "safe":
        allowed = _SAFE
        return lambda fn: fn in allowed
    if tier == "caveat":
        allowed = _SAFE | _CAVEAT
        return lambda fn: fn in allowed
    if tier == "all":
        return lambda fn: True
    raise ValueError(f"tier must be one of {TIER_ORDER}, got {tier!r}")


# --------------------------------------------------------------------------- #
# Collector                                                                   #
# --------------------------------------------------------------------------- #

def _overall_by_size(ela_cv, strategy, dimension, allow, cap=None):
    """Return (mean_by_size, lo_by_size, hi_by_size, n_by_size).

    mean = mean across per-feature medians (the overall CV line).
    lo/hi = 25th/75th percentile of those per-feature medians (spread band).
    n    = number of features contributing at that size.
    """
    raw = defaultdict(lambda: defaultdict(list))    # raw[size][feature] = CVs
    for cfg, by_inst in ela_cv.items():
        st, size = split_config_key(cfg)
        if st != strategy:
            continue
        for key, by_group in by_inst.items():
            if key[2] != dimension:
                continue
            for feats in by_group.values():
                for fn, v in feats.items():
                    if _is_runtime(fn) or not allow(fn):
                        continue
                    if not _finite(v) or (cap is not None and v > cap):
                        continue
                    raw[size][fn].append(v)

    mean_by, lo_by, hi_by, n_by = {}, {}, {}, {}
    for size, by_feat in raw.items():
        med = np.asarray([np.median(lst) for lst in by_feat.values() if lst], dtype=float)
        if med.size:
            mean_by[size] = float(med.mean())
            lo_by[size] = float(np.percentile(med, 25))
            hi_by[size] = float(np.percentile(med, 75))
            n_by[size] = int(med.size)
    return mean_by, lo_by, hi_by, n_by


# --------------------------------------------------------------------------- #
# Tier composition (audit which features land where, incl. "unclassified")    #
# --------------------------------------------------------------------------- #

def tier_composition(sources, dims=None, verbose=True):
    """Classify every non-runtime feature actually present in the data into
    safe / caveat / excluded / unclassified. `unclassified` surfaces anything
    the tier lists in ela_cv_features.py do not cover (e.g. a renamed key from
    a pflacco vs flacco difference) so nothing slips into "all" silently."""
    dims = sorted(sources) if dims is None else dims
    names = set()
    for d in dims:
        ela_cv = _resolve_source(sources, d)
        for by_inst in ela_cv.values():
            for by_group in by_inst.values():
                for feats in by_group.values():
                    names.update(feats.keys())
    names = {n for n in names if not _is_runtime(n)}

    comp = {
        "safe": sorted(names & _SAFE),
        "caveat": sorted(names & _CAVEAT),
        "excluded": sorted(names & _EXCLUDED),
        "unclassified": sorted(names - _SAFE - _CAVEAT - _EXCLUDED),
    }
    if verbose:
        print(f"Feature tier composition ({len(names)} non-runtime features):")
        for k in ("safe", "caveat", "excluded", "unclassified"):
            print(f"  {k:<13s} {len(comp[k]):>3d}")
        if comp["unclassified"]:
            print("  ! unclassified (not in any tier list) -> lands in 'all' only:")
            for f in comp["unclassified"]:
                print(f"      {f}")
    return comp


# --------------------------------------------------------------------------- #
# Public plot                                                                 #
# --------------------------------------------------------------------------- #

def plot_overall_cv_by_tier(sources, strategies=None, tiers=TIER_ORDER, cap=None,
                            show_band=True, logy=False, ylim=None, sharey="row",
                            width_per=5.0, height_per=3.8, title=None):
    """One line per tier (safe/caveat/all), rows=strategies, cols=dimensions.

    Parameters
    ----------
    sources     : mapping dimension -> pickle path (or already-loaded dict),
                  same object you pass to plot_cv_vs_size.
    strategies  : subset/order of sampling strategies; default = all present.
    tiers       : subset/order of ('safe','caveat','all').
    cap         : drop per-instance CV above this before aggregating. Strongly
                  recommended for 'all' (e.g. cap=1.0), since excluded features
                  (correlations, differences) divide by a near-zero mean and
                  blow up. Applies equally to all tiers so lines stay comparable.
    show_band   : shade the IQR of per-feature medians around each line.
    logy        : log y-axis; useful because 'all' can sit orders above 'safe'.
    sharey      : 'row' (default) keeps dims comparable within a strategy while
                  letting strategies differ; use False if 'all' squashes 'safe'.
    """
    dims = sorted(sources)
    loaded = {d: _resolve_source(sources, d) for d in dims}
    all_strats = sorted({split_config_key(c)[0] for d in dims for c in loaded[d]})
    strategies = strategies or all_strats
    tiers = [t for t in tiers if t in TIER_ORDER]
    allow = {t: _tier_allows(t) for t in tiers}

    comp = tier_composition(sources, dims, verbose=False)
    tier_n = {
        "safe": len(comp["safe"]),
        "caveat": len(comp["safe"]) + len(comp["caveat"]),
        "all": sum(len(comp[k]) for k in ("safe", "caveat", "excluded", "unclassified")),
    }

    nrows, ncols = len(strategies), len(dims)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_per * ncols, height_per * nrows),
                             squeeze=False, sharey=sharey, sharex=False)

    for r, st in enumerate(strategies):
        for c, dim in enumerate(dims):
            ax = axes[r][c]
            for t in tiers:
                mean_by, lo_by, hi_by, _ = _overall_by_size(loaded[dim], st, dim,
                                                            allow[t], cap=cap)
                if not mean_by:
                    continue
                xs = sorted(mean_by)
                ax.plot(xs, [mean_by[x] for x in xs], marker="o", markersize=5,
                        linewidth=1.9, color=TIER_COLOR[t], linestyle=TIER_STYLE[t],
                        label=t, zorder=3)
                if show_band:
                    ax.fill_between(xs, [lo_by[x] for x in xs], [hi_by[x] for x in xs],
                                    color=TIER_COLOR[t], alpha=0.13, linewidth=0, zorder=1)
            if r == 0:
                ax.set_title(f"dimension {dim}")
            if c == 0:
                ax.set_ylabel(f"{st}\noverall CV")
            if r == nrows - 1:
                ax.set_xlabel("sample size")
            if logy:
                ax.set_yscale("log")
            elif ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.3)

    handles = [Line2D([0], [0], color=TIER_COLOR[t], linestyle=TIER_STYLE[t],
                      marker="o", label=f"{t}  (n={tier_n[t]})") for t in tiers]
    if show_band:
        handles.append(Patch(facecolor="grey", alpha=0.25,
                             label="IQR of per-feature medians"))
    fig.legend(handles=handles, loc="upper right", title="feature tier", frameon=False)
    fig.suptitle(title or "Overall ELA within-instance CV by feature tier", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes


def _tier_feature_count(sources, tier, dims=None):
    comp = tier_composition(sources, dims, verbose=False)
    if tier == "safe":
        return len(comp["safe"])
    if tier == "caveat":
        return len(comp["safe"]) + len(comp["caveat"])
    if tier == "all":
        return sum(len(comp[k]) for k in ("safe", "caveat", "excluded", "unclassified"))
    raise ValueError(f"tier must be one of {TIER_ORDER}, got {tier!r}")


def plot_cv_vs_size_tier(sources, tier="safe", strategies=None, cap=None, ncols=None,
                         ylim=None, logy=False, width_per_plot=5.5, height_per_plot=4.2,
                         title=None, sharey=True):
    """Drop-in equivalent of plot_cv_vs_size, restricted to one feature `tier`.

    Identical shape: one panel per dimension, one line per strategy, y = overall
    CV = mean across features of (median over (func, inst)). The ONLY change is
    that features are filtered to the chosen tier ('safe' | 'caveat' | 'all')
    before aggregating; `all` reproduces the original plot_cv_vs_size output
    (everything except costs_runtime).

    `cap` drops per-instance CV above the cap before aggregating (e.g. cap=1.0);
    recommended for 'caveat'/'all'. `logy` helps when tiers span wide ranges.
    """
    dims = sorted(sources)
    n = len(dims); ncols = ncols or n; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_per_plot * ncols, height_per_plot * nrows),
                             squeeze=False, sharey=sharey)
    axf = axes.flatten()
    loaded = {d: _resolve_source(sources, d) for d in dims}
    all_strats = sorted({split_config_key(c)[0] for d in dims for c in loaded[d]})
    strategies = strategies or all_strats
    allow = _tier_allows(tier)
    cmap = plt.get_cmap("tab10")
    color_of = {s: cmap(i % 10) for i, s in enumerate(all_strats)}

    for ax, dim in zip(axf, dims):
        for st in strategies:
            mean_by, *_ = _overall_by_size(loaded[dim], st, dim, allow, cap=cap)
            if not mean_by:
                continue
            xs = sorted(mean_by)
            ax.plot(xs, [mean_by[x] for x in xs], marker="o", markersize=5,
                    linewidth=1.8, color=color_of[st], label=st)
        ax.set_title(f"dimension {dim}")
        ax.set_xlabel("sample size")
        ax.set_ylabel("overall CV (mean of feature medians)")
        if logy:
            ax.set_yscale("log")
        elif ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
    for ax in axf[n:]:
        ax.set_visible(False)

    handles = [Line2D([0], [0], color=color_of[s], marker="o", label=s)
               for s in strategies]
    fig.legend(handles=handles, loc="upper right", title="sampling strategy")
    n_feat = _tier_feature_count(sources, tier, dims)
    fig.suptitle(title or f"Overall ELA within-instance CV vs sample size  "
                          f"[{tier} tier, {n_feat} features]", fontsize=14)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Per BBOB function group                                                      #
# --------------------------------------------------------------------------- #

def _overall_by_size_fgroup(ela_cv, strategy, dimension, allow, cap=None):
    """out[bbob_group][size] = overall CV, scoped to that function group.

    Same aggregation as the overall line, but the median-per-feature is taken
    over (func-in-group x inst) rather than over all 24 functions:
        overall_CV(group, size) = mean_over_features(
            median_over(func in group, inst)( cv ) )
    Also returns the number of features contributing per (group, size).
    """
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for cfg, by_inst in ela_cv.items():
        st, size = split_config_key(cfg)
        if st != strategy:
            continue
        for key, by_group in by_inst.items():
            func_id, _, dim = key
            if dim != dimension:
                continue
            fg = _FUNC_TO_GROUP.get(func_id)
            if fg is None:
                continue
            for feats in by_group.values():
                for fn, v in feats.items():
                    if _is_runtime(fn) or not allow(fn):
                        continue
                    if not _finite(v) or (cap is not None and v > cap):
                        continue
                    raw[fg][size][fn].append(v)

    out, nfeat = {}, {}
    for fg, by_size in raw.items():
        out[fg], nfeat[fg] = {}, {}
        for size, by_feat in by_size.items():
            med = [np.median(lst) for lst in by_feat.values() if lst]
            if med:
                out[fg][size] = float(np.mean(med))
                nfeat[fg][size] = len(med)
    return out, nfeat


def plot_cv_vs_size_by_function_group(sources, strategy=None, tier="safe", groups=None,
                                      cap=None, ncols=None, ylim=None, logy=False,
                                      width_per_plot=5.5, height_per_plot=4.2,
                                      title=None, sharey=True):
    """Overall within-instance CV vs sample size, one line per BBOB function
    group, for a FIXED sampling strategy. One panel per dimension.

    Mirrors plot_cv_vs_size_tier, but the line dimension is the BBOB function
    group (separable / low-cond / high-cond unimodal / multimodal adequate /
    multimodal weak) instead of the strategy. Because the function group is now
    the line, the strategy must be fixed per figure.

    Parameters
    ----------
    strategy : str
        Sampling strategy to plot. If None and exactly one strategy is present,
        it is used; otherwise a ValueError lists the available strategies.
    tier     : 'safe' | 'caveat' | 'all'
        Feature subset entering the mean (default 'safe' -- CV-valid features
        give the cleanest cross-group comparison). 'all' uses every non-runtime
        feature.
    groups   : optional subset/order of BBOB function groups (defaults to
        BBOB_GROUP_ORDER).
    cap      : drop per-instance CV above this before aggregating.
    """
    dims = sorted(sources)
    loaded = {d: _resolve_source(sources, d) for d in dims}
    all_strats = sorted({split_config_key(c)[0] for d in dims for c in loaded[d]})
    if strategy is None:
        if len(all_strats) == 1:
            strategy = all_strats[0]
        else:
            raise ValueError(f"Multiple strategies present {all_strats}; "
                             f"pass strategy=... to pick one.")
    allow = _tier_allows(tier)

    n = len(dims); ncols = ncols or n; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_per_plot * ncols, height_per_plot * nrows),
                             squeeze=False, sharey=sharey)
    axf = axes.flatten()
    cmap = plt.get_cmap("tab10")
    color_of = {g: cmap(BBOB_GROUP_ORDER.index(g) % 10) for g in BBOB_GROUP_ORDER}

    present = set()
    for ax, dim in zip(axf, dims):
        out, _ = _overall_by_size_fgroup(loaded[dim], strategy, dim, allow, cap=cap)
        order = [g for g in (groups or BBOB_GROUP_ORDER) if g in out]
        for g in order:
            present.add(g)
            xs = sorted(out[g])
            ax.plot(xs, [out[g][x] for x in xs], marker="o", markersize=5,
                    linewidth=1.8, color=color_of[g], label=_bbob_label(g))
        ax.set_title(f"dimension {dim}")
        ax.set_xlabel("sample size")
        ax.set_ylabel("overall CV (mean of feature medians)")
        if logy:
            ax.set_yscale("log")
        elif ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
    for ax in axf[n:]:
        ax.set_visible(False)

    ordered_present = [g for g in BBOB_GROUP_ORDER if g in present]
    handles = [Line2D([0], [0], color=color_of[g], marker="o", label=_bbob_label(g))
               for g in ordered_present]
    fig.legend(handles=handles, loc="upper right", title="BBOB function group")
    n_feat = _tier_feature_count(sources, tier, dims)
    fig.suptitle(title or f"Overall ELA within-instance CV vs sample size by "
                          f"function group  [{strategy}, {tier} tier, "
                          f"{n_feat} features]", fontsize=13)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Heatmap: BBOB function group x sample size                                   #
# --------------------------------------------------------------------------- #

def heatmap_cv_by_function_group(sources, strategy=None, tier="safe", groups=None,
                                 cap=None, include_overall=True, annotate=True,
                                 cmap_name="RdYlGn_r", vmin=0.0, vmax=None,
                                 width_per=3.2, row_height=0.55, title=None):
    """Overall CV as a heatmap: rows = BBOB function groups, cols = sample size,
    one panel per dimension, fixed strategy + tier. Cell = overall CV scoped to
    that group (identical value to plot_cv_vs_size_by_function_group). Low CV =
    green (stable); NaN = grey. Values clipped into [vmin, vmax].

    include_overall adds an "all functions" row (the global overall CV) under a
    separator line, for reference against the per-group rows. vmax defaults to
    the data max; set it explicitly to keep the colour scale comparable across
    strategies/dimensions/tiers.
    """
    dims = sorted(sources)
    loaded = {d: _resolve_source(sources, d) for d in dims}
    all_strats = sorted({split_config_key(c)[0] for d in dims for c in loaded[d]})
    if strategy is None:
        if len(all_strats) == 1:
            strategy = all_strats[0]
        else:
            raise ValueError(f"Multiple strategies present {all_strats}; "
                             f"pass strategy=... to pick one.")
    allow = _tier_allows(tier)

    per_dim, sizes_all, grp_present = {}, set(), set()
    for d in dims:
        out, _ = _overall_by_size_fgroup(loaded[d], strategy, d, allow, cap=cap)
        ov = None
        if include_overall:
            mean_by, *_ = _overall_by_size(loaded[d], strategy, d, allow, cap=cap)
            ov = mean_by
        per_dim[d] = (out, ov)
        for g, by_s in out.items():
            grp_present.add(g); sizes_all.update(by_s)
        if ov:
            sizes_all.update(ov)
    sizes = sorted(sizes_all)
    row_groups = [g for g in (groups or BBOB_GROUP_ORDER) if g in grp_present]
    row_labels = [_bbob_label(g) for g in row_groups]
    if include_overall:
        row_labels = row_labels + ["all functions"]

    mats, finite = {}, []
    R = len(row_groups) + (1 if include_overall else 0)
    for d in dims:
        out, ov = per_dim[d]
        M = np.full((R, len(sizes)), np.nan)
        for ri, g in enumerate(row_groups):
            for ci, s in enumerate(sizes):
                if s in out.get(g, {}):
                    M[ri, ci] = out[g][s]
        if include_overall and ov:
            for ci, s in enumerate(sizes):
                if s in ov:
                    M[-1, ci] = ov[s]
        mats[d] = M
        finite.append(M[np.isfinite(M)])
    allv = np.concatenate([a for a in finite if a.size]) if finite else np.array([])
    if vmax is None:
        vmax = float(allv.max()) * 1.02 if allv.size else 1.0

    n = len(dims)
    fig, axes = plt.subplots(1, n, figsize=(width_per * n + 1.4,
                             max(2.4, row_height * len(row_labels) + 1.4)),
                             squeeze=False, sharey=True, layout="constrained")
    axf = axes.flatten()
    cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad("lightgrey")
    im = None
    for ax, d in zip(axf, dims):
        M = np.clip(mats[d], vmin, vmax)
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(f"dim {d}", fontsize=11)
        ax.set_xticks(range(len(sizes))); ax.set_xticklabels(sizes, fontsize=8)
        ax.set_xlabel("sample size")
        ax.set_yticks(range(len(row_labels)))
        if include_overall:
            ax.axhline(len(row_groups) - 0.5, color="black", lw=1.1)
        if annotate:
            for ri in range(mats[d].shape[0]):
                for ci in range(mats[d].shape[1]):
                    v = mats[d][ri, ci]
                    if np.isfinite(v):
                        norm = (min(v, vmax) - vmin) / (vmax - vmin + 1e-12)
                        ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                                fontsize=7.5,
                                color="white" if norm > 0.6 else "black")
    axes[0][0].set_yticklabels(row_labels, fontsize=8)
    cbar = fig.colorbar(im, ax=list(axf), fraction=0.03, pad=0.02)
    cbar.set_label(f"overall CV  (green=stable; grey=NaN; clipped at {vmax:.2g})")
    fig.suptitle(title or f"Overall CV by BBOB function group  "
                          f"[{strategy}, {tier} tier]", fontsize=13)
    return fig, axes


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # sources: {dimension: "path/to/ela_cv_dim{d}.pkl"} exactly as plot_cv_vs_size.
    # from your_paths import SOURCES
    #
    # tier_composition(SOURCES)                       # audit the split first
    #
    # # plot_cv_vs_size equivalent, one tier at a time (strategies = lines):
    # fig = plot_cv_vs_size_tier(SOURCES, tier="safe", cap=1.0)
    # fig.savefig("cv_vs_size_safe.png", dpi=150, bbox_inches="tight")
    #
    # # or the tier-comparison version (tiers = lines, faceted by strategy):
    # fig, _ = plot_overall_cv_by_tier(SOURCES, strategies=["sobol"], cap=1.0, logy=True)
    #
    # # per BBOB function group (lines = groups, fixed strategy):
    # fig = plot_cv_vs_size_by_function_group(SOURCES, strategy="sobol",
    #                                         tier="safe", cap=1.0)
    # fig.savefig("cv_vs_size_by_fgroup.png", dpi=150, bbox_inches="tight")
    pass