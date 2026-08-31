"""
scatter_cv_vs_icc.py

Join within-instance CV (reproducibility, low = good) against ICC (between-
instance reliability, high = good) per feature, and scatter them so you can see
which features are BOTH stable and discriminative -- the only ones that actually
earn their place.

Inputs
------
cv_sources  : {dimension: path_or_dict}  from compute_ela_cv.py
              structure: ela_cv[cfg][(func, inst, dim)][group][feature] = cv
icc_sources : {dimension: path_or_dict}  from compute_ela_icc.py
              structure: ela_icc[cfg][(func, dim)][group][feature] = icc
Both use config keys like "lhs_random_cd_75" (strategy + size).

Join key
--------
CV has an extra instance axis; it is collapsed by median so both sides live at
(feature, func). Two granularities are offered:
  level="feature_func" : one point per (feature, func)   -> dense joint cloud
  level="feature"      : one point per feature, aggregated across funcs
                         (CV = median over func x inst; ICC = median over func)

Quadrants (x = CV, low good; y = ICC, high good)
  top-left     low CV / high ICC : reliable & reproducible   <- keep
  bottom-left  low CV / low  ICC : stable but non-discriminative (decoration)
  top-right    high CV / high ICC: discriminative but noisy (needs more budget)
  bottom-right high CV / low  ICC: unreliable                <- drop
"""

import pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_icc_boxplots import split_config_key
from ela_cv_features import CV_SAFE, CV_CAVEATED, CV_EXCLUDED


_SAFE = set(CV_SAFE)
_CAVEAT = set(CV_CAVEATED)
_EXCLUDED = set(CV_EXCLUDED)

TIER_COLOR = {"safe": "#2ca02c", "caveat": "#ff7f0e",
              "excluded": "#d62728", "unclassified": "#7f7f7f"}


def _tier_of(fn):
    if fn in _SAFE:
        return "safe"
    if fn in _CAVEAT:
        return "caveat"
    if fn in _EXCLUDED:
        return "excluded"
    return "unclassified"


def _is_runtime(name):
    return "costs_runtime" in name


def _finite(v):
    return v is not None and np.isfinite(v)


def _load(obj, dimension):
    """Accept a path (str/Path) or an already-loaded dict, or a {dim: ...} map."""
    if isinstance(obj, dict) and dimension in obj:
        obj = obj[dimension]
    if isinstance(obj, dict):
        return obj
    with open(obj, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Collect + join                                                              #
# --------------------------------------------------------------------------- #

def _cv_feat_func(ela_cv, strategy, dimension, size, cap=None):
    """cv[feature][func] = median over instances (list first, then collapse)."""
    raw = defaultdict(lambda: defaultdict(list))
    for cfg, by_key in ela_cv.items():
        st, sz = split_config_key(cfg)
        if st != strategy or sz != size:
            continue
        for key, by_group in by_key.items():
            func, _, dim = key
            if dim != dimension:
                continue
            for feats in by_group.values():
                for fn, v in feats.items():
                    if _is_runtime(fn) or not _finite(v):
                        continue
                    if cap is not None and v > cap:
                        continue
                    raw[fn][func].append(v)
    return raw   # keep lists; caller decides collapse


def _icc_feat_func(ela_icc, strategy, dimension, size):
    """icc[feature][func] = icc value (already per (func, dim))."""
    out = defaultdict(dict)
    for cfg, by_key in ela_icc.items():
        st, sz = split_config_key(cfg)
        if st != strategy or sz != size:
            continue
        for key, by_group in by_key.items():
            func, dim = key            # NB: ICC key is (func, dim), 2-tuple
            if dim != dimension:
                continue
            for feats in by_group.values():
                for fn, v in feats.items():
                    if _is_runtime(fn) or not _finite(v):
                        continue
                    out[fn][func] = float(v)
    return out


def join_cv_icc(cv_sources, icc_sources, dimension, strategy, size,
                level="feature", cap=None):
    """Return a list of dict rows: feature, func (or None), cv, icc, tier.

    level="feature_func" keeps every matched (feature, func) pair.
    level="feature" collapses to one row per feature (CV = median over the
    pooled func x inst cells; ICC = median over funcs)."""
    ela_cv = _load(cv_sources, dimension)
    ela_icc = _load(icc_sources, dimension)
    cv_raw = _cv_feat_func(ela_cv, strategy, dimension, size, cap=cap)
    icc = _icc_feat_func(ela_icc, strategy, dimension, size)

    feats = sorted(set(cv_raw) & set(icc))
    rows = []
    if level == "feature_func":
        for fn in feats:
            for func in sorted(set(cv_raw[fn]) & set(icc[fn])):
                lst = cv_raw[fn][func]
                if lst:
                    rows.append({"feature": fn, "func": func,
                                 "cv": float(np.median(lst)),
                                 "icc": icc[fn][func], "tier": _tier_of(fn)})
    elif level == "feature":
        for fn in feats:
            pooled = [v for lst in cv_raw[fn].values() for v in lst]
            iccs = list(icc[fn].values())
            if pooled and iccs:
                rows.append({"feature": fn, "func": None,
                             "cv": float(np.median(pooled)),
                             "icc": float(np.median(iccs)), "tier": _tier_of(fn)})
    else:
        raise ValueError("level must be 'feature' or 'feature_func'")
    return rows, sorted(set(cv_raw) - set(icc))   # also report CV feats w/o ICC


# --------------------------------------------------------------------------- #
# Quadrants                                                                    #
# --------------------------------------------------------------------------- #

QUADRANT_ORDER = ["keep", "noisy", "decoration", "unreliable"]
QUADRANT_LABEL = {
    "keep": "reliable & reproducible",
    "noisy": "discriminative but noisy",
    "decoration": "stable but non-discriminative",
    "unreliable": "unreliable",
}


def _quadrant(cv, icc, cv_thresh, icc_thresh):
    low_cv, high_icc = cv < cv_thresh, icc >= icc_thresh
    if low_cv and high_icc:
        return "keep"
    if not low_cv and high_icc:
        return "noisy"
    if low_cv and not high_icc:
        return "decoration"
    return "unreliable"


def assign_quadrants(rows, cv_thresh, icc_thresh):
    """Tag each row in place with a 'quadrant' key; return the same list."""
    for r in rows:
        r["quadrant"] = _quadrant(r["cv"], r["icc"], cv_thresh, icc_thresh)
    return rows


def group_by_quadrant(rows):
    """Return {quadrant: [rows]} in canonical order, each sorted best-first
    (highest ICC, then lowest CV)."""
    out = {q: [] for q in QUADRANT_ORDER}
    for r in rows:
        out[r.get("quadrant", "unreliable")].append(r)
    for q in out:
        out[q].sort(key=lambda r: (-r["icc"], r["cv"]))
    return out


def print_quadrant_report(rows, header="", indent="  "):
    """Print every feature grouped by quadrant, with its CV / ICC / tier."""
    by_q = group_by_quadrant(rows)
    if header:
        print(header)
    for q in QUADRANT_ORDER:
        items = by_q[q]
        print(f"{indent}[{QUADRANT_LABEL[q]}]  {len(items)} feature(s)")
        for r in items:
            fq = f"  f{r['func']}" if r.get("func") is not None else ""
            print(f"{indent}    {r['feature']:<34s} CV={r['cv']:.3f}  "
                  f"ICC={r['icc']:.3f}  ({r['tier']}){fq}")


# --------------------------------------------------------------------------- #
# Draw                                                                        #
# --------------------------------------------------------------------------- #

def _draw(ax, rows, cv_thresh, icc_thresh, xlim, ylim, annotate, label_fs=7):
    for tier, col in TIER_COLOR.items():
        pts = [(r["cv"], r["icc"]) for r in rows if r["tier"] == tier]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=34, color=col, alpha=0.8, edgecolor="white",
                       linewidth=0.4, label=tier, zorder=3)
    ax.axvline(cv_thresh, color="grey", ls="--", lw=1, zorder=1)
    ax.axhline(icc_thresh, color="grey", ls="--", lw=1, zorder=1)
    x0, x1 = xlim; y0, y1 = ylim
    ax.axvspan(x0, cv_thresh, y0, y1, alpha=0)  # keep limits stable
    # shade the "keep" quadrant
    ax.add_patch(plt.Rectangle((x0, icc_thresh), cv_thresh - x0, y1 - icc_thresh,
                               color="#2ca02c", alpha=0.06, zorder=0))
    q = dict(fontsize=8, alpha=0.55, ha="center", va="center", style="italic")
    ax.text((x0 + cv_thresh) / 2, (icc_thresh + y1) / 2, "reliable &\nreproducible", **q)
    ax.text((cv_thresh + x1) / 2, (icc_thresh + y1) / 2, "discriminative\nbut noisy", **q)
    ax.text((x0 + cv_thresh) / 2, (y0 + icc_thresh) / 2, "stable but\nnon-discriminative", **q)
    ax.text((cv_thresh + x1) / 2, (y0 + icc_thresh) / 2, "unreliable", **q)

    if annotate != "none":
        for r in rows:
            outside = (r["cv"] > cv_thresh) or (r["icc"] < icc_thresh)
            if annotate == "all" or (annotate == "outliers" and outside):
                ax.annotate(r["feature"], (r["cv"], r["icc"]), fontsize=label_fs,
                            xytext=(3, 3), textcoords="offset points", alpha=0.8)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("within-instance CV  (low = reproducible)")
    ax.set_ylabel("ICC  (high = reliable / discriminative)")
    ax.grid(True, alpha=0.25)


def scatter_cv_vs_icc(cv_sources, icc_sources, dimension, strategy, size,
                      level="feature", cap=None, cv_thresh=0.10, icc_thresh=0.75,
                      xlim=None, ylim=(0, 1), annotate="outliers", report=True,
                      figsize=(7.5, 6.5), title=None):
    """Single CV-vs-ICC scatter for one (strategy, size, dimension).

    Points coloured by tier (safe/caveat/excluded/unclassified). Dashed lines at
    cv_thresh / icc_thresh split the plane into the four quadrants; the top-left
    "keep" quadrant is shaded. Each returned row carries a 'quadrant' key; set
    report=True to also print the per-quadrant feature listing.
    Returns (fig, ax, rows)."""
    rows, cv_only = join_cv_icc(cv_sources, icc_sources, dimension, strategy, size,
                                level=level, cap=cap)
    if not rows:
        raise ValueError(f"No matched features for strategy={strategy!r}, "
                         f"size={size}, dim={dimension}.")
    rows = assign_quadrants(rows, cv_thresh, icc_thresh)
    if xlim is None:
        xmax = max(r["cv"] for r in rows)
        xlim = (0, max(cv_thresh * 1.5, xmax * 1.08))

    fig, ax = plt.subplots(figsize=figsize)
    _draw(ax, rows, cv_thresh, icc_thresh, xlim, ylim, annotate)
    ax.set_title(title or f"CV vs ICC  [{strategy}, n={size}, dim {dimension}, "
                          f"{level}]  ({len(rows)} points)")
    ax.legend(title="feature tier", loc="lower right", frameon=True, framealpha=0.9)
    if cv_only:
        print(f"{len(cv_only)} CV feature(s) had no ICC match and were dropped: "
              f"{cv_only[:8]}{' ...' if len(cv_only) > 8 else ''}")
    if report:
        print_quadrant_report(
            rows, header=f"\n[{strategy}, n={size}, dim {dimension}]  "
                         f"(CV<{cv_thresh:g}, ICC>={icc_thresh:g})")
    fig.tight_layout()
    return fig, ax, rows


def scatter_cv_vs_icc_over_size(cv_sources, icc_sources, dimension, strategy,
                                sizes=None, level="feature", cap=None,
                                cv_thresh=0.20, icc_thresh=0.75, xlim=None,
                                ylim=(0, 1), annotate="none", report=True,
                                ncols=None, width_per=5.2, height_per=4.6, title=None):
    """One scatter panel per sample size (fixed strategy + dimension), so you can
    watch features migrate toward the top-left "keep" quadrant as budget grows.
    annotate defaults to 'none' here to keep the small multiples readable; set
    report=True to print the per-quadrant feature listing for each size.
    Returns (fig, axes, per_size) where per_size[size] is the quadrant-tagged
    list of rows."""
    ela_cv = _load(cv_sources, dimension)
    if sizes is None:
        sizes = sorted({split_config_key(c)[1] for c in ela_cv
                        if split_config_key(c)[0] == strategy})
    per_size = {s: join_cv_icc(cv_sources, icc_sources, dimension, strategy, s,
                               level=level, cap=cap)[0] for s in sizes}
    per_size = {s: assign_quadrants(r, cv_thresh, icc_thresh)
                for s, r in per_size.items() if r}
    sizes = sorted(per_size)
    if not sizes:
        raise ValueError(f"No matched data for strategy={strategy!r}, dim={dimension}.")
    if xlim is None:
        xmax = max(r["cv"] for rows in per_size.values() for r in rows)
        xlim = (0, max(cv_thresh * 1.5, xmax * 1.08))

    n = len(sizes); ncols = ncols or n; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_per * ncols, height_per * nrows),
                             squeeze=False, sharex=True, sharey=True)
    axf = axes.flatten()
    for ax, s in zip(axf, sizes):
        _draw(ax, per_size[s], cv_thresh, icc_thresh, xlim, ylim, annotate)
        ax.set_title(f"n = {s}  ({len(per_size[s])} pts)")
    for ax in axf[n:]:
        ax.set_visible(False)
    handles = [Line2D([0], [0], marker="o", linestyle="none", color=c, label=t)
               for t, c in TIER_COLOR.items()]
    fig.legend(handles=handles, title="feature tier", loc="upper right")
    fig.suptitle(title or f"CV vs ICC across sample size  [{strategy}, dim {dimension}]",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if report:
        for s in sizes:
            print_quadrant_report(
                per_size[s], header=f"\n[{strategy}, n={s}, dim {dimension}]  "
                                    f"(CV<{cv_thresh:g}, ICC>={icc_thresh:g})")
    return fig, axes, per_size


if __name__ == "__main__":
    # CV_SOURCES  = {2: "…/ela_cv_dim2.pkl",  5: "…"}   # compute_ela_cv.py
    # ICC_SOURCES = {2: "…/ela_icc_dim2.pkl", 5: "…"}   # compute_ela_icc.py
    #
    # fig, ax, rows = scatter_cv_vs_icc(CV_SOURCES, ICC_SOURCES, dimension=5,
    #                                   strategy="sobol", size=250, cap=1.0)
    # fig.savefig("cv_vs_icc.png", dpi=150, bbox_inches="tight")
    #
    # fig, _ = scatter_cv_vs_icc_over_size(CV_SOURCES, ICC_SOURCES, dimension=5,
    #                                      strategy="sobol", cap=1.0)
    pass