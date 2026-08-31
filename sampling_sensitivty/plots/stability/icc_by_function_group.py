"""
icc_by_function_group.py

Fix for the pooled-ceiling problem in the ICC analysis.

THE PROBLEM
-----------
ICC = sigma^2_between / (sigma^2_between + sigma^2_within), where the numerator
is fixed by how much a function's 100 BBOB instances genuinely differ. That
differs enormously by function:

  f1-f5   (separable, simple)      instances are near-trivial transformations
                                   -> tiny sigma^2_between -> low ICC CEILING
  f21-f24 (multimodal, weak struct) instances regenerate peak configurations
                                   -> large sigma^2_between -> high ICC ceiling

overall_icc_per_config() takes a median over ALL (func, feature) cells at once,
so it mixes 24 different ceilings into one number, and a strategy ranking on
that pooled median can be driven by the few high-ceiling functions.

THE FIX (two complementary outputs)
-----------------------------------
1. BREAKDOWN  -- report ICC per BBOB function group, never pooled across groups.
   icc_by_group() / plot_icc_by_function_group() / heatmap_icc_by_function_group()

2. CEILING-INVARIANT POOLING -- if you do want ONE number per config, do not
   average raw ICCs. Rank features WITHIN each function (where the ceiling is
   common to all features), then average those ranks across functions.
   icc_rank_within_function() returns a mean normalised rank in [0,1],
   comparable across functions by construction.

Also provides ceiling_diagnostic(), which quantifies the ceiling spread across
functions -- the evidence that the pooling was a problem in the first place.

Notes
-----
* ICC here is ICC(1,1) with 100 instances as subjects and 30 runs as replicates
  (see compute_ela_icc.py). Negatives are clipped to 0 at source, so a 0.0 is a
  CENSORED FLOOR, not a measurement: every summary below uses the median, and a
  zero_frac column exposes how much mass sits on that floor.
* No CV tier filtering is applied. The safe/caveat/excluded tiers encode
  ratio-scale validity, which ICC (a variance ratio) does not require.
"""

import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_icc_boxplots import (
    split_config_key, _FUNC_TO_GROUP, BBOB_GROUP_ORDER, _bbob_label,
)


def _is_runtime(name):
    return "costs_runtime" in name


def _finite(v):
    return v is not None and np.isfinite(v)


def _load(obj, dimension=None):
    """Accept a path, a loaded dict, or a {dim: path_or_dict} mapping."""
    if isinstance(obj, dict) and dimension is not None and dimension in obj:
        obj = obj[dimension]
    if isinstance(obj, dict):
        return obj
    with open(obj, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Long-form collection                                                        #
# --------------------------------------------------------------------------- #

def collect_icc_long(icc_sources, dimensions=None, features=None,
                     drop_nan=True):
    """Flatten the ICC pickles into a tidy DataFrame -- the base for everything.

    Columns: dimension, strategy, size, func, fgroup, feature_group, feature, icc
    `features` optionally restricts to a subset of feature names.
    """
    dims = sorted(icc_sources) if dimensions is None else dimensions
    keep = set(features) if features else None
    rows = []
    for d in dims:
        ela_icc = _load(icc_sources, d)
        for cfg, by_func in ela_icc.items():
            strategy, size = split_config_key(cfg)
            for key, by_group in by_func.items():
                func, dim = key                     # ICC key is (func, dim)
                if dim != d:
                    continue
                fgroup = _FUNC_TO_GROUP.get(func)
                if fgroup is None:
                    continue
                for grp, feats in by_group.items():
                    for fn, v in feats.items():
                        if _is_runtime(fn) or (keep and fn not in keep):
                            continue
                        if drop_nan and not _finite(v):
                            continue
                        rows.append({
                            "dimension": d, "strategy": strategy, "size": size,
                            "func": func, "fgroup": fgroup, "feature_group": grp,
                            "feature": fn,
                            "icc": float(v) if _finite(v) else np.nan,
                        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1. Breakdown by function group (the fix)                                    #
# --------------------------------------------------------------------------- #

def icc_by_group(long_df, agg="median"):
    """One row per (dimension, strategy, size, fgroup).

    icc      : agg of per-(func, feature) ICCs WITHIN that function group only
    zero_frac: share of cells sitting on the clipped-zero floor
    n_cells  : how many cells backed the estimate
    """
    aggf = "median" if agg == "median" else "mean"
    g = (long_df.groupby(["dimension", "strategy", "size", "fgroup"])
         .agg(icc=("icc", aggf),
              zero_frac=("icc", lambda s: float(np.mean(s <= 0))),
              n_cells=("icc", "size"))
         .reset_index())
    g["fgroup_label"] = g["fgroup"].map(_bbob_label)
    return g


def ceiling_diagnostic(long_df, at_size=None, q=0.90):
    """Quantify the per-function ICC ceiling -- the evidence for the fix.

    For each (dimension, func) take a high quantile of ICC across features and
    strategies (the best any feature achieves there) at the largest budget, plus
    the median. A wide spread of `ceiling` across functions means pooling raw
    ICCs across functions was mixing incomparable scales.
    """
    df = long_df
    if at_size is None:
        at_size = df["size"].max()
    df = df[df["size"] == at_size]
    out = (df.groupby(["dimension", "func", "fgroup"])
           .agg(ceiling=("icc", lambda s: float(np.quantile(s, q))),
                median_icc=("icc", "median"),
                zero_frac=("icc", lambda s: float(np.mean(s <= 0))),
                n_cells=("icc", "size"))
           .reset_index())
    out["fgroup_label"] = out["fgroup"].map(_bbob_label)
    out.attrs["at_size"] = at_size
    out.attrs["quantile"] = q
    return out.sort_values(["dimension", "ceiling"], ascending=[True, False])


# --------------------------------------------------------------------------- #
# 2. Ceiling-invariant pooling                                                #
# --------------------------------------------------------------------------- #

def icc_rank_within_function(long_df, by="feature"):
    """Ceiling-invariant aggregation.

    Within each (dimension, strategy, size, func) the ceiling is common to all
    features, so ranking features THERE is fair; those ranks are then comparable
    across functions. Rank is normalised to [0,1] (1 = best in that function),
    then averaged.

    by="feature"  -> mean normalised rank per feature (which features win,
                     ceiling-free)
    by="strategy" -> rank strategies within each (func, feature) instead, then
                     average: which sampler wins, free of both ceiling and
                     feature-composition effects.
    """
    df = long_df.copy()
    if by == "feature":
        keys = ["dimension", "strategy", "size", "func"]
        df["nrank"] = (df.groupby(keys)["icc"]
                       .rank(ascending=True, pct=True, method="average"))
        out = (df.groupby(["dimension", "strategy", "size", "feature"])
               .agg(mean_nrank=("nrank", "mean"),
                    median_icc=("icc", "median"),
                    n_func=("func", "nunique"))
               .reset_index())
        return out.sort_values(["dimension", "strategy", "size", "mean_nrank"],
                               ascending=[True, True, True, False])
    if by == "strategy":
        keys = ["dimension", "size", "func", "feature"]
        df["nrank"] = (df.groupby(keys)["icc"]
                       .rank(ascending=True, pct=True, method="average"))
        out = (df.groupby(["dimension", "size", "strategy"])
               .agg(mean_nrank=("nrank", "mean"),
                    median_icc=("icc", "median"),
                    n_cells=("nrank", "size"))
               .reset_index())
        out["rank"] = (out.groupby(["dimension", "size"])["mean_nrank"]
                       .rank(ascending=False, method="min").astype(int))
        return out.sort_values(["dimension", "size", "rank"])
    raise ValueError("by must be 'feature' or 'strategy'")


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #

def plot_icc_by_function_group(group_df, strategy=None, groups=None, ylim=(0, 1),
                               ncols=None, width_per=5.2, height_per=4.2,
                               title=None):
    """ICC vs budget, one line per BBOB function group, one panel per dimension.
    Fixed strategy (auto if only one present)."""
    df = group_df
    strats = sorted(df["strategy"].unique())
    if strategy is None:
        if len(strats) == 1:
            strategy = strats[0]
        else:
            raise ValueError(f"Multiple strategies {strats}; pass strategy=...")
    df = df[df["strategy"] == strategy]
    dims = sorted(df["dimension"].unique())
    order = [g for g in (groups or BBOB_GROUP_ORDER)
             if g in set(df["fgroup"].unique())]

    n = len(dims); ncols = ncols or n; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_per * ncols, height_per * nrows),
                             squeeze=False, sharey=True)
    axf = axes.flatten()
    cmap = plt.get_cmap("tab10")
    color_of = {g: cmap(BBOB_GROUP_ORDER.index(g) % 10) for g in BBOB_GROUP_ORDER}

    for ax, d in zip(axf, dims):
        sub = df[df["dimension"] == d]
        for g in order:
            s = sub[sub["fgroup"] == g].sort_values("size")
            if len(s):
                ax.plot(s["size"], s["icc"], marker="o", markersize=5,
                        linewidth=1.8, color=color_of[g], label=_bbob_label(g))
        ax.set_title(f"dimension {d}")
        ax.set_xlabel("budget multiplier (n = size x d)")
        ax.set_ylabel("ICC (median within group)")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
    for ax in axf[n:]:
        ax.set_visible(False)

    handles = [Line2D([0], [0], color=color_of[g], marker="o", label=_bbob_label(g))
               for g in order]
    fig.legend(handles=handles, loc="upper right", title="BBOB function group")
    fig.suptitle(title or f"ICC by BBOB function group  [{strategy}]", fontsize=13)
    fig.tight_layout()
    return fig, axes


def heatmap_icc_by_function_group(group_df, strategy=None, groups=None,
                                  annotate=True, cmap_name="RdYlGn",
                                  vmin=0.0, vmax=1.0, width_per=3.2,
                                  row_height=0.55, title=None):
    """Rows = BBOB function groups, cols = budget, panel = dimension.
    Green = high ICC (reliable). Fixed colour scale so panels are comparable."""
    df = group_df
    strats = sorted(df["strategy"].unique())
    if strategy is None:
        if len(strats) == 1:
            strategy = strats[0]
        else:
            raise ValueError(f"Multiple strategies {strats}; pass strategy=...")
    df = df[df["strategy"] == strategy]
    dims = sorted(df["dimension"].unique())
    sizes = sorted(df["size"].unique())
    order = [g for g in (groups or BBOB_GROUP_ORDER)
             if g in set(df["fgroup"].unique())]

    n = len(dims)
    fig, axes = plt.subplots(1, n, figsize=(width_per * n + 1.4,
                             max(2.4, row_height * len(order) + 1.6)),
                             squeeze=False, sharey=True, layout="constrained")
    axf = axes.flatten()
    cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad("lightgrey")
    im = None
    for ax, d in zip(axf, dims):
        sub = df[df["dimension"] == d]
        M = np.full((len(order), len(sizes)), np.nan)
        for ri, g in enumerate(order):
            for ci, s in enumerate(sizes):
                cell = sub[(sub["fgroup"] == g) & (sub["size"] == s)]
                if len(cell):
                    M[ri, ci] = cell["icc"].iloc[0]
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(f"dim {d}", fontsize=11)
        ax.set_xticks(range(len(sizes))); ax.set_xticklabels(sizes, fontsize=8)
        ax.set_xlabel("budget multiplier")
        ax.set_yticks(range(len(order)))
        if annotate:
            for ri in range(M.shape[0]):
                for ci in range(M.shape[1]):
                    if np.isfinite(M[ri, ci]):
                        norm = (M[ri, ci] - vmin) / (vmax - vmin + 1e-12)
                        ax.text(ci, ri, f"{M[ri, ci]:.2f}", ha="center",
                                va="center", fontsize=7.5,
                                color="white" if norm < 0.35 else "black")
    axes[0][0].set_yticklabels([_bbob_label(g) for g in order], fontsize=8)
    cbar = fig.colorbar(im, ax=list(axf), fraction=0.03, pad=0.02)
    cbar.set_label("ICC (green = reliable; grey = NaN)")
    fig.suptitle(title or f"ICC by BBOB function group  [{strategy}]", fontsize=13)
    return fig, axes


# --------------------------------------------------------------------------- #
# One-call entry point                                                        #
# --------------------------------------------------------------------------- #

def run_group_analysis(icc_sources, dimensions=None, strategy=None, agg="median",
                       features=None, verbose=True):
    """Collect once, return every table as a DataFrame.

    Returns dict:
      long        : tidy per-(func, feature) ICC rows
      by_group    : ICC per (dimension, strategy, size, fgroup)   <- the fix
      ceilings    : per-function ceiling diagnostic               <- the evidence
      rank_feature: ceiling-invariant mean normalised rank per feature
      rank_strategy: ceiling-invariant strategy ranking per (dim, size)
    """
    long = collect_icc_long(icc_sources, dimensions, features=features)
    if long.empty:
        raise ValueError("No ICC rows collected -- check paths/dimensions.")
    by_group = icc_by_group(long, agg=agg)
    ceilings = ceiling_diagnostic(long)
    rank_feature = icc_rank_within_function(long, by="feature")
    rank_strategy = icc_rank_within_function(long, by="strategy")

    if verbose:
        print(f"collected {len(long):,} ICC cells | "
              f"{long['feature'].nunique()} features | "
              f"{long['func'].nunique()} functions | "
              f"dims {sorted(long['dimension'].unique())}")
        sz = ceilings.attrs.get("at_size")
        print(f"\n=== per-function ICC ceiling (p90 across features, n={sz}) ===")
        for d in sorted(ceilings["dimension"].unique()):
            sub = ceilings[ceilings["dimension"] == d]
            spread = sub["ceiling"].max() - sub["ceiling"].min()
            print(f"  dim {d}: ceiling ranges {sub['ceiling'].min():.3f} .. "
                  f"{sub['ceiling'].max():.3f}  (spread {spread:.3f})")
            byg = (sub.groupby("fgroup_label")["ceiling"].median()
                   .sort_values(ascending=False))
            for lbl, v in byg.items():
                print(f"      {lbl:<34s} median ceiling {v:.3f}")
        print("\n  -> a wide ceiling spread means pooling raw ICCs across "
              "functions mixes incomparable scales; use by_group or the "
              "rank-based aggregation instead.")
    return {"long": long, "by_group": by_group, "ceilings": ceilings,
            "rank_feature": rank_feature, "rank_strategy": rank_strategy}


if __name__ == "__main__":
    # ICC_SOURCES = {2: ".../dim2featELA/ela_icc_results.pkl",
    #                5: ".../dim5featELA/ela_icc_results.pkl",
    #                10: ".../dim10featELA/ela_icc_results.pkl"}
    #
    # res = run_group_analysis(ICC_SOURCES)
    # res["by_group"]      # the un-pooled breakdown
    # res["ceilings"]      # evidence the ceilings differ
    # res["rank_strategy"] # ceiling-invariant sampler ranking
    #
    # fig, _ = plot_icc_by_function_group(res["by_group"], strategy="sobol")
    # fig, _ = heatmap_icc_by_function_group(res["by_group"], strategy="sobol")
    pass