"""
icc_plots.py

The ICC figure set, built on the DataFrames returned by
icc_by_function_group.run_group_analysis().

    res = run_group_analysis(ICC_SOURCES)

  1. plot_function_attainment(res["ceilings"])
     24 bars per dimension = p90 ICC at the largest budget, coloured by BBOB
     group. THE EVIDENCE FIGURE: shows the bimodal split (f1-f19 ~0.99 vs
     f20-f24 ~0.01) that makes pooling ICC across functions indefensible.

  2. plot_floor_map(res["long"])
     Share of cells sitting on the clipped-zero floor, features x function
     groups. Visualises the 13.4% floor finding: which features have no
     between-instance signal, and where.

  3. plot_icc_vs_budget(res["by_group"])
     ICC vs budget, one line per BBOB function group, WITH a second row showing
     zero_frac. Supersedes icc_by_function_group.plot_icc_by_function_group --
     a group median of 0.02 with 60% censored cells is not the same object as a
     clean 0.02, so the censoring must be shown alongside.

  4. heatmap_strategy_by_group(res["by_group"])
     Strategies x function groups, one panel per budget. THE RECOMMENDATION
     FIGURE: is the best sampler universal or conditional on landscape type?

Conventions applied throughout
------------------------------
* MEDIAN, never mean -- ICC negatives are clipped to 0 at source, so the
  distribution is left-censored.
* Fixed colour scale (vmin=0, vmax=1) on ICC heatmaps so panels are comparable.
* x-axis is the BUDGET MULTIPLIER (n = size x dimension), not absolute n.
* No aggregate ever spans BBOB function groups.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_icc_boxplots import BBOB_GROUP_ORDER, _bbob_label


def _group_colors(cmap_name="tab10"):
    cmap = plt.get_cmap(cmap_name)
    return {g: cmap(i % 10) for i, g in enumerate(BBOB_GROUP_ORDER)}


def _present_groups(df, col="fgroup", groups=None):
    have = set(df[col].unique())
    return [g for g in (groups or BBOB_GROUP_ORDER) if g in have]


def _txt_color(norm, light_bg_above=0.35):
    return "white" if norm < light_bg_above else "black"


# --------------------------------------------------------------------------- #
# 1. Per-function attainment (the evidence figure)                             #
# --------------------------------------------------------------------------- #

def plot_function_attainment(ceilings, dimensions=None, groups=None,
                             value="ceiling", ylim=(0, 1.05), ncols=None,
                             width_per=7.0, height_per=3.6, cmap_name="tab10",
                             legend="right", title=None):
    """24 bars per dimension: the ICC attained on each BBOB function.

    `value="ceiling"` is the p90 of ICC across features/strategies at the
    largest budget (the best any measurement achieves there); `value="median_icc"`
    is the typical feature instead. Bars are coloured by BBOB group and ordered
    by function id, with separators between groups, so the bimodal split is
    immediately visible.
    """
    df = ceilings.copy()
    dims = sorted(df["dimension"].unique()) if dimensions is None else dimensions
    order = _present_groups(df, "fgroup", groups)
    color_of = _group_colors(cmap_name)

    n = len(dims); ncols = ncols or 1; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_per * ncols, height_per * nrows),
                             squeeze=False, sharey=True)
    axf = axes.flatten()

    for ax, d in zip(axf, dims):
        sub = df[df["dimension"] == d].sort_values("func")
        funcs = sub["func"].tolist()
        vals = sub[value].tolist()
        cols = [color_of.get(g, "grey") for g in sub["fgroup"]]
        ax.bar(range(len(funcs)), vals, color=cols, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(funcs)))
        ax.set_xticklabels([f"f{f}" for f in funcs], fontsize=7, rotation=90)
        ax.set_ylabel(f"ICC attained ({'p90' if value == 'ceiling' else 'median'})")
        ax.set_title(f"dimension {d}")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=0.3)
        # separators between BBOB groups
        prev = None
        for i, g in enumerate(sub["fgroup"]):
            if prev is not None and g != prev:
                ax.axvline(i - 0.5, color="black", lw=0.8, alpha=0.5)
            prev = g
    for ax in axf[n:]:
        ax.set_visible(False)

    handles = [Patch(facecolor=color_of[g], label=_bbob_label(g)) for g in order]
    fig.suptitle(title or "ICC attainable per BBOB function (largest budget)",
                 fontsize=13)
    if legend == "right":
        # reserve the right margin, then anchor the legend into it
        fig.tight_layout(rect=[0, 0, 0.78, 0.95])
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.79, 0.5),
                   title="BBOB function group", frameon=False)
    elif legend == "top":
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94),
                   ncol=min(3, len(order)), title="BBOB function group",
                   frameon=False)
    elif legend == "bottom":
        fig.tight_layout(rect=[0, 0.07, 1, 0.95])
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                   ncol=min(3, len(order)), title="BBOB function group",
                   frameon=False)
    else:                                   # None/"none" -> no figure legend
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes


# --------------------------------------------------------------------------- #
# 2. Floor map (the censoring / invariance figure)                             #
# --------------------------------------------------------------------------- #

def floor_table(long_df, strategy=None, dimensions=None):
    """zero_frac per (dimension, feature, fgroup): share of cells clipped to 0."""
    df = long_df
    if strategy is not None:
        df = df[df["strategy"] == strategy]
    if dimensions is not None:
        df = df[df["dimension"].isin(dimensions)]
    out = (df.groupby(["dimension", "feature", "fgroup"])
           .agg(zero_frac=("icc", lambda s: float(np.mean(s <= 0))),
                median_icc=("icc", "median"), n_cells=("icc", "size"))
           .reset_index())
    return out


def plot_floor_map(long_df, strategy=None, dimensions=None, groups=None,
                   features=None, sort_by="zero_frac", annotate=True,
                   cmap_name="Reds", width_per=3.6, row_height=0.26, title=None):
    """Heatmap: rows = features, cols = BBOB function groups, cell = share of
    ICC cells on the clipped-zero floor. Dark = no between-instance signal.

    Pooled across strategies by default; pass strategy=... to isolate one (the
    pooled view is inflated by cma_random, which floors far more often).
    """
    ft = floor_table(long_df, strategy=strategy, dimensions=dimensions)
    dims = sorted(ft["dimension"].unique())
    order = _present_groups(ft, "fgroup", groups)

    rank = (ft.groupby("feature")["zero_frac"].mean()
            .sort_values(ascending=False))
    feats = [f for f in rank.index if features is None or f in set(features)]
    if sort_by != "zero_frac":
        feats = sorted(feats)

    n = len(dims)
    fig, axes = plt.subplots(1, n, figsize=(width_per * n + 2.6,
                             max(3.0, row_height * len(feats) + 1.8)),
                             squeeze=False, sharey=True, layout="constrained")
    axf = axes.flatten()
    cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad("lightgrey")
    im = None
    for ax, d in zip(axf, dims):
        sub = ft[ft["dimension"] == d]
        M = np.full((len(feats), len(order)), np.nan)
        for ri, f in enumerate(feats):
            for ci, g in enumerate(order):
                cell = sub[(sub["feature"] == f) & (sub["fgroup"] == g)]
                if len(cell):
                    M[ri, ci] = cell["zero_frac"].iloc[0]
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_title(f"dim {d}", fontsize=11)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([_bbob_label(g) for g in order], fontsize=7,
                           rotation=35, ha="right")
        ax.set_yticks(range(len(feats)))
        if annotate:
            for ri in range(M.shape[0]):
                for ci in range(M.shape[1]):
                    if np.isfinite(M[ri, ci]) and M[ri, ci] > 0:
                        ax.text(ci, ri, f"{M[ri, ci]*100:.0f}", ha="center",
                                va="center", fontsize=6,
                                color="white" if M[ri, ci] > 0.55 else "black")
    axes[0][0].set_yticklabels(feats, fontsize=6.5)
    cbar = fig.colorbar(im, ax=list(axf), fraction=0.03, pad=0.02)
    cbar.set_label("share of ICC cells clipped to 0 (dark = no signal)")
    tag = f" [{strategy}]" if strategy else " [all strategies]"
    fig.suptitle(title or f"Clipped-zero floor by feature and function group{tag}",
                 fontsize=13)
    return fig, axes


# --------------------------------------------------------------------------- #
# 3. ICC vs budget per group, with censoring shown                             #
# --------------------------------------------------------------------------- #

def plot_icc_vs_budget(by_group, strategy=None, groups=None, ylim=(0, 1),
                       show_zero_frac=True, width_per=5.0, height_per=3.6,
                       cmap_name="tab10", legend="right", title=None):
    """ICC vs budget multiplier, one line per BBOB function group, one column
    per dimension. Second row plots zero_frac for the same cells, so a median
    propped up (or held down) by censored cells is visible rather than hidden.
    """
    df = by_group
    strats = sorted(df["strategy"].unique())
    if strategy is None:
        if len(strats) == 1:
            strategy = strats[0]
        else:
            raise ValueError(f"Multiple strategies {strats}; pass strategy=...")
    df = df[df["strategy"] == strategy]
    dims = sorted(df["dimension"].unique())
    order = _present_groups(df, "fgroup", groups)
    color_of = _group_colors(cmap_name)

    nrows = 2 if show_zero_frac else 1
    fig, axes = plt.subplots(nrows, len(dims),
                             figsize=(width_per * len(dims), height_per * nrows),
                             squeeze=False, sharex="col", sharey="row")

    for ci, d in enumerate(dims):
        sub = df[df["dimension"] == d]
        ax = axes[0][ci]
        for g in order:
            s = sub[sub["fgroup"] == g].sort_values("size")
            if len(s):
                ax.plot(s["size"], s["icc"], marker="o", markersize=5,
                        linewidth=1.8, color=color_of[g], label=_bbob_label(g))
        ax.set_title(f"dimension {d}")
        ax.set_ylabel("ICC (median within group)")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        if show_zero_frac:
            ax2 = axes[1][ci]
            for g in order:
                s = sub[sub["fgroup"] == g].sort_values("size")
                if len(s):
                    ax2.plot(s["size"], s["zero_frac"], marker="s", markersize=4,
                             linewidth=1.4, linestyle="--", color=color_of[g])
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("share clipped to 0")
            ax2.set_xlabel("budget multiplier (n = size x d)")
            ax2.grid(True, alpha=0.3)
        else:
            ax.set_xlabel("budget multiplier (n = size x d)")

    handles = [Line2D([0], [0], color=color_of[g], marker="o",
                      label=_bbob_label(g)) for g in order]
    if show_zero_frac:
        handles.append(Line2D([0], [0], color="grey", marker="s", linestyle="--",
                              label="zero_frac (lower row)"))
    fig.suptitle(title or f"ICC vs budget by function group  [{strategy}]",
                 fontsize=13)
    if legend == "right":
        fig.tight_layout(rect=[0, 0, 0.82, 0.94])
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.83, 0.5),
                   title="BBOB function group", frameon=False)
    elif legend == "top":
        fig.tight_layout(rect=[0, 0, 1, 0.86])
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.93),
                   ncol=min(3, len(handles)), title="BBOB function group",
                   frameon=False)
    elif legend == "bottom":
        fig.tight_layout(rect=[0, 0.08, 1, 0.94])
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                   ncol=min(3, len(handles)), title="BBOB function group",
                   frameon=False)
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, axes


# --------------------------------------------------------------------------- #
# 4. Strategy x function group (the recommendation figure)                     #
# --------------------------------------------------------------------------- #

def heatmap_strategy_by_group(by_group, dimension, strategies=None, groups=None,
                              sizes=None, value="icc", annotate=True,
                              cmap_name="RdYlGn", vmin=0.0, vmax=1.0,
                              width_per=3.4, row_height=0.55, title=None):
    """Rows = sampling strategies, cols = BBOB function groups, one panel per
    budget, for a single dimension. Green = high ICC.

    Answers the recommendation question directly: is the best sampler universal
    or conditional on landscape type? Include `uniform` as the baseline -- if
    the low-discrepancy designs do not beat it, that is itself a result.
    """
    df = by_group[by_group["dimension"] == dimension]
    if df.empty:
        raise ValueError(f"No rows for dimension {dimension}.")
    strats = strategies or sorted(df["strategy"].unique())
    order = _present_groups(df, "fgroup", groups)
    sizes = sizes or sorted(df["size"].unique())

    n = len(sizes)
    fig, axes = plt.subplots(1, n, figsize=(width_per * n + 2.8,
                             max(2.6, row_height * len(strats) + 2.0)),
                             squeeze=False, sharey=True, layout="constrained")
    axf = axes.flatten()
    cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad("lightgrey")
    im = None
    for ax, sz in zip(axf, sizes):
        sub = df[df["size"] == sz]
        M = np.full((len(strats), len(order)), np.nan)
        for ri, st in enumerate(strats):
            for ci, g in enumerate(order):
                cell = sub[(sub["strategy"] == st) & (sub["fgroup"] == g)]
                if len(cell):
                    M[ri, ci] = cell[value].iloc[0]
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(f"budget x{sz}", fontsize=11)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([_bbob_label(g) for g in order], fontsize=7,
                           rotation=35, ha="right")
        ax.set_yticks(range(len(strats)))
        if annotate:
            for ri in range(M.shape[0]):
                for ci in range(M.shape[1]):
                    if np.isfinite(M[ri, ci]):
                        norm = (M[ri, ci] - vmin) / (vmax - vmin + 1e-12)
                        ax.text(ci, ri, f"{M[ri, ci]:.2f}", ha="center",
                                va="center", fontsize=7,
                                color=_txt_color(norm))
    axes[0][0].set_yticklabels(strats, fontsize=8)
    cbar = fig.colorbar(im, ax=list(axf), fraction=0.03, pad=0.02)
    cbar.set_label(f"{value} (green = reliable; grey = missing)")
    fig.suptitle(title or f"Sampling strategy x function group — dim {dimension}",
                 fontsize=13)
    return fig, axes


# --------------------------------------------------------------------------- #
# 5. Strategy comparison, with function scoping                                #
# --------------------------------------------------------------------------- #

def applicable_functions(ceilings, ceiling_min=0.5, shared_across_dims=False):
    """Which functions have enough between-instance signal to compare on.

    A function is "applicable" if its attainment ceiling (p90 ICC at the largest
    budget) reaches `ceiling_min`. Below that the ICC is floored for every
    strategy, so a strategy comparison there measures nothing -- all samplers tie
    at zero, and ties at a floor are not evidence of equivalence.

    Returns {dimension: set(func)}. With shared_across_dims=True every dimension
    uses the SAME function set (the intersection), so cross-dimension lines are
    computed on identical problems.

    NOTE: your ceilings are strongly bimodal (~0 or ~1), so the exact threshold
    barely matters -- anything in 0.1..0.7 gives the same partition.
    """
    per_dim = {}
    for d, g in ceilings.groupby("dimension"):
        per_dim[int(d)] = set(g.loc[g["ceiling"] >= ceiling_min, "func"].astype(int))
    if shared_across_dims and per_dim:
        common = set.intersection(*per_dim.values())
        per_dim = {d: set(common) for d in per_dim}
    return per_dim


def _strategy_curve(long_df, keep_funcs=None):
    """median ICC + zero_frac per (dimension, strategy, size), optionally
    restricted to keep_funcs = {dimension: set(func)}."""
    df = long_df
    if keep_funcs is not None:
        mask = df.apply(lambda r: r["func"] in keep_funcs.get(r["dimension"], set()),
                        axis=1)
        df = df[mask]
    return (df.groupby(["dimension", "strategy", "size"])
            .agg(icc=("icc", "median"),
                 zero_frac=("icc", lambda s: float(np.mean(s <= 0))),
                 n_func=("func", "nunique"), n_cells=("icc", "size"))
            .reset_index())


def plot_strategy_comparison(long_df, ceilings=None, scope="applicable",
                             ceiling_min=0.5, shared_across_dims=False,
                             strategies=None, ylim=(0, 1), show_zero_frac=False,
                             width_per=5.0, height_per=3.8, cmap_name="tab10",
                             title=None):
    """Median ICC vs budget, ONE LINE PER SAMPLING STRATEGY, panel per dimension.

    scope:
      "applicable" -- only functions whose ceiling >= ceiling_min (needs
                      `ceilings`). This is the meaningful comparison: strategies
                      are judged only where there is signal to detect.
      "all"        -- every function pooled. Comparable to the original
                      overall_icc_per_config number; included so you can show
                      how much the floored functions drag the curves down.
      "both"       -- two rows: applicable on top, all below, same axes.

    show_zero_frac adds a censoring row (share of cells clipped to 0).
    """
    scopes = ["applicable", "all"] if scope == "both" else [scope]
    if "applicable" in scopes and ceilings is None:
        raise ValueError("scope='applicable' needs the `ceilings` DataFrame.")

    keep = (applicable_functions(ceilings, ceiling_min, shared_across_dims)
            if ceilings is not None else None)
    curves = {}
    for sc in scopes:
        curves[sc] = _strategy_curve(long_df, keep if sc == "applicable" else None)

    dims = sorted(long_df["dimension"].unique())
    strats = strategies or sorted(long_df["strategy"].unique())
    cmap = plt.get_cmap(cmap_name)
    color_of = {s: cmap(i % 10) for i, s in enumerate(sorted(long_df["strategy"].unique()))}

    nrows = len(scopes) * (2 if show_zero_frac else 1)
    fig, axes = plt.subplots(nrows, len(dims),
                             figsize=(width_per * len(dims), height_per * nrows),
                             squeeze=False, sharex="col", sharey="row")

    row = 0
    for sc in scopes:
        cur = curves[sc]
        for ci, d in enumerate(dims):
            ax = axes[row][ci]
            sub = cur[cur["dimension"] == d]
            for st in strats:
                s = sub[sub["strategy"] == st].sort_values("size")
                if len(s):
                    ax.plot(s["size"], s["icc"], marker="o", markersize=5,
                            linewidth=1.8, color=color_of[st], label=st)
            nf = int(sub["n_func"].max()) if len(sub) else 0
            if row == 0:
                ax.set_title(f"dimension {d}")
            ax.set_ylabel(f"{sc}: median ICC\n({nf} functions)")
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("budget multiplier (n = size x d)")
        row += 1
        if show_zero_frac:
            for ci, d in enumerate(dims):
                ax = axes[row][ci]
                sub = cur[cur["dimension"] == d]
                for st in strats:
                    s = sub[sub["strategy"] == st].sort_values("size")
                    if len(s):
                        ax.plot(s["size"], s["zero_frac"], marker="s", markersize=4,
                                linestyle="--", linewidth=1.4, color=color_of[st])
                ax.set_ylim(0, 1)
                ax.set_ylabel(f"{sc}: share clipped to 0")
                ax.set_xlabel("budget multiplier (n = size x d)")
                ax.grid(True, alpha=0.3)
            row += 1

    handles = [Line2D([0], [0], color=color_of[s], marker="o", label=s)
               for s in strats]
    fig.tight_layout(rect=[0, 0, 0.84, 0.94])
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.85, 0.5),
               title="sampling strategy", frameon=False)
    excl = ""
    if keep is not None and "applicable" in scopes:
        dropped = sorted(set(long_df["func"].unique()) - keep.get(dims[0], set()))
        excl = f"  (excluded at dim {dims[0]}: {', '.join('f%d' % f for f in dropped)})"
    fig.suptitle(title or f"Sampling strategy comparison — median ICC vs budget{excl}",
                 fontsize=12)
    return fig, axes


if __name__ == "__main__":
    # from icc_by_function_group import run_group_analysis
    # res = run_group_analysis(ICC_SOURCES)
    #
    # fig, _ = plot_function_attainment(res["ceilings"])           # 1
    # fig, _ = plot_floor_map(res["long"], strategy="sobol")       # 2
    # fig, _ = plot_icc_vs_budget(res["by_group"], strategy="sobol")  # 3
    # fig, _ = heatmap_strategy_by_group(res["by_group"], dimension=5)  # 4
    #
    # # 5 — strategy comparison, only where there is signal to detect:
    # fig, _ = plot_strategy_comparison(res["long"], res["ceilings"],
    #                                   scope="applicable")
    # # both scopes stacked, to show how much the floored functions drag it down:
    # fig, _ = plot_strategy_comparison(res["long"], res["ceilings"], scope="both")
    pass