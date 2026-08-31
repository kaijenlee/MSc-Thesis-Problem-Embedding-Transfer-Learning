"""
repro_plots.py

Figure set for the per-function reproducibility study. Consumes the tables from
repro_analysis.prepare().

    from compute_ela_repro import load_repro, repro_to_dataframe
    from repro_analysis import prepare
    import repro_plots as rp

    df  = repro_to_dataframe(load_repro(path))
    tab = prepare(df)

DESCRIPTIVE
  F1 heatmap_feature_function   features x 24 functions, cell = normalised sigma
  F2 boxplot_features           per-feature spread ACROSS functions
SAMPLER COMPARISON (ordinal -- never pools sigma)
  F3 barplot_win_counts         wins across the 24 functions
  F4 heatmap_strategy_rank      strategies x functions, cell = rank
  F5 dotplot_per_function       unaggregated evidence for selected functions
CONVERGENCE RATE (alpha is dimensionless -> pooling IS legitimate)
  F6 loglog_convergence         log sigma vs log n with fitted slopes
  F7 alpha_distribution         violin of alpha per strategy, refs at 0.5 / 1.0
  F8 heatmap_alpha              features x functions, cell = alpha
DIAGNOSTICS (methods section)
  D1 plot_validity_audit        crosses_zero rate per feature
  D2 plot_sd_vs_mad             heavy-tail check
  D3 heatmap_n_unique           discrete/degenerate features

Conventions: median throughout (heavy tails); function axis always ordered by
id with BBOB group separators; legends outside the axes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from repro_analysis import (GROUP_ORDER, GROUP_LABEL, FUNC_TO_GROUP,
                            GROUP_BOUNDARIES)


def _group_colors(cmap_name="tab10"):
    cmap = plt.get_cmap(cmap_name)
    return {g: cmap(i % 10) for i, g in enumerate(GROUP_ORDER)}


def _strategy_colors(strats, cmap_name="tab10"):
    cmap = plt.get_cmap(cmap_name)
    return {s: cmap(i % 10) for i, s in enumerate(sorted(strats))}


def _func_separators(ax, funcs, axis="x"):
    """Vertical/horizontal lines at BBOB group boundaries."""
    pos = {f: i for i, f in enumerate(funcs)}
    for b in GROUP_BOUNDARIES:
        nxt = [f for f in funcs if f > b]
        if nxt and b in pos:
            line = pos[b] + 0.5
            (ax.axvline if axis == "x" else ax.axhline)(
                line, color="black", lw=0.8, alpha=0.5)


def _legend_right(fig, handles, title, rect=(0, 0, 0.82, 0.94), x=0.83):
    fig.tight_layout(rect=list(rect))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(x, 0.5),
               title=title, frameon=False)


# --------------------------------------------------------------------------- #
# F1 — feature x function heatmap                                              #
# --------------------------------------------------------------------------- #

def heatmap_feature_function(per_func, dimension, strategy, size, value="value_norm",
                             sort_features=True, vmax=None, cmap_name="magma_r",
                             annotate=False, width=13.0, row_height=0.24, title=None):
    """Rows = features, cols = 24 functions, cell = normalised sigma (low=good).

    The core descriptive figure: feature reproducibility AND its landscape
    dependence in one view, with nothing pooled. Group separators mark the BBOB
    boundaries so group-alignment can be seen rather than assumed.
    """
    d = per_func[(per_func["dimension"] == dimension) &
                 (per_func["strategy"] == strategy) &
                 (per_func["size"] == size)]
    if d.empty:
        raise ValueError(f"No rows for dim={dimension}, {strategy}, size={size}.")
    funcs = sorted(d["func"].unique())
    order = (d.groupby("feature")[value].median().sort_values().index.tolist()
             if sort_features else sorted(d["feature"].unique()))

    M = (d.pivot_table(index="feature", columns="func", values=value)
         .reindex(index=order, columns=funcs))
    if vmax is None:
        vmax = float(np.nanpercentile(M.to_numpy(), 95))

    fig, ax = plt.subplots(figsize=(width, max(3.0, row_height * len(order) + 2.0)))
    cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad("lightgrey")
    im = ax.imshow(M.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(funcs)))
    ax.set_xticklabels([f"f{f}" for f in funcs], fontsize=7, rotation=90)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6.5)
    _func_separators(ax, funcs, "x")
    if annotate:
        A = M.to_numpy()
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                            fontsize=5,
                            color="white" if A[i, j] > vmax * 0.6 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label(f"{value}  (run noise / feature range; low = reproducible)")
    ax.set_xlabel("BBOB function")
    ax.set_title(title or f"Feature reproducibility per function  "
                          f"[{strategy}, n={size}xd, dim {dimension}]")
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# F2 — per-feature boxplots across functions                                   #
# --------------------------------------------------------------------------- #

def boxplot_features(per_func, dimension, strategy, size, value="value_norm",
                     top=None, logy=True, width=13.0, height=6.0, title=None):
    """One box per feature = distribution across the 24 functions, sorted by
    median. Keeps cross-function variation visible as spread instead of hiding
    it in a mean."""
    d = per_func[(per_func["dimension"] == dimension) &
                 (per_func["strategy"] == strategy) &
                 (per_func["size"] == size)]
    d = d[np.isfinite(d[value])]
    order = d.groupby("feature")[value].median().sort_values().index.tolist()
    if top:
        order = order[:top]
    data = [d.loc[d["feature"] == f, value].to_numpy() for f in order]

    fig, ax = plt.subplots(figsize=(width, height))
    bp = ax.boxplot(data, patch_artist=True, widths=0.65,
                    medianprops=dict(color="black", lw=1.1),
                    flierprops=dict(marker="o", ms=2.5, alpha=0.4,
                                    markerfacecolor="grey", markeredgecolor="none"))
    for p in bp["boxes"]:
        p.set_facecolor("#4c72b0"); p.set_alpha(0.65); p.set_edgecolor("#4c72b0")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(f"{value} across functions")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(title or f"Per-feature reproducibility, spread across the 24 "
                          f"functions  [{strategy}, n={size}xd, dim {dimension}]")
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# F3 — win counts                                                              #
# --------------------------------------------------------------------------- #

def barplot_win_counts(wins, sizes=None, width_per=4.6, height_per=3.8, title=None):
    """Bars = strategies, height = number of functions won. Panels = budget,
    rows = dimension. Ordinal, so no sigma is ever averaged."""
    d = wins if sizes is None else wins[wins["size"].isin(sizes)]
    dims = sorted(d["dimension"].unique())
    szs = sorted(d["size"].unique())
    strats = sorted(d["strategy"].unique())
    color_of = _strategy_colors(strats)

    fig, axes = plt.subplots(len(dims), len(szs),
                             figsize=(width_per * len(szs), height_per * len(dims)),
                             squeeze=False, sharey=True)
    for r, dim in enumerate(dims):
        for c, sz in enumerate(szs):
            ax = axes[r][c]
            sub = d[(d["dimension"] == dim) & (d["size"] == sz)]
            sub = sub.set_index("strategy").reindex(strats)
            ax.bar(range(len(strats)), sub["wins"].fillna(0),
                   color=[color_of[s] for s in strats], edgecolor="white")
            ax.set_xticks(range(len(strats)))
            ax.set_xticklabels(strats, rotation=35, ha="right", fontsize=8)
            if r == 0:
                ax.set_title(f"budget x{sz}")
            if c == 0:
                ax.set_ylabel(f"dim {dim}\nfunctions won (of 24)")
            ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title or "Sampler comparison — functions won "
                          "(ranked within each function, never pooled)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes


# --------------------------------------------------------------------------- #
# F4 — strategy x function rank heatmap                                        #
# --------------------------------------------------------------------------- #

def heatmap_strategy_rank(sbf, dimension, size, value="func_rank",
                          cmap_name="RdYlGn_r", width=13.0, row_height=0.6,
                          annotate=True, title=None):
    """Rows = strategies, cols = 24 functions, cell = rank (1 = best).
    Exposes whether a sampler's advantage is universal or function-specific --
    the detail that F3's win count collapses."""
    d = sbf[(sbf["dimension"] == dimension) & (sbf["size"] == size)]
    if d.empty:
        raise ValueError(f"No rows for dim={dimension}, size={size}.")
    funcs = sorted(d["func"].unique())
    strats = sorted(d["strategy"].unique())
    M = (d.pivot_table(index="strategy", columns="func", values=value)
         .reindex(index=strats, columns=funcs))

    fig, ax = plt.subplots(figsize=(width, max(2.4, row_height * len(strats) + 1.8)))
    cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad("lightgrey")
    A = M.to_numpy()
    im = ax.imshow(A, aspect="auto", cmap=cmap, vmin=1, vmax=len(strats),
                   interpolation="nearest")
    ax.set_xticks(range(len(funcs)))
    ax.set_xticklabels([f"f{f}" for f in funcs], fontsize=7, rotation=90)
    ax.set_yticks(range(len(strats))); ax.set_yticklabels(strats, fontsize=8)
    _func_separators(ax, funcs, "x")
    if annotate:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    ax.text(j, i, f"{int(A[i, j])}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if A[i, j] > len(strats) * 0.7 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("rank within function (1 = most reproducible)")
    ax.set_xlabel("BBOB function")
    ax.set_title(title or f"Strategy rank per function  [n={size}xd, dim {dimension}]")
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# F5 — per-function dot plot (unaggregated evidence)                           #
# --------------------------------------------------------------------------- #

def dotplot_per_function(per_func, dimension, size, funcs, value="value_norm",
                         logy=True, ncols=4, width_per=3.4, height_per=3.0,
                         title=None):
    """For selected functions: one panel each, sigma per strategy with the
    across-feature spread as a boxplot. The raw evidence behind F3/F4."""
    d = per_func[(per_func["dimension"] == dimension) &
                 (per_func["size"] == size) & (per_func["func"].isin(funcs))]
    d = d[np.isfinite(d[value])]
    strats = sorted(d["strategy"].unique())
    color_of = _strategy_colors(strats)

    n = len(funcs); ncols = min(ncols, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_per * ncols, height_per * nrows),
                             squeeze=False, sharey=logy)
    axf = axes.flatten()
    for ax, f in zip(axf, funcs):
        sub = d[d["func"] == f]
        data = [sub.loc[sub["strategy"] == s, value].to_numpy() for s in strats]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", lw=1.0),
                        flierprops=dict(marker="o", ms=2, alpha=0.3,
                                        markeredgecolor="none"))
        for p, s in zip(bp["boxes"], strats):
            p.set_facecolor(color_of[s]); p.set_alpha(0.65)
            p.set_edgecolor(color_of[s])
        ax.set_xticks(range(1, len(strats) + 1))
        ax.set_xticklabels(strats, rotation=40, ha="right", fontsize=7)
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"f{f} — {GROUP_LABEL[FUNC_TO_GROUP[f]].split(' (')[0]}",
                     fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    for ax in axf[n:]:
        ax.set_visible(False)
    for r in range(nrows):
        axes[r][0].set_ylabel(value)
    fig.suptitle(title or f"Per-function sampler comparison  "
                          f"[n={size}xd, dim {dimension}]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes


# --------------------------------------------------------------------------- #
# F6 — log-log convergence                                                     #
# --------------------------------------------------------------------------- #

def loglog_convergence(per_func, alpha, dimension, funcs, feature=None,
                       value="value", ncols=4, width_per=3.4, height_per=3.0,
                       annotate_slope=True, title=None):
    """log sigma vs log n, one line per strategy, one panel per function.

    THE key figure: parallel lines => low-discrepancy designs buy a better
    CONSTANT; different slopes => they buy a better RATE. Points are drawn as
    well as the fit, because four budget values is a thin basis for a slope.

    If `feature` is None the per-panel value is the median across features
    (a level summary within one function, which is legitimate).
    """
    d = per_func[(per_func["dimension"] == dimension) & (per_func["func"].isin(funcs))]
    if feature:
        d = d[d["feature"] == feature]
    d = d[(d[value] > 0) & np.isfinite(d[value])]
    strats = sorted(d["strategy"].unique())
    color_of = _strategy_colors(strats)

    n = len(funcs); ncols = min(ncols, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_per * ncols, height_per * nrows),
                             squeeze=False)
    axf = axes.flatten()
    for ax, f in zip(axf, funcs):
        sub = d[d["func"] == f]
        for s in strats:
            ss = sub[sub["strategy"] == s]
            if ss.empty:
                continue
            agg = (ss.groupby("size")[value].median().reset_index()
                   .sort_values("size"))
            nn = agg["size"]
            ax.plot(nn, agg[value], marker="o", ms=4.5, lw=1.6,
                    color=color_of[s], label=s)
            if annotate_slope and len(agg) >= 3:
                x, y = np.log(nn.to_numpy(float)), np.log(agg[value].to_numpy(float))
                b = np.polyfit(x, y, 1)[0]
                ax.plot(nn, np.exp(np.polyval(np.polyfit(x, y, 1), x)),
                        color=color_of[s], lw=0.8, ls="--", alpha=0.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"f{f}", fontsize=10)
        ax.set_xlabel("budget multiplier (n = size x d)")
        ax.grid(True, which="both", alpha=0.25)
    for ax in axf[n:]:
        ax.set_visible(False)
    for r in range(nrows):
        axes[r][0].set_ylabel(f"sigma ({value})")

    handles = [Line2D([0], [0], color=color_of[s], marker="o", label=s)
               for s in strats]
    fig.suptitle(title or f"Convergence of run-to-run noise  "
                          f"[dim {dimension}{', ' + feature if feature else ''}]",
                 fontsize=12)
    _legend_right(fig, handles, "sampling strategy", rect=(0, 0, 0.84, 0.94), x=0.85)
    return fig, axes


# --------------------------------------------------------------------------- #
# F7 — alpha distribution                                                      #
# --------------------------------------------------------------------------- #

def alpha_distribution(alpha, dimension=None, strategies=None, r2_min=0.0,
                       xlim=(-0.2, 1.4), width=9.0, height=5.0, title=None):
    """Violin of fitted alpha per strategy, with reference lines at 0.5 (plain
    Monte Carlo) and 1.0 (best-case QMC / optimised LHS).

    alpha is dimensionless, so pooling across functions and features here IS
    legitimate -- this is the one figure that may do so."""
    d = alpha if dimension is None else alpha[alpha["dimension"] == dimension]
    d = d[np.isfinite(d["alpha"]) & (d["r2"] >= r2_min)]
    strats = strategies or sorted(d["strategy"].unique())
    color_of = _strategy_colors(strats)
    data = [d.loc[d["strategy"] == s, "alpha"].to_numpy() for s in strats]
    data = [a for a in data]

    fig, ax = plt.subplots(figsize=(width, height))
    parts = ax.violinplot(data, vert=False, showmedians=True, widths=0.85)
    for pc, s in zip(parts["bodies"], strats):
        pc.set_facecolor(color_of[s]); pc.set_alpha(0.65)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color("black"); parts[key].set_linewidth(1.0)
    ax.axvline(0.5, color="grey", ls="--", lw=1.2)
    ax.axvline(1.0, color="grey", ls=":", lw=1.2)
    ax.text(0.5, len(strats) + 0.55, "  Monte Carlo (1/sqrt(n))", fontsize=8,
            color="grey", va="center")
    ax.text(1.0, len(strats) + 0.55, "  QMC best case (1/n)", fontsize=8,
            color="grey", va="center")
    ax.set_yticks(range(1, len(strats) + 1)); ax.set_yticklabels(strats)
    ax.set_xlim(*xlim)
    ax.set_xlabel("convergence exponent alpha   (sigma ~ n^-alpha; higher = faster)")
    ax.grid(True, axis="x", alpha=0.3)
    med = d.groupby("strategy")["alpha"].median()
    for i, s in enumerate(strats, start=1):
        if s in med:
            ax.text(xlim[1] - 0.02, i, f"med {med[s]:.2f}", fontsize=8,
                    ha="right", va="center")
    dtag = f", dim {dimension}" if dimension is not None else ""
    ax.set_title(title or f"Convergence exponent by strategy{dtag} "
                          f"(pooled over functions & features)")
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# F8 — alpha heatmap                                                           #
# --------------------------------------------------------------------------- #

def heatmap_alpha(alpha, dimension, strategy, vmin=0.0, vmax=1.2,
                  cmap_name="viridis", width=13.0, row_height=0.24, title=None):
    """Features x functions, cell = alpha. Shows whether the convergence rate is
    a property of the FEATURE (row stripes), the LANDSCAPE (column stripes), or
    neither."""
    d = alpha[(alpha["dimension"] == dimension) & (alpha["strategy"] == strategy)]
    if d.empty:
        raise ValueError(f"No alpha fits for dim={dimension}, {strategy}.")
    funcs = sorted(d["func"].unique())
    order = d.groupby("feature")["alpha"].median().sort_values(
        ascending=False).index.tolist()
    M = (d.pivot_table(index="feature", columns="func", values="alpha")
         .reindex(index=order, columns=funcs))

    fig, ax = plt.subplots(figsize=(width, max(3.0, row_height * len(order) + 2.0)))
    cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad("lightgrey")
    im = ax.imshow(M.to_numpy(), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(funcs)))
    ax.set_xticklabels([f"f{f}" for f in funcs], fontsize=7, rotation=90)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6.5)
    _func_separators(ax, funcs, "x")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("alpha  (sigma ~ n^-alpha; 0.5 = Monte Carlo)")
    ax.set_xlabel("BBOB function")
    ax.set_title(title or f"Convergence exponent per feature and function  "
                          f"[{strategy}, dim {dimension}]")
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# Diagnostics                                                                  #
# --------------------------------------------------------------------------- #

def plot_validity_audit(df, dimension=None, tiers=None, width=12.0, height=5.0,
                        title=None):
    """D1 — share of cells where the feature's 30 runs straddle zero, per
    feature. Turns the CV validity tiering into an empirical check: a
    safe-tier feature with a nonzero rate is a tiering error.

    `tiers` optional {feature: tier} for colouring (e.g. from ela_cv_features).
    """
    d = df if dimension is None else df[df["dimension"] == dimension]
    g = (d.groupby("feature")
         .agg(crosses=("crosses_zero", "mean"),
              positive=("all_positive", "mean"))
         .reset_index().sort_values("crosses", ascending=False))
    tier_color = {"safe": "#2ca02c", "caveat": "#ff7f0e",
                  "excluded": "#d62728", "unclassified": "#7f7f7f"}
    cols = ([tier_color.get(tiers.get(f, "unclassified"), "#7f7f7f")
             for f in g["feature"]] if tiers else "#4c72b0")

    fig, ax = plt.subplots(figsize=(width, height))
    ax.bar(range(len(g)), g["crosses"], color=cols, edgecolor="white")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g["feature"], rotation=90, fontsize=7)
    ax.set_ylabel("share of cells whose 30 runs straddle zero")
    ax.grid(True, axis="y", alpha=0.3)
    if tiers:
        ax.legend(handles=[Patch(facecolor=c, label=t)
                           for t, c in tier_color.items()],
                  title="CV tier", frameon=False, loc="upper right")
    ax.set_title(title or "CV validity audit — sign-crossing features cannot "
                          "use CV")
    fig.tight_layout()
    return fig, ax


def plot_sd_vs_mad(df, dimension=None, sample=20000, width=6.8, height=6.4,
                   title=None):
    """D2 — sd vs mad, log-log, with y=x. Points above the diagonal are
    heavy-tailed, so median (not mean) aggregation is required for them."""
    d = df if dimension is None else df[df["dimension"] == dimension]
    d = d[(d["sd"] > 0) & (d["mad"] > 0)]
    if len(d) > sample:
        d = d.sample(sample, random_state=0)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.scatter(d["mad"], d["sd"], s=4, alpha=0.15, edgecolor="none",
               color="#4c72b0")
    lo = float(min(d["mad"].min(), d["sd"].min()))
    hi = float(max(d["mad"].max(), d["sd"].max()))
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls="--", label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("MAD (scaled, robust)"); ax.set_ylabel("SD")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title(title or "Heavy-tail diagnostic — points above y=x have "
                          "outlier-driven SD")
    fig.tight_layout()
    return fig, ax


def heatmap_n_unique(df, cmap_name="viridis", width=8.0, row_height=0.24,
                     vmax=30, title=None):
    """D3 — median distinct values among the 30 runs, features x dimension.
    Small counts mark discrete features where dispersion measures degenerate."""
    g = (df.groupby(["feature", "dimension"])["n_unique"].median()
         .reset_index())
    order = (g.groupby("feature")["n_unique"].min().sort_values()
             .index.tolist())
    dims = sorted(g["dimension"].unique())
    M = g.pivot_table(index="feature", columns="dimension",
                      values="n_unique").reindex(index=order, columns=dims)

    fig, ax = plt.subplots(figsize=(width, max(3.0, row_height * len(order) + 1.8)))
    cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad("lightgrey")
    A = M.to_numpy()
    im = ax.imshow(A, aspect="auto", cmap=cmap, vmin=1, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([f"dim {d}" for d in dims], fontsize=9)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6.5)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.isfinite(A[i, j]):
                ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center",
                        fontsize=6, color="white" if A[i, j] < vmax * 0.6 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("median distinct values among 30 runs")
    ax.set_title(title or "Degeneracy check — low counts mark discrete features")
    fig.tight_layout()
    return fig, ax


# =========================================================================== #
# TIER 1 — raw SD (comparable WITHIN a feature; no assumptions)               #
# =========================================================================== #

def plot_sd_vs_budget(per_func, dimension, feature, funcs=None, value="value",
                      ncols=6, width_per=2.6, height_per=2.3, logy=True,
                      title=None):
    """S1 — SD vs budget for ONE feature, one panel per function.

    Raw SD in feature units. Because the feature is fixed, every comparison in
    this figure (strategy, budget, function) is like-for-like and needs no
    normaliser -- the most defensible view in the set. Show 2-3 representative
    features rather than all 46.
    """
    d = per_func[(per_func["dimension"] == dimension) &
                 (per_func["feature"] == feature)]
    d = d[np.isfinite(d[value]) & (d[value] > 0)]
    if d.empty:
        raise ValueError(f"No rows for {feature!r} at dim {dimension}.")
    funcs = funcs or sorted(d["func"].unique())
    strats = sorted(d["strategy"].unique())
    color_of = _strategy_colors(strats)

    n = len(funcs); ncols = min(ncols, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_per * ncols, height_per * nrows),
                             squeeze=False, sharex=True, sharey=logy)
    axf = axes.flatten()
    for ax, f in zip(axf, funcs):
        sub = d[d["func"] == f]
        for s in strats:
            ss = sub[sub["strategy"] == s].sort_values("size")
            if len(ss):
                ax.plot(ss["size"], ss[value], marker="o", ms=3.5,
                        lw=1.4, color=color_of[s], label=s)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"f{f}", fontsize=9)
        ax.grid(True, which="both", alpha=0.25)
    for ax in axf[n:]:
        ax.set_visible(False)
    for r in range(nrows):
        axes[r][0].set_ylabel("SD (feature units)", fontsize=8)
    for c in range(ncols):
        axes[nrows - 1][c].set_xlabel("budget multiplier (n = size x d)", fontsize=8)

    handles = [Line2D([0], [0], color=color_of[s], marker="o", label=s)
               for s in strats]
    fig.suptitle(title or f"Run-to-run SD vs budget — {feature}  [dim {dimension}]",
                 fontsize=12)
    _legend_right(fig, handles, "sampling strategy", rect=(0, 0, 0.86, 0.95), x=0.87)
    return fig, axes


# =========================================================================== #
# TIER 2 — SD ratios (dimensionless -> poolable without a normaliser)         #
# =========================================================================== #

def boxplot_sampler_ratio(ratios, dimension=None, size=None, strategies=None,
                          logy=True, clip_pct=(1, 99), xlim=None,
                          showfliers=False, width=9.0, height=5.2, title=None):
    """R1 — SD_strategy / SD_baseline, pooled over features and functions.

    The ratio is dimensionless (numerator and denominator share units), so this
    pooling needs no invented normaliser. Reads directly: 0.4 means "40% of the
    baseline's noise". The dashed line at 1.0 is the baseline itself.

    `clip_pct` sets the x-limits from percentiles of the pooled ratios rather
    than from the extremes. Without it a handful of near-degenerate cells (a
    feature that barely moved for one sampler) stretch the log axis over a dozen
    decades and every box collapses to a sliver. Outliers are counted in the
    caption instead of being drawn. Pass xlim=... to override, or
    clip_pct=None to show the full range.
    """
    d = ratios
    if dimension is not None:
        d = d[d["dimension"] == dimension]
    if size is not None:
        d = d[d["size"] == size]
    d = d[np.isfinite(d["ratio"]) & (d["ratio"] > 0)]
    base = ratios.attrs.get("baseline", "baseline")
    strats = strategies or sorted(d["strategy"].unique())
    color_of = _strategy_colors(strats)
    data = [d.loc[d["strategy"] == s, "ratio"].to_numpy() for s in strats]

    if xlim is None and clip_pct is not None:
        allv = d["ratio"].to_numpy()
        lo, hi = np.percentile(allv, clip_pct)
        xlim = (lo / 1.6, hi * 1.6)

    fig, ax = plt.subplots(figsize=(width, height))
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.65,
                    showfliers=showfliers,
                    medianprops=dict(color="black", lw=1.2),
                    flierprops=dict(marker="o", ms=2, alpha=0.25,
                                    markeredgecolor="none"))
    for p, s in zip(bp["boxes"], strats):
        p.set_facecolor(color_of[s]); p.set_alpha(0.7); p.set_edgecolor(color_of[s])
    ax.axvline(1.0, color="black", ls="--", lw=1.2)
    ax.set_yticks(range(1, len(strats) + 1)); ax.set_yticklabels(strats)
    if logy:
        ax.set_xscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(f"SD ratio vs {base}   (<1 = less noisy than {base})")
    ax.grid(True, axis="x", alpha=0.3)
    med = d.groupby("strategy")["ratio"].median()
    for i, s in enumerate(strats, start=1):
        if s in med:
            ax.text(0.99, i + 0.32, f"median {med[s]:.2f}x", fontsize=8,
                    transform=ax.get_yaxis_transform(), ha="right")
    tag = []
    if dimension is not None:
        tag.append(f"dim {dimension}")
    if size is not None:
        tag.append(f"n={size}xd")
    ax.set_title(title or f"Sampler effect on run-to-run noise"
                          + (f"  [{', '.join(tag)}]" if tag else ""))
    # be explicit about what the axis hides
    notes = []
    if xlim is not None:
        out_n = int(((d["ratio"] < xlim[0]) | (d["ratio"] > xlim[1])).sum())
        if out_n:
            notes.append(f"{out_n:,} of {len(d):,} cells outside axis")
    n_deg = ratios.attrs.get("n_degenerate", 0)
    if n_deg:
        notes.append(f"{n_deg:,} degenerate cells (sd~0) excluded")
    if notes:
        ax.text(0.01, 0.02, "; ".join(notes), transform=ax.transAxes,
                fontsize=7.5, color="grey", va="bottom")
    fig.tight_layout()
    return fig, ax


def boxplot_budget_ratio(bratio, dimension=None, strategies=None,
                         show_alpha_axis=True, width=9.5, height=5.4, title=None):
    """R2 — SD(low budget) / SD(high budget), with the implied alpha marked.

    Convergence without a regression: quadrupling n multiplies noise by
    4^-alpha, so the ratio alone gives the exponent. Reference lines at
    2.0 (alpha=0.5, Monte Carlo) and 4.0 (alpha=1.0, QMC ideal).
    """
    d = bratio if dimension is None else bratio[bratio["dimension"] == dimension]
    d = d[np.isfinite(d["ratio"]) & (d["ratio"] > 0)]
    lo, hi = int(d["size_lo"].iloc[0]), int(d["size_hi"].iloc[0])
    fold = hi / lo
    strats = strategies or sorted(d["strategy"].unique())
    color_of = _strategy_colors(strats)
    data = [d.loc[d["strategy"] == s, "ratio"].to_numpy() for s in strats]

    fig, ax = plt.subplots(figsize=(width, height))
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.65,
                    medianprops=dict(color="black", lw=1.2),
                    flierprops=dict(marker="o", ms=2, alpha=0.25,
                                    markeredgecolor="none"))
    for p, s in zip(bp["boxes"], strats):
        p.set_facecolor(color_of[s]); p.set_alpha(0.7); p.set_edgecolor(color_of[s])
    for r, lab, ls in [(1.0, "no gain", ":"),
                       (fold ** 0.5, "alpha=0.5  (Monte Carlo)", "--"),
                       (fold ** 1.0, "alpha=1.0  (QMC ideal)", "-.")]:
        ax.axvline(r, color="grey", ls=ls, lw=1.2)
        ax.text(r, len(strats) + 0.62, f" {lab}", fontsize=8, color="grey",
                rotation=0, va="center")
    ax.set_xscale("log")
    ax.set_yticks(range(1, len(strats) + 1)); ax.set_yticklabels(strats)
    ax.set_xlabel(f"SD(n={lo}xd) / SD(n={hi}xd)   "
                  f"(budget x{fold:g}; higher = noise falls faster)")
    ax.grid(True, axis="x", alpha=0.3)
    med = d.groupby("implied_alpha").size()  # placeholder to keep pandas import used
    ma = d.groupby("strategy")["implied_alpha"].median()
    mr = d.groupby("strategy")["ratio"].median()
    for i, s in enumerate(strats, start=1):
        if s in ma:
            ax.text(0.99, i + 0.32, f"{mr[s]:.2f}x  ->  alpha={ma[s]:.2f}",
                    fontsize=8, transform=ax.get_yaxis_transform(), ha="right")
    dtag = f"  [dim {dimension}]" if dimension is not None else ""
    ax.set_title(title or f"Budget effect on run-to-run noise{dtag}")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig, ax


def heatmap_ratio_by_function(ratios, dimension, size, strategies=None,
                              vmin=0.0, vmax=2.0, cmap_name="RdYlGn_r",
                              annotate=True, width=13.0, row_height=0.6,
                              title=None):
    """R3 — strategies x 24 functions, cell = median SD ratio vs the baseline.

    Shows whether a sampler's advantage is universal or function-specific. Green
    = less noisy than baseline; the baseline's own row is 1.0 by construction.
    """
    d = ratios[(ratios["dimension"] == dimension) & (ratios["size"] == size)]
    if d.empty:
        raise ValueError(f"No rows for dim={dimension}, size={size}.")
    base = ratios.attrs.get("baseline", "baseline")
    funcs = sorted(d["func"].unique())
    strats = strategies or sorted(d["strategy"].unique())
    M = (d.pivot_table(index="strategy", columns="func", values="ratio",
                       aggfunc="median")
         .reindex(index=strats, columns=funcs))

    fig, ax = plt.subplots(figsize=(width, max(2.4, row_height * len(strats) + 1.8)))
    cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad("lightgrey")
    A = M.to_numpy()
    im = ax.imshow(A, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(funcs)))
    ax.set_xticklabels([f"f{f}" for f in funcs], fontsize=7, rotation=90)
    ax.set_yticks(range(len(strats))); ax.set_yticklabels(strats, fontsize=8)
    _func_separators(ax, funcs, "x")
    if annotate:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if A[i, j] > vmax * 0.7 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label(f"median SD ratio vs {base}  (green = less noisy)")
    ax.set_xlabel("BBOB function")
    ax.set_title(title or f"Sampler advantage per function  "
                          f"[n={size}xd, dim {dimension}]")
    fig.tight_layout()
    return fig, ax


# =========================================================================== #
# TIER 3 — CV (dimensionless by construction; safe tier only)                 #
# =========================================================================== #

def plot_cv_vs_budget(per_func, dimension, funcs, safe_features, value="cv",
                      ncols=4, width_per=3.2, height_per=2.8, logy=True,
                      title=None):
    """C2 — median CV across the SAFE-TIER features, lines = strategies, one
    panel per function.

    CV is dimensionless by construction, so aggregating across the safe features
    is principled rather than relying on an invented normaliser. Restricted to
    `safe_features` because CV is invalid for sign-changing / interval-scaled
    features.
    """
    d = per_func[(per_func["dimension"] == dimension) &
                 (per_func["func"].isin(funcs)) &
                 (per_func["feature"].isin(set(safe_features)))]
    d = d[np.isfinite(d[value]) & (d[value] > 0)]
    if d.empty:
        raise ValueError("No safe-tier CV rows for that selection.")
    strats = sorted(d["strategy"].unique())
    color_of = _strategy_colors(strats)

    n = len(funcs); ncols = min(ncols, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_per * ncols, height_per * nrows),
                             squeeze=False, sharex=True, sharey=True)
    axf = axes.flatten()
    for ax, f in zip(axf, funcs):
        sub = d[d["func"] == f]
        for s in strats:
            g = (sub[sub["strategy"] == s].groupby("size")[value]
                 .median().reset_index().sort_values("size"))
            if len(g):
                ax.plot(g["size"], g[value], marker="o", ms=4,
                        lw=1.5, color=color_of[s], label=s)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"f{f}", fontsize=9)
        ax.grid(True, which="both", alpha=0.25)
    for ax in axf[n:]:
        ax.set_visible(False)
    for r in range(nrows):
        axes[r][0].set_ylabel("median CV (safe tier)", fontsize=8)
    for c in range(ncols):
        axes[nrows - 1][c].set_xlabel("budget multiplier (n = size x d)", fontsize=8)

    handles = [Line2D([0], [0], color=color_of[s], marker="o", label=s)
               for s in strats]
    # count what SURVIVED, not what was requested: the per-dimension exclusions
    # (repro_analysis.OMIT_FEATURES) remove safe-tier features at some
    # dimensions, so len(safe_features) would overstate the aggregation.
    n_used = d["feature"].nunique()
    n_asked = len(set(safe_features))
    tag = (f"{n_used} of {n_asked}" if n_used != n_asked else f"{n_used}")
    fig.suptitle(title or f"CV vs budget, median over {tag} "
                          f"safe-tier features  [dim {dimension}]", fontsize=12)
    _legend_right(fig, handles, "sampling strategy", rect=(0, 0, 0.84, 0.94), x=0.85)
    return fig, axes


# =========================================================================== #
# TIER 4 — value_norm robustness                                              #
# =========================================================================== #

def scatter_normaliser_robustness(per_func, dimension=None, hows=("iqr", "range"),
                                  width=6.6, height=6.2, title=None):
    """V2 — value_norm under two different denominators, plotted against each
    other.

    value_norm's weak point is that the denominator is a convention (why IQR
    across functions rather than range?). This figure pre-empts the objection:
    if the points lie on a tight monotone line, the CHOICE does not change any
    ordering, and conclusions drawn from value_norm are robust to it. Spearman
    rho is annotated.
    """
    from repro_analysis import feature_scale
    d = per_func if dimension is None else per_func[per_func["dimension"] == dimension]
    cols = {}
    for how in hows:
        sc = feature_scale(d, how=how).rename(columns={"scale": f"scale_{how}"})
        d = d.merge(sc, on=["dimension", "feature"], how="left")
        cols[how] = f"norm_{how}"
        d[cols[how]] = np.where(np.abs(d[f"scale_{how}"]) > 1e-12,
                                d["value"] / d[f"scale_{how}"], np.nan)
    x, y = cols[hows[0]], cols[hows[1]]
    dd = d[np.isfinite(d[x]) & np.isfinite(d[y]) & (d[x] > 0) & (d[y] > 0)]

    fig, ax = plt.subplots(figsize=(width, height))
    ax.scatter(dd[x], dd[y], s=5, alpha=0.15, edgecolor="none", color="#4c72b0")
    lo = float(min(dd[x].min(), dd[y].min()))
    hi = float(max(dd[x].max(), dd[y].max()))
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls="--", label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"value_norm  (denominator = {hows[0]} across functions)")
    ax.set_ylabel(f"value_norm  (denominator = {hows[1]} across functions)")
    rho = dd[[x, y]].corr(method="spearman").iloc[0, 1]
    ax.text(0.03, 0.97, f"Spearman rho = {rho:.4f}\n(rank-preserving if ~1)",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    dtag = f"  [dim {dimension}]" if dimension is not None else ""
    ax.set_title(title or f"Normaliser robustness{dtag}")
    fig.tight_layout()
    return fig, ax


# =========================================================================== #
# TIER 5 — alpha vs dimension                                                 #
# =========================================================================== #

def plot_alpha_vs_dimension(alpha, strategies=None, r2_min=0.0, use="alpha",
                            ylim=(-0.1, 1.3), width=8.5, height=5.4, title=None):
    """A4 — convergence exponent against problem dimension, one line per
    strategy (median with IQR band).

    Theory predicts a decline: QMC star discrepancy carries (log N)^d factors, so
    the effective finite-n exponent falls as d grows. If alpha drops from dim 2
    to dim 10 that is theory CONFIRMED, not a defect -- and it quantifies how
    fast the low-discrepancy advantage erodes with dimension.

    `use="implied_alpha"` accepts the budget-ratio table instead of the
    regression fits.
    """
    d = alpha[np.isfinite(alpha[use])]
    if "r2" in d.columns and r2_min > 0:
        d = d[d["r2"] >= r2_min]
    strats = strategies or sorted(d["strategy"].unique())
    color_of = _strategy_colors(strats)

    fig, ax = plt.subplots(figsize=(width, height))
    for s in strats:
        g = (d[d["strategy"] == s].groupby("dimension")[use]
             .agg(med="median",
                  lo=lambda x: np.percentile(x, 25),
                  hi=lambda x: np.percentile(x, 75))
             .reset_index().sort_values("dimension"))
        if g.empty:
            continue
        ax.plot(g["dimension"], g["med"], marker="o", ms=6, lw=1.8,
                color=color_of[s], label=s)
        ax.fill_between(g["dimension"], g["lo"], g["hi"], color=color_of[s],
                        alpha=0.13, lw=0)
    ax.axhline(0.5, color="grey", ls="--", lw=1.2)
    ax.axhline(1.0, color="grey", ls=":", lw=1.2)
    ax.text(ax.get_xlim()[1], 0.5, " Monte Carlo", fontsize=8, color="grey",
            va="bottom", ha="right")
    ax.text(ax.get_xlim()[1], 1.0, " QMC ideal", fontsize=8, color="grey",
            va="bottom", ha="right")
    ax.set_xticks(sorted(d["dimension"].unique()))
    ax.set_xlabel("problem dimension")
    ax.set_ylabel(f"{use}  (sigma ~ n^-alpha)")
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    handles = [Line2D([0], [0], color=color_of[s], marker="o", label=s)
               for s in strats]
    ax.legend(handles=handles, title="strategy", frameon=False,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(title or "Convergence exponent vs dimension "
                          "(median, IQR band)")
    fig.tight_layout()
    return fig, ax


def plot_ratio_vs_budget(ratios, dimensions=None, strategies=None, agg="median",
                         show_band=True, logy=True, ylim=None, features=None,
                         ncols=None, width_per=5.0, height_per=4.0, title=None):
    """R4 — overall SD ratio vs budget: one line per strategy, one panel per
    dimension. The pooled headline version of R1.

    Each point is the `agg` of SD_strategy / SD_baseline over ALL (function,
    feature) cells at that (dimension, budget). Pooling is legitimate because
    the ratio is dimensionless -- both terms are SDs of the same quantity in the
    same units, matched cell by cell before aggregation.

    agg="median"  robust; the default.
    agg="geomean" geometric mean -- the correct "average" for ratios: an
                  arithmetic mean would make 0.5x and 2.0x average to 1.25x
                  rather than 1.0x.

    The band shows the interquartile range across cells, i.e. how consistent the
    advantage is, not uncertainty in the median. A dashed line marks the
    baseline at 1.0; a log y-axis keeps 0.5x and 2x visually symmetric about it.

    `features` optionally restricts to a subset (e.g. one CV tier).
    """
    d = ratios
    if dimensions is not None:
        d = d[d["dimension"].isin(dimensions)]
    if features is not None:
        d = d[d["feature"].isin(set(features))]
    d = d[np.isfinite(d["ratio"]) & (d["ratio"] > 0)]
    if d.empty:
        raise ValueError("No ratio rows for that selection.")
    base = ratios.attrs.get("baseline", "baseline")
    dims = sorted(d["dimension"].unique())
    strats = strategies or sorted(d["strategy"].unique())
    color_of = _strategy_colors(strats)

    def _agg(s):
        return (float(np.exp(np.mean(np.log(s)))) if agg == "geomean"
                else float(np.median(s)))

    n = len(dims); ncols = ncols or n; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_per * ncols, height_per * nrows),
                             squeeze=False, sharey=True)
    axf = axes.flatten()
    for ax, dim in zip(axf, dims):
        sub = d[d["dimension"] == dim]
        for s in strats:
            g = (sub[sub["strategy"] == s].groupby("size")["ratio"]
                 .agg(mid=_agg,
                      lo=lambda x: float(np.percentile(x, 25)),
                      hi=lambda x: float(np.percentile(x, 75)))
                 .reset_index().sort_values("size"))
            if g.empty:
                continue
            ax.plot(g["size"], g["mid"], marker="o", ms=5, lw=1.8,
                    color=color_of[s], label=s)
            if show_band:
                ax.fill_between(g["size"], g["lo"], g["hi"], color=color_of[s],
                                alpha=0.12, lw=0)
        ax.axhline(1.0, color="black", ls="--", lw=1.2)
        ax.set_title(f"dimension {dim}")
        ax.set_xlabel("budget multiplier (n = size x d)")
        ax.set_ylabel(f"SD ratio vs {base}  ({agg})")
        if logy:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, which="both", alpha=0.3)
    for ax in axf[n:]:
        ax.set_visible(False)

    handles = [Line2D([0], [0], color=color_of[s], marker="o", label=s)
               for s in strats]
    if show_band:
        handles.append(Patch(facecolor="grey", alpha=0.25,
                             label="IQR across (function, feature)"))
    nf = d["feature"].nunique()
    fig.suptitle(title or f"Sampler effect on run-to-run noise vs budget "
                          f"— pooled over {nf} features x "
                          f"{d['func'].nunique()} functions", fontsize=12)
    _legend_right(fig, handles, "sampling strategy", rect=(0, 0, 0.85, 0.93),
                  x=0.86)
    return fig, axes