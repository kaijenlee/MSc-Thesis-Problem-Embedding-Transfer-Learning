"""
icc_analysis.py

ICC(1,1) study, built on the variance components from compute_ela_icc.py.

    from compute_ela_icc import load_icc, components_to_dataframe
    import icc_analysis as ia

    df  = pd.concat([components_to_dataframe(load_icc(p)) for p in SOURCES.values()])
    tab = ia.prepare(df)

WHAT ICC(1,1) IS HERE
---------------------
    ICC = sigma_b^2 / (sigma_b^2 + sigma_w^2)

with a function's 100 BBOB instances as subjects and 30 runs as replicates. It
is the reliability of a SINGLE measurement: what share of the spread you see
across instances is real rather than sampling noise.

POOLING RULE -- THE INVERSE OF THE REPRODUCIBILITY STUDY
--------------------------------------------------------
ICC is dimensionless and bounded in [0, 1], so pooling across FEATURES is fine
(no units to reconcile). Pooling across FUNCTIONS is not: the attainable ICC
differs enormously by function because sigma_b is set by how much that
function's instances actually differ -- f1-f19 reach ~0.99 while f20/f23/f24
sit near 0. So: aggregate over features freely, keep functions on their own
axis. (In the sd study it was the opposite: units blocked both, and ratios
fixed both.)

WHY THE COMPONENTS MATTER
-------------------------
A low ICC has two causes the ratio alone cannot separate:
    sigma_w large   -> noisy measurement          (budget-limited)
    sigma_b ~ 0     -> instances do not differ    (structural; no budget helps)
Reporting sigma_b and sigma_w directly settles it. Negative sigma_b estimates
are KEPT (compute_ela_icc no longer clips), so `no_signal` = sigma2_between <= 0
is an honest count rather than a censored one.

FIGURES
  I1 heatmap_icc                features x functions, cell = ICC
  I2 plot_icc_vs_budget         panels = functions, lines = strategies  <- workhorse
  I3 barplot_win_counts         strategy ranking, ordinal across functions
  I4 scatter_signal_vs_noise    sigma_b vs sigma_w with ICC iso-lines    <- key
  I5 heatmap_sigma_b            "is there anything to detect at all"
  I6 plot_no_signal_rate        share of cells with sigma2_between <= 0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from repro_analysis import (GROUP_ORDER, GROUP_LABEL, FUNC_TO_GROUP,
                            GROUP_BOUNDARIES, OMIT_FEATURES, add_group)


def _strategy_colors(strats, cmap_name="tab10"):
    cmap = plt.get_cmap(cmap_name)
    return {s: cmap(i % 10) for i, s in enumerate(sorted(strats))}


def _func_separators(ax, funcs, axis="x"):
    pos = {f: i for i, f in enumerate(funcs)}
    for b in GROUP_BOUNDARIES:
        if b in pos and any(f > b for f in funcs):
            (ax.axvline if axis == "x" else ax.axhline)(
                pos[b] + 0.5, color="black", lw=0.8, alpha=0.5)


def _legend_right(fig, handles, title, rect=(0, 0, 0.84, 0.94), x=0.85):
    fig.tight_layout(rect=list(rect))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(x, 0.5),
               title=title, frameon=False)


# --------------------------------------------------------------------------- #
# Preparation                                                                  #
# --------------------------------------------------------------------------- #

def prepare(df, omit=None, verbose=True):
    """Tidy components table -> analysis-ready frame.

    Drops the per-dimension excluded features (shared with the reproducibility
    pipeline), attaches BBOB group labels, and adds:
      no_signal   : sigma2_between <= 0, i.e. nothing to detect on this cell
      icc_pos     : ICC with negatives floored at 0 -- for DISPLAY only
    """
    omit = OMIT_FEATURES if omit is None else omit
    d = df.copy()
    if omit:
        mask = pd.Series(False, index=d.index)
        for dim, feats in omit.items():
            mask |= (d["dimension"] == dim) & d["feature"].isin(feats)
        if verbose and mask.any():
            print(f"dropped {int(mask.sum()):,} rows for excluded features: "
                  f"{', '.join(sorted(d.loc[mask, 'feature'].unique()))}")
        d = d[~mask].copy()

    d = add_group(d)
    src = "icc_raw" if "icc_raw" in d.columns else "icc"
    d["no_signal"] = d["sigma2_between"] <= 0
    d["icc_pos"] = d[src].clip(lower=0)

    if verbose:
        print(f"{len(d):,} cells | {d['feature'].nunique()} features x "
              f"{d['func'].nunique()} functions x "
              f"{d['strategy'].nunique()} strategies | "
              f"dims {sorted(d['dimension'].unique())}")
        print(f"no-signal cells (sigma2_between <= 0): {d['no_signal'].mean():.1%}")
        if src == "icc_raw":
            print(f"  of which raw ICC < 0: {(d[src] < 0).mean():.1%} "
                  f"(kept, not clipped)")
    return d


def rank_within_function(df, value="icc_pos"):
    """Rank strategies WITHIN each (dimension, size, func, feature).

    Higher ICC = better = rank 1. Ordinal, so it never compares ICCs across
    functions with different attainable ceilings.
    """
    d = df[np.isfinite(df[value])].copy()
    keys = ["dimension", "size", "func", "feature"]
    d["rank"] = d.groupby(keys)[value].rank(ascending=False, method="min")
    d["is_win"] = d["rank"] == 1
    return d


def win_counts(ranked):
    """Two-stage ordinal aggregation: collapse features, then count functions
    won per (dimension, size, strategy)."""
    per_func = (ranked.groupby(["dimension", "size", "func", "strategy"])
                .agg(mean_rank=("rank", "mean"), n_feat=("rank", "size"))
                .reset_index())
    per_func["func_rank"] = (per_func
                             .groupby(["dimension", "size", "func"])["mean_rank"]
                             .rank(ascending=True, method="min").astype(int))
    g = (per_func.assign(won=lambda x: x["func_rank"] == 1)
         .groupby(["dimension", "size", "strategy"])
         .agg(wins=("won", "sum"), n_func=("func", "nunique"),
              mean_rank=("mean_rank", "mean"))
         .reset_index())
    g["overall_rank"] = (g.groupby(["dimension", "size"])["mean_rank"]
                         .rank(ascending=True, method="min").astype(int))
    return per_func, g.sort_values(["dimension", "size", "overall_rank"])


# =========================================================================== #
# I1 — ICC heatmap                                                            #
# =========================================================================== #

def heatmap_icc(df, dimension, strategy, size, value="icc_pos", sort_features=True,
                cmap_name="RdYlGn", vmin=0.0, vmax=1.0, annotate=False,
                width=13.0, row_height=0.24, title=None):
    """I1 — rows = features, cols = 24 functions, cell = ICC. Green = reliable.

    Fixed colour scale [0,1] so panels are comparable. Group separators mark the
    BBOB boundaries, so any group alignment can be SEEN rather than assumed --
    the split is not always along group lines (f21/f22 behave unlike f20/23/24).
    """
    d = df[(df["dimension"] == dimension) & (df["strategy"] == strategy) &
           (df["size"] == size)]
    if d.empty:
        raise ValueError(f"No rows for dim={dimension}, {strategy}, size={size}.")
    funcs = sorted(d["func"].unique())
    order = (d.groupby("feature")[value].median().sort_values(ascending=False)
             .index.tolist() if sort_features else sorted(d["feature"].unique()))
    M = (d.pivot_table(index="feature", columns="func", values=value)
         .reindex(index=order, columns=funcs))

    fig, ax = plt.subplots(figsize=(width, max(3.0, row_height * len(order) + 2.0)))
    cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad("lightgrey")
    A = M.to_numpy()
    im = ax.imshow(A, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(funcs)))
    ax.set_xticklabels([f"f{f}" for f in funcs], fontsize=7, rotation=90)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6.5)
    _func_separators(ax, funcs, "x")
    if annotate:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                            fontsize=5,
                            color="white" if A[i, j] < 0.35 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("ICC(1,1)  (green = a single run is reliable)")
    ax.set_xlabel("BBOB function")
    ax.set_title(title or f"ICC(1,1) per feature and function  "
                          f"[{strategy}, n={size}xd, dim {dimension}]")
    fig.tight_layout()
    return fig, ax


# =========================================================================== #
# I2 — ICC vs budget, per function (the workhorse)                            #
# =========================================================================== #

def plot_icc_vs_budget(df, dimension, funcs=None, value="icc_pos", features=None,
                       ncols=6, width_per=2.6, height_per=2.3, ylim=(0, 1),
                       title=None):
    """I2 — median ICC across features vs budget, one panel per function.

    The direct analogue of the reproducibility CV-vs-budget figure. Aggregating
    over features is legitimate (ICC is dimensionless and bounded); functions
    stay on their own panels because their attainable ICC differs by orders of
    magnitude.

    A flat line near zero means sigma_b ~ 0 -- no signal, and no budget will
    help. A rising line means the measurement was noise-limited and the budget
    is buying reliability.
    """
    d = df[df["dimension"] == dimension]
    if features is not None:
        d = d[d["feature"].isin(set(features))]
    d = d[np.isfinite(d[value])]
    funcs = funcs or sorted(d["func"].unique())
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
                ax.plot(g["size"], g[value], marker="o", ms=3.5, lw=1.4,
                        color=color_of[s], label=s)
        ax.set_title(f"f{f}", fontsize=9)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
    for ax in axf[n:]:
        ax.set_visible(False)
    for r in range(nrows):
        axes[r][0].set_ylabel("median ICC", fontsize=8)
    for c in range(ncols):
        axes[nrows - 1][c].set_xlabel("budget multiplier", fontsize=8)

    handles = [Line2D([0], [0], color=color_of[s], marker="o", label=s)
               for s in strats]
    nf = d["feature"].nunique()
    fig.suptitle(title or f"ICC(1,1) vs budget — median over {nf} features "
                          f"[dim {dimension}]", fontsize=12)
    _legend_right(fig, handles, "sampling strategy", rect=(0, 0, 0.87, 0.95), x=0.88)
    return fig, axes


# =========================================================================== #
# I3 — strategy win counts                                                    #
# =========================================================================== #

def barplot_win_counts(wins, width_per=4.4, height_per=3.6, title=None):
    """I3 — functions won per strategy. Ordinal: strategies are ranked within
    each function, so no ICC is ever compared across functions."""
    dims = sorted(wins["dimension"].unique())
    szs = sorted(wins["size"].unique())
    strats = sorted(wins["strategy"].unique())
    color_of = _strategy_colors(strats)

    fig, axes = plt.subplots(len(dims), len(szs),
                             figsize=(width_per * len(szs), height_per * len(dims)),
                             squeeze=False, sharey=True)
    for r, dim in enumerate(dims):
        for c, sz in enumerate(szs):
            ax = axes[r][c]
            sub = (wins[(wins["dimension"] == dim) & (wins["size"] == sz)]
                   .set_index("strategy").reindex(strats))
            ax.bar(range(len(strats)), sub["wins"].fillna(0),
                   color=[color_of[s] for s in strats], edgecolor="white")
            ax.set_xticks(range(len(strats)))
            ax.set_xticklabels(strats, rotation=35, ha="right", fontsize=8)
            if r == 0:
                ax.set_title(f"budget x{sz}")
            if c == 0:
                ax.set_ylabel(f"dim {dim}\nfunctions won")
            ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title or "Most reliable sampler — functions won (ICC, ranked "
                          "within each function)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes


# =========================================================================== #
# I4 — signal vs noise (the key figure)                                       #
# =========================================================================== #

def scatter_signal_vs_noise(df, dimension, size, strategy=None, funcs=None,
                            color_by="fgroup", iso_icc=(0.25, 0.5, 0.75, 0.9),
                            normalise=True, trend=True, hexbin=False,
                            sample=30000, width=8.0, height=7.4, title=None):
    """I4 — sigma_b (signal) against sigma_w (noise), log-log, one point per
    (function, feature).

    THE figure the variance components unlock. Because
        ICC = sd_b^2 / (sd_b^2 + sd_w^2),
    a constant ICC means a constant RATIO sd_b/sd_w, which on log-log axes is a
    straight line of slope 1. Those iso-ICC lines are drawn, so a point's
    position reads directly:

        bottom-right : low ICC because there is NO SIGNAL (sd_b tiny)
        top-right    : signal present but swamped by NOISE (budget-limited)
        top-left     : reliable -- strong signal, quiet measurement

    That distinction is invisible in the ICC value alone. It also shows whether
    a sampler wins by cutting sd_w (shift LEFT) or by inflating sd_b (shift UP).

    Cells with sigma2_between <= 0 cannot be plotted on a log axis; they are
    counted in the caption and are exactly the "no signal" set (see I6).

    SCALE CONFOUND -- why normalise=True is the default. sd_b and sd_w are both
    in FEATURE UNITS, so a large-magnitude feature has large values of both. A
    raw scatter therefore shows a strong slope-1 trend that is mostly the
    feature's scale rather than anything about signal and noise. Dividing both
    by |grand_mean| removes it, and leaves ICC untouched because ICC is a ratio
    -- the constant cancels, so the iso-ICC lines remain exactly valid. The
    normalised axes then read as familiar quantities: x is the CV (relative run
    noise) and y is the relative between-instance signal.

    `trend` fits log(sd_b) = a + b log(sd_w) by OLS and annotates the slope with
    Spearman rho. Slope ~ 1 means signal and noise scale together, so ICC is
    roughly independent of magnitude; slope < 1 means noisier cells have
    proportionally less signal, i.e. ICC falls as noise grows.

    `hexbin` replaces the points with a density map when overplotting hides the
    structure.
    """
    d = df[(df["dimension"] == dimension) & (df["size"] == size)]
    if strategy is not None:
        d = d[d["strategy"] == strategy]
    if funcs is not None:
        d = d[d["func"].isin(funcs)]
    n_all = len(d)
    d = d[(d["sigma2_between"] > 0) & (d["sigma2_within"] > 0)].copy()
    n_nosig = n_all - len(d)
    if d.empty:
        raise ValueError("No cells with positive signal to plot.")
    d["sd_b"] = np.sqrt(d["sigma2_between"])
    d["sd_w"] = np.sqrt(d["sigma2_within"])
    if normalise:
        sc = np.abs(d["grand_mean"])
        ok = sc > 1e-12
        d = d[ok].copy(); sc = sc[ok]
        d["sd_b"] = d["sd_b"] / sc
        d["sd_w"] = d["sd_w"] / sc
    if len(d) > sample:
        d = d.sample(sample, random_state=0)

    fig, ax = plt.subplots(figsize=(width, height))
    if hexbin:
        hb = ax.hexbin(d["sd_w"], d["sd_b"], xscale="log", yscale="log",
                       gridsize=45, cmap="viridis", mincnt=1, linewidths=0)
        fig.colorbar(hb, ax=ax, fraction=0.03, pad=0.02).set_label("cells")
        handles, leg_title = [], None
    elif color_by == "fgroup":
        cmap = plt.get_cmap("tab10")
        col = {g: cmap(i % 10) for i, g in enumerate(GROUP_ORDER)}
        for g in GROUP_ORDER:
            s = d[d["fgroup"] == g]
            if len(s):
                ax.scatter(s["sd_w"], s["sd_b"], s=7, alpha=0.35,
                           edgecolor="none", color=col[g], label=GROUP_LABEL[g])
        handles = [Patch(facecolor=col[g], label=GROUP_LABEL[g])
                   for g in GROUP_ORDER if (d["fgroup"] == g).any()]
        leg_title = "BBOB function group"
    else:
        vals = sorted(d[color_by].unique())
        col = _strategy_colors(vals)
        for v in vals:
            sv = d[d[color_by] == v]
            ax.scatter(sv["sd_w"], sv["sd_b"], s=7, alpha=0.35, edgecolor="none",
                       color=col[v], label=str(v))
        handles = [Patch(facecolor=col[v], label=str(v)) for v in vals]
        leg_title = color_by

    lo = float(min(d["sd_w"].min(), d["sd_b"].min()))
    hi = float(max(d["sd_w"].max(), d["sd_b"].max()))
    xs = np.array([lo, hi])
    for icc in iso_icc:
        c = np.sqrt(icc / (1 - icc))          # sd_b / sd_w at that ICC
        ax.plot(xs, c * xs, color="grey", lw=0.9, ls="--", alpha=0.8, zorder=0)
        ax.annotate(f"ICC={icc:g}", (xs[1], c * xs[1]), fontsize=7.5,
                    color="grey", ha="right", va="bottom")
    # --- fitted trend in log-log space ---
    if trend and len(d) >= 10:
        lx = np.log10(d["sd_w"].to_numpy()); ly = np.log10(d["sd_b"].to_numpy())
        # closed-form OLS: polyfit builds a Vandermonde matrix and warns about
        # conditioning on log-scaled inputs, which this avoids entirely
        xm, ym = lx.mean(), ly.mean()
        sxx = np.sum((lx - xm) ** 2)
        if sxx <= 0:
            trend = False
        else:
            b = float(np.sum((lx - xm) * (ly - ym)) / sxx)
            a = float(ym - b * xm)
            rho = d[["sd_w", "sd_b"]].corr(method="spearman").iloc[0, 1]
    if trend and len(d) >= 10:
        gx = np.linspace(lx.min(), lx.max(), 50)
        ax.plot(10 ** gx, 10 ** (a + b * gx), color="black", lw=2.0, zorder=5)
        ax.plot(10 ** gx, 10 ** (a + b * gx), color="white", lw=3.4, zorder=4)
        ax.text(0.02, 0.02,
                f"trend: slope = {b:.2f}   Spearman rho = {rho:.2f}\n"
                f"(slope 1 -> signal and noise scale together, ICC "
                f"independent of magnitude)",
                transform=ax.transAxes, fontsize=8, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey",
                          alpha=0.85))

    ax.set_xscale("log"); ax.set_yscale("log")
    unit = " / |mean|" if normalise else ""
    ax.set_xlabel(f"sigma_w{unit}   run-to-run noise  (right = noisier)"
                  + ("   [= CV]" if normalise else ""))
    ax.set_ylabel(f"sigma_b{unit}   between-instance signal  (up = more to detect)")
    ax.grid(True, which="both", alpha=0.2)
    if handles:
        ax.legend(handles=handles, title=leg_title, frameon=False, fontsize=8,
                  loc="lower right")
    if n_nosig:
        ax.text(0.01, 0.99, f"{n_nosig:,} of {n_all:,} cells have "
                            f"sigma2_between <= 0\n(no signal — cannot be shown "
                            f"on a log axis)",
                transform=ax.transAxes, va="top", fontsize=8, color="grey")
    tag = f"{strategy}, " if strategy else ""
    ax.set_title(title or f"Signal vs noise  [{tag}n={size}xd, dim {dimension}]")
    fig.tight_layout()
    return fig, ax


# =========================================================================== #
# I5 — sigma_b heatmap                                                        #
# =========================================================================== #

def heatmap_sigma_b(df, dimension, strategy, size, relative=True,
                    cmap_name="viridis", width=13.0, row_height=0.24,
                    title=None):
    """I5 — features x functions, cell = between-instance signal.

    Answers "is there anything to detect at all", independently of how noisy the
    measurement is. Transformation-invariant features (adj_r2, disp.diff_*,
    pca.expl_var*) should be dark across the board -- they do not vary between
    instances by construction, so no sampler or budget can give them a high ICC.

    relative=True divides sd_b by the feature's median |grand_mean| so rows are
    comparable; otherwise sd_b is in raw feature units and only comparable
    within a row.
    """
    d = df[(df["dimension"] == dimension) & (df["strategy"] == strategy) &
           (df["size"] == size)].copy()
    if d.empty:
        raise ValueError(f"No rows for dim={dimension}, {strategy}, size={size}.")
    d["sd_b"] = np.sqrt(d["sigma2_between"].clip(lower=0))
    if relative:
        scale = (d.groupby("feature")["grand_mean"]
                 .transform(lambda s: np.abs(s).median()))
        d["val"] = np.where(scale > 1e-12, d["sd_b"] / scale, np.nan)
        lab = "sd_b / |mean|   (relative between-instance signal)"
    else:
        d["val"] = d["sd_b"]
        lab = "sd_b  (raw feature units — compare within a row only)"

    funcs = sorted(d["func"].unique())
    order = (d.groupby("feature")["val"].median().sort_values(ascending=False)
             .index.tolist())
    M = (d.pivot_table(index="feature", columns="func", values="val")
         .reindex(index=order, columns=funcs))
    A = M.to_numpy()
    pos = A[np.isfinite(A) & (A > 0)]
    vmin = float(np.percentile(pos, 2)) if pos.size else 1e-6
    vmax = float(np.percentile(pos, 98)) if pos.size else 1.0

    fig, ax = plt.subplots(figsize=(width, max(3.0, row_height * len(order) + 2.0)))
    cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad("lightgrey")
    from matplotlib.colors import LogNorm
    im = ax.imshow(np.where(A > 0, A, np.nan), aspect="auto", cmap=cmap,
                   norm=LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, 1e-11)),
                   interpolation="nearest")
    ax.set_xticks(range(len(funcs)))
    ax.set_xticklabels([f"f{f}" for f in funcs], fontsize=7, rotation=90)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6.5)
    _func_separators(ax, funcs, "x")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label(lab + "   (grey = no signal)")
    ax.set_xlabel("BBOB function")
    ax.set_title(title or f"Between-instance signal  "
                          f"[{strategy}, n={size}xd, dim {dimension}]")
    fig.tight_layout()
    return fig, ax


# =========================================================================== #
# I6 — no-signal rate                                                         #
# =========================================================================== #

def plot_no_signal_rate(df, dimension=None, by="feature", tiers=None,
                        width=12.0, height=5.4, top=None, title=None):
    """I6 — share of cells with sigma2_between <= 0, i.e. no detectable
    between-instance signal.

    The honest, UNCLIPPED version of the old "clipped floor" count. Because the
    estimator is roughly symmetric about zero when the true signal is zero,
    clipping used to hide about half of these.

    by="feature"  -> which features are transformation-invariant
    by="func"     -> which landscapes have exchangeable instances
    `tiers` optional {feature: tier} to colour by CV tier.
    """
    d = df if dimension is None else df[df["dimension"] == dimension]
    g = d.groupby(by)["no_signal"].mean().sort_values(ascending=False)
    if top:
        g = g.head(top)
    labels = [f"f{v}" if by == "func" else str(v) for v in g.index]

    tier_color = {"safe": "#2ca02c", "caveat": "#ff7f0e",
                  "excluded": "#d62728", "unclassified": "#7f7f7f"}
    cols = ([tier_color.get(tiers.get(v, "unclassified"), "#7f7f7f")
             for v in g.index] if (tiers and by == "feature") else "#4c72b0")

    fig, ax = plt.subplots(figsize=(width, height))
    ax.bar(range(len(g)), g.to_numpy(), color=cols, edgecolor="white")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("share of cells with no between-instance signal")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    if tiers and by == "feature":
        ax.legend(handles=[Patch(facecolor=c, label=t)
                           for t, c in tier_color.items()],
                  title="CV tier", frameon=False, loc="upper right")
    dtag = f"  [dim {dimension}]" if dimension is not None else ""
    ax.set_title(title or f"No detectable between-instance signal "
                          f"(sigma_b^2 <= 0), by {by}{dtag}")
    fig.tight_layout()
    return fig, ax


# =========================================================================== #
# I7 — paired shift: does a sampler cut NOISE or gain SIGNAL?                 #
# =========================================================================== #

def plot_signal_noise_shift(df, dimension, size, target, baseline="uniform",
                            level="feature", normalise=True, funcs=None,
                            iso_icc=(0.25, 0.5, 0.75, 0.9), min_pairs=3,
                            width=8.0, height=7.4, title=None):
    """I7 — arrows from `baseline` to `target` in the (sigma_w, sigma_b) plane.

    The scatter (I4) shows where cells sit; this shows where a sampler MOVES
    them, which is the causal question. Because
        ICC = sd_b^2 / (sd_b^2 + sd_w^2),
    a sampler can raise ICC two ways, and they mean different things:

        arrow points LEFT  -> sigma_w fell: genuinely more precise measurement
        arrow points UP    -> sigma_b rose: the sampler changed what is being
                              measured, so instances now look more different.
                              That is not extra precision, and for an adaptive
                              sampler it may just mean the sampled region is
                              itself instance-dependent.

    Distinguishing those is exactly what the ICC value alone cannot do.

    level="feature" draws one arrow per feature (median across functions) so the
    figure stays readable; level="cell" draws every (function, feature) pair.
    Arrow colour marks whether ICC improved (green) or worsened (red).
    """
    d = df[(df["dimension"] == dimension) & (df["size"] == size) &
           (df["strategy"].isin([baseline, target]))].copy()
    if funcs is not None:
        d = d[d["func"].isin(funcs)]
    d = d[(d["sigma2_between"] > 0) & (d["sigma2_within"] > 0)].copy()
    if d.empty:
        raise ValueError("No cells with positive signal for that selection.")
    d["sd_b"] = np.sqrt(d["sigma2_between"])
    d["sd_w"] = np.sqrt(d["sigma2_within"])
    if normalise:
        sc = np.abs(d["grand_mean"])
        d = d[sc > 1e-12].copy()
        d["sd_b"] /= np.abs(d["grand_mean"]); d["sd_w"] /= np.abs(d["grand_mean"])

    keys = ["feature"] if level == "feature" else ["feature", "func"]
    piv = (d.groupby(keys + ["strategy"])[["sd_w", "sd_b"]]
           .median().reset_index())
    a = piv[piv["strategy"] == baseline].set_index(keys)[["sd_w", "sd_b"]]
    b = piv[piv["strategy"] == target].set_index(keys)[["sd_w", "sd_b"]]
    P = a.join(b, lsuffix="_0", rsuffix="_1").dropna()
    if len(P) < min_pairs:
        raise ValueError(f"Only {len(P)} paired cells — nothing to show.")

    icc0 = P["sd_b_0"] ** 2 / (P["sd_b_0"] ** 2 + P["sd_w_0"] ** 2)
    icc1 = P["sd_b_1"] ** 2 / (P["sd_b_1"] ** 2 + P["sd_w_1"] ** 2)
    better = (icc1 > icc0).to_numpy()

    fig, ax = plt.subplots(figsize=(width, height))
    lo = float(min(P[["sd_w_0", "sd_w_1", "sd_b_0", "sd_b_1"]].min()))
    hi = float(max(P[["sd_w_0", "sd_w_1", "sd_b_0", "sd_b_1"]].max()))
    xs = np.array([lo, hi])
    for icc in iso_icc:
        c = np.sqrt(icc / (1 - icc))
        ax.plot(xs, c * xs, color="grey", lw=0.9, ls="--", alpha=0.7, zorder=0)
        ax.annotate(f"ICC={icc:g}", (xs[1], c * xs[1]), fontsize=7.5,
                    color="grey", ha="right", va="bottom")

    for (x0, y0, x1, y1), ok in zip(
            P[["sd_w_0", "sd_b_0", "sd_w_1", "sd_b_1"]].to_numpy(), better):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.1,
                                    color="#2ca02c" if ok else "#d62728",
                                    alpha=0.75, shrinkA=0, shrinkB=0))
    ax.scatter(P["sd_w_0"], P["sd_b_0"], s=16, facecolor="white",
               edgecolor="black", lw=0.7, zorder=6, label=f"{baseline} (start)")

    ax.set_xscale("log"); ax.set_yscale("log")
    unit = " / |mean|" if normalise else ""
    ax.set_xlabel(f"sigma_w{unit}   run-to-run noise  (LEFT = sampler cut noise)")
    ax.set_ylabel(f"sigma_b{unit}   between-instance signal  (UP = signal grew)")
    ax.grid(True, which="both", alpha=0.2)

    # decompose the movement: how much of the ICC change came from each axis?
    dlw = np.log10(P["sd_w_1"] / P["sd_w_0"]).median()
    dlb = np.log10(P["sd_b_1"] / P["sd_b_0"]).median()
    ax.text(0.02, 0.02,
            f"median shift: noise x{10**dlw:.2f}, signal x{10**dlb:.2f}\n"
            f"{int(better.sum())}/{len(P)} cells improved in ICC",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85))
    handles = [Line2D([0], [0], color="#2ca02c", lw=1.5, label="ICC improved"),
               Line2D([0], [0], color="#d62728", lw=1.5, label="ICC worsened"),
               Line2D([0], [0], marker="o", lw=0, markerfacecolor="white",
                      markeredgecolor="black", label=f"{baseline} (start)")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    ax.set_title(title or f"{baseline} -> {target}: noise or signal?  "
                          f"[n={size}xd, dim {dimension}, per {level}]")
    fig.tight_layout()
    return fig, ax


# =========================================================================== #
# I8 — variance composition bars                                              #
# =========================================================================== #

def plot_variance_composition(df, dimension, strategy, size, by="func",
                              features=None, funcs=None, weight="equal",
                              show_total=True, sort_by=None, annotate=False,
                              width=13.0, height=6.6, title=None):
    """I8 — stacked bars: what fraction of the total variance is SIGNAL?

    Each bar partitions total variance into

        sigma_b^2   between-instance signal   (green, bottom)
        sigma_w^2   run-to-run noise          (grey, top)

    normalised to 100%. The green fraction IS the ICC -- this is the same number
    as I1/I2, drawn as a partition rather than a value.

    WHY VARIANCES, NOT SDs. sigma_b^2 + sigma_w^2 = total variance, so the parts
    genuinely add up. SDs do not (sigma_b + sigma_w is not the total SD), so a
    bar stacked from SDs would partition nothing.

    WHY THIS IS WORTH PLOTTING. High ICC is unambiguous -- it requires real
    signal AND a precise measurement. Low ICC has three causes the ratio cannot
    separate: no signal (sigma_b ~ 0), too much noise (sigma_w large), or both.
    Only the middle case is fixable with budget. The composition shows which,
    and `show_total` adds the magnitude the normalisation discards -- two bars
    can both read 30% signal while differing by orders of magnitude in absolute
    variance.

    AGGREGATION. With by="func" each bar pools ~45 features, which have
    different units. weight="equal" (default) first divides every feature's
    components by ITS OWN total variance, so each feature contributes equally
    and the bar is a true partition of normalised variance. weight="raw" sums
    the components as they are, which is a literal variance split but is
    dominated by whichever feature has the largest magnitude -- rarely what you
    want. Cells with sigma2_between <= 0 contribute 0 signal (not negative),
    and their share is reported as the "no signal" hatch.

    by="func"    one bar per BBOB function (group separators drawn)
    by="feature" one bar per feature
    """
    d = df[(df["dimension"] == dimension) & (df["strategy"] == strategy) &
           (df["size"] == size)].copy()
    if features is not None:
        d = d[d["feature"].isin(set(features))]
    if funcs is not None:
        d = d[d["func"].isin(funcs)]
    d = d[np.isfinite(d["sigma2_within"]) & (d["sigma2_within"] > 0)]
    if d.empty:
        raise ValueError(f"No cells for dim={dimension}, {strategy}, size={size}.")

    # negative signal estimates mean "no detectable signal", not negative variance
    d["s2b"] = d["sigma2_between"].clip(lower=0)
    d["s2w"] = d["sigma2_within"]
    d["no_sig"] = d["sigma2_between"] <= 0

    if weight == "equal":
        tot = d["s2b"] + d["s2w"]
        d["wb"] = np.where(tot > 0, d["s2b"] / tot, np.nan)
        d["ww"] = np.where(tot > 0, d["s2w"] / tot, np.nan)
    elif weight == "raw":
        d["wb"], d["ww"] = d["s2b"], d["s2w"]
    else:
        raise ValueError("weight must be 'equal' or 'raw'")

    g = (d.groupby(by)
         .agg(sig=("wb", "sum"), noise=("ww", "sum"),
              raw_total=("s2b", "sum"), raw_noise=("s2w", "sum"),
              no_sig=("no_sig", "mean"), n=("wb", "size"))
         .reset_index())
    tot = g["sig"] + g["noise"]
    g["frac_sig"] = g["sig"] / tot
    g["frac_noise"] = g["noise"] / tot
    g["total_var"] = g["raw_total"] + g["raw_noise"]

    if sort_by == "icc":
        g = g.sort_values("frac_sig", ascending=False)
    elif by == "func":
        g = g.sort_values("func")
    else:
        g = g.sort_values("frac_sig", ascending=False)
    labels = [f"f{v}" if by == "func" else str(v) for v in g[by]]
    x = np.arange(len(g))

    if show_total:
        fig, axes = plt.subplots(2, 1, figsize=(width, height), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1]))
        ax, ax2 = axes
    else:
        fig, ax = plt.subplots(figsize=(width, height * 0.75))
        ax2 = None

    ax.bar(x, g["frac_sig"], color="#2ca02c", alpha=0.85,
           edgecolor="white", lw=0.4, label="signal  $\\sigma_b^2$")
    ax.bar(x, g["frac_noise"], bottom=g["frac_sig"], color="#9e9e9e", alpha=0.85,
           edgecolor="white", lw=0.4, label="noise  $\\sigma_w^2$")
    ax.axhline(0.5, color="black", ls=":", lw=0.9, alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of total variance\n(green fraction = ICC)")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=9)
    if annotate:
        for xi, v in zip(x, g["frac_sig"]):
            ax.text(xi, min(v + 0.02, 0.96), f"{v:.2f}", ha="center",
                    va="bottom", fontsize=6, rotation=90)
    if by == "func":
        pos = {f: i for i, f in enumerate(g[by])}
        for b in GROUP_BOUNDARIES:
            if b in pos and any(f > b for f in g[by]):
                ax.axvline(pos[b] + 0.5, color="black", lw=0.8, alpha=0.5)
                if ax2 is not None:
                    ax2.axvline(pos[b] + 0.5, color="black", lw=0.8, alpha=0.5)

    if ax2 is not None:
        ax2.bar(x, g["total_var"], color="#4c72b0", alpha=0.75,
                edgecolor="white", lw=0.4)
        ax2.set_yscale("log")
        ax2.set_ylabel("total variance\n(raw units)", fontsize=8)
        ax2.grid(True, axis="y", alpha=0.3)
    tgt = ax2 if ax2 is not None else ax
    tgt.set_xticks(x)
    tgt.set_xticklabels(labels, rotation=90,
                        fontsize=7 if by == "func" else 6.5)
    tgt.set_xlabel("BBOB function" if by == "func" else "feature")

    nsg = float((d["no_sig"]).mean())
    fig.suptitle(title or f"Variance composition by {by}  "
                          f"[{strategy}, n={size}xd, dim {dimension}]  "
                          f"— {nsg:.0%} of cells have no detectable signal",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, (ax, ax2)


# =========================================================================== #
# Scoping: restrict to cells where a sampler comparison is possible            #
# =========================================================================== #

def scope_comparable(df, mode="all", baseline="uniform", verbose=True):
    """Keep only cells where a strategy comparison is MEANINGFUL.

    Where sigma_b ~ 0 -- f20/f23/f24 (exchangeable instances) and the PCA "_x"
    features (computed from the design matrix alone) -- every sampler floors at
    ICC ~ 0. They then all tie, and a tie at a floor is NOT evidence of
    equivalence: including those cells dilutes real differences toward zero and
    makes samplers look more alike than they are.

    Matching is done on (dimension, size, func, feature) so the surviving set is
    identical for every strategy -- otherwise strategies would be compared on
    different cells.

    mode="all"      keep a cell only if EVERY strategy has sigma2_between > 0.
                    Strictest and safest: no strategy is judged on a cell that
                    floored for it.
    mode="baseline" keep if the BASELINE has signal. Use when you want to ask
                    "does this sampler recover signal that uniform finds?" and
                    are willing to let a strategy score badly by flooring.
    mode="any"      keep if at least one strategy has signal -- the most
                    permissive, useful for asking whether a sampler CREATES
                    signal others miss.

    Returns the filtered frame with `.attrs["scope"]` describing what was kept.
    """
    keys = ["dimension", "size", "func", "feature"]
    d = df[np.isfinite(df["sigma2_within"]) & (df["sigma2_within"] > 0)].copy()
    has = d["sigma2_between"] > 0

    if mode == "all":
        ok = d.assign(_ok=has).groupby(keys)["_ok"].transform("all")
    elif mode == "any":
        ok = d.assign(_ok=has).groupby(keys)["_ok"].transform("any")
    elif mode == "baseline":
        b = (d[d["strategy"] == baseline].assign(_ok=lambda x: x["sigma2_between"] > 0)
             .set_index(keys)["_ok"])
        if b.empty:
            raise ValueError(f"baseline {baseline!r} not present")
        ok = d.set_index(keys).index.map(b).fillna(False).to_numpy()
    else:
        raise ValueError("mode must be 'all', 'baseline' or 'any'")

    out = d[np.asarray(ok)].copy()
    n_before = d.groupby(keys).ngroups
    n_after = out.groupby(keys).ngroups if len(out) else 0
    out.attrs["scope"] = dict(mode=mode, baseline=baseline,
                              cells_before=n_before, cells_after=n_after)
    if verbose:
        print(f"scope[{mode}]: kept {n_after:,} of {n_before:,} "
              f"(function, feature, size) cells "
              f"({n_after / max(n_before, 1):.1%}) with detectable signal")
        if n_after:
            lost_f = (d.groupby("func").ngroups - out.groupby("func").ngroups)
            gone = sorted(set(d["func"]) - set(out["func"]))
            if gone:
                print(f"  functions with NO comparable cells at all: "
                      f"{', '.join('f%d' % f for f in gone)}")
            drop = (d.groupby("func").size() - out.groupby("func").size()
                    ).sort_values(ascending=False).head(5)
            print("  most cells dropped: "
                  + ", ".join(f"f{f} ({int(c)})" for f, c in drop.items()))
    return out


# =========================================================================== #
# I9 — sigma_b and sigma_w ratios side by side                                #
# =========================================================================== #

def component_ratios(df, baseline="uniform", eps=1e-15):
    """Per-cell ratios of BOTH variance components against the baseline.

    Matched on (dimension, size, func, feature), so the units cancel and the
    ratios are dimensionless and poolable -- the same licence the sd ratios have
    in the reproducibility study.

    ratio_w < 1  -> the sampler measures more precisely (less run noise)
    ratio_b ~ 1  -> it measures THE SAME THING; sigma_b is a property of the
                    instances, so a sampler should leave it alone
    ratio_b != 1 -> it measures something DIFFERENT. For an adaptive sampler
                    this is the expected signature: the region it explores is
                    itself instance-dependent, so a higher ICC there is not
                    extra precision.

    Separating those two is the point: ICC alone cannot say whether a sampler
    won by reducing noise or by changing the signal.
    """
    keys = ["dimension", "size", "func", "feature"]
    d = df[(df["sigma2_within"] > 0) & np.isfinite(df["sigma2_between"])].copy()
    d["sd_b"] = np.sqrt(d["sigma2_between"].clip(lower=0))
    d["sd_w"] = np.sqrt(d["sigma2_within"])
    b = (d[d["strategy"] == baseline].set_index(keys)[["sd_b", "sd_w"]]
         .rename(columns={"sd_b": "b0", "sd_w": "w0"}))
    if b.empty:
        raise ValueError(f"baseline {baseline!r} not present")
    out = d.join(b, on=keys)
    out = out[(out["w0"] > eps)].copy()
    out["ratio_w"] = out["sd_w"] / out["w0"]
    out["ratio_b"] = np.where(out["b0"] > eps, out["sd_b"] / out["b0"], np.nan)
    out.attrs["baseline"] = baseline
    return out


def plot_component_ratios(ratios, dimension=None, size=None, strategies=None,
                          clip_pct=(2, 98), width=11.0, height=5.4, title=None):
    """I9 — side-by-side boxplots of ratio_w and ratio_b against the baseline.

    LEFT panel  (noise): below 1 means the sampler is more precise.
    RIGHT panel (signal): should sit AT 1 if the sampler measures the same
    quantity. A systematic departure means it does not, and any ICC advantage it
    shows is partly a change of target rather than a gain in precision.

    Read them together: an ICC win with ratio_w < 1 and ratio_b ~ 1 is a genuine
    precision win. An ICC win driven by ratio_b > 1 is not.
    """
    d = ratios
    if dimension is not None:
        d = d[d["dimension"] == dimension]
    if size is not None:
        d = d[d["size"] == size]
    base = ratios.attrs.get("baseline", "baseline")
    strats = strategies or sorted(d["strategy"].unique())
    color_of = _strategy_colors(strats)

    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharey=True)
    for ax, col, lab in [(axes[0], "ratio_w", "noise  $\\sigma_w$"),
                         (axes[1], "ratio_b", "signal  $\\sigma_b$")]:
        sub = d[np.isfinite(d[col]) & (d[col] > 0)]
        data = [sub.loc[sub["strategy"] == s, col].to_numpy() for s in strats]
        bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.62,
                        showfliers=False,
                        medianprops=dict(color="black", lw=1.2))
        for p, s in zip(bp["boxes"], strats):
            p.set_facecolor(color_of[s]); p.set_alpha(0.7)
            p.set_edgecolor(color_of[s])
        ax.axvline(1.0, color="black", ls="--", lw=1.2)
        ax.set_xscale("log")
        if clip_pct and len(sub):
            lo, hi = np.percentile(sub[col], clip_pct)
            ax.set_xlim(lo / 1.6, hi * 1.6)
        ax.set_xlabel(f"{lab}  ratio vs {base}")
        ax.grid(True, axis="x", alpha=0.3)
        med = sub.groupby("strategy")[col].median()
        for i, s in enumerate(strats, start=1):
            if s in med:
                ax.text(0.99, i + 0.3, f"{med[s]:.2f}x", fontsize=8,
                        transform=ax.get_yaxis_transform(), ha="right")
    axes[0].set_yticks(range(1, len(strats) + 1))
    axes[0].set_yticklabels(strats)
    axes[0].set_title("more precise  <-  |  ->  noisier", fontsize=9)
    axes[1].set_title("weaker signal  <-  |  ->  stronger signal", fontsize=9)
    tag = []
    if dimension is not None:
        tag.append(f"dim {dimension}")
    if size is not None:
        tag.append(f"n={size}xd")
    fig.suptitle(title or f"Did the sampler cut NOISE or change the SIGNAL?"
                          + (f"  [{', '.join(tag)}]" if tag else ""), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, axes


def plot_component_ratios_vs_budget(ratios, dimension=None, strategies=None,
                                    agg="median", show_band=False, logy=True,
                                    width_per=5.4, height=4.6, title=None):
    """I10 — sigma_w and sigma_b ratios against BUDGET, one line per strategy.

    The trend version of I9, and the figure that separates a better RATE from a
    better CONSTANT:

      LEFT  (noise): a line that FALLS with budget means the sampler's precision
            advantage compounds -- a better convergence rate. A FLAT line means a
            fixed multiplicative gain at every budget.
      RIGHT (signal): should stay AT 1.0. A non-adaptive design cannot change
            what is being measured, so any drift is either estimation bias (it
            shrinks toward 1 as n grows) or, if it moves AWAY from 1, evidence
            that the sampler is measuring a progressively different quantity.

    That second reading is what identifies an adaptive sampler: falling sigma_b
    with rising sigma_w means it is destroying the between-instance differences
    while becoming less repeatable -- not merely noisy, but measuring the wrong
    thing more and more as the budget grows.
    """
    d = ratios if dimension is None else ratios[ratios["dimension"] == dimension]
    strats = strategies or sorted(d["strategy"].unique())
    color_of = _strategy_colors(strats)

    def _agg(s):
        return (float(np.exp(np.mean(np.log(s)))) if agg == "geomean"
                else float(np.median(s)))

    fig, axes = plt.subplots(1, 2, figsize=(width_per * 2, height), sharex=True)
    for ax, col, lab in [(axes[0], "ratio_w", "noise  $\\sigma_w$ ratio"),
                         (axes[1], "ratio_b", "signal  $\\sigma_b$ ratio")]:
        sub = d[np.isfinite(d[col]) & (d[col] > 0)]
        for s in strats:
            g = (sub[sub["strategy"] == s].groupby("size")[col]
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
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("budget multiplier (n = size x d)")
        ax.set_ylabel(lab)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_title("falling = advantage compounds with budget", fontsize=9)
    axes[1].set_title("1.0 = measuring the same quantity", fontsize=9)

    handles = [Line2D([0], [0], color=color_of[s], marker="o", label=s)
               for s in strats]
    base = ratios.attrs.get("baseline", "baseline")
    dtag = f"  [dim {dimension}]" if dimension is not None else ""
    fig.suptitle(title or f"Noise and signal vs budget, relative to {base}{dtag}",
                 fontsize=12)
    _legend_right(fig, handles, "sampling strategy", rect=(0, 0, 0.86, 0.93),
                  x=0.87)
    return fig, axes


def heatmap_no_signal(df, dimension, strategy=None, size=None, features=None,
                      funcs=None, sort_features=True, aggregate=False,
                      width=13.0, row_height=0.24, title=None):
    """I11 — binary map of WHERE the between-instance signal is absent.

    Rows = features, columns = 24 functions. A cell is dark when
    sigma2_between <= 0, i.e. that feature cannot tell that function's instances
    apart at all.

    WHY BINARY. sigma_b is in feature units, so a magnitude heatmap needs a
    normaliser and the choice of normaliser then colours the reading. The
    question here is not "how much signal" but "is there any", which is a
    boolean -- so no scale, no normaliser, nothing to defend.

    WHY TWO AXES. The I6 bar charts collapse one axis each, so they cannot
    distinguish a feature that fails EVERYWHERE from one that fails only on the
    few exchangeable-instance functions. Here the structure is visible directly:

        dark ROW    -> the feature is invariant by construction (e.g. the PCA
                       "_x" variants are computed from the design matrix alone,
                       so they carry no function information at all)
        dark COLUMN -> the function's instances are statistically exchangeable
                       (f20/f23/f24 redraw their structure per instance)
        isolated cells -> a genuine feature-function interaction

    Margins show each row's and column's failure rate, so I6's information is
    kept without needing the separate bar charts.

    aggregate=True pools over strategies/sizes and shades by the FRACTION of
    configs in which the cell had no signal, rather than requiring a single
    (strategy, size).
    """
    d = df[df["dimension"] == dimension].copy()
    if strategy is not None:
        d = d[d["strategy"] == strategy]
    if size is not None:
        d = d[d["size"] == size]
    if features is not None:
        d = d[d["feature"].isin(set(features))]
    if funcs is not None:
        d = d[d["func"].isin(funcs)]
    if d.empty:
        raise ValueError("No rows for that selection.")
    if not aggregate and (strategy is None or size is None):
        raise ValueError("pass strategy= and size=, or set aggregate=True")

    M = (d.pivot_table(index="feature", columns="func", values="no_signal",
                       aggfunc="mean"))
    if sort_features:
        M = M.loc[M.mean(axis=1).sort_values(ascending=False).index]
    M = M[sorted(M.columns)]
    A = M.to_numpy(dtype=float)
    funcs_ = list(M.columns); feats_ = list(M.index)

    row_rate = np.nanmean(A, axis=1)
    col_rate = np.nanmean(A, axis=0)

    fig, axes = plt.subplots(
        2, 2, figsize=(width, max(3.4, row_height * len(feats_) + 2.6)),
        gridspec_kw=dict(width_ratios=[10, 1.1], height_ratios=[1.1, 10],
                         wspace=0.02, hspace=0.02))
    ax_top, ax_none = axes[0]
    ax, ax_right = axes[1]
    ax_none.set_visible(False)

    cmap = plt.get_cmap("Greys").copy(); cmap.set_bad("white")
    im = ax.imshow(A, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xticks(range(len(funcs_)))
    ax.set_xticklabels([f"f{f}" for f in funcs_], fontsize=7, rotation=90)
    ax.set_yticks(range(len(feats_)))
    ax.set_yticklabels(feats_, fontsize=6.5)
    _func_separators(ax, funcs_, "x")
    ax.set_xlabel("BBOB function")

    ax_top.bar(range(len(funcs_)), col_rate, color="#4c4c4c", width=0.85)
    ax_top.set_xlim(ax.get_xlim()); ax_top.set_ylim(0, 1)
    ax_top.set_xticks([]); ax_top.set_ylabel("per\nfunction", fontsize=7)
    ax_top.tick_params(labelsize=6); ax_top.grid(True, axis="y", alpha=0.3)
    for b in GROUP_BOUNDARIES:
        if b in funcs_ and any(f > b for f in funcs_):
            ax_top.axvline(funcs_.index(b) + 0.5, color="black", lw=0.8, alpha=0.5)

    ax_right.barh(range(len(feats_)), row_rate, color="#4c4c4c", height=0.85)
    ax_right.set_ylim(ax.get_ylim()); ax_right.set_xlim(0, 1)
    ax_right.set_yticks([]); ax_right.set_xlabel("per feature", fontsize=7)
    ax_right.tick_params(labelsize=6); ax_right.grid(True, axis="x", alpha=0.3)

    cb = fig.colorbar(im, ax=ax_right, fraction=0.35, pad=0.55)
    cb.set_label("share of configs with no signal" if aggregate
                 else "no signal (1 = absent)", fontsize=8)
    tag = f"{strategy}, n={size}xd" if not aggregate else "pooled over configs"
    overall = float(np.nanmean(A))
    fig.suptitle(title or f"Where the between-instance signal is absent  "
                          f"[{tag}, dim {dimension}]  — {overall:.0%} of cells",
                 fontsize=12)
    return fig, (ax, ax_top, ax_right)


def plot_icc_overall_vs_budget(scoped, dimension=None, strategies=None,
                               value="icc_pos", show_band=True, ncols=None,
                               width_per=5.4, height_per=4.2, title=None):
    """I12 — overall ICC vs budget: LEVEL (left) and CEILING-FREE RANK (right).

    MUST be given a SCOPED frame (scope_comparable). Cells with no
    between-instance signal floor at ICC ~ 0 for every strategy, so including
    them drags all curves toward zero by a common amount, compressing the
    differences -- and because the share of such cells is not constant across
    budgets, it distorts the trend as well, not just the level.

    Two panels, because a single aggregate cannot be both simple and safe:

      LEFT  median ICC across scoped cells. Easy to read, and fine for the
            TREND, but the level is contaminated: attainable ICC differs by
            function (ceilings run from ~0.005 to ~1.0 in this data), so the
            median partly reflects which functions have high ceilings rather
            than which sampler is better.

      RIGHT mean normalised rank. Strategies are ranked WITHIN each
            (func, feature) -- where the ceiling is shared by construction --
            then the ranks are averaged. Ceiling-free, so the ORDERING here is
            the defensible one. 1.0 = best on every cell, 0.0 = worst.

    If the two panels agree, report the simpler left one. If they disagree, the
    left panel was being driven by high-ceiling functions and the right is the
    result to quote.

    NOTE: sigma_w falls with budget while sigma_b is fixed by the instances, so
    ICC rising with budget is largely mechanical -- this figure restates the
    sigma_w result in a bounded form. Read it for the ORDERING and for where
    curves cross, not as independent evidence that budget buys reliability.
    """
    d = scoped if dimension is None else scoped[scoped["dimension"] == dimension]
    d = d[np.isfinite(d[value])].copy()
    if d.empty:
        raise ValueError("No scoped cells for that selection.")
    strats = strategies or sorted(d["strategy"].unique())
    color_of = _strategy_colors(strats)

    # ceiling-free: rank within each (func, feature) cell, normalise to [0,1]
    keys = ["dimension", "size", "func", "feature"]
    d["nrank"] = d.groupby(keys)[value].rank(ascending=True, pct=True,
                                             method="average")

    dims = sorted(d["dimension"].unique())
    fig, axes = plt.subplots(len(dims), 2,
                             figsize=(width_per * 2, height_per * len(dims)),
                             squeeze=False, sharex="col")
    for r_i, dim in enumerate(dims):
        sub = d[d["dimension"] == dim]
        for c_i, (col, lab, ylim) in enumerate(
                [(value, "median ICC (scoped cells)", (0, 1)),
                 ("nrank", "mean normalised rank\n(1 = best, ceiling-free)", (0, 1))]):
            ax = axes[r_i][c_i]
            for s in strats:
                g = (sub[sub["strategy"] == s].groupby("size")[col]
                     .agg(mid="median" if col == value else "mean",
                          lo=lambda x: float(np.percentile(x, 25)),
                          hi=lambda x: float(np.percentile(x, 75)))
                     .reset_index().sort_values("size"))
                if g.empty:
                    continue
                ax.plot(g["size"], g["mid"], marker="o", ms=5, lw=1.8,
                        color=color_of[s], label=s)
                if show_band and col == value:
                    ax.fill_between(g["size"], g["lo"], g["hi"],
                                    color=color_of[s], alpha=0.10, lw=0)
            ax.set_ylim(*ylim)
            ax.set_ylabel(lab if c_i == 0 else lab, fontsize=9)
            ax.set_xlabel("budget multiplier (n = size x d)")
            ax.grid(True, alpha=0.3)
            if r_i == 0:
                ax.set_title("level (ceiling-contaminated)" if c_i == 0
                             else "ordering (ceiling-free)", fontsize=10)
            if len(dims) > 1 and c_i == 0:
                ax.text(0.02, 0.95, f"dim {dim}", transform=ax.transAxes,
                        fontsize=9, va="top", fontweight="bold")

    handles = [Line2D([0], [0], color=color_of[s], marker="o", label=s)
               for s in strats]
    n_cells = d.groupby(keys).ngroups
    fig.suptitle(title or f"Overall ICC vs budget — {n_cells:,} signal-bearing "
                          f"cells", fontsize=12)
    _legend_right(fig, handles, "sampling strategy", rect=(0, 0, 0.86, 0.93),
                  x=0.87)
    return fig, axes


def no_signal_summary(df, dimension=None, strategy=None, size=None,
                      by="func", top=None, as_percent=True):
    """Summary TABLE of where the between-instance signal is absent.

    The quotable companion to heatmap_no_signal: same quantity, exact numbers.

    by="func"      one row per BBOB function
    by="feature"   one row per feature
    by="strategy"  one row per sampler -- does a sampler CREATE no-signal cells?
    by="overall"   one row per (dimension, strategy, size)

    Columns: n_cells, n_no_signal, pct_no_signal, and median ICC over the cells
    that DO have signal (so the two questions -- "is there signal" and "how
    reliable is it where there is" -- are not conflated).

    Filter to a single (strategy, size) for a per-config reading; leave them out
    to pool, but then say so, since pooling mixes samplers with very different
    no-signal rates.
    """
    d = df if dimension is None else df[df["dimension"] == dimension]
    if strategy is not None:
        d = d[d["strategy"] == strategy]
    if size is not None:
        d = d[d["size"] == size]
    if d.empty:
        raise ValueError("No rows for that selection.")

    keys = {"func": ["func"], "feature": ["feature"], "strategy": ["strategy"],
            "overall": ["dimension", "strategy", "size"]}[by]

    g = (d.groupby(keys)
         .agg(n_cells=("no_signal", "size"),
              n_no_signal=("no_signal", "sum"),
              med_icc_where_signal=("icc_pos",
                                    lambda s: np.nan))
         .reset_index())
    # median ICC over signal-bearing cells only
    sig = d[~d["no_signal"]]
    med = (sig.groupby(keys)["icc_pos"].median()
           .rename("med_icc_where_signal").reset_index())
    g = g.drop(columns=["med_icc_where_signal"]).merge(med, on=keys, how="left")

    g["pct_no_signal"] = g["n_no_signal"] / g["n_cells"]
    if as_percent:
        g["pct_no_signal"] = (100 * g["pct_no_signal"]).round(1)
        g["med_icc_where_signal"] = g["med_icc_where_signal"].round(3)
    if by == "func":
        g["group"] = g["func"].map(lambda f: GROUP_LABEL.get(FUNC_TO_GROUP.get(f), ""))
    g = g.sort_values("pct_no_signal", ascending=False).reset_index(drop=True)
    return g.head(top) if top else g


def plot_no_signal_by_config(df, dimensions=None, strategies=None, kind="line",
                             also_icc=False, ylim=None, ncols=None,
                             width_per=4.8, height_per=4.0, title=None):
    """I13 — no-signal rate per (strategy, budget), one panel per dimension.

    The plotted form of no_signal_summary(by="overall"). Answers a question the
    per-feature and per-function views cannot: does the SAMPLER itself affect
    how often the between-instance signal disappears?

    A FLAT line means the no-signal rate is a property of the features and
    functions, not of the sampling -- the expected result for a non-adaptive
    design, which cannot change what is there to be detected.

    A RISING line means the sampler destroys signal as the budget grows. For an
    adaptive sampler that is the signature of convergence: the more it adapts,
    the more it looks only near each instance's optimum, and the more the
    instances resemble one another.

    also_icc=True adds a second row with the median ICC over the cells that DO
    have signal, so "is there signal" and "how reliable where there is" stay
    separate -- a sampler can score well on one and badly on the other.
    """
    d = df if dimensions is None else df[df["dimension"].isin(dimensions)]
    if strategies is not None:
        d = d[d["strategy"].isin(strategies)]
    g = (d.groupby(["dimension", "strategy", "size"])
         .agg(pct=("no_signal", "mean"), n=("no_signal", "size"))
         .reset_index())
    sig = d[~d["no_signal"]]
    med = (sig.groupby(["dimension", "strategy", "size"])["icc_pos"]
           .median().rename("med_icc").reset_index())
    g = g.merge(med, on=["dimension", "strategy", "size"], how="left")

    dims = sorted(g["dimension"].unique())
    strats = strategies or sorted(g["strategy"].unique())
    color_of = _strategy_colors(strats)
    sizes = sorted(g["size"].unique())

    nrows = 2 if also_icc else 1
    n = len(dims); ncols = ncols or n
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_per * ncols, height_per * nrows),
                             squeeze=False, sharey="row", sharex=True)

    for ci, dim in enumerate(dims):
        sub = g[g["dimension"] == dim]
        ax = axes[0][ci]
        if kind == "line":
            for s in strats:
                q = sub[sub["strategy"] == s].sort_values("size")
                if len(q):
                    ax.plot(q["size"], q["pct"], marker="o", ms=5, lw=1.8,
                            color=color_of[s], label=s)
            ax.set_xticks(sizes)
        else:                                   # grouped bars
            w = 0.8 / max(len(strats), 1)
            x = np.arange(len(sizes))
            for i, s in enumerate(strats):
                q = (sub[sub["strategy"] == s].set_index("size")
                     .reindex(sizes)["pct"].fillna(0))
                ax.bar(x + (i - (len(strats) - 1) / 2) * w, q, width=w * 0.9,
                       color=color_of[s], label=s)
            ax.set_xticks(x); ax.set_xticklabels(sizes)
        ax.set_title(f"dimension {dim}")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=0.3)
        if ci == 0:
            ax.set_ylabel("share of cells with\nno between-instance signal")
        if not also_icc:
            ax.set_xlabel("budget multiplier (n = size x d)")

        if also_icc:
            ax2 = axes[1][ci]
            for s in strats:
                q = sub[sub["strategy"] == s].sort_values("size")
                if len(q):
                    ax2.plot(q["size"], q["med_icc"], marker="s", ms=4.5,
                             lw=1.6, ls="--", color=color_of[s])
            ax2.set_xticks(sizes)
            ax2.set_xlabel("budget multiplier (n = size x d)")
            ax2.grid(True, axis="y", alpha=0.3)
            if ci == 0:
                ax2.set_ylabel("median ICC\n(cells WITH signal)")

    handles = [Line2D([0], [0], color=color_of[s], marker="o", label=s)
               for s in strats]
    fig.suptitle(title or "Does the sampler affect how often the signal "
                          "disappears?", fontsize=12)
    _legend_right(fig, handles, "sampling strategy", rect=(0, 0, 0.86, 0.93),
                  x=0.87)
    return fig, axes