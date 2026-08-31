"""
repro_analysis.py

Foundation layer for the per-function reproducibility study. Consumes the tidy
table from compute_ela_repro.repro_to_dataframe().

CORE PRINCIPLE: FUNCTIONS ARE NEVER POOLED
------------------------------------------
sigma_run is in feature units and each function has its own scale, so averaging
sigma across functions is meaningless. Aggregation therefore stops at

    (dimension, strategy, size, func, feature)

Instances DO collapse (by median) -- the 100 instances of one function share a
landscape, so they are commensurable. Functions never collapse.

To make statements ACROSS functions there are exactly two legitimate routes:

  1. ORDINAL -- rank strategies within each function, then count wins across
     functions. Never adds incommensurable quantities.
  2. ALPHA -- the convergence exponent in  log sigma = log C - alpha * log n
     is DIMENSIONLESS, so it can be pooled, averaged and histogrammed freely.
     This is the one quantity that supports clean cross-function statements.

NORMALISATION
-------------
For cross-FEATURE comparisons sigma must be rescaled. `feature_scale` computes,
per (dimension, feature), the spread of the feature's typical value ACROSS
functions -- i.e. the usable dynamic range of that feature. sigma_norm =
sigma / scale then reads as "run noise as a share of the feature's usable
range". The scale is pooled over strategies and budgets so it is a fixed
per-feature constant and does not itself encode a strategy effect.

Deliberately NOT used: the spread across instances within a function. That is a
bijection with ICC and degenerates for transformation-invariant features.
"""

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# BBOB structure (labels only -- functions are never pooled for analysis)      #
# --------------------------------------------------------------------------- #

BBOB_GROUPS = {
    "separable": range(1, 6),
    "low_cond": range(6, 10),
    "high_cond": range(10, 15),
    "multi_global": range(15, 20),
    "multi_weak": range(20, 25),
}
GROUP_ORDER = ["separable", "low_cond", "high_cond", "multi_global", "multi_weak"]
GROUP_LABEL = {
    "separable": "separable (f1-f5)",
    "low_cond": "low/mod. conditioning (f6-f9)",
    "high_cond": "high cond. unimodal (f10-f14)",
    "multi_global": "multimodal w/ structure (f15-f19)",
    "multi_weak": "multimodal weak structure (f20-f24)",
}
FUNC_TO_GROUP = {f: g for g, rng in BBOB_GROUPS.items() for f in rng}
GROUP_BOUNDARIES = [5, 9, 14, 19]      # draw separators after these function ids


# --------------------------------------------------------------------------- #
# Feature exclusions (applied HERE, not at compute time)                       #
# --------------------------------------------------------------------------- #
# compute_ela_repro.py deliberately computes every feature, so the exclusion
# policy can change without a recompute. These are the features that cannot be
# computed reliably at a given dimension:
#
#   disp.*_02 at dim 2 -- the "02" variants use the best 2% of points. At dim 2
#     the smallest budget is n = 25*2 = 50, so 2% = 1 point and dispersion among
#     a single point is undefined. The feature is simply absent at that budget,
#     which also breaks the budget ratio and the alpha fit.
#
#   ela_meta.quad_simple.cond at every dimension -- a condition number, which
#     diverges when the quadratic design matrix is near-singular. It fails on
#     ~3-4% of cells, and the failing runs are the LARGEST values, so an sd
#     computed from the survivors is biased downward rather than merely noisy.
#
# Mirrors compute_ela_icc.get_omit_features so both pipelines share a feature
# set. ic.eps_ratio is NOT dropped (only 11 cells, 28-29/30 runs each -- the ICC
# list is over-conservative there).
OMIT_FEATURES = {
    2: {"disp.diff_mean_02", "disp.diff_median_02",
        "disp.ratio_mean_02", "disp.ratio_median_02",
        "ela_meta.quad_simple.cond"},
    5: {"ela_meta.quad_simple.cond"},
    10: {"ela_meta.quad_simple.cond"},
}


def drop_omitted(df, omit=None, verbose=False):
    """Remove the per-dimension excluded features from a tidy table.

    `omit` overrides OMIT_FEATURES; pass {} to keep everything.
    """
    omit = OMIT_FEATURES if omit is None else omit
    if not omit:
        return df
    mask = pd.Series(False, index=df.index)
    for dim, feats in omit.items():
        mask |= (df["dimension"] == dim) & df["feature"].isin(feats)
    if verbose and mask.any():
        n = int(mask.sum())
        which = sorted(df.loc[mask, "feature"].unique())
        print(f"dropped {n:,} rows for {len(which)} excluded feature(s): "
              f"{', '.join(which)}")
    return df[~mask].copy()


def add_group(df, col="func"):
    """Attach group key + label columns (annotation only)."""
    out = df.copy()
    out["fgroup"] = out[col].map(FUNC_TO_GROUP)
    out["fgroup_label"] = out["fgroup"].map(GROUP_LABEL)
    return out


# --------------------------------------------------------------------------- #
# 1. Aggregate instances -> per-function                                       #
# --------------------------------------------------------------------------- #

def aggregate_per_function(df, stat="sd", agg="median", min_cells=5):
    """Collapse the instance axis only.

    Returns one row per (dimension, strategy, size, func, feature) with:
      value        : agg of `stat` over the function's instances (the sigma)
      value_iqr    : IQR of `stat` across instances -- heteroscedasticity check
      level        : agg of the feature's own median value (its typical level,
                     needed for the normaliser and for CV)
      n_inst       : instances contributing
      frac_zero_sd : share of instances with sd == 0 (degenerate/discrete)

    `agg="median"` throughout: the sd-vs-mad diagnostic shows heavy tails, so a
    mean would be outlier-driven.
    """
    aggf = "median" if agg == "median" else "mean"
    d = df[np.isfinite(df[stat])]
    g = (d.groupby(["dimension", "strategy", "size", "func", "feature_group",
                    "feature"])
         .agg(value=(stat, aggf),
              value_iqr=(stat, lambda s: float(np.subtract(*np.percentile(s, [75, 25])))),
              level=("median", aggf),
              n_inst=(stat, "size"),
              frac_zero_sd=(stat, lambda s: float(np.mean(s == 0))))
         .reset_index())
    g = g[g["n_inst"] >= min_cells]
    return add_group(g)


# --------------------------------------------------------------------------- #
# 2. Normalisation                                                             #
# --------------------------------------------------------------------------- #

def feature_scale(per_func, how="iqr"):
    """Per (dimension, feature) scale = spread of the feature's typical value
    ACROSS functions. Pooled over strategies and budgets so it is a fixed
    constant per feature and carries no strategy effect.

    how="iqr"   robust (default)
    how="range" max-min across functions
    how="sd"    standard deviation across functions
    """
    # one level per (dimension, feature, func), pooled over strategy/size
    lvl = (per_func.groupby(["dimension", "feature", "func"])["level"]
           .median().reset_index())

    def _spread(s):
        s = s[np.isfinite(s)]
        if s.size < 2:
            return np.nan
        if how == "iqr":
            return float(np.subtract(*np.percentile(s, [75, 25])))
        if how == "range":
            return float(s.max() - s.min())
        return float(np.std(s, ddof=1))

    sc = (lvl.groupby(["dimension", "feature"])["level"]
          .apply(_spread).reset_index()
          .rename(columns={"level": "scale"}))
    return sc


def add_normalised(per_func, how="iqr", eps=1e-12):
    """Attach `scale` and `value_norm` = sigma / scale.

    value_norm reads as "run noise as a share of the feature's usable range
    across functions", and is comparable ACROSS FEATURES. It is still a
    per-function quantity -- it does not license averaging over functions.
    Also attaches `cv` = sigma / |level| for reference (valid only where the
    feature is ratio-scaled and strictly positive).
    """
    sc = feature_scale(per_func, how=how)
    out = per_func.merge(sc, on=["dimension", "feature"], how="left")
    out["value_norm"] = np.where(np.abs(out["scale"]) > eps,
                                 out["value"] / out["scale"], np.nan)
    out["cv"] = np.where(np.abs(out["level"]) > eps,
                         out["value"] / np.abs(out["level"]), np.nan)
    return out


# --------------------------------------------------------------------------- #
# 3. Ordinal cross-function summaries                                          #
# --------------------------------------------------------------------------- #

def rank_strategies(per_func, value="value", size=None):
    """Rank strategies WITHIN each (dimension, size, func, feature).

    Lower sigma = better = rank 1. This is the only legitimate way to compare
    strategies across functions, since it never adds incommensurable sigmas.
    """
    d = per_func if size is None else per_func[per_func["size"] == size]
    d = d[np.isfinite(d[value])].copy()
    keys = ["dimension", "size", "func", "feature"]
    d["rank"] = d.groupby(keys)[value].rank(ascending=True, method="min")
    d["n_strat"] = d.groupby(keys)[value].transform("size")
    d["is_win"] = d["rank"] == 1
    return d


def strategy_by_function(ranked):
    """Two-stage ordinal aggregation, stage 1: collapse the FEATURE axis.

    Per (dimension, size, func, strategy): mean rank across features, then rank
    the strategies within that function. Returns one row per
    (dimension, size, func, strategy) with `mean_rank` and `func_rank`.
    """
    g = (ranked.groupby(["dimension", "size", "func", "strategy"])
         .agg(mean_rank=("rank", "mean"),
              win_frac=("is_win", "mean"),
              n_feat=("rank", "size"))
         .reset_index())
    g["func_rank"] = (g.groupby(["dimension", "size", "func"])["mean_rank"]
                      .rank(ascending=True, method="min").astype(int))
    return add_group(g)


def win_counts(sbf):
    """Stage 2: count, per (dimension, size, strategy), how many of the 24
    functions that strategy ranks first on. Never averages sigma."""
    g = (sbf.assign(won=lambda d: d["func_rank"] == 1)
         .groupby(["dimension", "size", "strategy"])
         .agg(wins=("won", "sum"), n_func=("func", "nunique"),
              mean_rank=("mean_rank", "mean"))
         .reset_index())
    g["win_frac"] = g["wins"] / g["n_func"]
    g["overall_rank"] = (g.groupby(["dimension", "size"])["mean_rank"]
                         .rank(ascending=True, method="min").astype(int))
    return g.sort_values(["dimension", "size", "overall_rank"])


# --------------------------------------------------------------------------- #
# 4. Convergence rate                                                          #
# --------------------------------------------------------------------------- #

def fit_alpha(per_func, value="value", min_points=3, use_absolute_n=True):
    """Fit  log(sigma) = log C - alpha * log(n)  per
    (dimension, strategy, func, feature).

    alpha is DIMENSIONLESS, so unlike sigma it is comparable and poolable across
    functions, features and dimensions. Reference points: alpha = 0.5 for plain
    Monte Carlo, up to alpha = 1 for QMC / optimised LHS on well-behaved
    integrands.

    use_absolute_n=True fits against n = size * dimension. (The slope is
    identical either way, since log(size*d) = log(size) + log(d); only the
    intercept shifts.)

    Returns alpha, its standard error, r2 and n_points. WARNING: with only four
    budget values the slope is weakly identified -- always report se/CI and
    treat alpha as indicative.
    """
    rows = []
    keys = ["dimension", "strategy", "func", "feature_group", "feature"]
    for key, g in per_func.groupby(keys):
        g = g[(g[value] > 0) & np.isfinite(g[value])]
        if len(g) < min_points:
            continue
        n = g["size"] * g["dimension"] if use_absolute_n else g["size"]
        x = np.log(n.to_numpy(dtype=float))
        y = np.log(g[value].to_numpy(dtype=float))
        if np.ptp(x) == 0:
            continue
        # OLS with standard error on the slope
        xm, ym = x.mean(), y.mean()
        sxx = np.sum((x - xm) ** 2)
        b = np.sum((x - xm) * (y - ym)) / sxx
        a = ym - b * xm
        resid = y - (a + b * x)
        dof = len(x) - 2
        if dof > 0:
            s2 = np.sum(resid ** 2) / dof
            se_b = float(np.sqrt(s2 / sxx))
        else:
            se_b = np.nan
        sst = np.sum((y - ym) ** 2)
        r2 = float(1 - np.sum(resid ** 2) / sst) if sst > 0 else np.nan
        rows.append(dict(zip(keys, key)) | {
            "alpha": float(-b),            # sigma DECREASES with n -> alpha > 0
            "alpha_se": se_b,
            "log_C": float(a),
            "r2": r2,
            "n_points": int(len(x)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["alpha_lo"] = out["alpha"] - 1.96 * out["alpha_se"]
        out["alpha_hi"] = out["alpha"] + 1.96 * out["alpha_se"]
        out = add_group(out)
    return out


# --------------------------------------------------------------------------- #
# 5. Dimensionless SD ratios (the defensible way to aggregate)                 #
# --------------------------------------------------------------------------- #

def sd_ratio(per_func, baseline="uniform", value="value", eps=1e-15):
    """SD_strategy / SD_baseline, per (dimension, size, func, feature).

    Numerator and denominator share units, so the ratio is DIMENSIONLESS and can
    be pooled across features and functions without any normaliser -- the same
    licence alpha enjoys, and far easier to defend than an invented scale.

    ratio < 1  -> that strategy is less noisy than the baseline
    ratio = 1  -> identical
    `uniform` is the natural baseline: it is plain random sampling, so the ratio
    reads directly as "what do low-discrepancy designs buy over random?".
    """
    d = per_func[np.isfinite(per_func[value]) & (per_func[value] > 0)]
    keys = ["dimension", "size", "func", "feature_group", "feature"]
    base = (d[d["strategy"] == baseline]
            .set_index(keys)[value].rename("base"))
    if base.empty:
        raise ValueError(f"baseline strategy {baseline!r} not present")
    out = d.join(base, on=keys)
    out = out[np.isfinite(out["base"]) & (out["base"] > eps)].copy()
    out["ratio"] = out[value] / out["base"]
    out["log2_ratio"] = np.log2(out["ratio"])
    out.attrs["baseline"] = baseline
    return out


def budget_ratio(per_func, lo=None, hi=None, value="value", eps=1e-15):
    """SD(lo) / SD(hi) per (dimension, strategy, func, feature), plus the
    convergence exponent it implies.

    Since sigma ~ n^-alpha,

        SD(n_lo) / SD(n_hi) = (n_hi / n_lo) ^ alpha
        =>  alpha = log(ratio) / log(n_hi / n_lo)

    With lo=25 and hi=100 the budget quadruples, so:
        ratio 2.0 -> alpha 0.5  (Monte Carlo, 1/sqrt(n))
        ratio 2.8 -> alpha 0.75
        ratio 4.0 -> alpha 1.0  (QMC ideal, 1/n)

    This is the same information as the log-log regression, obtained by dividing
    two numbers -- no fit, no r2, nothing to defend beyond the two measurements.
    """
    sizes = sorted(per_func["size"].unique())
    lo = sizes[0] if lo is None else lo
    hi = sizes[-1] if hi is None else hi
    keys = ["dimension", "strategy", "func", "feature_group", "feature"]
    d = per_func[np.isfinite(per_func[value]) & (per_func[value] > 0)]
    a = d[d["size"] == lo].set_index(keys)[value].rename("sd_lo")
    b = d[d["size"] == hi].set_index(keys)[value].rename("sd_hi")
    out = pd.concat([a, b], axis=1).dropna().reset_index()
    out = out[out["sd_hi"] > eps].copy()
    out["ratio"] = out["sd_lo"] / out["sd_hi"]
    out["implied_alpha"] = np.log(out["ratio"]) / np.log(hi / lo)
    out["size_lo"], out["size_hi"] = lo, hi
    return add_group(out)


# --------------------------------------------------------------------------- #
# One-call pipeline                                                            #
# --------------------------------------------------------------------------- #

def prepare(df, stat="sd", how="iqr", baseline="uniform", omit=None,
            verbose=True):
    """df (from repro_to_dataframe) -> every table the figures need.

    Excluded features (OMIT_FEATURES) are dropped first; pass omit={} to keep
    them or a custom dict to override.
    """
    df = drop_omitted(df, omit=omit, verbose=verbose)
    per_func = add_normalised(aggregate_per_function(df, stat=stat), how=how)
    ranked = rank_strategies(per_func)
    sbf = strategy_by_function(ranked)
    wins = win_counts(sbf)
    alpha = fit_alpha(per_func)
    try:
        ratios = sd_ratio(per_func, baseline=baseline)
    except ValueError:
        ratios = pd.DataFrame()
    bratio = budget_ratio(per_func)
    if verbose:
        print(f"per-function rows : {len(per_func):,}  "
              f"({per_func['feature'].nunique()} features x "
              f"{per_func['func'].nunique()} functions x "
              f"{per_func['strategy'].nunique()} strategies)")
        if len(alpha):
            print(f"alpha fits        : {len(alpha):,}  "
                  f"(median r2 = {alpha['r2'].median():.3f})")
            print("median alpha by strategy (regression):")
            for st, a in alpha.groupby("strategy")["alpha"].median().sort_values(
                    ascending=False).items():
                print(f"    {st:<16s} {a:.3f}")
        if len(bratio):
            lo, hi = bratio["size_lo"].iloc[0], bratio["size_hi"].iloc[0]
            print(f"implied alpha from SD({lo}) / SD({hi}) ratio "
                  f"[no regression]:")
            for st, g in bratio.groupby("strategy"):
                print(f"    {st:<16s} ratio={g['ratio'].median():.2f}  "
                      f"alpha={g['implied_alpha'].median():.3f}")
        if len(ratios):
            print(f"SD ratio vs {baseline} (median, pooled):")
            for st, g in ratios.groupby("strategy"):
                print(f"    {st:<16s} {g['ratio'].median():.3f}x")
    return {"per_func": per_func, "ranked": ranked, "strategy_by_function": sbf,
            "wins": wins, "alpha": alpha, "ratios": ratios, "budget_ratio": bratio}