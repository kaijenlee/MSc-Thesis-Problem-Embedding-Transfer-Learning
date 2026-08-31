"""
rank_stability.py

Rank sampling strategies x budget (size = per-dimension multiplier, n = size*d)
by feature stability, in one pass computing three complementary metrics:

    overall_cv   reproducibility  (median per-feature within-instance CV; low good)
                 -- only meaningful on the safe tier, where CV is scale-valid.
    overall_icc  cross-instance reliability (median per-feature ICC(1,1); high good)
                 -- the binding constraint, since CV is broadly good for space-
                    filling designs.
    keep_count   joint criterion: # features that are BOTH reproducible
                 (CV < cv_thresh) AND reliable (ICC >= icc_thresh); high good.

`by` selects which metric ranks the table; all three are always reported.

Two guards, both on by default:
  * restrict_to_shared_features -- when comparing across dimensions, restrict to
    the feature set present (with finite CV and ICC) in EVERY dimension. This is
    essential because compute_ela_icc.get_omit_features drops six features at
    dim 2 (ic.eps_ratio, ela_meta.quad_simple.cond, the four disp.*_02), so the
    raw per-dim feature sets differ and an unguarded ranking mis-ranks dim 2.
  * clip-awareness -- ICC values are clipped to 0 at source (negative = noise
    >= signal). Aggregation uses the MEDIAN (robust to that left-censoring) and
    an `icc_zero_frac` column exposes how much of each cell sits on the floor,
    so a clipped 0 is never mistaken for a smooth measurement.

Per-feature aggregation matches the scatter/plots: CV = median over pooled
(func x inst); ICC = median over funcs. Built on scatter_cv_vs_icc.join_cv_icc.
"""

import numpy as np
import pandas as pd

from scatter_cv_vs_icc import join_cv_icc, _load, split_config_key


# metric -> higher_is_better
DIRECTION = {"overall_cv": False, "overall_icc": True,
             "keep_count": True, "keep_frac": True}


def _tier_ok(t, tier):
    if tier == "safe":
        return t == "safe"
    if tier == "caveat":
        return t in ("safe", "caveat")
    if tier == "all":
        return True
    raise ValueError("tier must be 'safe', 'caveat', or 'all'")


def _config_metrics(rows, cv_thresh, icc_thresh, feature_agg):
    agg = np.median if feature_agg == "median" else np.mean
    cvs = [r["cv"] for r in rows]
    iccs = [r["icc"] for r in rows]
    keep = sum(1 for r in rows if r["cv"] < cv_thresh and r["icc"] >= icc_thresh)
    n = len(rows)
    return {
        "n_feat": n,
        "overall_cv": float(agg(cvs)) if cvs else np.nan,
        "overall_icc": float(agg(iccs)) if iccs else np.nan,
        "keep_count": keep,
        "keep_frac": (keep / n) if n else np.nan,
        "icc_zero_frac": float(np.mean([r["icc"] <= 0 for r in rows])) if n else np.nan,
    }


# --------------------------------------------------------------------------- #
# Collect                                                                     #
# --------------------------------------------------------------------------- #

def collect_table(cv_sources, icc_sources, dimensions=None, strategies=None,
                  sizes=None, tier="safe", cap=None, cv_thresh=0.10,
                  icc_thresh=0.75, feature_agg="median",
                  restrict_to_shared_features=True):
    """Return (df, shared_features).

    df has one row per (dimension, strategy, size) with the metrics above.
    shared_features is the intersection used (or None if not restricting)."""
    dims = sorted(dimensions or cv_sources)
    loaded = {d: _load(cv_sources, d) for d in dims}

    configs = {}
    for d in dims:
        configs[d] = {split_config_key(c) for c in loaded[d]}
    all_strats = sorted({st for d in dims for (st, _) in configs[d]})
    all_sizes = sorted({sz for d in dims for (_, sz) in configs[d]})
    strategies = strategies or all_strats
    sizes = sizes or all_sizes

    # first pass: joined, tier-filtered rows per (dim, strategy, size)
    raw, feat_by_dim = {}, {d: set() for d in dims}
    for d in dims:
        for st in strategies:
            for sz in sizes:
                if (st, sz) not in configs[d]:
                    continue
                rows, _ = join_cv_icc(cv_sources, icc_sources, d, st, sz,
                                      level="feature", cap=cap)
                rows = [r for r in rows if _tier_ok(r["tier"], tier)]
                raw[(d, st, sz)] = rows
                feat_by_dim[d].update(r["feature"] for r in rows)

    shared = None
    if restrict_to_shared_features and len(dims) > 1:
        shared = set.intersection(*(feat_by_dim[d] for d in dims))

    records = []
    for (d, st, sz), rows in raw.items():
        use = [r for r in rows if shared is None or r["feature"] in shared]
        records.append({"dimension": d, "strategy": st, "size": sz,
                        **_config_metrics(use, cv_thresh, icc_thresh, feature_agg)})
    df = (pd.DataFrame(records)
          .sort_values(["dimension", "strategy", "size"])
          .reset_index(drop=True))
    return df, (sorted(shared) if shared is not None else None)


# --------------------------------------------------------------------------- #
# Rank + summarise                                                            #
# --------------------------------------------------------------------------- #

def min_budget(df, metric="overall_cv", target=0.10):
    """Smallest size (budget multiplier) at which each (dimension, strategy)
    first meets `target` on `metric`. None if never met within the sizes seen."""
    higher = DIRECTION[metric]
    out = []
    for (d, st), g in df.groupby(["dimension", "strategy"]):
        g = g.sort_values("size")
        hit = g[g[metric] >= target] if higher else g[g[metric] <= target]
        out.append({"dimension": d, "strategy": st,
                    "min_size": int(hit["size"].min()) if len(hit) else None})
    return pd.DataFrame(out).sort_values(["dimension", "strategy"]).reset_index(drop=True)


def strategy_summary(df, by="overall_cv"):
    """Collapse the size axis: per (dimension, strategy) report the best cell,
    the value at the largest budget, and the median across budgets; rank
    strategies within each dimension by the best cell."""
    higher = DIRECTION[by]
    rows = []
    for (d, st), g in df.groupby(["dimension", "strategy"]):
        g = g.sort_values("size")
        best = g[by].max() if higher else g[by].min()
        rows.append({"dimension": d, "strategy": st,
                     f"{by}_best": best,
                     f"{by}_at_max_budget": g[by].iloc[-1],
                     f"{by}_median": g[by].median()})
    out = pd.DataFrame(rows)
    out["rank"] = (out.groupby("dimension")[f"{by}_best"]
                   .rank(ascending=not higher, method="min").astype(int))
    return out.sort_values(["dimension", "rank"]).reset_index(drop=True)


def pivot_grid(df, by="overall_cv", dimension=None):
    """size x strategy matrix of `by` for one dimension (the ranking grid)."""
    d = df if dimension is None else df[df["dimension"] == dimension]
    return d.pivot_table(index="size", columns="strategy", values=by)


def rank_stability(cv_sources, icc_sources, dimensions=None, strategies=None,
                   sizes=None, tier="safe", by="overall_cv", cv_thresh=0.10,
                   icc_thresh=0.75, cap=None, feature_agg="median",
                   restrict_to_shared_features=True, min_budget_metric=None,
                   min_budget_target=None, verbose=True):
    """Top-level entry point.

    Returns a dict:
      by_config  : DataFrame, one row per (dimension, strategy, size) with all
                   metrics and a per-dimension 'rank' on `by`.
      by_strategy: DataFrame, size axis collapsed, strategies ranked per dim.
      min_budget : DataFrame, smallest budget meeting the target per strategy.
      shared_features / n_shared / ranked_by : provenance for the write-up.
    """
    if by not in DIRECTION:
        raise ValueError(f"by must be one of {list(DIRECTION)}")
    df, shared = collect_table(cv_sources, icc_sources, dimensions, strategies,
                               sizes, tier, cap, cv_thresh, icc_thresh,
                               feature_agg, restrict_to_shared_features)
    higher = DIRECTION[by]
    df["rank"] = (df.groupby("dimension")[by]
                  .rank(ascending=not higher, method="min").astype(int))
    df = df.sort_values(["dimension", "rank", "size"]).reset_index(drop=True)

    by_strategy = strategy_summary(df, by)
    mbm = min_budget_metric or by
    mbt = (min_budget_target if min_budget_target is not None
           else (cv_thresh if mbm == "overall_cv" else icc_thresh))
    mb = min_budget(df, mbm, mbt)

    if verbose:
        n_sh = "all" if shared is None else len(shared)
        arrow = "higher=better" if higher else "lower=better"
        print(f"Ranked by {by} ({arrow}) | tier={tier} | "
              f"features={n_sh} shared across dims | cap={cap}")
        print(f"Thresholds: CV<{cv_thresh:g}, ICC>={icc_thresh:g} | "
              f"per-feature agg={feature_agg} (median is clip-robust)")
        for d in sorted(df["dimension"].unique()):
            print(f"\n=== dimension {d} — size(=budget xdim) x strategy grid of {by} ===")
            grid = pivot_grid(df, by, d)
            with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
                print(grid.to_string())
            print(f"  strategy ranking (best {by} across budgets):")
            sub = by_strategy[by_strategy["dimension"] == d]
            for _, r in sub.iterrows():
                print(f"    {r['rank']}. {r['strategy']:<16s} "
                      f"best={r[f'{by}_best']:.3f}  "
                      f"@maxbudget={r[f'{by}_at_max_budget']:.3f}")
        print(f"\n=== minimum budget to reach {mbm} "
              f"{'>=' if DIRECTION[mbm] else '<='} {mbt:g} ===")
        with pd.option_context("display.max_rows", None):
            print(mb.to_string(index=False))

    return {"by_config": df, "by_strategy": by_strategy, "min_budget": mb,
            "shared_features": shared,
            "n_shared": (None if shared is None else len(shared)),
            "ranked_by": by}


if __name__ == "__main__":
    # from pathlib import Path
    # CV_SOURCES  = {2: Path("…/dim2featELA/ela_cv_results.pkl"),
    #                5: Path("…/dim5featELA/ela_cv_results.pkl"),
    #                10: Path("…/dim10featELA/ela_cv_results.pkl")}
    # ICC_SOURCES = {2: Path("…/dim2featELA/ela_icc_results.pkl"),
    #                5: Path("…/dim5featELA/ela_icc_results.pkl"),
    #                10: Path("…/dim10featELA/ela_icc_results.pkl")}
    #
    # # reproducibility ranking (the literal "stability" question):
    # res = rank_stability(CV_SOURCES, ICC_SOURCES, by="overall_cv", cap=1.0)
    #
    # # the binding constraint / recommendation ranking:
    # res = rank_stability(CV_SOURCES, ICC_SOURCES, by="overall_icc", cap=1.0)
    # res["by_strategy"].to_csv("strategy_ranking.csv", index=False)
    pass