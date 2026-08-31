"""
check_nonfinite.py

Decide, per feature, whether to EXCLUDE it or IMPUTE it — using evidence rather
than a rule of thumb.

Three questions, in order of how much they constrain the decision:

  1. INF or NAN?  Inf is a hard constraint: StandardScaler raises ValueError on
     infinity, so an inf-producing feature must be excluded (or clipped /
     log-transformed) whatever its rate. NaN is imputable at any rate.

  2. IS THE PATTERN UNIFORM OR CONCENTRATED?  A feature failing 4% of the time
     uniformly is safe to impute -- the imputed values are scattered. A feature
     failing 4% of the time CONCENTRATED ON ONE STRATEGY is not: mean-imputing
     exactly the cells where a sampler misbehaves erases the very differential
     the study is trying to measure. Concentration matters more than rate.

  3. HOW BAD IS THE RATE?  Only once 1 and 2 are answered.

Also reports whether the failures are total (all 30 runs of an instance) or
partial. Partial failures are the dangerous ones for any statistic computed
over runs, because the surviving runs are a NON-RANDOM subsample -- the runs
that fail are typically the extreme ones.

Usage:
  python check_nonfinite.py /path/to/dim5featELA --dimension 5
  python check_nonfinite.py /path/to/dim5featELA --dimension 2 --configs sobol_25 sobol_100
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

N_FUNCTIONS = 24
N_INSTANCES = 100
N_RUNS = 30
OMIT_GROUPS = {"levelset"}

STRATEGIES = ["cma_random", "ilhs", "lhs", "lhs_random_cd", "sobol", "uniform"]
SIZES = [25, 50, 75, 100]


def _is_runtime(name):
    return "costs_runtime" in name


def scan_config(path, dimension):
    """Return a long DataFrame: one row per (feature, func, instance) cell."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    rows = []
    for func in range(1, N_FUNCTIONS + 1):
        for inst in range(1, N_INSTANCES + 1):
            key = (func, inst, dimension)
            if key not in data:
                continue
            for grp, runs in data[key].items():
                if grp in OMIT_GROUPS:
                    continue
                for feat in runs[0].keys():
                    if _is_runtime(feat):
                        continue
                    vals = np.array([runs[r].get(feat, np.nan)
                                     for r in range(N_RUNS)], dtype=float)
                    n_nan = int(np.isnan(vals).sum())
                    n_inf = int(np.isinf(vals).sum())
                    if n_nan or n_inf:
                        rows.append({
                            "feature": feat, "group": grp, "func": func,
                            "instance": inst, "n_nan": n_nan, "n_inf": n_inf,
                            "n_bad": n_nan + n_inf,
                            "total": (n_nan + n_inf) == N_RUNS,
                        })
    del data
    return pd.DataFrame(rows)


def scan(input_dir, dimension, configs=None, verbose=True):
    """Scan every config and summarise per feature.

    Returns (summary, by_strategy, raw):
      summary    : per-feature verdict — kind, rate, concentration, partial share
      by_strategy: per (feature, strategy) failure rate — the concentration check
      raw        : the underlying per-cell rows
    """
    input_dir = Path(input_dir)
    configs = configs or [f"{s}_{z}" for s in STRATEGIES for z in SIZES]

    frames, totals = [], {}
    for cfg in configs:
        p = input_dir / f"{cfg}_ela.pkl"
        if not p.exists():
            if verbose:
                print(f"  (missing {p.name}, skipped)")
            continue
        strategy, size = cfg.rsplit("_", 1)
        df = scan_config(p, dimension)
        # cells scanned in this config = funcs x instances, per feature
        totals[cfg] = N_FUNCTIONS * N_INSTANCES
        if len(df):
            df["strategy"] = strategy
            df["size"] = int(size)
            df["config"] = cfg
            frames.append(df)
        if verbose:
            print(f"  {cfg:<22s} {len(df):>6,d} affected (feature, func, instance) cells")

    if not frames:
        print("\nNo non-finite values found anywhere. "
              "No feature needs excluding or imputing.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    n_cfg = len(totals)
    cells_per_feature = n_cfg * N_FUNCTIONS * N_INSTANCES

    g = (raw.groupby("feature")
         .agg(group=("group", "first"),
              cells_affected=("n_bad", "size"),
              n_inf_values=("n_inf", "sum"),
              n_nan_values=("n_nan", "sum"),
              total_failures=("total", "sum"),
              mean_runs_lost=("n_bad", "mean"))
         .reset_index())
    g["pct_cells"] = 100 * g["cells_affected"] / cells_per_feature
    g["kind"] = np.where(g["n_inf_values"] > 0,
                         np.where(g["n_nan_values"] > 0, "inf+nan", "inf"),
                         "nan")
    g["pct_partial"] = 100 * (1 - g["total_failures"] / g["cells_affected"])

    # concentration: is the failure rate strategy-specific?
    by_strat = (raw.groupby(["feature", "strategy"])["n_bad"].size()
                .rename("cells").reset_index())
    per_cfg_cells = N_FUNCTIONS * N_INSTANCES * len(SIZES)
    by_strat["pct"] = 100 * by_strat["cells"] / per_cfg_cells
    conc = (by_strat.pivot_table(index="feature", columns="strategy",
                                 values="pct", fill_value=0.0))
    # ratio of worst to mean strategy: 1.0 = uniform, >>1 = concentrated
    g = g.merge(
        pd.DataFrame({
            "feature": conc.index,
            "worst_strategy": conc.idxmax(axis=1).values,
            "concentration": (conc.max(axis=1) /
                              conc.mean(axis=1).replace(0, np.nan)).values,
        }), on="feature", how="left")

    g["verdict"] = [_verdict(r) for _, r in g.iterrows()]
    g = g.sort_values("pct_cells", ascending=False).reset_index(drop=True)

    if verbose:
        _report(g, conc, dimension, cells_per_feature)
    return g, conc, raw


def _verdict(r, high=50.0, conc_thresh=2.0):
    """Turn the evidence into a recommendation (see module docstring)."""
    if r["kind"] in ("inf", "inf+nan"):
        return "EXCLUDE (inf crashes StandardScaler)"
    if r["pct_cells"] >= high:
        return "EXCLUDE (majority of cells missing)"
    if np.isfinite(r["concentration"]) and r["concentration"] >= conc_thresh:
        return (f"EXCLUDE or flag (concentrated on {r['worst_strategy']}; "
                f"imputing would mask a sampler difference)")
    if r["pct_partial"] > 50:
        return "IMPUTE with care (mostly PARTIAL: survivors are a biased subsample)"
    return "IMPUTE (low rate, spread evenly)"


def _report(g, conc, dimension, cells_per_feature):
    print(f"\n{'='*78}\nNON-FINITE FEATURE REPORT — dimension {dimension}")
    print(f"{cells_per_feature:,} (config, function, instance) cells per feature")
    print(f"{'='*78}")
    if g.empty:
        print("nothing to report")
        return
    for _, r in g.iterrows():
        print(f"\n{r['feature']}   [{r['group']}]")
        print(f"  kind            : {r['kind']}"
              f"   ({int(r['n_inf_values']):,} inf, {int(r['n_nan_values']):,} nan values)")
        print(f"  cells affected  : {r['pct_cells']:.2f}%  "
              f"({int(r['cells_affected']):,} cells)")
        print(f"  runs lost/cell  : {r['mean_runs_lost']:.1f} of {N_RUNS}"
              f"   ({r['pct_partial']:.0f}% of affected cells are PARTIAL)")
        if np.isfinite(r["concentration"]):
            print(f"  concentration   : {r['concentration']:.2f}x "
                  f"(worst: {r['worst_strategy']})"
                  + ("   <- strategy-specific!" if r["concentration"] >= 2 else ""))
        print(f"  -> {r['verdict']}")

    print(f"\n{'='*78}\nPER-STRATEGY FAILURE RATE (% of cells)\n{'='*78}")
    with pd.option_context("display.float_format", lambda v: f"{v:6.2f}"):
        print(conc.loc[g["feature"]].to_string())

    print(f"\n{'='*78}\nSUGGESTED OMIT LIST FOR dim {dimension}\n{'='*78}")
    drop = g.loc[g["verdict"].str.startswith("EXCLUDE"), "feature"].tolist()
    print(f"    {dimension}: {set(drop)!r},")
    keep = g.loc[~g["verdict"].str.startswith("EXCLUDE"), "feature"].tolist()
    if keep:
        print(f"\n  imputable instead (currently may be over-excluded): {keep}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Classify non-finite ELA feature values by kind and pattern.")
    p.add_argument("input_dir", type=str)
    p.add_argument("--dimension", type=int, default=5)
    p.add_argument("--configs", nargs="+", default=None,
                   help="e.g. sobol_25 sobol_100 (default: all 24)")
    p.add_argument("--csv", type=str, default=None,
                   help="optional path to write the per-feature summary")
    a = p.parse_args()
    summary, conc, raw = scan(a.input_dir, a.dimension, a.configs)
    if a.csv and len(summary):
        summary.to_csv(a.csv, index=False)
        print(f"\nsummary written to {a.csv}")


# --------------------------------------------------------------------------- #
# Where do the failures sit? Function / size / strategy breakdown              #
# --------------------------------------------------------------------------- #

BBOB_GROUP = {**{f: "separable" for f in range(1, 6)},
              **{f: "low/mod cond" for f in range(6, 10)},
              **{f: "high cond unimodal" for f in range(10, 15)},
              **{f: "multimodal, structure" for f in range(15, 20)},
              **{f: "multimodal, weak" for f in range(20, 25)}}


def failure_breakdown(raw, by="func", dimension=None, top=8, verbose=True):
    """Where the non-finite values sit, per feature.

    A rate quoted over the whole grid can hide a very specific cause. If a
    feature fails on only one FUNCTION, the mechanism is a property of that
    landscape rather than of sampling in general -- for instance, fitting a
    quadratic model to a purely linear function leaves the quadratic terms
    unidentified, so the design matrix is ill-conditioned however the points
    were drawn. Likewise a failure confined to one SIZE is a resolution limit,
    and one confined to one STRATEGY is a sampler artefact.

    `by` may be "func", "size" or "strategy". Returns a per-feature summary with
    the number of distinct levels affected and the share carried by the worst,
    plus the full contingency table.
    """
    d = raw if dimension is None else raw[raw["dimension"] == dimension] \
        if "dimension" in raw.columns else raw
    if d.empty:
        print("no non-finite cells to break down")
        return pd.DataFrame(), pd.DataFrame()

    tab = (d.groupby(["feature", by]).size().rename("cells")
           .reset_index()
           .pivot(index="feature", columns=by, values="cells")
           .fillna(0).astype(int))

    rows = []
    for feat, r in tab.iterrows():
        nz = r[r > 0]
        rows.append({
            "feature": feat,
            "total_cells": int(r.sum()),
            f"n_{by}_affected": int((r > 0).sum()),
            f"n_{by}_possible": int(len(r)),
            "worst": nz.idxmax() if len(nz) else None,
            "worst_share": float(nz.max() / r.sum()) if r.sum() else np.nan,
        })
    summ = (pd.DataFrame(rows)
            .sort_values("total_cells", ascending=False)
            .reset_index(drop=True))

    if verbose:
        print(f"\n{'='*74}\nNON-FINITE CELLS BY {by.upper()}\n{'='*74}")
        for _, r in summ.iterrows():
            n_aff, n_pos = r[f"n_{by}_affected"], r[f"n_{by}_possible"]
            print(f"\n{r['feature']}")
            print(f"  {int(r['total_cells']):,} affected cells across "
                  f"{n_aff} of {n_pos} {by} values")
            worst = r["worst"]
            lbl = (f"f{worst} ({BBOB_GROUP.get(worst, '?')})"
                   if by == "func" else str(worst))
            print(f"  worst: {lbl} carries {r['worst_share']:.1%} of them")
            if n_aff <= 3:
                hit = tab.loc[r["feature"]]
                hit = hit[hit > 0]
                detail = ", ".join(
                    (f"f{k}" if by == "func" else str(k)) + f" ({int(v):,})"
                    for k, v in hit.items())
                print(f"  -> CONFINED to {detail}")
                if by == "func":
                    print(f"     a failure this localised is a property of the "
                          f"landscape, not of sampling")
            elif r["worst_share"] > 0.5:
                print(f"  -> dominated by a single {by}")
        print(f"\n{'='*74}\nCONTINGENCY TABLE (cells)\n{'='*74}")
        show = tab.copy()
        if by == "func":
            show.columns = [f"f{c}" for c in show.columns]
        print(show.loc[summ["feature"]].to_string())
    return summ, tab