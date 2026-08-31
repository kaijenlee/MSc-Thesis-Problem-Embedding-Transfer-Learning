"""
icc_degeneracy_diagnostic.py

Find and explain degenerate ICC cells before trusting the ICC results.

WHY
---
ICC(1,1) = (MSB - MSW) / (MSB + (k-1)*MSW). Three values are suspect:

  ICC == 1.000  requires MSW == 0, i.e. ZERO run-to-run variance across all 30
                runs for every instance. With random sampling that should not
                happen -- it usually means the feature is constant (e.g. an
                integer feature stuck at one value, or a degenerate fit). These
                cells inflate the per-function "ceiling" estimates.
  ICC == 0.000  is a CENSORED FLOOR, not a measurement: compute_ela_icc.py
                clips negative ICCs (run noise >= between-instance signal) to 0.
                A cell at 0.0 and a cell at 0.004 are not comparable.
  NaN           degenerate by construction: <2 usable instances, <2 finite runs,
                or no between-subject variance. compute_ela_icc.py already
                prints these per config; this collects them into a DataFrame.

STAGE 1 (fast)  scan_icc_degeneracy() -- reads only the ICC pickles, flags every
                cell, and breaks the counts down by feature / function / group /
                (strategy, size) so you can see whether degeneracy is
                concentrated or diffuse.

STAGE 2 (slow)  inspect_raw_cell() -- rebuilds the (100 instances x 30 runs)
                matrix from the ORIGINAL ELA pickle for one suspicious cell and
                reports MSB, MSW, the recovered variance components, and how
                many instances have literally constant runs. This is what turns
                "ICC is exactly 1" into "because 100/100 instances had zero
                run-to-run variance".

Usage
-----
    res = scan_icc_degeneracy(ICC_SOURCES)          # stage 1, prints summary
    res["by_feature"]                                # DataFrames for the notebook
    res["cells"].query("is_one")                     # the offending cells

    # stage 2, on one cell picked from the scan:
    inspect_raw_cell("/path/to/dim5featELA", "sobol_100", func=1, dimension=5,
                     feature="ela_distr.number_of_peaks")
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from plot_icc_boxplots import split_config_key, _FUNC_TO_GROUP, _bbob_label


N_INSTANCES = 100
N_RUNS = 30


def _is_runtime(name):
    return "costs_runtime" in name


def _load(obj, dimension=None):
    if isinstance(obj, dict) and dimension is not None and dimension in obj:
        obj = obj[dimension]
    if isinstance(obj, dict):
        return obj
    with open(obj, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Stage 1: scan the ICC pickles                                                #
# --------------------------------------------------------------------------- #

def scan_icc_degeneracy(icc_sources, dimensions=None, eps=1e-9,
                        near_one=0.999, verbose=True):
    """Flag every ICC cell and summarise where degeneracy concentrates.

    Flags per cell:
      is_nan    : ICC could not be computed (degenerate design)
      is_one    : ICC == 1.0 exactly (MSW == 0)   -> suspect
      is_near1  : ICC >= near_one but not exactly 1 -> worth a look
      is_zero   : ICC == 0.0 exactly (clipped floor, or true zero)

    Returns dict of DataFrames: cells, by_feature, by_function, by_config,
    by_feature_group.
    """
    dims = sorted(icc_sources) if dimensions is None else dimensions
    rows = []
    for d in dims:
        ela_icc = _load(icc_sources, d)
        for cfg, by_func in ela_icc.items():
            strategy, size = split_config_key(cfg)
            for key, by_group in by_func.items():
                func, dim = key
                if dim != d:
                    continue
                for grp, feats in by_group.items():
                    for fn, v in feats.items():
                        if _is_runtime(fn):
                            continue
                        val = float(v) if v is not None else np.nan
                        isnan = not np.isfinite(val)
                        rows.append({
                            "dimension": d, "strategy": strategy, "size": size,
                            "func": func,
                            "fgroup": _bbob_label(_FUNC_TO_GROUP.get(func, "")),
                            "feature_group": grp, "feature": fn, "icc": val,
                            "is_nan": isnan,
                            "is_one": (not isnan) and abs(val - 1.0) <= eps,
                            "is_near1": (not isnan) and (near_one <= val < 1.0 - eps),
                            "is_zero": (not isnan) and abs(val) <= eps,
                        })
    cells = pd.DataFrame(rows)
    if cells.empty:
        raise ValueError("No ICC cells found -- check paths/dimensions.")

    flags = ["is_nan", "is_one", "is_near1", "is_zero"]

    def _summ(keys):
        g = (cells.groupby(keys)
             .agg(n_cells=("icc", "size"),
                  **{f: (f, "sum") for f in flags},
                  median_icc=("icc", "median"))
             .reset_index())
        for f in flags:
            g[f.replace("is_", "frac_")] = g[f] / g["n_cells"]
        return g

    by_feature = _summ(["feature"]).sort_values(
        ["frac_one", "frac_nan"], ascending=False).reset_index(drop=True)
    by_function = _summ(["dimension", "func", "fgroup"]).sort_values(
        ["dimension", "frac_one"], ascending=[True, False]).reset_index(drop=True)
    by_config = _summ(["dimension", "strategy", "size"]).sort_values(
        ["dimension", "strategy", "size"]).reset_index(drop=True)
    by_feature_group = _summ(["feature_group"]).sort_values(
        "frac_one", ascending=False).reset_index(drop=True)

    if verbose:
        n = len(cells)
        print(f"scanned {n:,} ICC cells | {cells['feature'].nunique()} features | "
              f"dims {sorted(cells['dimension'].unique())}")
        for f, label in [("is_one", "ICC == 1.000 (MSW == 0, suspect)"),
                         ("is_near1", f"ICC in [{near_one}, 1) (near-degenerate)"),
                         ("is_zero", "ICC == 0.000 (clipped floor)"),
                         ("is_nan", "NaN (degenerate design)")]:
            c = int(cells[f].sum())
            print(f"  {label:<44s} {c:>7,d}  ({c / n:.2%})")

        top = by_feature[by_feature["is_one"] > 0].head(10)
        if len(top):
            print("\n  features most often ICC == 1.000:")
            for _, r in top.iterrows():
                print(f"    {r['feature']:<34s} {r['frac_one']:6.1%} "
                      f"({int(r['is_one'])}/{int(r['n_cells'])} cells)")
        else:
            print("\n  no cells with ICC == 1.000 -- ceilings are not inflated "
                  "by zero-variance features.")

        topz = by_feature[by_feature["is_zero"] > 0].head(10)
        if len(topz):
            print("\n  features most often on the clipped floor (ICC == 0):")
            for _, r in topz.sort_values("frac_zero", ascending=False).head(10).iterrows():
                print(f"    {r['feature']:<34s} {r['frac_zero']:6.1%} "
                      f"({int(r['is_zero'])}/{int(r['n_cells'])} cells)")
    return {"cells": cells, "by_feature": by_feature, "by_function": by_function,
            "by_config": by_config, "by_feature_group": by_feature_group}


def suspect_cells(scan, which="is_one", top=20):
    """Pull the individual offending cells so you can feed one to stage 2."""
    c = scan["cells"]
    return (c[c[which]]
            .loc[:, ["dimension", "strategy", "size", "func", "fgroup",
                     "feature_group", "feature", "icc"]]
            .head(top).reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Stage 2: rebuild the raw matrix and explain the degeneracy                    #
# --------------------------------------------------------------------------- #

def _build_matrix(data, func, dimension, group, feature,
                  n_instances=N_INSTANCES, n_runs=N_RUNS):
    """Mirror compute_ela_icc.py's matrix construction: rows = instances,
    cols = runs."""
    matrix = []
    for inst in range(1, n_instances + 1):
        key = (func, inst, dimension)
        if key not in data:
            continue
        inst_data = data[key]
        if group not in inst_data:
            continue
        gd = inst_data[group]
        row = []
        for r in range(n_runs):
            try:
                row.append(gd[r][feature])
            except (IndexError, KeyError):
                row.append(np.nan)
        matrix.append(row)
    return np.asarray(matrix, dtype=float)


def _find_group(ela_dir, config_key, func, dimension, feature, data):
    """Locate which feature_group holds `feature` (so callers need not pass it)."""
    for inst in range(1, N_INSTANCES + 1):
        key = (func, inst, dimension)
        if key in data:
            for grp, gd in data[key].items():
                try:
                    if feature in gd[0]:
                        return grp
                except (IndexError, KeyError, TypeError):
                    continue
            break
    return None


def inspect_raw_cell(ela_dir, config_key, func, dimension, feature,
                     group=None, verbose=True):
    """Rebuild one (instances x runs) matrix and report the variance components.

    Confirms WHY a cell degenerated: reports MSB, MSW, the recovered
    sigma^2_between = (MSB - MSW)/k and sigma^2_within = MSW, how many instances
    have literally constant runs, and how many distinct values the feature takes
    overall (a tiny count means a discrete/degenerate feature).
    """
    path = Path(ela_dir) / f"{config_key}_ela.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with open(path, "rb") as f:
        data = pickle.load(f)

    grp = group or _find_group(ela_dir, config_key, func, dimension, feature, data)
    if grp is None:
        raise ValueError(f"Could not locate feature {feature!r} in {path.name}")

    M = _build_matrix(data, func, dimension, grp, feature)
    del data
    if M.size == 0:
        raise ValueError("Empty matrix -- no instances found for that (func, dim).")

    finite_per_row = np.isfinite(M).sum(axis=1)
    M = M[finite_per_row >= 2]
    if M.shape[0] < 2:
        raise ValueError("Fewer than 2 usable instances -- degenerate by design.")
    k = int(np.min(np.isfinite(M).sum(axis=1)))
    M = np.asarray([r[np.isfinite(r)][:k] for r in M], dtype=float)
    n = M.shape[0]

    grand = M.mean()
    row_means = M.mean(axis=1)
    ss_b = k * np.sum((row_means - grand) ** 2)
    ss_w = np.sum((M - row_means[:, None]) ** 2)
    msb = ss_b / (n - 1)
    msw = ss_w / (n * (k - 1))
    denom = msb + (k - 1) * msw
    icc = (msb - msw) / denom if denom > 0 else np.nan

    row_ranges = M.max(axis=1) - M.min(axis=1)
    n_const_rows = int(np.sum(row_ranges == 0))
    n_unique = int(np.unique(M).size)

    out = {
        "feature": feature, "feature_group": grp, "config": config_key,
        "func": func, "dimension": dimension,
        "n_instances": n, "n_runs": k,
        "MSB": float(msb), "MSW": float(msw),
        "sigma2_between": float((msb - msw) / k),
        "sigma2_within": float(msw),
        "icc_recomputed": float(icc),
        "n_constant_rows": n_const_rows,
        "frac_constant_rows": n_const_rows / n,
        "n_unique_values": n_unique,
        "value_min": float(M.min()), "value_max": float(M.max()),
    }

    if verbose:
        print(f"{config_key} | f{func} dim{dimension} | {grp}.{feature}")
        print(f"  matrix {n} instances x {k} runs")
        print(f"  MSB={msb:.6g}  MSW={msw:.6g}  -> ICC={icc:.6f}")
        print(f"  sigma^2_between={out['sigma2_between']:.6g}  "
              f"sigma^2_within={out['sigma2_within']:.6g}")
        print(f"  instances with CONSTANT runs: {n_const_rows}/{n} "
              f"({out['frac_constant_rows']:.1%})")
        print(f"  distinct values overall: {n_unique} "
              f"(range {out['value_min']:.6g} .. {out['value_max']:.6g})")
        if msw == 0:
            print("  => MSW == 0: zero run-to-run variance. ICC is 1 by "
                  "construction; the feature is constant within every instance.")
        elif n_unique <= 5:
            print("  => very few distinct values: discrete/degenerate feature; "
                  "ICC is unstable here.")
        elif out["sigma2_between"] <= 0:
            print("  => sigma^2_between <= 0: run noise exceeds between-instance "
                  "signal (ICC clipped to 0 at source).")
    return out


def inspect_many(ela_dirs, suspects, verbose=False):
    """Run inspect_raw_cell over a DataFrame of suspects (from suspect_cells).

    ela_dirs: {dimension: path_to_ELA_pkl_directory}
    Returns a DataFrame, one row per inspected cell."""
    out = []
    for _, r in suspects.iterrows():
        d = int(r["dimension"])
        cfg = f"{r['strategy']}_{int(r['size'])}"
        try:
            out.append(inspect_raw_cell(ela_dirs[d], cfg, int(r["func"]), d,
                                        r["feature"], group=r.get("feature_group"),
                                        verbose=verbose))
        except Exception as e:                      # keep going on missing files
            out.append({"feature": r["feature"], "config": cfg,
                        "func": int(r["func"]), "dimension": d, "error": str(e)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    # ICC_SOURCES = {2: ".../dim2featELA/ela_icc_results.pkl",
    #                5: ".../dim5featELA/ela_icc_results.pkl",
    #                10: ".../dim10featELA/ela_icc_results.pkl"}
    # ELA_DIRS    = {2: ".../dim2featELA", 5: ".../dim5featELA",
    #                10: ".../dim10featELA"}
    #
    # scan = scan_icc_degeneracy(ICC_SOURCES)      # stage 1
    # sus  = suspect_cells(scan, "is_one", top=10)
    # inspect_many(ELA_DIRS, sus)                  # stage 2, confirms the cause
    pass