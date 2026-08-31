"""
Compute per-instance reproducibility statistics for ELA features across the
30 runs.

UNIT OF ANALYSIS
----------------
One row per (config, function, instance, dimension, feature): the 30 run values
for that cell are summarised into descriptive statistics. Unlike ICC -- which
needs the whole set of instances to form a between-subject term -- these are
computable for a single instance in isolation.

  stats[config_key][(func, inst, dim)][feature_group][feature_name] = {
      "mean", "median", "sd", "var", "min", "q1", "q3", "max", "iqr", "mad",
      "cv", "cv_abs", "range", "n_finite", "n_unique", "all_positive",
      "crosses_zero",
  }

Quartiles/min/max are stored so boxplots can be drawn downstream without
re-reading the raw ELA pickles.

WHY NO NORMALISATION HERE
-------------------------
Normalising sd by the RANGE OF THE SAME 30 VALUES is not informative: for a
fixed n the ratio sd/(max-min) is nearly constant (~0.25 for normal data at
n=30), so it measures distribution shape rather than dispersion, and it cancels
precisely the scale it is meant to normalise. A meaningful scale-free measure
needs a denominator from a DIFFERENT level:

  * spread across instances (within a function)
        -> a bijection with ICC, since ICC = 1 / (1 + (sd_w/sd_b)^2).
           Adds no information, and degenerates for the transformation-invariant
           features where sd_b ~ 0.
  * spread across functions (global feature scale)
        -> stays well-defined for invariant features; this is the one that
           yields something new.

Both are downstream aggregations over this table, so they are deliberately left
out of this script.

CV VALIDITY
-----------
`cv` = sd/mean is stored for every feature because it is cheap, but it is only
INTERPRETABLE for ratio-scaled, strictly positive features whose mean is away
from zero. `all_positive` and `crosses_zero` are stored so the downstream filter
can be applied explicitly rather than assumed. `cv_abs` = sd/|mean| is provided
for features with a consistent negative sign; it is still meaningless for
sign-changing features.

Only `costs_runtime` features are skipped. The dimension-specific omissions in
compute_ela_icc.py exist because those features are degenerate for ICC (no
between-instance variance); they are perfectly well-defined here.

Usage:
  python compute_ela_repro.py /path/to/dim5feat --output-dir /path/to/out --dimension 5
"""

import argparse
import os
import pickle
import re
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_FUNCTIONS = 24
N_INSTANCES = 100
N_RUNS = 30

OMIT_GROUPS = {"levelset"}


def _is_runtime(feat_name):
    return "costs_runtime" in feat_name


ELA_FILES = {
    "cma_random_25": "cma_random_25_ela.pkl",
    "cma_random_50": "cma_random_50_ela.pkl",
    "cma_random_75": "cma_random_75_ela.pkl",
    "cma_random_100": "cma_random_100_ela.pkl",
    "ilhs_25": "ilhs_25_ela.pkl",
    "ilhs_50": "ilhs_50_ela.pkl",
    "ilhs_75": "ilhs_75_ela.pkl",
    "ilhs_100": "ilhs_100_ela.pkl",
    "lhs_25": "lhs_25_ela.pkl",
    "lhs_50": "lhs_50_ela.pkl",
    "lhs_75": "lhs_75_ela.pkl",
    "lhs_100": "lhs_100_ela.pkl",
    "lhs_random_cd_25": "lhs_random_cd_25_ela.pkl",
    "lhs_random_cd_50": "lhs_random_cd_50_ela.pkl",
    "lhs_random_cd_75": "lhs_random_cd_75_ela.pkl",
    "lhs_random_cd_100": "lhs_random_cd_100_ela.pkl",
    "sobol_25": "sobol_25_ela.pkl",
    "sobol_50": "sobol_50_ela.pkl",
    "sobol_75": "sobol_75_ela.pkl",
    "sobol_100": "sobol_100_ela.pkl",
    "uniform_25": "uniform_25_ela.pkl",
    "uniform_50": "uniform_50_ela.pkl",
    "uniform_75": "uniform_75_ela.pkl",
    "uniform_100": "uniform_100_ela.pkl",
}

ELA_FEATURE_GROUPS = {
    "ela_dist": [
        "ela_distr.skewness", "ela_distr.kurtosis",
        "ela_distr.number_of_peaks", "ela_distr.costs_runtime",
    ],
    "meta": [
        "ela_meta.lin_simple.adj_r2", "ela_meta.lin_simple.intercept",
        "ela_meta.lin_simple.coef.min", "ela_meta.lin_simple.coef.max",
        "ela_meta.lin_simple.coef.max_by_min", "ela_meta.lin_w_interact.adj_r2",
        "ela_meta.quad_simple.adj_r2", "ela_meta.quad_simple.cond",
        "ela_meta.quad_w_interact.adj_r2", "ela_meta.costs_runtime",
    ],
    "disp": [
        "disp.ratio_mean_02", "disp.ratio_mean_05", "disp.ratio_mean_10",
        "disp.ratio_mean_25", "disp.ratio_median_02", "disp.ratio_median_05",
        "disp.ratio_median_10", "disp.ratio_median_25", "disp.diff_mean_02",
        "disp.diff_mean_05", "disp.diff_mean_10", "disp.diff_mean_25",
        "disp.diff_median_02", "disp.diff_median_05", "disp.diff_median_10",
        "disp.diff_median_25", "disp.costs_runtime",
    ],
    "nbc": [
        "nbc.nn_nb.sd_ratio", "nbc.nn_nb.mean_ratio", "nbc.nn_nb.cor",
        "nbc.dist_ratio.coeff_var", "nbc.nb_fitness.cor", "nbc.costs_runtime",
    ],
    "ic": [
        "ic.h_max", "ic.eps_s", "ic.eps_max", "ic.eps_ratio",
        "ic.m0", "ic.costs_runtime",
    ],
    "pca": [
        "pca.expl_var.cov_x", "pca.expl_var.cor_x", "pca.expl_var.cov_init",
        "pca.expl_var.cor_init", "pca.expl_var_PC1.cov_x",
        "pca.expl_var_PC1.cor_x", "pca.expl_var_PC1.cov_init",
        "pca.expl_var_PC1.cor_init", "pca.costs_runtime",
    ],
}

_FIELDS = ("mean", "median", "sd", "var", "min", "q1", "q3", "max", "iqr",
           "mad", "cv", "cv_abs", "range", "n_finite", "n_unique",
           "all_positive", "crosses_zero")


def _nan_stats():
    d = {f: np.nan for f in _FIELDS}
    d["n_finite"] = 0
    d["n_unique"] = 0
    d["all_positive"] = False
    d["crosses_zero"] = False
    return d


# ---------------------------------------------------------------------------
# Per-instance statistics
# ---------------------------------------------------------------------------

def run_stats(values, min_finite=2, mad_scale=1.4826):
    """Summarise one instance's run values (length N_RUNS, NaNs tolerated).

    sd/var use ddof=1 (sample statistics -- these 30 runs estimate the
    run-to-run variance, they are not the population). `mad` is the median
    absolute deviation scaled to be a consistent estimator of sigma under
    normality, so it is directly comparable to `sd`.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < min_finite:
        d = _nan_stats()
        d["n_finite"] = int(n)
        if n == 1:
            d["mean"] = d["median"] = d["min"] = d["max"] = float(v[0])
            d["n_unique"] = 1
        return d

    mean = float(np.mean(v))
    median = float(np.median(v))
    sd = float(np.std(v, ddof=1))
    q1, q3 = (float(x) for x in np.percentile(v, [25, 75]))
    vmin, vmax = float(v.min()), float(v.max())
    mad = float(np.median(np.abs(v - median)) * mad_scale)

    # CV: only interpretable for ratio-scaled, strictly positive features.
    cv = sd / mean if mean != 0 else np.nan
    cv_abs = sd / abs(mean) if mean != 0 else np.nan

    return {
        "mean": mean,
        "median": median,
        "sd": sd,
        "var": float(sd ** 2),
        "min": vmin,
        "q1": q1,
        "q3": q3,
        "max": vmax,
        "iqr": float(q3 - q1),
        "mad": mad,
        "cv": float(cv) if np.isfinite(cv) else np.nan,
        "cv_abs": float(cv_abs) if np.isfinite(cv_abs) else np.nan,
        "range": float(vmax - vmin),
        "n_finite": int(n),
        "n_unique": int(np.unique(v).size),
        "all_positive": bool(vmin > 0),
        "crosses_zero": bool(vmin < 0 < vmax),
    }


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_repro(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def project(stats, field="sd"):
    """Project one field into the nested layout the plotting scripts consume:
        out[config_key][(func, inst, dim)][feature_group][feature_name] = float
    """
    out = {}
    for cfg, by_key in stats.items():
        out[cfg] = {}
        for key, by_group in by_key.items():
            out[cfg][key] = {}
            for grp, feats in by_group.items():
                out[cfg][key][grp] = {fn: float(st.get(field, np.nan))
                                      if st.get(field) is not None else np.nan
                                      for fn, st in feats.items()}
    return out


def repro_to_dataframe(stats, dimension=None, features=None):
    """Flatten to a tidy DataFrame: one row per (config, func, inst, feature).

    This is the table to feed boxplots (min/q1/median/q3/max are all present)
    and any downstream normalisation."""
    import pandas as pd
    keep = set(features) if features else None
    rows = []
    for cfg, by_key in stats.items():
        m = re.match(r"^(.*)_(\d+)$", cfg)
        strategy, size = (m.group(1), int(m.group(2))) if m else (cfg, np.nan)
        for (func, inst, dim), by_group in by_key.items():
            if dimension is not None and dim != dimension:
                continue
            for grp, feats in by_group.items():
                for fn, st in feats.items():
                    if keep and fn not in keep:
                        continue
                    rows.append({"strategy": strategy, "size": size,
                                 "func": func, "instance": inst,
                                 "dimension": dim, "feature_group": grp,
                                 "feature": fn, **st})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir=None, dimension=5):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    input_dir = Path(input_dir)
    output_file = Path(output_dir) / "ela_stats.pkl"

    all_stats = {}

    for config_key, filename in ELA_FILES.items():
        filepath = input_dir / filename
        if not filepath.exists():
            print(f"WARNING: {filepath} not found, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {filename} ({config_key})")
        print(f"{'='*60}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        config_stats = {}
        n_cells = 0

        for func_id in range(1, N_FUNCTIONS + 1):
            for inst_id in range(1, N_INSTANCES + 1):
                instance_key = (func_id, inst_id, dimension)
                if instance_key not in data:
                    continue
                instance_data = data[instance_key]

                inst_stats = {}
                for group_name, feature_names in ELA_FEATURE_GROUPS.items():
                    if group_name in OMIT_GROUPS or group_name not in instance_data:
                        continue
                    group_data = instance_data[group_name]

                    group_stats = {}
                    for feat_name in feature_names:
                        if _is_runtime(feat_name):
                            continue
                        vals = []
                        for run_idx in range(N_RUNS):
                            try:
                                vals.append(group_data[run_idx][feat_name])
                            except (IndexError, KeyError):
                                vals.append(np.nan)
                        group_stats[feat_name] = run_stats(vals)
                        n_cells += 1

                    if group_stats:
                        inst_stats[group_name] = group_stats

                if inst_stats:
                    config_stats[instance_key] = inst_stats

        all_stats[config_key] = config_stats

        # ---- summary ----
        cells = [st for by_group in config_stats.values()
                 for grp in by_group.values() for st in grp.values()]
        ok = [s for s in cells if s["n_finite"] >= 2]
        if ok:
            sds = np.array([s["sd"] for s in ok])
            cvs = np.array([s["cv"] for s in ok if np.isfinite(s["cv"])])
            const = float(np.mean([s["sd"] == 0 for s in ok]))
            pos = float(np.mean([s["all_positive"] for s in ok]))
            crs = float(np.mean([s["crosses_zero"] for s in ok]))
            print(f"  Instances: {len(config_stats)}   cells: {n_cells:,}   "
                  f"usable: {len(ok):,}   degenerate: {len(cells) - len(ok):,}")
            print(f"  median sd  : {np.median(sds):.6g}   "
                  f"(scale-dependent -- not comparable across features)")
            if cvs.size:
                print(f"  median CV  : {np.median(cvs):.4f}   "
                      f"[p90 {np.percentile(cvs, 90):.4f}]  "
                      f"(interpretable only where all_positive)")
            print(f"  strictly positive cells : {pos:.1%}   "
                  f"sign-crossing cells: {crs:.1%}")
            print(f"  zero-variance cells     : {const:.1%}  "
                  f"(all 30 runs identical)")

            for group_name in ELA_FEATURE_GROUPS:
                grp = [st for by_group in config_stats.values()
                       if group_name in by_group
                       for st in by_group[group_name].values()
                       if st["n_finite"] >= 2]
                if grp:
                    gsd = np.median([s["sd"] for s in grp])
                    gcv = [s["cv"] for s in grp if np.isfinite(s["cv"])]
                    gcvs = f"{np.median(gcv):.4f}" if gcv else "n/a"
                    print(f"    {group_name:>10s}: median sd={gsd:.6g}, "
                          f"median CV={gcvs}")

        del data
        print("  Done.")

    with open(output_file, "wb") as f:
        pickle.dump(all_stats, f)

    print(f"\n\nResults saved to: {output_file}")
    print("\nAccess format:")
    print("  stats[config_key][(func, inst, dim)][feature_group][feature_name] -> dict")
    print(f"  fields: {', '.join(_FIELDS)}")
    print("\n  Tidy DataFrame (for boxplots / normalisation):")
    print("    from compute_ela_repro import load_repro, repro_to_dataframe")
    print("    df = repro_to_dataframe(load_repro(path))")
    print("\n  Nested floats (for the existing plotting scripts):")
    print("    from compute_ela_repro import project")
    print("    ela_sd = project(load_repro(path), 'sd')   # or 'cv', 'iqr', ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-instance reproducibility statistics for ELA features "
                    "across the 30 runs (one row per config/function/instance/"
                    "feature)."
    )
    parser.add_argument("input_dir", type=str,
                        help="Directory containing the ELA pkl files.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for the output pkl. Defaults to input_dir.")
    parser.add_argument("--dimension", type=int, default=5,
                        help="Problem dimensionality (default: 5).")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir,
         dimension=args.dimension)