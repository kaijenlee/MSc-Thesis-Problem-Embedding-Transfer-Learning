"""
Compute per-feature within-config reliability (ICC) for ELA features across
30 runs, REPORTING THE VARIANCE COMPONENTS as well as the ICC itself.

A single ICC is a *between-instances* quantity: it contrasts the variance
between a function's instances (signal) with the run-to-run variance within
each instance (sampling noise). The natural unit is one ICC per
(config, dimension, function, feature), with that function's 100 instances as
the subjects and the 30 runs as the replicate measurements.

Model: one-way random-effects ICC(1,1) (Shrout & Fleiss). The runs are
independent draws with no identity across instances, so this is the correct
form (not ICC(2,1)/ICC(3,1)).

    MSB = between-instance mean square      (df = n - 1)
    MSW = within-instance mean square       (df = n * (k_est - 1))

    sigma^2_within  = MSW
    sigma^2_between = (MSB - MSW) / k_est

    ICC(1,1) = (MSB - MSW) / (MSB + (k_est - 1) * MSW)
             = sigma^2_between / (sigma^2_between + sigma^2_within)

WHY THE COMPONENTS MATTER
-------------------------
The ICC alone cannot distinguish the two causes of a low value:
  * sigma^2_within large   -> noisy measurement (budget-limited)
  * sigma^2_between ~ 0    -> the instances genuinely do not differ, so there
                              is nothing to detect (structural; no budget helps)
Reporting MSB/MSW makes that distinction directly readable, and also lets a
sampler's advantage be attributed to reduced noise (sigma^2_within down) rather
than inflated signal (sigma^2_between up).

CAUTION: MSW is SCALE-DEPENDENT, so it is not comparable across features whose
units differ. `grand_mean` and `sd_within` are stored so a relative noise level
can be formed where the feature is ratio-scaled.

TWO DIFFERENT k's -- DO NOT CONFLATE
------------------------------------
  k_est     = runs per instance used to ESTIMATE the components (N_RUNS = 30).
  k_average = runs AVERAGED to form one training vector downstream
              (K_AVERAGE = [1, 2, 3, 5], matching the classifier sweep).

ICC(1,k) is the reliability of a k-run AVERAGE, obtained from the components:

    ICC(1,k) = sigma^2_between / (sigma^2_between + sigma^2_within / k)

(equivalently the Spearman-Brown transform of ICC(1,1)). Averaging shrinks only
the noise term, so:

    noise_frac      = 1 - ICC(1,1)                  # ceiling on removable noise
    removable_at_k  = (1 - 1/k) * noise_frac        # actually removed at finite k

Low ICC(1,1) -> averaging pays. High ICC(1,1) -> spend the budget on more
instances instead.

CLIPPING: OFF BY DEFAULT
------------------------
Negative ICCs are KEPT. Under sigma^2_between = 0 the estimator is roughly
symmetric about zero, so sampling error puts about half the no-signal cells
slightly positive and half slightly negative. Clipping collapses only the
negative half onto 0, which HALVES the apparent no-signal rate and biases every
aggregate upward. Use `sigma2_between <= 0` as the no-signal indicator, and clip
only at display time (plots already pin vmin=0 / ylim=(0,1)).
Pass --clip-negative to restore the old behaviour.

OUTPUT (one pickle)
-------------------
  ela_icc_results.pkl:
      comp[config_key][(func, dim)][feature_group][feature_name] = {
          "icc", "icc_raw", "msb", "msw", "sigma2_between", "sigma2_within",
          "sd_between", "sd_within", "grand_mean", "n_instances", "k_est",
          "noise_frac", "icc_k": {k: ...}, "removable_at_k": {k: ...},
          "n_constant_rows", "n_unique_values",
      }

Downstream code that expects the old bare-float layout needs one line:

      from compute_ela_icc import load_icc, project
      ela_icc = project(load_icc(path))              # -> [cfg][(f,d)][grp][feat] = float
      ela_icc = project(load_icc(path), "msw")       # or any other field
      ela_icc = project(load_icc(path), "icc_k", k=5)

Usage:
  python compute_ela_icc.py /path/to/dim2feat --output-dir /path/to/out --dimension 2
  python compute_ela_icc.py ... --k-average 1 2 3 5 --clip-negative
"""

import argparse
import pickle
import numpy as np
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_FUNCTIONS = 24
N_INSTANCES = 100
N_RUNS = 30                      # k_est: replicates used to estimate components

# k_average: runs averaged into one downstream training vector.
# Mirrors DEFAULT_N_RUNS_TRAIN in the classification setup.
K_AVERAGE = [1, 2, 3, 5]


def get_omit_features(dimension):
    if dimension == 2:
        return {
            'disp.diff_median_02', 'disp.ratio_median_02',
            'disp.ratio_mean_02', 'ela_meta.quad_simple.cond',
            'disp.diff_mean_02', 'ic.eps_ratio'
        }
    elif dimension in (5, 10):
        return {'ela_meta.quad_simple.cond'}
    return set()


OMIT_GROUPS = {"levelset"}


# costs_runtime features are not reproducibility-relevant; skip them everywhere.
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
    "lhs_random_cd_25": "lhs_random_cd_25_ela.pkl", "lhs_random_cd_50": "lhs_random_cd_50_ela.pkl",
    "lhs_random_cd_75": "lhs_random_cd_75_ela.pkl", "lhs_random_cd_100": "lhs_random_cd_100_ela.pkl",
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


# ---------------------------------------------------------------------------
# ICC computation
# ---------------------------------------------------------------------------

def _nan_result(k_average):
    """Uniform shape for degenerate cells, so consumers never KeyError."""
    return {
        "icc": np.nan, "icc_raw": np.nan, "msb": np.nan, "msw": np.nan,
        "sigma2_between": np.nan, "sigma2_within": np.nan,
        "sd_between": np.nan, "sd_within": np.nan, "grand_mean": np.nan,
        "n_instances": 0, "k_est": 0, "noise_frac": np.nan,
        "icc_k": {k: np.nan for k in k_average},
        "removable_at_k": {k: np.nan for k in k_average},
        "n_constant_rows": 0, "n_unique_values": 0,
    }


def compute_icc_components(value_matrix, clip_negative=False, k_average=K_AVERAGE):
    """
    One-way random-effects ICC(1,1) + variance components for a
    (subjects x runs) matrix.

    Rows = subjects (a function's instances), columns = the runs. Subjects with
    fewer than 2 finite runs are dropped; a balanced design is assumed and each
    retained subject is trimmed to the common run count k_est.

    Returns a dict (see module docstring). All-NaN on a degenerate matrix
    (<2 usable subjects, <2 usable runs, or no positive denominator).

    `icc` is clipped to 0 when clip_negative (matching the legacy behaviour);
    `icc_raw` always keeps the signed value, so the censoring can be quantified.
    """
    nanres = _nan_result(k_average)
    M = np.asarray(value_matrix, dtype=float)
    if M.ndim != 2:
        return nanres

    finite_per_row = np.isfinite(M).sum(axis=1)
    M = M[finite_per_row >= 2]
    if M.shape[0] < 2:
        return nanres

    k_est = int(np.min(np.isfinite(M).sum(axis=1)))
    if k_est < 2:
        return nanres
    M = np.asarray([r[np.isfinite(r)][:k_est] for r in M], dtype=float)
    n = M.shape[0]

    grand = M.mean()
    row_means = M.mean(axis=1)
    ss_between = k_est * np.sum((row_means - grand) ** 2)
    ss_within = np.sum((M - row_means[:, None]) ** 2)
    df_between = n - 1
    df_within = n * (k_est - 1)
    if df_between <= 0 or df_within <= 0:
        return nanres

    msb = ss_between / df_between
    msw = ss_within / df_within
    denom = msb + (k_est - 1) * msw
    if denom <= 0:
        return nanres

    icc_raw = (msb - msw) / denom
    icc = 0.0 if (clip_negative and icc_raw < 0) else icc_raw

    # Variance components (sigma2_between may be negative -> no detectable signal)
    s2_w = msw
    s2_b = (msb - msw) / k_est

    # ICC(1,k): reliability of a k-run average. Averaging shrinks ONLY the noise.
    icc_k, removable = {}, {}
    for k in k_average:
        d = s2_b + s2_w / k
        v = (s2_b / d) if d > 0 else np.nan
        if clip_negative and np.isfinite(v) and v < 0:
            v = 0.0
        icc_k[k] = float(v) if np.isfinite(v) else np.nan
        # fraction of single-run within-class noise removed by averaging k runs
        removable[k] = float((1.0 - 1.0 / k) * (1.0 - icc)) if np.isfinite(icc) else np.nan

    row_ranges = M.max(axis=1) - M.min(axis=1)

    return {
        "icc": float(icc),
        "icc_raw": float(icc_raw),
        "msb": float(msb),
        "msw": float(msw),
        "sigma2_between": float(s2_b),
        "sigma2_within": float(s2_w),
        "sd_between": float(np.sqrt(s2_b)) if s2_b > 0 else 0.0,
        "sd_within": float(np.sqrt(s2_w)),
        "grand_mean": float(grand),
        "n_instances": int(n),
        "k_est": int(k_est),
        # 1 - ICC(1,1): ceiling on the share of noise that averaging can remove
        "noise_frac": float(1.0 - icc),
        "icc_k": icc_k,
        "removable_at_k": removable,
        "n_constant_rows": int(np.sum(row_ranges == 0)),
        "n_unique_values": int(np.unique(M).size),
    }


def reliability_band(icc):
    """Koo & Li interpretation band for an ICC value."""
    if icc is None or not np.isfinite(icc):
        return "n/a"
    if icc >= 0.90:
        return "excellent"
    if icc >= 0.75:
        return "good"
    if icc >= 0.50:
        return "moderate"
    return "poor"


# ---------------------------------------------------------------------------
# Convenience: flatten the components pickle into a tidy DataFrame
# ---------------------------------------------------------------------------

def load_icc(path):
    """Load the components pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def project(components, field="icc", k=None, clip=False):
    """Project one field out of the components into the nested float layout

        out[config_key][(func, dim)][feature_group][feature_name] = float

    which every plotting/analysis script in this project consumes. `field` may
    be any scalar key ("icc", "icc_raw", "msw", "msb", "sigma2_between",
    "sigma2_within", "noise_frac", ...) or a per-k dict field ("icc_k",
    "removable_at_k") in which case `k` selects the entry.

    clip=True floors the projected value at 0 -- use at DISPLAY time only,
    never before computing no-signal statistics.
    """
    out = {}
    for cfg, by_func in components.items():
        out[cfg] = {}
        for fk, by_group in by_func.items():
            out[cfg][fk] = {}
            for grp, feats in by_group.items():
                d = {}
                for fn, st in feats.items():
                    v = st.get(field, np.nan)
                    if isinstance(v, dict):
                        if k is None:
                            raise ValueError(
                                f"field {field!r} is per-k; pass k=... "
                                f"(available: {sorted(v)})")
                        v = v.get(k, np.nan)
                    v = float(v) if v is not None else np.nan
                    if clip and np.isfinite(v) and v < 0:
                        v = 0.0
                    d[fn] = v
                out[cfg][fk][grp] = d
    return out


def components_to_dataframe(components, dimension=None):
    """Flatten comp[config][(func,dim)][group][feature] -> tidy DataFrame.

    One row per cell, with icc / msb / msw / sigma2_* / noise_frac plus one
    column per k ("icc_k1", "removable_k1", ...). Requires pandas.
    """
    import pandas as pd
    import re
    rows = []
    for cfg, by_func in components.items():
        m = re.match(r"^(.*)_(\d+)$", cfg)
        strategy, size = (m.group(1), int(m.group(2))) if m else (cfg, np.nan)
        for (func, dim), by_group in by_func.items():
            if dimension is not None and dim != dimension:
                continue
            for grp, feats in by_group.items():
                for fn, st in feats.items():
                    row = {"strategy": strategy, "size": size, "func": func,
                           "dimension": dim, "feature_group": grp, "feature": fn}
                    for key, val in st.items():
                        if key in ("icc_k", "removable_at_k"):
                            prefix = "icc_k" if key == "icc_k" else "removable_k"
                            for k, v in val.items():
                                row[f"{prefix}{k}"] = v
                        else:
                            row[key] = val
                    rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir=None, dimension=5, clip_negative=False,
         k_average=K_AVERAGE):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    input_dir = Path(input_dir)
    output_file = Path(output_dir) / "ela_icc_results.pkl"

    ela_comp = {}
    omit_features = get_omit_features(dimension)

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

        config_comp = {}

        for func_id in range(1, N_FUNCTIONS + 1):
            func_key = (func_id, dimension)

            func_comp = {}
            for group_name, feature_names in ELA_FEATURE_GROUPS.items():
                if group_name in OMIT_GROUPS:
                    continue

                group_comp = {}
                for feat_name in feature_names:
                    if feat_name in omit_features or _is_runtime(feat_name):
                        continue

                    matrix = []  # one row per instance: its 30 run values
                    for inst_id in range(1, N_INSTANCES + 1):
                        instance_key = (func_id, inst_id, dimension)
                        if instance_key not in data:
                            continue
                        instance_data = data[instance_key]
                        if group_name not in instance_data:
                            continue
                        group_data = instance_data[group_name]
                        row = []
                        for run_idx in range(N_RUNS):
                            try:
                                row.append(group_data[run_idx][feat_name])
                            except (IndexError, KeyError):
                                row.append(np.nan)
                        matrix.append(row)

                    if len(matrix) >= 2:
                        stats = compute_icc_components(
                            matrix, clip_negative=clip_negative,
                            k_average=k_average,
                        )
                    else:
                        # too few instances present -> degenerate
                        stats = _nan_result(k_average)

                    group_comp[feat_name] = stats

                if group_comp:
                    func_comp[group_name] = group_comp

            config_comp[func_key] = func_comp

        ela_comp[config_key] = config_comp

        # ---- summary ----
        cells = [s for fk in config_comp.values()
                 for grp in fk.values() for s in grp.values()]
        ok = [s for s in cells if np.isfinite(s["icc"])]
        if ok:
            iccs = np.array([s["icc"] for s in ok])
            msws = np.array([s["msw"] for s in ok])
            s2b = np.array([s["sigma2_between"] for s in ok])
            nan_count = len(cells) - len(ok)

            print(f"  Functions processed: {len(config_comp)}")
            print(f"  Overall median ICC(1,1): {np.median(iccs):.4f} "
                  f"({reliability_band(np.median(iccs))})")
            print(f"  Overall mean ICC(1,1):   {np.mean(iccs):.4f}")
            print(f"  Overall 10th pct:        {np.percentile(iccs, 10):.4f}")
            print(f"  NaN/degenerate count:    {nan_count}")

            # --- noise decomposition ---
            zero_frac = float(np.mean(iccs <= 0))
            neg_frac = float(np.mean([s["icc_raw"] < 0 for s in ok]))
            nosig = float(np.mean(s2b <= 0))
            print(f"  Noise decomposition:")
            print(f"    median noise_frac (1 - ICC):     {np.median(1 - iccs):.4f}")
            print(f"    cells on the clipped floor:      {zero_frac:.2%}")
            print(f"    cells with raw ICC < 0:          {neg_frac:.2%}")
            print(f"    cells with sigma^2_between <= 0: {nosig:.2%}  "
                  f"(no detectable between-instance signal)")
            print(f"    median MSW (scale-dependent):    {np.median(msws):.6g}")

            # --- averaging: ICC(1,k) ---
            print(f"  Reliability of a k-run average (median across cells):")
            for k in k_average:
                vals = np.array([s["icc_k"][k] for s in ok
                                 if np.isfinite(s["icc_k"][k])])
                rem = np.array([s["removable_at_k"][k] for s in ok
                                if np.isfinite(s["removable_at_k"][k])])
                if vals.size:
                    print(f"    k={k:<2d} ICC(1,k)={np.median(vals):.4f}   "
                          f"noise removed={np.median(rem):.1%}")

            for group_name in ELA_FEATURE_GROUPS:
                grp = [s for fk in config_comp.values() if group_name in fk
                       for s in fk[group_name].values() if np.isfinite(s["icc"])]
                if grp:
                    gi = np.array([s["icc"] for s in grp])
                    gz = float(np.mean([s["sigma2_between"] <= 0 for s in grp]))
                    print(f"    {group_name:>10s}: median={np.median(gi):.4f}, "
                          f"mean={np.mean(gi):.4f}, "
                          f"p10={np.percentile(gi, 10):.4f}, "
                          f"no-signal={gz:.1%}")

            # ---- per-group NaN/degenerate breakdown ----
            nan_by_feature, group_of = {}, {}
            for fk in config_comp.values():
                for grp_name, grp in fk.items():
                    for feat_name, s in grp.items():
                        group_of[feat_name] = grp_name
                        if not np.isfinite(s["icc"]):
                            nan_by_feature[feat_name] = nan_by_feature.get(feat_name, 0) + 1

            if nan_count:
                print(f"  Degenerate breakdown ({nan_count} NaN cells across "
                      f"{len(nan_by_feature)} feature(s)):")
                by_group_nan = {}
                for feat_name, cnt in nan_by_feature.items():
                    by_group_nan.setdefault(group_of[feat_name], []).append((feat_name, cnt))
                for grp_name in ELA_FEATURE_GROUPS:
                    offenders = by_group_nan.get(grp_name)
                    if not offenders:
                        continue
                    offenders.sort(key=lambda t: t[1], reverse=True)
                    total = sum(c for _, c in offenders)
                    detail = ", ".join(f"{fn} ({c}/{len(config_comp)} func)"
                                       for fn, c in offenders)
                    print(f"    {grp_name:>10s}: {total} NaN -> {detail}")
            else:
                print("  Degenerate breakdown: none (all cells computable).")

        del data
        print("  Done.")

    with open(output_file, "wb") as f:
        pickle.dump(ela_comp, f)

    print(f"\n\nResults saved to: {output_file}")
    print("\nAccess format:")
    print("  comp[config_key][(func, dim)][feature_group][feature_name] -> dict with")
    print("    icc, icc_raw, msb, msw, sigma2_between, sigma2_within,")
    print(f"    noise_frac, icc_k{{{','.join(map(str, k_average))}}}, removable_at_k{{...}}")
    print(f"  Negatives {'CLIPPED to 0' if clip_negative else 'KEPT'}; "
          f"use sigma2_between <= 0 as the no-signal indicator.")
    print("\n  For scripts expecting the old bare-float layout:")
    print("    from compute_ela_icc import load_icc, project")
    print("    ela_icc = project(load_icc(path))            # field='icc'")
    print("    ela_icc = project(load_icc(path), 'msw')     # or any field")
    print("    ela_icc = project(load_icc(path), 'icc_k', k=5)")
    print("\n  Tidy DataFrame:")
    print("    from compute_ela_icc import components_to_dataframe")
    print("    df = components_to_dataframe(load_icc(path))")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-feature within-config ICC + variance components "
                    "for ELA features across 30 runs "
                    "(subjects = a function's instances)."
    )
    parser.add_argument(
        "input_dir", type=str,
        help="Directory containing the ELA pkl files.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output pkl files. Defaults to input_dir.",
    )
    parser.add_argument("--dimension", type=int, default=5,
                        help="Problem dimensionality (default: 5).")
    parser.add_argument("--clip-negative", action="store_true",
                        help="Clip negative ICCs to 0 (OFF by default). Clipping "
                             "halves the apparent no-signal rate and biases "
                             "aggregates upward; prefer clipping at display time.")
    parser.add_argument("--k-average", type=int, nargs="+", default=K_AVERAGE,
                        metavar="K",
                        help="Run-averaging sizes for ICC(1,k), matching the "
                             "downstream classifier sweep (default: 1 2 3 5).")
    args = parser.parse_args()
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dimension=args.dimension,
        clip_negative=args.clip_negative,
        k_average=args.k_average,
    )