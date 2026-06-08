"""
Compute per-feature within-config reliability (ICC) for ELA features across
30 runs.

A single ICC is a *between-instances* quantity: it contrasts the variance
between a function's instances (signal) with the run-to-run variance within
each instance (sampling noise). So, unlike CV, it cannot be produced for one
instance in isolation -- the natural unit is one ICC per
(config, dimension, function, feature), with that function's 100 instances as
the subjects and the 30 runs as the replicate measurements.

Model: one-way random-effects ICC(1,1) (Shrout & Fleiss). The runs are
independent draws with no identity across instances, so this is the correct
form (not ICC(2,1)/ICC(3,1)). It is identical to pingouin's "ICC(1,1)" and to
R psych's ICC1, just computed in closed form for speed.

    ICC = (MSB - MSW) / (MSB + (k - 1) * MSW)
        = sigma^2_between / (sigma^2_between + sigma^2_within)

Interpretation: ICC ~ 1 means the feature value is essentially fixed by which
instance it is and barely moved by resampling; ICC ~ 0 means it is mostly
sampling noise. Bands (Koo & Li): >0.9 excellent, 0.75-0.9 good,
0.5-0.75 moderate, <0.5 poor. A negative value means run noise exceeds the
between-instance signal (reported as ~0 after optional clipping).

Output format (pkl):
  ela_icc[config_key][(func, dim)][feature_group][feature_name] = icc_value

Usage:
  python compute_ela_icc.py /path/to/dim2feat --output-dir /path/to/output --dimension 2
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
N_RUNS = 30


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

def compute_icc(value_matrix, clip_negative=True):
    """
    One-way random-effects ICC(1,1) for a (subjects x runs) matrix.

    Rows = subjects (a function's instances), columns = the 30 runs. Subjects
    with fewer than 2 finite runs are dropped; a balanced design is assumed and
    each retained subject is trimmed to the common run count.

    Returns np.nan if the matrix is degenerate (fewer than 2 usable subjects,
    or no between-subject variance). Negative ICCs are clipped to 0 by default
    (run noise >= between-instance signal).
    """
    M = np.asarray(value_matrix, dtype=float)
    if M.ndim != 2:
        return np.nan

    finite_per_row = np.isfinite(M).sum(axis=1)
    M = M[finite_per_row >= 2]
    if M.shape[0] < 2:
        return np.nan

    k = int(np.min(np.isfinite(M).sum(axis=1)))
    if k < 2:
        return np.nan
    rows = [r[np.isfinite(r)][:k] for r in M]
    M = np.asarray(rows, dtype=float)
    n = M.shape[0]

    grand = M.mean()
    row_means = M.mean(axis=1)
    ss_between = k * np.sum((row_means - grand) ** 2)
    ss_within = np.sum((M - row_means[:, None]) ** 2)
    df_between = n - 1
    df_within = n * (k - 1)
    if df_between <= 0 or df_within <= 0:
        return np.nan

    msb = ss_between / df_between
    msw = ss_within / df_within
    denom = msb + (k - 1) * msw
    if denom <= 0:
        return np.nan

    icc = (msb - msw) / denom
    if clip_negative and icc < 0:
        icc = 0.0
    return float(icc)


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
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir=None, dimension=5, clip_negative=True):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    input_dir = Path(input_dir)
    output_file = Path(output_dir) / "ela_icc_results.pkl"

    ela_icc = {}
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

        config_icc = {}

        for func_id in range(1, N_FUNCTIONS + 1):
            func_key = (func_id, dimension)

            # Gather, per group/feature, a (instance x run) matrix for this function.
            func_icc = {}
            for group_name, feature_names in ELA_FEATURE_GROUPS.items():
                if group_name in OMIT_GROUPS:
                    continue

                group_icc = {}
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
                        group_icc[feat_name] = compute_icc(
                            matrix, clip_negative=clip_negative
                        )
                    else:
                        # too few instances present -> degenerate, record as NaN
                        group_icc[feat_name] = np.nan

                if group_icc:
                    func_icc[group_name] = group_icc

            config_icc[func_key] = func_icc

        ela_icc[config_key] = config_icc

        # ---- summary ----
        all_iccs = [
            v for fk in config_icc.values()
            for grp in fk.values()
            for v in grp.values()
            if not np.isnan(v)
        ]
        if all_iccs:
            print(f"  Functions processed: {len(config_icc)}")
            print(f"  Overall median ICC: {np.median(all_iccs):.4f} "
                  f"({reliability_band(np.median(all_iccs))})")
            print(f"  Overall mean ICC:   {np.mean(all_iccs):.4f}")
            print(f"  Overall 10th pct:   {np.percentile(all_iccs, 10):.4f}")
            nan_count = sum(
                1 for fk in config_icc.values()
                for grp in fk.values()
                for v in grp.values()
                if np.isnan(v)
            )
            print(f"  NaN/degenerate count: {nan_count}")

            for group_name in ELA_FEATURE_GROUPS:
                grp_iccs = [
                    v for fk in config_icc.values()
                    if group_name in fk
                    for v in fk[group_name].values()
                    if not np.isnan(v)
                ]
                if grp_iccs:
                    print(f"    {group_name:>10s}: median={np.median(grp_iccs):.4f}, "
                          f"mean={np.mean(grp_iccs):.4f}, "
                          f"p10={np.percentile(grp_iccs, 10):.4f}")

            # ---- per-group NaN/degenerate breakdown ----
            # Count, per feature, on how many functions its ICC was NaN.
            nan_by_feature = {}   # feature_name -> count of functions degenerate
            group_of = {}
            for fk in config_icc.values():
                for grp_name, grp in fk.items():
                    for feat_name, v in grp.items():
                        group_of[feat_name] = grp_name
                        if np.isnan(v):
                            nan_by_feature[feat_name] = nan_by_feature.get(feat_name, 0) + 1

            if nan_count:
                print(f"  Degenerate breakdown ({nan_count} NaN cells across "
                      f"{len(nan_by_feature)} feature(s)):")
                # group the offending features by their feature group
                by_group_nan = {}
                for feat_name, cnt in nan_by_feature.items():
                    by_group_nan.setdefault(group_of[feat_name], []).append((feat_name, cnt))
                for grp_name in ELA_FEATURE_GROUPS:
                    offenders = by_group_nan.get(grp_name)
                    if not offenders:
                        continue
                    offenders.sort(key=lambda t: t[1], reverse=True)
                    total = sum(c for _, c in offenders)
                    detail = ", ".join(f"{fn} ({c}/{len(config_icc)} func)"
                                       for fn, c in offenders)
                    print(f"    {grp_name:>10s}: {total} NaN -> {detail}")
            else:
                print("  Degenerate breakdown: none (all cells computable).")

        del data
        print("  Done.")

    with open(output_file, "wb") as f:
        pickle.dump(ela_icc, f)

    print(f"\n\nResults saved to: {output_file}")
    print("\nAccess format:")
    print("  ela_icc[config_key][(func, dim)][feature_group][feature_name]")
    print("  e.g. ela_icc['ilhs_50'][(1, 2)]['meta']['ela_meta.lin_simple.adj_r2']")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-feature within-config ICC for ELA features "
                    "across 30 runs (subjects = a function's instances)."
    )
    parser.add_argument(
        "input_dir", type=str,
        help="Directory containing the ELA pkl files.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output pkl file. Defaults to input_dir.",
    )
    parser.add_argument("--dimension", type=int, default=5,
                        help="Problem dimensionality (default: 5).")
    parser.add_argument("--no-clip-negative", action="store_true",
                        help="Keep negative ICCs instead of clipping them to 0.")
    args = parser.parse_args()
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dimension=args.dimension,
        clip_negative=not args.no_clip_negative,
    )