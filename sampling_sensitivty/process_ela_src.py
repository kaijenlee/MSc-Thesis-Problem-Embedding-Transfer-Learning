"""
Compute per-feature within-config rank-stability (Spearman) for ELA features
across 30 runs.

Where ICC asks whether a feature's *values* are reproducible, rank-stability
asks whether its *ordering of the instances* is reproducible across independent
samples. For a fixed (config, dimension, function, feature) we build the
100-instances x 30-runs matrix, take every pair of runs, compute the Spearman
rank correlation between their two instance-value vectors (i.e. do the two runs
rank the instances the same way?), and average over all run-pairs. This is then
the rank-stability of that feature on that function.

Rationale (vs ICC):
  - Monotone-invariant: immune to scale and to any systematic per-run/strategy
    shift (a feature whose values drift but whose ordering holds scores ~1).
  - Not depressed by small between-instance signal in the same way ICC is, so a
    low ICC + high rank-stability flags "values jitter but ordering is stable".
  - Matches downstream use: ELA features are used to *compare / rank* problems
    (e.g. random-forest splits are pure threshold comparisons), so reproducible
    ordering is the decision-relevant reliability property.

The subject being ranked is the function's instances; each "variable" is a run.
A correlation therefore needs >= 3 instances and is undefined for a run whose
values are constant across instances (no order to compare).

Output format (pkl):
  ela_rho[config_key][(func, dim)][feature_group][feature_name] = mean_spearman

Usage:
  python process_ela_src.py /path/to/dim2feat --output-dir /path/to/output --dimension 2
"""

import argparse
import pickle
import numpy as np
import os
from itertools import combinations
from pathlib import Path

from scipy.stats import spearmanr, kendalltau

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_FUNCTIONS = 24
N_INSTANCES = 100
N_RUNS = 30

# Minimum instances needed for a meaningful rank correlation.
MIN_SUBJECTS = 3


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
# Rank-stability computation
# ---------------------------------------------------------------------------

def compute_rank_stability(value_matrix, method="spearman", max_run_pairs=None,
                           rng=None):
    """
    Mean pairwise rank correlation across runs for a (subjects x runs) matrix.

    Rows = subjects (a function's instances), columns = the runs. For each pair
    of run-columns we compute the rank correlation between their instance
    vectors (over the instances finite in both runs), then average over pairs.

    method        : "spearman" (default) or "kendall".
    max_run_pairs : cap on the number of run-pairs averaged (None = all pairs;
                    30 runs -> 435 pairs). Sampling is reproducible via rng.

    Returns np.nan if the matrix is degenerate (fewer than MIN_SUBJECTS usable
    instances, or no run-pair yields a defined correlation -- e.g. a feature
    constant across instances, which has no order to compare).
    """
    corr_fn = kendalltau if method == "kendall" else spearmanr
    M = np.asarray(value_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        return np.nan

    n_runs = M.shape[1]
    pairs = list(combinations(range(n_runs), 2))
    if max_run_pairs is not None and len(pairs) > max_run_pairs:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(pairs), size=max_run_pairs, replace=False)
        pairs = [pairs[t] for t in idx]

    cors = []
    for i, j in pairs:
        a = M[:, i]
        b = M[:, j]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < MIN_SUBJECTS:
            continue
        av, bv = a[ok], b[ok]
        # A run with no variance across instances has undefined ranks -> skip.
        if np.ptp(av) == 0 or np.ptp(bv) == 0:
            continue
        c = corr_fn(av, bv).correlation
        if np.isfinite(c):
            cors.append(c)

    if not cors:
        return np.nan
    return float(np.mean(cors))


def stability_band(rho):
    """Rough interpretation band for a rank-stability value (rule of thumb)."""
    if rho is None or not np.isfinite(rho):
        return "n/a"
    if rho >= 0.90:
        return "excellent"
    if rho >= 0.75:
        return "good"
    if rho >= 0.50:
        return "moderate"
    return "poor"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir=None, dimension=5, method="spearman",
         max_run_pairs=None):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    input_dir = Path(input_dir)
    output_file = Path(output_dir) / "ela_rank_stability_results.pkl"

    ela_rho = {}
    omit_features = get_omit_features(dimension)
    rng = np.random.default_rng(0)

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

        config_rho = {}

        for func_id in range(1, N_FUNCTIONS + 1):
            func_key = (func_id, dimension)

            func_rho = {}
            for group_name, feature_names in ELA_FEATURE_GROUPS.items():
                if group_name in OMIT_GROUPS:
                    continue

                group_rho = {}
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

                    if len(matrix) >= MIN_SUBJECTS:
                        group_rho[feat_name] = compute_rank_stability(
                            matrix, method=method,
                            max_run_pairs=max_run_pairs, rng=rng,
                        )
                    else:
                        # too few instances present -> degenerate, record as NaN
                        group_rho[feat_name] = np.nan

                if group_rho:
                    func_rho[group_name] = group_rho

            config_rho[func_key] = func_rho

        ela_rho[config_key] = config_rho

        # ---- summary ----
        all_rhos = [
            v for fk in config_rho.values()
            for grp in fk.values()
            for v in grp.values()
            if not np.isnan(v)
        ]
        if all_rhos:
            print(f"  Functions processed: {len(config_rho)}")
            print(f"  Overall median rho: {np.median(all_rhos):.4f} "
                  f"({stability_band(np.median(all_rhos))})")
            print(f"  Overall mean rho:   {np.mean(all_rhos):.4f}")
            print(f"  Overall 10th pct:   {np.percentile(all_rhos, 10):.4f}")
            nan_count = sum(
                1 for fk in config_rho.values()
                for grp in fk.values()
                for v in grp.values()
                if np.isnan(v)
            )
            print(f"  NaN/degenerate count: {nan_count}")

            for group_name in ELA_FEATURE_GROUPS:
                grp_rhos = [
                    v for fk in config_rho.values()
                    if group_name in fk
                    for v in fk[group_name].values()
                    if not np.isnan(v)
                ]
                if grp_rhos:
                    print(f"    {group_name:>10s}: median={np.median(grp_rhos):.4f}, "
                          f"mean={np.mean(grp_rhos):.4f}, "
                          f"p10={np.percentile(grp_rhos, 10):.4f}")

            # ---- per-group NaN/degenerate breakdown ----
            nan_by_feature = {}
            group_of = {}
            for fk in config_rho.values():
                for grp_name, grp in fk.items():
                    for feat_name, v in grp.items():
                        group_of[feat_name] = grp_name
                        if np.isnan(v):
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
                    detail = ", ".join(f"{fn} ({c}/{len(config_rho)} func)"
                                       for fn, c in offenders)
                    print(f"    {grp_name:>10s}: {total} NaN -> {detail}")
            else:
                print("  Degenerate breakdown: none (all cells computable).")

        del data
        print("  Done.")

    with open(output_file, "wb") as f:
        pickle.dump(ela_rho, f)

    print(f"\n\nResults saved to: {output_file}")
    print("\nAccess format:")
    print("  ela_rho[config_key][(func, dim)][feature_group][feature_name]")
    print("  e.g. ela_rho['ilhs_50'][(1, 2)]['meta']['ela_meta.lin_simple.adj_r2']")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-feature within-config rank-stability (mean "
                    "pairwise Spearman across runs) for ELA features. Subjects "
                    "= a function's instances; variables = the 30 runs."
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
    parser.add_argument("--method", default="spearman",
                        choices=["spearman", "kendall"],
                        help="Rank correlation method (default: spearman).")
    parser.add_argument("--max-run-pairs", type=int, default=None,
                        help="Cap on run-pairs averaged per feature for speed "
                             "(default: all 435 pairs).")
    args = parser.parse_args()
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dimension=args.dimension,
        method=args.method,
        max_run_pairs=args.max_run_pairs,
    )