"""
Compute per-feature CV for ELA features across 30 runs.

Output format (pkl):
  ela_cv[config_key][(func, inst, dim)][feature_group][feature_name] = cv_value

Usage:
  python compute_ela_cv.py /path/to/dim2feat --output-dir /path/to/output
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
DIMENSION = 2

OMIT_FEATURES = {
    "disp.diff_median_02", "disp.ratio_median_02",
    "ela_level.lda_qda_50", "ela_level.lda_qda_25",
    "ic.eps_ratio", "disp.ratio_mean_02", "disp.diff_mean_02",
    "ela_level.lda_qda_10", "ela_meta.quad_simple.cond",
}
OMIT_GROUPS = {"levelset"}

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
    "levelset": [
        "ela_level.mmce_lda_10", "ela_level.mmce_qda_10", "ela_level.lda_qda_10",
        "ela_level.mmce_lda_25", "ela_level.mmce_qda_25", "ela_level.lda_qda_25",
        "ela_level.mmce_lda_50", "ela_level.mmce_qda_50", "ela_level.lda_qda_50",
        "ela_level.costs_runtime",
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
}


# ---------------------------------------------------------------------------
# CV computation
# ---------------------------------------------------------------------------

def compute_cv(values):
    """
    Compute coefficient of variation (std / mean) for an array of values.
    Returns np.nan if mean is zero or if values contain NaN.
    """
    values = np.array(values, dtype=float)
    if np.any(np.isnan(values)):
        return np.nan
    mean = np.mean(values)
    if mean == 0:
        return np.nan
    return np.std(values, ddof=0) / abs(mean)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    input_dir = Path(input_dir)
    output_file = Path(output_dir) / "ela_cv_results.pkl"

    ela_cv = {}

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

        config_cv = {}

        for func_id in range(1, N_FUNCTIONS + 1):
            for inst_id in range(1, N_INSTANCES + 1):
                instance_key = (func_id, inst_id, DIMENSION)

                if instance_key not in data:
                    print(f"  WARNING: {instance_key} not found in {filename}")
                    continue

                instance_data = data[instance_key]
                instance_cv = {}

                for group_name, feature_names in ELA_FEATURE_GROUPS.items():
                    if group_name in OMIT_GROUPS:
                        continue

                    if group_name not in instance_data:
                        print(f"  WARNING: {group_name} not found for {instance_key}")
                        continue

                    group_data = instance_data[group_name]
                    group_cv = {}

                    for feat_name in feature_names:
                        if feat_name in OMIT_FEATURES:
                            continue

                        # Collect this feature's value across 30 runs
                        values = []
                        for run_idx in range(N_RUNS):
                            values.append(group_data[run_idx][feat_name])

                        group_cv[feat_name] = compute_cv(values)

                    instance_cv[group_name] = group_cv

                config_cv[instance_key] = instance_cv

        ela_cv[config_key] = config_cv

        # Print summary
        n_instances = len(config_cv)
        all_cvs = []
        for inst_cv in config_cv.values():
            for grp_cv in inst_cv.values():
                all_cvs.extend(v for v in grp_cv.values() if not np.isnan(v))
        if all_cvs:
            print(f"  Instances processed: {n_instances}")
            print(f"  Overall median CV: {np.median(all_cvs):.4f}")
            print(f"  Overall mean CV:   {np.mean(all_cvs):.4f}")
            print(f"  Overall 90th pct:  {np.percentile(all_cvs, 90):.4f}")
            print(f"  NaN count: {sum(1 for inst_cv in config_cv.values() for grp_cv in inst_cv.values() for v in grp_cv.values() if np.isnan(v))}")

            # Per-group summary
            for group_name in ELA_FEATURE_GROUPS:
                grp_cvs = []
                for inst_cv in config_cv.values():
                    if group_name in inst_cv:
                        grp_cvs.extend(
                            v for v in inst_cv[group_name].values()
                            if not np.isnan(v)
                        )
                if grp_cvs:
                    print(f"    {group_name:>10s}: median={np.median(grp_cvs):.4f}, "
                          f"mean={np.mean(grp_cvs):.4f}, "
                          f"p90={np.percentile(grp_cvs, 90):.4f}")

        del data
        print(f"  Done.")

    # Save results
    with open(output_file, "wb") as f:
        pickle.dump(ela_cv, f)

    print(f"\n\nResults saved to: {output_file}")
    print(f"\nAccess format:")
    print(f"  ela_cv[config_key][(func, inst, dim)][feature_group][feature_name]")
    print(f"  e.g. ela_cv['ilhs_50'][(1, 1, 2)]['meta']['ela_meta.lin_simple.adj_r2']")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-feature CV for ELA features across 30 runs."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing the ELA pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output pkl file. Defaults to input_dir.",
    )
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir)