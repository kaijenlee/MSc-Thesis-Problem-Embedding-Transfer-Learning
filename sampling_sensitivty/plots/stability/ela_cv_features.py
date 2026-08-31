"""
Filtered ELA features for Coefficient of Variation (CV = sigma / mu) analysis
and cross-feature aggregation.

Selection rule
--------------
CV is only interpretable when a feature is ratio-scaled (a true, meaningful
zero denoting *absence*) AND strictly positive (single sign, mean well away
from zero). Features that change sign, that use an interval-scale zero
(0 = "symmetric" / "no correlation", not "none"), or that have an arbitrary
offset are excluded because their CV is unstable and not comparable across
features when aggregated.

Every *.costs_runtime is dropped: runtime is positive and CV-computable, but
it describes the *computation* (hardware, implementation, load, sample size),
not the *landscape*, so it is non-reproducible and would inject environment
noise into a cross-feature aggregate.

Two tiers are provided:
  * CV_SAFE      -> clean, positive, ratio-scaled. Use these by default.
  * CV_CAVEATED  -> CV is computable but interpret with care (bounded
                    proportions, log-scaled ranges, scale-dependence, or a
                    quantity that is itself already a CV). Opt in explicitly.
"""

# --- Tier 1: clean, positive, ratio-scaled -----------------------------------
CV_SAFE = [
    "ela_distr.number_of_peaks",          # count >= 1, true zero
    "disp.ratio_mean_02",                 # dispersion ratios, strictly positive
    "disp.ratio_mean_05",
    "disp.ratio_mean_10",
    "disp.ratio_mean_25",
    "disp.ratio_median_02",
    "disp.ratio_median_05",
    "disp.ratio_median_10",
    "disp.ratio_median_25",
    "nbc.nn_nb.sd_ratio",                 # ratio of distances, positive
    "nbc.nn_nb.mean_ratio",
    "ic.h_max",                           # entropy-like, non-negative, true zero
    "ic.eps_ratio",                       # partial-information ratio, positive
    "ela_meta.quad_simple.cond",          # condition number >= 1
    "ela_meta.lin_simple.coef.max_by_min",# max/min ratio >= 1
]

# --- Tier 2: computable but interpret with care ------------------------------
# Each entry maps to the reason to be cautious when aggregating.
CV_CAVEATED = {
    "pca.expl_var.cov_x":        "bounded proportion in (0,1]; clustering near bound deflates CV",
    "pca.expl_var.cor_x":        "bounded proportion in (0,1]; clustering near bound deflates CV",
    "pca.expl_var.cov_init":     "bounded proportion in (0,1]; clustering near bound deflates CV",
    "pca.expl_var.cor_init":     "bounded proportion in (0,1]; clustering near bound deflates CV",
    "pca.expl_var_PC1.cov_x":    "bounded proportion in (0,1]; clustering near bound deflates CV",
    "pca.expl_var_PC1.cor_x":    "bounded proportion in (0,1]; clustering near bound deflates CV",
    "pca.expl_var_PC1.cov_init": "bounded proportion in (0,1]; clustering near bound deflates CV",
    "pca.expl_var_PC1.cor_init": "bounded proportion in (0,1]; clustering near bound deflates CV",
    "ic.m0":                     "bounded in [0,1]; same deflation caveat as proportions",
    "ic.eps_s":                  "log-scaled, spans orders of magnitude and can hit 0; prefer CV(log) or robust spread",
    "ic.eps_max":                "log-scaled, spans orders of magnitude; prefer CV(log) or robust spread",
    "ela_meta.lin_simple.coef.min": "scale-dependent on objective magnitude; near-zero values blow up CV (normalize first)",
    "ela_meta.lin_simple.coef.max": "scale-dependent on objective magnitude (normalize first)",
    "nbc.dist_ratio.coeff_var":  "feature is itself a CV; taking a CV of a CV is semantically nested",
}

# --- Excluded (kept explicit for auditability) -------------------------------
CV_EXCLUDED = {
    "ela_distr.skewness":              "sign-changing, interval-scale (0 = symmetric)",
    "ela_distr.kurtosis":              "sign-changing, interval-scale (0 = mesokurtic)",
    "ela_meta.lin_simple.intercept":   "arbitrary sign and scale, no true zero",
    "ela_meta.lin_simple.adj_r2":      "can be negative; bounded above by 1; not ratio-scaled",
    "ela_meta.lin_w_interact.adj_r2":  "can be negative; not ratio-scaled",
    "ela_meta.quad_simple.adj_r2":     "can be negative; not ratio-scaled",
    "ela_meta.quad_w_interact.adj_r2": "can be negative; not ratio-scaled",
    "disp.diff_mean_02":    "difference, can be negative; 0 = no difference",
    "disp.diff_mean_05":    "difference, can be negative; 0 = no difference",
    "disp.diff_mean_10":    "difference, can be negative; 0 = no difference",
    "disp.diff_mean_25":    "difference, can be negative; 0 = no difference",
    "disp.diff_median_02":  "difference, can be negative; 0 = no difference",
    "disp.diff_median_05":  "difference, can be negative; 0 = no difference",
    "disp.diff_median_10":  "difference, can be negative; 0 = no difference",
    "disp.diff_median_25":  "difference, can be negative; 0 = no difference",
    "nbc.nn_nb.cor":        "correlation in [-1,1], sign-changing; 0 = no correlation",
    "nbc.nb_fitness.cor":   "correlation in [-1,1], sign-changing; 0 = no correlation",
    # all *.costs_runtime dropped as environment-dependent (see module docstring)
}

ELA_FEATURE_GROUPS = {
    "ela_dist": [
        "ela_distr.skewness", "ela_distr.kurtosis",
        "ela_distr.number_of_peaks",
    ],
    "meta": [
        "ela_meta.lin_simple.adj_r2", "ela_meta.lin_simple.intercept",
        "ela_meta.lin_simple.coef.min", "ela_meta.lin_simple.coef.max",
        "ela_meta.lin_simple.coef.max_by_min", "ela_meta.lin_w_interact.adj_r2",
        "ela_meta.quad_simple.adj_r2", "ela_meta.quad_simple.cond",
        "ela_meta.quad_w_interact.adj_r2"
    ],
    "disp": [
        "disp.ratio_mean_02", "disp.ratio_mean_05", "disp.ratio_mean_10",
        "disp.ratio_mean_25", "disp.ratio_median_02", "disp.ratio_median_05",
        "disp.ratio_median_10", "disp.ratio_median_25", "disp.diff_mean_02",
        "disp.diff_mean_05", "disp.diff_mean_10", "disp.diff_mean_25",
        "disp.diff_median_02", "disp.diff_median_05", "disp.diff_median_10",
        "disp.diff_median_25"
    ],
    "nbc": [
        "nbc.nn_nb.sd_ratio", "nbc.nn_nb.mean_ratio", "nbc.nn_nb.cor",
        "nbc.dist_ratio.coeff_var", "nbc.nb_fitness.cor"
    ],
    "ic": [
        "ic.h_max", "ic.eps_s", "ic.eps_max", "ic.eps_ratio",
        "ic.m0"
    ],
    "pca": [
        "pca.expl_var.cov_x", "pca.expl_var.cor_x", "pca.expl_var.cov_init",
        "pca.expl_var.cor_init", "pca.expl_var_PC1.cov_x",
        "pca.expl_var_PC1.cor_x", "pca.expl_var_PC1.cov_init",
        "pca.expl_var_PC1.cor_init"
    ]
}


def get_cv_features(include_caveated: bool = False) -> list:
    """Return the flat list of feature keys eligible for CV aggregation.

    Parameters
    ----------
    include_caveated : bool
        If True, append the Tier-2 (caveated) features to the clean set.
    """
    feats = list(CV_SAFE)
    if include_caveated:
        feats += list(CV_CAVEATED.keys())
    return feats


def filter_groups(ela_feature_groups: dict, include_caveated: bool = False) -> dict:
    """Filter an ELA_FEATURE_GROUPS-style dict down to CV-eligible features.

    Keeps the original group structure, drops empty groups, and validates that
    every selected key actually exists in the input.
    """
    keep = set(get_cv_features(include_caveated))
    all_keys = {k for v in ela_feature_groups.values() for k in v}
    missing = keep - all_keys
    if missing:
        raise KeyError(f"Selected features not present in input groups: {sorted(missing)}")

    out = {}
    for group, feats in ela_feature_groups.items():
        kept = [f for f in feats if f in keep]
        if kept:
            out[group] = kept
    return out


if __name__ == "__main__":
    print(f"CV_SAFE:      {len(CV_SAFE)} features")
    print(f"CV_CAVEATED:  {len(CV_CAVEATED)} features")
    print(f"CV_EXCLUDED:  {len(CV_EXCLUDED)} features (+ all *.costs_runtime)")