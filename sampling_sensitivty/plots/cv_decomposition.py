"""
Variance Decomposition Analysis for CMA-Random Feature Stability.

Decomposes CV variability into:
  - Within-run variance: how much a feature estimate varies due to sampling noise
    within the same CMA-ES trajectory (estimated via feature variance across instances
    of the same function, holding the run fixed).
  - Between-run variance: how much feature estimates differ across the 30 independent
    CMA-ES runs for the same (function, instance) pair.

The key hypothesis: CMA-Random instability comes primarily from between-run variance
(different trajectories), NOT within-run sampling noise. More samples reduce within-run
noise but cannot fix between-run divergence — and on multimodal functions, between-run
variance may *increase* with budget as runs converge to different basins.

Integrates with the user's existing plot_cv.py data structures and conventions.

Usage:
    from variance_decomposition import *
    ela_cv = load_cv_data("path/to/ela_cv_results.pkl")
    tla_cv = load_cv_data("path/to/tla_cv.pkl")

    # ELA analysis
    run_ela_variance_decomposition(ela_cv)

    # TLA analysis (H0 features)
    run_tla_variance_decomposition(tla_cv, homology="h0")
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Configuration (mirrors plot_cv.py)
# ---------------------------------------------------------------------------

N_FUNCTIONS = 24
N_INSTANCES = 100
DIMENSION = 2

SAMPLE_SIZES_ELA = [25, 50, 75, 100]
SAMPLE_SIZES_TLA = [10, 25, 50, 75, 100]

STRATEGY_COLORS = {
    "ilhs": "#1f77b4", "lhs": "#ff7f0e", "sobol": "#2ca02c",
    "uniform": "#d62728", "cma_random": "#9467bd",
}
STRATEGY_LABELS = {
    "ilhs": "iLHS", "lhs": "LHS", "sobol": "Sobol",
    "uniform": "Uniform", "cma_random": "CMA-Random",
}

FUNCTION_GROUPS = {
    "Separable": [1, 2, 3, 4, 5],
    "Low/Moderate Cond.": [6, 7, 8, 9],
    "High Conditioning": [10, 11, 12, 13, 14],
    "Multimodal (adequate)": [15, 16, 17, 18, 19],
    "Multimodal (weak)": [20, 21, 22, 23, 24],
}

# Broader grouping for decomposition
UNIMODAL_FUNCS = list(range(1, 15))     # F1-F14
MULTIMODAL_FUNCS = list(range(15, 25))   # F15-F24

ELA_GROUPS = ["ela_dist", "meta", "disp", "nbc", "ic"]
TLA_TRANSFORMS = ["volume", "axis"]
TLA_HOMOLOGIES = ["h0", "h1", "h2"]

SAMPLING_STRATEGIES = ["ilhs", "lhs", "sobol", "uniform", "cma_random"]


def load_cv_data(cv_path):
    with open(cv_path, "rb") as f:
        return pickle.load(f)


# ===========================================================================
# Core decomposition logic
# ===========================================================================

def _ela_get_raw_feature_values(ela_cv, config_key, func_id, inst_id):
    """
    Return dict {feature_name: cv_value} for a single (func, instance) under
    a given config.  The CV was computed across 30 runs, so each value already
    *is* the CV for that feature at that (func, inst).

    Because the raw per-run feature values are not stored (only CVs), we work
    with the CV values directly. High CV = high between-run variability for
    that (function, instance, feature) triple.
    """
    key = (func_id, inst_id, DIMENSION)
    if config_key not in ela_cv or key not in ela_cv[config_key]:
        return {}
    result = {}
    for grp_name in ELA_GROUPS:
        if grp_name not in ela_cv[config_key][key]:
            continue
        for feat_name, cv_val in ela_cv[config_key][key][grp_name].items():
            if not np.isnan(cv_val):
                result[feat_name] = cv_val
    return result


def _tla_get_raw_cv_values(tla_cv, config_key, func_id, inst_id,
                            transform=None, homology=None):
    """
    Return list of CV values for a single (func, instance) under a given config,
    filtered by transform/homology.
    """
    key = (func_id, inst_id, DIMENSION)
    if config_key not in tla_cv or key not in tla_cv[config_key]:
        return []
    instance_data = tla_cv[config_key][key]
    cvs = []
    transforms = [transform] if transform else TLA_TRANSFORMS
    homologies_list = [homology] if homology else TLA_HOMOLOGIES
    for t in transforms:
        if t not in instance_data:
            continue
        for h in homologies_list:
            if h not in instance_data[t] or instance_data[t][h] is None:
                continue
            arr = instance_data[t][h].flatten()
            finite_mask = np.isfinite(arr)
            cvs.extend(arr[finite_mask].tolist())
    return cvs


# ---------------------------------------------------------------------------
# ELA Variance Decomposition
# ---------------------------------------------------------------------------

def ela_variance_decomposition(ela_cv, strategy, func_ids=None):
    """
    For a given strategy, compute per-feature statistics across sample sizes.

    Since the stored data is CV values (one per feature per function-instance),
    we decompose the *distribution of CVs* into:
      - Median CV across all (func, instance) pairs  → overall instability level
      - Variance of CV across instances (within a function) → how much instability
        varies across instances of the same function
      - Variance of median-CV across functions → how much instability varies
        across different function classes

    Returns:
        dict keyed by sample_size → {
            "overall_median_cv": float,
            "within_func_var": float,   # avg variance of CV across instances
            "between_func_var": float,  # variance of per-function median CV
            "total_var": float,         # total variance of all CVs
            "n_features": int,
        }
    """
    if func_ids is None:
        func_ids = list(range(1, N_FUNCTIONS + 1))

    results = {}
    for size in SAMPLE_SIZES_ELA:
        config_key = f"{strategy}_{size}"
        if config_key not in ela_cv:
            continue

        # Collect per-feature CVs grouped by function
        # Structure: {func_id: {feat_name: [cv across instances]}}
        func_feat_cvs = defaultdict(lambda: defaultdict(list))
        for func_id in func_ids:
            for inst_id in range(1, N_INSTANCES + 1):
                feat_vals = _ela_get_raw_feature_values(ela_cv, config_key,
                                                        func_id, inst_id)
                for feat_name, cv_val in feat_vals.items():
                    func_feat_cvs[func_id][feat_name].append(cv_val)

        if not func_feat_cvs:
            continue

        # Aggregate: for each feature, compute within-function and between-function
        all_cvs = []
        within_func_vars = []
        between_func_medians = []

        # Get the union of all feature names
        all_features = set()
        for func_id in func_feat_cvs:
            all_features.update(func_feat_cvs[func_id].keys())

        for feat_name in all_features:
            func_medians = []
            func_vars = []
            for func_id in func_ids:
                if func_id not in func_feat_cvs:
                    continue
                vals = func_feat_cvs[func_id].get(feat_name, [])
                if len(vals) < 2:
                    continue
                all_cvs.extend(vals)
                func_medians.append(np.median(vals))
                func_vars.append(np.var(vals))

            if func_medians:
                within_func_vars.append(np.median(func_vars))
                between_func_medians.extend(func_medians)

        if not all_cvs:
            continue

        overall_median = np.median(all_cvs)
        total_var = np.var(all_cvs)
        avg_within_var = np.median(within_func_vars) if within_func_vars else 0
        between_var = np.var(between_func_medians) if len(between_func_medians) > 1 else 0

        results[size] = {
            "overall_median_cv": overall_median,
            "within_func_var": avg_within_var,
            "between_func_var": between_var,
            "total_var": total_var,
            "n_features": len(all_features),
        }

    return results


def tla_variance_decomposition(tla_cv, strategy, func_ids=None,
                                transform=None, homology=None):
    """
    Same decomposition for TLA features.
    """
    if func_ids is None:
        func_ids = list(range(1, N_FUNCTIONS + 1))

    results = {}
    for size in SAMPLE_SIZES_TLA:
        config_key = f"{strategy}_{size}"
        if config_key not in tla_cv:
            continue

        # Collect CVs grouped by function
        func_cvs = defaultdict(list)
        for func_id in func_ids:
            for inst_id in range(1, N_INSTANCES + 1):
                cvs = _tla_get_raw_cv_values(tla_cv, config_key, func_id, inst_id,
                                              transform, homology)
                if cvs:
                    # Use median CV per instance as the summary
                    func_cvs[func_id].append(np.median(cvs))

        if not func_cvs:
            continue

        all_cvs = []
        within_func_vars = []
        func_medians = []

        for func_id in func_ids:
            vals = func_cvs.get(func_id, [])
            if len(vals) < 2:
                continue
            all_cvs.extend(vals)
            func_medians.append(np.median(vals))
            within_func_vars.append(np.var(vals))

        if not all_cvs:
            continue

        results[size] = {
            "overall_median_cv": np.median(all_cvs),
            "within_func_var": np.median(within_func_vars) if within_func_vars else 0,
            "between_func_var": np.var(func_medians) if len(func_medians) > 1 else 0,
            "total_var": np.var(all_cvs),
            "n_obs": len(all_cvs),
        }

    return results


# ===========================================================================
# Plotting
# ===========================================================================

def plot_ela_decomposition_by_strategy(ela_cv, func_ids=None, func_label="All",
                                       strategies=None):
    """
    Bar chart: for each strategy × sample size, show stacked within-func and
    between-func variance components.
    """
    if strategies is None:
        strategies = SAMPLING_STRATEGIES

    fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 5),
                             sharey=True)
    if len(strategies) == 1:
        axes = [axes]

    for ax_idx, strategy in enumerate(strategies):
        ax = axes[ax_idx]
        decomp = ela_variance_decomposition(ela_cv, strategy, func_ids)
        if not decomp:
            ax.set_title(f"{STRATEGY_LABELS[strategy]}\n(no data)")
            continue

        sizes = sorted(decomp.keys())
        within_vals = [decomp[s]["within_func_var"] for s in sizes]
        between_vals = [decomp[s]["between_func_var"] for s in sizes]
        median_cvs = [decomp[s]["overall_median_cv"] for s in sizes]

        x = np.arange(len(sizes))
        width = 0.6
        bars_within = ax.bar(x, within_vals, width, label="Within-function var",
                             color="#4ECDC4", edgecolor="white")
        bars_between = ax.bar(x, between_vals, width, bottom=within_vals,
                              label="Between-function var", color="#FF6B6B",
                              edgecolor="white")

        # Overlay median CV as line on secondary axis
        ax2 = ax.twinx()
        ax2.plot(x, median_cvs, "ko-", linewidth=2, markersize=6, label="Median CV")
        if ax_idx == len(strategies) - 1:
            ax2.set_ylabel("Median CV", fontsize=10)
        ax2.tick_params(axis="y", labelsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}d" for s in sizes], fontsize=10)
        ax.set_xlabel("Sample Size (×d)", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Variance of CV", fontsize=10)
        ax.set_title(STRATEGY_LABELS[strategy], fontsize=12,
                     color=STRATEGY_COLORS[strategy], fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

    # Shared legend
    handles_bars = [
        Patch(facecolor="#4ECDC4", label="Within-function var"),
        Patch(facecolor="#FF6B6B", label="Between-function var"),
        plt.Line2D([0], [0], color="black", marker="o", label="Median CV"),
    ]
    fig.legend(handles=handles_bars, loc="upper center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle(f"ELA Variance Decomposition — {func_label} Functions",
                 fontsize=14, y=1.08)
    plt.tight_layout()
    plt.show()


def plot_ela_decomposition_unimodal_vs_multimodal(ela_cv, strategies=None):
    """
    Side-by-side: unimodal (F1-14) vs multimodal (F15-24) for each strategy.
    Focused comparison to isolate the multimodality effect.
    """
    if strategies is None:
        strategies = SAMPLING_STRATEGIES

    fig, axes = plt.subplots(2, len(strategies),
                             figsize=(4.5 * len(strategies), 9),
                             sharey="row")
    if len(strategies) == 1:
        axes = axes.reshape(-1, 1)

    for col, strategy in enumerate(strategies):
        for row, (func_ids, label) in enumerate([
            (UNIMODAL_FUNCS, "Unimodal (F1–F14)"),
            (MULTIMODAL_FUNCS, "Multimodal (F15–F24)"),
        ]):
            ax = axes[row, col]
            decomp = ela_variance_decomposition(ela_cv, strategy, func_ids)
            if not decomp:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            sizes = sorted(decomp.keys())
            within_vals = [decomp[s]["within_func_var"] for s in sizes]
            between_vals = [decomp[s]["between_func_var"] for s in sizes]
            total_vals = [decomp[s]["total_var"] for s in sizes]

            x = np.arange(len(sizes))
            width = 0.6
            ax.bar(x, within_vals, width, label="Within-func",
                   color="#4ECDC4", edgecolor="white")
            ax.bar(x, between_vals, width, bottom=within_vals,
                   label="Between-func", color="#FF6B6B", edgecolor="white")

            # Show proportion text
            for i, s in enumerate(sizes):
                total = within_vals[i] + between_vals[i]
                if total > 0:
                    pct_between = between_vals[i] / total * 100
                    ax.text(i, total + 0.001, f"{pct_between:.0f}%",
                            ha="center", va="bottom", fontsize=8,
                            fontweight="bold", color="#FF6B6B")

            ax.set_xticks(x)
            ax.set_xticklabels([f"{s}d" for s in sizes], fontsize=9)
            if row == 1:
                ax.set_xlabel("Sample Size (×d)", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Variance of CV\n({label})", fontsize=10)
            if row == 0:
                ax.set_title(STRATEGY_LABELS[strategy], fontsize=12,
                             color=STRATEGY_COLORS[strategy], fontweight="bold")
            ax.grid(True, alpha=0.2, axis="y")

    handles = [
        Patch(facecolor="#4ECDC4", label="Within-function variance"),
        Patch(facecolor="#FF6B6B", label="Between-function variance"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, 1.01), frameon=True)
    fig.suptitle("ELA Variance Decomposition — Unimodal vs Multimodal",
                 fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()


def plot_ela_median_cv_trend_comparison(ela_cv, strategies=None):
    """
    Line plot: median CV over sample size, split by unimodal/multimodal,
    one line per strategy. Shows the divergence pattern.
    """
    if strategies is None:
        strategies = SAMPLING_STRATEGIES

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (func_ids, label) in zip(axes, [
        (UNIMODAL_FUNCS, "Unimodal (F1–F14)"),
        (MULTIMODAL_FUNCS, "Multimodal (F15–F24)"),
    ]):
        for strategy in strategies:
            decomp = ela_variance_decomposition(ela_cv, strategy, func_ids)
            if not decomp:
                continue
            sizes = sorted(decomp.keys())
            medians = [decomp[s]["overall_median_cv"] for s in sizes]
            ax.plot(sizes, medians, marker="o", linewidth=2, markersize=6,
                    color=STRATEGY_COLORS[strategy],
                    label=STRATEGY_LABELS[strategy])
        ax.set_xlabel("Sample Size (×d)", fontsize=12)
        ax.set_ylabel("Median CV", fontsize=12)
        ax.set_title(label, fontsize=13)
        ax.set_xticks(SAMPLE_SIZES_ELA)
        ax.set_xticklabels([f"{s}d" for s in SAMPLE_SIZES_ELA])
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Median CV Trend — Unimodal vs Multimodal", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# TLA decomposition plots
# ---------------------------------------------------------------------------

def plot_tla_decomposition_by_strategy(tla_cv, transform=None, homology=None,
                                        func_ids=None, func_label="All",
                                        strategies=None):
    """Bar chart decomposition for TLA features."""
    if strategies is None:
        strategies = SAMPLING_STRATEGIES

    seg_parts = []
    if transform:
        seg_parts.append(transform.title())
    if homology:
        seg_parts.append(homology.upper())
    seg_label = " ".join(seg_parts) if seg_parts else "All"

    fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 5),
                             sharey=True)
    if len(strategies) == 1:
        axes = [axes]

    for ax_idx, strategy in enumerate(strategies):
        ax = axes[ax_idx]
        decomp = tla_variance_decomposition(tla_cv, strategy, func_ids,
                                             transform, homology)
        if not decomp:
            ax.set_title(f"{STRATEGY_LABELS[strategy]}\n(no data)")
            continue

        sizes = sorted(decomp.keys())
        within_vals = [decomp[s]["within_func_var"] for s in sizes]
        between_vals = [decomp[s]["between_func_var"] for s in sizes]
        median_cvs = [decomp[s]["overall_median_cv"] for s in sizes]

        x = np.arange(len(sizes))
        width = 0.6
        ax.bar(x, within_vals, width, label="Within-function var",
               color="#4ECDC4", edgecolor="white")
        ax.bar(x, between_vals, width, bottom=within_vals,
               label="Between-function var", color="#FF6B6B", edgecolor="white")

        ax2 = ax.twinx()
        ax2.plot(x, median_cvs, "ko-", linewidth=2, markersize=6, label="Median CV")
        if ax_idx == len(strategies) - 1:
            ax2.set_ylabel("Median CV", fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}d" for s in sizes], fontsize=10)
        ax.set_xlabel("Sample Size (×d)", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Variance of CV", fontsize=10)
        ax.set_title(STRATEGY_LABELS[strategy], fontsize=12,
                     color=STRATEGY_COLORS[strategy], fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

    handles_bars = [
        Patch(facecolor="#4ECDC4", label="Within-function var"),
        Patch(facecolor="#FF6B6B", label="Between-function var"),
        plt.Line2D([0], [0], color="black", marker="o", label="Median CV"),
    ]
    fig.legend(handles=handles_bars, loc="upper center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle(f"TLA Variance Decomposition — {func_label} Functions ({seg_label})",
                 fontsize=14, y=1.08)
    plt.tight_layout()
    plt.show()


def plot_tla_decomposition_unimodal_vs_multimodal(tla_cv, transform=None,
                                                    homology=None, strategies=None):
    """Side-by-side unimodal vs multimodal for TLA."""
    if strategies is None:
        strategies = SAMPLING_STRATEGIES

    seg_parts = []
    if transform:
        seg_parts.append(transform.title())
    if homology:
        seg_parts.append(homology.upper())
    seg_label = " ".join(seg_parts) if seg_parts else "All"

    fig, axes = plt.subplots(2, len(strategies),
                             figsize=(4.5 * len(strategies), 9),
                             sharey="row")
    if len(strategies) == 1:
        axes = axes.reshape(-1, 1)

    for col, strategy in enumerate(strategies):
        for row, (func_ids, label) in enumerate([
            (UNIMODAL_FUNCS, "Unimodal (F1–F14)"),
            (MULTIMODAL_FUNCS, "Multimodal (F15–F24)"),
        ]):
            ax = axes[row, col]
            decomp = tla_variance_decomposition(tla_cv, strategy, func_ids,
                                                 transform, homology)
            if not decomp:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            sizes = sorted(decomp.keys())
            within_vals = [decomp[s]["within_func_var"] for s in sizes]
            between_vals = [decomp[s]["between_func_var"] for s in sizes]

            x = np.arange(len(sizes))
            width = 0.6
            ax.bar(x, within_vals, width, color="#4ECDC4", edgecolor="white")
            ax.bar(x, between_vals, width, bottom=within_vals,
                   color="#FF6B6B", edgecolor="white")

            for i, s in enumerate(sizes):
                total = within_vals[i] + between_vals[i]
                if total > 0:
                    pct_between = between_vals[i] / total * 100
                    ax.text(i, total + 0.001, f"{pct_between:.0f}%",
                            ha="center", va="bottom", fontsize=8,
                            fontweight="bold", color="#FF6B6B")

            ax.set_xticks(x)
            ax.set_xticklabels([f"{s}d" for s in sizes], fontsize=9)
            if row == 1:
                ax.set_xlabel("Sample Size (×d)", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Variance of CV\n({label})", fontsize=10)
            if row == 0:
                ax.set_title(STRATEGY_LABELS[strategy], fontsize=12,
                             color=STRATEGY_COLORS[strategy], fontweight="bold")
            ax.grid(True, alpha=0.2, axis="y")

    handles = [
        Patch(facecolor="#4ECDC4", label="Within-function variance"),
        Patch(facecolor="#FF6B6B", label="Between-function variance"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, 1.01), frameon=True)
    fig.suptitle(f"TLA Variance Decomposition — Unimodal vs Multimodal ({seg_label})",
                 fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Per-function-group detailed decomposition
# ---------------------------------------------------------------------------

def plot_decomposition_per_function_group(ela_cv, strategy="cma_random"):
    """
    One panel per BBOB function group, showing how within/between variance
    evolves with sample size for a single strategy (default: CMA-Random).
    """
    n_groups = len(FUNCTION_GROUPS)
    fig, axes = plt.subplots(1, n_groups, figsize=(4.5 * n_groups, 5), sharey=True)

    for idx, (group_name, func_ids) in enumerate(FUNCTION_GROUPS.items()):
        ax = axes[idx]
        decomp = ela_variance_decomposition(ela_cv, strategy, func_ids)
        if not decomp:
            ax.set_title(f"{group_name}\n(no data)")
            continue

        sizes = sorted(decomp.keys())
        within_vals = [decomp[s]["within_func_var"] for s in sizes]
        between_vals = [decomp[s]["between_func_var"] for s in sizes]

        x = np.arange(len(sizes))
        width = 0.6
        ax.bar(x, within_vals, width, color="#4ECDC4", edgecolor="white",
               label="Within-func")
        ax.bar(x, between_vals, width, bottom=within_vals, color="#FF6B6B",
               edgecolor="white", label="Between-func")

        for i in range(len(sizes)):
            total = within_vals[i] + between_vals[i]
            if total > 0:
                pct = between_vals[i] / total * 100
                ax.text(i, total + 0.0005, f"{pct:.0f}%", ha="center",
                        va="bottom", fontsize=8, fontweight="bold",
                        color="#FF6B6B")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}d" for s in sizes], fontsize=9)
        ax.set_xlabel("Sample Size (×d)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Variance of CV", fontsize=10)
        ax.set_title(group_name, fontsize=11)
        ax.grid(True, alpha=0.2, axis="y")

    handles = [
        Patch(facecolor="#4ECDC4", label="Within-function variance"),
        Patch(facecolor="#FF6B6B", label="Between-function variance"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle(
        f"ELA Variance Decomposition per Function Group — {STRATEGY_LABELS[strategy]}",
        fontsize=14, y=1.07)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Summary table (text)
# ---------------------------------------------------------------------------

def print_decomposition_summary(ela_cv=None, tla_cv=None, homology="h0",
                                 strategies=None):
    """Print a summary table of the decomposition results."""
    if strategies is None:
        strategies = SAMPLING_STRATEGIES

    print(f"\n{'='*80}")
    print("VARIANCE DECOMPOSITION SUMMARY")
    print(f"{'='*80}")

    for feat_type, cv_data, sample_sizes in [
        ("ELA", ela_cv, SAMPLE_SIZES_ELA),
        ("TLA", tla_cv, SAMPLE_SIZES_TLA),
    ]:
        if cv_data is None:
            continue
        print(f"\n--- {feat_type} (homology={homology if feat_type=='TLA' else 'N/A'}) ---")
        print(f"{'Strategy':<15} {'Size':<8} {'Median CV':<10} "
              f"{'Within-F Var':<14} {'Between-F Var':<15} {'% Between':<12}")
        print("-" * 74)

        for strategy in strategies:
            if feat_type == "ELA":
                decomp = ela_variance_decomposition(cv_data, strategy)
            else:
                decomp = tla_variance_decomposition(cv_data, strategy,
                                                     homology=homology)
            for size in sample_sizes:
                if size not in decomp:
                    continue
                d = decomp[size]
                total = d["within_func_var"] + d["between_func_var"]
                pct = (d["between_func_var"] / total * 100) if total > 0 else 0
                print(f"{STRATEGY_LABELS[strategy]:<15} {size}d"
                      f"{'':>4} {d['overall_median_cv']:<10.4f} "
                      f"{d['within_func_var']:<14.6f} "
                      f"{d['between_func_var']:<15.6f} {pct:<12.1f}")
        print()
    print("=" * 80)


# ===========================================================================
# Run-all convenience functions
# ===========================================================================

def run_ela_variance_decomposition(ela_cv, strategies=None):
    """Run all ELA variance decomposition analyses."""
    print("=" * 60 + "\nELA VARIANCE DECOMPOSITION\n" + "=" * 60)

    # 1. Per-strategy decomposition (all functions)
    plot_ela_decomposition_by_strategy(ela_cv, strategies=strategies)

    # 2. Unimodal vs multimodal comparison
    plot_ela_decomposition_unimodal_vs_multimodal(ela_cv, strategies=strategies)

    # 3. Median CV trend comparison
    plot_ela_median_cv_trend_comparison(ela_cv, strategies=strategies)

    # 4. Per function group for CMA-Random
    plot_decomposition_per_function_group(ela_cv, strategy="cma_random")

    # 5. Summary table
    print_decomposition_summary(ela_cv, strategies=strategies)


def run_tla_variance_decomposition(tla_cv, homology="h0", strategies=None):
    """Run all TLA variance decomposition analyses for a given homology."""
    print("=" * 60 + f"\nTLA VARIANCE DECOMPOSITION (homology={homology})\n" + "=" * 60)

    # 1. Per-strategy decomposition (all functions)
    for transform in [None, "volume", "axis"]:
        t_label = transform.title() if transform else "All"
        plot_tla_decomposition_by_strategy(
            tla_cv, transform=transform, homology=homology,
            func_label="All", strategies=strategies)

    # 2. Unimodal vs multimodal for volume and axis
    for transform in ["volume", "axis"]:
        plot_tla_decomposition_unimodal_vs_multimodal(
            tla_cv, transform=transform, homology=homology,
            strategies=strategies)

    # 3. Summary table
    print_decomposition_summary(tla_cv=tla_cv, homology=homology,
                                 strategies=strategies)


def run_all_variance_decomposition(ela_cv, tla_cv, homology="h0",
                                    strategies=None):
    """Run complete variance decomposition for both ELA and TLA."""
    run_ela_variance_decomposition(ela_cv, strategies=strategies)
    print("\n\n")
    run_tla_variance_decomposition(tla_cv, homology=homology,
                                    strategies=strategies)
    print("\n\nCombined summary:")
    print_decomposition_summary(ela_cv, tla_cv, homology=homology,
                                 strategies=strategies)