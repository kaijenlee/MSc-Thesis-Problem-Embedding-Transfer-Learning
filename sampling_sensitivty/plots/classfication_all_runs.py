"""
Plotting functions for ELA allruns classification results.

Usage in notebook:
    from ela_plots import (
        load_results,
        plot_accuracy_median,
        plot_accuracy_allruns,
        plot_consistency_boxplots,
        plot_per_class_heatmap,
        plot_strategy_size_heatmap,
    )

    results = load_results("ela_classification_results_allruns.h5")
    plot_accuracy_median(results)
    plot_accuracy_allruns(results)
    plot_consistency_boxplots(results)
    plot_per_class_heatmap(results, metric="median")
    plot_strategy_size_heatmap(results, metric="median")
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


N_FUNCTIONS = 24
N_RUNS = 30

STRATEGIES = ["cma_random", "ilhs", "lhs", "sobol", "uniform"]
SIZES = [25, 50, 75, 100]

STRATEGY_COLORS = {
    "cma_random": "#1f77b4",
    "ilhs":       "#ff7f0e",
    "lhs":        "#2ca02c",
    "sobol":      "#d62728",
    "uniform":    "#9467bd",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def parse_config_key(config_key):
    """'cma_random_50' -> ('cma_random', 50). 'ilhs_25' -> ('ilhs', 25)."""
    parts = config_key.rsplit("_", 1)
    return parts[0], int(parts[1])


def load_results(h5_path):
    """
    Load all configs from the results file into a dict.

    Returns
    -------
    dict mapping config_key -> dict of arrays/scalars
    """
    results = {}
    with h5py.File(h5_path, "r") as f:
        for config_key in f.keys():
            grp = f[config_key]
            entry = {
                "fold_accuracies_median":   grp["fold_accuracies_median"][:],
                "fold_accuracies_all_runs": grp["fold_accuracies_all_runs"][:],
                "fold_consistency":         grp["fold_consistency"][:],
                "per_instance_consistency": grp["per_instance_consistency"][:],
                "true_labels":              grp["true_labels"][:],
                "pred_median":              grp["pred_median"][:],
                "pred_runs":                grp["pred_runs"][:],
                "pred_majority_vote":       grp["pred_majority_vote"][:],
            }
            for k, v in grp.attrs.items():
                entry[k] = v
            results[config_key] = entry
    return results


def sort_configs(config_keys):
    """Sort by (strategy, size) using STRATEGIES order."""
    def key_fn(c):
        strat, size = parse_config_key(c)
        strat_idx = STRATEGIES.index(strat) if strat in STRATEGIES else 999
        return (strat_idx, size)
    return sorted(config_keys, key=key_fn)


# ---------------------------------------------------------------------------
# Accuracy bar charts
# ---------------------------------------------------------------------------

def _accuracy_bar(results, metric_key, title, ylabel, ax=None):
    """Shared bar-chart code for either median or all-runs accuracy."""
    config_keys = sort_configs(list(results.keys()))
    accs = [float(results[c][metric_key]) for c in config_keys]
    colors = [STRATEGY_COLORS.get(parse_config_key(c)[0], "gray")
              for c in config_keys]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(config_keys)), 5))

    x = np.arange(len(config_keys))
    bars = ax.bar(x, accs, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(config_keys, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(1 / N_FUNCTIONS, color="red", linestyle="--",
               linewidth=1, label=f"chance ({1/N_FUNCTIONS:.3f})")

    # Legend for strategies
    handles = [plt.Rectangle((0, 0), 1, 1, color=STRATEGY_COLORS[s])
               for s in STRATEGIES if s in {parse_config_key(c)[0] for c in config_keys}]
    labels = [s for s in STRATEGIES
              if s in {parse_config_key(c)[0] for c in config_keys}]
    handles.append(plt.Line2D([0], [0], color="red", linestyle="--"))
    labels.append("chance")
    ax.legend(handles, labels, loc="lower right", fontsize=8, framealpha=0.9)

    # Annotate values on bars
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    return ax


def plot_accuracy_median(results, ax=None):
    """Bar chart of overall accuracy when testing on median features."""
    return _accuracy_bar(
        results,
        metric_key="overall_accuracy_median",
        title="Classification accuracy — testing on median features",
        ylabel="Accuracy",
        ax=ax,
    )


def plot_accuracy_allruns(results, ax=None):
    """Bar chart of overall accuracy when testing on all 30 individual runs."""
    return _accuracy_bar(
        results,
        metric_key="overall_accuracy_all_runs",
        title="Classification accuracy — testing on all individual runs",
        ylabel="Accuracy (averaged over runs)",
        ax=ax,
    )


# ---------------------------------------------------------------------------
# Per-instance consistency boxplots
# ---------------------------------------------------------------------------

def plot_consistency_boxplots(results, ax=None):
    """
    Boxplot of per-instance consistency (fraction of 30 runs predicted
    correctly) for each config.
    """
    config_keys = sort_configs(list(results.keys()))
    data = [results[c]["per_instance_consistency"] for c in config_keys]
    colors = [STRATEGY_COLORS.get(parse_config_key(c)[0], "gray")
              for c in config_keys]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(config_keys)), 5))

    bp = ax.boxplot(data, patch_artist=True, showfliers=True,
                    medianprops=dict(color="black"),
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(np.arange(1, len(config_keys) + 1))
    ax.set_xticklabels(config_keys, rotation=45, ha="right")
    ax.set_ylabel("Per-instance consistency")
    ax.set_title("Per-instance consistency across the 30 runs")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Per-class (BBOB function) accuracy heatmap
# ---------------------------------------------------------------------------

def per_class_accuracy(true_labels, predictions):
    """Accuracy per class (0..N_FUNCTIONS-1)."""
    accs = np.full(N_FUNCTIONS, np.nan)
    for c in range(N_FUNCTIONS):
        mask = true_labels == c
        if mask.any():
            accs[c] = (predictions[mask] == c).mean()
    return accs


def plot_per_class_heatmap(results, metric="median", ax=None):
    """
    Heatmap: rows = configs, columns = BBOB function (1..24),
    values = per-class accuracy.

    metric : "median" uses pred_median; "all_runs" uses pred_runs (flattened);
             "majority_vote" uses pred_majority_vote.
    """
    config_keys = sort_configs(list(results.keys()))
    matrix = np.zeros((len(config_keys), N_FUNCTIONS))

    for i, c in enumerate(config_keys):
        entry = results[c]
        y_true = entry["true_labels"]
        if metric == "median":
            y_pred = entry["pred_median"]
        elif metric == "majority_vote":
            y_pred = entry["pred_majority_vote"]
        elif metric == "all_runs":
            y_true = np.repeat(y_true, N_RUNS)
            y_pred = entry["pred_runs"].flatten()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        matrix[i] = per_class_accuracy(y_true, y_pred)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, max(5, 0.3 * len(config_keys))))

    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(N_FUNCTIONS))
    ax.set_xticklabels([f"f{i + 1}" for i in range(N_FUNCTIONS)], fontsize=8)
    ax.set_yticks(np.arange(len(config_keys)))
    ax.set_yticklabels(config_keys, fontsize=8)
    ax.set_xlabel("BBOB function")
    ax.set_title(f"Per-class accuracy ({metric})")

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Accuracy")

    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Strategy × size heatmap
# ---------------------------------------------------------------------------

def plot_strategy_size_heatmap(results, metric="median", ax=None,
                                strategies=None, sizes=None):
    """
    Heatmap: rows = sampling strategy, columns = sample size,
    values = overall accuracy.

    metric : "median" -> overall_accuracy_median
             "all_runs" -> overall_accuracy_all_runs
             "majority_vote" -> overall_accuracy_majority_vote
    """
    if metric == "median":
        attr = "overall_accuracy_median"
        title = "Accuracy (median test) by strategy × size"
    elif metric == "all_runs":
        attr = "overall_accuracy_all_runs"
        title = "Accuracy (all runs test) by strategy × size"
    elif metric == "majority_vote":
        attr = "overall_accuracy_majority_vote"
        title = "Accuracy (majority vote) by strategy × size"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Auto-detect available strategies/sizes
    present = [parse_config_key(c) for c in results.keys()]
    if strategies is None:
        strategies = [s for s in STRATEGIES if any(p[0] == s for p in present)]
    if sizes is None:
        sizes = sorted({p[1] for p in present})

    matrix = np.full((len(strategies), len(sizes)), np.nan)
    for i, strat in enumerate(strategies):
        for j, size in enumerate(sizes):
            key = f"{strat}_{size}"
            if key in results:
                matrix[i, j] = float(results[key][attr])

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.2 * len(sizes) + 2,
                                          0.6 * len(strategies) + 2))

    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_yticklabels(strategies)
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Sampling strategy")
    ax.set_title(title)

    # Annotate cells
    for i in range(len(strategies)):
        for j in range(len(sizes)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Accuracy")

    plt.tight_layout()
    return ax