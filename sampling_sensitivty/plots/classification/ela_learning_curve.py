"""
Plot ELA learning curve results.

For a fixed instance_count, plot accuracy vs run_count as a 2x4 grid:
  - 4 columns: sample sizes [25d, 50d, 75d, 100d]
  - 2 rows:    top = acc_median, bottom = acc_allruns
  - One line per sampling strategy in each subplot.

Aggregation (Option 2): mean of per-fold means across n_repeats,
                         SEM across the 5 folds as error bars.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt


SAMPLERS = ["cma_random", "ilhs", "lhs", "sobol", "uniform"]
SAMPLE_SIZES = [25, 50, 75, 100]
SAMPLER_COLORS = {
    "cma_random": "#1f77b4",
    "ilhs":       "#ff7f0e",
    "lhs":        "#2ca02c",
    "sobol":      "#d62728",
    "uniform":    "#9467bd",
}
SAMPLER_LABELS = {
    "cma_random": "CMA-random",
    "ilhs":       "iLHS",
    "lhs":        "LHS",
    "sobol":      "Sobol",
    "uniform":    "Uniform",
}


def _cell_stats(fold_rep_block):
    """
    Option 2 aggregation on a (n_folds, n_repeats) block.
    Returns (mean, sem).
    """
    fold_means = fold_rep_block.mean(axis=1)              # (n_folds,)
    return fold_means.mean(), fold_means.std(ddof=1) / np.sqrt(len(fold_means))


def _locate_cell(acc_array, inst_axis, runs_axis, instance_count, run_count):
    """
    Return the (n_folds, n_repeats) block for a given (instance, run) cell,
    or None if the cell isn't present. Handles both storage layouts:

    - 4-D grid:  shape (n_instances, n_runs, n_folds, n_repeats)
                 with inst_axis/runs_axis as 1-D axis coordinates.
    - 3-D flat:  shape (n_pairs, n_folds, n_repeats)
                 with inst_axis/runs_axis as length-n_pairs parallel arrays.
    """
    if acc_array.ndim == 4:
        i_matches = np.where(inst_axis == instance_count)[0]
        r_matches = np.where(runs_axis == run_count)[0]
        if len(i_matches) == 0 or len(r_matches) == 0:
            return None
        return acc_array[i_matches[0], r_matches[0]]
    elif acc_array.ndim == 3:
        matches = np.where((inst_axis == instance_count) &
                           (runs_axis == run_count))[0]
        if len(matches) == 0:
            return None
        return acc_array[matches[0]]
    else:
        raise ValueError(
            f"Unexpected acc array ndim={acc_array.ndim}, shape={acc_array.shape}"
        )


def collect_curve(h5_group, instance_count, run_counts):
    """
    For a single config group, return (means, sems) arrays over the
    requested run_counts at the given fixed instance_count.

    Returns two dicts: {"median": (means, sems), "allruns": (means, sems)},
    each with arrays of length len(run_counts). NaN where the cell is missing.
    """
    inst = h5_group["instance_counts"][:]
    runs = h5_group["run_counts"][:]
    acc_med = h5_group["acc_median"][:]
    acc_run = h5_group["acc_allruns"][:]

    means_med = np.full(len(run_counts), np.nan)
    sems_med  = np.full(len(run_counts), np.nan)
    means_run = np.full(len(run_counts), np.nan)
    sems_run  = np.full(len(run_counts), np.nan)

    for i, r in enumerate(run_counts):
        block_med = _locate_cell(acc_med, inst, runs, instance_count, r)
        if block_med is not None:
            means_med[i], sems_med[i] = _cell_stats(block_med)

        block_run = _locate_cell(acc_run, inst, runs, instance_count, r)
        if block_run is not None:
            means_run[i], sems_run[i] = _cell_stats(block_run)

    return {
        "median":  (means_med, sems_med),
        "allruns": (means_run, sems_run),
    }


def plot_learning_curve(
    h5_path,
    instance_count,
    run_counts,
    samplers=SAMPLERS,
    sample_sizes=SAMPLE_SIZES,
    figsize=(18, 8),
    ylim_median=None,
    ylim_allruns=None,
    save_path=None,
    show=False,
):
    """
    Build the 2x4 subplot grid.

    Parameters
    ----------
    h5_path : str or Path
    instance_count : int
        Fixed instance count to slice on.
    run_counts : list of int
        x-axis values (must exist as columns in the stored grid).
    samplers : list of str
        Sampler prefixes to include (one line each).
    sample_sizes : list of int
        Sample-size budgets, one per column. Length should be 4.
    figsize : tuple
    ylim_median : tuple or None
        (ymin, ymax) for the top row (median features). None = auto.
    ylim_allruns : tuple or None
        (ymin, ymax) for the bottom row (all runs). None = auto.
    save_path : str or None
        If given, save figure to this path.
    show : bool
        If True, call plt.show() at the end. In notebooks the figure
        renders automatically from the return value, so leave this False.
    """
    n_rows, n_cols = 2, len(sample_sizes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             sharex=True, sharey="row")

    row_labels = ["Median features", "All runs"]
    row_keys = ["median", "allruns"]

    with h5py.File(h5_path, "r") as f:
        for col, size in enumerate(sample_sizes):
            for sampler in samplers:
                config_key = f"{sampler}_{size}"
                if config_key not in f:
                    continue
                curves = collect_curve(f[config_key], instance_count, run_counts)

                for row, key in enumerate(row_keys):
                    means, sems = curves[key]
                    ax = axes[row, col]
                    ax.errorbar(
                        run_counts, means, yerr=sems,
                        marker="o", markersize=5, linewidth=1.5,
                        capsize=3, color=SAMPLER_COLORS.get(sampler),
                        label=SAMPLER_LABELS.get(sampler, sampler),
                    )

    # Cosmetic touches
    for col, size in enumerate(sample_sizes):
        axes[0, col].set_title(f"{size}d samples", fontsize=12)
        axes[-1, col].set_xlabel("Number of runs")
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(f"{label}\nAccuracy")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xticks(run_counts)

    # Per-row y-limits (sharey="row" means setting one axis per row is enough)
    if ylim_median is not None:
        axes[0, 0].set_ylim(ylim_median)
    if ylim_allruns is not None:
        axes[1, 0].set_ylim(ylim_allruns)

    # Single shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=len(samplers), bbox_to_anchor=(0.5, -0.02),
                   frameon=False)

    fig.suptitle(
        f"Learning curve at {instance_count} training instances per class",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()

    return fig, axes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=str)
    parser.add_argument("--instance-count", type=int, required=True)
    parser.add_argument("--run-counts", nargs="+", type=int,
                        default=[1, 5, 10, 20, 30])
    parser.add_argument("--ylim-median", nargs=2, type=float, default=None,
                        metavar=("YMIN", "YMAX"))
    parser.add_argument("--ylim-allruns", nargs=2, type=float, default=None,
                        metavar=("YMIN", "YMAX"))
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    plot_learning_curve(
        h5_path=args.h5_path,
        instance_count=args.instance_count,
        run_counts=args.run_counts,
        ylim_median=tuple(args.ylim_median) if args.ylim_median else None,
        ylim_allruns=tuple(args.ylim_allruns) if args.ylim_allruns else None,
        save_path=args.save,
        show=True,
    )