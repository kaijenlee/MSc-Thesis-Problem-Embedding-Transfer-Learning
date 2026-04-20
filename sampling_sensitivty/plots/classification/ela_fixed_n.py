"""
Bar-plot view of the fixed-N experiments.

For a single sampling strategy, plot a 2x4 grid of bar charts:
  - 4 columns: sample sizes [25d, 50d, 75d, 100d]
  - 2 rows:    top = acc_median, bottom = acc_allruns
  - Each subplot has one cluster of bars per N value.
  - Within a cluster, bars are ordered from most-instances/fewest-runs
    on the left to fewest-instances/most-runs on the right.
  - Each bar is annotated with its "inst x runs" pair.

Aggregation: mean of per-fold means (Option 2), no error bars.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


SAMPLE_SIZES = [25, 50, 75, 100]


def aggregate_cell(acc_array, pair_idx):
    """Mean of per-fold means for one cell."""
    return acc_array[pair_idx].mean(axis=1).mean()


def collect_n_data(h5_path, sampler, sample_size):
    """
    For one config (sampler_size), return the per-pair accuracy in the
    order they were stored in the file (which matches the bash script's
    left-to-right ordering: most instances -> fewest instances).

    Returns
    -------
    inst : np.ndarray (n_pairs,)
    runs : np.ndarray (n_pairs,)
    acc_median  : np.ndarray (n_pairs,)
    acc_allruns : np.ndarray (n_pairs,)
    """
    config_key = f"{sampler}_{sample_size}"
    with h5py.File(h5_path, "r") as f:
        if config_key not in f:
            return None
        grp = f[config_key]
        inst = grp["instance_counts"][:]
        runs = grp["run_counts"][:]
        med = grp["acc_median"][:]
        run = grp["acc_allruns"][:]

    n_pairs = len(inst)
    acc_med = np.array([aggregate_cell(med, i) for i in range(n_pairs)])
    acc_run = np.array([aggregate_cell(run, i) for i in range(n_pairs)])
    return inst, runs, acc_med, acc_run


def plot_fixed_n_bars(
    n_to_h5,
    sampler,
    sample_sizes=SAMPLE_SIZES,
    figsize=(20, 9),
    ylim_median=None,
    ylim_allruns=None,
    cluster_width=0.8,
    annotate_fontsize=7,
    save_path=None,
    show=False,
):
    """
    Parameters
    ----------
    n_to_h5 : dict[int, str or Path]
        Mapping from N (total rows per class) to the h5 file produced by
        the corresponding pairs-mode run. E.g.
            {6:  "ela_learning_N6.h5",
             12: "ela_learning_N12.h5",
             24: "ela_learning_N24.h5",
             60: "ela_learning_N60.h5"}
        Order of insertion = order of clusters along the x-axis.
    sampler : str
        Which sampling strategy to plot (e.g. "ilhs", "lhs", "sobol",
        "uniform", "cma_random").
    sample_sizes : list of int
        One per column.
    figsize : tuple
    ylim_median, ylim_allruns : tuple or None
        Y-axis limits for the two rows.
    cluster_width : float
        Total horizontal span of one cluster (in x-axis units, where
        cluster centers are at integer positions).
    annotate_fontsize : int
    save_path : str or None
    show : bool
    """
    n_values = list(n_to_h5.keys())
    n_clusters = len(n_values)
    n_rows, n_cols = 2, len(sample_sizes)

    # Pre-load data for every (N, sample_size) combination
    # data[n][size] -> (inst, runs, acc_med, acc_run)  or None
    data = {}
    for n, h5 in n_to_h5.items():
        data[n] = {}
        for size in sample_sizes:
            data[n][size] = collect_n_data(h5, sampler, size)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             sharex=True, sharey="row")
    row_keys = ["median", "allruns"]
    row_labels = ["Median features", "All runs"]

    cluster_centers = np.arange(n_clusters)

    for col, size in enumerate(sample_sizes):
        for row, key in enumerate(row_keys):
            ax = axes[row, col]

            for ci, n in enumerate(n_values):
                entry = data[n][size]
                if entry is None:
                    continue
                inst, runs, acc_med, acc_run = entry
                values = acc_med if key == "median" else acc_run
                n_bars = len(values)
                if n_bars == 0:
                    continue

                # Bar positions within this cluster, evenly spaced
                bar_w = cluster_width / n_bars
                offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_w
                xs = cluster_centers[ci] + offsets

                # Color bars within a cluster on a gradient so the
                # left->right progression is visually obvious.
                cmap = plt.get_cmap("viridis")
                colors = [cmap(j / max(n_bars - 1, 1)) for j in range(n_bars)]

                ax.bar(xs, values, width=bar_w * 0.9,
                       color=colors, edgecolor="black", linewidth=0.4)

                # Annotate each bar with its (inst x runs) pair
                for x, v, i_val, r_val in zip(xs, values, inst, runs):
                    ax.text(
                        x, v + 0.005,
                        f"{int(i_val)}×{int(r_val)}",
                        ha="center", va="bottom",
                        rotation=90, fontsize=annotate_fontsize,
                    )

    # Cosmetic touches
    for col, size in enumerate(sample_sizes):
        axes[0, col].set_title(f"{size}d samples", fontsize=12)
        axes[-1, col].set_xlabel("N (rows per class)")
        axes[-1, col].set_xticks(cluster_centers)
        axes[-1, col].set_xticklabels([f"N={n}" for n in n_values])

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(f"{label}\nAccuracy")

    for ax in axes.flat:
        ax.grid(True, axis="y", alpha=0.3)

    if ylim_median is not None:
        axes[0, 0].set_ylim(ylim_median)
    if ylim_allruns is not None:
        axes[1, 0].set_ylim(ylim_allruns)

    fig.suptitle(
        f"Fixed-N learning curves — sampler: {sampler}",
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
    parser.add_argument("--sampler", type=str, required=True)
    parser.add_argument("--n-files", nargs="+", required=True,
                        help="Pairs of N:path, e.g. 6:ela_learning_N6.h5 "
                             "12:ela_learning_N12.h5 ...")
    parser.add_argument("--ylim-median", nargs=2, type=float, default=None)
    parser.add_argument("--ylim-allruns", nargs=2, type=float, default=None)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    n_to_h5 = {}
    for entry in args.n_files:
        n_str, path = entry.split(":", 1)
        n_to_h5[int(n_str)] = path

    plot_fixed_n_bars(
        n_to_h5=n_to_h5,
        sampler=args.sampler,
        ylim_median=tuple(args.ylim_median) if args.ylim_median else None,
        ylim_allruns=tuple(args.ylim_allruns) if args.ylim_allruns else None,
        save_path=args.save,
        show=True,
    )