"""
Plot t-SNE embeddings for ELA and TLA features, colored by BBOB function class.

Loads precomputed embeddings from h5 files (produced by process_ela_tsne.py
and process_tla_tsne.py) and provides four plot types:

  1) Single-panel scatter — one configuration, one mode
  2) Grid — rows = sampling strategies, columns = sample sizes
  3) Side-by-side ELA vs TLA — same configuration, same mode
  4) Mean vs all_runs comparison — same configuration, two modes

Usage (in a notebook):
    from plot_tsne import *

    # --- Load data ---
    ela = load_ela_embeddings("path/to/ela_tsne_embeddings.h5")
    tla = load_tla_embeddings("path/to/tla_tsne_embeddings_volume_h0.h5")

    # --- Single panel ---
    plot_single(ela, "ilhs_50", mode="mean")
    plot_single(tla, "ilhs_50", mode="mean", title_prefix="TLA (volume_h0)")

    # --- Grid ---
    plot_grid(ela, mode="mean", title="ELA — Mean")
    plot_grid(tla, mode="mean", title="TLA (volume_h0) — Mean")

    # --- Side-by-side ELA vs TLA ---
    plot_side_by_side(ela, tla, "ilhs_50", mode="mean",
                      labels=("ELA", "TLA (volume_h0)"))

    # --- Mean vs all_runs ---
    plot_mean_vs_allruns(ela, "ilhs_50", title_prefix="ELA")
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_FUNCTIONS = 24
N_INSTANCES = 100
N_RUNS = 30

SAMPLING_STRATEGIES = ["ilhs", "lhs", "sobol", "uniform", "cma_random"]
SAMPLE_SIZES_ELA = [25, 50, 75, 100]
SAMPLE_SIZES_TLA = [10, 25, 50, 75, 100]

STRATEGY_LABELS = {
    "ilhs": "iLHS",
    "lhs": "LHS",
    "sobol": "Sobol",
    "uniform": "Uniform",
    "cma_random": "CMA-Random",
}

FUNCTION_GROUPS = {
    "Separable": [1, 2, 3, 4, 5],
    "Low/Moderate Cond.": [6, 7, 8, 9],
    "High Conditioning": [10, 11, 12, 13, 14],
    "Multimodal (adequate)": [15, 16, 17, 18, 19],
    "Multimodal (weak)": [20, 21, 22, 23, 24],
}

# ---------------------------------------------------------------------------
# Color palette — 24 maximally separated colors
# ---------------------------------------------------------------------------

# Hand-picked palette: 24 colors that are visually distinguishable.
# Built from a combination of tab20, Set1, and manually adjusted hues.
_PALETTE_24 = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    "#e6beff", "#1ce6ff", "#ff34ff", "#008080",
]

FUNCTION_COLORS = {i: _PALETTE_24[i] for i in range(N_FUNCTIONS)}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ela_embeddings(h5_path):
    """
    Load precomputed ELA t-SNE embeddings.

    Returns dict:
        {
            "labels": np.array (2400,),
            "labels_all_runs": np.array (72000,),
            "config_key": {
                "mean": np.array (2400, 2),
                "median": np.array (2400, 2),
                "all_runs": np.array (72000, 2),
            },
            ...
        }
    """
    data = {}
    with h5py.File(h5_path, "r") as f:
        if "labels" in f:
            data["labels"] = f["labels"][:]
        else:
            data["labels"] = np.repeat(np.arange(N_FUNCTIONS), N_INSTANCES)

        if "labels_all_runs" in f:
            data["labels_all_runs"] = f["labels_all_runs"][:]
        else:
            data["labels_all_runs"] = np.repeat(
                np.arange(N_FUNCTIONS), N_INSTANCES * N_RUNS
            )

        for key in f.keys():
            if key.startswith("labels"):
                continue
            grp = f[key]
            data[key] = {}
            for mode in ["mean", "median", "all_runs"]:
                if mode in grp:
                    data[key][mode] = grp[mode][:]
    return data


def load_tla_embeddings(h5_path):
    """
    Load precomputed TLA t-SNE embeddings.
    Same structure as ELA; also reads segment metadata if present.
    """
    data = load_ela_embeddings(h5_path)  # same h5 layout

    # Read segment metadata
    with h5py.File(h5_path, "r") as f:
        if "segment" in f.attrs:
            data["_segment"] = f.attrs["segment"]
        if "segment_length" in f.attrs:
            data["_segment_length"] = int(f.attrs["segment_length"])

    return data


def available_configs(data):
    """List config keys present in the loaded data."""
    return [k for k in data.keys() if not k.startswith(("labels", "_"))]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_embedding(data, config_key, mode):
    """Retrieve embedding and matching labels."""
    if config_key not in data:
        raise KeyError(
            f"Config '{config_key}' not found. "
            f"Available: {available_configs(data)}"
        )
    if mode not in data[config_key]:
        raise KeyError(
            f"Mode '{mode}' not found for config '{config_key}'. "
            f"Available: {list(data[config_key].keys())}"
        )

    emb = data[config_key][mode]
    labels = data["labels_all_runs"] if mode == "all_runs" else data["labels"]
    return emb, labels


def _scatter_tsne(ax, emb, labels, mode="mean", point_size=None, alpha=None,
                  show_legend=False, legend_loc="best", legend_ncol=4,
                  title=None):
    """
    Core scatter plot on a given axes.

    Parameters
    ----------
    ax : matplotlib Axes
    emb : (n, 2) array of t-SNE coordinates
    labels : (n,) array of function indices (0-based)
    mode : 'mean', 'median', or 'all_runs' — controls default size/alpha
    point_size : override marker size
    alpha : override transparency
    show_legend : whether to add a legend
    title : axes title
    """
    n = len(labels)

    # Defaults that scale with point count
    if point_size is None:
        point_size = 4 if n > 10000 else (10 if n > 3000 else 15)
    if alpha is None:
        alpha = 0.25 if n > 10000 else (0.5 if n > 3000 else 0.7)

    # Color array
    colors = [FUNCTION_COLORS[l] for l in labels]

    # Plot in shuffled order so no class systematically covers another
    rng = np.random.RandomState(42)
    order = rng.permutation(n)
    ax.scatter(
        emb[order, 0], emb[order, 1],
        c=[colors[i] for i in order],
        s=point_size, alpha=alpha, edgecolors="none", rasterized=True,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        ax.set_title(title, fontsize=10, pad=4)

    if show_legend:
        _add_legend(ax, loc=legend_loc, ncol=legend_ncol)


def _add_legend(ax, loc="best", ncol=4, fontsize=7, markersize=4):
    """Add function-class legend."""
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=FUNCTION_COLORS[i], markersize=markersize,
               label=f"F{i + 1}")
        for i in range(N_FUNCTIONS)
    ]
    ax.legend(
        handles=handles, loc=loc, ncol=ncol, fontsize=fontsize,
        framealpha=0.8, handletextpad=0.2, columnspacing=0.8,
        borderpad=0.3, labelspacing=0.3,
    )


def _parse_config(config_key):
    """Parse 'ilhs_50' → ('ilhs', 50)."""
    parts = config_key.rsplit("_", 1)
    return parts[0], int(parts[1])


# ---------------------------------------------------------------------------
# Plot 1: Single-panel scatter
# ---------------------------------------------------------------------------

def plot_single(data, config_key, mode="mean", title_prefix=None,
                point_size=None, alpha=None, figsize=(7, 6),
                show_legend=True, legend_ncol=6, save_path=None):
    """
    Single t-SNE scatter plot for one configuration and mode.

    Parameters
    ----------
    data : dict from load_ela_embeddings or load_tla_embeddings
    config_key : e.g. 'ilhs_50'
    mode : 'mean', 'median', or 'all_runs'
    title_prefix : e.g. 'ELA' or 'TLA (volume_h0)'
    """
    emb, labels = _get_embedding(data, config_key, mode)
    strategy, size = _parse_config(config_key)

    if title_prefix is None:
        title_prefix = ""
    mode_label = mode.replace("_", " ").title()
    title = f"{title_prefix} — {STRATEGY_LABELS.get(strategy, strategy)}, "
    title += f"n={size}d — {mode_label}"

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _scatter_tsne(
        ax, emb, labels, mode=mode,
        point_size=point_size, alpha=alpha,
        show_legend=show_legend, legend_ncol=legend_ncol,
        title=title,
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 2: Grid — strategies × sample sizes
# ---------------------------------------------------------------------------

def plot_grid(data, mode="mean", title=None, strategies=None,
              sample_sizes=None, point_size=None, alpha=None,
              figsize_per_panel=(3, 3), show_legend=True,
              legend_ncol=6, save_path=None):
    """
    Grid of t-SNE plots: rows = sampling strategies, columns = sample sizes.

    Parameters
    ----------
    data : dict from load_ela_embeddings or load_tla_embeddings
    mode : 'mean', 'median', or 'all_runs'
    strategies : list of strategy names (default: all 5)
    sample_sizes : list of ints (default: auto-detect from data)
    """
    if strategies is None:
        strategies = SAMPLING_STRATEGIES
    if sample_sizes is None:
        # Auto-detect from available configs
        all_sizes = set()
        for ck in available_configs(data):
            _, sz = _parse_config(ck)
            all_sizes.add(sz)
        sample_sizes = sorted(all_sizes)

    nrows = len(strategies)
    ncols = len(sample_sizes)
    fw = figsize_per_panel[0] * ncols
    fh = figsize_per_panel[1] * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    if ncols == 1:
        axes = axes[:, np.newaxis]

    for r, strat in enumerate(strategies):
        for c, sz in enumerate(sample_sizes):
            ax = axes[r, c]
            config_key = f"{strat}_{sz}"

            if config_key in data and mode in data.get(config_key, {}):
                emb, labels = _get_embedding(data, config_key, mode)
                _scatter_tsne(ax, emb, labels, mode=mode,
                              point_size=point_size, alpha=alpha)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=12, color="gray", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            # Row labels (left)
            if c == 0:
                ax.set_ylabel(STRATEGY_LABELS.get(strat, strat),
                              fontsize=10, rotation=0, labelpad=50,
                              va="center")

            # Column headers (top)
            if r == 0:
                ax.set_title(f"n = {sz}d", fontsize=10, pad=6)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    # Shared legend at the bottom
    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=FUNCTION_COLORS[i], markersize=5,
                   label=f"F{i + 1}")
            for i in range(N_FUNCTIONS)
        ]
        fig.legend(
            handles=handles, loc="lower center",
            ncol=12, fontsize=7, framealpha=0.8,
            bbox_to_anchor=(0.5, -0.04),
            handletextpad=0.2, columnspacing=0.8,
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 3: Side-by-side ELA vs TLA
# ---------------------------------------------------------------------------

def plot_side_by_side(data_left, data_right, config_key, mode="mean",
                      labels=("ELA", "TLA"), point_size=None, alpha=None,
                      figsize=(12, 5), show_legend=True, legend_ncol=12,
                      save_path=None):
    """
    Two panels: left = data_left (e.g. ELA), right = data_right (e.g. TLA).
    Same config_key and mode for direct comparison.

    Parameters
    ----------
    data_left, data_right : dicts from load_*_embeddings
    config_key : e.g. 'ilhs_50'
    labels : tuple of (left_label, right_label)
    """
    strategy, size = _parse_config(config_key)
    strat_label = STRATEGY_LABELS.get(strategy, strategy)
    mode_label = mode.replace("_", " ").title()

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)

    for ax, d, lbl in [(ax_l, data_left, labels[0]),
                        (ax_r, data_right, labels[1])]:
        if config_key in d and mode in d.get(config_key, {}):
            emb, lab = _get_embedding(d, config_key, mode)
            _scatter_tsne(ax, emb, lab, mode=mode,
                          point_size=point_size, alpha=alpha,
                          title=f"{lbl} — {strat_label}, n={size}d")
        else:
            ax.text(0.5, 0.5, f"{lbl}\nN/A", ha="center", va="center",
                    fontsize=12, color="gray", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"{lbl} — {strat_label}, n={size}d", fontsize=10)

    suptitle = f"ELA vs TLA — {strat_label}, n={size}d — {mode_label}"
    fig.suptitle(suptitle, fontsize=12, y=1.02)

    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=FUNCTION_COLORS[i], markersize=5,
                   label=f"F{i + 1}")
            for i in range(N_FUNCTIONS)
        ]
        fig.legend(
            handles=handles, loc="lower center",
            ncol=legend_ncol, fontsize=7, framealpha=0.8,
            bbox_to_anchor=(0.5, -0.06),
            handletextpad=0.2, columnspacing=0.8,
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_side_by_side_multi(data_dict, config_key, mode="mean",
                            point_size=None, alpha=None,
                            figsize_per_panel=(4, 4), show_legend=True,
                            legend_ncol=12, save_path=None):
    """
    N panels side-by-side for comparing multiple feature types / segments.

    Parameters
    ----------
    data_dict : OrderedDict or dict of {panel_label: data_dict}
        e.g. {"ELA": ela_data, "TLA (all)": tla_all, "TLA (vol_h0)": tla_vh0}
    config_key : e.g. 'ilhs_50'
    mode : 'mean', 'median', or 'all_runs'
    """
    panel_labels = list(data_dict.keys())
    n_panels = len(panel_labels)

    strategy, size = _parse_config(config_key)
    strat_label = STRATEGY_LABELS.get(strategy, strategy)
    mode_label = mode.replace("_", " ").title()

    fw = figsize_per_panel[0] * n_panels
    fh = figsize_per_panel[1]
    fig, axes = plt.subplots(1, n_panels, figsize=(fw, fh))
    if n_panels == 1:
        axes = [axes]

    for ax, lbl in zip(axes, panel_labels):
        d = data_dict[lbl]
        if config_key in d and mode in d.get(config_key, {}):
            emb, lab = _get_embedding(d, config_key, mode)
            _scatter_tsne(ax, emb, lab, mode=mode,
                          point_size=point_size, alpha=alpha,
                          title=f"{lbl}")
        else:
            ax.text(0.5, 0.5, f"{lbl}\nN/A", ha="center", va="center",
                    fontsize=12, color="gray", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(lbl, fontsize=10)

    suptitle = f"{strat_label}, n={size}d — {mode_label}"
    fig.suptitle(suptitle, fontsize=12, y=1.03)

    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=FUNCTION_COLORS[i], markersize=5,
                   label=f"F{i + 1}")
            for i in range(N_FUNCTIONS)
        ]
        fig.legend(
            handles=handles, loc="lower center",
            ncol=legend_ncol, fontsize=7, framealpha=0.8,
            bbox_to_anchor=(0.5, -0.06),
            handletextpad=0.2, columnspacing=0.8,
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 4: Mean vs all_runs comparison
# ---------------------------------------------------------------------------

def plot_mean_vs_allruns(data, config_key, title_prefix="",
                         point_size_mean=None, point_size_all=None,
                         alpha_mean=None, alpha_all=None,
                         figsize=(12, 5), show_legend=True,
                         legend_ncol=12, save_path=None):
    """
    Two panels: left = mean (clean clusters), right = all_runs (noisy clouds).
    Directly visualizes feature stability.

    Parameters
    ----------
    data : dict from load_*_embeddings (must have both 'mean' and 'all_runs')
    config_key : e.g. 'ilhs_50'
    title_prefix : e.g. 'ELA' or 'TLA (volume_h0)'
    """
    strategy, size = _parse_config(config_key)
    strat_label = STRATEGY_LABELS.get(strategy, strategy)

    fig, (ax_mean, ax_all) = plt.subplots(1, 2, figsize=figsize)

    # Mean panel
    if "mean" in data.get(config_key, {}):
        emb_m, lab_m = _get_embedding(data, config_key, "mean")
        _scatter_tsne(ax_mean, emb_m, lab_m, mode="mean",
                      point_size=point_size_mean, alpha=alpha_mean,
                      title="Mean (2,400 points)")
    else:
        ax_mean.text(0.5, 0.5, "Mean\nN/A", ha="center", va="center",
                     fontsize=12, color="gray", transform=ax_mean.transAxes)
        ax_mean.set_xticks([])
        ax_mean.set_yticks([])

    # All runs panel
    if "all_runs" in data.get(config_key, {}):
        emb_a, lab_a = _get_embedding(data, config_key, "all_runs")
        _scatter_tsne(ax_all, emb_a, lab_a, mode="all_runs",
                      point_size=point_size_all, alpha=alpha_all,
                      title="All Runs (72,000 points)")
    else:
        ax_all.text(0.5, 0.5, "All Runs\nN/A", ha="center", va="center",
                    fontsize=12, color="gray", transform=ax_all.transAxes)
        ax_all.set_xticks([])
        ax_all.set_yticks([])

    prefix = f"{title_prefix} — " if title_prefix else ""
    suptitle = f"{prefix}{strat_label}, n={size}d — Mean vs All Runs"
    fig.suptitle(suptitle, fontsize=12, y=1.02)

    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=FUNCTION_COLORS[i], markersize=5,
                   label=f"F{i + 1}")
            for i in range(N_FUNCTIONS)
        ]
        fig.legend(
            handles=handles, loc="lower center",
            ncol=legend_ncol, fontsize=7, framealpha=0.8,
            bbox_to_anchor=(0.5, -0.06),
            handletextpad=0.2, columnspacing=0.8,
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Convenience: batch grid for all modes
# ---------------------------------------------------------------------------

def plot_all_grids(data, title_prefix="", modes=None, save_dir=None, **kwargs):
    """
    Generate grid plots for each mode.

    Parameters
    ----------
    data : dict from load_*_embeddings
    title_prefix : e.g. 'ELA' or 'TLA (volume_h0)'
    modes : list of modes to plot (default: all available)
    save_dir : if provided, save PNGs here
    """
    if modes is None:
        # Find modes available in at least one config
        modes_found = set()
        for ck in available_configs(data):
            modes_found.update(data[ck].keys())
        modes = sorted(modes_found)

    for mode in modes:
        mode_label = mode.replace("_", " ").title()
        title = f"{title_prefix} — {mode_label}" if title_prefix else mode_label
        save_path = None
        if save_dir:
            save_path = f"{save_dir}/{title_prefix.lower().replace(' ', '_')}_{mode}_grid.png"

        print(f"Plotting grid: {title}")
        plot_grid(data, mode=mode, title=title, save_path=save_path, **kwargs)