"""
Plot overall ICC against sample size, one panel per dimension, one line per
sampling strategy.

Consumes the pickles written by compute_ela_icc.py, whose structure is:

    ela_icc[config_key][(func, dim)][feature_group][feature_name] = icc_value

The config_key bundles strategy and size, e.g. "lhs_random_cd_75" ->
strategy "lhs_random_cd", size 75. For each (dimension, strategy, size) the
"overall ICC" is the aggregate (median by default) of every finite per-(func,
feature) ICC in that config — i.e. the headline number from the summary.
"""

import re
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Strategies omitted by default (override via omit_strategies=... or --omit).
DEFAULT_OMIT_STRATEGIES = {"lhs_random_cd"}


# Known multi-word strategy prefixes so the trailing integer is split off
# correctly (e.g. "lhs_random_cd_75" -> "lhs_random_cd", 75).
def split_config_key(config_key):
    """Split a config key like 'lhs_random_cd_75' into ('lhs_random_cd', 75)."""
    m = re.match(r"^(.*)_(\d+)$", config_key)
    if not m:
        raise ValueError(f"Cannot parse strategy/size from config key {config_key!r}")
    return m.group(1), int(m.group(2))


def load_icc_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def overall_icc_per_config(ela_icc, agg="median", omit_strategies=None, only_strategies=None):
    """
    Collapse each config's per-(func, feature) ICCs into one overall value.

    omit_strategies : iterable of strategy names to exclude (e.g. {"cma_random"}).
    only_strategies : if given, keep ONLY these strategies (applied after omit).

    Returns {strategy: {size: overall_icc}}.
    """
    aggf = np.nanmedian if agg == "median" else np.nanmean
    omit = set(omit_strategies or ())
    only = set(only_strategies) if only_strategies else None
    out = {}
    for config_key, by_func in ela_icc.items():
        strategy, size = split_config_key(config_key)
        if strategy in omit:
            continue
        if only is not None and strategy not in only:
            continue
        vals = [
            v
            for by_group in by_func.values()
            for grp in by_group.values()
            for v in grp.values()
            if v is not None and np.isfinite(v)
        ]
        if not vals:
            continue
        out.setdefault(strategy, {})[size] = float(aggf(vals))
    return out


def plot_icc_vs_size(
    sources,
    agg="median",
    omit_strategies="__default__",
    only_strategies=None,
    ncols=None,
    width_per_plot=5.5,
    height_per_plot=4.2,
    title=None,
    sharey=True,
    ylim=(0, 1),
):
    """
    Parameters
    ----------
    sources : dict {dimension: path_to_ela_icc_results.pkl}
        One ICC pickle per dimension (e.g. {2: "...dim2.../ela_icc_results.pkl", ...}).
    agg : "median" | "mean"
        How to collapse per-(func, feature) ICCs into the overall value.
    omit_strategies : iterable of strategy names to exclude from all panels.
        Defaults to DEFAULT_OMIT_STRATEGIES ({'lhs_random_cd'}). Pass an explicit
        set (e.g. set() to omit nothing, or your own list) to override.
    only_strategies : if given, plot ONLY these strategies (applied after omit).
    ncols : int | None
        Columns in the subplot grid (default: one row of all dimensions).
    title : str | None
        Figure title (auto if None).
    sharey, ylim : passed through to the axes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    dims = sorted(sources)
    n = len(dims)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(width_per_plot * ncols, height_per_plot * nrows),
        squeeze=False, sharey=sharey,
    )
    axes_flat = axes.flatten()

    if omit_strategies == "__default__":
        omit_strategies = DEFAULT_OMIT_STRATEGIES

    # Consistent strategy -> color across all panels.
    per_dim = {dim: overall_icc_per_config(load_icc_pkl(p), agg=agg,
                                            omit_strategies=omit_strategies,
                                            only_strategies=only_strategies)
               for dim, p in sources.items()}
    strategies = sorted({s for d in per_dim.values() for s in d})
    cmap = plt.get_cmap("tab10")
    color_of = {st: cmap(i % 10) for i, st in enumerate(strategies)}

    for ax, dim in zip(axes_flat, dims):
        data = per_dim[dim]
        for st in strategies:
            if st not in data:
                continue
            sizes = sorted(data[st])
            ys = [data[st][s] for s in sizes]
            ax.plot(sizes, ys, marker="o", markersize=5, linewidth=1.8,
                    color=color_of[st], label=st)
        ax.set_title(f"dimension {dim}")
        ax.set_xlabel("sample size")
        ax.set_ylabel(f"overall ICC ({agg})")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        # ticks at the actual sizes present
        all_sizes = sorted({s for st in data for s in data[st]})
        if all_sizes:
            ax.set_xticks(all_sizes)
            ax.set_xticklabels(all_sizes)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    handles = [plt.Line2D([0], [0], color=color_of[st], marker="o", label=st)
               for st in strategies]
    fig.legend(handles=handles, loc="upper right", title="sampling strategy")
    fig.suptitle(title or "Overall ELA feature reliability (ICC) vs sample size",
                 fontsize=14)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot overall ICC vs sample size per dimension.")
    parser.add_argument("--pkl", action="append", nargs=2, metavar=("DIM", "PATH"),
                        required=True, help="dimension and path, repeatable: --pkl 2 a.pkl --pkl 5 b.pkl")
    parser.add_argument("--agg", default="median", choices=["median", "mean"])
    parser.add_argument("--omit", nargs="*", default="__default__",
                        help="strategy names to exclude (default: lhs_random_cd). "
                             "Pass --omit with no names to omit nothing.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="plot only these strategies")
    parser.add_argument("--out", default="icc_vs_size.png")
    args = parser.parse_args()
    # --omit absent -> module default; --omit with no args -> omit nothing.
    if args.omit == "__default__":
        omit = "__default__"
    else:
        omit = set(args.omit)
    sources = {int(d): p for d, p in args.pkl}
    fig = plot_icc_vs_size(sources, agg=args.agg,
                           omit_strategies=omit, only_strategies=args.only)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print("saved", args.out)