"""
ela_viz.py

Visualize the distribution of ELA (Exploratory Landscape Analysis) feature
values across BBOB functions.

Each .pkl file is produced by `extract_ela_features` and corresponds to one
(sampling_method, sample_size, dimension) combination. Its contents are a dict:

    {
        (function, instance, dimension): {
            "ela_dist": [run_0_dict, run_1_dict, ..., run_29_dict],  # 30 runs
            "meta":     [...],
            "disp":     [...],
            "ic":       [...],
            "nbc":      [...],
            "pca":      [...],
        },
        ...
    }

where each `run_i_dict` maps a feature name (e.g. "ela_distr.skewness") to a
scalar value, as returned by the pflacco `calculate_*` functions.

For a chosen feature group, every individual feature in that group is drawn as
its own box plot: x-axis = function number, y-axis = feature value. Each box
aggregates every instance x run for that function. Feature names containing
"costs_runtime" are omitted.
"""

from __future__ import annotations

import math
import pickle
from typing import List, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def get_omit_features(dimension: int) -> set:
    """
    Features known to contain NaN / inf for a given dimension and therefore
    excluded from the plots (in addition to any *.costs_runtime feature).
    """
    if dimension == 2:
        return {
            'disp.diff_median_02', 'disp.ratio_median_02',
            'disp.ratio_mean_02', 'ela_meta.quad_simple.cond',
            'disp.diff_mean_02', 'ic.eps_ratio'
        }
    elif dimension in (5, 10):
        return {'ela_meta.quad_simple.cond'}
    return set()


def load_features(pkl_path) -> dict:
    """Load the pickled feature dictionary from disk."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _as_features(pkl_path_or_dict) -> dict:
    """Accept either a loaded dict or a path/str pointing to a pickle."""
    if isinstance(pkl_path_or_dict, dict):
        return pkl_path_or_dict
    return load_features(pkl_path_or_dict)


def list_feature_groups(pkl_path_or_dict) -> List[str]:
    """Return the feature-group names available in a file (ela_dist, meta, ...)."""
    features = _as_features(pkl_path_or_dict)
    groups = set()
    for group_dict in features.values():
        groups.update(group_dict.keys())
    return sorted(groups)


def list_feature_names(
    pkl_path_or_dict,
    feature_group: str,
    dimension: int | None = None,
    include_runtime: bool = False,
    include_omitted: bool = False,
) -> List[str]:
    """
    Return the individual feature names inside a group (e.g. for "ela_dist":
    ela_distr.skewness, ela_distr.kurtosis, ...).

    By default this drops:
      - any *.costs_runtime feature (unless include_runtime=True), and
      - the dimension-specific NaN/inf features from get_omit_features()
        (unless include_omitted=True).
    """
    features = _as_features(pkl_path_or_dict)
    omit = set() if include_omitted else get_omit_features(dimension)
    for (func, inst, dim), group_dict in features.items():
        if dimension is not None and dim != dimension:
            continue
        runs = group_dict.get(feature_group)
        if runs:  # non-empty list of per-run dicts
            names = list(runs[0].keys())
            if not include_runtime:
                names = [n for n in names if "costs_runtime" not in n]
            names = [n for n in names if n not in omit]
            return names
    raise ValueError(
        f"No data found for feature group {feature_group!r} "
        f"(dimension={dimension}). Available groups: {list_feature_groups(features)}"
    )


def _collect_values(
    features: dict,
    feature_group: str,
    feature_name: str,
    function: int,
    dimension: int,
    instances: Sequence[int],
) -> np.ndarray:
    """Gather all instance x run values of one feature for one function."""
    values = []
    for inst in instances:
        group_dict = features.get((function, inst, dimension))
        if group_dict is None:
            continue
        runs = group_dict.get(feature_group)
        if not runs:
            continue
        for run_dict in runs:
            values.append(run_dict.get(feature_name, np.nan))
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]  # drop NaN / inf so boxplot doesn't choke


def visualize_feature_distribution(
    pkl_path,
    dimension: int,
    feature_group: str,
    functions: Sequence[int] = range(1, 25),
    instances: Sequence[int] = range(1, 101),
    ncols: int = 1,
    showfliers: bool = False,
    showmeans: bool = True,
    width_per_plot: float = 14.0,
    height_per_plot: float = 3.2,
    title: str | None = None,
    ylim=None,
):
    """
    Draw one box plot per feature in `feature_group`.

    Parameters
    ----------
    pkl_path : str | Path | dict
        Path to a pickle written by `extract_ela_features`, or an already
        loaded features dict.
    dimension : int
        Problem dimension to plot (2, 5, 10). Used to filter the keys.
    feature_group : str
        One of "ela_dist", "meta", "disp", "ic", "nbc", "pca".
    functions : sequence of int
        Function numbers to show on the x-axis (default 1..24).
    instances : sequence of int
        Instances to aggregate per box (default 1..100).
    ncols : int
        Number of columns in the subplot grid. Default 1 (each plot full width).
    showfliers : bool
        Whether to draw outlier points. Default False (3000 points per box is
        a lot of clutter).
    showmeans : bool
        Whether to overlay the mean of each box as a dashed line (distinct from
        the solid median line). Default True.
    width_per_plot, height_per_plot : float
        Size in inches of a single subplot.
    title : str | None
        Figure title. Auto-generated if None.
    ylim : None | (ymin, ymax) | dict
        Fix the y-axis range. Pass a single (ymin, ymax) tuple to apply the same
        limits to every plot, or a dict {feature_name: (ymin, ymax)} to set
        limits per feature (features absent from the dict are left autoscaled).
        Either bound may be None to autoscale only that side, e.g. (0, None).

    Returns
    -------
    matplotlib.figure.Figure
    """
    features = _as_features(pkl_path)
    functions = list(functions)
    instances = list(instances)

    feature_names = list_feature_names(features, feature_group, dimension=dimension)
    if not feature_names:
        raise ValueError(
            f"Feature group {feature_group!r} has no plottable features "
            f"(all were costs_runtime?)."
        )

    n = len(feature_names)
    ncols = max(1, ncols)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width_per_plot * ncols, height_per_plot * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    positions = list(range(1, len(functions) + 1))

    for ax, feat in zip(axes_flat, feature_names):
        data = [
            _collect_values(features, feature_group, feat, fn, dimension, instances)
            for fn in functions
        ]
        # Keep x-position alignment but skip empty arrays (boxplot errors on them).
        plot_data, plot_pos = [], []
        for pos, arr in zip(positions, data):
            if arr.size:
                plot_data.append(arr)
                plot_pos.append(pos)

        if plot_data:
            ax.boxplot(
                plot_data,
                positions=plot_pos,
                showfliers=showfliers,
                showmeans=showmeans,
                meanline=showmeans,
                meanprops=dict(color="tab:red", linestyle="--", linewidth=1.2),
            )

        ax.set_xlim(0.5, len(functions) + 0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(functions)
        ax.set_title(feat)
        ax.set_xlabel("BBOB function")
        ax.set_ylabel("feature value")
        ax.grid(axis="y", alpha=0.3)

        if isinstance(ylim, dict):
            feat_ylim = ylim.get(feat)
        else:
            feat_ylim = ylim
        if feat_ylim is not None:
            ax.set_ylim(feat_ylim[0], feat_ylim[1])

    # Hide any unused axes (when n doesn't fill the grid).
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    if title is None:
        title = f"{feature_group} — dimension {dimension}"
    fig.suptitle(title, fontsize=14)

    if showmeans:
        handles = [
            Line2D([0], [0], color="tab:orange", linewidth=1.2, label="median"),
            Line2D([0], [0], color="tab:red", linestyle="--", linewidth=1.2, label="mean"),
        ]
        fig.legend(handles=handles, loc="upper right")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Value vs. sample size, one line per sampling strategy
# ---------------------------------------------------------------------------

_STRATEGY_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan",
]


def parse_sources_from_paths(paths) -> list:
    """
    Build a `sources` list from file paths named `{strategy}_{size}_ela.pkl`
    (the convention used by extract_ela_features). Strategy names may contain
    underscores; the trailing integer before `_ela` is taken as the size.
    """
    import re
    sources = []
    for p in paths:
        stem = str(p).split("/")[-1]
        m = re.match(r"^(.*)_(\d+)_ela\.pkl$", stem)
        if not m:
            raise ValueError(
                f"Cannot parse strategy/size from {stem!r}; "
                f"expected '<strategy>_<size>_ela.pkl'."
            )
        sources.append({"path": p, "strategy": m.group(1), "size": int(m.group(2))})
    return sources


def _normalize_sources(sources) -> list:
    """Accept nested dict {strategy: {size: path}}, list of dicts, or list of tuples."""
    out = []
    if isinstance(sources, dict):
        for strategy, by_size in sources.items():
            for size, path in by_size.items():
                out.append({"path": path, "strategy": strategy, "size": int(size)})
        return out
    for s in sources:
        if isinstance(s, dict):
            out.append({"path": s["path"], "strategy": s["strategy"], "size": int(s["size"])})
        else:  # (path, strategy, size)
            path, strategy, size = s
            out.append({"path": path, "strategy": strategy, "size": int(size)})
    return out


def _function_stats(features, group, feature, function, dimension, instances, center, band):
    """
    Aggregate one feature over the given instances (each instance contributes its
    30-run statistics) for a single function.

    Returns (center_value, band_value) or None if no data.
      center : "mean" | "median" of the per-instance run-means
      band   : "run_std"      -> mean over instances of the within-run std (stability)
               "instance_std" -> std across instances of the run-means (instance spread)
               "none"         -> 0
    """
    per_instance_mean, per_instance_runstd = [], []
    for inst in instances:
        gd = features.get((function, inst, dimension))
        if gd is None:
            continue
        runs = gd.get(group)
        if not runs:
            continue
        vals = np.asarray([r.get(feature, np.nan) for r in runs], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        per_instance_mean.append(vals.mean())
        per_instance_runstd.append(vals.std(ddof=1) if vals.size > 1 else 0.0)

    if not per_instance_mean:
        return None
    pim = np.asarray(per_instance_mean)
    c = np.median(pim) if center == "median" else pim.mean()
    if band == "instance_std":
        b = pim.std(ddof=1) if pim.size > 1 else 0.0
    elif band == "none":
        b = 0.0
    else:  # "run_std"
        b = float(np.mean(per_instance_runstd))
    return float(c), float(b)


def plot_value_vs_size(
    sources,
    dimension,
    feature_group,
    function,
    instance=None,
    center="mean",
    band="run_std",
    logx=True,
    ncols=2,
    width_per_plot=6.5,
    height_per_plot=3.6,
    title=None,
):
    """
    For one BBOB function, plot each feature's value against sample size, with one
    line per sampling strategy and a shaded band showing stability.

    Parameters
    ----------
    sources :
        Where to find the per-(strategy, size) pickles. One of:
          - nested dict {strategy: {size: path}}
          - list of dicts [{"path": ..., "strategy": ..., "size": ...}, ...]
          - list of (path, strategy, size) tuples
        (Use parse_sources_from_paths(paths) to build this from filenames.)
    dimension : int
        Problem dimension (2, 5, 10).
    feature_group : str
        "ela_dist" | "meta" | "disp" | "ic" | "nbc" | "pca".
    function : int
        Single BBOB function number to plot (values can't be pooled across
        functions, so one must be chosen).
    instance : int | None
        Restrict to one instance, or None to aggregate over instances 1..100.
    center : str
        "mean" or "median" line.
    band : str
        Shaded band: "run_std" (run-to-run stability, the default),
        "instance_std" (spread across instances), or "none".
    logx : bool
        Log-scale the sample-size axis (sizes usually span a wide range).
    ncols : int
        Columns in the subplot grid.
    width_per_plot, height_per_plot : float
        Per-subplot size in inches.
    title : str | None
        Figure title (auto-generated if None).

    Returns
    -------
    matplotlib.figure.Figure
    """
    src = _normalize_sources(sources)
    instances = [instance] if instance is not None else list(range(1, 101))
    strategies = sorted({s["strategy"] for s in src})
    color_of = {st: _STRATEGY_COLORS[i % len(_STRATEGY_COLORS)] for i, st in enumerate(strategies)}

    # Load each pickle once.
    cache = {}
    for s in src:
        key = (s["strategy"], s["size"])
        if key not in cache:
            cache[key] = load_features(s["path"])

    # Discover features (drops costs_runtime + dimension-specific omitted ones).
    feature_names = None
    for s in src:
        try:
            feature_names = list_feature_names(cache[(s["strategy"], s["size"])], feature_group, dimension=dimension)
            break
        except ValueError:
            continue
    if not feature_names:
        raise ValueError(f"No plottable features for group {feature_group!r} at dimension {dimension}.")

    # results[feature][strategy] = sorted [(size, center, band), ...]
    results = {f: {st: [] for st in strategies} for f in feature_names}
    for s in src:
        feats = cache[(s["strategy"], s["size"])]
        for feature in feature_names:
            stat = _function_stats(feats, feature_group, feature, function, dimension, instances, center, band)
            if stat is not None:
                results[feature][s["strategy"]].append((s["size"], stat[0], stat[1]))
    for feature in feature_names:
        for st in strategies:
            results[feature][st].sort(key=lambda t: t[0])

    n = len(feature_names)
    ncols = max(1, ncols)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(width_per_plot * ncols, height_per_plot * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    all_sizes = sorted({s["size"] for s in src})
    for ax, feature in zip(axes_flat, feature_names):
        for st in strategies:
            pts = results[feature][st]
            if not pts:
                continue
            sizes = np.array([p[0] for p in pts], dtype=float)
            cen = np.array([p[1] for p in pts], dtype=float)
            bnd = np.array([p[2] for p in pts], dtype=float)
            ax.plot(sizes, cen, marker="o", markersize=4, linewidth=1.5,
                    color=color_of[st], label=st)
            if band != "none":
                ax.fill_between(sizes, cen - bnd, cen + bnd, color=color_of[st], alpha=0.15)
        if logx:
            ax.set_xscale("log")
        ax.set_xticks(all_sizes)
        ax.set_xticklabels(all_sizes)
        ax.set_title(feature)
        ax.set_xlabel("sample size")
        ax.set_ylabel("feature value")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    if title is None:
        band_lbl = {"run_std": "band = run-to-run std",
                    "instance_std": "band = instance spread",
                    "none": ""}.get(band, "")
        scope = f"instance {instance}" if instance is not None else "instances 1–100 (aggregated)"
        title = f"{feature_group} — function {function}, dimension {dimension} — {scope}"
        if band_lbl:
            title += f"\n{band_lbl}"

    handles = [Line2D([0], [0], color=color_of[st], marker="o", markersize=4, label=st)
               for st in strategies]
    fig.legend(handles=handles, loc="upper right", title="sampling strategy")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Overall stability: one number per (strategy, size) across all features
# ---------------------------------------------------------------------------

_ALL_GROUPS = ["ela_dist", "meta", "disp", "ic", "nbc", "pca"]



def _icc_oneway(value_matrix, min_runs=2):
    """
    One-way random-effects ICC(1,1) for a subjects x runs matrix.

    Rows = subjects (e.g. instances), columns = replicate runs. NaNs are allowed;
    a balanced design is assumed (equal runs per subject) for the standard ANOVA
    estimator. Returns ICC in (-inf, 1]; clip to 0 for interpretation since a
    negative value means run noise >= between-subject signal.

        ICC = (MSB - MSW) / (MSB + (k - 1) * MSW)

    Returns None if the matrix is too small or degenerate (no between-subject
    variance to speak of).
    """
    M = np.asarray(value_matrix, dtype=float)
    if M.ndim != 2:
        return None
    # Keep subjects (rows) that have at least min_runs finite values.
    finite_per_row = np.isfinite(M).sum(axis=1)
    M = M[finite_per_row >= min_runs]
    if M.shape[0] < 2:
        return None

    # Use the common run count k across subjects (balanced design from 30 runs).
    k = int(np.min(np.isfinite(M).sum(axis=1)))
    if k < min_runs:
        return None
    # Trim each row to its first k finite values for a balanced estimator.
    rows = []
    for r in M:
        fv = r[np.isfinite(r)]
        rows.append(fv[:k])
    M = np.asarray(rows, dtype=float)
    n = M.shape[0]

    grand = M.mean()
    row_means = M.mean(axis=1)
    ss_between = k * np.sum((row_means - grand) ** 2)
    ss_within = np.sum((M - row_means[:, None]) ** 2)
    df_between = n - 1
    df_within = n * (k - 1)
    if df_between <= 0 or df_within <= 0:
        return None
    msb = ss_between / df_between
    msw = ss_within / df_within
    denom = msb + (k - 1) * msw
    if denom <= 0:
        return None
    return (msb - msw) / denom


# ---------------------------------------------------------------------------
# Overall stability via within-config reliability (ICC) — Renau-safe
# ---------------------------------------------------------------------------

_ALL_GROUPS = ["ela_dist", "meta", "disp", "ic", "nbc", "pca"]

_RELIABILITY_BANDS = [
    (0.90, "excellent"),
    (0.75, "good"),
    (0.50, "moderate"),
    (-np.inf, "poor"),
]


def reliability_band(icc):
    """Map an ICC value to the Koo & Li interpretation band."""
    if icc is None or not np.isfinite(icc):
        return "n/a"
    for thr, label in _RELIABILITY_BANDS:
        if icc >= thr:
            return label
    return "poor"


def overall_stability(
    sources,
    dimension,
    feature_groups=None,
    functions=range(1, 25),
    instances=range(1, 101),
    subject="instance",
    agg="median",
    by_group=False,
    clip_negative=True,
):
    """
    Collapse the experiment into a within-config reliability (ICC) score per
    (sampling strategy, sample size). HIGHER = more stable.

    For every (strategy, size, feature) we compute a one-way random-effects
    ICC(1,1) where the *subject* is the entity the feature should reproducibly
    characterize and the 30 runs are replicate measurements:

      subject="instance" (default, recommended for RQ1):
          ICC is computed within each function (subjects = that function's
          instances, runs = replicates), then averaged over functions. This is
          the stringent, per-instance reliability.
      subject="function":
          ICC is computed with functions as subjects (one representative value
          per function-run, averaged over instances). Coarser: does the feature
          reproducibly separate function classes.

    The ICC is intrinsically unitless and computed ENTIRELY within one strategy's
    context, so nothing is pooled across strategies (Renau-safe). A constant
    per-strategy offset leaves it unchanged. *.costs_runtime and dimension-omitted
    features are excluded as everywhere else.

    Interpretation: ICC ~ 1 means the feature value is essentially fixed by the
    subject and barely moved by resampling; ICC ~ 0 means it is mostly sampling
    noise. Bands (Koo & Li): >0.9 excellent, 0.75-0.9 good, 0.5-0.75 moderate,
    <0.5 poor. See reliability_band().

    Note: this measures reproducibility, NOT accuracy/bias. Pair with a
    convergence check if closeness to a reference matters.

    Returns
    -------
    pandas.DataFrame
        by_group=False -> strategy (rows) x size (cols) of mean ICC.
        by_group=True  -> tidy frame [strategy, size, group, icc].
    """
    import pandas as pd

    src = _normalize_sources(sources)
    groups = list(feature_groups) if feature_groups is not None else _ALL_GROUPS
    functions = list(functions)
    instances = list(instances)
    aggf = np.nanmedian if agg == "median" else np.nanmean

    rows = []
    for s in src:
        feats = load_features(s["path"])
        # feature -> group, discovered once per file
        names_by_group = {}
        for group in groups:
            try:
                names_by_group[group] = list_feature_names(feats, group, dimension=dimension)
            except ValueError:
                names_by_group[group] = []

        # Per feature: collect ICC(s) then average over functions (instance subject).
        per_feature_icc = {}   # feature -> mean ICC over functions
        feat_group = {}
        for group, names in names_by_group.items():
            for name in names:
                feat_group[name] = group
                fn_iccs = []

                if subject == "function":
                    # Build subjects=function x runs matrix: per function, average
                    # over instances within each run to get one value per (func, run).
                    mat = []
                    for fn in functions:
                        # need a consistent run count; use the per-run instance-mean
                        run_vals = None
                        counts = None
                        for inst in instances:
                            gd = feats.get((fn, inst, dimension))
                            if gd is None:
                                continue
                            runs = gd.get(group)
                            if not runs:
                                continue
                            v = np.asarray([r.get(name, np.nan) for r in runs], dtype=float)
                            if run_vals is None:
                                run_vals = np.zeros_like(v)
                                counts = np.zeros_like(v)
                            finite = np.isfinite(v)
                            run_vals[finite] += v[finite]
                            counts[finite] += 1
                        if run_vals is not None:
                            with np.errstate(invalid="ignore"):
                                mat.append(np.where(counts > 0, run_vals / counts, np.nan))
                    icc = _icc_oneway(np.asarray(mat, dtype=float)) if len(mat) >= 2 else None
                    if icc is not None:
                        fn_iccs.append(icc)

                else:  # subject == "instance": ICC within each function, then average
                    for fn in functions:
                        mat = []
                        for inst in instances:
                            gd = feats.get((fn, inst, dimension))
                            if gd is None:
                                continue
                            runs = gd.get(group)
                            if not runs:
                                continue
                            v = np.asarray([r.get(name, np.nan) for r in runs], dtype=float)
                            mat.append(v)
                        if len(mat) >= 2:
                            icc = _icc_oneway(np.asarray(mat, dtype=float))
                            if icc is not None:
                                fn_iccs.append(icc)

                if fn_iccs:
                    val = float(np.mean(fn_iccs))
                    if clip_negative:
                        val = max(val, 0.0)
                    per_feature_icc[name] = val

        if by_group:
            for group in groups:
                vals = [v for nm, v in per_feature_icc.items() if feat_group.get(nm) == group]
                if vals:
                    rows.append({"strategy": s["strategy"], "size": s["size"],
                                 "group": group, "icc": float(aggf(vals))})
        else:
            vals = list(per_feature_icc.values())
            rows.append({"strategy": s["strategy"], "size": s["size"],
                         "icc": float(aggf(vals)) if vals else float("nan")})
        del feats

    df = pd.DataFrame(rows)
    if by_group:
        return df.sort_values(["group", "icc"], ascending=[True, False]).reset_index(drop=True)
    pivot = df.pivot(index="strategy", columns="size", values="icc")
    return pivot.reindex(sorted(pivot.columns), axis=1).sort_index()


def plot_overall_stability(stability_table, title=None, cmap="YlGn"):
    """
    Heatmap of a strategy x size ICC table from overall_stability(by_group=False).
    Higher / greener = more stable (more reproducible).
    """
    df = stability_table
    data = df.values.astype(float)
    fig, ax = plt.subplots(figsize=(1.3 * data.shape[1] + 3, 0.7 * data.shape[0] + 2))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(df.index)
    ax.set_xlabel("sample size")
    ax.set_ylabel("sampling strategy")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.6 else "black", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("within-config reliability ICC (higher = more stable)")
    ax.set_title(title or "Overall feature reliability by (strategy, size)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Rank stability: do features reproduce the ORDERING of subjects across runs?
# ---------------------------------------------------------------------------

def rank_stability(
    sources,
    dimension,
    feature_groups=None,
    functions=range(1, 25),
    instances=range(1, 101),
    method="spearman",
    max_run_pairs=None,
    agg="median",
    by_group=False,
):
    """
    Per (strategy, size, feature): how reproducibly does the feature ORDER a
    function's instances across independent runs? For each function we correlate
    the instance-value vectors of run i vs run j (over all run pairs), average the
    pairwise correlations, then average over functions and aggregate over features.

    Bias- and scale-invariant (monotone): a feature whose absolute value shifts
    but whose ordering holds still scores ~1. Computed within strategy, so it is
    Renau-safe. Returns values in [-1, 1]; higher = more stable ordering.

    method : "spearman" (default) or "kendall".
    max_run_pairs : cap the number of random run pairs per function for speed
                    (None = all pairs; 30 runs -> 435 pairs).
    """
    import pandas as pd
    from itertools import combinations
    from scipy.stats import spearmanr, kendalltau

    corr_fn = kendalltau if method == "kendall" else spearmanr
    src = _normalize_sources(sources)
    groups = list(feature_groups) if feature_groups is not None else _ALL_GROUPS
    functions = list(functions)
    instances = list(instances)
    aggf = np.nanmedian if agg == "median" else np.nanmean
    rng = np.random.default_rng(0)

    rows = []
    for s in src:
        feats = load_features(s["path"])
        names_by_group = {}
        for group in groups:
            try:
                names_by_group[group] = list_feature_names(feats, group, dimension=dimension)
            except ValueError:
                names_by_group[group] = []

        per_feature = {}
        feat_group = {}
        for group, names in names_by_group.items():
            for name in names:
                feat_group[name] = group
                fn_scores = []
                for fn in functions:
                    # instances (rows) x runs (cols)
                    mat = []
                    for inst in instances:
                        gd = feats.get((fn, inst, dimension))
                        if gd is None:
                            continue
                        runs = gd.get(group)
                        if not runs:
                            continue
                        mat.append([r.get(name, np.nan) for r in runs])
                    if len(mat) < 3:
                        continue
                    A = np.asarray(mat, dtype=float)  # instances x runs
                    n_runs = A.shape[1]
                    pairs = list(combinations(range(n_runs), 2))
                    if max_run_pairs is not None and len(pairs) > max_run_pairs:
                        idx = rng.choice(len(pairs), size=max_run_pairs, replace=False)
                        pairs = [pairs[t] for t in idx]
                    cors = []
                    for i, j in pairs:
                        a, b = A[:, i], A[:, j]
                        ok = np.isfinite(a) & np.isfinite(b)
                        if ok.sum() < 3:
                            continue
                        c = corr_fn(a[ok], b[ok]).correlation
                        if np.isfinite(c):
                            cors.append(c)
                    if cors:
                        fn_scores.append(np.mean(cors))
                if fn_scores:
                    per_feature[name] = float(np.mean(fn_scores))

        if by_group:
            for group in groups:
                vals = [v for nm, v in per_feature.items() if feat_group.get(nm) == group]
                if vals:
                    rows.append({"strategy": s["strategy"], "size": s["size"],
                                 "group": group, "rank_stability": float(aggf(vals))})
        else:
            vals = list(per_feature.values())
            rows.append({"strategy": s["strategy"], "size": s["size"],
                         "rank_stability": float(aggf(vals)) if vals else float("nan")})
        del feats

    df = pd.DataFrame(rows)
    if by_group:
        return df.sort_values(["group", "rank_stability"], ascending=[True, False]).reset_index(drop=True)
    pivot = df.pivot(index="strategy", columns="size", values="rank_stability")
    return pivot.reindex(sorted(pivot.columns), axis=1).sort_index()