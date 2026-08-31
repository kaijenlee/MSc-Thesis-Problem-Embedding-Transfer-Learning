"""
classify_plots.py

Loader + figure set for the classification experiment
(classify_ela_subsample_parallel.py -> ela_classification_subsample.h5).

    import classify_plots as cp
    df = cp.load_results({2: "...dim2.../ela_classification_subsample.h5",
                          5: "...", 10: "..."})

    cp.plot_accuracy_vs_budget(df, dimension=5)          # C1
    cp.plot_instances_vs_runs(df, dimension=5)           # C2
    cp.heatmap_grid(df, "sobol", 100, 5)                 # C3
    cp.plot_protocols(df, dimension=5)                   # C4
    cp.plot_consistency(paths, "sobol_100")              # C5
    cp.plot_confusion(paths, "sobol_100")                # C6
    cp.plot_reliability_vs_accuracy(df, ratios, 5, 100)  # C7

THE BUDGET AXIS
---------------
n_feval_train = 24 * n_instances_train * n_runs_train * size * dimension

so a point at (n_inst=10, n_runs=1) costs exactly as much as (5, 2) or (2, 5).
Plotting accuracy against this budget puts every grid point on a common
footing, which is what makes the instances-vs-runs trade-off readable: at a
given x, the vertical spread between colours IS the answer to "more problems
once, or fewer problems repeatedly?".

THE THREE PROTOCOLS
-------------------
  all_runs      predict from ONE run's features   -- one shot at the landscape
  median        predict from the median of 30     -- you can afford to resample
  majority_vote predict per run, take the mode    -- ensemble at inference
The gap between all_runs and the others is what run-to-run noise costs at test
time. Runs enter TRAINING as separate rows (augmentation), so `n_runs_train` is
a different mechanism from the median/vote protocols, which act at test time.
"""

import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

N_FUNCTIONS = 24
STRATEGY_ORDER = ["sobol", "ilhs", "lhs_random_cd", "lhs", "uniform", "cma_random"]

ACC_COLS = {
    "median": "acc_median",
    "all_runs": "acc_all_runs",
    "majority_vote": "acc_majority_vote",
}


def _split_config(key):
    m = re.match(r"^(.*)_(\d+)$", key)
    if not m:
        raise ValueError(f"cannot parse config key {key!r}")
    return m.group(1), int(m.group(2))


def _colors(vals, cmap_name="tab10"):
    cmap = plt.get_cmap(cmap_name)
    return {v: cmap(i % 10) for i, v in enumerate(vals)}


def _strategy_colors(strats):
    ordered = [s for s in STRATEGY_ORDER if s in strats]
    ordered += [s for s in sorted(strats) if s not in ordered]
    return _colors(ordered)


# --------------------------------------------------------------------------- #
# Loader                                                                       #
# --------------------------------------------------------------------------- #

def load_results(paths):
    """Flatten the HDF5 file(s) into a tidy DataFrame.

    `paths` : {dimension: path} or a single path.
    One row per (dimension, strategy, size, n_instances_train, n_runs_train):

      acc_median / acc_all_runs / acc_majority_vote : the three protocols
      consistency_mean / consistency_std            : per-instance agreement
      fold_acc_*_std                                : spread across the 5 folds
      n_feval_train                                 : the common budget axis
    """
    if not isinstance(paths, dict):
        paths = {None: paths}
    rows = []
    for dim, path in paths.items():
        with h5py.File(str(path), "r") as f:
            for cfg in f:
                grp = f[cfg]
                if not isinstance(grp, h5py.Group):
                    continue
                strategy, size = _split_config(cfg)
                for sub in grp:
                    g = grp[sub]
                    if not isinstance(g, h5py.Group) or "fold_accuracies_median" not in g:
                        continue
                    a = dict(g.attrs)
                    rows.append({
                        "dimension": int(a.get("dimension", dim if dim else -1)),
                        "strategy": strategy,
                        "size": int(a.get("sample_size_per_dim", size)),
                        "n_instances_train": int(a["n_instances_train"]),
                        "n_runs_train": int(a["n_runs_train"]),
                        "n_feval_train": int(a["n_feval_train"]),
                        "acc_median": float(a["overall_accuracy_median"]),
                        "acc_all_runs": float(a["overall_accuracy_all_runs"]),
                        "acc_majority_vote": float(
                            a.get("overall_accuracy_majority_vote", np.nan)),
                        "consistency_mean": float(a["overall_consistency_mean"]),
                        "consistency_std": float(a["overall_consistency_std"]),
                        "fold_acc_median_std": float(
                            np.std(g["fold_accuracies_median"][:])),
                        "fold_acc_all_runs_std": float(
                            np.std(g["fold_accuracies_all_runs"][:])),
                        "n_features": int(grp.attrs.get("n_features", -1)),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no results found — check the paths")
    return df.sort_values(["dimension", "strategy", "size",
                           "n_instances_train", "n_runs_train"]
                          ).reset_index(drop=True)


def load_predictions(path, config_key, n_inst, n_runs):
    """Per-instance arrays for one grid point (for C5/C6)."""
    sub = f"inst_{n_inst:02d}_runs_{n_runs:02d}"
    with h5py.File(str(path), "r") as f:
        g = f[config_key][sub]
        return {k: g[k][:] for k in
                ("true_labels", "pred_median", "pred_majority_vote",
                 "per_instance_consistency")} | {"pred_runs": g["pred_runs"][:]}


# --------------------------------------------------------------------------- #
# C1 — accuracy vs training budget                                             #
# --------------------------------------------------------------------------- #

def plot_accuracy_vs_budget(df, dimension=None, metric="acc_all_runs",
                            agg="max", sizes=None, strategies=None,
                            logx=True, width_per=5.4, height=4.4, title=None):
    """C1 — accuracy against the TRAINING budget, one line per strategy.

    Every (n_inst, n_runs) pair that costs the same number of evaluations is
    collapsed with `agg`; use agg="max" to read the figure as an efficiency
    frontier ("the best you can do for this many evaluations") and agg="mean"
    to read it as an average over allocations.

    Note the budget already includes `size`, so a strategy appears at several
    x-positions -- larger samples cost more per feature vector. This is the
    fairest sampler comparison available: equal evaluations, not equal rows.
    """
    d = df if dimension is None else df[df["dimension"] == dimension]
    if sizes is not None:
        d = d[d["size"].isin(sizes)]
    strats = strategies or [s for s in STRATEGY_ORDER
                            if s in set(d["strategy"])]
    color_of = _strategy_colors(set(d["strategy"]))
    dims = sorted(d["dimension"].unique())

    fig, axes = plt.subplots(1, len(dims), figsize=(width_per * len(dims), height),
                             squeeze=False, sharey=True)
    for ax, dim in zip(axes[0], dims):
        sub = d[d["dimension"] == dim]
        for s in strats:
            g = (sub[sub["strategy"] == s]
                 .groupby("n_feval_train")[metric].agg(agg)
                 .reset_index().sort_values("n_feval_train"))
            if len(g):
                ax.plot(g["n_feval_train"], g[metric], marker="o", ms=4,
                        lw=1.6, color=color_of[s], label=s)
        if logx:
            ax.set_xscale("log")
        ax.set_title(f"dimension {dim}")
        ax.set_xlabel("training budget (function evaluations)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].set_ylabel(f"{metric}  ({agg} over allocations)")
    handles = [Line2D([0], [0], color=color_of[s], marker="o", label=s)
               for s in strats]
    fig.suptitle(title or "Classification accuracy vs training budget", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.87, 0.94])
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.88, 0.5),
               title="sampling strategy", frameon=False)
    return fig, axes


# --------------------------------------------------------------------------- #
# C2 — instances vs runs at equal budget                                       #
# --------------------------------------------------------------------------- #

def plot_instances_vs_runs(df, dimension, strategy=None, size=None,
                           metric="acc_all_runs", connect=True,
                           annotate_instances=True, link_isobudget=True,
                           label_fs=7, width=8.4, height=5.6, title=None ,ylim=None, logx=True):
    """C2 — THE budget-allocation figure.

    Accuracy against `n_feval_train`, coloured by `n_runs_train` and labelled
    with `n_instances_train`. Because the budget already accounts for both axes,
    points that share an x-position cost the SAME number of evaluations but
    spend it differently: (10 instances x 1 run) sits at the same x as (5 x 2)
    and (2 x 5).

    Read the VERTICAL spread at a fixed x -- the grey connectors join exactly
    those iso-budget groups, and the number beside each point is how many
    instances it bought:
      1-run points on top  -> spend the budget on MORE PROBLEMS (diversity wins)
      5-run points on top  -> spend it on REPEATS (averaging out noise wins)
      no separation        -> the allocation does not matter, only the total

    Runs enter training as separate rows, so "repeats" here means augmentation
    rather than denoising -- the classifier sees more, noisier rows.
    """
    d = df[df["dimension"] == dimension]
    if strategy:
        d = d[d["strategy"] == strategy]
    if size:
        d = d[d["size"] == size]
    d = d[np.isfinite(d[metric])]
    if d.empty:
        raise ValueError("no rows for that selection")
    runs = sorted(d["n_runs_train"].unique())
    color_of = _colors(runs, "viridis") if len(runs) > 4 else _colors(runs)

    fig, ax = plt.subplots(figsize=(width, height))

    # iso-budget connectors: same cost, different allocation
    if link_isobudget:
        for b, g in d.groupby("n_feval_train"):
            if len(g) > 1:
                ax.plot([b, b], [g[metric].min(), g[metric].max()],
                        color="grey", lw=1.0, alpha=0.45, zorder=1)

    for r in runs:
        s = d[d["n_runs_train"] == r].sort_values("n_feval_train")
        ax.scatter(s["n_feval_train"], s[metric], s=42, color=color_of[r],
                   edgecolor="white", lw=0.6, zorder=3,
                   label=f"{r} run{'s' if r > 1 else ''}")
        if connect:
            ax.plot(s["n_feval_train"], s[metric], color=color_of[r],
                    lw=1.0, alpha=0.45, zorder=2)
        if annotate_instances:
            for _, row in s.iterrows():
                ax.annotate(f"{int(row['n_instances_train'])}",
                            (row["n_feval_train"], row[metric]),
                            xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=label_fs,
                            color=color_of[r], zorder=4)

    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("training budget (function evaluations)")
    ax.set_ylabel("Mean accuracy across folds")
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, which="both", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if link_isobudget:
        handles.append(Line2D([0], [0], color="grey", lw=1.0, alpha=0.6))
        labels.append("equal budget")
    leg = ax.legend(handles, labels, title="runs per instance", frameon=False,
                    loc="lower right")
    if annotate_instances:
        ax.text(0.02, 0.03, "number beside each point = training instances",
                transform=ax.transAxes, va="top", fontsize=8, color="grey")
    tag = " ".join(x for x in [strategy, f"n={size}xd" if size else ""] if x)
    ax.set_title(title or f"Instances vs repetitions at equal budget"
                          + (f"  [{tag}, dim {dimension}]" if tag else
                             f"  [dim {dimension}]"))
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# C3 — the grid                                                                #
# --------------------------------------------------------------------------- #

def heatmap_grid(df, strategy, size, dimension, metric="acc_all_runs",
                 annotate=True, iso_budget=True, cmap_name="viridis",
                 width=7.2, height=5.4, title=None):
    """C3 — rows = instances, cols = runs, cell = accuracy, for one config.

    Anti-diagonal contours join cells of equal cost (n_inst x n_runs constant),
    so an iso-budget comparison is a move ALONG a contour. If accuracy is flat
    along a contour the allocation is irrelevant; if it rises toward one corner,
    that corner is where the budget should go.
    """
    d = df[(df["dimension"] == dimension) & (df["strategy"] == strategy)
           & (df["size"] == size)]
    if d.empty:
        raise ValueError(f"no rows for {strategy}_{size} dim {dimension}")
    M = d.pivot_table(index="n_instances_train", columns="n_runs_train",
                      values=metric)
    inst, runs = list(M.index), list(M.columns)

    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(M.to_numpy(), aspect="auto", cmap=cmap_name,
                   origin="lower", interpolation="nearest")
    ax.set_xticks(range(len(runs))); ax.set_xticklabels(runs)
    ax.set_yticks(range(len(inst))); ax.set_yticklabels(inst)
    ax.set_xlabel("runs per training instance")
    ax.set_ylabel("training instances per class")
    if annotate:
        A = M.to_numpy()
        lo, hi = np.nanmin(A), np.nanmax(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    norm = (A[i, j] - lo) / (hi - lo + 1e-12)
                    ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if norm < 0.55 else "black")
    if iso_budget:
        gi, gj = np.meshgrid(np.arange(len(runs)), np.arange(len(inst)))
        cost = np.outer(np.array(inst, float), np.array(runs, float))
        ax.contour(gi, gj, np.log10(cost), levels=6, colors="white",
                   linewidths=0.7, alpha=0.55)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(metric)
    ax.set_title(title or f"Budget allocation grid  "
                          f"[{strategy}, n={size}xd, dim {dimension}]\n"
                          f"white contours = equal cost")
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# C4 — the three test protocols                                                #
# --------------------------------------------------------------------------- #

def plot_protocols(df, dimension, size=None, n_runs_train=None,
                   strategies=None, width=9.0, height=5.2, title=None):
    """C4 — what does run-to-run noise cost at INFERENCE?

    For each strategy, the best accuracy under each protocol:
      all_runs      one shot at the landscape
      median        median feature vector over 30 runs
      majority_vote 30 predictions, modal label

    The gap between all_runs and the other two is the price of test-time noise.
    A large gap means resampling the new problem before classifying is worth
    real accuracy; a small one means a single evaluation of the landscape
    suffices.
    """
    d = df[df["dimension"] == dimension]
    if size:
        d = d[d["size"] == size]
    if n_runs_train:
        d = d[d["n_runs_train"] == n_runs_train]
    strats = strategies or [s for s in STRATEGY_ORDER if s in set(d["strategy"])]
    protos = list(ACC_COLS)
    g = d.groupby("strategy")[[ACC_COLS[p] for p in protos]].max()

    fig, ax = plt.subplots(figsize=(width, height))
    x = np.arange(len(strats)); w = 0.8 / len(protos)
    pc = _colors(protos, "Set2")
    for i, p in enumerate(protos):
        vals = [g.loc[s, ACC_COLS[p]] if s in g.index else np.nan for s in strats]
        ax.bar(x + (i - (len(protos) - 1) / 2) * w, vals, width=w * 0.9,
               color=pc[p], edgecolor="white", label=p)
    ax.set_xticks(x); ax.set_xticklabels(strats, rotation=25, ha="right")
    ax.set_ylabel("best accuracy")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="test protocol", frameon=False)
    ax.set_title(title or f"Cost of test-time noise by protocol  [dim {dimension}]")
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# C5 / C6 — per-instance detail                                                #
# --------------------------------------------------------------------------- #

def plot_consistency(path, config_key, n_inst=20, n_runs=5, bins=31,
                     width=8.0, height=4.6, title=None):
    """C5 — distribution of per-instance consistency: the fraction of an
    instance's 30 runs that receive the correct label.

    A mass at 1.0 means those instances are classified identically however they
    are sampled. Mass near 0 means an instance is CONSISTENTLY misclassified --
    a systematic confusion, not noise. Mass in the middle is the genuinely
    unstable set, which is what test-time averaging can rescue.
    """
    p = load_predictions(path, config_key, n_inst, n_runs)
    c = p["per_instance_consistency"]
    fig, ax = plt.subplots(figsize=(width, height))
    ax.hist(c, bins=bins, range=(0, 1), color="#4c72b0", edgecolor="white")
    ax.axvline(float(np.mean(c)), color="black", ls="--", lw=1.4,
               label=f"mean {np.mean(c):.3f}")
    ax.set_xlabel("fraction of the 30 runs classified correctly")
    ax.set_ylabel("test instances")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    ax.text(0.02, 0.95, f"always right: {(c == 1).mean():.1%}\n"
                        f"always wrong: {(c == 0).mean():.1%}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=.85))
    ax.set_title(title or f"Per-instance consistency  [{config_key}, "
                          f"{n_inst} inst x {n_runs} runs]")
    fig.tight_layout()
    return fig, ax


def plot_confusion(path, config_key, n_inst=20, n_runs=5, protocol="all_runs",
                   normalise=True, grid=True, group_lines=True,
                   grid_color="#b0b0b0", width=8.6, height=7.6, title=None):
    """C6 — which BBOB functions get confused with which.

    Comparable to Renau et al. (2020), Fig. 5. Off-diagonal structure is the
    interesting part: a pair of functions confused in BOTH directions is a
    genuine similarity in feature space, not a classifier deficiency.

    `grid` draws thin lines on the cell boundaries so a cell can be traced back
    to its row and column without counting -- necessary at 24x24. `group_lines`
    adds heavier rules at the BBOB group boundaries (after f5, f9, f14, f19), so
    confusion WITHIN a problem class is visually separable from confusion
    ACROSS classes; the former is expected, the latter is not.
    `protocol` selects which prediction to tabulate:
      "all_runs"      every (instance, run) prediction -- ONE SHOT at the
                      landscape, and the protocol the cost-to-target analysis is
                      built on, so this is the consistent default.
      "median"        one prediction per instance, from the median feature
                      vector over the 30 runs (test-time denoising).
      "majority_vote" one prediction per instance, the modal label over runs.

    Comparing "all_runs" against "median" is informative in itself: if the same
    confusion pairs appear but weaker, run-to-run noise is merely blurring an
    existing structure. If NEW pairs appear under "all_runs", noise is actively
    pushing instances across particular decision boundaries.
    """
    p = load_predictions(path, config_key, n_inst, n_runs)
    if protocol == "all_runs":
        pr = p["pred_runs"]  # (n_instances, n_runs_test)
        y = np.repeat(p["true_labels"], pr.shape[1])
        pred = pr.ravel()
    elif protocol in ("median", "majority_vote"):
        y = p["true_labels"]
        pred = p[f"pred_{protocol}"]
    else:
        raise ValueError("protocol must be 'all_runs', 'median' or "
                         "'majority_vote'")
    C = np.zeros((N_FUNCTIONS, N_FUNCTIONS))
    for t, q in zip(y, pred):
        C[int(t), int(q)] += 1
    if normalise:
        C = C / np.maximum(C.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1 if normalise else None)
    ax.set_xticks(range(N_FUNCTIONS))
    ax.set_xticklabels([f"{i + 1}" for i in range(N_FUNCTIONS)], fontsize=7)
    ax.set_yticks(range(N_FUNCTIONS))
    ax.set_yticklabels([f"f{i + 1}" for i in range(N_FUNCTIONS)], fontsize=7)
    ax.set_xlabel("predicted");
    ax.set_ylabel("actual")

    # Thin rules on every cell boundary, drawn as minor-tick gridlines so they
    # sit BETWEEN cells rather than through their centres. The colour must
    # contrast with the light end of the colormap -- white lines vanish on the
    # near-empty cells, which is most of a 24x24 confusion matrix.
    if grid:
        ax.set_xticks(np.arange(-0.5, N_FUNCTIONS, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, N_FUNCTIONS, 1), minor=True)
        ax.grid(which="minor", color=grid_color, linestyle="-", linewidth=0.5,
                alpha=0.9)
        ax.tick_params(which="minor", length=0)
        ax.set_axisbelow(False)  # draw the grid over the image
    if group_lines:
        for b in (5, 9, 14, 19):  # after f5, f9, f14, f19
            ax.axvline(b - 0.5, color="black", lw=1.1, alpha=0.65)
            ax.axhline(b - 0.5, color="black", lw=1.1, alpha=0.65)

    for i in range(N_FUNCTIONS):
        for j in range(N_FUNCTIONS):
            if C[i, j] > 0.01:
                ax.text(j, i, f"{C[i, j] * 100:.0f}", ha="center", va="center",
                        fontsize=5.5, color="white" if C[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label(
        "share of the class" if normalise else "count")
    acc = float((y == pred).mean())
    ax.set_title(title or f"Confusion  [{config_key}, {protocol}, acc={acc:.3f}]"
                 + ("\nheavy rules = BBOB group boundaries"
                    if group_lines else ""))
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# C7 — does reliability predict accuracy?                                      #
# --------------------------------------------------------------------------- #

def plot_reliability_vs_accuracy(df, ratios, dimension, size,
                                 metric="acc_all_runs", ratio_col="ratio_w",
                                 annotate=True, width=7.6, height=6.0,
                                 title=None):
    """C7 — the figure that links the two chapters.

    x : the strategy's noise ratio against uniform, from the ICC variance
        components (repro/ICC chapter)
    y : its classification accuracy at the same budget (this chapter)

    The reliability chapter makes a falsifiable PREDICTION: if run-to-run noise
    is what limits the classifier, then strategies with lower sigma_w should
    score higher, and the points should fall on a downward line. If they do not,
    reliability and usefulness are separate properties -- which is itself a
    result, and the more interesting one.

    `ratios` is the DataFrame from icc_analysis.component_ratios(); pass
    ratio_col="ratio_b" to test the signal side instead.
    """
    acc = (df[(df["dimension"] == dimension) & (df["size"] == size)]
           .groupby("strategy")[metric].max())
    r = (ratios[(ratios["dimension"] == dimension) & (ratios["size"] == size)]
         .groupby("strategy")[ratio_col].median())
    common = sorted(set(acc.index) & set(r.index))
    if len(common) < 3:
        raise ValueError(f"only {len(common)} strategies in common")
    color_of = _strategy_colors(common)

    fig, ax = plt.subplots(figsize=(width, height))
    for s in common:
        ax.scatter(r[s], acc[s], s=110, color=color_of[s], edgecolor="black",
                   lw=0.7, zorder=3, label=s)
        if annotate:
            ax.annotate(s, (r[s], acc[s]), fontsize=8, xytext=(6, 5),
                        textcoords="offset points")
    x = np.array([r[s] for s in common]); y = np.array([acc[s] for s in common])
    if len(common) >= 3 and np.ptp(x) > 0:
        lx = np.log10(x)
        b = np.sum((lx - lx.mean()) * (y - y.mean())) / np.sum((lx - lx.mean())**2)
        a = y.mean() - b * lx.mean()
        gx = np.linspace(lx.min(), lx.max(), 40)
        ax.plot(10**gx, a + b * gx, color="black", lw=1.2, ls="--", zorder=1)
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        ax.text(0.03, 0.05, f"Spearman rho = {rho:.2f}\n"
                            f"(negative = less noise -> better accuracy)",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey",
                          alpha=0.85))
    ax.axvline(1.0, color="grey", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(f"{ratio_col} vs uniform  (<1 = less run-to-run noise)")
    ax.set_ylabel(f"{metric}")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(title or f"Does measurement reliability predict classification "
                          f"accuracy?  [dim {dimension}, n={size}xd]")
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# C1b — paired difference against a baseline                                   #
# --------------------------------------------------------------------------- #

def accuracy_delta(df, baseline="uniform", metric="acc_all_runs"):
    """Paired difference in accuracy against `baseline`.

    Pairing is on the EXACT grid point (dimension, size, n_instances_train,
    n_runs_train), not merely on budget -- so the two configs being compared
    used the same number of instances, the same number of runs and the same
    sample size, and differ only in the sampling strategy. That removes the
    allocation and sample-size variation that otherwise dominates the plot.

    Adds `delta` = acc(strategy) - acc(baseline).
    """
    keys = ["dimension", "size", "n_instances_train", "n_runs_train"]
    b = (df[df["strategy"] == baseline].set_index(keys)[metric]
         .rename("_base"))
    if b.empty:
        raise ValueError(f"baseline {baseline!r} not present")
    out = df.join(b, on=keys)
    out = out[np.isfinite(out["_base"])].copy()
    out["delta"] = out[metric] - out["_base"]
    out.attrs["baseline"] = baseline
    out.attrs["metric"] = metric
    return out


def plot_accuracy_delta(df, baseline="uniform", metric="acc_all_runs",
                        dimension=None, kind="box", strategies=None,
                        width_per=4.6, height=4.6, title=None):
    """C1b — how much better than `baseline`, paired grid point by grid point.

    Absolute accuracy hides small differences: a 0.01 gap is invisible on a
    0..1 axis but may be perfectly consistent. This plots the PAIRED difference
    instead, so the y-axis spans only the effect.

    kind="box"  distribution of the advantage over all grid points -- a box
                that sits entirely above 0 means the strategy beats the
                baseline everywhere, not just on average.
    kind="line" the advantage against training budget, which shows whether it
                grows, shrinks, or vanishes as budget increases.

    A near-zero box for the space-filling designs is a RESULT: it says the
    measurement-noise differences between them (which the reliability analysis
    does detect) are too small to matter for this task.
    """
    d = accuracy_delta(df, baseline, metric)
    if dimension is not None:
        d = d[d["dimension"] == dimension]
    d = d[d["strategy"] != baseline]
    strats = [s for s in STRATEGY_ORDER if s in set(d["strategy"])]
    strats += [s for s in sorted(set(d["strategy"])) if s not in strats]
    if strategies:
        strats = [s for s in strats if s in strategies]
    color_of = _strategy_colors(set(d["strategy"]) | {baseline})
    dims = sorted(d["dimension"].unique())

    fig, axes = plt.subplots(1, len(dims), figsize=(width_per * len(dims), height),
                             squeeze=False, sharey=True)
    for ax, dim in zip(axes[0], dims):
        sub = d[d["dimension"] == dim]
        if kind == "box":
            data = [sub.loc[sub["strategy"] == s, "delta"].to_numpy()
                    for s in strats]
            bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                            medianprops=dict(color="black", lw=1.2),
                            flierprops=dict(marker="o", ms=2, alpha=0.3,
                                            markeredgecolor="none"))
            for p, s in zip(bp["boxes"], strats):
                p.set_facecolor(color_of[s]); p.set_alpha(0.7)
                p.set_edgecolor(color_of[s])
            ax.set_xticks(range(1, len(strats) + 1))
            ax.set_xticklabels(strats, rotation=30, ha="right", fontsize=8)
        else:
            for s in strats:
                g = (sub[sub["strategy"] == s].groupby("n_feval_train")["delta"]
                     .median().reset_index().sort_values("n_feval_train"))
                if len(g):
                    ax.plot(g["n_feval_train"], g["delta"], marker="o", ms=4,
                            lw=1.5, color=color_of[s], label=s)
            ax.set_xscale("log")
            ax.set_xlabel("training budget (function evaluations)")
        ax.axhline(0.0, color="black", ls="--", lw=1.2)
        ax.set_title(f"dimension {dim}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0][0].set_ylabel(f"{metric} minus {baseline}\n(paired on the grid point)")
    if kind == "line":
        handles = [Line2D([0], [0], color=color_of[s], marker="o", label=s)
                   for s in strats]
        fig.tight_layout(rect=[0, 0, 0.87, 0.94])
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.88, 0.5),
                   title="sampling strategy", frameon=False)
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle(title or f"Accuracy advantage over {baseline} "
                          f"(paired, {len(d) // max(len(strats), 1)} grid points each)",
                 fontsize=12)
    return fig, axes


# --------------------------------------------------------------------------- #
# C1c — budget curve without the sample-size confound                          #
# --------------------------------------------------------------------------- #

def plot_accuracy_by_size(df, dimension, metric="acc_all_runs",
                          n_runs_train=None, strategies=None, sizes=None,
                          width_per=4.2, height=4.0, title=None):
    """C1c — accuracy vs budget, one PANEL PER SAMPLE SIZE.

    In the pooled budget plot a single x-position mixes configurations that
    bought many CHEAP feature vectors with ones that bought few EXPENSIVE ones,
    because n_feval = 24 * n_inst * n_runs * size * d confounds all three. That
    mixing is a large part of the sawtooth.

    Faceting by `size` fixes the per-vector cost, so within a panel the budget
    axis varies only the amount of training data and the curve is smooth. The
    comparison ACROSS panels then answers a separate question: for a fixed total
    budget, is it better to buy many rough feature vectors or few precise ones?
    """
    d = df[df["dimension"] == dimension]
    if n_runs_train is not None:
        d = d[d["n_runs_train"] == n_runs_train]
    sizes = sizes or sorted(d["size"].unique())
    strats = [s for s in STRATEGY_ORDER if s in set(d["strategy"])]
    if strategies:
        strats = [s for s in strats if s in strategies]
    color_of = _strategy_colors(set(d["strategy"]))

    fig, axes = plt.subplots(1, len(sizes), figsize=(width_per * len(sizes), height),
                             squeeze=False, sharey=True, sharex=True)
    for ax, sz in zip(axes[0], sizes):
        sub = d[d["size"] == sz]
        for s in strats:
            g = (sub[sub["strategy"] == s]
                 .groupby("n_feval_train")[metric].max()
                 .reset_index().sort_values("n_feval_train"))
            if len(g):
                ax.plot(g["n_feval_train"], g[metric], marker="o", ms=3.5,
                        lw=1.5, color=color_of[s], label=s)
        ax.set_xscale("log")
        ax.set_title(f"n = {sz}xd")
        ax.set_xlabel("training budget (fevals)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].set_ylabel(metric)
    handles = [Line2D([0], [0], color=color_of[s], marker="o", label=s)
               for s in strats]
    fig.suptitle(title or f"Accuracy vs budget, per sample size  "
                          f"[dim {dimension}"
                          + (f", {n_runs_train} run(s)" if n_runs_train else "")
                          + "]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.88, 0.93])
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.89, 0.5),
               title="sampling strategy", frameon=False)
    return fig, axes


# --------------------------------------------------------------------------- #
# C1d — budget curve faceted by one allocation axis                            #
# --------------------------------------------------------------------------- #

def plot_accuracy_faceted(df, dimension, facet_by="n_instances_train",
                          metric="acc_all_runs", agg="max", fixed=None,
                          strategies=None, facet_values=None, ncols=4,
                          width_per=3.4, height_per=3.0, sharey=True,
                          title=None):
    """C1d — accuracy vs budget with one allocation axis held fixed per panel.

    The pooled C1 curve is sawtoothed because a single x-position mixes
    configurations that differ in THREE ways at once:

        n_feval = 24 * n_instances * n_runs * size * dimension

    Faceting removes one of them. `facet_by` may be "n_instances_train",
    "n_runs_train" or "size":

      facet_by="n_instances_train"  fixes DIVERSITY. Within a panel the budget
            grows by buying repeats or bigger samples, so the panel answers
            "given N training problems, is extra budget better spent on runs or
            on sample size?"
      facet_by="n_runs_train"       fixes REPETITION -- the cleanest strategy
            comparison, since n_runs=1 is the allocation that C2 shows to be
            best anyway.
      facet_by="size"               fixes the COST PER FEATURE VECTOR
            (equivalent to plot_accuracy_by_size).

    Two of the three still vary inside a panel, so the curve is smoother but
    not fully de-confounded. Pass `fixed` to pin a second axis and remove the
    remaining mixing entirely, e.g.

        plot_accuracy_faceted(df, 5, facet_by="size",
                              fixed={"n_runs_train": 1})

    which leaves n_instances as the only thing the budget buys.
    """
    d = df[df["dimension"] == dimension]
    for k, v in (fixed or {}).items():
        d = d[d[k] == v]
    if d.empty:
        raise ValueError("no rows for that selection")
    vals = facet_values or sorted(d[facet_by].unique())
    strats = [s for s in STRATEGY_ORDER if s in set(d["strategy"])]
    if strategies:
        strats = [s for s in strats if s in strategies]
    color_of = _strategy_colors(set(d["strategy"]))

    n = len(vals); ncols = min(ncols, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_per * ncols, height_per * nrows),
                             squeeze=False, sharey=sharey, sharex=True)
    axf = axes.flatten()
    unit = {"n_instances_train": "instances", "n_runs_train": "run(s)",
            "size": "xd"}.get(facet_by, "")
    for ax, v in zip(axf, vals):
        sub = d[d[facet_by] == v]
        for s in strats:
            g = (sub[sub["strategy"] == s].groupby("n_feval_train")[metric]
                 .agg(agg).reset_index().sort_values("n_feval_train"))
            if len(g):
                ax.plot(g["n_feval_train"], g[metric], marker="o", ms=3.5,
                        lw=1.5, color=color_of[s], label=s)
        ax.set_xscale("log")
        ax.set_title(f"{v} {unit}", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
    for ax in axf[n:]:
        ax.set_visible(False)
    for r in range(nrows):
        axes[r][0].set_ylabel(metric, fontsize=9)
    for c in range(ncols):
        axes[nrows - 1][c].set_xlabel("budget (fevals)", fontsize=9)

    handles = [Line2D([0], [0], color=color_of[s], marker="o", label=s)
               for s in strats]
    fx = ("  " + ", ".join(f"{k}={v}" for k, v in fixed.items())) if fixed else ""
    fig.suptitle(title or f"Accuracy vs budget, faceted by {facet_by}"
                          f"  [dim {dimension}{fx}]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.88, 0.94])
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.89, 0.5),
               title="sampling strategy", frameon=False)
    return fig, axes


# --------------------------------------------------------------------------- #
# C8 — which factor matters most?                                              #
# --------------------------------------------------------------------------- #

FACTORS = ["strategy", "size", "n_instances_train", "n_runs_train"]


def _eta_squared(d, factors, response):
    """Proportion of variance in `response` explained by each factor alone.

    For a balanced full factorial the one-way sums of squares are orthogonal,
    so these are directly comparable and (with the interaction remainder) sum
    to 1. Computed by hand rather than via a model formula so the balance
    assumption stays visible.
    """
    y = d[response].to_numpy(float)
    ss_total = np.sum((y - y.mean()) ** 2)
    out = {}
    for f in factors:
        ss = 0.0
        for _, g in d.groupby(f, observed=True):
            ss += len(g) * (g[response].mean() - y.mean()) ** 2
        out[f] = ss / ss_total if ss_total > 0 else np.nan
    out["_residual"] = 1.0 - sum(out.values())
    return out


def factor_importance(df, dimension=None, response="acc_all_runs",
                      factors=None, control_budget=True, verbose=True):
    """Which of the four design factors explains most of the accuracy?

    THE TRAP. Three factors cost evaluations and one does not:

        n_feval = 24 * n_instances * n_runs * size * dimension

    so an unconditional ranking mostly reports which factor spans the widest
    range in the grid (n_instances varies 1..20, so it "wins" by construction),
    not which one MATTERS. Sampling strategy is free, so it is structurally
    handicapped in that comparison.

    Two decompositions are therefore returned:

      raw        eta^2 on the accuracy itself. Reads as "what drives the spread
                 across everything I ran" -- dominated by total budget.
      controlled eta^2 on the residual after regressing out log(n_feval).
                 Because log(budget) = const + log(n_inst) + log(n_runs) +
                 log(size), removing it strips out exactly the TOTAL SPEND and
                 leaves the ALLOCATION. This is the practitioner's question:
                 "given a budget I can afford, what should I care about?"

    Strategy is orthogonal to budget (every strategy was run on the same grid),
    so its eta^2 is essentially unchanged between the two -- which is what makes
    the controlled column a fair comparison rather than a handicapped one.
    """
    d = df if dimension is None else df[df["dimension"] == dimension]
    d = d[np.isfinite(d[response])].copy()
    factors = factors or FACTORS
    if d.empty:
        raise ValueError("no rows")

    raw = _eta_squared(d, factors, response)

    controlled = None
    if control_budget:
        x = np.log(d["n_feval_train"].to_numpy(float))
        y = d[response].to_numpy(float)
        b = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
        d["_resid"] = y - (y.mean() + b * (x - x.mean()))
        controlled = _eta_squared(d, factors, "_resid")

    tab = pd.DataFrame({"factor": factors + ["_residual"]})
    tab["eta2_raw"] = tab["factor"].map(raw)
    if controlled:
        tab["eta2_budget_controlled"] = tab["factor"].map(controlled)
    tab["n_levels"] = tab["factor"].map(
        lambda f: d[f].nunique() if f in d.columns else np.nan)

    if verbose:
        n_dim = "all dims" if dimension is None else f"dim {dimension}"
        print(f"Variance in {response} explained ({n_dim}, {len(d):,} grid points)")
        print(f"{'factor':<22s}{'raw':>10s}{'budget-controlled':>20s}{'levels':>9s}")
        for _, r in tab.iterrows():
            c = (f"{r['eta2_budget_controlled']:19.1%}"
                 if controlled else "".rjust(20))
            lv = "" if not np.isfinite(r["n_levels"]) else f"{int(r['n_levels']):9d}"
            print(f"{r['factor']:<22s}{r['eta2_raw']:9.1%}{c}{lv}")
        if controlled:
            print("\n  raw        = includes total budget, so the wide-range "
                  "factors dominate")
            print("  controlled = log(budget) regressed out, so this is the "
                  "ALLOCATION effect")
    return tab


def plot_factor_importance(df, dimensions=None, response="acc_all_runs",
                           factors=None, width_per=4.4, height=4.6, title=None):
    """C8 — the two decompositions side by side, one panel per dimension.

    Read the CONTROLLED bars for the practical ranking. A factor that is large
    raw but small controlled only mattered because it bought more evaluations;
    a factor that stays large under control genuinely changes the outcome at
    equal cost.
    """
    d = df if dimensions is None else df[df["dimension"].isin(dimensions)]
    dims = sorted(d["dimension"].unique())
    factors = factors or FACTORS

    fig, axes = plt.subplots(1, len(dims), figsize=(width_per * len(dims), height),
                             squeeze=False, sharey=True)
    for ax, dim in zip(axes[0], dims):
        t = factor_importance(d, dim, response, factors, verbose=False)
        t = t[t["factor"] != "_residual"]
        x = np.arange(len(t)); w = 0.38
        ax.bar(x - w / 2, t["eta2_raw"], width=w, color="#b0b0b0",
               edgecolor="white", label="raw")
        ax.bar(x + w / 2, t["eta2_budget_controlled"], width=w, color="#4c72b0",
               edgecolor="white", label="budget-controlled")
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace("_train", "").replace("n_", "")
                            for f in t["factor"]], rotation=25, ha="right",
                           fontsize=9)
        ax.set_title(f"dimension {dim}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0][0].set_ylabel(f"share of variance in {response}")
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.suptitle(title or "Which design factor drives accuracy?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, axes


# --------------------------------------------------------------------------- #
# C9 — the practical answer: how should I spend a budget?                      #
# --------------------------------------------------------------------------- #

def budget_frontier(df, dimension, metric="acc_all_runs", targets=None,
                    exclude_strategies=None, verbose=True):
    """Which (strategy, size, instances, runs) is the best use of a budget?

    Returns (frontier, cheapest):

      frontier  the non-dominated configurations: those for which no CHEAPER
                configuration achieved equal or better accuracy. Everything off
                the frontier is strictly wasteful -- some other setting got at
                least as far for less. Reading down the frontier tells you how
                the recommendation CHANGES as the budget grows.

      cheapest  for each accuracy target, the cheapest configuration that
                reaches it. This is the form a practitioner actually wants:
                "to hit 90%, spend this much, allocated like this."

    Note the frontier is a single realisation, not an estimate with error bars:
    neighbouring configurations often differ by less than fold-to-fold noise, so
    treat the PATTERN down the frontier (which factors move, which stay put) as
    the finding, not the exact winning cell.
    """
    d = df[(df["dimension"] == dimension) & np.isfinite(df[metric])].copy()
    if exclude_strategies:
        d = d[~d["strategy"].isin(exclude_strategies)]
    if d.empty:
        raise ValueError("no rows for that selection")

    # Several allocations can cost exactly the same. Collapse each budget to its
    # BEST configuration first -- otherwise a frontier point could be one that
    # merely happened to be encountered first at that cost, while a strictly
    # better allocation of the same evaluations was passed over.
    idx_best = d.groupby("n_feval_train")[metric].idxmax()
    best_at = d.loc[idx_best].sort_values("n_feval_train")
    n_at = d.groupby("n_feval_train").size().rename("n_configs")

    running = -np.inf
    keep = []
    for _, r in best_at.iterrows():
        if r[metric] > running:
            keep.append(r)
            running = r[metric]
    cols = ["n_feval_train", metric, "strategy", "size",
            "n_instances_train", "n_runs_train"]
    frontier = pd.DataFrame(keep)[cols].reset_index(drop=True)

    targets = targets or [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    best_map = best_at.set_index("n_feval_train")[metric]
    rows = []
    for t in targets:
        ok = d[d[metric] >= t]
        if len(ok):
            cost = ok["n_feval_train"].min()
            # among the configurations that cost this much, take the BEST one
            tie = d[d["n_feval_train"] == cost]
            r = tie.loc[tie[metric].idxmax()]
            rows.append({"target": t, "n_feval_train": int(cost),
                         metric: r[metric],
                         "best_at_this_cost": float(best_map.get(cost, np.nan)),
                         "n_configs_at_cost": int(n_at.get(cost, 1)),
                         "strategy": r["strategy"], "size": int(r["size"]),
                         "n_instances_train": int(r["n_instances_train"]),
                         "n_runs_train": int(r["n_runs_train"])})
        else:
            rows.append({"target": t, "n_feval_train": None, metric: np.nan,
                         "best_at_this_cost": np.nan, "n_configs_at_cost": 0,
                         "strategy": "-", "size": None,
                         "n_instances_train": None, "n_runs_train": None})
    cheapest = pd.DataFrame(rows)

    if verbose:
        ex = f" (excluding {', '.join(exclude_strategies)})" if exclude_strategies else ""
        print(f"dim {dimension}{ex} — cheapest configuration reaching each target\n")
        print(f"{'target':>7s}{'fevals':>12s}{'acc':>8s}{'best@cost':>11s}"
              f"{'#cfg':>6s}   {'strategy':<16s}{'size':>5s}{'inst':>6s}{'runs':>6s}")
        for _, r in cheapest.iterrows():
            if pd.isna(r["n_feval_train"]):
                print(f"{r['target']:7.0%}{'not reached':>12s}")
                continue
            print(f"{r['target']:7.0%}{int(r['n_feval_train']):12,d}"
                  f"{r[metric]:8.3f}{r['best_at_this_cost']:11.3f}"
                  f"{int(r['n_configs_at_cost']):6d}   {r['strategy']:<16s}"
                  f"{int(r['size']):5d}{int(r['n_instances_train']):6d}"
                  f"{int(r['n_runs_train']):6d}")
        print("\n  acc = the reported configuration; best@cost = the best of the "
              "#cfg allocations\n  that cost the same. Equal values mean no "
              "better use of those evaluations existed.")
        print(f"\n  frontier: {len(frontier)} non-dominated of "
              f"{d['n_feval_train'].nunique()} distinct budgets "
              f"({len(d)} configurations)")
        vc = frontier["n_runs_train"].value_counts().sort_index()
        print(f"  runs on the frontier: "
              + ", ".join(f"{k} run(s) x{v}" for k, v in vc.items()))
        vs = frontier["size"].value_counts().sort_index()
        print(f"  sizes on the frontier: "
              + ", ".join(f"{k}xd x{v}" for k, v in vs.items()))
    return frontier, cheapest


def plot_budget_frontier(df, dimension, metric="acc_all_runs",
                         exclude_strategies=None, annotate=True,
                         width=9.0, height=5.8, title=None):
    """C9 — every configuration as a grey point, the efficient frontier joined.

    A configuration below the line is dominated: something cheaper did at least
    as well. Labels give the winning allocation as size / instances x runs, so
    the shift in recommendation along the frontier is readable directly.
    """
    d = df[(df["dimension"] == dimension) & np.isfinite(df[metric])]
    if exclude_strategies:
        d = d[~d["strategy"].isin(exclude_strategies)]
    fr, _ = budget_frontier(df, dimension, metric,
                            exclude_strategies=exclude_strategies, verbose=False)
    color_of = _strategy_colors(set(d["strategy"]))

    fig, ax = plt.subplots(figsize=(width, height))
    ax.scatter(d["n_feval_train"], d[metric], s=14, color="lightgrey",
               edgecolor="none", zorder=1, label="all configurations")
    ax.plot(fr["n_feval_train"], fr[metric], color="black", lw=1.2,
            ls="--", zorder=2)
    for s in sorted(set(fr["strategy"])):
        q = fr[fr["strategy"] == s]
        ax.scatter(q["n_feval_train"], q[metric], s=60, color=color_of[s],
                   edgecolor="black", lw=0.6, zorder=3, label=s)
    if annotate:
        for _, r in fr.iterrows():
            ax.annotate(f"{int(r['size'])}d/{int(r['n_instances_train'])}"
                        f"x{int(r['n_runs_train'])}",
                        (r["n_feval_train"], r[metric]),
                        xytext=(4, -10), textcoords="offset points",
                        fontsize=6.5, color="black")
    ax.set_xscale("log")
    ax.set_xlabel("training budget (function evaluations)")
    ax.set_ylabel(metric)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="lower right",
              title="on the frontier")
    ex = f", excl. {','.join(exclude_strategies)}" if exclude_strategies else ""
    ax.set_title(title or f"Best use of a training budget  [dim {dimension}{ex}]"
                          f"\nlabels: size/instances x runs")
    fig.tight_layout()
    return fig, ax


def cost_to_target(df, dimension, targets=None, metric="acc_all_runs",
                   strategies=None, verbose=True):
    """Cheapest budget at which EACH strategy reaches each accuracy target.

    The per-strategy version of budget_frontier. Where that answers "what is
    the best use of a budget", this answers "how much does each sampler cost to
    get me to X%", which is the head-to-head comparison.

    It also converts the null result into a number: if every space-filling
    design reaches 90% at a similar cost, the earlier eta^2 of 1.3% for strategy
    becomes concrete -- "no sampler saves you meaningful evaluations" rather
    than "the variance attributable to strategy is small".

    Returns (tidy, pivot_cost, pivot_ratio):
      tidy        one row per (strategy, target) with cost, accuracy, allocation
      pivot_cost  targets x strategies, cheapest evaluations (NaN = unreached)
      pivot_ratio the same, divided by the cheapest strategy in that row, so
                  1.00 marks the winner and 2.00 means "twice the cost"
    """
    d = df[(df["dimension"] == dimension) & np.isfinite(df[metric])]
    if strategies:
        d = d[d["strategy"].isin(strategies)]
    if d.empty:
        raise ValueError("no rows for that selection")
    targets = targets or [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    strats = [s for s in STRATEGY_ORDER if s in set(d["strategy"])]
    strats += [s for s in sorted(set(d["strategy"])) if s not in strats]

    rows = []
    for s in strats:
        ds = d[d["strategy"] == s]
        for t in targets:
            ok = ds[ds[metric] >= t]
            if len(ok):
                cost = ok["n_feval_train"].min()
                tie = ds[ds["n_feval_train"] == cost]
                r = tie.loc[tie[metric].idxmax()]
                rows.append({"strategy": s, "target": t,
                             "n_feval_train": int(cost), metric: r[metric],
                             "size": int(r["size"]),
                             "n_instances_train": int(r["n_instances_train"]),
                             "n_runs_train": int(r["n_runs_train"])})
            else:
                rows.append({"strategy": s, "target": t,
                             "n_feval_train": np.nan, metric: np.nan,
                             "size": np.nan, "n_instances_train": np.nan,
                             "n_runs_train": np.nan})
    tidy = pd.DataFrame(rows)
    pivot = tidy.pivot(index="target", columns="strategy",
                       values="n_feval_train")[strats]
    ratio = pivot.div(pivot.min(axis=1), axis=0)

    if verbose:
        print(f"dim {dimension} — cheapest budget for each strategy to reach "
              f"each target\n")
        show = pivot.copy()
        show.index = [f"{t:.0%}" for t in show.index]      # keep the formatter
        show.index.name = "target"                          # off the index
        print(show.map(lambda v: "—" if pd.isna(v) else f"{v:,.0f}").to_string())
        print(f"\nrelative to the cheapest strategy in each row "
              f"(1.00 = winner):\n")
        showr = ratio.copy()
        showr.index = [f"{t:.0%}" for t in showr.index]
        showr.index.name = "target"
        print(showr.map(lambda v: "—" if pd.isna(v) else f"{v:.2f}").to_string())
        sf = [s for s in strats if s != "cma_random"]
        if len(sf) > 1:
            sub = ratio[sf].to_numpy()
            sub = sub[np.isfinite(sub)]
            if sub.size:
                print(f"\n  space-filling designs: worst/best cost ratio is "
                      f"{sub.max():.2f}x at most, median {np.median(sub):.2f}x")
    return tidy, pivot, ratio


def plot_cost_to_target(df, dimension, targets=None, metric="acc_all_runs",
                        strategies=None, width=8.6, height=5.4, title=None):
    """C10 — evaluations needed to reach each accuracy target, per strategy.

    Lines that sit on top of one another mean the samplers are interchangeable
    in cost; a line displaced upward means that sampler needs systematically
    more evaluations for the same result. Missing points are targets a strategy
    never reached within the grid.
    """
    tidy, pivot, _ = cost_to_target(df, dimension, targets, metric,
                                    strategies, verbose=False)
    strats = list(pivot.columns)
    color_of = _strategy_colors(set(strats))

    fig, ax = plt.subplots(figsize=(width, height))
    for s in strats:
        q = tidy[(tidy["strategy"] == s) & np.isfinite(tidy["n_feval_train"])]
        if len(q):
            ax.plot(q["target"], q["n_feval_train"], marker="o", ms=5, lw=1.7,
                    color=color_of[s], label=s)
    ax.set_yscale("log")
    ax.set_xlabel("accuracy target")
    ax.set_ylabel("cheapest training budget (function evaluations)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="upper left",
              title="sampling strategy")
    ax.set_title(title or f"Cost to reach each accuracy target  [dim {dimension}]"
                          f"\nmissing points = target never reached")
    fig.tight_layout()
    return fig, ax


def best_per_strategy(df, dimension, metric="acc_all_runs", strategies=None,
                      as_markdown=False, verbose=True):
    """Highest accuracy each strategy reaches anywhere in the grid.

    One row per strategy: the single best-performing configuration, its
    accuracy, and what it cost. This is the CEILING per strategy -- what is
    achievable with the most generous allocation swept -- as opposed to
    cost_to_target, which reports the CHEAPEST route to a given accuracy.

    Ties on accuracy are broken by cost, so the row reports the cheapest of the
    equally-best configurations rather than an arbitrary one.

    Note these maxima are bounded by the grid: n_instances_train tops out at 20
    (the training fold size), so a strategy pinned at 20 instances may not have
    plateaued.
    """
    d = df[(df["dimension"] == dimension) & np.isfinite(df[metric])]
    if strategies:
        d = d[d["strategy"].isin(strategies)]
    if d.empty:
        raise ValueError("no rows for that selection")
    strats = [s for s in STRATEGY_ORDER if s in set(d["strategy"])]
    strats += [s for s in sorted(set(d["strategy"])) if s not in strats]

    rows = []
    for s in strats:
        ds = d[d["strategy"] == s]
        best = ds[metric].max()
        tie = ds[np.isclose(ds[metric], best)]
        r = tie.loc[tie["n_feval_train"].idxmin()]  # cheapest of the best
        rows.append({
            "strategy": s,
            metric: float(r[metric]),
            "n_feval_train": int(r["n_feval_train"]),
            "size": int(r["size"]),
            "n_instances_train": int(r["n_instances_train"]),
            "n_runs_train": int(r["n_runs_train"]),
            "n_tied": int(len(tie)),
            "at_grid_max": bool(r["n_instances_train"] == d["n_instances_train"].max()),
        })
    out = pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)

    if verbose and not as_markdown:
        print(f"dim {dimension} — best configuration per strategy\n")
        print(f"{'strategy':<16s}{'acc':>8s}{'fevals':>12s}"
              f"{'size':>6s}{'inst':>6s}{'runs':>6s}   note")
        for _, r in out.iterrows():
            note = "at grid max instances" if r["at_grid_max"] else ""
            print(f"{r['strategy']:<16s}{r[metric]:8.4f}"
                  f"{r['n_feval_train']:12,d}{r['size']:6d}"
                  f"{r['n_instances_train']:6d}{r['n_runs_train']:6d}   {note}")
    if as_markdown:
        show = out.copy()
        show["allocation"] = (show["n_instances_train"].astype(str) + "@"
                              + show["size"].astype(str) + "×"
                              + show["n_runs_train"].astype(str))
        show["fevals"] = show["n_feval_train"].map(lambda v: f"{v:,}")
        show[metric] = show[metric].map(lambda v: f"{v:.4f}")
        cols = ["strategy", metric, "fevals", "allocation"]
        try:
            print(show[cols].to_markdown(index=False))
        except ImportError:
            print(show[cols].to_string(index=False))
    return out


def ceiling_table(paths, metric="acc_all_runs", strategies=None,
                  as_markdown=False, verbose=True):
    """Maximum attainable accuracy, from a maximum-data run.

    Intended for HDF5 files produced with the largest sample size, all 20
    training instances and all 30 runs -- i.e. every training row the design
    permits. Reports one row per (dimension, strategy).

    THE 20-INSTANCE LIMIT IS THE BINDING ONE, not the runs. The 20/80 split caps
    training at 20 instances per class, and the cost-to-target tables show
    accuracy still climbing in instances at dimension 2 (every 90% row sits at
    exactly 20). Going further would need a different split, which changes the
    test set and breaks comparability with the rest of the sweep. So this is the
    ceiling GIVEN that limit, not an unconditional one.

    `paths` : {dimension: path to the maximum-data h5}
    """
    df = load_results(paths)
    if strategies:
        df = df[df["strategy"].isin(strategies)]
    if df.empty:
        raise ValueError("no rows found")

    strats = [s for s in STRATEGY_ORDER if s in set(df["strategy"])]
    strats += [s for s in sorted(set(df["strategy"])) if s not in strats]
    dims = sorted(df["dimension"].unique())

    # keep the largest-data cell per (dimension, strategy)
    rows = []
    for d in dims:
        for s in strats:
            q = df[(df["dimension"] == d) & (df["strategy"] == s)]
            if q.empty:
                continue
            r = q.loc[q["n_feval_train"].idxmax()]
            rows.append({
                "dimension": d, "strategy": s,
                metric: float(r[metric]),
                "acc_median": float(r["acc_median"]),
                "consistency": float(r["consistency_mean"]),
                "fold_std": float(r["fold_acc_all_runs_std"]),
                "n_feval_train": int(r["n_feval_train"]),
                "size": int(r["size"]),
                "n_instances_train": int(r["n_instances_train"]),
                "n_runs_train": int(r["n_runs_train"]),
            })
    out = pd.DataFrame(rows)

    pivot = out.pivot(index="strategy", columns="dimension",
                      values=metric).reindex(strats)

    if verbose and not as_markdown:
        for d in dims:
            sub = out[out["dimension"] == d].sort_values(metric, ascending=False)
            cfg = sub.iloc[0]
            print(f"\ndim {d} — maximum attainable "
                  f"({cfg['n_instances_train']} inst x {cfg['n_runs_train']} runs "
                  f"@ {cfg['size']}xd, {cfg['n_feval_train']:,} fevals)\n")
            print(f"{'strategy':<16s}{'acc':>9s}{'+/- fold':>10s}"
                  f"{'consistency':>13s}")
            for _, r in sub.iterrows():
                print(f"{r['strategy']:<16s}{r[metric]:9.4f}"
                      f"{r['fold_std']:10.4f}{r['consistency']:13.4f}")
            sf = sub[sub["strategy"] != "cma_random"][metric]
            if len(sf) > 1:
                print(f"  space-filling spread: {sf.max() - sf.min():.4f} "
                      f"(max fold sd {sub['fold_std'].max():.4f})")
        print("\nceiling by dimension (space-filling best):")
        for d in dims:
            sub = out[(out["dimension"] == d) & (out["strategy"] != "cma_random")]
            print(f"  dim {d:>2}: {sub[metric].max():.4f}")

    if as_markdown:
        show = pivot.copy()
        show.columns = [f"dim {c}" for c in show.columns]
        show = show.round(4)
        try:
            print(show.to_markdown())
        except ImportError:
            print(show.to_string())
    return out, pivot


def stability_cost_table(df, ratios, dimension, targets=None,
                         metric="acc_all_runs", exclude=("cma_random",),
                         as_markdown=True):
    """Join cost-to-target with run-to-run noise, per accuracy target.

    Built directly from the two DataFrames so that no value is transcribed by
    hand -- the cost, the allocation and the sigma_w that goes with it always
    come from the same row.

    For each target: the cheapest strategy (or strategies, on a tie), the
    allocation it used, its sigma_w AT THE SAMPLE SIZE THAT ALLOCATION USED, and
    -- for comparison -- the least noisy strategy at that same size and what it
    cost. Where the two differ, stability is not determining cost.

    `ratios` is icc_analysis.component_ratios() output; sigma_w is read as
    ratio_w, the median ratio against uniform at the matching (dimension, size).
    """
    d = df[(df["dimension"] == dimension) & np.isfinite(df[metric])]
    d = d[~d["strategy"].isin(exclude)]
    rr = ratios[ratios["dimension"] == dimension]
    sig = rr.groupby(["strategy", "size"])["ratio_w"].median()
    targets = targets or [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

    rows = []
    for t in targets:
        ok = d[d[metric] >= t]
        if ok.empty:
            rows.append({"target": f"{t:.0%}", "cheapest": "—", "allocation": "—",
                         "sigma_w": "—", "fevals": "—", "min_sigma_w": "—",
                         "most_stable": "—", "its_allocation": "—",
                         "its_fevals": "—"})
            continue
        best = ok["n_feval_train"].min()
        tie = ok[ok["n_feval_train"] == best]
        # one representative allocation per winning strategy
        winners = sorted(tie["strategy"].unique())
        rep = tie.loc[tie.groupby("strategy")[metric].idxmax()]
        allocs = sorted({f"{int(r.n_instances_train)}@{int(r['size'])}"
                         f"x{int(r.n_runs_train)}" for _, r in rep.iterrows()})
        sizes = sorted({int(r["size"]) for _, r in rep.iterrows()})
        wsig = sorted(round(float(sig.get((r.strategy, int(r["size"])), np.nan)), 3)
                      for _, r in rep.iterrows())

        # least noisy strategy at the winner's size (first, if several sizes tie)
        size0 = sizes[0]
        cand = {s: sig.get((s, size0), np.nan) for s in d["strategy"].unique()}
        cand = {k: v for k, v in cand.items() if np.isfinite(v)}
        ms = min(cand, key=cand.get) if cand else None
        if ms is not None:
            mo = d[(d["strategy"] == ms) & (d[metric] >= t)]
            if len(mo):
                mr = mo.loc[mo["n_feval_train"].idxmin()]
                ms_alloc = (f"{int(mr.n_instances_train)}@{int(mr['size'])}"
                            f"x{int(mr.n_runs_train)}")
                ms_fev = f"{int(mr.n_feval_train):,}"
            else:
                ms_alloc, ms_fev = "—", "—"
        else:
            ms_alloc, ms_fev = "—", "—"

        rows.append({
            "target": f"{t:.0%}",
            "cheapest": ", ".join(winners) if len(winners) < 4 else f"{len(winners)}-way tie",
            "allocation": " / ".join(allocs),
            "sigma_w": f"{wsig[0]:.3f}" if len(wsig) == 1 else f"{wsig[0]:.3f}–{wsig[-1]:.3f}",
            "fevals": f"{int(best):,}",
            "min_sigma_w": f"{cand[ms]:.3f}" if ms else "—",
            "most_stable": ms or "—",
            "its_allocation": ms_alloc,
            "its_fevals": ms_fev,
        })

    out = pd.DataFrame(rows)
    if as_markdown:
        try:
            print(f"### Dimension {dimension}\n")
            print(out.to_markdown(index=False))
        except ImportError:
            print(out.to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# C10 — within-sampler: every allocation, accuracy against budget              #
# --------------------------------------------------------------------------- #

_SIZE_MARKERS = {25: "o", 50: "s", 75: "^", 100: "D"}
_RUN_COLORS = {1: "#d62728", 2: "#ff7f0e", 3: "#2ca02c", 5: "#1f77b4",
               10: "#9467bd", 30: "#8c564b"}


def plot_allocation_scatter(df, dimension, metric="acc_all_runs",
                            strategies=None, ncols=2, logx=False, ylim=None,
                            connect=True, color_by="n_runs_train",
                            annotate_instances=False, ms=42, label_fs=6,
                            width_per=4.2, height_per=3.2, title=None):
    """C10 — one panel per sampling strategy: EVERY configuration, not just the
    frontier.

    The cost-to-target tables report only the cheapest route to each accuracy,
    which hides how much is going on underneath: a configuration reaching 89%
    for a tenth of the cost of one reaching 90% is invisible there. This shows
    the whole cloud.

    ENCODING. Three design variables on two axes:
      colour   runs per instance (default) -- given the most salient channel
               because it carries the main result: the 1-run points should form
               the upper envelope at every budget.
      marker   sample size (25/50/75/100 x d), the cost per feature vector
      position along a line -- the instance count, since at fixed size and runs
               the budget is proportional to it. Lines join constant
               (size, runs), so each traces the effect of adding instances.

    Set color_by="size" to swap the colour and marker channels, and
    annotate_instances=True to label each point with its instance count where
    reading it off the line is not enough.

    Read the VERTICAL spread at a fixed x: that is the accuracy cost of
    allocating a given budget badly. Read the SLOPE of each line: steep means
    instances are still buying accuracy, flat means that combination has
    saturated.

    NOTE ON THE X-AXIS. Linear by default, so that the cost differences are
    read at face value. Budgets span roughly two and a half orders of magnitude,
    however, so the cheap configurations crowd into the left margin; pass
    logx=True to spread them out, at the price of visually compressing the
    expensive end.
    """
    d = df[(df["dimension"] == dimension) & np.isfinite(df[metric])]
    strats = [s for s in STRATEGY_ORDER if s in set(d["strategy"])]
    strats += [s for s in sorted(set(d["strategy"])) if s not in strats]
    if strategies:
        strats = [s for s in strats if s in strategies]
    sizes = sorted(d["size"].unique())
    runs = sorted(d["n_runs_train"].unique())

    if color_by == "n_runs_train":
        col_of = {r: _RUN_COLORS.get(r, "grey") for r in runs}
        mark_of = {s: _SIZE_MARKERS.get(s, "o") for s in sizes}
        col_key, mark_key = "n_runs_train", "size"
        col_title, mark_title = "runs / instance", "sample size"
        col_lab = {r: f"{r} run{'s' if r > 1 else ''}" for r in runs}
        mark_lab = {s: f"{s}xd" for s in sizes}
    else:
        cmap = plt.get_cmap("viridis")
        col_of = {s: cmap(i / max(len(sizes) - 1, 1)) for i, s in enumerate(sizes)}
        mark_of = {r: _SIZE_MARKERS.get(list(_SIZE_MARKERS)[i % 4], "o")
                   for i, r in enumerate(runs)}
        col_key, mark_key = "size", "n_runs_train"
        col_title, mark_title = "sample size", "runs / instance"
        col_lab = {s: f"{s}xd" for s in sizes}
        mark_lab = {r: f"{r} run{'s' if r > 1 else ''}" for r in runs}

    n = len(strats);
    ncols = min(ncols, n);
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_per * ncols, height_per * nrows),
                             squeeze=False, sharex=True, sharey=True)
    axf = axes.flatten()
    for ax, st in zip(axf, strats):
        sub = d[d["strategy"] == st]
        for sz in sizes:
            for nr in runs:
                g = (sub[(sub["size"] == sz) & (sub["n_runs_train"] == nr)]
                     .sort_values("n_feval_train"))
                if g.empty:
                    continue
                cv = nr if col_key == "n_runs_train" else sz
                mv = sz if mark_key == "size" else nr
                if connect:
                    ax.plot(g["n_feval_train"], g[metric], color=col_of[cv],
                            lw=1.0, alpha=0.5, zorder=1)
                ax.scatter(g["n_feval_train"], g[metric], s=ms,
                           color=col_of[cv], marker=mark_of[mv],
                           edgecolor="white", lw=0.5, zorder=3)
                if annotate_instances:
                    for _, r in g.iterrows():
                        ax.annotate(f"{int(r['n_instances_train'])}",
                                    (r["n_feval_train"], r[metric]),
                                    xytext=(0, 6), textcoords="offset points",
                                    ha="center", fontsize=label_fs,
                                    color=col_of[cv], zorder=4)
        if logx:
            ax.set_xscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(st, fontsize=16)
        ax.grid(True, which="both", alpha=0.3)
    for ax in axf[n:]:
        ax.set_visible(False)
    for r in range(nrows):
        axes[r][0].set_ylabel("Mean Accuracy across folds", fontsize=16)
    for c in range(ncols):
        axes[nrows - 1][c].set_xlabel("training budget (function evaluations)", fontsize=16)

    col_h = [Line2D([0], [0], color=col_of[v], lw=3, label=col_lab[v])
             for v in (runs if col_key == "n_runs_train" else sizes)]
    mark_h = [Line2D([0], [0], color="grey", marker=mark_of[v], lw=0, ms=7,
                     label=mark_lab[v])
              for v in (sizes if mark_key == "size" else runs)]
    # Instances get their own legend block even though they are not a colour or
    # marker channel: with 8 levels they cannot be, so they are encoded as
    # POSITION ALONG A LINE. Stating that beside the other two keys makes all
    # three design variables findable in one place rather than leaving one of
    # them only in the subtitle.
    inst_vals = sorted(d["n_instances_train"].unique())
    inst_h = [Line2D([0], [0], color="grey", lw=1.4, marker=">", ms=6,
                     markevery=[-1],
                     label=f"[1, 2, 3, 5, 7, 10, 15, 20] along line")]

    fig.tight_layout(rect=[0, 0, 0.80, 0.94])
    leg1 = fig.legend(handles=col_h, loc="upper left",
                      bbox_to_anchor=(0.815, 0.90), title=col_title,
                      frameon=False, fontsize=16, title_fontsize=16)
    fig.add_artist(leg1)
    leg2 = fig.legend(handles=mark_h, loc="upper left",
                      bbox_to_anchor=(0.815, 0.63), title=mark_title,
                      frameon=False, fontsize=16, title_fontsize=16)
    fig.add_artist(leg2)
    fig.legend(handles=inst_h, loc="upper left", bbox_to_anchor=(0.815, 0.36),
               title="instances", frameon=False, fontsize=16,  title_fontsize=16)
    extra = "" if not annotate_instances else "  (labels = instances)"
    fig.suptitle(title or f"Accuracy against functions evaluations "+ "\n" + f"for every allocation of each sampling strategy at Dimension {dimension}"
                          f"{extra}", fontsize=24)
    return fig, axes


# --------------------------------------------------------------------------- #
# C11 — between-sampler: one efficient frontier per strategy                   #
# --------------------------------------------------------------------------- #

def plot_frontier_by_strategy(df, dimension, metric="acc_all_runs",
                              strategies=None, show_cloud=True, logx=True,
                              ylim=None, width=9.0, height=5.8, title=None):
    """C11 — the efficient frontier of EACH sampler, overlaid.

    plot_budget_frontier draws a single frontier over all configurations, which
    answers "what is the best use of a budget" but says nothing about which
    sampler achieves it. Here each strategy gets its own frontier: the
    configurations for which no cheaper configuration of THAT SAMPLER did as
    well.

    A frontier that sits above and to the left is better at every budget. Where
    frontiers coincide the samplers are interchangeable; where they separate,
    the horizontal gap is the extra budget the lower one needs for the same
    accuracy -- which is the cost-to-target result, read continuously rather
    than at fixed thresholds.
    """
    d = df[(df["dimension"] == dimension) & np.isfinite(df[metric])]
    strats = [s for s in STRATEGY_ORDER if s in set(d["strategy"])]
    strats += [s for s in sorted(set(d["strategy"])) if s not in strats]
    if strategies:
        strats = [s for s in strats if s in strategies]
    color_of = _strategy_colors(set(d["strategy"]))

    fig, ax = plt.subplots(figsize=(width, height))
    if show_cloud:
        ax.scatter(d["n_feval_train"], d[metric], s=8, color="lightgrey",
                   edgecolor="none", zorder=1)
    for st in strats:
        sub = d[d["strategy"] == st]
        # best per budget, then running maximum
        best = sub.loc[sub.groupby("n_feval_train")[metric].idxmax()]
        best = best.sort_values("n_feval_train")
        keep, run_max = [], -np.inf
        for _, r in best.iterrows():
            if r[metric] > run_max:
                keep.append(r);
                run_max = r[metric]
        fr = pd.DataFrame(keep)
        ax.plot(fr["n_feval_train"], fr[metric], marker="o", ms=4.5, lw=1.7,
                color=color_of[st], label=st, zorder=3)
    if logx:
        ax.set_xscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("training budget (function evaluations)")
    ax.set_ylabel(metric)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="sampling strategy", frameon=False, loc="lower right",
              fontsize=9)
    ax.set_title(title or f"Efficient frontier per sampler  [dim {dimension}]"
                          f"\nhorizontal gap = extra budget for the same accuracy")
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# C12 — does run-to-run noise predict accuracy?                                #
# --------------------------------------------------------------------------- #

def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def noise_vs_accuracy(df, ratios, dimension=None, metric="acc_all_runs",
                      exclude=("cma_random",), verbose=True):
    """Join each configuration's accuracy to the run-to-run noise of its sampler.

    THE CONFOUND, and why the correlation must be computed WITHIN groups.
    ratio_w varies only over (strategy, size, dimension) -- it knows nothing
    about instances or runs. Accuracy varies over all five. Worse, ratio_w
    FALLS with sample size while accuracy RISES with it, so a correlation
    pooled over sizes would be driven almost entirely by that shared dependence
    and would look strong regardless of whether noise matters at all.

    At a fixed (size, instances, runs) every strategy has the SAME budget, so
    the six points in such a group differ only in sampling strategy. The
    within-group correlation is therefore the controlled one, and it is what
    this function reports; the pooled value is returned alongside purely to
    show how misleading it is.

    Returns (merged, summary) where summary holds the within-group Spearman
    correlations, one row per (size, instances, runs).
    """
    d = df if dimension is None else df[df["dimension"] == dimension]
    d = d[~d["strategy"].isin(exclude)]
    sig = (ratios.groupby(["dimension", "strategy", "size"])["ratio_w"]
           .median().rename("ratio_w").reset_index())
    m = d.merge(sig, on=["dimension", "strategy", "size"], how="inner")
    m = m[np.isfinite(m[metric]) & np.isfinite(m["ratio_w"])]
    if m.empty:
        raise ValueError("no overlap between the accuracy and ratio tables")

    keys = ["dimension", "size", "n_instances_train", "n_runs_train"]
    rows = []
    for k, g in m.groupby(keys):
        rows.append(dict(zip(keys, k)) | {
            "n_strategies": len(g),
            "rho": _spearman(g["ratio_w"], g[metric]),
            "acc_spread": float(g[metric].max() - g[metric].min()),
        })
    summary = pd.DataFrame(rows)

    if verbose:
        pooled = _spearman(m["ratio_w"], m[metric])
        w = summary["rho"].dropna()
        print(f"dim {dimension} — noise against accuracy "
              f"({len(m):,} configurations)\n")
        print(f"  pooled Spearman rho          : {pooled:+.3f}   "
              f"<- CONFOUNDED by sample size")
        print(f"  within (size, inst, runs)    : median {w.median():+.3f}, "
              f"mean {w.mean():+.3f}  (n = {len(w)} groups)")
        print(f"  groups with rho > 0          : {(w > 0).mean():.0%}")
        print(f"  median accuracy spread across strategies within a group: "
              f"{summary['acc_spread'].median():.4f}")
        print("\n  A negative rho means LESS noise goes with HIGHER accuracy.")
        print("  Compare the spread against the SEM (~0.006): if it is of the "
              "same order,\n  the strategies are not separable within a group "
              "whatever the sign of rho.")
    return m, summary


def plot_noise_vs_accuracy(df, ratios, dimension, metric="acc_all_runs",
                           n_runs_train=1, instances=None,
                           exclude=("cma_random",), annotate=False, ncols=4,
                           width_per=3.2, height_per=2.9, title=None):
    """C12 — accuracy against sampler noise, one panel per instance count.

    Within a panel and a colour, the points are the individual sampling
    strategies at IDENTICAL budget -- so any trend there is attributable to the
    sampler and not to how much was spent. Colour is sample size, which is the
    variable ratio_w depends on and which must therefore be held constant for
    the comparison to mean anything.

    A downward trend within a colour means quieter samplers score higher. A
    flat cloud means they do not, and given the fold-to-fold SEM of roughly
    0.006, a vertical spread of that order within a colour is indistinguishable
    from noise regardless of any apparent slope.
    """
    m, summary = noise_vs_accuracy(df, ratios, dimension, metric, exclude,
                                   verbose=False)
    if n_runs_train is not None:
        m = m[m["n_runs_train"] == n_runs_train]
    inst = instances or sorted(m["n_instances_train"].unique())
    sizes = sorted(m["size"].unique())
    cmap = plt.get_cmap("viridis")
    col_of = {s: cmap(i / max(len(sizes) - 1, 1)) for i, s in enumerate(sizes)}

    n = len(inst);
    ncols = min(ncols, n);
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_per * ncols, height_per * nrows),
                             squeeze=False, sharex=True, sharey=True)
    axf = axes.flatten()
    for ax, ni in zip(axf, inst):
        sub = m[m["n_instances_train"] == ni]
        for sz in sizes:
            g = sub[sub["size"] == sz].sort_values("ratio_w")
            if g.empty:
                continue
            ax.plot(g["ratio_w"], g[metric], color=col_of[sz], lw=1.0,
                    alpha=0.5, zorder=1)
            ax.scatter(g["ratio_w"], g[metric], s=34, color=col_of[sz],
                       edgecolor="white", lw=0.5, zorder=3)
            if annotate:
                for _, r in g.iterrows():
                    ax.annotate(r["strategy"][:4], (r["ratio_w"], r[metric]),
                                xytext=(3, 3), textcoords="offset points",
                                fontsize=5.5, color=col_of[sz])
        rho = summary[(summary["n_instances_train"] == ni)
                      & (summary["n_runs_train"] == (n_runs_train or 1))]["rho"]
        tag = f"  (rho {rho.median():+.2f})" if len(rho.dropna()) else ""
        ax.set_title(f"{ni} instance{'s' if ni > 1 else ''}{tag}", fontsize=9.5)
        ax.grid(True, alpha=0.3)
    for ax in axf[n:]:
        ax.set_visible(False)
    for r in range(nrows):
        axes[r][0].set_ylabel(metric, fontsize=9)
    for c in range(ncols):
        axes[nrows - 1][c].set_xlabel("sigma_w ratio vs uniform", fontsize=9)

    handles = [Line2D([0], [0], color=col_of[s], lw=3, label=f"{s}xd")
               for s in sizes]
    fig.tight_layout(rect=[0, 0, 0.87, 0.92])
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.88, 0.5),
               title="sample size", frameon=False)
    rt = f", {n_runs_train} run" if n_runs_train else ""
    fig.suptitle(title or f"Sampler noise against accuracy  [dim {dimension}{rt}]"
                          f"\nwithin a colour the budget is identical; points "
                          f"are the sampling strategies", fontsize=11)
    return fig, axes


def plot_noise_accuracy_summary(df, ratios, dimensions=(2, 5, 10),
                                metric="acc_all_runs", scale="sem", n_folds=5,
                                exclude=("cma_random",), n_runs_train=None,
                                show_absolute=False,
                                width=9.6, height=4.4, title=None):
    """C13 — the whole stability-cost argument in one figure.

    LEFT: distribution of the within-group Spearman correlation between a
    sampler's run-to-run noise and its classification accuracy. Each group is
    the set of strategies at one (size, instances, runs), i.e. at IDENTICAL
    budget, so a group's correlation is attributable to the sampler alone.
    Negative means quieter samplers score higher. The dashed line at zero is
    the null.

    RIGHT: the spread in accuracy across strategies within those same groups,
    against the fold-to-fold standard error. This is the check that decides
    whether the left panel means anything: a strong rank correlation over
    differences smaller than the measurement precision is not a finding. Where
    the box sits at the SEM line, the strategies are not merely uncorrelated
    with their noise -- they are indistinguishable from one another.

    Read the two together. A negative correlation with spread well above the
    reference is a real effect; a negative correlation with spread at the
    reference is ranking noise.

    `scale` selects the reference against which the spread is measured:

      "sem" (default)  the standard error of the mean, s/sqrt(k). This is the
            uncertainty of the REPORTED accuracy, and therefore the right scale
            for judging whether two configurations differ. Reference lines are
            drawn at 1 and at 2*sqrt(2), the approximate 95% threshold for a
            difference between two such means.
      "sd"  the fold-to-fold standard deviation itself. Use this where the rest
            of the document reports mean +/- SD, so that the figure is on the
            same scale as the tables. It is the more conservative choice by a
            factor of sqrt(k-1) = 2, and a spread of one SD is a much weaker
            statement than a spread of one SEM.

    show_absolute=True adds a third panel giving the spread in accuracy units
    rather than in multiples of the measurement error. The normalised panel
    answers "can the difference be detected"; the absolute one answers "how much
    accuracy is at stake", and the two can diverge, since the measurement error
    itself falls as the training set grows. Reporting only the normalised figure
    can therefore make a shrinking difference look constant, and reporting only
    the absolute one can make a difference look meaningful when it is smaller
    than the experiment can resolve.
    """
    if scale not in ("sem", "sd"):
        raise ValueError("scale must be 'sem' or 'sd'")
    # Per-group SEM rather than one global constant: precision varies by more
    # than an order of magnitude across configurations, and is worst at small
    # training sets -- exactly where the accuracy differences are largest.
    dsem = add_sem(df, n_folds)
    col = "sem_all_runs" if scale == "sem" else "fold_acc_all_runs_std"
    keys = ["dimension", "size", "n_instances_train", "n_runs_train"]
    gsem = (dsem[~dsem["strategy"].isin(exclude)]
            .groupby(keys)[col].median().rename("sem").reset_index())

    rows = []
    for d in dimensions:
        _, summ = noise_vs_accuracy(df, ratios, d, metric, exclude,
                                    verbose=False)
        if n_runs_train is not None:
            summ = summ[summ["n_runs_train"] == n_runs_train]
        summ["dimension"] = d
        rows.append(summ)
    S = pd.concat(rows, ignore_index=True).merge(gsem, on=keys, how="left")
    S["spread_in_sem"] = S["acc_spread"] / S["sem"]
    sem_val = float(S["sem"].median())
    dims = sorted(S["dimension"].unique())
    cmap = plt.get_cmap("tab10")
    col = {d: cmap(i % 10) for i, d in enumerate(dims)}

    ncols = 3 if show_absolute else 2
    fig, axes = plt.subplots(1, ncols,
                             figsize=(width * ncols / 2, height))

    # --- left: correlation ---
    ax = axes[0]
    data = [S.loc[S["dimension"] == d, "rho"].dropna().to_numpy() for d in dims]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", lw=1.3),
                    flierprops=dict(marker="o", ms=2.5, alpha=0.35,
                                    markeredgecolor="none"))
    for p, d in zip(bp["boxes"], dims):
        p.set_facecolor(col[d]);
        p.set_alpha(0.65);
        p.set_edgecolor(col[d])
    ax.axhline(0, color="black", ls="--", lw=1.2)
    ax.set_xticks(range(1, len(dims) + 1))
    ax.set_xticklabels([f"dim {d}" for d in dims])
    ax.set_ylabel("Spearman rho within a budget group")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("noise vs accuracy, at equal budget", fontsize=10)
    for i, d in enumerate(dims, start=1):
        v = S.loc[S["dimension"] == d, "rho"].dropna()
        ax.text(i, 0.97, f"{(v > 0).mean():.0%} positive", ha="center",
                fontsize=7.5, color="grey")

        # --- optional third panel: the spread in accuracy units ---
    if show_absolute:
        ax3 = axes[2]
        data3 = [S.loc[S["dimension"] == d, "acc_spread"].dropna().to_numpy()
                 for d in dims]
        bp3 = ax3.boxplot(data3, patch_artist=True, widths=0.55,
                          medianprops=dict(color="black", lw=1.3),
                          flierprops=dict(marker="o", ms=2.5, alpha=0.35,
                                          markeredgecolor="none"))
        for p, d in zip(bp3["boxes"], dims):
            p.set_facecolor(col[d]);
            p.set_alpha(0.65);
            p.set_edgecolor(col[d])
        ax3.set_xticks(range(1, len(dims) + 1))
        ax3.set_xticklabels([f"dim {d}" for d in dims])
        ax3.set_ylabel("accuracy spread across strategies")
        ax3.grid(True, axis="y", alpha=0.3)
        ax3.set_title("how much accuracy is at stake?", fontsize=10)
        for i in range(len(dims)):
            med = float(np.median(data3[i]))
            ax3.text(i + 1, med, f"  {med:.3f}", fontsize=7.5, color="grey",
                     va="bottom")

    # --- right: is the difference even measurable? ---
    ax2 = axes[1]
    data2 = [S.loc[S["dimension"] == d, "spread_in_sem"].dropna().to_numpy()
             for d in dims]
    bp2 = ax2.boxplot(data2, patch_artist=True, widths=0.55,
                      medianprops=dict(color="black", lw=1.3),
                      flierprops=dict(marker="o", ms=2.5, alpha=0.35,
                                      markeredgecolor="none"))
    for p, d in zip(bp2["boxes"], dims):
        p.set_facecolor(col[d]);
        p.set_alpha(0.65);
        p.set_edgecolor(col[d])
    lbl = "SEM" if scale == "sem" else "SD"
    ax2.axhline(1.0, color="#d62728", ls="--", lw=1.4)
    ax2.text(len(dims) + 0.45, 1.0, f" 1 {lbl}", fontsize=8, color="#d62728",
             va="center")
    if scale == "sem":
        ax2.axhline(2 * np.sqrt(2), color="#d62728", ls=":", lw=1.2, alpha=0.8)
        ax2.text(len(dims) + 0.45, 2 * np.sqrt(2), " 2\u221a2 SEM\n (95%)",
                 fontsize=8, color="#d62728", va="center")
    ax2.set_xticks(range(1, len(dims) + 1))
    ax2.set_xticklabels([f"dim {d}" for d in dims])
    ax2.set_ylabel(f"accuracy spread / {lbl} of that group")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_title(f"is the difference measurable?  "
                  f"(median {lbl} {sem_val:.4f})", fontsize=10)
    for i, d in enumerate(dims, start=1):
        med = float(np.median(data2[i - 1]))
        ax2.text(i, med, f"  {med:.1f}x", fontsize=7.5, color="grey",
                 va="bottom")



    fig.suptitle(title or "Does sampler stability predict classification "
                          "accuracy?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig, axes


def add_sem(df, n_folds=5, inplace=False):
    """Attach the standard error of each reported accuracy.

    The reported accuracy is the MEAN of `n_folds` fold accuracies, so its
    uncertainty is the standard error of that mean, not the fold-to-fold spread
    itself. `load_results` stores the fold standard deviation using numpy's
    default ddof=0, which gives a convenient identity:

        SEM = s(ddof=1)/sqrt(n) = s(ddof=0)*sqrt(n/(n-1))/sqrt(n)
            = s(ddof=0)/sqrt(n-1)

    so with five folds the SEM is simply half the stored value.

    Adds `sem_all_runs` and `sem_median`. Note this captures variation across
    the five partitions of a FIXED set of instances under a fixed seed; it is a
    lower bound on the uncertainty that re-running the whole experiment would
    show.
    """
    out = df if inplace else df.copy()
    k = np.sqrt(n_folds - 1)
    if "fold_acc_all_runs_std" in out.columns:
        out["sem_all_runs"] = out["fold_acc_all_runs_std"] / k
    if "fold_acc_median_std" in out.columns:
        out["sem_median"] = out["fold_acc_median_std"] / k
    return out


def sem_report(df, n_folds=5, by=None, verbose=True):
    """Summarise the standard error, overall and by whichever factor you name.

    Worth running by "n_instances_train": precision is expected to be worst at
    small training sets, which is exactly where the accuracy differences between
    configurations are largest, so a single pooled SEM can understate the
    uncertainty precisely where it matters most.
    """
    d = add_sem(df, n_folds)
    col = "sem_all_runs"
    if verbose:
        s = d[col].dropna()
        print(f"standard error of the reported accuracy ({len(s):,} configs)\n")
        print(f"  median {s.median():.4f}   mean {s.mean():.4f}   "
              f"IQR {s.quantile(.25):.4f}–{s.quantile(.75):.4f}   "
              f"max {s.max():.4f}")
        print(f"  a difference is interpretable at roughly "
              f"2*sqrt(2)*SEM = {2 * np.sqrt(2) * s.median():.4f}")
        if by:
            print(f"\n  by {by}:")
            g = d.groupby(by)[col].median()
            for k_, v in g.items():
                print(f"    {k_:>8}: {v:.4f}")
    return d.groupby(by)[col].median() if by else d[col].median()


def plot_instances_vs_runs_grid(df, dimensions=(2, 5, 10), strategy=None,
                                size=None, metric="acc_all_runs", connect=True,
                                annotate_instances=True, link_isobudget=True,
                                label_fs=6.5, sharex=False, ylim=None,
                                width_per=4.6, height=5.0, title=None):
    """C2 across several dimensions, as one row of panels.

    Same encoding as plot_instances_vs_runs: colour is runs per instance, the
    label beside each point is the instance count, and the grey connectors join
    settings of identical cost. The panels share a y-axis so the vertical gap
    between the single-run points and the rest is comparable across dimensions.

    `sharex` is False by default because the budget ranges differ by a factor of
    d between dimensions; setting it True puts them on a common axis, at the
    cost of compressing the low-dimensional panel.
    """
    dims = list(dimensions)
    d_all = df[df["dimension"].isin(dims)]
    if strategy:
        d_all = d_all[d_all["strategy"] == strategy]
    if size:
        d_all = d_all[d_all["size"] == size]
    d_all = d_all[np.isfinite(d_all[metric])]
    if d_all.empty:
        raise ValueError("no rows for that selection")
    runs = sorted(d_all["n_runs_train"].unique())
    color_of = ({r: _RUN_COLORS.get(r, "grey") for r in runs}
                if len(runs) <= 6 else _colors(runs, "viridis"))

    fig, axes = plt.subplots(1, len(dims), figsize=(width_per * len(dims), height),
                             squeeze=False, sharey=True, sharex=sharex)
    axf = axes[0]
    for ax, dim in zip(axf, dims):
        d = d_all[d_all["dimension"] == dim]
        if link_isobudget:
            for b, g in d.groupby("n_feval_train"):
                if len(g) > 1:
                    ax.plot([b, b], [g[metric].min(), g[metric].max()],
                            color="grey", lw=1.0, alpha=0.45, zorder=1)
        for r in runs:
            s = d[d["n_runs_train"] == r].sort_values("n_feval_train")
            if s.empty:
                continue
            ax.scatter(s["n_feval_train"], s[metric], s=42, color=color_of[r],
                       edgecolor="white", lw=0.6, zorder=3,
                       label=f"{r} run{'s' if r > 1 else ''}")
            if connect:
                ax.plot(s["n_feval_train"], s[metric], color=color_of[r],
                        lw=1.0, alpha=0.45, zorder=2)
            if annotate_instances:
                for _, row in s.iterrows():
                    ax.annotate(f"{int(row['n_instances_train'])}",
                                (row["n_feval_train"], row[metric]),
                                xytext=(0, 7), textcoords="offset points",
                                ha="center", fontsize=label_fs,
                                color=color_of[r], zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel("training budget (function evaluations)")
        ax.set_title(f"dimension {dim}", fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
    axf[0].set_ylabel(metric)

    handles = [Line2D([0], [0], marker="o", lw=0, color=color_of[r],
                      label=f"{r} run{'s' if r > 1 else ''}") for r in runs]
    if link_isobudget:
        handles.append(Line2D([0], [0], color="grey", lw=1.0, alpha=0.6,
                              label="equal budget"))
    fig.tight_layout(rect=[0, 0, 0.87, 0.92])
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.88, 0.5),
               title="runs per instance", frameon=False)
    tag = " ".join(x for x in [strategy, f"n={size}xd" if size else ""] if x)
    extra = "; labels = instances" if annotate_instances else ""
    fig.suptitle(title or f"Diversity vs repetition at equal budget"
                 + (f"  [{tag}]" if tag else "") + extra, fontsize=12)
    return fig, axes


def plot_instances_vs_size_grid(df, dimensions=(2, 5, 10), strategy=None,
                                n_runs_train=1, metric="acc_all_runs",
                                connect=True, annotate_instances=True,
                                link_isobudget=True, label_fs=6.5,
                                sharex=False, ylim=None, width_per=4.6,
                                height=5.0, title=None):
    """The diversity-versus-precision counterpart of
    plot_instances_vs_runs_grid.

    Instead of trading instances against REPETITION, this trades them against
    SAMPLE SIZE. The two are different questions: an extra run buys another
    near-duplicate feature vector of the same instance, whereas a larger sample
    buys a more precise feature vector of it. Both are paid for out of the same
    budget, so at a fixed cost a setting may hold many feature vectors computed
    from small samples, or few computed from large ones.

    Colour is the sample size and the label beside each point is the instance
    count, so a grey connector joining two points shows the same budget spent
    on many rough feature vectors versus few precise ones. Runs are held fixed
    (at one by default) so that only two factors vary.

    Read the ORDER of the colours at a fixed budget: if the small-sample points
    sit above, buy more feature vectors; if the large-sample points sit above,
    buy better ones. A crossover along the budget axis indicates that the answer
    depends on how much is available.
    """
    dims = list(dimensions)
    d_all = df[df["dimension"].isin(dims)]
    if strategy:
        d_all = d_all[d_all["strategy"] == strategy]
    if n_runs_train is not None:
        d_all = d_all[d_all["n_runs_train"] == n_runs_train]
    d_all = d_all[np.isfinite(d_all[metric])]
    if d_all.empty:
        raise ValueError("no rows for that selection")
    sizes = sorted(d_all["size"].unique())
    cmap = plt.get_cmap("viridis")
    color_of = {s: cmap(i / max(len(sizes) - 1, 1)) for i, s in enumerate(sizes)}

    fig, axes = plt.subplots(1, len(dims), figsize=(width_per * len(dims), height),
                             squeeze=False, sharey=True, sharex=sharex)
    axf = axes[0]
    for ax, dim in zip(axf, dims):
        d = d_all[d_all["dimension"] == dim]
        if link_isobudget:
            for b, g in d.groupby("n_feval_train"):
                if len(g) > 1:
                    ax.plot([b, b], [g[metric].min(), g[metric].max()],
                            color="grey", lw=1.0, alpha=0.45, zorder=1)
        for sz in sizes:
            s = d[d["size"] == sz].sort_values("n_feval_train")
            if s.empty:
                continue
            ax.scatter(s["n_feval_train"], s[metric], s=42, color=color_of[sz],
                       edgecolor="white", lw=0.6, zorder=3)
            if connect:
                ax.plot(s["n_feval_train"], s[metric], color=color_of[sz],
                        lw=1.0, alpha=0.45, zorder=2)
            if annotate_instances:
                for _, row in s.iterrows():
                    ax.annotate(f"{int(row['n_instances_train'])}",
                                (row["n_feval_train"], row[metric]),
                                xytext=(0, 7), textcoords="offset points",
                                ha="center", fontsize=label_fs,
                                color=color_of[sz], zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel("training budget (function evaluations)")
        ax.set_title(f"dimension {dim}", fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
    axf[0].set_ylabel(metric)

    handles = [Line2D([0], [0], marker="o", lw=0, color=color_of[s],
                      label=f"{s}$\\times$d") for s in sizes]
    if link_isobudget:
        handles.append(Line2D([0], [0], color="grey", lw=1.0, alpha=0.6,
                              label="equal budget"))
    fig.tight_layout(rect=[0, 0, 0.87, 0.92])
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.88, 0.5),
               title="sample size", frameon=False)
    rt = f"{n_runs_train} run" if n_runs_train else "all runs"
    tag = " ".join(x for x in [strategy, rt] if x)
    extra = "; labels = instances" if annotate_instances else ""
    fig.suptitle(title or f"Diversity vs precision at equal budget"
                 + (f"  [{tag}]" if tag else "") + extra, fontsize=12)
    return fig, axes


def plot_runs_vs_size_grid(df, dimensions=(2, 5, 10), strategy=None,
                           n_instances_train=20, metric="acc_all_runs",
                           annotate_size=True, link_isobudget=True,
                           label_fs=7, ylim=None, width_per=4.6, height=5.0,
                           title=None):
    """The third pairwise trade-off: repetition against sample size, with the
    number of instances held fixed.

    This is the most direct test of whether repeated sampling can substitute for
    precision. With instances fixed, the cost of a setting is proportional to
    runs x size, so a number of exact equal-cost pairs exist in the grid:

        1 run at 50xd   vs  2 runs at 25xd
        1 run at 75xd   vs  3 runs at 25xd
        1 run at 100xd  vs  2 runs at 50xd
        2 runs at 75xd  vs  3 runs at 50xd

    Each pair spends the same evaluations either on ONE precise feature vector
    or on SEVERAL rough ones. The grey connectors join them. If repetition were
    a substitute for precision, the two members of a pair would score alike; if
    precision matters, the single-run member sits higher.

    Note this differs from Section~\\ref{sec:repetition}, where repetition was
    traded against instance DIVERSITY. Here diversity is held constant and the
    comparison is between two ways of spending the same budget on the same
    instances.
    """
    dims = list(dimensions)
    d_all = df[df["dimension"].isin(dims)]
    if strategy:
        d_all = d_all[d_all["strategy"] == strategy]
    if n_instances_train is not None:
        d_all = d_all[d_all["n_instances_train"] == n_instances_train]
    d_all = d_all[np.isfinite(d_all[metric])]
    if d_all.empty:
        raise ValueError("no rows for that selection")
    runs = sorted(d_all["n_runs_train"].unique())
    sizes = sorted(d_all["size"].unique())
    color_of = {r: _RUN_COLORS.get(r, "grey") for r in runs}
    mark_of = {s: _SIZE_MARKERS.get(s, "o") for s in sizes}

    fig, axes = plt.subplots(1, len(dims), figsize=(width_per * len(dims), height),
                             squeeze=False, sharey=True)
    axf = axes[0]
    for ax, dim in zip(axf, dims):
        d = d_all[d_all["dimension"] == dim]
        if link_isobudget:
            for b, g in d.groupby("n_feval_train"):
                if len(g) > 1:
                    ax.plot([b, b], [g[metric].min(), g[metric].max()],
                            color="grey", lw=1.2, alpha=0.55, zorder=1)
        for r in runs:
            for sz in sizes:
                g = d[(d["n_runs_train"] == r) & (d["size"] == sz)]
                if g.empty:
                    continue
                ax.scatter(g["n_feval_train"], g[metric], s=60,
                           color=color_of[r], marker=mark_of[sz],
                           edgecolor="white", lw=0.6, zorder=3)
                if annotate_size:
                    for _, row in g.iterrows():
                        ax.annotate(f"{int(row['size'])}",
                                    (row["n_feval_train"], row[metric]),
                                    xytext=(0, 8), textcoords="offset points",
                                    ha="center", fontsize=label_fs,
                                    color=color_of[r], zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel("training budget (function evaluations)")
        ax.set_title(f"dimension {dim}", fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
    axf[0].set_ylabel(metric)

    handles = [Line2D([0], [0], marker="o", lw=0, color=color_of[r],
                      label=f"{r} run{'s' if r > 1 else ''}") for r in runs]
    handles += [Line2D([0], [0], marker=mark_of[s], lw=0, color="grey",
                       label=f"{s}$\\times$d") for s in sizes]
    if link_isobudget:
        handles.append(Line2D([0], [0], color="grey", lw=1.2, alpha=0.6,
                              label="equal budget"))
    fig.tight_layout(rect=[0, 0, 0.86, 0.92])
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.87, 0.5),
               frameon=False, fontsize=9)
    tag = " ".join(x for x in [strategy,
                               f"{n_instances_train} instances"
                               if n_instances_train else ""] if x)
    extra = "; labels = sample size" if annotate_size else ""
    fig.suptitle(title or f"Repetition vs sample size at equal budget"
                 + (f"  [{tag}]" if tag else "") + extra, fontsize=12)
    return fig, axes


def plot_frontier_by_strategy_grid(df, dimensions=(2, 5, 10),
                                   metric="acc_all_runs", strategies=None,
                                   show_cloud=False, logx=True, ylim=None,
                                   sharex=False, width_per=4.6, height=5.0,
                                   title=None):
    """C11 across several dimensions, as one row of panels.

    Each panel shows, for every sampling strategy, the settings that no cheaper
    setting of that same strategy matched. A frontier lying above and to the
    left is better at every budget; where two frontiers coincide the strategies
    are interchangeable, and where they separate the HORIZONTAL gap is the extra
    budget the lower one needs for the same accuracy.

    The panels share a y-axis so that the separation between frontiers is
    comparable across dimensions, which is the point of showing them together.
    `sharex` is False by default because the budget ranges differ by a factor of
    d between dimensions.
    """
    dims = list(dimensions)
    d_all = df[df["dimension"].isin(dims)]
    d_all = d_all[np.isfinite(d_all[metric])]
    if strategies:
        d_all = d_all[d_all["strategy"].isin(strategies)]
    if d_all.empty:
        raise ValueError("no rows for that selection")
    strats = [s for s in STRATEGY_ORDER if s in set(d_all["strategy"])]
    strats += [s for s in sorted(set(d_all["strategy"])) if s not in strats]
    color_of = _strategy_colors(set(d_all["strategy"]))

    fig, axes = plt.subplots(1, len(dims), figsize=(width_per * len(dims), height),
                             squeeze=False, sharey=True, sharex=sharex)
    axf = axes[0]
    for ax, dim in zip(axf, dims):
        d = d_all[d_all["dimension"] == dim]
        if show_cloud:
            ax.scatter(d["n_feval_train"], d[metric], s=7, color="lightgrey",
                       edgecolor="none", zorder=1)
        for st in strats:
            sub = d[d["strategy"] == st]
            if sub.empty:
                continue
            # best setting at each budget, then the running maximum
            best = (sub.loc[sub.groupby("n_feval_train")[metric].idxmax()]
                    .sort_values("n_feval_train"))
            keep, run_max = [], -np.inf
            for _, r in best.iterrows():
                if r[metric] > run_max:
                    keep.append(r)
                    run_max = r[metric]
            fr = pd.DataFrame(keep)
            ax.plot(fr["n_feval_train"], fr[metric], marker="o", ms=4,
                    lw=1.6, color=color_of[st], label=st, zorder=3)
        if logx:
            ax.set_xscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("training budget (function evaluations)", fontsize=16)
        ax.set_title(f"dimension {dim}", fontsize=16)
        ax.grid(True, which="both", alpha=0.3)
    axf[0].set_ylabel("Mean accuracy across folds" , fontsize=16)

    handles = [Line2D([0], [0], color=color_of[s], marker="o", label=s)
               for s in strats]
    fig.tight_layout(rect=[0, 0, 0.85, 0.92])
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.86, 0.5),
               title="sampling strategy", frameon=False, fontsize=16)
    fig.suptitle(title or "Pareto-efficient frontier per sampling strategy",
                 fontsize=24)
    return fig, axes


def plot_spread_vs_budget(df, ratios, dimensions=(2, 5, 10),
                          metric="acc_all_runs", scale="sd", n_folds=5,
                          exclude=("cma_random",), n_runs_train=1,
                          normalise=True, show_rho=True, fit=True,
                          width=10.4, height=4.4, title=None):
    """How the difference between sampling strategies decays with budget.

    Each point is one budget group -- a fixed (dimension, size, instances, runs)
    at which every strategy incurs the same cost. The vertical axis is the
    spread in accuracy across the strategies within that group, and the
    horizontal axis is what the group cost.

    This unifies three observations that otherwise appear separate. The
    frontiers of Section~\\ref{sec:frontier} converge as the budget grows; the
    spread between strategies falls with dimension; and, as this figure shows,
    it also falls with budget WITHIN a dimension. The dimensional trend is not a
    separate effect but the same one aggregated, since the higher dimensions
    reach their ceiling at a smaller fraction of the budget range examined.

    `show_rho` colours each point by its within-group rank correlation, which
    separates the two things that decay differently: the ORDERING persists at
    every budget, while the SIZE of the difference does not.

    normalise=True divides the spread by the fold-to-fold variation of the
    group, so that a value of one marks the point at which the strategies cease
    to be distinguishable; set it False to read the spread in accuracy units.
    """
    dsem = add_sem(df, n_folds)
    col = "sem_all_runs" if scale == "sem" else "fold_acc_all_runs_std"
    keys = ["dimension", "size", "n_instances_train", "n_runs_train"]
    gsem = (dsem[~dsem["strategy"].isin(exclude)]
            .groupby(keys)[col].median().rename("ref").reset_index())

    rows = []
    for d in dimensions:
        _, summ = noise_vs_accuracy(df, ratios, d, metric, exclude, verbose=False)
        if n_runs_train is not None:
            summ = summ[summ["n_runs_train"] == n_runs_train]
        summ["dimension"] = d
        rows.append(summ)
    S = pd.concat(rows, ignore_index=True).merge(gsem, on=keys, how="left")
    S["budget"] = (24 * S["n_instances_train"] * S["n_runs_train"]
                   * S["size"] * S["dimension"])
    S["y"] = S["acc_spread"] / S["ref"] if normalise else S["acc_spread"]
    S = S[np.isfinite(S["y"]) & (S["budget"] > 0)]

    dims = sorted(S["dimension"].unique())
    cmap = plt.get_cmap("tab10")
    col_of = {d: cmap(i % 10) for i, d in enumerate(dims)}

    fig, ax = plt.subplots(figsize=(width, height))
    for d in dims:
        g = S[S["dimension"] == d].sort_values("budget")
        if show_rho:
            sc = ax.scatter(g["budget"], g["y"], c=g["rho"], cmap="coolwarm_r",
                            vmin=-1, vmax=0, s=46, edgecolor=col_of[d], lw=1.3,
                            zorder=3)
        else:
            ax.scatter(g["budget"], g["y"], s=40, color=col_of[d],
                       edgecolor="white", lw=0.5, zorder=3)
        if fit and len(g) > 2:
            lx, ly = np.log10(g["budget"]), np.log10(g["y"].clip(lower=1e-6))
            b = (np.sum((lx - lx.mean()) * (ly - ly.mean()))
                 / np.sum((lx - lx.mean()) ** 2))
            a = ly.mean() - b * lx.mean()
            xs = np.linspace(lx.min(), lx.max(), 40)
            ax.plot(10 ** xs, 10 ** (a + b * xs), color=col_of[d], lw=1.6,
                    alpha=0.75, zorder=2,
                    label=f"dim {d}  (slope {b:.2f})")
        else:
            ax.plot([], [], color=col_of[d], lw=1.6, label=f"dim {d}")

    if normalise:
        ax.axhline(1.0, color="#d62728", ls="--", lw=1.3)
        ax.text(ax.get_xlim()[1], 1.0,
                f"  1 {'SEM' if scale == 'sem' else 'SD'}", fontsize=8,
                color="#d62728", va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training budget (function evaluations)")
    ax.set_ylabel("accuracy spread across strategies"
                  + (f" / {'SEM' if scale == 'sem' else 'SD'}" if normalise else ""))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    if show_rho:
        cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("within-group Spearman $\\rho$", fontsize=9)
    fig.suptitle(title or "The difference between sampling strategies decays "
                          "with budget", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, ax


def confusion_pair(path, config_key, n_inst, n_runs, a=10, b=11,
                   protocol="all_runs"):
    """The 2x2 confusion block for one pair of functions, plus leakage.

    Returns the four rates within the pair -- P(predict a | true a),
    P(predict b | true a), and the two for true b -- together with the share of
    each class assigned to some THIRD function. The last matters: a fall in the
    a->b rate is only an improvement if it does not simply move the error
    elsewhere.

    `a` and `b` are 1-based BBOB function numbers.
    """
    p = load_predictions(path, config_key, n_inst, n_runs)
    if protocol == "all_runs":
        pr = p["pred_runs"]
        y = np.repeat(p["true_labels"], pr.shape[1])
        pred = pr.ravel()
    else:
        y, pred = p["true_labels"], p[f"pred_{protocol}"]
    ia, ib = a - 1, b - 1

    out = {}
    for name, t, o in ((f"f{a}", ia, ib), (f"f{b}", ib, ia)):
        m = y == t
        n = int(m.sum())
        if not n:
            out[name] = dict(correct=np.nan, to_other=np.nan, elsewhere=np.nan,
                             n=0)
            continue
        out[name] = dict(
            correct=float((pred[m] == t).mean()),
            to_other=float((pred[m] == o).mean()),
            elsewhere=float(((pred[m] != t) & (pred[m] != o)).mean()),
            n=n)
    return out

def plot_confusion_pair_grid(paths, config_key="lhs_random_cd_100",
                             a=10, b=11, instances=(1, 2, 3, 5, 7, 10, 15, 20),
                             n_runs=1, dimensions=None, show_elsewhere=True,
                             width_per=4.0, height=4.0, title=None):
    """How the confusion between two functions responds to training data.

    Rather than reading a single 24x24 matrix, this extracts the block for one
    pair and traces it across allocations. The upper line is the share of the
    class classified correctly and the lower the share assigned to the other
    member of the pair; where the two meet, the classifier is doing no better
    than chance between them.

    `show_elsewhere` adds the share assigned to any THIRD function as a dashed
    line. This distinguishes a genuine improvement from a redistribution: if the
    a->b rate falls while the elsewhere rate rises, the error has moved rather
    than gone.

    One panel per dimension, sharing a vertical axis so the levels at which the
    curves settle can be compared directly --- that comparison being the point,
    since it is the plateau LEVEL rather than its existence that differs between
    dimensions.
    """
    dims = sorted(dimensions or paths.keys())
    fig, axes = plt.subplots(1, len(dims), figsize=(width_per * len(dims), height),
                             squeeze=False, sharey=True)
    axf = axes[0]
    for ax, d in zip(axf, dims):
        rows = []
        for ni in instances:
            try:
                r = confusion_pair(paths[d], config_key, ni, n_runs, a, b)
            except (KeyError, OSError):
                continue
            rows.append((ni, r))
        if not rows:
            ax.set_visible(False)
            continue
        xs = [ni for ni, _ in rows]
        for cls, colour in ((f"f{a}", "#1f77b4"), (f"f{b}", "#d62728")):
            ax.plot(xs, [r[cls]["correct"] for _, r in rows], marker="o", ms=4.5,
                    lw=1.7, color=colour, label=f"{cls} correct")
            ax.plot(xs, [r[cls]["to_other"] for _, r in rows], marker="s", ms=4,
                    lw=1.5, ls="--", color=colour, alpha=0.75,
                    label=f"{cls} $\\rightarrow$ f{b if cls == f'f{a}' else a}")
            if show_elsewhere:
                ax.plot(xs, [r[cls]["elsewhere"] for _, r in rows], marker="^",
                        ms=3.5, lw=1.0, ls=":", color=colour, alpha=0.5,
                        label=f"{cls} elsewhere")
        ax.axhline(0.5, color="grey", ls=":", lw=1.0, alpha=0.7)
        ax.set_xlabel("training instances per function")
        ax.set_title(f"dimension {d}", fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axf[0].set_ylabel("share of the class")
    h, l = axf[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 0.82, 0.92])
    fig.legend(h, l, loc="center left", bbox_to_anchor=(0.83, 0.5),
               frameon=False, fontsize=8.5)
    fig.suptitle(title or f"Confusion between f{a} and f{b}  "
                          f"[{config_key}, {n_runs} run per instance]",
                 fontsize=12)
    return fig, axes


def _eta_squared_interactions(d, factors, response):
    """One-way shares plus every two-way interaction, for a BALANCED factorial.

    The one-way decomposition leaves a residual that mixes interaction with
    noise, and reporting it as "unexplained" concedes more than necessary. On a
    balanced full factorial the two-way terms are also orthogonal to the main
    effects and to each other, so they can be added by the same construction:

        SS(A x B) = SS(cells of A,B) - SS(A) - SS(B)

    where the first term treats each (A, B) combination as a group. What remains
    after the main effects and every two-way term is three-way and higher
    interaction plus noise.

    BALANCE IS REQUIRED. Where the design is unbalanced the terms overlap, the
    shares double-count and the residual can come out NEGATIVE, which is the
    signal that this construction does not apply. The caller should check for it.
    """
    from itertools import combinations
    y = d[response].to_numpy(float)
    grand = y.mean()
    ss_total = float(np.sum((y - grand) ** 2))
    if ss_total <= 0:
        return {}, {}

    def ss_of(cols):
        s = 0.0
        for _, g in d.groupby(list(cols), observed=True):
            s += len(g) * (g[response].mean() - grand) ** 2
        return float(s)

    main = {f: ss_of([f]) / ss_total for f in factors}
    inter = {}
    for a, b in combinations(factors, 2):
        cell = ss_of([a, b]) / ss_total
        inter[f"{a} x {b}"] = cell - main[a] - main[b]
    return main, inter


def factor_importance_2way(df, dimension=None, response="acc_all_runs",
                           factors=None, control_budget=True, budget_degree=1,
                           top=None, verbose=True):
    """Factor importance including every two-way interaction.

    Extends factor_importance by attributing part of what it reports as
    residual. A large interaction between the number of instances and the budget
    would, for instance, be the signature of diminishing returns: instances help
    a great deal when few have been used and little once many have.

    Requires a balanced full factorial. A negative residual indicates the design
    is not balanced -- most commonly because the frame has been filtered on
    something correlated with the factors, such as a budget range -- and the
    result should be discarded rather than interpreted.

    `budget_degree` sets the order of the polynomial in log(cost) that is
    removed. Degree 1 is the conservative default: accuracy saturates with
    budget, so a straight line under-removes the budget effect and leaves some
    curvature in the residual. A higher degree removes more of it, but the
    number of instances is collinear with the budget, so a more flexible budget
    term also absorbs part of the allocation effect the decomposition is meant
    to isolate. Running both is therefore a robustness check rather than a
    choice: if the ranking is unchanged, the leftover curvature is not driving
    it.
    """
    d = df if dimension is None else df[df["dimension"] == dimension]
    d = d[np.isfinite(d[response])].copy()
    factors = list(factors or FACTORS)
    if d.empty:
        raise ValueError("no rows")

    # balance check: every combination present the same number of times
    counts = d.groupby(factors, observed=True).size()
    balanced = bool(counts.nunique() == 1 and
                    len(counts) == int(np.prod([d[f].nunique() for f in factors])))

    resp = response
    r2 = np.nan
    if control_budget:
        x = np.log(d["n_feval_train"].to_numpy(float))
        y = d[response].to_numpy(float)
        coef = np.polyfit(x, y, budget_degree)
        fit = np.polyval(coef, x)
        d["_resid"] = y - fit
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum((y - fit) ** 2)) / ss_tot if ss_tot > 0 else np.nan
        resp = "_resid"

    main, inter = _eta_squared_interactions(d, factors, resp)
    residual = 1.0 - sum(main.values()) - sum(inter.values())

    rows = [{"term": f, "kind": "main", "share": v} for f, v in main.items()]
    rows += [{"term": k, "kind": "2-way", "share": v} for k, v in inter.items()]
    rows.append({"term": "3-way and higher, plus noise", "kind": "residual",
                 "share": residual})
    tab = pd.DataFrame(rows)
    tab.attrs["budget_r2"] = r2
    tab.attrs["budget_degree"] = budget_degree

    if verbose:
        n_dim = "all dims" if dimension is None else f"dim {dimension}"
        ctl = "budget-controlled" if control_budget else "raw"
        deg = "" if budget_degree == 1 else f", degree {budget_degree}"
        print(f"Variance in {response} explained, {ctl}{deg} "
              f"({n_dim}, {len(d):,} settings)")
        if control_budget:
            print(f"  budget term accounts for {r2:.1%} of the variance; "
                  f"the shares below divide the remaining {1 - r2:.1%}")
        if not balanced:
            print("  WARNING: the design is not balanced, so the terms are not "
                  "orthogonal.\n           Shares double-count and a negative "
                  "residual is possible; do not interpret.")
        print()
        m = tab[tab.kind == "main"].sort_values("share", ascending=False)
        print("  main effects")
        for _, r in m.iterrows():
            print(f"    {r['term']:<34s}{r['share']:7.1%}")
        i = tab[tab.kind == "2-way"].sort_values("share", ascending=False)
        if top:
            i = i.head(top)
        print("\n  two-way interactions")
        for _, r in i.iterrows():
            print(f"    {r['term']:<34s}{r['share']:7.1%}")
        print(f"\n  {'3-way and higher, plus noise':<34s}{residual:7.1%}")
        if residual < 0:
            print("\n  A negative residual confirms the design is unbalanced.")
    return tab