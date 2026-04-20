"""
Confusion matrix analysis for ELA classification results.

Usage in a Jupyter notebook:
    from confusion_matrix_analysis import ConfusionMatrixAnalysis
    cma = ConfusionMatrixAnalysis("/path/to/dim2/results")
    cma.load()

    # Heatmap for a single config
    cma.plot_confusion_matrix("ilhs_50", n_runs_train=5)

    # Side-by-side dim2 vs dim5
    cma2 = ConfusionMatrixAnalysis("/path/to/dim2/results")
    cma5 = ConfusionMatrixAnalysis("/path/to/dim5/results")
    cma2.load(); cma5.load()
    plot_dimension_comparison(cma2, cma5, "ilhs_50", n_runs_train=5)

    # Top confused pairs
    cma.top_confused_pairs("ilhs_50", n_runs_train=5, top_n=15)

    # Summary across configs
    cma.plot_n_confused_pairs_vs_runs("ilhs_50")
    cma.plot_confusion_by_bbob_group("ilhs_50", n_runs_train=5)
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# BBOB function metadata
# ---------------------------------------------------------------------------

BBOB_FUNCTION_NAMES = {
    1: "Sphere", 2: "Ellipsoidal (sep)", 3: "Rastrigin (sep)",
    4: "Büche-Rastrigin", 5: "Linear Slope",
    6: "Attractive Sector", 7: "Step Ellipsoidal",
    8: "Rosenbrock", 9: "Rosenbrock (rot)",
    10: "Ellipsoidal", 11: "Discus", 12: "Bent Cigar",
    13: "Sharp Ridge", 14: "Different Powers",
    15: "Rastrigin", 16: "Weierstrass",
    17: "Schaffers F7", 18: "Schaffers F7 (mod)",
    19: "Griewank-Rosenbrock",
    20: "Schwefel", 21: "Gallagher 101",
    22: "Gallagher 21", 23: "Katsuura", 24: "Lunacek bi-Rastrigin",
}

# BBOB function groups (0-indexed class labels)
BBOB_GROUPS = {
    "Separable": [0, 1, 2, 3, 4],
    "Low/mod. cond.": [5, 6, 7, 8],
    "High cond. unimodal": [9, 10, 11, 12, 13],
    "Multimodal (adequate)": [14, 15, 16, 17, 18],
    "Multimodal (weak)": [19, 20, 21, 22, 23],
}

GROUP_OF_FUNC = {}
for group_name, func_indices in BBOB_GROUPS.items():
    for fi in func_indices:
        GROUP_OF_FUNC[fi] = group_name

FUNC_LABELS_SHORT = [f"f{i+1}" for i in range(24)]
FUNC_LABELS_FULL = [f"f{i+1} {BBOB_FUNCTION_NAMES[i+1]}" for i in range(24)]


# ---------------------------------------------------------------------------
# Core analysis class
# ---------------------------------------------------------------------------

class ConfusionMatrixAnalysis:
    """Load and analyze confusion matrices from classification HDF5 results."""

    def __init__(self, result_dir, dimension=None):
        """
        Parameters
        ----------
        result_dir : str or Path
            Directory containing ela_classification_results_allruns.h5
        dimension : int, optional
            Dimension label for plot titles.
        """
        self.result_dir = Path(result_dir)
        self.h5_path = self.result_dir / "ela_classification_results_allruns.h5"
        self.dimension = dimension
        self.h5 = None
        self._configs = None

    def load(self):
        """Open the HDF5 file."""
        if not self.h5_path.exists():
            raise FileNotFoundError(f"{self.h5_path} not found")
        self.h5 = h5py.File(self.h5_path, "r")
        self._configs = list(self.h5.keys())
        print(f"Loaded {self.h5_path}")
        print(f"Configs: {self._configs}")
        # Show available n_runs_train for first config
        first = self._configs[0]
        subkeys = [k for k in self.h5[first].keys()]
        print(f"Example subkeys for '{first}': {subkeys}")

    def close(self):
        if self.h5 is not None:
            self.h5.close()

    @property
    def configs(self):
        return self._configs

    def get_n_runs_values(self, config_key):
        """Return available n_runs_train values for a config."""
        return sorted([
            int(k.split("_")[-1])
            for k in self.h5[config_key].keys()
            if k.startswith("n_runs_")
        ])

    def get_confusion_matrix(self, config_key, n_runs_train,
                             pred_type="median"):
        """
        Build the 24x24 confusion matrix from stored predictions.

        Parameters
        ----------
        config_key : str
        n_runs_train : int
        pred_type : str
            "median" — predict on median feature vector (one prediction per
                       instance).
            "runs"   — predict on each of the 30 individual runs. True labels
                       are repeated to match (30 predictions per instance).
            "majority_vote" — majority vote across 30 runs (one prediction
                              per instance).

        Returns
        -------
        cm : np.ndarray, shape (24, 24)
            Confusion matrix (rows=true, cols=predicted).
        """
        subkey = f"n_runs_{n_runs_train:02d}"
        grp = self.h5[config_key][subkey]
        true_labels = grp["true_labels"][:]

        if pred_type == "median":
            pred_labels = grp["pred_median"][:]
        elif pred_type == "runs":
            pred_runs = grp["pred_runs"][:]        # (n_instances, 30)
            true_labels = np.repeat(true_labels, pred_runs.shape[1])
            pred_labels = pred_runs.flatten()
        elif pred_type == "majority_vote":
            pred_labels = grp["pred_majority_vote"][:]
        else:
            raise ValueError(
                f"pred_type must be 'median', 'runs', or 'majority_vote', "
                f"got '{pred_type}'")

        cm = confusion_matrix(true_labels, pred_labels, labels=np.arange(24))
        return cm

    def get_confusion_matrix_normalized(self, config_key, n_runs_train,
                                        pred_type="median"):
        """Row-normalized confusion matrix (each row sums to 1)."""
        cm = self.get_confusion_matrix(config_key, n_runs_train, pred_type)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return cm / row_sums

    # -------------------------------------------------------------------
    # Plot: single confusion matrix heatmap
    # -------------------------------------------------------------------

    def plot_confusion_matrix(self, config_key, n_runs_train,
                              pred_type="median",
                              normalized=True, ax=None, figsize=(12, 10),
                              title=None, show_values=True, cmap="Blues"):
        """
        Plot a 24x24 confusion matrix heatmap.

        Parameters
        ----------
        config_key : str
        n_runs_train : int
        pred_type : str
            "median", "runs", or "majority_vote"
        normalized : bool
            If True, row-normalize (rates). If False, raw counts.
        """
        if normalized:
            cm = self.get_confusion_matrix_normalized(config_key, n_runs_train,
                                                      pred_type)
            fmt = ".2f"
            vmin, vmax = 0, 1
        else:
            cm = self.get_confusion_matrix(config_key, n_runs_train, pred_type)
            fmt = "d"
            vmin, vmax = None, None

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Mask diagonal for off-diagonal emphasis (optional: set diagonal
        # separately). We'll show the full matrix but use annotation control.
        sns.heatmap(
            cm, ax=ax, cmap=cmap,
            xticklabels=FUNC_LABELS_SHORT,
            yticklabels=FUNC_LABELS_SHORT,
            vmin=vmin, vmax=vmax,
            annot=show_values if cm.shape[0] <= 24 else False,
            fmt=fmt,
            annot_kws={"size": 7},
            linewidths=0.3, linecolor="white",
            cbar_kws={"shrink": 0.8},
        )

        # Add BBOB group separators
        group_boundaries = [0, 5, 9, 14, 19, 24]
        for b in group_boundaries[1:-1]:
            ax.axhline(y=b, color="black", linewidth=1.5)
            ax.axvline(x=b, color="black", linewidth=1.5)

        dim_str = f"d={self.dimension}" if self.dimension else ""
        if title is None:
            title = f"Confusion Matrix — {config_key}, runs={n_runs_train}, pred={pred_type} {dim_str}"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        plt.tight_layout()
        return fig, ax

    # -------------------------------------------------------------------
    # Top confused pairs
    # -------------------------------------------------------------------

    def top_confused_pairs(self, config_key, n_runs_train, top_n=15,
                           pred_type="median"):
        """
        Return a DataFrame of the most confused (true, predicted) pairs.

        Parameters
        ----------
        config_key : str
        n_runs_train : int
        top_n : int

        Returns
        -------
        df : pd.DataFrame
        """
        cm_norm = self.get_confusion_matrix_normalized(config_key, n_runs_train, pred_type)
        cm_raw = self.get_confusion_matrix(config_key, n_runs_train, pred_type)

        pairs = []
        for i in range(24):
            for j in range(24):
                if i == j:
                    continue
                if cm_norm[i, j] > 0:
                    pairs.append({
                        "true_func": f"f{i+1}",
                        "true_name": BBOB_FUNCTION_NAMES[i + 1],
                        "true_group": GROUP_OF_FUNC[i],
                        "pred_func": f"f{j+1}",
                        "pred_name": BBOB_FUNCTION_NAMES[j + 1],
                        "pred_group": GROUP_OF_FUNC[j],
                        "same_group": GROUP_OF_FUNC[i] == GROUP_OF_FUNC[j],
                        "error_rate": cm_norm[i, j],
                        "error_count": cm_raw[i, j],
                    })

        df = pd.DataFrame(pairs)
        df.sort_values("error_rate", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df.head(top_n)

    # -------------------------------------------------------------------
    # Group-level confusion
    # -------------------------------------------------------------------

    def get_group_confusion_matrix(self, config_key, n_runs_train,
                                    pred_type="median"):
        """
        Build a 5x5 confusion matrix at the BBOB group level.

        Returns
        -------
        group_cm : np.ndarray, shape (5, 5)
        group_names : list of str
        """
        cm = self.get_confusion_matrix(config_key, n_runs_train, pred_type)
        group_names = list(BBOB_GROUPS.keys())
        group_cm = np.zeros((5, 5))

        for gi, (gname_i, funcs_i) in enumerate(BBOB_GROUPS.items()):
            for gj, (gname_j, funcs_j) in enumerate(BBOB_GROUPS.items()):
                group_cm[gi, gj] = sum(
                    cm[fi, fj] for fi in funcs_i for fj in funcs_j
                )

        # Row-normalize
        row_sums = group_cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        group_cm_norm = group_cm / row_sums
        return group_cm_norm, group_names

    def plot_confusion_by_bbob_group(self, config_key, n_runs_train,
                                      pred_type="median",
                                      ax=None, figsize=(8, 6), title=None):
        """Plot a 5x5 group-level confusion matrix."""
        group_cm, group_names = self.get_group_confusion_matrix(
            config_key, n_runs_train, pred_type)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        sns.heatmap(
            group_cm, ax=ax, cmap="Blues",
            xticklabels=group_names, yticklabels=group_names,
            annot=True, fmt=".3f",
            vmin=0, vmax=1,
            linewidths=1, linecolor="white",
            cbar_kws={"shrink": 0.8},
        )

        dim_str = f"d={self.dimension}" if self.dimension else ""
        if title is None:
            title = (f"Group Confusion — {config_key}, "
                     f"runs={n_runs_train} {dim_str}")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted group", fontsize=11)
        ax.set_ylabel("True group", fontsize=11)
        plt.tight_layout()
        return fig, ax

    # -------------------------------------------------------------------
    # Number of confused pairs vs n_runs_train
    # -------------------------------------------------------------------

    def count_confused_pairs(self, config_key, n_runs_train, threshold=0.05,
                             pred_type="median"):
        """Count function pairs with off-diagonal error rate > threshold."""
        cm_norm = self.get_confusion_matrix_normalized(config_key, n_runs_train,
                                                       pred_type)
        np.fill_diagonal(cm_norm, 0)
        return np.sum(cm_norm > threshold)

    def plot_n_confused_pairs_vs_runs(self, config_key, threshold=0.05,
                                      pred_type="median",
                                      ax=None, figsize=(8, 5)):
        """
        Plot number of confused pairs (error > threshold) vs n_runs_train.
        """
        n_runs_values = self.get_n_runs_values(config_key)
        counts = [
            self.count_confused_pairs(config_key, nr, threshold, pred_type)
            for nr in n_runs_values
        ]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        ax.plot(n_runs_values, counts, "o-", linewidth=2, markersize=8)
        ax.set_xlabel("Runs per training instance", fontsize=11)
        ax.set_ylabel(f"Confused pairs (error > {threshold:.0%})", fontsize=11)
        dim_str = f"d={self.dimension}" if self.dimension else ""
        ax.set_title(
            f"Number of confused pairs — {config_key} {dim_str}",
            fontsize=13, fontweight="bold")
        ax.set_xticks(n_runs_values)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax

    # -------------------------------------------------------------------
    # Per-function accuracy
    # -------------------------------------------------------------------

    def per_function_accuracy(self, config_key, n_runs_train,
                              pred_type="median"):
        """Return per-function accuracy as a Series."""
        cm = self.get_confusion_matrix(config_key, n_runs_train, pred_type)
        correct = np.diag(cm)
        total = cm.sum(axis=1)
        total[total == 0] = 1
        acc = correct / total
        return pd.Series(acc, index=FUNC_LABELS_SHORT, name="accuracy")

    def plot_per_function_accuracy(self, config_key, n_runs_train,
                                    pred_type="median",
                                    ax=None, figsize=(14, 5)):
        """Bar chart of per-function accuracy."""
        acc = self.per_function_accuracy(config_key, n_runs_train, pred_type)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        colors = []
        group_colors = {
            "Separable": "#4e79a7",
            "Low/mod. cond.": "#f28e2b",
            "High cond. unimodal": "#e15759",
            "Multimodal (adequate)": "#76b7b2",
            "Multimodal (weak)": "#59a14f",
        }
        for i in range(24):
            colors.append(group_colors[GROUP_OF_FUNC[i]])

        ax.bar(range(24), acc.values, color=colors, edgecolor="white",
               linewidth=0.5)
        ax.set_xticks(range(24))
        ax.set_xticklabels(FUNC_LABELS_SHORT, rotation=45, ha="right")
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=acc.mean(), color="black", linestyle="--", alpha=0.5,
                    label=f"Mean: {acc.mean():.3f}")
        ax.legend()

        # Add group labels
        group_boundaries = [0, 5, 9, 14, 19, 24]
        for gi, (gname, _) in enumerate(BBOB_GROUPS.items()):
            mid = (group_boundaries[gi] + group_boundaries[gi + 1]) / 2 - 0.5
            ax.text(mid, -0.12, gname, ha="center", va="top", fontsize=8,
                    transform=ax.get_xaxis_transform(), fontstyle="italic")

        dim_str = f"d={self.dimension}" if self.dimension else ""
        ax.set_title(
            f"Per-function accuracy — {config_key}, runs={n_runs_train} "
            f"{dim_str}",
            fontsize=13, fontweight="bold")
        plt.tight_layout()
        return fig, ax

    # -------------------------------------------------------------------
    # Off-diagonal error heatmap (diagonal zeroed)
    # -------------------------------------------------------------------

    def plot_error_matrix(self, config_key, n_runs_train, pred_type="median",
                           ax=None, figsize=(12, 10), title=None):
        """
        Plot only off-diagonal errors (diagonal set to 0).
        Uses a diverging colormap to highlight misclassification hotspots.
        """
        cm_norm = self.get_confusion_matrix_normalized(config_key, n_runs_train,
                                                       pred_type)
        np.fill_diagonal(cm_norm, 0)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        max_err = cm_norm.max()
        sns.heatmap(
            cm_norm, ax=ax, cmap="Reds",
            xticklabels=FUNC_LABELS_SHORT,
            yticklabels=FUNC_LABELS_SHORT,
            vmin=0, vmax=max(max_err, 0.1),
            annot=True, fmt=".2f",
            annot_kws={"size": 7},
            linewidths=0.3, linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "Error rate"},
        )

        group_boundaries = [0, 5, 9, 14, 19, 24]
        for b in group_boundaries[1:-1]:
            ax.axhline(y=b, color="black", linewidth=1.5)
            ax.axvline(x=b, color="black", linewidth=1.5)

        dim_str = f"d={self.dimension}" if self.dimension else ""
        if title is None:
            title = (f"Misclassification rates — {config_key}, "
                     f"runs={n_runs_train} {dim_str}")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        plt.tight_layout()
        return fig, ax

    # -------------------------------------------------------------------
    # Compare across all sampling strategies
    # -------------------------------------------------------------------

    def plot_confused_pairs_all_configs(self, n_runs_train=5, threshold=0.05,
                                         pred_type="median",
                                         ax=None, figsize=(10, 6)):
        """
        Bar chart of number of confused pairs per config for a given n_runs.
        """
        configs = sorted(self._configs)
        counts = []
        for c in configs:
            nrv = self.get_n_runs_values(c)
            if n_runs_train in nrv:
                counts.append(self.count_confused_pairs(c, n_runs_train,
                                                        threshold, pred_type))
            else:
                counts.append(np.nan)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        ax.barh(range(len(configs)), counts, color="#4e79a7",
                edgecolor="white")
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(configs, fontsize=9)
        ax.set_xlabel(f"Confused pairs (error > {threshold:.0%})", fontsize=11)
        dim_str = f"d={self.dimension}" if self.dimension else ""
        ax.set_title(
            f"Confused pairs per config — runs={n_runs_train} {dim_str}",
            fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        return fig, ax

    # -------------------------------------------------------------------
    # Persistent confusions across configs
    # -------------------------------------------------------------------

    def persistent_confusions(self, n_runs_train=5, threshold=0.05,
                              pred_type="median"):
        """
        Find function pairs that are confused across ALL configs.

        Returns a DataFrame of pairs sorted by mean error rate.
        """
        all_cms = {}
        for config in self._configs:
            nrv = self.get_n_runs_values(config)
            if n_runs_train in nrv:
                all_cms[config] = self.get_confusion_matrix_normalized(
                    config, n_runs_train, pred_type)

        if not all_cms:
            return pd.DataFrame()

        pairs = []
        for i in range(24):
            for j in range(24):
                if i == j:
                    continue
                rates = [cm[i, j] for cm in all_cms.values()]
                n_above = sum(r > threshold for r in rates)
                if n_above > 0:
                    pairs.append({
                        "true_func": f"f{i+1}",
                        "true_name": BBOB_FUNCTION_NAMES[i + 1],
                        "pred_func": f"f{j+1}",
                        "pred_name": BBOB_FUNCTION_NAMES[j + 1],
                        "same_group": GROUP_OF_FUNC[i] == GROUP_OF_FUNC[j],
                        "mean_error": np.mean(rates),
                        "max_error": np.max(rates),
                        "n_configs_above_threshold": n_above,
                        "n_configs_total": len(all_cms),
                    })

        df = pd.DataFrame(pairs)
        df.sort_values(["n_configs_above_threshold", "mean_error"],
                       ascending=[False, False], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


# ---------------------------------------------------------------------------
# Cross-dimension comparison (standalone function)
# ---------------------------------------------------------------------------

def plot_dimension_comparison(cma_dim2, cma_dim5, config_key, n_runs_train,
                               pred_type="median", figsize=(22, 9)):
    """
    Side-by-side confusion matrices for dim=2 and dim=5.

    Parameters
    ----------
    cma_dim2, cma_dim5 : ConfusionMatrixAnalysis
    config_key : str
    n_runs_train : int
    pred_type : str
        "median", "runs", or "majority_vote"
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    cma_dim2.plot_confusion_matrix(
        config_key, n_runs_train, pred_type=pred_type, ax=axes[0],
        title=f"{config_key}, runs={n_runs_train}, pred={pred_type}, d=2")
    cma_dim5.plot_confusion_matrix(
        config_key, n_runs_train, pred_type=pred_type, ax=axes[1],
        title=f"{config_key}, runs={n_runs_train}, pred={pred_type}, d=5")

    plt.tight_layout()
    return fig, axes


def plot_error_comparison(cma_dim2, cma_dim5, config_key, n_runs_train,
                           pred_type="median", figsize=(22, 9)):
    """
    Side-by-side error matrices (diagonal zeroed) for dim=2 and dim=5.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Get both matrices to set a shared color scale
    cm2 = cma_dim2.get_confusion_matrix_normalized(config_key, n_runs_train,
                                                    pred_type)
    cm5 = cma_dim5.get_confusion_matrix_normalized(config_key, n_runs_train,
                                                    pred_type)
    np.fill_diagonal(cm2, 0)
    np.fill_diagonal(cm5, 0)
    shared_max = max(cm2.max(), cm5.max(), 0.1)

    for ax_i, (cma, cm, dim_label) in enumerate([
        (cma_dim2, cm2, "d=2"), (cma_dim5, cm5, "d=5")
    ]):
        sns.heatmap(
            cm, ax=axes[ax_i], cmap="Reds",
            xticklabels=FUNC_LABELS_SHORT,
            yticklabels=FUNC_LABELS_SHORT,
            vmin=0, vmax=shared_max,
            annot=True, fmt=".2f", annot_kws={"size": 7},
            linewidths=0.3, linecolor="white",
            cbar_kws={"shrink": 0.8},
        )
        group_boundaries = [0, 5, 9, 14, 19, 24]
        for b in group_boundaries[1:-1]:
            axes[ax_i].axhline(y=b, color="black", linewidth=1.5)
            axes[ax_i].axvline(x=b, color="black", linewidth=1.5)
        axes[ax_i].set_title(
            f"Errors — {config_key}, runs={n_runs_train}, {dim_label}",
            fontsize=13, fontweight="bold")
        axes[ax_i].set_xlabel("Predicted", fontsize=11)
        axes[ax_i].set_ylabel("True", fontsize=11)

    plt.tight_layout()
    return fig, axes


def compare_top_confusions(cma_dim2, cma_dim5, config_key, n_runs_train,
                            pred_type="median", top_n=15):
    """
    Compare top confused pairs between two dimensions.

    Returns merged DataFrame showing which confusions persist or resolve.
    """
    df2 = cma_dim2.top_confused_pairs(config_key, n_runs_train, top_n=100,
                                       pred_type=pred_type)
    df5 = cma_dim5.top_confused_pairs(config_key, n_runs_train, top_n=100,
                                       pred_type=pred_type)

    df2 = df2.rename(columns={"error_rate": "error_d2", "error_count": "count_d2"})
    df5 = df5.rename(columns={"error_rate": "error_d5", "error_count": "count_d5"})

    merged = pd.merge(
        df2[["true_func", "pred_func", "true_name", "pred_name",
             "same_group", "error_d2", "count_d2"]],
        df5[["true_func", "pred_func", "error_d5", "count_d5"]],
        on=["true_func", "pred_func"],
        how="outer",
    )
    merged["error_d2"] = merged["error_d2"].fillna(0)
    merged["error_d5"] = merged["error_d5"].fillna(0)
    merged["resolved_in_5d"] = (merged["error_d2"] > 0.05) & (merged["error_d5"] < 0.01)
    merged["worse_in_5d"] = merged["error_d5"] > merged["error_d2"]
    merged.sort_values("error_d2", ascending=False, inplace=True)
    merged.reset_index(drop=True, inplace=True)

    return merged.head(top_n)