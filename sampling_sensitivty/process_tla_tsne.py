"""
Precompute t-SNE embeddings for TLA features.

Three modes per configuration:
  1) instance-level mean (2400 points)
  2) instance-level median (2400 points)
  3) all runs (72000 points)

Uses openTSNE for speed. Standardizes features, applies PCA pre-reduction,
then computes 2D embeddings. Memory-efficient: loads one run at a time for
all_runs mode.

Usage:
  python precompute_tsne_tla.py /path/to/tla/h5/dir
  python precompute_tsne_tla.py /path/to/tla/h5/dir --output-dir /path/to/output
  python precompute_tsne_tla.py /path/to/tla/h5/dir --segment volume_h0
  python precompute_tsne_tla.py /path/to/tla/h5/dir --configs ilhs_50 sobol_50
  python precompute_tsne_tla.py /path/to/tla/h5/dir --modes mean median
"""

import argparse
import numpy as np
import h5py
import os
import gc
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from openTSNE import TSNE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_FUNCTIONS = 24
N_INSTANCES = 100
N_RUNS = 30
DIMENSION = 2

TSNE_PERPLEXITY = 40
TSNE_RANDOM_STATE = 42
TSNE_N_ITER = 1000
PCA_COMPONENTS = 50

TLA_PERSPECTIVES = ["volume", "axis"]
TLA_HOMOLOGIES = ["h0", "h1", "h2"]
TLA_FEATURE_LENGTHS = {"h0": 100, "h1": 10000, "h2": 10000}

TLA_SEGMENTS = {
    "all": (None, None),
    "volume_all": ("volume", None),
    "axis_all": ("axis", None),
    "volume_h0": ("volume", "h0"),
    "volume_h1": ("volume", "h1"),
    "volume_h2": ("volume", "h2"),
    "axis_h0": ("axis", "h0"),
    "axis_h1": ("axis", "h1"),
    "axis_h2": ("axis", "h2"),
}

TLA_FILES = {
    "ilhs_10": "ilhs_10_tla.h5", "ilhs_25": "ilhs_25_tla.h5",
    "ilhs_50": "ilhs_50_tla.h5", "ilhs_75": "ilhs_75_tla.h5",
    "ilhs_100": "ilhs_100_tla.h5",
    "lhs_10": "lhs_10_tla.h5", "lhs_25": "lhs_25_tla.h5",
    "lhs_50": "lhs_50_tla.h5", "lhs_75": "lhs_75_tla.h5",
    "lhs_100": "lhs_100_tla.h5",
    "sobol_10": "sobol_10_tla.h5", "sobol_25": "sobol_25_tla.h5",
    "sobol_50": "sobol_50_tla.h5", "sobol_75": "sobol_75_tla.h5",
    "sobol_100": "sobol_100_tla.h5",
    "uniform_10": "uniform_10_tla.h5", "uniform_25": "uniform_25_tla.h5",
    "uniform_50": "uniform_50_tla.h5", "uniform_75": "uniform_75_tla.h5",
    "uniform_100": "uniform_100_tla.h5",
    "cma_random_10": "cma_random_10_tla.h5", "cma_random_25": "cma_random_25_tla.h5",
    "cma_random_50": "cma_random_50_tla.h5", "cma_random_75": "cma_random_75_tla.h5",
    "cma_random_100": "cma_random_100_tla.h5",
}

ALL_MODES = ["mean", "median", "all_runs"]

TRUE_LABELS = np.repeat(np.arange(N_FUNCTIONS), N_INSTANCES)
TRUE_LABELS_ALL = np.repeat(np.arange(N_FUNCTIONS), N_INSTANCES * N_RUNS)


# ---------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------

def _get_segment_specs(perspective_filter, homology_filter):
    """Return list of (perspective, homology) pairs for a segment."""
    specs = []
    for persp in TLA_PERSPECTIVES:
        if perspective_filter is not None and persp != perspective_filter:
            continue
        for hom in TLA_HOMOLOGIES:
            if homology_filter is not None and hom != homology_filter:
                continue
            specs.append((persp, hom))
    return specs


def _get_segment_length(specs):
    return sum(TLA_FEATURE_LENGTHS[hom] for _, hom in specs)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_instance_data(h5_file, func_id, inst_id, specs):
    """
    Load all runs for one instance, one segment.
    Returns (N_RUNS, seg_len) array.
    """
    key = f"{func_id}_{inst_id}_{DIMENSION}"
    group = h5_file[key]
    seg_len = _get_segment_length(specs)
    data = np.empty((N_RUNS, seg_len))

    offset = 0
    for persp, hom in specs:
        feat_len = TLA_FEATURE_LENGTHS[hom]
        arr = group[persp][hom][:]
        arr = arr.reshape(arr.shape[0], -1)  # (N_RUNS, feat_len)
        data[:, offset:offset + feat_len] = arr
        offset += feat_len

    return data


def _build_collapsed_matrix(filepath, specs, agg_func):
    """
    Build (2400, seg_len) matrix with runs collapsed by agg_func.
    Loads one instance at a time for memory efficiency.
    """
    seg_len = _get_segment_length(specs)
    matrix = np.empty((N_FUNCTIONS * N_INSTANCES, seg_len))

    with h5py.File(filepath, "r") as f:
        for func_idx in range(N_FUNCTIONS):
            func_id = func_idx + 1
            for inst_idx in range(N_INSTANCES):
                inst_id = inst_idx + 1
                row = func_idx * N_INSTANCES + inst_idx
                inst_data = _load_instance_data(f, func_id, inst_id, specs)
                matrix[row] = agg_func(inst_data, axis=0)

    return matrix


def _build_all_runs_matrix(filepath, specs):
    """
    Build (72000, seg_len) matrix — all runs, all instances.
    Row order: func0_inst0_run0..run29, func0_inst1_run0..run29, ...
    """
    seg_len = _get_segment_length(specs)
    matrix = np.empty((N_FUNCTIONS * N_INSTANCES * N_RUNS, seg_len))

    with h5py.File(filepath, "r") as f:
        for func_idx in range(N_FUNCTIONS):
            func_id = func_idx + 1
            for inst_idx in range(N_INSTANCES):
                inst_id = inst_idx + 1
                base_row = (func_idx * N_INSTANCES + inst_idx) * N_RUNS
                inst_data = _load_instance_data(f, func_id, inst_id, specs)
                matrix[base_row:base_row + N_RUNS] = inst_data

    return matrix


# ---------------------------------------------------------------------------
# t-SNE computation
# ---------------------------------------------------------------------------

def _compute_tsne(features, pca_components=PCA_COMPONENTS, perplexity=TSNE_PERPLEXITY):
    """Standardize, PCA, then t-SNE to 2D."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Remove constant/NaN columns
    nan_cols = np.any(~np.isfinite(X), axis=0)
    if np.any(nan_cols):
        n_removed = int(np.sum(nan_cols))
        print(f"    Removed {n_removed} non-finite columns")
        X = X[:, ~nan_cols]

    X = np.nan_to_num(X, nan=0.0)

    # PCA pre-reduction
    if pca_components is not None and X.shape[1] > pca_components:
        pca = PCA(n_components=pca_components, random_state=TSNE_RANDOM_STATE)
        X = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_.sum()
        print(f"    PCA: {features.shape[1]} -> {X.shape[1]} ({explained:.1%} variance)")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=TSNE_RANDOM_STATE,
        n_iter=TSNE_N_ITER,
        initialization="pca",
        n_jobs=-1,
    )
    embedding = tsne.fit(X)
    return np.array(embedding)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir=None, segment="all", configs=None, modes=None,
         pca_components=PCA_COMPONENTS):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    input_dir = Path(input_dir)
    output_file = Path(output_dir) / f"tla_tsne_embeddings_{segment}.h5"
    modes = modes or ALL_MODES

    # Parse segment
    if segment not in TLA_SEGMENTS:
        print(f"Unknown segment: {segment}")
        print(f"Available: {list(TLA_SEGMENTS.keys())}")
        return
    perspective_filter, homology_filter = TLA_SEGMENTS[segment]
    specs = _get_segment_specs(perspective_filter, homology_filter)
    seg_len = _get_segment_length(specs)

    # Determine configs
    if configs:
        config_keys = [c for c in configs if c in TLA_FILES]
    else:
        config_keys = list(TLA_FILES.keys())

    # Memory estimates
    collapsed_gb = N_FUNCTIONS * N_INSTANCES * seg_len * 8 / (1024**3)
    all_runs_gb = N_FUNCTIONS * N_INSTANCES * N_RUNS * seg_len * 8 / (1024**3)

    print(f"TLA t-SNE Preprocessing")
    print(f"Segment: {segment} ({seg_len} features)")
    print(f"Specs: {specs}")
    print(f"Modes: {modes}")
    print(f"Configs: {len(config_keys)}")
    print(f"PCA components: {pca_components}")
    print(f"Perplexity: {TSNE_PERPLEXITY}")
    print(f"Memory estimates: collapsed={collapsed_gb:.2f} GB, all_runs={all_runs_gb:.2f} GB")
    print()

    with h5py.File(output_file, "a") as out:
        # Save labels once
        if "labels" not in out:
            out.create_dataset("labels", data=TRUE_LABELS)
        if "labels_all_runs" not in out:
            out.create_dataset("labels_all_runs", data=TRUE_LABELS_ALL)
        out.attrs["segment"] = segment
        out.attrs["segment_length"] = seg_len

        for config_key in config_keys:
            filename = TLA_FILES[config_key]
            filepath = input_dir / filename
            if not filepath.exists():
                print(f"WARNING: {filepath} not found, skipping.")
                continue

            print(f"{'='*60}")
            print(f"Processing: {config_key} ({segment})")
            print(f"{'='*60}")

            if config_key in out:
                config_grp = out[config_key]
            else:
                config_grp = out.create_group(config_key)

            # Mean
            if "mean" in modes:
                if "mean" in config_grp:
                    print(f"  mean: already exists, skipping")
                else:
                    print(f"  Building mean matrix...")
                    matrix = _build_collapsed_matrix(filepath, specs, np.mean)
                    print(f"    Shape: {matrix.shape}")
                    print(f"  Computing t-SNE (mean)...")
                    embedding = _compute_tsne(matrix, pca_components)
                    config_grp.create_dataset("mean", data=embedding)
                    print(f"    Done. Embedding shape: {embedding.shape}")
                    del matrix, embedding
                    gc.collect()

            # Median
            if "median" in modes:
                if "median" in config_grp:
                    print(f"  median: already exists, skipping")
                else:
                    print(f"  Building median matrix...")
                    matrix = _build_collapsed_matrix(filepath, specs, np.median)
                    print(f"    Shape: {matrix.shape}")
                    print(f"  Computing t-SNE (median)...")
                    embedding = _compute_tsne(matrix, pca_components)
                    config_grp.create_dataset("median", data=embedding)
                    print(f"    Done. Embedding shape: {embedding.shape}")
                    del matrix, embedding
                    gc.collect()

            # All runs
            if "all_runs" in modes:
                if "all_runs" in config_grp:
                    print(f"  all_runs: already exists, skipping")
                else:
                    print(f"  Building all-runs matrix...")
                    matrix = _build_all_runs_matrix(filepath, specs)
                    print(f"    Shape: {matrix.shape}")
                    print(f"  Computing t-SNE (all_runs) — this may take a while...")
                    embedding = _compute_tsne(matrix, pca_components)
                    config_grp.create_dataset("all_runs", data=embedding)
                    print(f"    Done. Embedding shape: {embedding.shape}")
                    del matrix, embedding
                    gc.collect()

            print()

    print(f"\nResults saved to: {output_file}")
    print(f"\nOutput h5 structure:")
    print(f"  labels              ({N_FUNCTIONS * N_INSTANCES},)")
    print(f"  labels_all_runs     ({N_FUNCTIONS * N_INSTANCES * N_RUNS},)")
    print(f"  attrs: segment, segment_length")
    print(f"  {{config_key}}/")
    print(f"    mean              ({N_FUNCTIONS * N_INSTANCES}, 2)")
    print(f"    median            ({N_FUNCTIONS * N_INSTANCES}, 2)")
    print(f"    all_runs          ({N_FUNCTIONS * N_INSTANCES * N_RUNS}, 2)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute t-SNE embeddings for TLA features."
    )
    parser.add_argument("input_dir", type=str,
                        help="Directory containing TLA h5 files.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output h5. Defaults to input_dir.")
    parser.add_argument("--segment", type=str, default="all",
                        choices=list(TLA_SEGMENTS.keys()),
                        help="Which TLA segment to process (default: all).")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific configs to process (e.g. ilhs_50 sobol_50).")
    parser.add_argument("--modes", nargs="+", default=None,
                        choices=ALL_MODES,
                        help="Which modes to compute (default: all three).")
    parser.add_argument("--pca-components", type=int, default=PCA_COMPONENTS,
                        help=f"PCA components for pre-reduction (default: {PCA_COMPONENTS}).")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir,
         segment=args.segment, configs=args.configs, modes=args.modes,
         pca_components=args.pca_components)