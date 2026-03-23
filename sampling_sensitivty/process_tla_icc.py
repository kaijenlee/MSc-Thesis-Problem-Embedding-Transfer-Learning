"""
Compute ICC for TLA features across 3 perspectives:
  Perspective 1: Overall (all 2400 instances) — computed in feature segments
  Perspective 2: Per BBOB function group (5 groups)
  Perspective 3: Per function class (24 classes)

Memory-efficient: never loads more than one function group at a time.
Perspective 1 processes features in segments to stay within memory limits.

Peak memory usage:
  - Perspective 3: ~0.96 GB (1 class: 100 x 30 x 40200 x 8 bytes)
  - Perspective 2: ~4.8 GB  (max 5 classes stacked)
  - Perspective 1: ~5.7 GB  (2400 x 30 x 10000 for largest segment)

Output: h5 file with per-element ICC values and summary statistics.
ICC implementation is validated against pingouin at startup.
"""

import argparse
import numpy as np
import pandas as pd
import pingouin as pg
import h5py
import os
import gc
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FILES = [
    "ilhs_10_tla.h5", "ilhs_25_tla.h5", "ilhs_50_tla.h5", "ilhs_75_tla.h5", "ilhs_100_tla.h5",
    "lhs_10_tla.h5", "lhs_25_tla.h5", "lhs_50_tla.h5", "lhs_75_tla.h5", "lhs_100_tla.h5",
    "sobol_10_tla.h5", "sobol_25_tla.h5", "sobol_50_tla.h5", "sobol_75_tla.h5", "sobol_100_tla.h5",
    "uniform_10_tla.h5", "uniform_25_tla.h5", "uniform_50_tla.h5", "uniform_75_tla.h5", "uniform_100_tla.h5",
    "cma_random_10_tla.h5", "cma_random_25_tla.h5", "cma_random_50_tla.h5", "cma_random_75_tla.h5", "cma_random_100_tla.h5",
]

N_FUNCTIONS = 24
N_INSTANCES = 100
N_RUNS = 30
DIMENSION = 2

PERSPECTIVES_LIST = ["volume", "axis"]
HOMOLOGIES = ["h0", "h1", "h2"]
FEATURE_LENGTHS = {"h0": 100, "h1": 10000, "h2": 10000}
TOTAL_FEATURES = 2 * (100 + 10000 + 10000)  # 40200

# Feature segments for chunk-based processing
FEATURE_SEGMENTS = []
_offset = 0
for _persp in PERSPECTIVES_LIST:
    for _hom in HOMOLOGIES:
        _flen = FEATURE_LENGTHS[_hom]
        FEATURE_SEGMENTS.append((_persp, _hom, _offset, _flen))
        _offset += _flen

# BBOB function groups
FUNCTION_GROUPS = {
    "separable": [1, 2, 3, 4, 5],
    "low_moderate_conditioning": [6, 7, 8, 9],
    "high_conditioning": [10, 11, 12, 13, 14],
    "multimodal_adequate": [15, 16, 17, 18, 19],
    "multimodal_weak": [20, 21, 22, 23, 24],
}


# ---------------------------------------------------------------------------
# ICC(2,1) computation
# ---------------------------------------------------------------------------

def compute_icc_2_1(data):
    """
    Compute ICC(2,1) two-way random effects, absolute agreement, single measures.

    Parameters
    ----------
    data : np.ndarray, shape (n_subjects, n_raters)

    Returns
    -------
    float
        ICC(2,1) value. Returns np.nan if degenerate.
    """
    n, k = data.shape
    if n < 2 or k < 2:
        return np.nan

    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    df_rows = n - 1
    df_cols = k - 1
    df_error = df_rows * df_cols

    if df_rows == 0 or df_error == 0:
        return np.nan

    ms_rows = ss_rows / df_rows
    ms_cols = ss_cols / df_cols
    ms_error = ss_error / df_error

    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denominator == 0:
        return np.nan

    icc = (ms_rows - ms_error) / denominator
    return icc


def compute_icc_batch(data_3d):
    """
    Compute ICC(2,1) for each feature element.

    Parameters
    ----------
    data_3d : np.ndarray, shape (n_instances, n_runs, n_features)

    Returns
    -------
    np.ndarray, shape (n_features,)
    """
    n_instances, n_runs, n_features = data_3d.shape
    icc_values = np.empty(n_features)

    for f in range(n_features):
        icc_values[f] = compute_icc_2_1(data_3d[:, :, f])

    return icc_values


def validate_icc_against_pingouin(n_tests=5, n_subjects=50, n_raters=30, atol=1e-6):
    """
    Validate manual ICC(2,1) against pingouin. Raises AssertionError on mismatch.
    """
    print("Validating ICC implementation against pingouin...")
    np.random.seed(42)

    for i in range(n_tests):
        true_vals = np.random.randn(n_subjects) * (i + 1)
        noise_scale = 0.1 * (i + 1)
        data = true_vals[:, None] + np.random.randn(n_subjects, n_raters) * noise_scale

        icc_manual = compute_icc_2_1(data)

        records = []
        for subj in range(n_subjects):
            for rater in range(n_raters):
                records.append({
                    "targets": subj, "raters": rater,
                    "ratings": data[subj, rater],
                })
        df = pd.DataFrame(records)
        icc_table = pg.intraclass_corr(
            data=df, targets="targets", raters="raters", ratings="ratings"
        )
        icc_pingouin = icc_table.loc[icc_table["Type"] == "ICC2", "ICC"].values[0]

        diff = abs(icc_manual - icc_pingouin)
        status = "PASS" if diff < atol else "FAIL"
        print(f"  Test {i+1}: manual={icc_manual:.8f}, pingouin={icc_pingouin:.8f}, "
              f"diff={diff:.2e} [{status}]")

        if diff >= atol:
            raise AssertionError(
                f"ICC mismatch in test {i+1}: manual={icc_manual}, "
                f"pingouin={icc_pingouin}, diff={diff}"
            )

    print("  All tests passed!\n")


# ---------------------------------------------------------------------------
# Data loading — memory efficient
# ---------------------------------------------------------------------------

def load_class_data(h5_path, func_id):
    """
    Load all features for one function class.

    Returns
    -------
    np.ndarray, shape (N_INSTANCES, N_RUNS, TOTAL_FEATURES) — ~0.96 GB
    """
    data = np.empty((N_INSTANCES, N_RUNS, TOTAL_FEATURES))

    with h5py.File(h5_path, "r") as f:
        for inst_idx in range(N_INSTANCES):
            inst_id = inst_idx + 1
            key = f"{func_id}_{inst_id}_{DIMENSION}"
            group = f[key]

            offset = 0
            for perspective in PERSPECTIVES_LIST:
                for homology in HOMOLOGIES:
                    feat_len = FEATURE_LENGTHS[homology]
                    arr = group[perspective][homology][:]
                    # Reshape to (N_RUNS, feat_len) regardless of stored shape
                    # e.g. (30, 1, 100) -> (30, 100) or (30, 100, 100) -> (30, 10000)
                    arr = arr.reshape(arr.shape[0], -1)

                    if arr.shape == (N_RUNS, feat_len):
                        data[inst_idx, :, offset:offset + feat_len] = arr
                    elif arr.shape == (feat_len, N_RUNS):
                        data[inst_idx, :, offset:offset + feat_len] = arr.T
                    else:
                        raise ValueError(
                            f"Unexpected shape {arr.shape} (after reshape) for "
                            f"{key}/{perspective}/{homology}. "
                            f"Expected ({N_RUNS}, {feat_len}) or ({feat_len}, {N_RUNS})."
                        )

                    offset += feat_len

    return data


def load_segment_all_classes(h5_path, perspective, homology):
    """
    Load one feature segment across all 24 classes.

    Returns
    -------
    np.ndarray, shape (2400, N_RUNS, feat_len)
        For h1/h2 this is ~5.7 GB. For h0 this is ~57 MB.
    """
    feat_len = FEATURE_LENGTHS[homology]
    data = np.empty((N_FUNCTIONS * N_INSTANCES, N_RUNS, feat_len))

    with h5py.File(h5_path, "r") as f:
        for func_idx in range(N_FUNCTIONS):
            func_id = func_idx + 1
            start_row = func_idx * N_INSTANCES

            for inst_idx in range(N_INSTANCES):
                inst_id = inst_idx + 1
                key = f"{func_id}_{inst_id}_{DIMENSION}"
                arr = f[key][perspective][homology][:]
                arr = arr.reshape(arr.shape[0], -1)  # flatten feature dims

                if arr.shape == (N_RUNS, feat_len):
                    data[start_row + inst_idx] = arr
                elif arr.shape == (feat_len, N_RUNS):
                    data[start_row + inst_idx] = arr.T
                else:
                    raise ValueError(
                        f"Unexpected shape {arr.shape} (after reshape) for "
                        f"{key}/{perspective}/{homology}."
                    )

    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_filename(filename):
    """Extract sampling strategy and sample size from filename."""
    parts = filename.replace("_tla.h5", "").rsplit("_", 1)
    return parts[0], parts[1]


def compute_summaries(icc_values):
    """Compute summary statistics from an array of ICC values."""
    valid = icc_values[~np.isnan(icc_values)]
    if len(valid) == 0:
        return {
            "median": np.nan, "mean": np.nan,
            "p10": np.nan, "p90": np.nan,
            "prop_above_0.5": np.nan, "prop_above_0.75": np.nan,
            "prop_above_0.9": np.nan,
            "n_valid": 0, "n_nan": int(np.sum(np.isnan(icc_values))),
        }
    return {
        "median": float(np.median(valid)),
        "mean": float(np.mean(valid)),
        "p10": float(np.percentile(valid, 10)),
        "p90": float(np.percentile(valid, 90)),
        "prop_above_0.5": float(np.mean(valid > 0.5)),
        "prop_above_0.75": float(np.mean(valid > 0.75)),
        "prop_above_0.9": float(np.mean(valid > 0.9)),
        "n_valid": int(len(valid)),
        "n_nan": int(np.sum(np.isnan(icc_values))),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir
    output_file = os.path.join(output_dir, "tla_icc_results.h5")

    os.makedirs(output_dir, exist_ok=True)

    validate_icc_against_pingouin()

    # Print memory estimates
    max_group_size = max(len(v) for v in FUNCTION_GROUPS.values())
    group_gb = max_group_size * N_INSTANCES * N_RUNS * TOTAL_FEATURES * 8 / (1024**3)
    max_seg_len = max(FEATURE_LENGTHS.values())
    p1_seg_gb = N_FUNCTIONS * N_INSTANCES * N_RUNS * max_seg_len * 8 / (1024**3)
    print(f"Memory estimates:")
    print(f"  Perspective 3 (per class):  {N_INSTANCES * N_RUNS * TOTAL_FEATURES * 8 / (1024**3):.2f} GB")
    print(f"  Perspective 2 (max group):  {group_gb:.2f} GB")
    print(f"  Perspective 1 (max segment): {p1_seg_gb:.2f} GB")
    print(f"  Feature vector length: {TOTAL_FEATURES}")
    print()

    with h5py.File(output_file, "w") as out:
        for filename in FILES:
            filepath = os.path.join(input_dir, filename)
            if not os.path.exists(filepath):
                print(f"WARNING: {filepath} not found, skipping.")
                continue

            sampling_strategy, sample_size = parse_filename(filename)
            config_key = f"{sampling_strategy}_{sample_size}"

            print(f"\n{'='*60}")
            print(f"Processing: {filename} ({config_key})")
            print(f"{'='*60}")

            config_grp = out.create_group(config_key)
            config_grp.attrs["sampling_strategy"] = sampling_strategy
            config_grp.attrs["sample_size"] = int(sample_size)
            config_grp.attrs["filename"] = filename

            # ---------------------------------------------------------------
            # Perspective 3: Per function class
            # One class at a time (~0.96 GB)
            # ---------------------------------------------------------------
            print("  Perspective 3 (per function class)...")
            p3_grp = config_grp.create_group("perspective_3_classes")

            for func_idx in range(N_FUNCTIONS):
                func_id = func_idx + 1
                data_class = load_class_data(filepath, func_id)
                icc_class = compute_icc_batch(data_class)
                del data_class
                gc.collect()

                cls_grp = p3_grp.create_group(f"f{func_id}")
                cls_grp.create_dataset("icc_per_element", data=icc_class)

                summaries = compute_summaries(icc_class)
                for k, v in summaries.items():
                    cls_grp.attrs[k] = v

            print("    Median ICCs: ", end="")
            for func_idx in range(N_FUNCTIONS):
                func_id = func_idx + 1
                m = p3_grp[f"f{func_id}"].attrs["median"]
                print(f"f{func_id}:{m:.3f}", end=" ")
            print()

            # ---------------------------------------------------------------
            # Perspective 2: Per function group
            # Stack classes in group (~max 4.8 GB)
            # ---------------------------------------------------------------
            print("  Perspective 2 (per function group)...")
            p2_grp = config_grp.create_group("perspective_2_groups")

            for group_name, func_ids in FUNCTION_GROUPS.items():
                group_data_list = []
                for func_id in func_ids:
                    group_data_list.append(load_class_data(filepath, func_id))

                group_data = np.concatenate(group_data_list, axis=0)
                del group_data_list
                gc.collect()

                icc_group = compute_icc_batch(group_data)
                del group_data
                gc.collect()

                grp = p2_grp.create_group(group_name)
                grp.create_dataset("icc_per_element", data=icc_group)

                summaries = compute_summaries(icc_group)
                for k, v in summaries.items():
                    grp.attrs[k] = v

                print(f"    {group_name}: Median={summaries['median']:.4f}, "
                      f"P90={summaries['p90']:.4f}")

            # ---------------------------------------------------------------
            # Perspective 1: Overall — one feature segment at a time
            # Largest segment (h1/h2): ~5.7 GB
            # ---------------------------------------------------------------
            print("  Perspective 1 (overall) — segment by segment...")
            p1_grp = config_grp.create_group("perspective_1_overall")
            icc_overall = np.empty(TOTAL_FEATURES)

            for persp, hom, offset, flen in FEATURE_SEGMENTS:
                seg_gb = N_FUNCTIONS * N_INSTANCES * N_RUNS * flen * 8 / (1024**3)
                print(f"    {persp}/{hom} (len={flen}, ~{seg_gb:.2f} GB)...")

                seg_data = load_segment_all_classes(filepath, persp, hom)
                icc_seg = compute_icc_batch(seg_data)
                icc_overall[offset:offset + flen] = icc_seg
                del seg_data, icc_seg
                gc.collect()

            p1_grp.create_dataset("icc_per_element", data=icc_overall)
            summaries = compute_summaries(icc_overall)
            for k, v in summaries.items():
                p1_grp.attrs[k] = v

            print(f"    Overall: Median={summaries['median']:.4f}, "
                  f"P90={summaries['p90']:.4f}, "
                  f"Prop>0.75={summaries['prop_above_0.75']:.2%}")

            del icc_overall
            gc.collect()

    print(f"\n\nResults saved to: {output_file}")
    print(f"\nOutput h5 structure:")
    print(f"  {{config_key}}/")
    print(f"    perspective_1_overall/")
    print(f"      icc_per_element          ({TOTAL_FEATURES},)")
    print(f"      attrs: median, mean, p10, p90, prop_above_0.5/0.75/0.9")
    print(f"    perspective_2_groups/")
    print(f"      {{group_name}}/")
    print(f"        icc_per_element        ({TOTAL_FEATURES},)")
    print(f"        attrs: same as above")
    print(f"    perspective_3_classes/")
    print(f"      f{{1-24}}/")
    print(f"        icc_per_element        ({TOTAL_FEATURES},)")
    print(f"        attrs: same as above")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute ICC(2,1) for TLA features across 3 perspectives."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing the TLA h5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output h5 file. Defaults to input_dir.",
    )
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir)