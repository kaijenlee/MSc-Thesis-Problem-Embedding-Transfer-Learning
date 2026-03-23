import argparse
from pathlib import Path
import numpy as np
import h5py
import pickle
from tqdm.auto import tqdm
import gc


def compute_cv(features_array):
    """
    Compute coefficient of variation for features across runs.

    Args:
        features_array: numpy array of shape (n_runs, ...)

    Returns:
        CV array with same shape as single run
    """
    mean = np.mean(features_array, axis=0)
    std = np.std(features_array, axis=0)

    # Avoid division by zero
    cv = np.where(np.abs(mean) > 1e-10, std / np.abs(mean), np.nan)

    return cv


def process_h5_file(h5_path):
    """
    Process a single HDF5 file and compute CV for all features.
    """
    print(f"Processing: {h5_path.name}")

    cv_dict = {}

    with h5py.File(h5_path, 'r') as f:
        # Iterate over all groups (function_instance_dimension keys)
        for key_str in f.keys():
            # Parse the key string back to tuple
            parts = key_str.split('_')
            # Assuming format: function_instance_dimension
            # You may need to adjust this parsing based on your actual key format
            function = int(parts[0])
            instance = int(parts[1])
            dimension = int(parts[2])
            key = (function, instance, dimension)

            # Initialize nested structure
            cv_dict[key] = {
                'volume': {'h0': None, 'h1': None, 'h2': None},
                'axis': {'h0': None, 'h1': None, 'h2': None}
            }

            grp = f[key_str]

            # Process each transformation type (volume, axis)
            for transform in ['volume', 'axis']:
                if transform not in grp:
                    continue

                transform_grp = grp[transform]

                # Process each homology dimension (h0, h1, h2)
                for homology in ['h0', 'h1', 'h2']:
                    if homology not in transform_grp:
                        continue

                    # Load the dataset
                    features_array = transform_grp[homology][:]

                    if features_array.size == 0:
                        continue

                    # Compute CV
                    cv = compute_cv(features_array)

                    cv_dict[key][transform][homology] = cv

                    # Clean up
                    del features_array

    # Force garbage collection
    gc.collect()

    return cv_dict


def main(data_dir):
    """
    Main function to process TinyTLA features.

    Args:
        data_dir: Path to directory containing HDF5 files
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    files = [
        "ilhs_10_tla.h5",
        "ilhs_25_tla.h5",
        "ilhs_50_tla.h5",
        "ilhs_75_tla.h5",
        "ilhs_100_tla.h5",
        "lhs_10_tla.h5",
        "lhs_25_tla.h5",
        "lhs_50_tla.h5",
        "lhs_75_tla.h5",
        "lhs_100_tla.h5",
        "sobol_10_tla.h5",
        "sobol_25_tla.h5",
        "sobol_50_tla.h5",
        "sobol_75_tla.h5",
        "sobol_100_tla.h5",
        "uniform_10_tla.h5",
        "uniform_25_tla.h5",
        "uniform_50_tla.h5",
        "uniform_75_tla.h5",
        "uniform_100_tla.h5",
        "cma_random_10_tla.h5",
        "cma_random_25_tla.h5",
        "cma_random_50_tla.h5",
        "cma_random_75_tla.h5",
        "cma_random_100_tla.h5",
    ]

    output = {}
    failed_files = []

    for filename in tqdm(files):
        h5_path = data_path / filename

        if not h5_path.exists():
            print(f"Warning: {filename} not found, skipping...")
            failed_files.append((filename, "File not found"))
            continue

        # Extract key name (e.g., "cma_10" from "cma_10_tla.h5")
        key_name = filename.replace('_tla.h5', '').replace('_ela.h5', '')

        try:
            # Process file and compute CV
            cv_dict = process_h5_file(h5_path)

            # Store in output
            output[key_name] = cv_dict

            print(f"Completed: {key_name}\n")

            # Clean up after each file
            del cv_dict
            gc.collect()

        except Exception as e:
            print(f"ERROR: Failed to process {filename}: {e}")
            print(f"Skipping...\n")
            failed_files.append((filename, f"Error: {str(e)}"))
            continue

    output_file = data_path / "cv" / "tla_cv.pkl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(output, f)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_file}")
    print(f"Successfully processed: {len(output)}/{len(files)} files")

    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for fname, reason in failed_files:
            print(f"  - {fname}: {reason}")
    else:
        print("\nAll files processed successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze TinyTLA feature variance across runs"
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to directory containing HDF5 files"
    )

    args = parser.parse_args()
    main(args.data_dir)