import argparse
from pathlib import Path
import numpy as np
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


def process_pickle_file(pkl_path):
    """
    Process a single pickle file and compute CV for all features.
    """
    print(f"Processing: {pkl_path.name}")

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"  ERROR: Failed to load pickle file: {e}")
        raise  # Re-raise so main() can catch it

    cv_dict = {}

    # Get all keys upfront
    keys = list(data.keys())

    # Iterate over all (function, instance, dimension) keys
    for key in keys:
        function, instance, dim = key

        # Initialize nested structure
        cv_dict[key] = {
            'volume': {'h0': None, 'h1': None, 'h2': None},
            'axis': {'h0': None, 'h1': None, 'h2': None}
        }

        # Process each transformation type (volume, axis)
        for transform in ['volume', 'axis']:
            if transform not in data[key]:
                continue

            # Process each homology dimension (h0, h1, h2)
            for homology in ['h0', 'h1', 'h2']:
                if homology not in data[key][transform]:
                    continue

                features_list = data[key][transform][homology]

                if len(features_list) == 0:
                    continue

                # Convert list of arrays to single array (n_runs, ...)
                features_array = np.array(features_list)

                # Compute CV
                cv = compute_cv(features_array)

                cv_dict[key][transform][homology] = cv

                # print(f"  {key} - {transform}/{homology}: shape {cv.shape}, "
                #       f"mean CV = {np.nanmean(cv):.4f}")

                # Clean up large intermediate arrays
                del features_array

        # Delete processed key's data immediately
        del data[key]

    # Explicitly delete the large data dictionary
    del data

    # Force garbage collection to free memory
    gc.collect()

    return cv_dict


def main(data_dir):
    """
    Main function to process TinyTLA features.

    Args:
        data_dir: Path to directory containing pickle files
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    files = [
        "cma_10_tla.pkl",
        "cma_25_tla.pkl",
        "cma_50_tla.pkl",
        "ilhs_25_tla.pkl",
        "ilhs_50_tla.pkl",
        "lhs_10_tla.pkl",
        "lhs_25_tla.pkl",
        "lhs_50_tla.pkl",
        "sobol_10_tla.pkl",
        "sobol_25_tla.pkl",
        "sobol_50_tla.pkl",
        "uniform_10_tla.pkl",
        "uniform_25_tla.pkl",
        "uniform_50_tla.pkl",
    ]

    output = {}
    failed_files = []

    for filename in tqdm(files):
        pkl_path = data_path / filename

        if not pkl_path.exists():
            print(f"Warning: {filename} not found, skipping...")
            failed_files.append((filename, "File not found"))
            continue

        # Extract key name (e.g., "cma_10" from "cma_10_tla.pkl")
        key_name = filename.replace('_tla.pkl', '').replace('_ela.pkl', '')

        try:
            # Process file and compute CV
            cv_dict = process_pickle_file(pkl_path)

            # Store in output
            output[key_name] = cv_dict

            print(f"Completed: {key_name}\n")

            # Clean up after each file
            del cv_dict
            gc.collect()

        except (pickle.UnpicklingError, EOFError) as e:
            print(f"ERROR: Failed to process {filename}: {e}")
            print(f"This file appears to be corrupted. Skipping...\n")
            failed_files.append((filename, f"Corrupted: {str(e)}"))
            continue
        except Exception as e:
            print(f"ERROR: Unexpected error processing {filename}: {e}")
            print(f"Skipping...\n")
            failed_files.append((filename, f"Unexpected error: {str(e)}"))
            continue

    # Save output
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
        help="Path to directory containing pickle files"
    )

    args = parser.parse_args()
    main(args.data_dir)