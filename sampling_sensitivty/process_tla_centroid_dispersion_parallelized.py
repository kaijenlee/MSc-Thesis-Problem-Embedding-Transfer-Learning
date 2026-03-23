import argparse
from pathlib import Path
import numpy as np
import h5py
from tqdm.auto import tqdm
import gc
from scipy.spatial.distance import cosine, euclidean
from multiprocessing import Pool, cpu_count
import tempfile
import shutil


def flatten_features(features_dict, transform_filter=None, homology_filter=None):
    """
    Flatten the nested feature dictionary into a single vector.

    Args:
        features_dict: Dictionary with structure {transform: {homology: array}}
        transform_filter: List of transforms to include (e.g., ['volume'] or ['axis'])
        homology_filter: List of homologies to include (e.g., ['h0'])

    Returns:
        Flattened 1D numpy array
    """
    vectors = []

    transforms = transform_filter if transform_filter else ['volume', 'axis']
    homologies = homology_filter if homology_filter else ['h0', 'h1', 'h2']

    for transform in transforms:
        if transform not in features_dict:
            continue
        for homology in homologies:
            if homology not in features_dict[transform]:
                continue
            arr = features_dict[transform][homology]
            if arr is not None and arr.size > 0:
                vectors.append(arr.flatten())

    if len(vectors) == 0:
        return None

    return np.concatenate(vectors)


def compute_stats_for_group(run_vectors):
    """
    Compute centroids and distances for a group of run vectors.

    Args:
        run_vectors: Array of shape (n_runs, feature_dim)

    Returns:
        Dictionary with centroids and distances
    """
    if len(run_vectors) == 0:
        return None

    run_vectors = np.array(run_vectors)

    # Compute centroids
    centroid_mean = np.mean(run_vectors, axis=0)
    centroid_median = np.median(run_vectors, axis=0)

    # Compute distances
    cosine_dist_to_mean = []
    cosine_dist_to_median = []
    euclidean_dist_to_mean = []
    euclidean_dist_to_median = []

    for vec in run_vectors:
        # Cosine distances
        cosine_dist_to_mean.append(cosine(vec, centroid_mean))
        cosine_dist_to_median.append(cosine(vec, centroid_median))

        # Euclidean distances
        euclidean_dist_to_mean.append(euclidean(vec, centroid_mean))
        euclidean_dist_to_median.append(euclidean(vec, centroid_median))

    return {
        'centroid_mean': centroid_mean,
        'centroid_median': centroid_median,
        'cosine_dist_to_mean': np.array(cosine_dist_to_mean),
        'cosine_dist_to_median': np.array(cosine_dist_to_median),
        'euclidean_dist_to_mean': np.array(euclidean_dist_to_mean),
        'euclidean_dist_to_median': np.array(euclidean_dist_to_median),
    }


def process_single_problem_instance_worker(args):
    """
    Worker function to process a single problem instance.
    Returns results as a dictionary instead of writing to HDF5.

    Args:
        args: Tuple of (input_h5_path, key_str, setting_name)

    Returns:
        Tuple of (key_str, results_dict) or None if failed
    """
    input_h5_path, key_str, setting_name = args

    try:
        with h5py.File(input_h5_path, 'r') as input_f:
            if key_str not in input_f:
                return None

            input_grp = input_f[key_str]

            # Read data for volume and axis transforms
            features_dict = {
                'volume': {},
                'axis': {}
            }

            for transform in ['volume', 'axis']:
                if transform not in input_grp:
                    continue

                transform_grp = input_grp[transform]

                for homology in ['h0', 'h1', 'h2']:
                    if homology not in transform_grp:
                        continue

                    # Load the dataset (shape: n_runs x feature_dims...)
                    data = transform_grp[homology][:]
                    features_dict[transform][homology] = data

            # Get number of runs from any available dataset
            n_runs = None
            for transform in ['volume', 'axis']:
                if transform in features_dict:
                    for homology in ['h0', 'h1', 'h2']:
                        if homology in features_dict[transform]:
                            arr = features_dict[transform][homology]
                            if arr.size > 0:
                                n_runs = arr.shape[0]
                                break
                if n_runs is not None:
                    break

            if n_runs is None or n_runs == 0:
                return None

            # Define all groups to process
            groups = {
                'all': {'transform': None, 'homology': None},
                'volume_all': {'transform': ['volume'], 'homology': None},
                'volume_h0': {'transform': ['volume'], 'homology': ['h0']},
                'volume_h1': {'transform': ['volume'], 'homology': ['h1']},
                'volume_h2': {'transform': ['volume'], 'homology': ['h2']},
                'axis_all': {'transform': ['axis'], 'homology': None},
                'axis_h0': {'transform': ['axis'], 'homology': ['h0']},
                'axis_h1': {'transform': ['axis'], 'homology': ['h1']},
                'axis_h2': {'transform': ['axis'], 'homology': ['h2']},
            }

            results = {}

            # Process each group
            for group_name, filters in groups.items():
                run_vectors = []

                # Extract feature vector for each run
                for run_idx in range(n_runs):
                    run_features = {
                        'volume': {},
                        'axis': {}
                    }

                    for transform in ['volume', 'axis']:
                        if transform in features_dict:
                            for homology in ['h0', 'h1', 'h2']:
                                if homology in features_dict[transform]:
                                    arr = features_dict[transform][homology]
                                    if arr.size > 0:
                                        run_features[transform][homology] = arr[run_idx]

                    # Flatten this run's features for the current group
                    flat_vector = flatten_features(
                        run_features,
                        transform_filter=filters['transform'],
                        homology_filter=filters['homology']
                    )

                    if flat_vector is not None:
                        run_vectors.append(flat_vector)

                # Compute stats for this group
                if len(run_vectors) > 0:
                    stats = compute_stats_for_group(run_vectors)
                    if stats is not None:
                        stats['n_runs'] = len(run_vectors)
                        stats['feature_dim'] = run_vectors[0].shape[0]
                        results[group_name] = stats

                # Clean up run vectors
                del run_vectors

            # Clean up features_dict
            del features_dict
            gc.collect()

            return (key_str, results)

    except Exception as e:
        print(f"Error processing {key_str} in {setting_name}: {e}")
        return None


def write_results_to_h5(output_h5, setting_name, key_str, results):
    """
    Write results for a single problem instance to HDF5.
    """
    # Create output group for this setting if it doesn't exist
    if setting_name not in output_h5:
        setting_grp = output_h5.create_group(setting_name)
    else:
        setting_grp = output_h5[setting_name]

    # Create group for this problem instance
    instance_grp = setting_grp.create_group(key_str)

    # Write all group results
    for group_name, stats in results.items():
        group_grp = instance_grp.create_group(group_name)

        # Save all statistics
        group_grp.create_dataset('centroid_mean', data=stats['centroid_mean'])
        group_grp.create_dataset('centroid_median', data=stats['centroid_median'])
        group_grp.create_dataset('cosine_dist_to_mean', data=stats['cosine_dist_to_mean'])
        group_grp.create_dataset('cosine_dist_to_median', data=stats['cosine_dist_to_median'])
        group_grp.create_dataset('euclidean_dist_to_mean', data=stats['euclidean_dist_to_mean'])
        group_grp.create_dataset('euclidean_dist_to_median', data=stats['euclidean_dist_to_median'])
        group_grp.attrs['n_runs'] = stats['n_runs']
        group_grp.attrs['feature_dim'] = stats['feature_dim']


def process_file_parallel(h5_path, output_h5_file, setting_name, n_processes):
    """
    Process a single HDF5 file in parallel and write results.

    Args:
        h5_path: Path to input HDF5 file
        output_h5_file: Open output HDF5 file handle
        setting_name: Name of the setting (e.g., "cma_10")
        n_processes: Number of parallel processes
    """
    print(f"Processing: {h5_path.name} with {n_processes} processes")

    # Get all keys from the input file
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())

    # Prepare arguments for parallel processing
    args_list = [(str(h5_path), key_str, setting_name) for key_str in keys]

    # Process in parallel
    processed_count = 0
    with Pool(processes=n_processes) as pool:
        # Use imap_unordered for better progress tracking
        results_iter = pool.imap_unordered(
            process_single_problem_instance_worker,
            args_list,
            chunksize=1
        )

        # Process results as they come in and write to HDF5
        for result in tqdm(results_iter, total=len(args_list),
                           desc=f"  {setting_name}", leave=False):
            if result is not None:
                key_str, results_dict = result
                write_results_to_h5(output_h5_file, setting_name, key_str, results_dict)
                processed_count += 1

                # Flush periodically (every 10 instances)
                if processed_count % 10 == 0:
                    output_h5_file.flush()

    # Final flush
    output_h5_file.flush()

    print(f"  Completed: {setting_name} ({processed_count} problem instances)")
    return processed_count


def main(data_dir, n_processes=None):
    """
    Main function to process TinyTLA features and compute centroids/distances.

    Args:
        data_dir: Path to directory containing HDF5 files
        n_processes: Number of parallel processes (default: cpu_count())
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    if n_processes is None:
        n_processes = cpu_count()

    print(f"Using {n_processes} parallel processes")

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

    # Create output directory
    output_dir = data_path / "centroids"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "tla_centroids_distances.h5"

    failed_files = []
    total_processed = 0

    # Open output HDF5 file once for all writes
    with h5py.File(output_file, 'a') as output_h5:
        for filename in tqdm(files, desc="Overall Progress"):
            h5_path = data_path / filename

            if not h5_path.exists():
                print(f"Warning: {filename} not found, skipping...")
                failed_files.append((filename, "File not found"))
                continue

            # Extract key name (e.g., "cma_10" from "cma_10_tla.h5")
            key_name = filename.replace('_tla.h5', '')
            if key_name in output_h5:
                print(f"Skipping {filename} (already in output)")
                continue

            try:
                # Process file in parallel and write results
                count = process_file_parallel(h5_path, output_h5, key_name, n_processes)
                total_processed += count

                print()

            except Exception as e:
                print(f"ERROR: Failed to process {filename}: {e}")
                import traceback
                traceback.print_exc()
                print(f"Skipping...\n")
                failed_files.append((filename, f"Error: {str(e)}"))
                continue

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_file}")
    print(f"Total problem instances processed: {total_processed}")
    print(f"Successfully processed: {len(files) - len(failed_files)}/{len(files)} files")

    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for fname, reason in failed_files:
            print(f"  - {fname}: {reason}")
    else:
        print("\nAll files processed successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute centroids and distances for TinyTLA features (parallel)"
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to directory containing HDF5 files"
    )
    parser.add_argument(
        "-p", "--processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: all CPUs)"
    )

    args = parser.parse_args()
    main(args.data_dir, args.processes)