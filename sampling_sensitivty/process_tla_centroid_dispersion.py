import argparse
from pathlib import Path
import numpy as np
import h5py
from tqdm.auto import tqdm
import gc
from scipy.spatial.distance import cosine, euclidean


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


def process_single_problem_instance(input_grp, key_str, output_h5, setting_name):
    """
    Process a single problem instance and write results to HDF5.

    Args:
        input_grp: HDF5 group for this problem instance
        key_str: String key (e.g., "1_1_10")
        output_h5: Open HDF5 file handle for output
        setting_name: Name of the setting (e.g., "cma_10")
    """
    # Parse the key string back to tuple
    parts = key_str.split('_')
    function = int(parts[0])
    instance = int(parts[1])
    dimension = int(parts[2])

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
        return False

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

    # Create output group for this setting if it doesn't exist
    if setting_name not in output_h5:
        setting_grp = output_h5.create_group(setting_name)
    else:
        setting_grp = output_h5[setting_name]

    # Create group for this problem instance
    instance_grp = setting_grp.create_group(key_str)

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
                # Create group for this feature group
                group_grp = instance_grp.create_group(group_name)

                # Save all statistics
                group_grp.create_dataset('centroid_mean', data=stats['centroid_mean'])
                group_grp.create_dataset('centroid_median', data=stats['centroid_median'])
                group_grp.create_dataset('cosine_dist_to_mean', data=stats['cosine_dist_to_mean'])
                group_grp.create_dataset('cosine_dist_to_median', data=stats['cosine_dist_to_median'])
                group_grp.create_dataset('euclidean_dist_to_mean', data=stats['euclidean_dist_to_mean'])
                group_grp.create_dataset('euclidean_dist_to_median', data=stats['euclidean_dist_to_median'])
                group_grp.attrs['n_runs'] = len(run_vectors)
                group_grp.attrs['feature_dim'] = run_vectors[0].shape[0]

                # Clean up
                del stats

        # Clean up run vectors
        del run_vectors

    # Clean up features_dict
    del features_dict
    gc.collect()

    return True


def compute_centroids_and_distances(h5_path, output_h5_file, setting_name):
    """
    Process a single HDF5 file and write results incrementally.

    Args:
        h5_path: Path to input HDF5 file
        output_h5_file: Open output HDF5 file handle
        setting_name: Name of the setting (e.g., "cma_10")
    """
    print(f"Processing: {h5_path.name}")

    processed_count = 0

    with h5py.File(h5_path, 'r') as input_f:
        # Get all keys
        keys = list(input_f.keys())

        # Process each problem instance one at a time
        for key_str in tqdm(keys, desc=f"  {setting_name}", leave=False):
            grp = input_f[key_str]

            success = process_single_problem_instance(
                grp, key_str, output_h5_file, setting_name
            )

            if success:
                processed_count += 1

            # Force garbage collection after each instance
            gc.collect()

    print(f"  Completed: {setting_name} ({processed_count} problem instances)")
    return processed_count


def main(data_dir):
    """
    Main function to process TinyTLA features and compute centroids/distances.

    Args:
        data_dir: Path to directory containing HDF5 files
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    files = [
        "cma_10_tla.h5",
        "cma_25_tla.h5",
        "cma_50_tla.h5",
        "ilhs_10_tla.h5",
        "ilhs_25_tla.h5",
        "ilhs_50_tla.h5",
        "lhs_10_tla.h5",
        "lhs_25_tla.h5",
        "lhs_50_tla.h5",
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
    ]

    # Create output directory
    output_dir = data_path / "centroids"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "tla_centroids_distances.h5"

    failed_files = []
    total_processed = 0

    # Open output HDF5 file once for all writes
    with h5py.File(output_file, 'w') as output_h5:
        for filename in tqdm(files, desc="Overall Progress"):
            h5_path = data_path / filename

            if not h5_path.exists():
                print(f"Warning: {filename} not found, skipping...")
                failed_files.append((filename, "File not found"))
                continue

            # Extract key name (e.g., "cma_10" from "cma_10_tla.h5")
            key_name = filename.replace('_tla.h5', '')

            try:
                # Process file and write results incrementally
                count = compute_centroids_and_distances(h5_path, output_h5, key_name)
                total_processed += count

                # Flush to disk after each file
                output_h5.flush()

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
        description="Compute centroids and distances for TinyTLA features"
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to directory containing HDF5 files"
    )

    args = parser.parse_args()
    main(args.data_dir)

'''
Key changes for minimal memory usage:

1. **Incremental writing**: Opens the output HDF5 file once and writes after each problem instance is processed
2. **Process one instance at a time**: `process_single_problem_instance()` handles one problem at a time
3. **Immediate cleanup**: Deletes variables and calls `gc.collect()` after each instance
4. **No in-memory accumulation**: Results are written directly to disk instead of accumulating in memory
5. **Flush to disk**: Calls `output_h5.flush()` after each file to ensure data is written

The output HDF5 structure will be:
```
tla_centroids_distances.h5
├── cma_10/
│   ├── 1_1_10/
│   │   ├── all/
│   │   │   ├── centroid_mean
│   │   │   ├── centroid_median
│   │   │   ├── cosine_dist_to_mean
│   │   │   ├── cosine_dist_to_median
│   │   │   ├── euclidean_dist_to_mean
│   │   │   └── euclidean_dist_to_median
│   │   ├── volume_all/
│   │   ├── volume_h0/
│   │   └── ...
│   └── ...
└── ...
'''
