import pickle
import h5py
import numpy as np
from pathlib import Path
import argparse
import sys


def convert_pickle_to_h5(pickle_file, h5_file, overwrite=False):
    """
    Convert pickle file with nested dict structure to HDF5 format.

    Args:
        pickle_file (str): Path to input pickle file
        h5_file (str): Path to output HDF5 file
        overwrite (bool): If True, overwrite existing h5 file. If False, append to existing file.
    """
    # Load pickle file
    print(f"Loading pickle file: {pickle_file}")
    with open(pickle_file, 'rb') as f:
        features = pickle.load(f)

    print(f"Loaded {len(features)} entries")

    # Check if h5 file exists
    mode = 'w' if overwrite or not Path(h5_file).exists() else 'a'

    # Write to HDF5
    print(f"Writing to HDF5 file: {h5_file} (mode: {mode})")
    with h5py.File(h5_file, mode) as f:
        for (function, instance, dimension), feature_dict in features.items():
            group_name = f"{function}_{instance}_{dimension}"

            # Skip if group already exists (in append mode)
            if group_name in f:
                print(f"  Skipping {group_name} (already exists)")
                continue

            print(f"  Creating group: {group_name}")
            grp = f.create_group(group_name)

            # Iterate over first level keys (e.g., 'volume', 'axis')
            for k1, val1 in feature_dict.items():
                grp2 = grp.create_group(k1)

                # Iterate over second level keys (e.g., 'h0', 'h1', 'h2')
                for k2, val2 in val1.items():
                    # Convert to numpy array if needed
                    if not isinstance(val2, np.ndarray):
                        val2 = np.array(val2)

                    grp2.create_dataset(k2, data=val2)

    print(f"Conversion complete! Saved to {h5_file}")


def batch_convert_directory(directory, pattern="*tla.pkl", overwrite=False, recursive=False):
    """
    Convert all pickle files in a directory to HDF5 format.

    Args:
        directory (str or Path): Directory containing pickle files
        pattern (str): Glob pattern to match pickle files (default: "*tla.pkl")
        overwrite (bool): If True, overwrite existing h5 files
        recursive (bool): If True, search subdirectories recursively

    Returns:
        tuple: (num_converted, num_failed, num_skipped)
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    # Find all pickle files
    if recursive:
        pickle_files = list(directory.rglob(pattern))
    else:
        pickle_files = list(directory.glob(pattern))

    if not pickle_files:
        print(f"No pickle files found in {directory} matching pattern '{pattern}'")
        return 0, 0, 0

    print(f"Found {len(pickle_files)} pickle file(s) in {directory}")
    print(f"Overwrite mode: {overwrite}")
    print("-" * 80)

    num_converted = 0
    num_failed = 0
    num_skipped = 0

    for i, pickle_file in enumerate(pickle_files, 1):
        print(f"\n[{i}/{len(pickle_files)}] Processing: {pickle_file.name}")

        # Generate h5 filename (same directory, replace extension)
        h5_file = pickle_file.with_suffix('.h5')

        # Skip if h5 file exists and overwrite is False
        if h5_file.exists() and not overwrite:
            print(f"  Skipping (h5 file already exists): {h5_file.name}")
            num_skipped += 1
            continue

        try:
            convert_pickle_to_h5(str(pickle_file), str(h5_file), overwrite=overwrite)
            num_converted += 1
        except Exception as e:
            print(f"  ERROR: Failed to convert {pickle_file.name}")
            print(f"  Error message: {str(e)}")
            num_failed += 1

    print("\n" + "=" * 80)
    print(f"Batch conversion complete!")
    print(f"  Successfully converted: {num_converted}")
    print(f"  Skipped (already exists): {num_skipped}")
    print(f"  Failed: {num_failed}")
    print("=" * 80)

    return num_converted, num_failed, num_skipped


def verify_h5_structure(h5_file, num_samples=3):
    """
    Verify the structure of the created HDF5 file.

    Args:
        h5_file (str): Path to HDF5 file to verify
        num_samples (int): Number of sample groups to display
    """
    print(f"\nVerifying HDF5 file: {h5_file}")
    with h5py.File(h5_file, 'r') as f:
        print(f"Total number of groups: {len(f.keys())}")

        # Sample a few groups
        for i, group_name in enumerate(list(f.keys())[:num_samples]):
            print(f"\n  Group {i + 1}: {group_name}")
            grp = f[group_name]

            for k1 in grp.keys():
                print(f"    └── {k1}/")
                grp2 = grp[k1]

                for k2 in grp2.keys():
                    dataset = grp2[k2]
                    print(f"        └── {k2}: shape={dataset.shape}, dtype={dataset.dtype}")


def load_specific_features(h5_file, function, instance, dimension):
    """
    Load features for a specific (function, instance, dimension) tuple.

    Args:
        h5_file (str): Path to HDF5 file
        function (int): Function ID
        instance (int): Instance ID
        dimension (int): Dimension

    Returns:
        dict: Nested dictionary with the feature structure
    """
    group_name = f"{function}_{instance}_{dimension}"

    with h5py.File(h5_file, 'r') as f:
        if group_name not in f:
            raise KeyError(f"Group {group_name} not found in HDF5 file")

        grp = f[group_name]
        features = {}

        for k1 in grp.keys():
            features[k1] = {}
            grp2 = grp[k1]

            for k2 in grp2.keys():
                features[k1][k2] = grp2[k2][:]  # Load dataset into memory

    return features


def main():
    """Command-line interface for batch conversion."""
    parser = argparse.ArgumentParser(
        description='Convert pickle files to HDF5 format in batch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all tla.pkl files in a directory
  python convert_pickle_to_h5.py /path/to/data

  # Convert with overwrite
  python convert_pickle_to_h5.py /path/to/data --overwrite

  # Convert recursively in subdirectories
  python convert_pickle_to_h5.py /path/to/data --recursive

  # Custom pattern
  python convert_pickle_to_h5.py /path/to/data --pattern "features_*.pkl"

  # Verify a specific h5 file
  python convert_pickle_to_h5.py --verify /path/to/file.h5
        """
    )

    parser.add_argument(
        'directory',
        type=str,
        help='Directory containing pickle files to convert'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default='*tla.pkl',
        help='Glob pattern to match pickle files (default: *tla.pkl)'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing h5 files'
    )

    parser.add_argument(
        '--recursive',
        '-r',
        action='store_true',
        help='Search subdirectories recursively'
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the structure of converted files after conversion'
    )

    args = parser.parse_args()

    try:
        # Perform batch conversion
        num_converted, num_failed, num_skipped = batch_convert_directory(
            directory=args.directory,
            pattern=args.pattern,
            overwrite=args.overwrite,
            recursive=args.recursive
        )

        # Verify converted files if requested
        if args.verify and num_converted > 0:
            print("\n" + "=" * 80)
            print("Verifying converted files...")
            print("=" * 80)

            directory = Path(args.directory)
            if args.recursive:
                h5_files = list(directory.rglob('*.h5'))
            else:
                h5_files = list(directory.glob('*.h5'))

            for h5_file in h5_files[:3]:  # Verify first 3 files
                try:
                    verify_h5_structure(str(h5_file), num_samples=2)
                except Exception as e:
                    print(f"Error verifying {h5_file}: {e}")

        # Exit with appropriate code
        if num_failed > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()