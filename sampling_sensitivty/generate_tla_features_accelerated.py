import argparse
import os
from pathlib import Path
from ripser import Rips
from persim import PersistenceImager
from pflacco.classical_ela_features import *
import pickle
from scipy.spatial.distance import cdist
import h5py
import gc
import warnings
import numpy as np
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# Optional ripser++ import — falls back gracefully if not installed
try:
    import ripserplusplus as rpp
    RIPSER_PLUS_PLUS_AVAILABLE = True
except ImportError:
    RIPSER_PLUS_PLUS_AVAILABLE = False


def axis_transform(X):
    pca = PCA(n_components=X.shape[1])
    centered_points = pca.fit_transform(X)
    scaled = centered_points / np.abs(centered_points).max(axis=0)
    return scaled


def volume_transform(X):
    _, n_dimensions = X.shape
    pca = PCA(n_components=n_dimensions)
    centered_points = pca.fit_transform(X)
    min_vals = centered_points.min(axis=0)
    max_vals = centered_points.max(axis=0)
    box_dimensions = max_vals - min_vals
    current_hypervolume = np.prod(box_dimensions)
    if current_hypervolume > 0:
        scale_factor = (1.0 / current_hypervolume) ** (1.0 / n_dimensions)
    else:
        scale_factor = 1.0
    return centered_points * scale_factor


def get_distance(X_D, Y_D, alpha=0.2):
    return alpha * X_D + (1 - alpha) * Y_D


def compute_persistence_diagrams(distance_matrix, maxdim, use_rpp):
    """
    Compute persistence diagrams from a distance matrix.

    If use_rpp is True and ripser++ is available, use the GPU implementation.
    Otherwise fall back to CPU ripser. Both paths return the same structure:
    a list [d0, d1, ..., d_maxdim] of (n_features, 2) arrays.
    """
    if use_rpp and RIPSER_PLUS_PLUS_AVAILABLE:
        # ripser++ takes the distance matrix directly and a format string
        result = rpp.run(
            f"--format distance --dim {maxdim}",
            distance_matrix.astype(np.float32),
        )
        # ripser++ returns a dict-like with one entry per dimension
        diagrams = []
        for d in range(maxdim + 1):
            dgm = np.asarray(result[d], dtype=np.float64)
            if dgm.ndim == 1 and dgm.size == 0:
                dgm = np.empty((0, 2), dtype=np.float64)
            diagrams.append(dgm)
        return diagrams
    else:
        rips = Rips(maxdim=maxdim, coeff=2, verbose=False)
        return rips.fit_transform(distance_matrix, distance_matrix=True)


def compute_one_run(samples, sample_size, dimension, alpha,
                    pimgr0, pimgr1, pimgr2, use_rpp):
    """
    Compute the volume- and axis-transformed persistence images for a single run.
    Pure function — safe to call from a joblib worker.
    """
    X = samples['X'][:sample_size * dimension]
    Y = samples['Y'][:sample_size * dimension]

    X_volume = volume_transform(X)
    X_axis = axis_transform(X)

    X_volume_D = cdist(X_volume, X_volume, "euclidean")
    X_volume_D_norm = X_volume_D / np.abs(X_volume_D).max(axis=0)
    X_axis_D = cdist(X_axis, X_axis, "euclidean")
    X_axis_D_norm = X_axis_D / np.abs(X_axis_D).max(axis=0)

    Y_D = cdist(np.asmatrix(Y).T, np.asmatrix(Y).T, "euclidean")
    Y_D_norm = Y_D / Y_D.max()

    distances_volume = get_distance(X_volume_D_norm, Y_D_norm, alpha=alpha)
    distances_axis = get_distance(X_axis_D_norm, Y_D_norm, alpha=alpha)

    diagrams_volume = compute_persistence_diagrams(np.asarray(distances_volume), maxdim=2, use_rpp=use_rpp)
    diagrams_axis = compute_persistence_diagrams(np.asarray(distances_axis), maxdim=2, use_rpp=use_rpp)

    def to_image(diagram, pimgr):
        if diagram.shape[0] == 0:
            return pimgr.transform(diagram)
        sel = np.isfinite(diagram.sum(axis=1))
        return pimgr.transform(diagram[sel, :])

    return {
        'volume': {
            'h0': to_image(diagrams_volume[0], pimgr0),
            'h1': to_image(diagrams_volume[1], pimgr1),
            'h2': to_image(diagrams_volume[2], pimgr2),
        },
        'axis': {
            'h0': to_image(diagrams_axis[0], pimgr0),
            'h1': to_image(diagrams_axis[1], pimgr1),
            'h2': to_image(diagrams_axis[2], pimgr2),
        },
    }


def extract_ela_features(samples_src, sampling_method, sample_size, data_dir, output_dir, dimension):
    features = {}
    for function in range(1, 25):
        for instance in range(1, 101):
            filename = output_dir / "temp" / f"ela_{sampling_method}_{sample_size}_{function}_{instance}_{dimension}.pkl"
            if filename.exists():
                try:
                    with open(filename, 'rb') as f:
                        file_done = pickle.load(f)
                        features[(function, instance, dimension)] = file_done
                        continue
                except EOFError:
                    print(f"{filename} is empty or corrupted")

            features[(function, instance, dimension)] = {
                "ela_dist": [], "meta": [], "disp": [],
                "ic": [], "nbc": [], "pca": [],
            }
            for runs in range(0, 30):
                samples = samples_src[(function, instance, dimension, runs)]
                X = samples['X'][:sample_size * dimension]
                Y = samples['Y'][:sample_size * dimension]
                features[(function, instance, dimension)]["ela_dist"].append(calculate_ela_distribution(X, Y))
                features[(function, instance, dimension)]["meta"].append(calculate_ela_meta(X, Y))
                features[(function, instance, dimension)]["disp"].append(calculate_dispersion(X, Y))
                features[(function, instance, dimension)]["ic"].append(calculate_information_content(X, Y, seed=100))
                features[(function, instance, dimension)]["nbc"].append(calculate_nbc(X, Y))
                features[(function, instance, dimension)]["pca"].append(calculate_pca(X, Y))

            with open(filename, 'wb') as f:
                pickle.dump(features[(function, instance, dimension)], f)

    with open(output_dir / f"{sampling_method}_{sample_size}_ela.pkl", 'wb') as f:
        pickle.dump(features, f)


def extract_tla_features(samples_src, sampling_method, sample_size, data_dir,
                         output_dir, dimension, n_jobs, use_rpp):
    """
    Extract TLA features with run-level joblib parallelism.

    The 30 runs for each (function, instance, dimension) are computed in parallel
    by worker processes. The main process aggregates the results and writes them
    to the h5 file — workers never touch h5, so there's no concurrency issue.
    """
    h5_data_file = output_dir / f"{sampling_method}_{sample_size}_tla.h5"
    if not h5_data_file.exists():
        with h5py.File(h5_data_file, 'w') as f:
            pass

    kernel_size = 0.0002
    max_range = 1.0
    alpha = 0.2
    pimgr0 = PersistenceImager(
        pixel_size=0.01,
        birth_range=(0.0, 0.01),
        pers_range=(0.0, max_range),
        kernel_params={"sigma": [[kernel_size, 0.0], [0.0, kernel_size]]},
    )
    pimgr1 = PersistenceImager(
        pixel_size=0.01,
        birth_range=(0.0, max_range),
        pers_range=(0.0, max_range),
        kernel_params={"sigma": [[kernel_size, 0.0], [0.0, kernel_size]]},
    )
    pimgr2 = PersistenceImager(
        pixel_size=0.01,
        birth_range=(0.0, max_range),
        pers_range=(0.0, max_range),
        kernel_params={"sigma": [[kernel_size, 0.0], [0.0, kernel_size]]},
    )

    backend_msg = "ripser++ (GPU)" if (use_rpp and RIPSER_PLUS_PLUS_AVAILABLE) else "ripser (CPU)"
    if use_rpp and not RIPSER_PLUS_PLUS_AVAILABLE:
        print("WARNING: --use-ripser-plus-plus requested but ripserplusplus is not installed. Falling back to CPU ripser.")
    print(f"TLA backend: {backend_msg}, joblib n_jobs={n_jobs}")

    # Persistent worker pool — kept alive across all (function, instance) iterations
    # so workers aren't respawned 2400 times.
    with Parallel(n_jobs=n_jobs) as parallel:
        for function in range(1, 25):
            for instance in range(1, 101):
                # Skip if already computed
                with h5py.File(h5_data_file, 'r') as f:
                    if f"{function}_{instance}_{dimension}" in f:
                        continue

                # Parallel computation of all 30 runs
                run_results = parallel(
                    delayed(compute_one_run)(
                        samples_src[(function, instance, dimension, r)],
                        sample_size, dimension, alpha,
                        pimgr0, pimgr1, pimgr2, use_rpp,
                    )
                    for r in range(30)
                )

                # Aggregate the 30 results into the per-instance features dict
                features = {
                    'volume': {'h0': [], 'h1': [], 'h2': []},
                    'axis':   {'h0': [], 'h1': [], 'h2': []},
                }
                for r in run_results:
                    for transform in ('volume', 'axis'):
                        for h in ('h0', 'h1', 'h2'):
                            features[transform][h].append(r[transform][h])

                # Main-process-only h5 write
                with h5py.File(h5_data_file, 'a') as f:
                    grp = f.create_group(f"{function}_{instance}_{dimension}")
                    for k1, val1 in features.items():
                        grp2 = grp.create_group(k1)
                        for k2, val2 in val1.items():
                            grp2.create_dataset(k2, data=np.asarray(val2))

                del features, run_results
                gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Extract landscape features from optimization problems")
    parser.add_argument("--feature-type", type=str, required=True, choices=["ela", "tla"])
    parser.add_argument("--sampling-method", type=str, required=True,
                        choices=["uniform", "lhs", "ilhs", "sobol", "cma", "cma_random"])
    parser.add_argument("--sample-size", type=int, required=True, choices=[10, 25, 50, 75, 100])
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Number of parallel workers for TLA (default -1 = all cores). "
                             "Capped at 30 since there are 30 runs per instance.")
    parser.add_argument("--use-ripser-plus-plus", action="store_true",
                        help="Use ripser++ (GPU) instead of CPU ripser. Requires the "
                             "ripserplusplus package and a CUDA-capable GPU.")
    args = parser.parse_args()

    dimension = args.dimension
    data_dir = Path(args.data_dir).resolve()
    file_name = data_dir / "features" / "pickles" / f"{args.sampling_method}_{args.sample_size}_{args.feature_type}.pkl"
    if file_name.exists():
        print(f"{args.sampling_method}_{args.sample_size}_{args.feature_type}.pkl already exists. Skipping.")
        return

    output_dir = Path(args.output_dir).resolve()

    processed_files = Path(output_dir / 'processed_files.txt')
    if processed_files.exists():
        processed_files_entries = set(processed_files.read_text().splitlines())
        if f"{args.sampling_method}_{args.sample_size}_{args.feature_type}" in processed_files_entries:
            print(f"{args.sampling_method}_{args.sample_size}_{args.feature_type} already in processed_files.txt. Skipping.")
            return

    sampling_method = "cma_single" if args.sampling_method == "cma" else args.sampling_method
    sample_size = args.sample_size
    runs = 30

    os.makedirs(output_dir / "temp", exist_ok=True)

    pickle_file = (
        f"{sampling_method}_100_{runs}.pkl"
        if args.sampling_method in ("cma", "cma_random")
        else f"{sampling_method}_{sample_size}_{runs}.pkl"
    )
    with open(data_dir / pickle_file, "rb") as f:
        samples = pickle.load(f)

    print(f"Running {args.feature_type} with {args.sampling_method} sampling, sample size {args.sample_size}")

    if args.feature_type == "ela":
        extract_ela_features(samples, sampling_method, sample_size, data_dir, output_dir, dimension)
    else:
        extract_tla_features(
            samples, sampling_method, sample_size, data_dir, output_dir, dimension,
            n_jobs=args.n_jobs, use_rpp=args.use_ripser_plus_plus,
        )

    del samples
    gc.collect()
    with open(processed_files, 'a') as f:
        f.write(f"{args.sampling_method}_{args.sample_size}_{args.feature_type}\n")
    print(f"{args.sampling_method}_{args.sample_size}_{args.feature_type} done")


if __name__ == "__main__":
    main()