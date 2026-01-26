import argparse
import os
from pathlib import Path
from ripser import Rips
from tqdm.auto import tqdm
from persim import PersistenceImager
import numpy as np
from pflacco.classical_ela_features import *
import pickle
from scipy.spatial.distance import cdist


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

    # Bounding box dimensions along each axis
    box_dimensions = max_vals - min_vals

    current_hypervolume = np.prod(box_dimensions)

    if current_hypervolume > 0:
        scale_factor = (1.0 / current_hypervolume) ** (1.0 / n_dimensions)
    else:
        scale_factor = 1.0

    X_scaled = centered_points * scale_factor
    return X_scaled


## Distance between sample
def get_distance(X_D, Y_D, alpha=0.2):
    return alpha * X_D + (1 - alpha) * Y_D


def extract_ela_features(data, sampling_method, sample_size, data_dir):
    """
    Extract ELA (Exploratory Landscape Analysis) features.
    """
    features = {}
    for dimension in [2]:
        # for function in tqdm(range(1, 25), position=0):
        for function in range(1, 25):
            # for instance in tqdm(range(1, 101), position=1, desc=f"ELA Sampling {sampling_method}, {sample_size} - Function {function}, dimension {dimension}"):
            for instance in range(1,101):
                filename = data_dir / "features" / "pickles" / f"ela_{sampling_method}_{sample_size}_{function}_{instance}_{dimension}.pkl"
                #
                # if filename.exists():
                #     # print(
                #     #     f"Skipping as ELA - Sampling {sampling_method}, {sample_size} - Function {function} - Instance {instance} - Dimension {dimension} exists")
                #     try:
                #         with open(filename, 'rb') as f:
                #             file_done = pickle.load(f)
                #             features[(function, instance, dimension)] = file_done
                #             continue
                #     except EOFError:
                #         print(f"{filename} is empty or corrupted")

                # print(
                #     f"Processing ELA - Sampling {sampling_method}, {sample_size} - Function {function} - Instance {instance} - Dimension {dimension}...")
                features[(function, instance, dimension)] = {
                    "ela_dist": [],
                    "levelset": [],
                    "meta": [],
                    "disp": [],
                    "ic": [],
                    "nbc": [],
                }
                for runs in range(0, 30):
                    samples = data[(function, instance, dimension, runs)]
                    X = samples['X'] if sampling_method != "cma" else samples['X'][:sample_size * dimension]
                    Y = samples['Y'] if sampling_method != "cma" else samples['Y'][:sample_size * dimension]

                    features[(function, instance, dimension)]["ela_dist"].append(calculate_ela_distribution(X, Y))
                    try:
                        levelset_features = calculate_ela_level(X, Y, ela_level_resample_iterations=5)
                        features[(function, instance, dimension)]["levelset"].append(levelset_features)
                    except Exception as e:
                        print(f"Error in levelset for {sampling_method}-{sample_size}: {function}-{instance}-{dimension}: {e}")
                        features[(function, instance, dimension)]["levelset"].append({})
                    features[(function, instance, dimension)]["meta"].append(calculate_ela_meta(X, Y))
                    features[(function, instance, dimension)]["disp"].append(calculate_dispersion(X, Y))
                    features[(function, instance, dimension)]["ic"].append(
                        calculate_information_content(X, Y, seed=100))
                    features[(function, instance, dimension)]["nbc"].append(calculate_nbc(X, Y))

                # with open(filename, 'wb') as f:
                #     pickle.dump(features[(function, instance, dimension)], f)

    return features


def extract_tla_features(data, sampling_method, sample_size, data_dir):
    """
    Extract TLA (Topological Landscape Analysis) features.

    TODO: Implement TLA feature extraction logic
    """
    max_range = 1.0
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

    features = {}
    for dimension in [2]:
        # for function in tqdm(range(1, 25), position=0):
        for function in range(1, 25):
            # for instance in tqdm(range(1, 101), position=1, desc=f"TLA Sampling {sampling_method}, {sample_size} - Function {function}, dimension {dimension}"):
            for instance in range(1, 101):
                filename = data_dir / "features" / "pickles" / f"tla_{sampling_method}_{sample_size}_{function}_{instance}_{dimension}.pkl"

                if filename.exists():
                    # print(
                    #     f"Skipping as TLA - Sampling {sampling_method}, {sample_size} - Function {function} - Instance {instance} - Dimension {dimension} exists")
                    try:
                        with open(filename, 'rb') as f:
                            file_done = pickle.load(f)
                            features[(function, instance, dimension)] = file_done
                            continue
                    except EOFError:
                        print(f"{filename} is empty or corrupted")

                # print(
                #     f"Processing TLA - Sampling {sampling_method}, {sample_size} - Function {function} - Instance {instance} - Dimension {dimension}...")
                features[(function, instance, dimension)] = {
                    'volume': {
                        'h0': [],
                        'h1': [],
                        'h2': []
                    },
                    'axis': {
                        'h0': [],
                        'h1': [],
                        'h2': []
                    }
                }
                for runs in range(0, 30):
                    samples = data[(function, instance, dimension, runs)]
                    X = samples['X'] if sampling_method != "cma" else samples['X'][:sample_size * dimension]
                    Y = samples['Y'] if sampling_method != "cma" else samples['Y'][:sample_size * dimension]

                    X_volume = volume_transform(X)
                    X_axis = axis_transform(X)

                    # Normalizing
                    X_volume_D = cdist(X_volume, X_volume, "euclidean")
                    X_volume_D_norm = X_volume_D / np.abs(X_volume_D).max(axis=0)
                    X_axis_D = cdist(X_axis, X_axis, "euclidean")
                    X_axis_D_norm = X_axis_D / np.abs(X_axis_D).max(axis=0)

                    Y_D = cdist(np.asmatrix(Y).T, np.asmatrix(Y).T, "euclidean")
                    Y_D_norm = Y_D / Y_D.max()

                    rips_volume = Rips(maxdim=dimension, coeff=2, verbose=False)
                    rips_axis = Rips(maxdim=dimension, coeff=2, verbose=False)

                    distances_volume = get_distance(X_volume_D_norm, Y_D_norm, alpha=alpha)
                    distances_axis = get_distance(X_axis_D_norm, Y_D_norm, alpha=alpha)

                    diagrams_volume = rips_volume.fit_transform(distances_volume, distance_matrix=True)
                    diagrams_axis = rips_axis.fit_transform(distances_axis, distance_matrix=True)

                    d0_volume = diagrams_volume[0]
                    d1_volume = diagrams_volume[1]
                    d2_volume = diagrams_volume[2]

                    d0_axis = diagrams_axis[0]
                    d1_axis = diagrams_axis[1]
                    d2_axis = diagrams_axis[2]

                    sel0_volume = np.isfinite(d0_volume.sum(axis=1))
                    img0_volume = pimgr0.transform(d0_volume[sel0_volume, :])

                    sel1_volume = np.isfinite(d1_volume.sum(axis=1))
                    img1_volume = pimgr1.transform(d1_volume[sel1_volume, :])

                    sel2_volume = np.isfinite(d2_volume.sum(axis=1))
                    img2_volume = pimgr2.transform(d2_volume[sel2_volume, :])

                    sel0_axis = np.isfinite(d0_axis.sum(axis=1))
                    img0_axis = pimgr0.transform(d0_axis[sel0_axis, :])

                    sel1_axis = np.isfinite(d1_axis.sum(axis=1))
                    img1_axis = pimgr1.transform(d1_axis[sel1_axis, :])

                    sel2_axis = np.isfinite(d2_axis.sum(axis=1))
                    img2_axis = pimgr2.transform(d2_axis[sel2_axis, :])

                    features[(function, instance, dimension)]['volume']['h0'].append(img0_volume)
                    features[(function, instance, dimension)]['volume']['h1'].append(img1_volume)
                    features[(function, instance, dimension)]['volume']['h2'].append(img2_volume)
                    features[(function, instance, dimension)]['axis']['h0'].append(img0_axis)
                    features[(function, instance, dimension)]['axis']['h1'].append(img1_axis)
                    features[(function, instance, dimension)]['axis']['h2'].append(img2_axis)
                    
                # with open(filename, 'wb') as f:
                #     pickle.dump(features[(function, instance, dimension)], f)

    return features


def main():
    parser = argparse.ArgumentParser(
        description="Extract landscape features from optimization problems"
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        required=True,
        choices=["ela", "tla"],
        help="Type of features to extract: 'ela' for Exploratory Landscape Analysis or 'tla' for Topological Landscape Analysis"
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        required=True,
        choices=["uniform", "lhs", "ilhs", "sobol", "cma", "cma_random"],
        help="Sampling method to use: uniform, lhs (Latin Hypercube Sampling), ilhs (Improved Latin Hypercube Sampling), sobol, or cma"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        required=True,
        choices=[10, 25, 50, 75, 100],
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing CSV files"
    )

    args = parser.parse_args()

    # Convert data_dir to Path and resolve relative paths
    data_dir = Path(args.data_dir).resolve()
    file_name = data_dir / "features" / "pickles" / f"{args.sampling_method}_{args.sample_size}_{args.feature_type}.pkl"
    if file_name.exists():
        print(f"{args.sampling_method}_{args.sample_size}_{args.feature_type}.pkl already exists. Skipping feature extraction.")
        return

    # TODO fix this
    sampling_method = "cma_single" if args.sampling_method == "cma" else args.sampling_method
    sample_size = args.sample_size
    runs = 30

    os.makedirs(data_dir / "features", exist_ok=True)
    os.makedirs(data_dir / "features" / "pickles", exist_ok=True)

    pickle_file = f"{sampling_method}_100_{runs}.pkl" if args.sampling_method == "cma" else f"{sampling_method}_{sample_size}_{runs}.pkl"

    with open(data_dir / pickle_file, "rb") as f:
        data = pickle.load(f)

    print(f"Running feature extraction: {args.feature_type} with {args.sampling_method} sampling and sample size {args.sample_size}")
    features = extract_ela_features(data, sampling_method, sample_size, data_dir) if args.feature_type == "ela" else extract_tla_features(data, sampling_method, sample_size, data_dir)

    with open(data_dir / "features" / "pickles" / f"{args.sampling_method}_{args.sample_size}_{args.feature_type}.pkl",
              'wb') as f:
        pickle.dump(features, f)

    print(f"{args.sampling_method}_{args.sample_size}_{args.feature_type}.pkl done!")


if __name__ == "__main__":
    main()
