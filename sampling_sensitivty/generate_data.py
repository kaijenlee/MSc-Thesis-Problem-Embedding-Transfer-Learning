import argparse
import re
import sys

import cma
import cocoex
import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm
import pandas as pd
import os
import pickle

LOWER_BOUND = -5
UPPER_BOUND = 5


def parse_arguments():
    """
    Parse command line arguments for data generation.

    Returns:
        argparse.Namespace: Parsed arguments containing sampling_method, feature_method, and sample_size
    """
    parser = argparse.ArgumentParser(
        description='Generate data using specified sampling and feature extraction methods.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
              python generate_data.py -s cma_single -f ela -n 100
              python generate_data.py --sampling-method uniform --feature-method ela --sample-size 200
        """
    )

    parser.add_argument(
        '-s', '--sampling-method',
        type=str,
        required=True,
        choices=['cma_single', 'cma_random', 'uniform', 'lhs', 'ilhs', 'lhs_random_cd', 'sobol'],
        help='Sampling method to use for data generation',
        metavar='METHOD'
    )

    parser.add_argument(
        '-f', '--feature-method',
        type=str,
        required=True,
        choices=['ela', 'tla'],
        help='Feature extraction method (ela: Exploratory Landscape Analysis, cla: Topology Landscape Analysis)',
        metavar='METHOD'
    )

    parser.add_argument(
        '-n', '--sample-size',
        type=int,
        required=True,
        help='Number of samples multiplied by problem dimension to generate (integer)',
        metavar='SIZE'
    )

    parser.add_argument(
        '-r', '--runs',
        type=int,
        required=False,
        default=1,
        help='Number of runs',
        metavar='SIZE'
    )

    # Check if no arguments were provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Additional validation for sample_size
    if args.sample_size <= 0:
        parser.error(f"Sample size must be a positive integer, got {args.sample_size}")

    return args


def parse_problem_id(problem_id):
    """
    Parse a problem ID string in the format 'bbob_f001_i02_d02'.

    Args:
        problem_id: String in format 'bbob_f{function}_i{instance}_d{dimension}'

    Returns:
        tuple: (function, instance, dimension) as integers
    """
    # Method 1: Using regex
    match = re.match(r'bbob_f(\d+)_i(\d+)_d(\d+)', problem_id)
    if match:
        function = int(match.group(1))
        instance = int(match.group(2))
        dimension = int(match.group(3))
        return function, instance, dimension
    else:
        raise ValueError(f"Invalid problem ID format: {problem_id}")


def generate_random_samples(suite: cocoex.Suite, sample_size, runs):
    samples = {}
    for problem in suite:
        function, instance, dimension = parse_problem_id(problem.id)
        Xs = []
        Ys = []
        for run in range(runs):
            X = np.random.uniform(
                LOWER_BOUND,
                UPPER_BOUND,
                (sample_size * dimension, dimension)
            )
            Y = np.array([problem(x) for x in X])
            Xs.append(X)
            Ys.append(Y)
            samples[(function, instance, dimension, run)] = {'X': X, 'Y': Y}
        df = pd.DataFrame({
            "X": Xs,
            "Y": Ys
        })
        # df.to_csv(f"data/samples/random_{function}_{instance}_{dimension}_{sample_size}.csv")
        # print(f"Saved random_{function}_{instance}_{dimension}_{sample_size}.csv")
    return samples


def generate_lhs_samples(suite, sample_size, runs):
    """
    Generate samples using Latin Hypercube Sampling.

    Args:
        suite: COCO problem suite
        sample_size: Number of samples per dimension

    Returns:
        dict: Dictionary with (function, instance, dimension) as keys and {'X': X, 'Y': Y} as values
    """
    samples = {}
    for problem in suite:
        function, instance, dimension = parse_problem_id(problem.id)
        Xs = []
        Ys = []

        for run in range(runs):
            # Create Latin Hypercube sampler
            sampler = qmc.LatinHypercube(d=dimension)

            # Generate samples in [0, 1]^d
            X_unit = sampler.random(n=sample_size * dimension)

            # Scale to [LOWER_BOUND, UPPER_BOUND]
            X = qmc.scale(X_unit, LOWER_BOUND, UPPER_BOUND)

            # Evaluate the problem
            Y = np.array([problem(x) for x in X])
            Xs.append(X)
            Ys.append(Y)
            samples[(function, instance, dimension, run)] = {'X': X, 'Y': Y}
        df = pd.DataFrame({
            "X": Xs,
            "Y": Ys
        })
        # df.to_csv(f"data/samples/lhs_{function}_{instance}_{dimension}_{sample_size}.csv")
        # print(f"Saved lhs_{function}_{instance}_{dimension}_{sample_size}.csv")

    return samples


def _improved_lhs(n, k, dup=1):
    """
    Improved Latin Hypercube Sampling (Beachkofski-Grandhi algorithm).

    This implements the algorithm from:
        Beachkofski, B., Grandhi, R. (2002) "Improved Distributed Hypercube Sampling"
        American Institute of Aeronautics and Astronautics Paper 2002-1274.

    Based on the MATLAB implementation by John Burkardt and the R `lhs` package
    implementation (improvedLHS) used by the R `flacco` package.

    The algorithm greedily constructs a Latin Hypercube design one point at a time.
    At each step, it generates `dup` candidate points satisfying the LHS constraint,
    then selects the candidate whose nearest-neighbor distance to existing points is
    closest to the optimal spacing: opt = n / n^(1/k).

    Args:
        n: Number of sample points to generate.
        k: Number of dimensions.
        dup: Duplication factor controlling the number of candidate points per step.
             Higher values give better space-filling but slower computation.
             A value of 5 is reasonable (Burkardt). Default is 1.

    Returns:
        np.ndarray: An (n, k) array with values in [0, 1].
    """
    # Optimal spacing between points
    opt = n / n ** (1.0 / k)

    # Result matrix: stores the integer bin assignments (0-indexed) for each point
    # result[i, j] = the bin index for point i in dimension j
    result = np.empty((n, k), dtype=int)

    # Track which bins are available in each dimension
    # available[j] = list of available bin indices for dimension j
    available = [list(range(n)) for _ in range(k)]

    # Place the first point randomly
    for j in range(k):
        idx = np.random.randint(len(available[j]))
        result[0, j] = available[j].pop(idx)

    # Greedily place remaining points
    for i in range(1, n):
        # Number of candidates to generate
        n_candidates = dup * (n - i)

        # Generate candidate points: for each dimension, sample from available bins
        candidates = np.empty((n_candidates, k), dtype=int)
        for j in range(k):
            n_avail = len(available[j])
            # Sample bin indices (with replacement from available bins)
            rand_indices = np.random.randint(0, n_avail, size=n_candidates)
            candidates[:, j] = np.array(available[j])[rand_indices]

        # Compute distances from each candidate to all existing points
        # Use center-of-bin coordinates for distance calculation
        existing_coords = (result[:i, :] + 0.5) / n
        candidate_coords = (candidates + 0.5) / n

        # Compute minimum distance from each candidate to existing points
        dists = cdist(candidate_coords, existing_coords)
        min_dists = dists.min(axis=1)

        # Select the candidate whose min distance is closest to opt/n
        # (opt is in bin-space, so we compare in the same scale)
        target_dist = opt / n
        best_idx = np.argmin(np.abs(min_dists - target_dist))

        best_candidate = candidates[best_idx]

        # Record the chosen point
        result[i, :] = best_candidate

        # Remove chosen bins from available lists
        for j in range(k):
            chosen_bin = best_candidate[j]
            if chosen_bin in available[j]:
                available[j].remove(chosen_bin)

    # Transform integer bins to [0, 1] by adding a random offset within each cell
    design = np.empty((n, k))
    for i in range(n):
        for j in range(k):
            design[i, j] = (result[i, j] + np.random.uniform()) / n

    return design


def generate_ilhs_samples(suite, sample_size, runs):
    """
    Generate samples using Improved Latin Hypercube Sampling (Beachkofski-Grandhi).

    This is the iLHS method referenced in:
        Renau et al. (2020) "Exploratory Landscape Analysis is Strongly Sensitive
        to the Sampling Strategy", PPSN 2020.

    It uses the "improved" LHS designs from the R `lhs` package (improvedLHS),
    which implements the Beachkofski-Grandhi greedy algorithm that selects points
    to be as close to an optimal even spacing as possible.

    Args:
        suite: COCO problem suite
        sample_size: Number of samples per dimension
        runs: Number of independent runs

    Returns:
        dict: Dictionary with (function, instance, dimension, run) as keys
              and {'X': X, 'Y': Y} as values
    """
    samples = {}
    for problem in tqdm(suite):
        function, instance, dimension = parse_problem_id(problem.id)
        Xs = []
        Ys = []

        for run in range(runs):
            n = sample_size * dimension

            # Generate improved LHS design in [0, 1]^d
            # dup=5 is a reasonable default per Burkardt's recommendation,
            # matching typical usage in the R lhs package
            X_unit = _improved_lhs(n=n, k=dimension, dup=5)

            # Scale to [LOWER_BOUND, UPPER_BOUND]
            X = X_unit * (UPPER_BOUND - LOWER_BOUND) + LOWER_BOUND

            # Evaluate the problem
            Y = np.array([problem(x) for x in X])
            Xs.append(X)
            Ys.append(Y)
            samples[(function, instance, dimension, run)] = {'X': X, 'Y': Y}
        df = pd.DataFrame({
            "X": Xs,
            "Y": Ys
        })

    return samples


def generate_lhs_random_cd_samples(suite, sample_size, runs):
    """
    Generate samples using Latin Hypercube Sampling optimized via random
    centered-discrepancy criterion (scipy's optimization="random-cd").

    Note: This was previously called "ilhs" but has been renamed to avoid
    confusion with the Beachkofski-Grandhi Improved LHS algorithm used
    in the R flacco/lhs packages.

    Args:
        suite: COCO problem suite
        sample_size: Number of samples per dimension
        runs: Number of independent runs

    Returns:
        dict: Dictionary with (function, instance, dimension, run) as keys
              and {'X': X, 'Y': Y} as values
    """
    samples = {}
    for problem in tqdm(suite):
        function, instance, dimension = parse_problem_id(problem.id)
        Xs = []
        Ys = []

        for run in range(runs):
            # Create Latin Hypercube sampler with optimization for better space-filling
            sampler = qmc.LatinHypercube(d=dimension, optimization="random-cd")

            # Generate samples in [0, 1]^d
            X_unit = sampler.random(n=sample_size * dimension)

            # Scale to [LOWER_BOUND, UPPER_BOUND]
            X = qmc.scale(X_unit, LOWER_BOUND, UPPER_BOUND)

            # Evaluate the problem
            Y = np.array([problem(x) for x in X])
            Xs.append(X)
            Ys.append(Y)
            samples[(function, instance, dimension, run)] = {'X': X, 'Y': Y}
        df = pd.DataFrame({
            "X": Xs,
            "Y": Ys
        })

    return samples


def generate_sobol_samples(suite, sample_size, runs):
    """
    Generate samples using Sobol sequence (quasi-random low-discrepancy sequence).

    Args:
        suite: COCO problem suite
        sample_size: Number of samples per dimension

    Returns:
        dict: Dictionary with (function, instance, dimension) as keys and {'X': X, 'Y': Y} as values
    """
    samples = {}
    for problem in suite:
        function, instance, dimension = parse_problem_id(problem.id)
        Xs = []
        Ys = []
        for run in range(runs):
            # Create Sobol sampler
            sampler = qmc.Sobol(d=dimension, scramble=True)

            # Generate samples in [0, 1]^d
            # Note: Sobol sequences work best with powers of 2
            X_unit = sampler.random(n=sample_size * dimension)

            # Scale to [LOWER_BOUND, UPPER_BOUND]
            X = qmc.scale(X_unit, LOWER_BOUND, UPPER_BOUND)

            # Evaluate the problem
            Y = np.array([problem(x) for x in X])
            samples[(function, instance, dimension, run)] = {'X': X, 'Y': Y}
            Xs.append(X)
            Ys.append(Y)

        df = pd.DataFrame({
            "X": Xs,
            "Y": Ys
        })
        # df.to_csv(f"data/samples/sobol_{function}_{instance}_{dimension}_{sample_size}.csv")
        # print(f"Saved sobol_{function}_{instance}_{dimension}_{sample_size}.csv")

    return samples


def generate_cma_single_samples(suite, sample_size, runs, random_start_point=False):
    """
    Generate samples using CMA-ES with a single run.

    Args:
        suite: COCO problem suite
        sample_size: Number of samples per dimension

    Returns:
        dict: Dictionary with (function, instance, dimension) as keys and {'X': X, 'Y': Y} as values
    """
    # TODO: Implement CMA-ES sampling
    samples = {}
    for problem in tqdm(suite):
        function, instance, dimension = parse_problem_id(problem.id)
        Xs = []
        Ys = []
        for run in range(runs):
            samples[(function, instance, dimension, run)] = {'X': [], 'Y': []}
            # x0 = np.random.uniform(LOWER_BOUND, UPPER_BOUND, size=dimension)
            X_list = []
            Y_list = []
            budget = sample_size * dimension
            starting_point = np.random.uniform(
                LOWER_BOUND,
                UPPER_BOUND,
                dimension
            ) if random_start_point else dimension * [0]

            # Disable all stopping criteria
            opts = {
                'bounds': [LOWER_BOUND, UPPER_BOUND],
                'tolfun': 0,  # no function value tolerance
                'tolx': 0,  # no parameter change tolerance
                'tolstagnation': np.inf,  # no stagnation check
                'maxiter': np.inf,  # no iteration limit
                'maxfevals': np.inf,  # no function evaluation limit
                'verbose': -9  # suppress all output
            }
            es = cma.CMAEvolutionStrategy(starting_point, 2, opts)
            while budget > 0:
                X = es.ask()
                X_list.extend(X)
                Y = [problem(x) for x in X]
                Y_list.extend(Y)
                budget -= len(X)
                es.tell(X, Y)

            # Trim to exact size
            target_size = sample_size * dimension
            samples[(function, instance, dimension, run)] = {
                'X': np.array(X_list[:target_size]),
                'Y': np.array(Y_list[:target_size])
            }
            Xs.append(np.array(X_list[:target_size]))
            Ys.append(np.array(Y_list[:target_size]))
        df = pd.DataFrame({
            "X": Xs,
            "Y": Ys
        })
        # df.to_csv(f"data/samples/cma_{function}_{instance}_{dimension}_{sample_size}.csv")
        # print(f"Saved cma_{function}_{instance}_{dimension}_{sample_size}.csv")

    return samples


def generate_cma_indp_samples(suite, sample_size):
    """
    Generate samples using independent CMA-ES runs.

    Args:
        suite: COCO problem suite
        sample_size: Number of samples per dimension

    Returns:
        dict: Dictionary with (function, instance, dimension) as keys and {'X': X, 'Y': Y} as values
    """
    # TODO: Implement independent CMA-ES sampling
    # This requires the cma library: pip install cma
    raise NotImplementedError("CMA-ES independent runs sampling not yet implemented")


if __name__ == "__main__":
    args = parse_arguments()

    print(f"Sampling Method: {args.sampling_method}")
    print(f"Feature Method: {args.feature_method}")
    print(f"Sample Size: {args.sample_size}")
    os.makedirs('data/samples', exist_ok=True)
    os.makedirs('data/samples/pickles', exist_ok=True)

    suite = cocoex.Suite(
        "bbob",
        "year: 2009 instances: 1-100",
        "function_indices: 1-24 "
        "dimensions: 2,5,10 "  # TODO increase number of dimensions? 
        "instance_indices: 1-100"
    )
    samples = {}

    match args.sampling_method:
        case 'cma_random':  # random starting point
            samples = generate_cma_single_samples(suite, args.sample_size, args.runs, random_start_point=True)
        case 'cma_single':
            samples = generate_cma_single_samples(suite, args.sample_size, args.runs)
        case 'uniform':
            samples = generate_random_samples(suite, args.sample_size, args.runs)
        case 'lhs':
            samples = generate_lhs_samples(suite, args.sample_size, args.runs)
        case 'ilhs':
            samples = generate_ilhs_samples(suite, args.sample_size, args.runs)
        case 'lhs_random_cd':
            samples = generate_lhs_random_cd_samples(suite, args.sample_size, args.runs)
        case 'sobol':
            samples = generate_sobol_samples(suite, args.sample_size, args.runs)
        case _:
            raise ValueError(f"Unknown sampling method: {args.sampling_method}")

    print(f"Generated {len(samples)} problem samples for {args.runs} runs.")
    with open(f'data/samples/pickles/{args.sampling_method}_{args.sample_size}_{args.runs}.pkl', 'wb') as f:
        pickle.dump(samples, f)