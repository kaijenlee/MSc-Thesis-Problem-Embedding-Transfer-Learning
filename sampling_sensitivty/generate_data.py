import argparse
import re
import sys

import cma
import cocoex
import numpy as np
from scipy.stats import qmc
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
        choices=['cma_single', 'cma_indp', 'uniform', 'lhs', 'ilhs', 'sobol'],
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
        df.to_csv(f"data/samples/random_{function}_{instance}_{dimension}_{sample_size}.csv")
        print(f"Saved random_{function}_{instance}_{dimension}_{sample_size}.csv")
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
        df.to_csv(f"data/samples/lhs_{function}_{instance}_{dimension}_{sample_size}.csv")
        print(f"Saved lhs_{function}_{instance}_{dimension}_{sample_size}.csv")

    return samples


def generate_ilhs_samples(suite, sample_size, runs):
    """
    Generate samples using Improved Latin Hypercube Sampling.

    Args:
        suite: COCO problem suite
        sample_size: Number of samples per dimension

    Returns:
        dict: Dictionary with (function, instance, dimension) as keys and {'X': X, 'Y': Y} as values
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
        df.to_csv(f"data/samples/ilhs_{function}_{instance}_{dimension}_{sample_size}.csv")
        print(f"Saved ilhs_{function}_{instance}_{dimension}_{sample_size}.csv")

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
        df.to_csv(f"data/samples/sobol_{function}_{instance}_{dimension}_{sample_size}.csv")
        print(f"Saved sobol_{function}_{instance}_{dimension}_{sample_size}.csv")

    return samples


def generate_cma_single_samples(suite, sample_size, runs):
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
            starting_point = dimension * [0]

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
            es = cma.CMAEvolutionStrategy(starting_point, 1, opts)
            while budget > 0:
                X = es.ask()
                X_list.extend(X)
                Y = [problem(x) for x in X]
                Y_list.extend(Y)
                budget -= len(X)
                es.tell(X, Y)

                if es.stop():
                    es.stop().clear()  # force CMA-ES to clear stopping conditions

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
        df.to_csv(f"data/samples/cma_{function}_{instance}_{dimension}_{sample_size}.csv")
        print(f"Saved cma_{function}_{instance}_{dimension}_{sample_size}.csv")

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
        "dimensions: 2,3,5 "  # TODO increase number of dimensions? 
        "instance_indices: 1-100"
    )
    samples = {}

    match args.sampling_method:
        case 'cma_single':
            samples = generate_cma_single_samples(suite, args.sample_size, args.runs)
        case 'uniform':
            samples = generate_random_samples(suite, args.sample_size, args.runs)
        case 'lhs':
            samples = generate_lhs_samples(suite, args.sample_size, args.runs)
        case 'ilhs':
            samples = generate_ilhs_samples(suite, args.sample_size, args.runs)
        case 'sobol':
            samples = generate_sobol_samples(suite, args.sample_size, args.runs)
        case _:
            raise ValueError(f"Unknown sampling method: {args.sampling_method}")

    print(f"Generated {len(samples)} problem samples for {args.runs} runs.")
    with open(f'data/samples/pickles/{args.sampling_method}_{args.sample_size}_{args.runs}.pkl', 'wb') as f:
        pickle.dump(samples, f)


