import argparse
import re
import sys

import cocoex
from scipy.stats import qmc
import numpy as np

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

def generate_random_samples(suite: cocoex.Suite, sample_size):
    samples = {}
    for problem in suite:
        function, instance, dimension = parse_problem_id(problem.id)
        X = np.random.uniform (
            LOWER_BOUND,
            UPPER_BOUND,
            (sample_size * dimension, dimension)
        )
        Y = np.array([problem(x) for x in X])
        samples[(function, instance, dimension)] = {'X':X, 'Y':Y}
    return samples

def generate_lhs_samples(suite, sample_size):
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

        # Create Latin Hypercube sampler
        sampler = qmc.LatinHypercube(d=dimension)

        # Generate samples in [0, 1]^d
        X_unit = sampler.random(n=sample_size * dimension)

        # Scale to [LOWER_BOUND, UPPER_BOUND]
        X = qmc.scale(X_unit, LOWER_BOUND, UPPER_BOUND)

        # Evaluate the problem
        Y = np.array([problem(x) for x in X])
        samples[(function, instance, dimension)] = {'X': X, 'Y': Y}

    return samples


def generate_ilhs_samples(suite, sample_size):
    """
    Generate samples using Improved Latin Hypercube Sampling.

    Args:
        suite: COCO problem suite
        sample_size: Number of samples per dimension

    Returns:
        dict: Dictionary with (function, instance, dimension) as keys and {'X': X, 'Y': Y} as values
    """
    samples = {}
    for problem in suite:
        function, instance, dimension = parse_problem_id(problem.id)

        # Create Latin Hypercube sampler with optimization for better space-filling
        sampler = qmc.LatinHypercube(d=dimension, optimization="random-cd")

        # Generate samples in [0, 1]^d
        X_unit = sampler.random(n=sample_size * dimension)

        # Scale to [LOWER_BOUND, UPPER_BOUND]
        X = qmc.scale(X_unit, LOWER_BOUND, UPPER_BOUND)

        # Evaluate the problem
        Y = np.array([problem(x) for x in X])
        samples[(function, instance, dimension)] = {'X': X, 'Y': Y}

    return samples


def generate_sobol_samples(suite, sample_size):
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

        # Create Sobol sampler
        sampler = qmc.Sobol(d=dimension, scramble=True)

        # Generate samples in [0, 1]^d
        # Note: Sobol sequences work best with powers of 2
        X_unit = sampler.random(n=sample_size * dimension)

        # Scale to [LOWER_BOUND, UPPER_BOUND]
        X = qmc.scale(X_unit, LOWER_BOUND, UPPER_BOUND)

        # Evaluate the problem
        Y = np.array([problem(x) for x in X])
        samples[(function, instance, dimension)] = {'X': X, 'Y': Y}

    return samples


def generate_cma_single_samples(suite, sample_size):
    """
    Generate samples using CMA-ES with a single run.

    Args:
        suite: COCO problem suite
        sample_size: Number of samples per dimension

    Returns:
        dict: Dictionary with (function, instance, dimension) as keys and {'X': X, 'Y': Y} as values
    """
    # TODO: Implement CMA-ES sampling
    # This requires the cma library: pip install cma
    raise NotImplementedError("CMA-ES single run sampling not yet implemented")


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
            samples = generate_cma_single_samples(suite, args.sample_size)
        case 'cma_indp':
            samples = generate_cma_indp_samples(suite, args.sample_size)
        case 'uniform':
            samples = generate_random_samples(suite, args.sample_size)
        case 'lhs':
            samples = generate_lhs_samples(suite, args.sample_size)
        case 'ilhs':
            samples = generate_ilhs_samples(suite, args.sample_size)
        case 'sobol':
            samples = generate_sobol_samples(suite, args.sample_size)
        case _:
            raise ValueError(f"Unknown sampling method: {args.sampling_method}")

    print(f"Generated {len(samples)} problem samples")

    # TODO: Add feature extraction and save results
