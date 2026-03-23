"""
Extend existing sample pkl files with dimension 10 samples.

Loads an existing pkl file, generates dimension 10 samples using the same
sampling method and sample size, and saves the merged result.

Usage:
  python extend_samples_dim10.py -s ilhs -n 50 -r 30 --input-dir data/samples/pickles
  python extend_samples_dim10.py -s sobol -n 100 -r 30 --input-dir data/samples/pickles --output-dir data/samples/pickles
"""

import argparse
import os
import sys
import pickle
import re

import cma
import cocoex
import numpy as np
from scipy.stats import qmc
from tqdm.auto import tqdm

LOWER_BOUND = -5
UPPER_BOUND = 5
TARGET_DIMENSION = 10


def parse_problem_id(problem_id):
    match = re.match(r'bbob_f(\d+)_i(\d+)_d(\d+)', problem_id)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    raise ValueError(f"Invalid problem ID format: {problem_id}")


# ---------------------------------------------------------------------------
# Sampling methods (same as original script)
# ---------------------------------------------------------------------------

def generate_random_samples(suite, sample_size, runs):
    samples = {}
    for problem in tqdm(suite, desc="Uniform"):
        function, instance, dimension = parse_problem_id(problem.id)
        for run in range(runs):
            X = np.random.uniform(LOWER_BOUND, UPPER_BOUND,
                                  (sample_size * dimension, dimension))
            Y = np.array([problem(x) for x in X])
            samples[(function, instance, dimension, run)] = {'X': X, 'Y': Y}
    return samples


def generate_lhs_samples(suite, sample_size, runs):
    samples = {}
    for problem in tqdm(suite, desc="LHS"):
        function, instance, dimension = parse_problem_id(problem.id)
        for run in range(runs):
            sampler = qmc.LatinHypercube(d=dimension)
            X_unit = sampler.random(n=sample_size * dimension)
            X = qmc.scale(X_unit, LOWER_BOUND, UPPER_BOUND)
            Y = np.array([problem(x) for x in X])
            samples[(function, instance, dimension, run)] = {'X': X, 'Y': Y}
    return samples


def generate_ilhs_samples(suite, sample_size, runs):
    samples = {}
    for problem in tqdm(suite, desc="iLHS"):
        function, instance, dimension = parse_problem_id(problem.id)
        for run in range(runs):
            sampler = qmc.LatinHypercube(d=dimension, optimization="random-cd")
            X_unit = sampler.random(n=sample_size * dimension)
            X = qmc.scale(X_unit, LOWER_BOUND, UPPER_BOUND)
            Y = np.array([problem(x) for x in X])
            samples[(function, instance, dimension, run)] = {'X': X, 'Y': Y}
    return samples


def generate_sobol_samples(suite, sample_size, runs):
    samples = {}
    for problem in tqdm(suite, desc="Sobol"):
        function, instance, dimension = parse_problem_id(problem.id)
        for run in range(runs):
            sampler = qmc.Sobol(d=dimension, scramble=True)
            X_unit = sampler.random(n=sample_size * dimension)
            X = qmc.scale(X_unit, LOWER_BOUND, UPPER_BOUND)
            Y = np.array([problem(x) for x in X])
            samples[(function, instance, dimension, run)] = {'X': X, 'Y': Y}
    return samples


def generate_cma_random_samples(suite, sample_size, runs):
    samples = {}
    for problem in tqdm(suite, desc="CMA-Random"):
        function, instance, dimension = parse_problem_id(problem.id)
        for run in range(runs):
            X_list, Y_list = [], []
            budget = sample_size * dimension
            starting_point = np.random.uniform(LOWER_BOUND, UPPER_BOUND, dimension)
            opts = {
                'bounds': [LOWER_BOUND, UPPER_BOUND],
                'tolfun': 0, 'tolx': 0,
                'tolstagnation': np.inf,
                'maxiter': np.inf, 'maxfevals': np.inf,
                'verbose': -9,
            }
            es = cma.CMAEvolutionStrategy(starting_point, 2, opts)
            while budget > 0:
                X = es.ask()
                X_list.extend(X)
                Y = [problem(x) for x in X]
                Y_list.extend(Y)
                budget -= len(X)
                es.tell(X, Y)
            target_size = sample_size * dimension
            samples[(function, instance, dimension, run)] = {
                'X': np.array(X_list[:target_size]),
                'Y': np.array(Y_list[:target_size]),
            }
    return samples


SAMPLING_FUNCTIONS = {
    "uniform": generate_random_samples,
    "lhs": generate_lhs_samples,
    "ilhs": generate_ilhs_samples,
    "sobol": generate_sobol_samples,
    "cma_random": generate_cma_random_samples,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extend existing sample pkl files with dimension 10 samples."
    )
    parser.add_argument("-s", "--sampling-method", type=str, required=True,
                        choices=list(SAMPLING_FUNCTIONS.keys()),
                        help="Sampling method.")
    parser.add_argument("-n", "--sample-size", type=int, required=True,
                        help="Sample size multiplier (n * dimension).")
    parser.add_argument("-r", "--runs", type=int, default=30,
                        help="Number of runs (default: 30).")
    parser.add_argument("--input-dir", type=str, default="data/samples/pickles",
                        help="Directory containing existing pkl files.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Defaults to input-dir.")
    parser.add_argument("--dim", type=int, default=TARGET_DIMENSION,
                        help=f"Dimension to add (default: {TARGET_DIMENSION}).")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    pkl_filename = f"{args.sampling_method}_{args.sample_size}_{args.runs}.pkl"
    pkl_path = os.path.join(args.input_dir, pkl_filename)

    # Load existing data
    if os.path.exists(pkl_path):
        print(f"Loading existing data from: {pkl_path}")
        with open(pkl_path, "rb") as f:
            existing_samples = pickle.load(f)
        print(f"  Existing keys: {len(existing_samples)}")

        # Check which dimensions already exist
        existing_dims = set()
        for key in existing_samples:
            existing_dims.add(key[2])  # dimension is index 2
        print(f"  Existing dimensions: {sorted(existing_dims)}")

        if args.dim in existing_dims:
            # Check if all expected keys for this dimension exist
            expected_count = 24 * 100 * args.runs  # functions * instances * runs
            dim_keys = [k for k in existing_samples if k[2] == args.dim]
            if len(dim_keys) >= expected_count:
                print(f"  Dimension {args.dim} already complete "
                      f"({len(dim_keys)} keys). Nothing to do.")
                return
            else:
                print(f"  Dimension {args.dim} partially exists "
                      f"({len(dim_keys)}/{expected_count} keys). "
                      f"Will generate missing entries.")
    else:
        print(f"No existing file at {pkl_path}. Creating new file.")
        existing_samples = {}

    # Create suite for target dimension only
    print(f"\nGenerating dimension {args.dim} samples...")
    print(f"  Method: {args.sampling_method}")
    print(f"  Sample size: {args.sample_size} × d = {args.sample_size * args.dim}")
    print(f"  Runs: {args.runs}")

    suite = cocoex.Suite(
        "bbob",
        "year: 2009 instances: 1-100",
        f"function_indices: 1-24 "
        f"dimensions: {args.dim} "
        f"instance_indices: 1-100"
    )

    # Generate new samples
    sampling_func = SAMPLING_FUNCTIONS[args.sampling_method]
    new_samples = sampling_func(suite, args.sample_size, args.runs)
    print(f"  Generated {len(new_samples)} new keys")

    # Merge
    merged = {**existing_samples, **new_samples}
    print(f"\nMerged: {len(existing_samples)} existing + {len(new_samples)} new "
          f"= {len(merged)} total")

    # Verify dimensions
    merged_dims = set()
    for key in merged:
        merged_dims.add(key[2])
    print(f"Dimensions in merged file: {sorted(merged_dims)}")

    for dim in sorted(merged_dims):
        dim_keys = [k for k in merged if k[2] == dim]
        print(f"  Dimension {dim}: {len(dim_keys)} keys")

    # Save
    output_path = os.path.join(output_dir, pkl_filename)
    print(f"\nSaving to: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(merged, f)
    print("Done!")


if __name__ == "__main__":
    main()