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

# Master seed folded into every per-design seed. Default 42 makes the whole
# experiment reproducible out of the box; override with --base-seed to generate
# a different, fully-reproducible-but-independent realization of all runs
# (e.g. to confirm stability conclusions aren't an artifact of one seed family).
BASE_SEED = 42

# --- rpy2 bridge for canonical iLHS (R lhs::improvedLHS) ---------------------
# rpy2 / R are imported lazily inside the helpers below, so this script only
# requires R + rpy2 installed when you actually run `-s ilhs`. All other
# sampling methods work with a plain Python environment.
_LHS_PKG = None


def _get_lhs():
    """Import and cache R's `lhs` package (loaded lazily on first iLHS call)."""
    global _LHS_PKG
    if _LHS_PKG is None:
        from rpy2.robjects.packages import importr
        _LHS_PKG = importr("lhs")
    return _LHS_PKG


def _design_seed(function, instance, dimension, run):
    """Deterministic, well-mixed seed unique to each design.

    Built from BASE_SEED plus the (function, instance, dimension, run)
    coordinates via SeedSequence, so structurally-close keys (e.g. consecutive
    runs) still yield statistically independent RNG streams. The same integer is
    fed to numpy, scipy.stats.qmc, R's set.seed, and pycma -- so EVERY sampler is
    fully reproducible from BASE_SEED + these coordinates, and re-running the
    script reproduces the exact same samples.

    COCO's problem(x) is deterministic, so seeding the sampler is sufficient.
    """
    ss = np.random.SeedSequence([int(BASE_SEED), int(function), int(instance),
                                 int(dimension), int(run)])
    # uint32; avoid 0 because pycma treats seed=0 as "use a random seed".
    return int(ss.generate_state(1)[0]) or 1


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

    parser.add_argument(
        '--base-seed',
        type=int,
        required=False,
        default=42,
        help='Master seed folded into every per-design seed (default: 42).',
        metavar='SEED'
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
            rng = np.random.default_rng(
                _design_seed(function, instance, dimension, run))
            X = rng.uniform(
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
            # Create Latin Hypercube sampler (seeded for reproducibility)
            sampler = qmc.LatinHypercube(
                d=dimension,
                seed=_design_seed(function, instance, dimension, run))

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


def _improved_lhs(n, k, dup=1, seed=None):
    """
    Improved Latin Hypercube Sampling (Beachkofski-Grandhi) via R's `lhs` package.

    Delegates to lhs::improvedLHS through rpy2 -- the exact implementation that
    flacco::createInitialSample(type="lhs") calls internally -- instead of a
    hand-rolled port. Verified distributionally identical to the lhslib C++
    reference (Mann-Whitney p > 0.17 across configs) and bit-identical to
    standalone Rscript under the same set.seed.

    dup=1 matches flacco (it calls improvedLHS(n, k) with no dup argument).
    Pass a `seed` for a reproducible design.

    Args:
        n: Number of sample points.
        k: Number of dimensions.
        dup: Duplication factor (candidates per placement step). 1 = flacco.
        seed: Optional integer; sets R's RNG for a reproducible design.

    Returns:
        np.ndarray: An (n, k) array with values in [0, 1].
    """
    import rpy2.robjects as ro
    lhs = _get_lhs()
    if seed is not None:
        # R's set.seed requires a signed 32-bit int; our seeds are uint32,
        # so fold into [1, 2**31 - 1] deterministically.
        r_seed = int(seed) % (2**31 - 1) or 1
        ro.r(f"set.seed({r_seed})")
    X_unit = np.array(lhs.improvedLHS(int(n), int(k), dup=int(dup)))  # [0, 1]^k
    return X_unit


def generate_ilhs_samples(suite, sample_size, runs, dup=1):
    """
    Generate samples using Improved Latin Hypercube Sampling (Beachkofski-Grandhi)
    via R's `lhs` package (lhs::improvedLHS through rpy2).

    This is the iLHS method referenced in:
        Renau et al. (2020) "Exploratory Landscape Analysis is Strongly Sensitive
        to the Sampling Strategy", PPSN 2020.

    Uses the same implementation the R `flacco` package calls internally, rather
    than a hand-rolled port. Each (function, instance, run) gets a fresh,
    deterministically-seeded design -- so results are reproducible and each
    design is independent -- matching how the other samplers in this script
    draw a new design per problem.

    Args:
        suite: COCO problem suite
        sample_size: Number of samples per dimension
        runs: Number of independent runs
        dup: Duplication factor for improvedLHS (1 = flacco default)

    Returns:
        dict: Dictionary with (function, instance, dimension, run) as keys
              and {'X': X, 'Y': Y} as values
    """
    samples = {}
    for problem in tqdm(suite):
        function, instance, dimension = parse_problem_id(problem.id)
        for run in range(runs):
            n = sample_size * dimension

            # Unique, reproducible seed per design -> reproducible + independent
            seed = _design_seed(function, instance, dimension, run)

            # Improved LHS design in [0, 1]^d via R lhs::improvedLHS
            X_unit = _improved_lhs(n=n, k=dimension, dup=dup, seed=seed)

            # Scale to [LOWER_BOUND, UPPER_BOUND]
            X = X_unit * (UPPER_BOUND - LOWER_BOUND) + LOWER_BOUND

            # Evaluate the problem
            Y = np.array([problem(x) for x in X])
            samples[(function, instance, dimension, run)] = {'X': X, 'Y': Y}

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
            sampler = qmc.LatinHypercube(
                d=dimension, optimization="random-cd",
                seed=_design_seed(function, instance, dimension, run))

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
            # Create Sobol sampler (seeded scramble for reproducibility)
            sampler = qmc.Sobol(
                d=dimension, scramble=True,
                seed=_design_seed(function, instance, dimension, run))

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
            seed = _design_seed(function, instance, dimension, run)
            rng = np.random.default_rng(seed)
            # x0 = np.random.uniform(LOWER_BOUND, UPPER_BOUND, size=dimension)
            X_list = []
            Y_list = []
            budget = sample_size * dimension
            starting_point = rng.uniform(
                LOWER_BOUND,
                UPPER_BOUND,
                dimension
            ) if random_start_point else dimension * [0]

            # Disable all stopping criteria; 'seed' makes CMA-ES sampling reproducible
            opts = {
                'bounds': [LOWER_BOUND, UPPER_BOUND],
                'tolfun': 0,  # no function value tolerance
                'tolx': 0,  # no parameter change tolerance
                'tolstagnation': np.inf,  # no stagnation check
                'maxiter': np.inf,  # no iteration limit
                'maxfevals': np.inf,  # no function evaluation limit
                'seed': int(seed),  # reproducible CMA-ES sampling (0 would randomize)
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
    # Runs at module scope, so this rebinds the global BASE_SEED that
    # _design_seed reads. Set it before any sampler is called.
    BASE_SEED = args.base_seed
    print(f"Base Seed: {BASE_SEED}")
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