from sklearn.decomposition import PCA
import numpy as np
import time

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

def calculate_basic_features(X, y, lower_bounds=None, upper_bounds=None,
                             blocks=None, minimize=True, return_dict=False):
    """
    Calculate basic features following flacco's specification.
    Works for ANY number of dimensions.

    Parameters:
    -----------
    X : numpy array of shape (n_observations, n_dimensions)
        Can be 1D, 2D, 3D, ..., nD
    y : numpy array of objective values
    lower_bounds : array-like, optional lower bounds per dimension
    upper_bounds : array-like, optional upper bounds per dimension
    blocks : array-like or int, optional number of blocks per dimension
    minimize : bool, whether minimizing (default True)
    return_dict : bool, if True returns dict, otherwise returns vector

    Returns:
    --------
    If return_dict=False: numpy array of 15 feature values
    If return_dict=True: dictionary with feature names as keys
    """
    start_time = time.time()

    n_obs, n_dim = X.shape

    # Set defaults if not provided
    if lower_bounds is None:
        lower_bounds = X.min(axis=0)
    else:
        lower_bounds = np.array(lower_bounds)

    if upper_bounds is None:
        upper_bounds = X.max(axis=0)
    else:
        upper_bounds = np.array(upper_bounds)

    if blocks is None:
        blocks = np.array([1] * n_dim)
    elif isinstance(blocks, int):
        blocks = np.array([blocks] * n_dim)
    else:
        blocks = np.array(blocks)

    # Calculate cells_filled
    cells_total = int(np.prod(blocks))

    if cells_total == 1:
        # No cell mapping - single cell contains all points
        cells_filled = 1
    else:
        # Assign each observation to a cell
        cell_assignments = _assign_observations_to_cells(X, lower_bounds, upper_bounds, blocks)
        # Count unique non-empty cells
        cells_filled = len(np.unique(cell_assignments))

    runtime = time.time() - start_time

    # Calculate features in the order they appear in flacco
    feature_values = [
        n_dim,  # dim
        n_obs,  # observations
        np.min(lower_bounds),  # lower_min
        np.max(lower_bounds),  # lower_max
        np.min(upper_bounds),  # upper_min
        np.max(upper_bounds),  # upper_max
        np.min(y),  # objective_min
        np.max(y),  # objective_max
        np.min(blocks),  # blocks_min
        np.max(blocks),  # blocks_max
        float(cells_filled),  # cells_filled
        float(cells_total),  # cells_total
        float(minimize),  # minimize_fun (convert bool to float)
        0.0,  # costs_fun_evals
        runtime  # costs_runtime
    ]

    if return_dict:
        feature_names = [
            'basic.dim',
            'basic.observations',
            'basic.lower_min',
            'basic.lower_max',
            'basic.upper_min',
            'basic.upper_max',
            'basic.objective_min',
            'basic.objective_max',
            'basic.blocks_min',
            'basic.blocks_max',
            'basic.cells_filled',
            'basic.cells_total',
            'basic.minimize_fun',
            'basic.costs_fun_evals',
            'basic.costs_runtime'
        ]
        return dict(zip(feature_names, feature_values))
    else:
        return np.array(feature_values)


def _assign_observations_to_cells(X, lower_bounds, upper_bounds, blocks):
    """
    Assign each observation to a cell in the grid.
    Works for ANY number of dimensions.

    Parameters:
    -----------
    X : numpy array of shape (n_observations, n_dimensions)
    lower_bounds : numpy array of lower bounds per dimension
    upper_bounds : numpy array of upper bounds per dimension
    blocks : numpy array of number of blocks per dimension

    Returns:
    --------
    cell_ids : numpy array of shape (n_observations,)
        Linear index of the cell each observation belongs to
    """
    n_obs, n_dim = X.shape

    # Calculate cell size per dimension
    cell_sizes = (upper_bounds - lower_bounds) / blocks

    # Determine which cell each observation belongs to in each dimension
    # Cell indices start at 0
    cell_indices = np.floor((X - lower_bounds) / cell_sizes).astype(int)

    # Handle edge case: observations exactly at upper bound
    for d in range(n_dim):
        cell_indices[cell_indices[:, d] >= blocks[d], d] = blocks[d] - 1

    # Convert multi-dimensional cell indices to linear indices
    # This is like converting (i, j, k) to a single number
    cell_ids = np.zeros(n_obs, dtype=int)
    multiplier = 1
    for d in range(n_dim):
        cell_ids += cell_indices[:, d] * multiplier
        multiplier *= blocks[d]

    return cell_ids