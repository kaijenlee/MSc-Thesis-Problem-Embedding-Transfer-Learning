from sklearn.decomposition import PCA
import numpy as np

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