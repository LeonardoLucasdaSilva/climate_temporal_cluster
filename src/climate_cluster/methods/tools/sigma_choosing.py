"""Helpers for choosing spectral-clustering sigma values."""

from __future__ import annotations

import numpy as np
import pandas as pd

from climate_cluster.methods.tools.sliding_windows import create_windows


SIGMA_WINDOW_SIZE = 5
SIGMA_LOWER_QUANTILE = 0.01
SIGMA_UPPER_QUANTILE = 0.20


def euclidian_distances(data: np.ndarray) -> np.ndarray:
    """Calculate pairwise Euclidean distances between data points.

    Args:
        data: Input data array of shape (n_samples, n_features).

    Returns:
        Distance matrix with shape (n_samples, n_samples), where each entry is
        the Euclidean distance between two rows.
    """
    n_samples = len(data)
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            distances[i, j] = np.linalg.norm(data[i] - data[j])
    return distances


def take_sigma(distance_matrix: np.ndarray) -> np.ndarray:
    """Generate sigma candidates from a pairwise distance distribution.

    The returned values span the 1st to 20th percentile of upper-triangular
    pairwise distances, matching the original project heuristic.

    Args:
        distance_matrix: Square pairwise distance matrix.

    Returns:
        Array of 20 sigma values suitable for Gaussian-kernel sweeps.
    """
    distances = distance_matrix[np.triu_indices(len(distance_matrix), k=1)].copy()
    distances.sort()

    n_distances = len(distances)
    if n_distances == 0:
        raise ValueError("At least two samples are needed to calculate sigma values.")

    start = distances[int(0.01 * n_distances)]
    end = distances[int(0.2 * n_distances)]
    step = (end - start) / 19
    sigmas = np.array([(start + i * step) for i in range(20)])
    sigmas[sigmas == 0] = 1e-2

    return sigmas


def sigma_values_from_distance_distribution(
    distance_matrix: np.ndarray,
    n_values: int = 20,
    lower_quantile: float = SIGMA_LOWER_QUANTILE,
    upper_quantile: float = SIGMA_UPPER_QUANTILE,
) -> np.ndarray:
    """Choose sigma values from the lower tail of pairwise distances.

    Args:
        distance_matrix: Square pairwise distance matrix.
        n_values: Number of sigma values to generate.
        lower_quantile: Lower distance quantile used as the first sigma.
        upper_quantile: Upper distance quantile used as the last sigma.

    Returns:
        Linearly spaced sigma candidates across the selected quantile range.
    """
    if n_values <= 0:
        raise ValueError(f"n_values must be positive, got {n_values}")
    if (
        n_values == 20
        and lower_quantile == SIGMA_LOWER_QUANTILE
        and upper_quantile == SIGMA_UPPER_QUANTILE
    ):
        return take_sigma(distance_matrix)

    distances = distance_matrix[np.triu_indices(len(distance_matrix), k=1)].copy()
    distances.sort()
    if len(distances) == 0:
        raise ValueError("At least two windows are needed to calculate sigma values.")

    start_idx = min(int(lower_quantile * len(distances)), len(distances) - 1)
    end_idx = min(int(upper_quantile * len(distances)), len(distances) - 1)
    start = distances[start_idx]
    end = distances[end_idx]

    sigmas = np.linspace(start, end, n_values)
    sigmas[sigmas == 0] = 1e-2
    return sigmas


def calculate_sigma_values(
    df: pd.DataFrame,
    n_values: int = 20,
    window_size: int = SIGMA_WINDOW_SIZE,
    normalize: bool = True,
) -> np.ndarray:
    """Calculate distance-based sigma candidates for spectral clustering.

    Args:
        df: Daily station dataframe used to build the sigma-selection windows.
        n_values: Number of sigma values to generate.
        window_size: Number of days per window before distance calculation.
        normalize: Whether to standardize feature columns before windowing.

    Returns:
        Array of sigma candidates for spectral clustering.
    """
    windows, _ = create_windows(df, window_size=window_size, normalize=normalize)
    windows_flat = windows.reshape(windows.shape[0], -1)
    distances = euclidian_distances(windows_flat)
    return sigma_values_from_distance_distribution(distances, n_values=n_values)
