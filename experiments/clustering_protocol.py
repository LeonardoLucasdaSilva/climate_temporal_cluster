"""Shared clustering protocol for RS A801 experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from climate_cluster.clustering.ng import spectral_clustering
from climate_cluster.features.window_features import create_windows


SUPPORTED_CLUSTERING_ALGORITHMS = ("kmeans", "spectral")
PCA_VARIANCE_THRESHOLD = 0.90
KMEANS_N_INIT = 10
SIGMA_WINDOW_SIZE = 5
SIGMA_LOWER_QUANTILE = 0.01
SIGMA_UPPER_QUANTILE = 0.20


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the numeric weather columns used by clustering experiments."""
    return [
        col
        for col in df.columns
        if col != "Data" and pd.api.types.is_numeric_dtype(df[col])
    ]


def create_cluster_feature_matrix(
    df: pd.DataFrame,
    window_size: int,
    columns: list[str] | None = None,
    normalize: bool = True,
    variance_threshold: float = PCA_VARIANCE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, object, object, list[str]]:
    """Create the exact window feature matrix used for clustering."""
    feature_columns = columns if columns is not None else numeric_feature_columns(df)
    windows, (scaler, pca) = create_windows(
        df,
        window_size=window_size,
        columns=feature_columns,
        normalize=normalize,
        variance_threshold=variance_threshold,
    )

    if windows.ndim == 3:
        windows_flat = windows.reshape(windows.shape[0], -1)
    else:
        windows_flat = windows

    return windows, windows_flat, scaler, pca, feature_columns


def pairwise_euclidean_distances(feature_matrix: np.ndarray) -> np.ndarray:
    """Return the full pairwise Euclidean distance matrix."""
    n_samples = len(feature_matrix)
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            distances[i, j] = np.linalg.norm(feature_matrix[i] - feature_matrix[j])
    return distances


def sigma_values_from_distance_distribution(
    distance_matrix: np.ndarray,
    n_values: int = 20,
    lower_quantile: float = SIGMA_LOWER_QUANTILE,
    upper_quantile: float = SIGMA_UPPER_QUANTILE,
) -> np.ndarray:
    """Choose sigma values from the lower tail of pairwise distances.

    This preserves the original experiment logic: use the 1st to 20th
    percentiles of upper-triangular pairwise distances, then linearly space
    candidate sigma values across that interval.
    """
    if n_values <= 0:
        raise ValueError(f"n_values must be positive, got {n_values}")

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
    """Calculate distance-based sigma candidates for spectral clustering."""
    windows, _ = create_windows(df, window_size=window_size, normalize=normalize)
    windows_flat = windows.reshape(windows.shape[0], -1)
    distances = pairwise_euclidean_distances(windows_flat)
    return sigma_values_from_distance_distribution(distances, n_values=n_values)


def cluster_feature_matrix(
    feature_matrix: np.ndarray,
    n_clusters: int,
    algorithm: str = "kmeans",
    sigma: float | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster windows using one of the supported algorithms."""
    algorithm = algorithm.lower()

    if algorithm == "kmeans":
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=KMEANS_N_INIT,
        )
        return kmeans.fit_predict(feature_matrix)

    if algorithm == "spectral":
        if sigma is None:
            raise ValueError("sigma must be provided when algorithm='spectral'")
        return spectral_clustering(
            feature_matrix,
            sigma=sigma,
            k=n_clusters,
            random_state=random_state,
        )

    supported = ", ".join(SUPPORTED_CLUSTERING_ALGORITHMS)
    raise ValueError(f"Unsupported clustering algorithm: {algorithm!r}. Use one of: {supported}")
