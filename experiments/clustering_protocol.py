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
