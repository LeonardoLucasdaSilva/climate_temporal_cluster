"""Dimensionality reduction helpers used by feature pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def select_numeric_columns(
    df: pd.DataFrame,
    exclude: tuple[str, ...] = ("Data",),
) -> list[str]:
    """Return numeric dataframe columns excluding metadata columns."""
    return [
        col
        for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    """Flatten a 3D window tensor to a 2D feature matrix."""
    if windows.ndim == 2:
        return windows
    if windows.ndim != 3:
        raise ValueError(f"windows must be 2D or 3D, got shape {windows.shape}")
    return windows.reshape(windows.shape[0], -1)


def determine_pca_components(
    features: np.ndarray,
    variance_threshold: float = 0.90,
) -> int:
    """Return the smallest PCA component count that reaches the threshold."""
    if not 0 < variance_threshold < 1:
        raise ValueError(
            f"variance_threshold must be between 0 and 1, got {variance_threshold}"
        )

    max_components = min(features.shape)
    pca_full = PCA(n_components=max_components)
    pca_full.fit(features)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    return int(np.argmax(cumulative_variance >= variance_threshold) + 1)


def determine_n_components(
    df: pd.DataFrame,
    window_size: int = 4,
    columns: list[str] | None = None,
    normalize: bool = True,
    variance_threshold: float = 0.90,
) -> int:
    """Determine the PCA component count for flattened sliding windows."""
    if columns is None:
        columns = select_numeric_columns(df)

    if not columns:
        raise ValueError("No numeric columns found in dataframe")

    data = df[columns].values

    if len(data) < window_size:
        raise ValueError(f"DataFrame has {len(data)} rows but window_size is {window_size}")

    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    n_windows = len(data) - window_size + 1
    windows_flat = np.zeros((n_windows, window_size * len(columns)))

    for i in range(n_windows):
        windows_flat[i] = data[i : i + window_size].flatten()

    return determine_pca_components(windows_flat, variance_threshold)


def fit_pca_by_variance(
    features: np.ndarray,
    variance_threshold: float = 0.90,
) -> tuple[np.ndarray, PCA]:
    """Fit PCA using enough components to retain the requested variance."""
    n_components = determine_pca_components(features, variance_threshold)
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features), pca

