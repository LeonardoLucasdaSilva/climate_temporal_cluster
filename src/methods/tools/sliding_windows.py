"""Create sliding window features from daily station data."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from methods.tools.dimensionality_reduction_tools import (
    determine_pca_components,
    flatten_windows,
    select_numeric_columns,
)


def validate_window_stride(window_stride: int) -> int:
    """Return a positive number of days between consecutive window starts."""
    if isinstance(window_stride, bool) or not isinstance(
        window_stride,
        (int, np.integer),
    ):
        raise ValueError("window_stride must be a positive integer.")
    normalized = int(window_stride)
    if normalized <= 0:
        raise ValueError("window_stride must be a positive integer.")
    return normalized


def create_windows(
    df: pd.DataFrame,
    window_size: int = 4,
    columns: List[str] | None = None,
    normalize: bool = True,
    n_components: int | None = None,
    variance_threshold: float | None = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Tuple[StandardScaler | None, PCA | None]]:
    """Create sliding windows and optionally reduce them with PCA."""
    if columns is None:
        columns = select_numeric_columns(df)

    if not columns:
        raise ValueError("No numeric columns found in dataframe")

    data = df[columns].values

    if len(data) < window_size:
        raise ValueError(f"DataFrame has {len(data)} rows but window_size is {window_size}")

    scaler = None
    if normalize:
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)
    else:
        data_normalized = data

    n_windows = len(data_normalized) - window_size + 1
    windows = np.zeros((n_windows, window_size, len(columns)))

    for i in range(n_windows):
        windows[i] = data_normalized[i : i + window_size]

    pca = None
    if variance_threshold is not None or n_components is not None:
        windows_flat = flatten_windows(windows)
        original_dim = windows_flat.shape[1]

        if variance_threshold is not None:
            n_components = determine_pca_components(windows_flat, variance_threshold)

        pca = PCA(n_components=n_components)
        windows = pca.fit_transform(windows_flat)

        if verbose:
            print("PCA Dimensionality Reduction:")
            print(f"  Original dimensions: {original_dim}")
            print(f"  New dimensions: {n_components}")
            print(
                f"  Reduction: {original_dim} -> {n_components} "
                f"({100 * (1 - n_components / original_dim):.1f}% reduction)"
            )
            if variance_threshold is not None:
                explained_var = pca.explained_variance_ratio_.sum()
                print(
                    f"  Explained variance: {explained_var * 100:.2f}% "
                    f"(threshold: {variance_threshold * 100:.1f}%)"
                )

    return windows, (scaler, pca)


def windows_to_dataframe(
    windows: np.ndarray,
    columns: List[str],
    scaler: StandardScaler | None = None,
    window_size: int | None = None,
) -> pd.DataFrame:
    """Convert windows back to a DataFrame with flattened features."""
    if window_size is None:
        window_size = windows.shape[1]

    n_windows, _, n_features = windows.shape

    if scaler is not None:
        windows_flat = windows.reshape(n_windows * window_size, n_features)
        windows_flat = scaler.inverse_transform(windows_flat)
        windows = windows_flat.reshape(n_windows, window_size, n_features)

    data_dict = {}
    for col_idx, col in enumerate(columns):
        for day_idx in range(window_size):
            col_name = f"{col}_day{day_idx}"
            data_dict[col_name] = windows[:, day_idx, col_idx]

    return pd.DataFrame(data_dict)


def create_normalized_windows(
    df: pd.DataFrame,
    window_size: int = 4,
    columns: List[str] | None = None,
) -> Tuple[np.ndarray, StandardScaler]:
    """Create normalized windows and return the fitted scaler."""
    windows, (scaler, _) = create_windows(df, window_size, columns, normalize=True)
    if scaler is None:
        raise RuntimeError("Scaler should not be None when normalize=True")
    return windows, scaler
