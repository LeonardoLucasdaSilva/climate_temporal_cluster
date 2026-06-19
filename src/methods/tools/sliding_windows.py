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


def determine_n_components(
    df: pd.DataFrame,
    window_size: int = 4,
    columns: List[str] | None = None,
    normalize: bool = True,
    variance_threshold: float = 0.90,
) -> int:
    """Determine optimal number of PCA components based on variance threshold.

    Args:
        df: DataFrame with daily climate data
        window_size: Number of consecutive days per sample
        columns: Columns to include. If None, uses all numeric columns except 'Data'
        normalize: If True, normalize data before PCA
        variance_threshold: Fraction of variance to retain (e.g., 0.90 for 90%). Must be between 0 and 1.

    Returns:
        Optimal number of components needed to retain the specified variance threshold

    Example:
        >>> df = load_station_daily_data(state='SP', station_id='A701', data_root=DATA_ROOT)
        >>> n_comp = determine_n_components(df, window_size=4, variance_threshold=0.90)
        >>> windows, (scaler, pca) = create_windows(df, window_size=4, n_components=n_comp)
    """
    if not 0 < variance_threshold < 1:
        raise ValueError(f"variance_threshold must be between 0 and 1, got {variance_threshold}")

    if columns is None:
        columns = select_numeric_columns(df)

    if not columns:
        raise ValueError("No numeric columns found in dataframe")

    # Extract the data
    data = df[columns].values

    if len(data) < window_size:
        raise ValueError(f"DataFrame has {len(data)} rows but window_size is {window_size}")

    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    # Create sliding windows and flatten
    n_windows = len(data) - window_size + 1
    windows_flat = np.zeros((n_windows, window_size * len(columns)))

    for i in range(n_windows):
        windows_flat[i] = data[i : i + window_size].flatten()

    return determine_pca_components(windows_flat, variance_threshold)


def create_windows(
    df: pd.DataFrame,
    window_size: int = 4,
    columns: List[str] | None = None,
    normalize: bool = True,
    n_components: int | None = None,
    variance_threshold: float | None = None,
) -> Tuple[np.ndarray, Tuple[StandardScaler | None, PCA | None]]:
    """Create sliding windows of consecutive days.

    Args:
        df: DataFrame with daily climate data (must have 'Data' column)
        window_size: Number of consecutive days per sample (e.g., 4)
        columns: Columns to include in windows. If None, uses all except 'Data'.
                 Numeric columns are automatically selected.
        normalize: If True, normalize each window by column statistics.
        n_components: Number of PCA components for dimensionality reduction. If None, no PCA is applied.
                      Must be less than or equal to (window_size * n_features).
                      If both n_components and variance_threshold are None, no PCA is applied.
        variance_threshold: If specified (e.g., 0.90 for 90%), automatically determines n_components
                           to retain this fraction of variance. Overrides n_components if both are specified.
                           Must be between 0 and 1.

    Returns:
        Tuple of (windows, (scaler, pca)):
        - windows: numpy array of shape (n_windows, n_components) if PCA is applied,
                   or (n_windows, window_size, n_features) if PCA is not applied
        - scaler: StandardScaler object (for inverse transform), or None if normalize=False
        - pca: PCA object (for inverse transform), or None if n_components=None

    Example:
        >>> df = load_station_daily_data(state='SP', station_id='A701', data_root=DATA_ROOT)
        >>> # Method 1: Specify exact number of components
        >>> windows, (scaler, pca) = create_windows(
        ...     df,
        ...     window_size=4,
        ...     columns=['TEMPERATURA_MAXIMA', 'TEMPERATURA_MIN', 'PRECIPITACAO_TOTAL'],
        ...     normalize=True,
        ...     n_components=5
        ... )
        >>> # Method 2: Specify variance threshold (automatically determines n_components)
        >>> windows, (scaler, pca) = create_windows(
        ...     df,
        ...     window_size=4,
        ...     columns=['TEMPERATURA_MAXIMA', 'TEMPERATURA_MIN', 'PRECIPITACAO_TOTAL'],
        ...     normalize=True,
        ...     variance_threshold=0.90
        ... )
    """
    if columns is None:
        columns = select_numeric_columns(df)

    if not columns:
        raise ValueError("No numeric columns found in dataframe")

    # Extract the data
    data = df[columns].values  # shape: (n_days, n_features)

    if len(data) < window_size:
        raise ValueError(f"DataFrame has {len(data)} rows but window_size is {window_size}")

    # Normalize the full data if requested
    scaler = None
    if normalize:
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)
    else:
        data_normalized = data

    # Create sliding windows
    n_windows = len(data_normalized) - window_size + 1
    windows = np.zeros((n_windows, window_size, len(columns)))

    for i in range(n_windows):
        windows[i] = data_normalized[i : i + window_size]

    # Apply PCA if n_components or variance_threshold is specified
    pca = None
    if variance_threshold is not None or n_components is not None:
        windows_flat = flatten_windows(windows)
        original_dim = windows_flat.shape[1]

        # If variance_threshold is specified, determine n_components
        if variance_threshold is not None:
            n_components = determine_pca_components(windows_flat, variance_threshold)

        # Fit and apply PCA with determined n_components
        pca = PCA(n_components=n_components)
        windows = pca.fit_transform(windows_flat)

        # Print dimensionality reduction information
        print(f"PCA Dimensionality Reduction:")
        print(f"  Original dimensions: {original_dim}")
        print(f"  New dimensions: {n_components}")
        print(f"  Reduction: {original_dim} → {n_components} ({100 * (1 - n_components / original_dim):.1f}% reduction)")
        if variance_threshold is not None:
            explained_var = pca.explained_variance_ratio_.sum()
            print(f"  Explained variance: {explained_var * 100:.2f}% (threshold: {variance_threshold * 100:.1f}%)")

    return windows, (scaler, pca)


def windows_to_dataframe(
    windows: np.ndarray,
    columns: List[str],
    scaler: StandardScaler | None = None,
    window_size: int | None = None,
) -> pd.DataFrame:
    """Convert windows back to a DataFrame with flattened features.

    Args:
        windows: Array of shape (n_windows, window_size, n_features)
        columns: Original column names
        scaler: Optional scaler for inverse transform (denormalization)
        window_size: Will be inferred from windows shape if not provided

    Returns:
        DataFrame with columns like: col1_day0, col1_day1, ..., col2_day0, col2_day1, ...
    """
    if window_size is None:
        window_size = windows.shape[1]

    n_windows, _, n_features = windows.shape

    # Denormalize if scaler is provided
    if scaler is not None:
        windows_flat = windows.reshape(n_windows * window_size, n_features)
        windows_flat = scaler.inverse_transform(windows_flat)
        windows = windows_flat.reshape(n_windows, window_size, n_features)

    # Flatten windows to dataframe
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
    """Convenience function: create normalized windows and return both.

    Args:
        df: Daily climate data
        window_size: Number of days per window
        columns: Columns to use

    Returns:
        Tuple of (windows, scaler)
    """
    windows, (scaler, _) = create_windows(df, window_size, columns, normalize=True)
    if scaler is None:
        raise RuntimeError("Scaler should not be None when normalize=True")
    return windows, scaler

