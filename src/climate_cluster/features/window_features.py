"""Create sliding window features from daily station data."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_windows(
    df: pd.DataFrame,
    window_size: int = 4,
    columns: List[str] | None = None,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Create sliding windows of consecutive days.

    Args:
        df: DataFrame with daily climate data (must have 'Data' column)
        window_size: Number of consecutive days per sample (e.g., 4)
        columns: Columns to include in windows. If None, uses all except 'Data'.
                 Numeric columns are automatically selected.
        normalize: If True, normalize each window by column statistics.

    Returns:
        Tuple of (windows, scaler):
        - windows: numpy array of shape (n_windows, window_size, n_features)
        - scaler: StandardScaler object (for inverse transform), or None if normalize=False

    Example:
        >>> df = load_single_station(state='SP', station_id='A701', data_root=DATA_ROOT)
        >>> windows, scaler = create_windows(
        ...     df,
        ...     window_size=4,
        ...     columns=['TEMPERATURA_MAXIMA', 'TEMPERATURA_MIN', 'PRECIPITACAO_TOTAL'],
        ...     normalize=True
        ... )
        >>> print(windows.shape)  # (n_windows, 4, 3)
    """
    # Select numeric columns if not specified
    if columns is None:
        columns = [col for col in df.columns if col != "Data" and pd.api.types.is_numeric_dtype(df[col])]

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

    return windows, scaler


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
    windows, scaler = create_windows(df, window_size, columns, normalize=True)
    if scaler is None:
        raise RuntimeError("Scaler should not be None when normalize=True")
    return windows, scaler

