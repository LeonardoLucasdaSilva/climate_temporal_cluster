"""Precipitation target and distribution helpers.

The LSTM and manual-clustering workflows both start from sliding windows. A
window covers consecutive dataframe rows, while the supervised precipitation
target is observed after the window ends. This module centralizes the indexing
rules so clustering and modeling use the same target alignment.

The default precipitation column is the cleaned INMET daily total rainfall
column, measured in millimeters. Helpers accept a custom column name for tests
or alternative datasets with the same row ordering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_PRECIPITATION_COLUMN = "PRECIPITACAO_TOTAL"


def horizon_precipitation(
    df: pd.DataFrame,
    window_size: int,
    horizon: int = 1,
    precipitation_column: str = DEFAULT_PRECIPITATION_COLUMN,
) -> np.ndarray:
    """Return precipitation at the selected horizon for every sliding window.

    A window beginning at dataframe row ``i`` contains rows ``i`` through
    ``i + window_size - 1``. The target for ``horizon=1`` is the next row after
    the window, ``i + window_size``. More generally, the target row is
    ``i + window_size - 1 + horizon``.

    Args:
        df: Daily station dataframe ordered in time.
        window_size: Number of rows included in each sliding window.
        horizon: Forecast horizon counted after the final row of the window.
            ``1`` means the day immediately after the window.
        precipitation_column: Column containing precipitation totals in
            millimeters.

    Returns:
        A one-dimensional array with one entry per possible sliding window.
        Entries whose target row would fall beyond the dataframe are ``NaN``.
        Non-numeric precipitation values are coerced to ``NaN`` as well.

    Raises:
        ValueError: If window size or horizon is not positive, if the
            precipitation column is missing, or if the dataframe is shorter
            than the requested window size.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if precipitation_column not in df.columns:
        raise ValueError(
            f"Dataframe does not contain precipitation column "
            f"{precipitation_column!r}."
        )
    if len(df) < window_size:
        raise ValueError(
            f"Dataframe has {len(df)} rows but window_size is {window_size}."
        )

    n_windows = len(df) - window_size + 1
    targets = np.full(n_windows, np.nan, dtype=float)
    available = max(n_windows - horizon, 0)
    if available:
        # Only windows with a target row inside the dataframe receive a value.
        target_indices = np.arange(available) + window_size - 1 + horizon
        targets[:available] = pd.to_numeric(
            df.iloc[target_indices][precipitation_column],
            errors="coerce",
        ).to_numpy(dtype=float)
    return targets


def precipitation_targets(
    df: pd.DataFrame,
    window_size: int,
    n_windows: int,
    horizon: int = 1,
    precipitation_column: str = DEFAULT_PRECIPITATION_COLUMN,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid window indices and finite precipitation targets.

    This is the supervised-learning version of :func:`horizon_precipitation`.
    It drops windows whose target is unavailable or missing, returning the
    indices needed to filter the matching feature matrix.

    Args:
        df: Daily station dataframe ordered in time.
        window_size: Number of rows included in each sliding window.
        n_windows: Number of feature windows already created by the caller.
            The raw target vector is sliced to this length so target alignment
            matches the feature matrix exactly.
        horizon: Forecast horizon counted after the final row of the window.
        precipitation_column: Column containing precipitation totals in
            millimeters.

    Returns:
        ``(valid_indices, targets)`` where ``valid_indices`` are integer
        offsets into the caller's window feature matrix and ``targets`` are the
        corresponding finite precipitation values.
    """
    all_targets = horizon_precipitation(
        df,
        window_size=window_size,
        horizon=horizon,
        precipitation_column=precipitation_column,
    )[:n_windows]
    valid_indices = np.flatnonzero(np.isfinite(all_targets))
    return valid_indices, all_targets[valid_indices]


def next_day_precipitation_targets(
    df: pd.DataFrame,
    window_size: int,
    n_windows: int,
    precipitation_column: str = DEFAULT_PRECIPITATION_COLUMN,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite next-day precipitation targets.

    This convenience wrapper is equivalent to calling
    :func:`precipitation_targets` with ``horizon=1``. It is kept for call sites
    and notebooks that use next-day precipitation terminology explicitly.

    Args:
        df: Daily station dataframe ordered in time.
        window_size: Number of rows included in each sliding window.
        n_windows: Number of feature windows already created by the caller.
        precipitation_column: Column containing precipitation totals in
            millimeters.

    Returns:
        ``(valid_indices, targets)`` for the day immediately after each window.
    """
    return precipitation_targets(
        df,
        window_size,
        n_windows,
        horizon=1,
        precipitation_column=precipitation_column,
    )


def precipitation_bin_edges(values: np.ndarray) -> np.ndarray:
    """Return readable shared precipitation bins for histograms.

    The output is intended for comparing precipitation distributions across
    clusters using the same y-axis/bin scale. Non-finite values are ignored.
    Degenerate inputs, such as empty arrays or all-zero precipitation, return
    ``[0.0, 1.0]`` so plotting code still has a valid range.

    For non-degenerate inputs, the Freedman-Diaconis rule gives a data-driven
    starting point. The number of bins is then constrained to a readable range
    of 8 to 35 and the final edges are evenly spaced from zero to the maximum
    finite precipitation value.

    Args:
        values: Precipitation values in millimeters.

    Returns:
        A one-dimensional array of histogram bin edges.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([0.0, 1.0])

    max_value = float(values.max())
    if max_value <= 0:
        return np.array([0.0, 1.0])

    raw_edges = np.histogram_bin_edges(values, bins="fd")
    n_bins = max(8, min(len(raw_edges) - 1, 35))
    return np.linspace(0.0, max_value, n_bins + 1)
