"""Dynamic Time Warping distances for multivariate weather windows."""

from __future__ import annotations

import numpy as np


SUPPORTED_DISSIMILARITY_METRICS = ("euclidean", "dtw")
_DISSIMILARITY_METRIC_ALIASES = {"dwt": "dtw"}


def normalize_dissimilarity_metric(metric: str) -> str:
    """Return a supported canonical window-dissimilarity metric name."""
    normalized = str(metric).strip().lower()
    normalized = _DISSIMILARITY_METRIC_ALIASES.get(normalized, normalized)
    if normalized in SUPPORTED_DISSIMILARITY_METRICS:
        return normalized

    supported = ", ".join(SUPPORTED_DISSIMILARITY_METRICS)
    raise ValueError(
        f"Unsupported cluster_dissimilarity_metric: {metric!r}. "
        f"Use one of: {supported}"
    )


def _as_window(window: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(window, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional.")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one day and one feature.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _as_window_collection(windows: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(
            f"{name} must have shape (n_windows, window_size, n_features)."
        )
    if values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError(f"{name} must contain at least one day and one feature.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def dtw_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return exact dependent multivariate DTW distance between two windows."""
    first_values = _as_window(first, "first")
    second_values = _as_window(second, "second")
    if first_values.shape[1] != second_values.shape[1]:
        raise ValueError("DTW windows must contain the same number of features.")

    n_second = second_values.shape[0]
    differences = first_values[:, None, :] - second_values[None, :, :]
    local_costs = np.einsum(
        "ijk,ijk->ij",
        differences,
        differences,
        optimize=True,
    )
    previous = np.full(n_second + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0

    for first_index in range(first_values.shape[0]):
        current = np.full(n_second + 1, np.inf, dtype=np.float64)
        for second_index in range(1, n_second + 1):
            current[second_index] = local_costs[
                first_index,
                second_index - 1,
            ] + min(
                previous[second_index],
                current[second_index - 1],
                previous[second_index - 1],
            )
        previous = current

    return float(np.sqrt(previous[-1]))


def pairwise_dtw_distances(windows: np.ndarray) -> np.ndarray:
    """Return the symmetric pairwise DTW distance matrix for window tensors."""
    values = _as_window_collection(windows, "windows")
    n_windows = len(values)
    distances = np.zeros((n_windows, n_windows), dtype=np.float64)
    for first_index in range(n_windows):
        for second_index in range(first_index + 1, n_windows):
            distance = dtw_distance(values[first_index], values[second_index])
            distances[first_index, second_index] = distance
            distances[second_index, first_index] = distance
    return distances


def cross_dtw_distances(
    first_windows: np.ndarray,
    second_windows: np.ndarray,
) -> np.ndarray:
    """Return DTW distances from every first window to every second window."""
    first_values = _as_window_collection(first_windows, "first_windows")
    second_values = _as_window_collection(second_windows, "second_windows")
    if first_values.shape[2] != second_values.shape[2]:
        raise ValueError("DTW window collections must use the same features.")

    distances = np.empty(
        (len(first_values), len(second_values)),
        dtype=np.float64,
    )
    for first_index, first_window in enumerate(first_values):
        for second_index, second_window in enumerate(second_values):
            distances[first_index, second_index] = dtw_distance(
                first_window,
                second_window,
            )
    return distances
