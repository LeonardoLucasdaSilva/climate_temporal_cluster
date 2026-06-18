"""Backward-compatible sliding-window feature API.

New code should import from `climate_cluster.methods.tools.sliding_window_tools`.
This module preserves the older return shape used by the original tests and
examples: `create_windows(...) -> (windows, scaler)`.
"""

from __future__ import annotations

from climate_cluster.methods.tools import sliding_windows as _sliding_windows
from climate_cluster.methods.tools.sliding_windows import (  # noqa: F401
    create_normalized_windows,
    determine_n_components,
)


def create_windows(*args, **kwargs):
    """Create windows and return the historical `(windows, scaler)` tuple."""
    windows, (scaler, _pca) = _sliding_windows.create_windows(*args, **kwargs)
    return windows, scaler


def windows_to_dataframe(windows, columns, scaler=None, window_size=None):
    """Convert windows to a dataframe, accepting either scaler or `(scaler, pca)`."""
    if isinstance(scaler, tuple):
        scaler = scaler[0]
    return _sliding_windows.windows_to_dataframe(
        windows,
        columns=columns,
        scaler=scaler,
        window_size=window_size,
    )
