"""Configuration and single-station data loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from climate_cluster.data.load_inmet import load_station_daily_data


def load_single_station(
    state: str,
    station_id: str,
    data_root: Path,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load a single INMET station's daily data.

    Adapted from the original `config()` function to work with one station
    and daily grouping instead of weekly.

    Args:
        state: State code (e.g., 'SP', 'TO')
        station_id: Station ID (e.g., 'A701')
        data_root: INMET data root path
        cols: Columns to load (None = all)

    Returns:
        DataFrame with daily aggregated data.
    """
    return load_station_daily_data(
        state=state,
        station_id=station_id,
        data_root=data_root,
        cols=cols,
    )


