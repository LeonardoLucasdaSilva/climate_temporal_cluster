"""Utility functions for climate cluster project."""

import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import DATA_ROOT
from data.load_data import load_station_daily_data


def get_highest_precipitation_days(
    state: str = "RS",
    station_id: str = "A801",
    n_days: int = 10,
    data_root: Path = DATA_ROOT,
) -> pd.DataFrame:
    """Load station data and return the n days with highest precipitation.

    Args:
        state: State code (default: 'RS')
        station_id: Station ID (default: 'A801')
        n_days: Number of top precipitation days to return (default: 5)
        data_root: Root path to INMET data (default: DATA_ROOT from config)

    Returns:
        DataFrame with the top n days sorted by PRECIPITACAO_TOTAL in descending order.
        Includes all columns from the station data.
    """
    # Load the station data
    df = load_station_daily_data(
        state=state,
        station_id=station_id,
        data_root=data_root,
    )

    # Sort by PRECIPITACAO_TOTAL in descending order and get top n days
    top_days = df.nlargest(n_days, "PRECIPITACAO_TOTAL")

    return top_days


if __name__ == "__main__":
    # Load and display the 5 days with highest precipitation for RS A801
    print(f"Loading RS A801 station data...")
    result = get_highest_precipitation_days()

    print(f"\nTop 5 days with highest PRECIPITACAO_TOTAL at RS/A801:")
    print("=" * 80)
    print(result)

    print(f"\nSummary:")
    print(f"  Total days: {len(result)}")
    print(f"  Max precipitation: {result['PRECIPITACAO_TOTAL'].max():.2f} mm")
    print(f"  Min precipitation: {result['PRECIPITACAO_TOTAL'].min():.2f} mm")
    print(f"  Average precipitation: {result['PRECIPITACAO_TOTAL'].mean():.2f} mm")

