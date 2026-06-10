"""Utility functions for climate cluster project."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from climate_cluster.config import DATA_ROOT
from climate_cluster.config_data import load_single_station


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
    df = load_single_station(
        state=state,
        station_id=station_id,
        data_root=data_root,
    )

    # Sort by PRECIPITACAO_TOTAL in descending order and get top n days
    top_days = df.nlargest(n_days, "PRECIPITACAO_TOTAL")

    return top_days


def d_euclidiana(dados: np.ndarray) -> np.ndarray:
    """Calculate pairwise Euclidean distances between data points.

    Args:
        dados: Input data array of shape (n, features)

    Returns:
        Distance matrix D of shape (n, n) where D[i, j] is the Euclidean
        distance between point i and point j
    """
    n = len(dados)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Calculate distance between vector i and vector j
            D[i, j] = np.linalg.norm(dados[i] - dados[j])
    return D


def take_sigma(D: np.ndarray) -> np.ndarray:
    """Determine optimal sigma values for Gaussian kernel based on distance distribution.

    This function analyzes the distance matrix and generates 20 sigma values
    spanning from the 1st to 20th percentile of pairwise distances.

    Args:
        D: Distance matrix of shape (n, n)

    Returns:
        Array of 20 sigma values suitable for Gaussian kernel parameter sweeps
    """
    # Extract upper triangular distances (excluding diagonal)
    dist = D[np.triu_indices(len(D), k=1)].copy()
    dist.sort()

    m = len(dist)
    start = dist[int(0.01 * m)]  # 1st percentile
    end = dist[int(0.2 * m)]      # 20th percentile

    # Generate 20 linearly spaced sigma values
    r = (end - start) / 19
    Sigma = np.array([(start + i * r) for i in range(20)])

    # Ensure no zero values
    for i in range(len(Sigma)):
        if Sigma[i] == 0:
            Sigma[i] = 1e-2

    return Sigma


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

