"""Complete clustering pipeline: load → window → cluster."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from climate_cluster.clustering.ng import fit_predict
from climate_cluster.config import DATA_ROOT, OUTPUTS_DIR
from climate_cluster.config_data import load_single_station
from climate_cluster.features.window_features import create_normalized_windows


def run_clustering_pipeline(
    state: str,
    station_id: str,
    window_size: int = 4,
    n_clusters: int = 3,
    sigma: float = 1.0,
    columns: list[str] | None = None,
    data_root: Path = DATA_ROOT,
) -> dict:
    """Run complete spectral clustering pipeline on a single station.

    Args:
        state: State code (e.g., 'SP', 'TO')
        station_id: Station ID (e.g., 'A701')
        window_size: Days per window (default: 4)
        n_clusters: Number of clusters (default: 3)
        sigma: Affinity bandwidth (default: 1.0)
        columns: Columns to use (default: auto-select numeric)
        data_root: Root data path

    Returns:
        Dictionary with results:
        - labels: cluster assignments
        - windows: windowed data (3D)
        - windows_flat: flattened data (2D)
        - scaler: denormalization scaler
    """
    # Load data
    print(f"Loading {state}/{station_id}...")
    df = load_single_station(state=state, station_id=station_id, data_root=data_root)
    print(f"  ✓ {len(df)} days loaded")

    # Create windows
    print(f"Creating windows (size={window_size})...")
    windows, scaler = create_normalized_windows(
        df,
        window_size=window_size,
        columns=columns,
    )
    print(f"  ✓ {len(windows)} windows created")

    # Flatten
    windows_flat = windows.reshape(windows.shape[0], -1)
    print(f"  ✓ Flattened to shape {windows_flat.shape}")

    # Cluster
    print(f"Running spectral clustering (k={n_clusters}, sigma={sigma})...")
    labels = fit_predict(windows_flat, sigma=sigma, k=n_clusters)
    print(f"  ✓ Clustering complete")

    # Summary
    for i in range(n_clusters):
        count = sum(l == i for l in labels)
        print(f"    Cluster {i}: {count} samples")

    return {
        "labels": labels,
        "windows": windows,
        "windows_flat": windows_flat,
        "scaler": scaler,
        "df": df,
    }


def main() -> None:
    """CLI entrypoint for clustering pipeline."""
    parser = argparse.ArgumentParser(description="Spectral clustering pipeline")
    parser.add_argument("--state", default="SP", help="State code")
    parser.add_argument("--station-id", default="A701", help="Station ID")
    parser.add_argument("--window-size", type=int, default=4, help="Days per window")
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters")
    parser.add_argument("--sigma", type=float, default=1.0, help="Affinity bandwidth")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Columns to use",
    )

    args = parser.parse_args()

    results = run_clustering_pipeline(
        state=args.state,
        station_id=args.station_id,
        window_size=args.window_size,
        n_clusters=args.clusters,
        sigma=args.sigma,
        columns=args.columns,
    )

    print("\n" + "=" * 80)
    print("✓ Pipeline completed successfully!")
    print(f"Results returned with {len(results['labels'])} cluster assignments")
    print("=" * 80)


if __name__ == "__main__":
    main()

