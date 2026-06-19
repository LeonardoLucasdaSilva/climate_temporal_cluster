"""Cluster feature preparation and algorithm dispatch helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from climate_cluster.config import DATA_ROOT
from climate_cluster.config_data import load_single_station
from climate_cluster.methods.cluster.ng import spectral_clustering
from climate_cluster.methods.tools.sliding_windows import create_windows


SUPPORTED_CLUSTERING_ALGORITHMS = ("kmeans", "spectral")
PCA_VARIANCE_THRESHOLD = 0.90
KMEANS_N_INIT = 10


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric weather columns used by clustering experiments.

    Args:
        df: Daily station dataframe. The date column is expected to be named
            `Data` and is excluded from the result.

    Returns:
        List of numeric feature column names.
    """
    return [
        col
        for col in df.columns
        if col != "Data" and pd.api.types.is_numeric_dtype(df[col])
    ]


def create_cluster_feature_matrix(
    df: pd.DataFrame,
    window_size: int,
    columns: list[str] | None = None,
    normalize: bool = True,
    variance_threshold: float | None = PCA_VARIANCE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, object, object, list[str]]:
    """Create window features in the format expected by clustering algorithms.

    Args:
        df: Daily station dataframe.
        window_size: Number of consecutive days per sample.
        columns: Feature columns to include. If omitted, all numeric columns
            except `Data` are used.
        normalize: Whether to standardize feature columns before windowing.
        variance_threshold: PCA explained-variance threshold passed to
            `create_windows`.

    Returns:
        Tuple containing:
        - windows: original output from `create_windows`;
        - windows_flat: 2D matrix with one row per window;
        - scaler: fitted scaler when normalization is enabled, else `None`;
        - pca: fitted PCA when dimensionality reduction is enabled, else `None`;
        - feature_columns: column names used to build the windows.
    """
    feature_columns = columns if columns is not None else numeric_feature_columns(df)
    windows, (scaler, pca) = create_windows(
        df,
        window_size=window_size,
        columns=feature_columns,
        normalize=normalize,
        variance_threshold=variance_threshold,
    )

    if windows.ndim == 3:
        windows_flat = windows.reshape(windows.shape[0], -1)
    else:
        windows_flat = windows

    return windows, windows_flat, scaler, pca, feature_columns


def cluster_feature_matrix(
    feature_matrix: np.ndarray,
    n_clusters: int,
    algorithm: str = "kmeans",
    sigma: float | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster a feature matrix with K-means or spectral clustering.

    Args:
        feature_matrix: 2D matrix with shape `(n_samples, n_features)`.
        n_clusters: Number of clusters to produce.
        algorithm: Clustering algorithm name. Supported values are `kmeans`
            and `spectral`.
        sigma: Gaussian-kernel bandwidth for spectral clustering. Required
            when `algorithm="spectral"`.
        random_state: Random seed used by K-means and spectral clustering.

    Returns:
        One-dimensional array of cluster labels with length `n_samples`.
    """
    algorithm = algorithm.lower()

    if algorithm == "kmeans":
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=KMEANS_N_INIT,
        )
        return kmeans.fit_predict(feature_matrix)

    if algorithm == "spectral":
        if sigma is None:
            raise ValueError("sigma must be provided when algorithm='spectral'")
        return spectral_clustering(
            feature_matrix,
            sigma=sigma,
            k=n_clusters,
            random_state=random_state,
        )

    supported = ", ".join(SUPPORTED_CLUSTERING_ALGORITHMS)
    raise ValueError(f"Unsupported clustering algorithm: {algorithm!r}. Use one of: {supported}")


def run_clustering_pipeline(
    state: str,
    station_id: str,
    window_size: int = 4,
    n_clusters: int = 3,
    sigma: float = 1.0,
    columns: list[str] | None = None,
    data_root: Path = DATA_ROOT,
) -> dict:
    """Load one station, create windows, and run spectral clustering.

    Args:
        state: Brazilian state code, such as `SP` or `RS`.
        station_id: INMET station id, such as `A701` or `A801`.
        window_size: Number of consecutive days per window.
        n_clusters: Number of clusters to produce.
        sigma: Gaussian-kernel bandwidth for spectral clustering.
        columns: Optional feature columns. If omitted, numeric columns are
            selected automatically.
        data_root: Root path containing station data.

    Returns:
        Dictionary containing the loaded dataframe, window tensors, flattened
        feature matrix, fitted scaler/PCA objects, feature column names, and
        cluster labels.
    """
    print(f"Loading {state}/{station_id}...")
    df = load_single_station(state=state, station_id=station_id, data_root=data_root)
    print(f"  Loaded {len(df)} days")

    print(f"Creating windows (size={window_size})...")
    windows, windows_flat, scaler, pca, feature_columns = create_cluster_feature_matrix(
        df,
        window_size=window_size,
        columns=columns,
        normalize=True,
        variance_threshold=None,
    )
    print(f"  Created {len(windows)} windows")
    print(f"  Flattened to shape {windows_flat.shape}")

    print(f"Running spectral clustering (k={n_clusters}, sigma={sigma})...")
    labels = cluster_feature_matrix(
        windows_flat,
        n_clusters=n_clusters,
        algorithm="spectral",
        sigma=sigma,
    )
    print("  Clustering complete")

    for cluster_id in range(n_clusters):
        count = int(np.sum(labels == cluster_id))
        print(f"    Cluster {cluster_id}: {count} samples")

    return {
        "df": df,
        "labels": labels,
        "windows": windows,
        "windows_flat": windows_flat,
        "scaler": scaler,
        "pca": pca,
        "feature_columns": feature_columns,
    }


def main() -> None:
    """CLI entrypoint for the spectral clustering pipeline."""
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
    print("Pipeline completed successfully.")
    print(f"Results returned with {len(results['labels'])} cluster assignments")
    print("=" * 80)
