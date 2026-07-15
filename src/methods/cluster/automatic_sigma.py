"""Generate automatic spectral-clustering sigma candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_ROOT
from data.load_data import load_station_daily_data
from methods.cluster.cluster_pipeline import create_pipeline_clustering_features
from methods.tools.sigma_choosing import (
    SIGMA_LOWER_QUANTILE,
    SIGMA_UPPER_QUANTILE,
    euclidian_distances,
    sigma_values_from_distance_distribution,
)


STATE = "RS"
STATION_ID = "A801"
WINDOW_SIZE = 15
N_SIGMA_VALUES = 5
NORMALIZE = True
SCALER_TYPE = "standard"
PRECIPITATION_SCALER: str | None = None
TRAIN_RATIO = 0.6
PCA_VARIANCE_THRESHOLD: float | None = None
FEATURE_COLUMNS: list[str] | None = None
LOWER_QUANTILE = SIGMA_LOWER_QUANTILE
UPPER_QUANTILE = SIGMA_UPPER_QUANTILE


def generate_sigma_candidates_from_features(
    features: np.ndarray,
    n_values: int = N_SIGMA_VALUES,
    lower_quantile: float = LOWER_QUANTILE,
    upper_quantile: float = UPPER_QUANTILE,
) -> np.ndarray:
    """Generate sigma candidates from an already-prepared feature matrix."""
    distances = euclidian_distances(features)
    return sigma_values_from_distance_distribution(
        distances,
        n_values=n_values,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )


def generate_sigma_candidates(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    n_values: int = N_SIGMA_VALUES,
    normalize: bool = NORMALIZE,
    columns: list[str] | None = FEATURE_COLUMNS,
    scaler_type: str = SCALER_TYPE,
    precipitation_scaler_type: str | None = PRECIPITATION_SCALER,
    train_ratio: float = TRAIN_RATIO,
    pca_variance_threshold: float | None = PCA_VARIANCE_THRESHOLD,
    lower_quantile: float = LOWER_QUANTILE,
    upper_quantile: float = UPPER_QUANTILE,
) -> np.ndarray:
    """Prepare pipeline-equivalent training features and choose sigmas."""
    features, _ = create_pipeline_clustering_features(
        df,
        window_size=window_size,
        columns=columns,
        normalize=normalize,
        scaler_type=scaler_type,
        precipitation_scaler_type=precipitation_scaler_type,
        train_ratio=train_ratio,
        pca_variance_threshold=pca_variance_threshold,
    )
    return generate_sigma_candidates_from_features(
        features,
        n_values=n_values,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )


def run_automatic_sigma_selection(
    state: str = STATE,
    station_id: str = STATION_ID,
    window_size: int = WINDOW_SIZE,
    n_values: int = N_SIGMA_VALUES,
    normalize: bool = NORMALIZE,
    columns: list[str] | None = FEATURE_COLUMNS,
    scaler_type: str = SCALER_TYPE,
    precipitation_scaler_type: str | None = PRECIPITATION_SCALER,
    train_ratio: float = TRAIN_RATIO,
    pca_variance_threshold: float | None = PCA_VARIANCE_THRESHOLD,
    lower_quantile: float = LOWER_QUANTILE,
    upper_quantile: float = UPPER_QUANTILE,
    data_root: Path = DATA_ROOT,
) -> np.ndarray:
    """Load one station, calculate sigma candidates, and print them."""
    df = load_station_daily_data(
        state=state,
        station_id=station_id,
        data_root=data_root,
    )
    sigmas = generate_sigma_candidates(
        df,
        window_size=window_size,
        n_values=n_values,
        normalize=normalize,
        columns=columns,
        scaler_type=scaler_type,
        precipitation_scaler_type=precipitation_scaler_type,
        train_ratio=train_ratio,
        pca_variance_threshold=pca_variance_threshold,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )

    print(f"Automatic sigma selection for {state}/{station_id}")
    print(f"Window size: {window_size}")
    print(f"Training ratio: {train_ratio:g}")
    print(
        "Normalization: off"
        if not normalize
        else (
            f"Normalization: covariates={scaler_type}, "
            f"precipitation={precipitation_scaler_type or 'none'}"
        )
    )
    print(
        "PCA: off"
        if pca_variance_threshold is None
        else f"PCA variance threshold: {pca_variance_threshold:g}"
    )
    print(
        "Distance quantile range: "
        f"{lower_quantile:.1%} to {upper_quantile:.1%}"
    )
    print(f"Sigma candidates ({len(sigmas)}):")
    for index, sigma in enumerate(sigmas, start=1):
        print(f"  {index}: {sigma:.10g}")
    return sigmas


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate distance-based sigma candidates for one station."
    )
    parser.add_argument("--state", default=STATE)
    parser.add_argument("--station-id", default=STATION_ID)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--n-values", type=int, default=N_SIGMA_VALUES)
    parser.add_argument("--columns", nargs="+", default=FEATURE_COLUMNS)
    parser.add_argument(
        "--scaler-type",
        choices=("standard", "minmax"),
        default=SCALER_TYPE,
    )
    parser.add_argument(
        "--precipitation-scaler",
        choices=("none", "standard", "minmax"),
        default=PRECIPITATION_SCALER,
    )
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument(
        "--pca-variance-threshold",
        type=float,
        default=PCA_VARIANCE_THRESHOLD,
    )
    parser.add_argument("--lower-quantile", type=float, default=LOWER_QUANTILE)
    parser.add_argument("--upper-quantile", type=float, default=UPPER_QUANTILE)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Build windows without the pipeline's configured normalization.",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    return parser.parse_args()


def main() -> None:
    """Run automatic sigma selection from command-line arguments."""
    args = _parse_args()
    run_automatic_sigma_selection(
        state=args.state,
        station_id=args.station_id,
        window_size=args.window_size,
        n_values=args.n_values,
        normalize=not args.no_normalize,
        columns=args.columns,
        scaler_type=args.scaler_type,
        precipitation_scaler_type=args.precipitation_scaler,
        train_ratio=args.train_ratio,
        pca_variance_threshold=args.pca_variance_threshold,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()
