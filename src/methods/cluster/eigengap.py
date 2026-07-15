"""Analyze spectral eigengaps across multiple sliding-window sizes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigsh

from config import DATA_ROOT, OUTPUTS_DIR
from data.load_data import load_station_daily_data
from methods.cluster.cluster_pipeline import (
    create_cluster_feature_matrix,
    numeric_feature_columns,
)
from methods.cluster.ng import affinity_matrix, normalized_laplacian
from methods.tools.feature_scaling import (
    FeatureScalingState,
    normalize_precipitation_scaler_type,
    scale_weather_features,
)


STATE = "RS"
STATION_ID = "A801"
WINDOW_SIZES = [15, 30, 45]
SIGMA = 1
MAX_GAPS = 20
NORMALIZE = True
SCALER_TYPE = "standard"
PRECIPITATION_SCALER: str | None = None
TRAIN_RATIO = 0.6
PCA_VARIANCE_THRESHOLD: float | None = None
FEATURE_COLUMNS: list[str] | None = None
OUTPUT_DIR = OUTPUTS_DIR / "eigengap"


@dataclass(frozen=True)
class EigengapResult:
    """Eigengap values and the heuristic cluster recommendation."""

    window_size: int
    cluster_counts: np.ndarray
    gaps: np.ndarray
    eigenvalues: np.ndarray
    best_n_clusters: int | None
    best_gap: float | None
    warning: str | None = None


def _degenerate_eigengap_result(
    window_size: int,
    gap_count: int,
    warning: str,
) -> EigengapResult:
    """Return an explicit result for an affinity graph with no usable gaps."""
    return EigengapResult(
        window_size=window_size,
        cluster_counts=np.arange(1, gap_count + 1, dtype=int),
        gaps=np.zeros(gap_count, dtype=float),
        eigenvalues=np.zeros(gap_count + 1, dtype=float),
        best_n_clusters=None,
        best_gap=None,
        warning=warning,
    )


def calculate_eigengaps(
    samples: np.ndarray,
    sigma: float,
    window_size: int,
    max_gaps: int = MAX_GAPS,
) -> EigengapResult:
    """Return the leading normalized-affinity eigengaps for window samples."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(f"samples must be a 2D array, got shape {samples.shape}")
    if len(samples) < 2:
        raise ValueError("At least two samples are required for eigengap analysis.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if max_gaps <= 0:
        raise ValueError("max_gaps must be positive.")

    gap_count = min(max_gaps, len(samples) - 1)
    affinities = affinity_matrix(samples, sigma=sigma)
    if not np.any(affinities):
        return _degenerate_eigengap_result(
            window_size,
            gap_count,
            (
                f"Affinity matrix is degenerate for window size {window_size} "
                f"at sigma={sigma:g}: it has no non-zero edges. Increase sigma."
            ),
        )
    normalized_affinities = normalized_laplacian(affinities, copy=False)

    n_eigenvalues = gap_count + 1
    if n_eigenvalues < len(samples):
        try:
            eigenvalues = np.sort(
                eigsh(
                    normalized_affinities,
                    k=n_eigenvalues,
                    which="LA",
                    return_eigenvectors=False,
                )
            )[::-1]
        except (ArpackError, ArpackNoConvergence) as error:
            return _degenerate_eigengap_result(
                window_size,
                gap_count,
                (
                    f"Affinity matrix is numerically degenerate for window size "
                    f"{window_size} at sigma={sigma:g}; ARPACK could not compute "
                    f"reliable eigenvalues ({error}). Increase sigma."
                ),
            )
    else:
        eigenvalues = eigvalsh(
            normalized_affinities,
            check_finite=False,
        )[::-1]
    gaps = eigenvalues[:-1] - eigenvalues[1:]
    cluster_counts = np.arange(1, gap_count + 1, dtype=int)
    best_offset = int(np.argmax(gaps))

    return EigengapResult(
        window_size=window_size,
        cluster_counts=cluster_counts,
        gaps=gaps,
        eigenvalues=eigenvalues,
        best_n_clusters=int(cluster_counts[best_offset]),
        best_gap=float(gaps[best_offset]),
    )


def create_pipeline_clustering_features(
    df: pd.DataFrame,
    window_size: int,
    columns: list[str] | None,
    normalize: bool,
    scaler_type: str,
    precipitation_scaler_type: str | None,
    train_ratio: float,
    pca_variance_threshold: float | None,
) -> tuple[np.ndarray, list[str]]:
    """Build clustering features with the LSTM pipeline's preprocessing."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    train_end = int(np.floor(len(df) * train_ratio))
    if train_end < window_size:
        raise ValueError(
            f"Training split has {train_end} rows, fewer than window size "
            f"{window_size}."
        )

    feature_columns = columns if columns is not None else numeric_feature_columns(df)
    if not feature_columns:
        raise ValueError("No numeric feature columns are available.")

    precipitation_scaler_type = normalize_precipitation_scaler_type(
        precipitation_scaler_type
    )
    training_df = df.iloc[:train_end].reset_index(drop=True)
    scaled_features, _ = scale_weather_features(
        training_df,
        feature_columns,
        scalers=FeatureScalingState(),
        covariate_scaler_type=scaler_type,
        precipitation_scaler_type=precipitation_scaler_type,
        fit_scalers=normalize,
    )
    _, windows_flat, _, _, _ = create_cluster_feature_matrix(
        scaled_features,
        window_size=window_size,
        columns=feature_columns,
        normalize=False,
        variance_threshold=pca_variance_threshold,
        verbose=False,
    )
    return windows_flat, feature_columns


def plot_eigengaps(result: EigengapResult, output_path: Path) -> Path:
    """Save an eigengap plot with the largest gap highlighted."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = np.full(len(result.gaps), "#4c78a8", dtype=object)
    if result.best_n_clusters is not None:
        colors[result.best_n_clusters - 1] = "#e45756"

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(result.cluster_counts, result.gaps, color=colors, edgecolor="black")
    if result.best_n_clusters is None:
        ax.text(
            0.5,
            0.55,
            "No reliable eigengap\nDegenerate affinity matrix\nIncrease sigma",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#b22222",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.5", "fc": "white", "alpha": 0.95},
        )
    else:
        ax.scatter(
            [result.best_n_clusters],
            [result.best_gap],
            color="#e45756",
            marker="*",
            s=180,
            zorder=3,
            label=(
                f"Largest gap: k={result.best_n_clusters}, "
                f"gap={result.best_gap:.6g}"
            ),
        )
        ax.annotate(
            f"Suggested k = {result.best_n_clusters}\nGap = {result.best_gap:.6g}",
            xy=(result.best_n_clusters, result.best_gap),
            xytext=(12, 16),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#e45756"},
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.9},
        )
    ax.set_xticks(result.cluster_counts)
    ax.set_xlabel(r"Candidate number of clusters $k$")
    ax.set_ylabel(r"Eigengap $\lambda_k - \lambda_{k+1}$")
    ax.set_title(
        f"Spectral Eigengaps - Window Size {result.window_size} "
        f"(first {len(result.gaps)} gaps)"
    )
    ax.grid(axis="y", alpha=0.3)
    if result.best_n_clusters is not None:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_eigengap_analysis(
    state: str,
    station_id: str,
    window_sizes: Sequence[int],
    sigma: float,
    output_dir: Path,
    data_root: Path = DATA_ROOT,
    columns: list[str] | None = None,
    normalize: bool = True,
    max_gaps: int = MAX_GAPS,
    scaler_type: str = SCALER_TYPE,
    precipitation_scaler_type: str | None = PRECIPITATION_SCALER,
    train_ratio: float = TRAIN_RATIO,
    pca_variance_threshold: float | None = PCA_VARIANCE_THRESHOLD,
) -> list[EigengapResult]:
    """Analyze and plot eigengaps for every configured window size."""
    if not window_sizes:
        raise ValueError("window_sizes must contain at least one value.")
    if any(window_size <= 0 for window_size in window_sizes):
        raise ValueError("Every window size must be positive.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_station_daily_data(
        state=state,
        station_id=station_id,
        data_root=data_root,
    )

    print(f"Eigengap analysis for {state}/{station_id}")
    print(f"Sigma: {sigma:g}")
    print(f"Window sizes: {list(window_sizes)}")
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
    print(f"Output directory: {output_dir}")

    results: list[EigengapResult] = []
    for window_size in window_sizes:
        print(f"\nProcessing window size {window_size}...")
        windows_flat, _ = create_pipeline_clustering_features(
            df,
            window_size=window_size,
            columns=columns,
            normalize=normalize,
            scaler_type=scaler_type,
            precipitation_scaler_type=precipitation_scaler_type,
            train_ratio=train_ratio,
            pca_variance_threshold=pca_variance_threshold,
        )
        result = calculate_eigengaps(
            windows_flat,
            sigma=sigma,
            window_size=window_size,
            max_gaps=max_gaps,
        )
        plot_path = plot_eigengaps(
            result,
            output_dir / f"eigengaps_window_{window_size:03d}.png",
        )
        results.append(result)
        if result.warning is not None:
            print(f"  WARNING: {result.warning}")
            print("  Suggested clusters: N/A")
        else:
            print(
                f"  Largest gap: {result.best_gap:.6g} "
                f"at k={result.best_n_clusters}"
            )
        print(f"  Plot: {plot_path}")

    print("\nEigengap summary")
    print("Window size | Suggested clusters | Largest gap")
    print("------------|--------------------|------------")
    for result in results:
        suggested_clusters = (
            "N/A" if result.best_n_clusters is None else str(result.best_n_clusters)
        )
        largest_gap = "N/A" if result.best_gap is None else f"{result.best_gap:.6g}"
        print(
            f"{result.window_size:11d} | "
            f"{suggested_clusters:>18} | {largest_gap}"
        )
    return results


def main() -> None:
    """Run eigengap analysis from command-line or module defaults."""
    parser = argparse.ArgumentParser(
        description="Plot spectral eigengaps for multiple window sizes."
    )
    parser.add_argument("--state", default=STATE, help="State code")
    parser.add_argument("--station-id", default=STATION_ID, help="Station ID")
    parser.add_argument(
        "--window-sizes",
        nargs="+",
        type=int,
        default=WINDOW_SIZES,
        help="Sliding-window sizes to analyze",
    )
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--max-gaps", type=int, default=MAX_GAPS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable feature standardization before window creation",
    )
    args = parser.parse_args()

    run_eigengap_analysis(
        state=args.state,
        station_id=args.station_id,
        window_sizes=args.window_sizes,
        sigma=args.sigma,
        output_dir=args.output_dir,
        columns=args.columns,
        normalize=not args.no_normalize,
        max_gaps=args.max_gaps,
        scaler_type=args.scaler_type,
        precipitation_scaler_type=args.precipitation_scaler,
        train_ratio=args.train_ratio,
        pca_variance_threshold=args.pca_variance_threshold,
    )


if __name__ == "__main__":
    main()
