"""Analyze spectral eigengaps across multiple sliding-window sizes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvalsh
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigsh

from config import DATA_ROOT, OUTPUTS_DIR
from data.load_data import load_station_daily_data
from methods.cluster.cluster_pipeline import (
    create_pipeline_clustering_features,
)
from methods.cluster.ng import affinity_matrix, normalized_laplacian


STATE = "RS"
STATION_ID = "A801"
WINDOW_SIZES = [15, 30, 45]
MAX_GAPS = 20
NORMALIZE = True
SCALER_TYPE = "standard"
PRECIPITATION_SCALER: str | None = None
TRAIN_RATIO = 0.6
PCA_VARIANCE_THRESHOLD: float | None = None
FEATURE_COLUMNS: list[str] | None = None
SIGMA_BOUNDS = (1e-6, 100.0)
MAX_SIGMA_EVALUATIONS = 300
SIGMA_SCOUT_POINTS = 40
EIGEN_MAX_ITERATIONS = 300
EIGEN_TOLERANCE = 1e-3
OUTPUT_DIR = OUTPUTS_DIR / "eigengap"


@dataclass(frozen=True)
class EigengapResult:
    """Eigengap values and the heuristic cluster recommendation."""

    window_size: int
    sigma: float
    cluster_counts: np.ndarray
    gaps: np.ndarray
    eigenvalues: np.ndarray
    best_n_clusters: int | None
    best_gap: float | None
    warning: str | None = None


def combine_sigma_values(
    automatic_values: Sequence[float],
    additional_values: Sequence[float] | None = None,
) -> np.ndarray:
    """Return valid automatic and manual sigma values without duplicates."""
    automatic = np.asarray(automatic_values, dtype=float)
    additional = np.asarray(
        [] if additional_values is None else additional_values,
        dtype=float,
    )
    if automatic.ndim != 1 or additional.ndim != 1:
        raise ValueError("Sigma values must be one-dimensional sequences.")

    combined: list[float] = []
    for value in np.concatenate([automatic, additional]):
        sigma = float(value)
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(
                f"Every sigma value must be positive and finite, got {sigma}."
            )
        if not any(
            np.isclose(sigma, existing, rtol=1e-12, atol=0.0)
            for existing in combined
        ):
            combined.append(sigma)
    return np.asarray(combined, dtype=float)


def _degenerate_eigengap_result(
    window_size: int,
    sigma: float,
    gap_count: int,
    warning: str,
) -> EigengapResult:
    """Return an explicit result for an affinity graph with no usable gaps."""
    return EigengapResult(
        window_size=window_size,
        sigma=sigma,
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
    eigen_max_iterations: int = EIGEN_MAX_ITERATIONS,
    eigen_tolerance: float = EIGEN_TOLERANCE,
) -> EigengapResult:
    """Return eigengaps, ignoring the uninformative one-cluster gap."""
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
    if eigen_max_iterations <= 0:
        raise ValueError("eigen_max_iterations must be positive.")

    gap_count = min(max_gaps, len(samples) - 1)
    affinities = affinity_matrix(samples, sigma=sigma)
    if not np.any(affinities):
        return _degenerate_eigengap_result(
            window_size,
            sigma,
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
                    maxiter=eigen_max_iterations,
                    tol=eigen_tolerance,
                )
            )[::-1]
        except (ArpackError, ArpackNoConvergence) as error:
            return _degenerate_eigengap_result(
                window_size,
                sigma,
                gap_count,
                (
                    f"Affinity matrix is numerically degenerate for window size "
                    f"{window_size} at sigma={sigma:g}; ARPACK could not compute "
                    f"reliable eigenvalues within {eigen_max_iterations} "
                    f"iterations ({error}). This sigma trial was skipped."
                ),
            )
    else:
        eigenvalues = eigvalsh(
            normalized_affinities,
            check_finite=False,
        )[::-1]
    gaps = eigenvalues[:-1] - eigenvalues[1:]
    cluster_counts = np.arange(1, gap_count + 1, dtype=int)
    # k=1 only separates a single cluster from the rest and is not useful for
    # choosing a clustering. Start the maximization at k=2.
    usable_offsets = np.flatnonzero(cluster_counts >= 2)
    if not len(usable_offsets):
        return EigengapResult(
            window_size=window_size,
            sigma=sigma,
            cluster_counts=cluster_counts,
            gaps=gaps,
            eigenvalues=eigenvalues,
            best_n_clusters=None,
            best_gap=None,
            warning="At least two eigengaps are required to ignore the k=1 gap.",
        )
    best_offset = int(usable_offsets[np.argmax(gaps[usable_offsets])])

    return EigengapResult(
        window_size=window_size,
        sigma=sigma,
        cluster_counts=cluster_counts,
        gaps=gaps,
        eigenvalues=eigenvalues,
        best_n_clusters=int(cluster_counts[best_offset]),
        best_gap=float(gaps[best_offset]),
    )


def optimize_sigma_for_eigengap(
    samples: np.ndarray,
    window_size: int,
    sigma_bounds: tuple[float, float] = SIGMA_BOUNDS,
    max_sigma_evaluations: int = MAX_SIGMA_EVALUATIONS,
    sigma_scout_points: int = SIGMA_SCOUT_POINTS,
    max_gaps: int = MAX_GAPS,
    eigen_max_iterations: int = EIGEN_MAX_ITERATIONS,
    eigen_tolerance: float = EIGEN_TOLERANCE,
) -> tuple[EigengapResult, list[EigengapResult]]:
    """Numerically find the sigma with the largest usable eigengap."""
    lower, upper = map(float, sigma_bounds)
    if not (np.isfinite(lower) and np.isfinite(upper) and 0 < lower < upper):
        raise ValueError("sigma_bounds must be finite, positive, and increasing.")
    if max_sigma_evaluations < 3:
        raise ValueError("max_sigma_evaluations must be at least 3.")
    if sigma_scout_points < 3:
        raise ValueError("sigma_scout_points must be at least 3.")

    evaluations: list[EigengapResult] = []
    cache: dict[float, EigengapResult] = {}

    def evaluate(sigma: float) -> EigengapResult:
        key = float(sigma)
        if key not in cache:
            cache[key] = calculate_eigengaps(
                samples,
                sigma=key,
                window_size=window_size,
                max_gaps=max_gaps,
                eigen_max_iterations=eigen_max_iterations,
                eigen_tolerance=eigen_tolerance,
            )
            evaluations.append(cache[key])
            result = cache[key]
            status = (
                f"gap={result.best_gap:.6g} at k={result.best_n_clusters}"
                if result.best_gap is not None
                else f"skipped ({result.warning})"
            )
            print(f"    sigma={key:.8g}: {status}")
        return cache[key]

    def objective(sigma: float) -> float:
        result = evaluate(sigma)
        return np.inf if result.best_gap is None else -result.best_gap

    # Scout the full interval before continuous local refinement. This is more
    # robust than assuming the objective is unimodal across the whole range.
    scout_count = min(sigma_scout_points, max_sigma_evaluations)
    scout_sigmas = np.linspace(lower, upper, num=scout_count)
    scout_values = np.asarray([objective(sigma) for sigma in scout_sigmas])
    best_scout = int(np.argmin(scout_values))
    remaining_evaluations = max_sigma_evaluations - len(evaluations)
    if remaining_evaluations >= 3:
        local_lower = float(scout_sigmas[max(0, best_scout - 1)])
        local_upper = float(scout_sigmas[min(scout_count - 1, best_scout + 1)])
        minimize_scalar(
            objective,
            bounds=(local_lower, local_upper),
            method="bounded",
            options={
                "maxiter": remaining_evaluations,
                "xatol": max((upper - lower) * 1e-6, np.finfo(float).eps),
            },
        )

    valid = [result for result in evaluations if result.best_gap is not None]
    if not valid:
        best = evaluations[0]
    else:
        best = max(valid, key=lambda result: float(result.best_gap))
    return best, evaluations


def plot_eigengaps(result: EigengapResult, output_path: Path) -> Path:
    """Save an eigengap plot with the largest gap highlighted."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = np.full(len(result.gaps), "#4c78a8", dtype=object)
    if len(colors):
        colors[0] = "#b8b8b8"
    if result.best_n_clusters is not None:
        colors[result.best_n_clusters - 1] = "#e45756"

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(result.cluster_counts, result.gaps, color=colors, edgecolor="black")
    if len(result.gaps):
        ax.annotate(
            "ignored",
            xy=(1, result.gaps[0]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            color="#666666",
        )
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
        f"- Sigma {result.sigma:.6g} (first {len(result.gaps)} gaps)"
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
    output_dir: Path,
    data_root: Path = DATA_ROOT,
    columns: list[str] | None = None,
    normalize: bool = True,
    max_gaps: int = MAX_GAPS,
    scaler_type: str = SCALER_TYPE,
    precipitation_scaler_type: str | None = PRECIPITATION_SCALER,
    train_ratio: float = TRAIN_RATIO,
    pca_variance_threshold: float | None = PCA_VARIANCE_THRESHOLD,
    sigma_bounds: tuple[float, float] = SIGMA_BOUNDS,
    max_sigma_evaluations: int = MAX_SIGMA_EVALUATIONS,
    sigma_scout_points: int = SIGMA_SCOUT_POINTS,
    eigen_max_iterations: int = EIGEN_MAX_ITERATIONS,
    eigen_tolerance: float = EIGEN_TOLERANCE,
) -> list[EigengapResult]:
    """Optimize sigma independently for every window size."""
    if not window_sizes:
        raise ValueError("window_sizes must contain at least one value.")
    if any(window_size <= 0 for window_size in window_sizes):
        raise ValueError("Every window size must be positive.")
    if max_sigma_evaluations < 3:
        raise ValueError("max_sigma_evaluations must be at least 3.")
    if sigma_scout_points < 3:
        raise ValueError("sigma_scout_points must be at least 3.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_station_daily_data(
        state=state,
        station_id=station_id,
        data_root=data_root,
    )

    print(f"Eigengap analysis for {state}/{station_id}")
    print(f"Window sizes: {list(window_sizes)}")
    print(f"Sigma search interval: [{sigma_bounds[0]:g}, {sigma_bounds[1]:g}]")
    print(f"Maximum sigma evaluations per window: {max_sigma_evaluations}")
    print(f"Starting sigma scout points: {sigma_scout_points}")
    print(
        f"Eigenvalue solver cap: {eigen_max_iterations} iterations "
        f"(tolerance={eigen_tolerance:g})"
    )
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
        print("  Optimizing sigma (the k=1 gap is ignored):")
        best_result, evaluations = optimize_sigma_for_eigengap(
            windows_flat,
            window_size=window_size,
            sigma_bounds=sigma_bounds,
            max_sigma_evaluations=max_sigma_evaluations,
            sigma_scout_points=sigma_scout_points,
            max_gaps=max_gaps,
            eigen_max_iterations=eigen_max_iterations,
            eigen_tolerance=eigen_tolerance,
        )
        plot_path = plot_eigengaps(
            best_result,
            output_dir / f"eigengaps_window_{window_size:03d}_optimal.png",
        )
        results.append(best_result)
        skipped = sum(result.warning is not None for result in evaluations)
        if best_result.best_gap is None:
            print(
                f"  No usable sigma found; evaluations={len(evaluations)}; "
                f"skipped={skipped}"
            )
        else:
            print(
                f"  Best sigma={best_result.sigma:.8g}; "
                f"gap={best_result.best_gap:.6g}; "
                f"suggested k={best_result.best_n_clusters}; "
                f"evaluations={len(evaluations)}; skipped={skipped}"
            )
        print(f"  Plot: {plot_path}")

    print("\nEigengap summary")
    print("Window size | Optimal sigma | Suggested clusters | Max gap (k >= 2)")
    print("------------|---------------|--------------------|-----------------")
    for result in results:
        suggested_clusters = (
            "N/A" if result.best_n_clusters is None else str(result.best_n_clusters)
        )
        largest_gap = "N/A" if result.best_gap is None else f"{result.best_gap:.6g}"
        print(
            f"{result.window_size:11d} | "
            f"{result.sigma:12.6g} | "
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
    parser.add_argument(
        "--sigma-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=SIGMA_BOUNDS,
        help="Positive interval used by the bounded sigma optimizer",
    )
    parser.add_argument(
        "--max-sigma-evaluations",
        type=int,
        default=MAX_SIGMA_EVALUATIONS,
        help=(
            "Maximum number of sigma trials per window "
            f"(default: {MAX_SIGMA_EVALUATIONS})"
        ),
    )
    parser.add_argument(
        "--sigma-scout-points",
        type=int,
        default=SIGMA_SCOUT_POINTS,
        help=(
            "Evenly spaced starting sigma values "
            f"(default: {SIGMA_SCOUT_POINTS})"
        ),
    )
    parser.add_argument(
        "--eigen-max-iterations",
        type=int,
        default=EIGEN_MAX_ITERATIONS,
        help="Skip an ARPACK sigma trial after this many iterations",
    )
    parser.add_argument(
        "--eigen-tolerance",
        type=float,
        default=EIGEN_TOLERANCE,
        help="ARPACK convergence tolerance",
    )
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
        output_dir=args.output_dir,
        columns=args.columns,
        normalize=not args.no_normalize,
        max_gaps=args.max_gaps,
        scaler_type=args.scaler_type,
        precipitation_scaler_type=args.precipitation_scaler,
        train_ratio=args.train_ratio,
        pca_variance_threshold=args.pca_variance_threshold,
        sigma_bounds=tuple(args.sigma_bounds),
        max_sigma_evaluations=args.max_sigma_evaluations,
        sigma_scout_points=args.sigma_scout_points,
        eigen_max_iterations=args.eigen_max_iterations,
        eigen_tolerance=args.eigen_tolerance,
    )


if __name__ == "__main__":
    main()
