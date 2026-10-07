"""Spectral clustering algorithm for climate data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.cluster import KMeans

try:
    from scipy.linalg import eigh as scipy_eigh
except ImportError:  # pragma: no cover - scipy is installed with scikit-learn.
    scipy_eigh = None


DEFAULT_AFFINITY_CHUNK_SIZE = 1024
EIGENVALUE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SpectralTrainingGraph:
    """Reusable normalized training graph and its leading eigensystem."""

    sigma: float
    samples: np.ndarray
    training_degrees: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    @property
    def max_k(self) -> int:
        """Return the largest K supported by this cached eigensystem."""
        return int(self.eigenvectors.shape[1])


def euclidean_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors.

    Args:
        first: First vector.
        second: Second vector.

    Returns:
        Euclidean distance (L2 norm of difference)
    """
    return np.linalg.norm(first - second)


def affinity_matrix(
    samples: np.ndarray,
    sigma: float,
    chunk_size: int = DEFAULT_AFFINITY_CHUNK_SIZE,
) -> np.ndarray:
    """Create affinity matrix using Gaussian kernel.

    Args:
        samples: Data matrix of shape (n_samples, n_features).
        sigma: Bandwidth parameter for Gaussian kernel
        chunk_size: Number of rows processed at once. Larger chunks can be
            faster but use more temporary memory.

    Returns:
        Affinity matrix A of shape (n_samples, n_samples)
        A[i,j] = exp(-||s_i - s_j||^2 / (2*sigma^2)) for i != j
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    samples = np.asarray(samples, dtype=np.float64)
    n_samples = samples.shape[0]
    squared_norms = np.einsum("ij,ij->i", samples, samples)
    gamma = 1.0 / (2.0 * sigma**2)
    affinities = np.empty((n_samples, n_samples), dtype=np.float64)

    # Compute pairwise squared distances in row chunks:
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        distances_sq = (
            squared_norms[start:stop, None]
            + squared_norms[None, :]
            - 2.0 * samples[start:stop] @ samples.T
        )
        np.maximum(distances_sq, 0.0, out=distances_sq)
        affinities[start:stop] = np.exp(-gamma * distances_sq)

    np.fill_diagonal(affinities, 0.0)
    return affinities


def affinity_matrix_from_distances(
    distance_matrix: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Create a Gaussian affinity matrix from precomputed distances."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be square.")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values.")
    if np.any(distances < 0):
        raise ValueError("distance_matrix cannot contain negative values.")
    if not np.allclose(distances, distances.T):
        raise ValueError("distance_matrix must be symmetric.")

    gamma = 1.0 / (2.0 * sigma**2)
    affinities = np.exp(-gamma * distances**2)
    np.fill_diagonal(affinities, 0.0)
    return affinities


def cross_affinity_matrix(
    samples: np.ndarray,
    reference_samples: np.ndarray,
    sigma: float,
    chunk_size: int = DEFAULT_AFFINITY_CHUNK_SIZE,
) -> np.ndarray:
    """Create Gaussian affinities from samples to fitted reference samples."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    samples = np.asarray(samples, dtype=np.float64)
    references = np.asarray(reference_samples, dtype=np.float64)
    if samples.ndim != 2 or references.ndim != 2:
        raise ValueError("samples and reference_samples must be 2D arrays.")
    if samples.shape[1] != references.shape[1]:
        raise ValueError("samples and reference_samples must have matching features.")

    affinities = np.empty((len(samples), len(references)), dtype=np.float64)
    sample_norms = np.einsum("ij,ij->i", samples, samples)
    reference_norms = np.einsum("ij,ij->i", references, references)
    gamma = 1.0 / (2.0 * sigma**2)
    for start in range(0, len(samples), chunk_size):
        stop = min(start + chunk_size, len(samples))
        distances_sq = (
            sample_norms[start:stop, None]
            + reference_norms[None, :]
            - 2.0 * samples[start:stop] @ references.T
        )
        np.maximum(distances_sq, 0.0, out=distances_sq)
        affinities[start:stop] = np.exp(-gamma * distances_sq)
    return affinities


def cross_affinity_matrix_from_distances(
    distance_matrix: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Create Gaussian cross-affinities from held-out-to-training distances."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2:
        raise ValueError("distance_matrix must be 2D.")
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("distance_matrix must contain finite non-negative values.")
    return np.exp(-(distances**2) / (2.0 * sigma**2))


def normalized_laplacian(
    affinities: np.ndarray,
    copy: bool = True,
) -> np.ndarray:
    """Create normalized Laplacian matrix.

    Args:
        affinities: Affinity matrix of shape (n_samples, n_samples).
        copy: Whether to preserve the input matrix. Disable this for large
            one-use affinity matrices to reduce peak memory usage.

    Returns:
        Normalized Laplacian L = D^(-1/2) * A * D^(-1/2)
        where D is the degree matrix (diagonal of row sums)
    """
    # Compute degree (row sums)
    degrees = np.sum(affinities, axis=1)

    # Create inverse sqrt of degree for normalization
    diag = np.zeros_like(degrees)
    mask = degrees > 0
    diag[mask] = degrees[mask] ** (-0.5)

    # Normalized Laplacian: L = D^(-1/2) * A * D^(-1/2)
    if copy:
        laplacian = np.asarray(affinities, dtype=np.float64).copy()
    elif np.issubdtype(affinities.dtype, np.floating):
        laplacian = affinities
    else:
        laplacian = affinities.astype(np.float64)
    laplacian *= diag[:, None]
    laplacian *= diag[None, :]

    return laplacian


def eigenvector_embedding(laplacian: np.ndarray, k: int) -> np.ndarray:
    """Extract k eigenvectors from normalized Laplacian.

    Args:
        laplacian: Normalized Laplacian matrix of shape (n_samples, n_samples).
        k: Number of eigenvectors to extract, which matches the number of
            clusters.

    Returns:
        Matrix Y of shape (n_samples, k) with row-normalized eigenvectors
        corresponding to k largest eigenvalues
    """
    n_samples = laplacian.shape[0]
    if not 0 < k <= n_samples:
        raise ValueError(f"k must be between 1 and n_samples ({n_samples}), got {k}")

    # L is real and symmetric, so a Hermitian eigensolver is faster and more
    # numerically stable than the general eigensolver. SciPy can compute only
    # the top-k eigenvectors; NumPy is kept as a fallback.
    if scipy_eigh is not None:
        _, eigenvectors = scipy_eigh(
            laplacian,
            subset_by_index=[n_samples - k, n_samples - 1],
            check_finite=False,
        )
        embedding = eigenvectors
    else:
        _, eigenvectors = np.linalg.eigh(laplacian)
        embedding = eigenvectors[:, -k:]

    # Row-normalize: each row becomes unit vector
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    normalized_embedding = np.zeros_like(embedding)
    nonzero = norms[:, 0] != 0
    normalized_embedding[nonzero, :] = embedding[nonzero, :] / norms[nonzero]

    return normalized_embedding


def build_spectral_training_graph(
    samples: np.ndarray,
    sigma: float,
    max_k: int,
    *,
    chunk_size: int = DEFAULT_AFFINITY_CHUNK_SIZE,
    distance_matrix: np.ndarray | None = None,
) -> SpectralTrainingGraph:
    """Build one graph/eigendecomposition reusable by every ``K <= max_k``."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(f"samples must be a 2D array, got shape {samples.shape}")
    if len(samples) == 0:
        raise ValueError("samples cannot be empty.")
    if not 0 < int(max_k) <= len(samples):
        raise ValueError(
            f"max_k must be between 1 and n_samples ({len(samples)}), got {max_k}"
        )

    if distance_matrix is None:
        affinities = affinity_matrix(samples, sigma, chunk_size=chunk_size)
    else:
        distances = np.asarray(distance_matrix, dtype=np.float64)
        if distances.shape != (len(samples), len(samples)):
            raise ValueError(
                "distance_matrix shape must match the number of samples."
            )
        affinities = affinity_matrix_from_distances(distances, sigma)

    degrees = affinities.sum(axis=1)
    laplacian = normalized_laplacian(affinities, copy=False)
    if scipy_eigh is not None:
        eigenvalues, eigenvectors = scipy_eigh(
            laplacian,
            subset_by_index=[len(samples) - int(max_k), len(samples) - 1],
            check_finite=False,
        )
    else:
        all_values, all_vectors = np.linalg.eigh(laplacian)
        eigenvalues = all_values[-int(max_k) :]
        eigenvectors = all_vectors[:, -int(max_k) :]
    return SpectralTrainingGraph(
        sigma=float(sigma),
        samples=samples,
        training_degrees=degrees,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
    )


class SpectralClusteringModel:
    """Training-graph spectral clustering with Nyström out-of-sample routing."""

    def __init__(
        self,
        sigma: float = 1.0,
        k: int = 3,
        random_state: int = 42,
        chunk_size: int = DEFAULT_AFFINITY_CHUNK_SIZE,
    ) -> None:
        self.sigma = float(sigma)
        self.k = int(k)
        self.random_state = int(random_state)
        self.chunk_size = int(chunk_size)

    def fit(
        self,
        samples: np.ndarray,
        *,
        distance_matrix: np.ndarray | None = None,
    ) -> "SpectralClusteringModel":
        """Fit the training graph, eigensystem, and embedding-space K-means."""
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2:
            raise ValueError(f"samples must be a 2D array, got shape {samples.shape}")
        if len(samples) == 0:
            raise ValueError("samples cannot be empty.")
        if not 0 < self.k <= len(samples):
            raise ValueError(
                f"k must be between 1 and n_samples ({len(samples)}), got {self.k}"
            )

        graph = build_spectral_training_graph(
            samples,
            self.sigma,
            self.k,
            chunk_size=self.chunk_size,
            distance_matrix=distance_matrix,
        )
        return self.fit_from_training_graph(graph)

    def fit_from_training_graph(
        self,
        graph: SpectralTrainingGraph,
    ) -> "SpectralClusteringModel":
        """Fit embedding K-means from a graph cached for this sigma and K."""
        if not np.isclose(graph.sigma, self.sigma, rtol=1e-12, atol=0.0):
            raise ValueError("Cached spectral graph sigma does not match the model.")
        if self.k > graph.max_k:
            raise ValueError(
                f"Cached spectral graph supports K <= {graph.max_k}, got {self.k}."
            )

        eigenvalues = graph.eigenvalues[-self.k :]
        eigenvectors = graph.eigenvectors[:, -self.k :]

        norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
        embedding = np.zeros_like(eigenvectors)
        nonzero = norms[:, 0] > 0
        embedding[nonzero] = eigenvectors[nonzero] / norms[nonzero]

        self.samples_ = graph.samples
        self.training_degrees_ = graph.training_degrees
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        self.embedding_ = embedding
        self.kmeans_ = KMeans(
            n_clusters=self.k,
            random_state=self.random_state,
            n_init=10,
        ).fit(embedding)
        self.labels_ = np.asarray(self.kmeans_.labels_, dtype=int)
        return self

    def transform(
        self,
        samples: np.ndarray,
        *,
        cross_distance_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Nyström-extend held-out samples into the fitted spectral embedding."""
        if not hasattr(self, "eigenvectors_"):
            raise RuntimeError("SpectralClusteringModel must be fitted first.")
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2:
            raise ValueError("samples must be a 2D array.")
        if len(samples) == 0:
            return np.empty((0, self.k), dtype=np.float64)

        if cross_distance_matrix is None:
            cross_affinities = cross_affinity_matrix(
                samples,
                self.samples_,
                self.sigma,
                chunk_size=self.chunk_size,
            )
        else:
            distances = np.asarray(cross_distance_matrix, dtype=np.float64)
            if distances.shape != (len(samples), len(self.samples_)):
                raise ValueError(
                    "cross_distance_matrix must have shape "
                    "(n_held_out, n_training)."
                )
            cross_affinities = cross_affinity_matrix_from_distances(
                distances,
                self.sigma,
            )

        held_out_degrees = cross_affinities.sum(axis=1)
        train_scale = np.zeros_like(self.training_degrees_)
        train_mask = self.training_degrees_ > 0
        train_scale[train_mask] = self.training_degrees_[train_mask] ** -0.5
        held_out_scale = np.zeros_like(held_out_degrees)
        held_out_mask = held_out_degrees > 0
        held_out_scale[held_out_mask] = held_out_degrees[held_out_mask] ** -0.5
        normalized_cross = cross_affinities * train_scale[None, :]
        normalized_cross *= held_out_scale[:, None]

        inverse_eigenvalues = np.zeros_like(self.eigenvalues_)
        stable = np.abs(self.eigenvalues_) > EIGENVALUE_TOLERANCE
        inverse_eigenvalues[stable] = 1.0 / self.eigenvalues_[stable]
        embedding = (
            normalized_cross @ self.eigenvectors_
        ) * inverse_eigenvalues[None, :]
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        normalized = np.zeros_like(embedding)
        nonzero = norms[:, 0] > 0
        normalized[nonzero] = embedding[nonzero] / norms[nonzero]
        return normalized

    def predict(
        self,
        samples: np.ndarray,
        *,
        cross_distance_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Assign held-out samples using K-means in Nyström spectral space."""
        embedding = self.transform(
            samples,
            cross_distance_matrix=cross_distance_matrix,
        )
        if len(embedding) == 0:
            return np.array([], dtype=int)
        return np.asarray(self.kmeans_.predict(embedding), dtype=int)


def spectral_clustering(
    samples: np.ndarray,
    sigma: float = 1.0,
    k: int = 3,
    random_state: int = 42,
    chunk_size: int = DEFAULT_AFFINITY_CHUNK_SIZE,
    distance_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Perform spectral clustering using normalized Laplacian eigenvectors.

    Algorithm:
        1. Build affinity matrix using Gaussian kernel
        2. Compute normalized Laplacian
        3. Extract k eigenvectors with largest eigenvalues
        4. Apply K-means clustering on eigenvectors

    Args:
        samples: Input data of shape (n_samples, n_features).
        sigma: Bandwidth parameter for Gaussian kernel (default: 1.0)
        k: Number of clusters (default: 3)
        random_state: Random seed for KMeans reproducibility
        chunk_size: Number of rows to process per affinity-matrix chunk.
        distance_matrix: Optional precomputed pairwise distances. When given,
            these replace Euclidean distances between sample rows.

    Returns:
        Cluster labels of shape (n_samples,) with values in [0, k-1]
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(f"samples must be a 2D array, got shape {samples.shape}")
    if len(samples) == 0:
        return np.array([], dtype=int)
    model = SpectralClusteringModel(
        sigma=sigma,
        k=k,
        random_state=random_state,
        chunk_size=chunk_size,
    ).fit(samples, distance_matrix=distance_matrix)
    return model.labels_


def fit_predict(
    feature_rows: np.ndarray | list,
    sigma: float = 1.0,
    k: int = 3,
    random_state: int = 42,
    chunk_size: int = DEFAULT_AFFINITY_CHUNK_SIZE,
) -> List[int]:
    """Spectral clustering entrypoint for the pipeline.

    This function adapts the spectral clustering algorithm to match the
    pipeline interface.

    Args:
        feature_rows: Input data of shape (n_samples, n_features)
                     Can be a 2D numpy array or list of samples
        sigma: Bandwidth parameter for Gaussian kernel (default: 1.0)
               Controls neighborhood size in affinity matrix
               - Smaller sigma: focus on nearby points
               - Larger sigma: global structure
        k: Number of clusters (default: 3)
        random_state: Random seed for reproducibility
        chunk_size: Number of rows to process per affinity-matrix chunk.

    Returns:
        List of cluster labels, one per sample

    Example:
        >>> windows_flat = windows.reshape(windows.shape[0], -1)  # (9494, 20)
        >>> labels = fit_predict(windows_flat, sigma=1.0, k=3)
        >>> print(len(labels))  # 9494
    """
    # Convert to numpy array if needed
    if not isinstance(feature_rows, np.ndarray):
        samples = np.array(feature_rows)
    else:
        samples = feature_rows

    if len(samples) == 0:
        return []

    # Run spectral clustering
    labels = spectral_clustering(
        samples,
        sigma=sigma,
        k=k,
        random_state=random_state,
        chunk_size=chunk_size,
    )

    # Return as list of ints
    return labels.tolist()

