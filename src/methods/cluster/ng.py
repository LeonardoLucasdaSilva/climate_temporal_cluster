"""Spectral clustering algorithm for climate data."""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.cluster import KMeans

try:
    from scipy.linalg import eigh as scipy_eigh
except ImportError:  # pragma: no cover - scipy is installed with scikit-learn.
    scipy_eigh = None


DEFAULT_AFFINITY_CHUNK_SIZE = 1024


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


def spectral_clustering(
    samples: np.ndarray,
    sigma: float = 1.0,
    k: int = 3,
    random_state: int = 42,
    chunk_size: int = DEFAULT_AFFINITY_CHUNK_SIZE,
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

    Returns:
        Cluster labels of shape (n_samples,) with values in [0, k-1]
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(f"samples must be a 2D array, got shape {samples.shape}")
    if len(samples) == 0:
        return np.array([], dtype=int)

    # Step 1: Affinity matrix
    affinities = affinity_matrix(samples, sigma, chunk_size=chunk_size)

    # Step 2: Normalized Laplacian
    laplacian = normalized_laplacian(affinities)

    # Step 3: Eigenvector matrix
    embedding = eigenvector_embedding(laplacian, k)

    # Step 4: KMeans on eigenvectors
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embedding)

    return labels


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

