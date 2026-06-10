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


def metrica(si: np.ndarray, sj: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors.

    Args:
        si: First vector
        sj: Second vector

    Returns:
        Euclidean distance (L2 norm of difference)
    """
    return np.linalg.norm(si - sj)


def matriz_afinidade(
    S: np.ndarray,
    sigma: float,
    chunk_size: int = DEFAULT_AFFINITY_CHUNK_SIZE,
) -> np.ndarray:
    """Create affinity matrix using Gaussian kernel.

    Args:
        S: Data matrix of shape (n_samples, n_features)
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

    S = np.asarray(S, dtype=np.float64)
    n = S.shape[0]
    squared_norms = np.einsum("ij,ij->i", S, S)
    gamma = 1.0 / (2.0 * sigma**2)
    A = np.empty((n, n), dtype=np.float64)

    # Compute pairwise squared distances in row chunks:
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        distances_sq = (
            squared_norms[start:stop, None]
            + squared_norms[None, :]
            - 2.0 * S[start:stop] @ S.T
        )
        np.maximum(distances_sq, 0.0, out=distances_sq)
        A[start:stop] = np.exp(-gamma * distances_sq)

    np.fill_diagonal(A, 0.0)
    return A


def matriz_L(A: np.ndarray) -> np.ndarray:
    """Create normalized Laplacian matrix.

    Args:
        A: Affinity matrix of shape (n_samples, n_samples)

    Returns:
        Normalized Laplacian L = D^(-1/2) * A * D^(-1/2)
        where D is the degree matrix (diagonal of row sums)
    """
    # Compute degree (row sums)
    degrees = np.sum(A, axis=1)

    # Create inverse sqrt of degree for normalization
    diag = np.zeros_like(degrees)
    mask = degrees > 0
    diag[mask] = degrees[mask] ** (-0.5)

    # Normalized Laplacian: L = D^(-1/2) * A * D^(-1/2)
    L = A * diag[:, None] * diag[None, :]

    return L


def matriz_Y(L: np.ndarray, k: int) -> np.ndarray:
    """Extract k eigenvectors from normalized Laplacian.

    Args:
        L: Normalized Laplacian matrix of shape (n_samples, n_samples)
        k: Number of eigenvectors to extract (number of clusters)

    Returns:
        Matrix Y of shape (n_samples, k) with row-normalized eigenvectors
        corresponding to k largest eigenvalues
    """
    n = L.shape[0]
    if not 0 < k <= n:
        raise ValueError(f"k must be between 1 and n_samples ({n}), got {k}")

    # L is real and symmetric, so a Hermitian eigensolver is faster and more
    # numerically stable than the general eigensolver. SciPy can compute only
    # the top-k eigenvectors; NumPy is kept as a fallback.
    if scipy_eigh is not None:
        _, eigenvectors = scipy_eigh(
            L,
            subset_by_index=[n - k, n - 1],
            check_finite=False,
        )
        X = eigenvectors
    else:
        _, eigenvectors = np.linalg.eigh(L)
        X = eigenvectors[:, -k:]

    # Row-normalize: each row becomes unit vector
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Y = np.zeros_like(X)
    nonzero = norms[:, 0] != 0
    Y[nonzero, :] = X[nonzero, :] / norms[nonzero]

    return Y


def spectral_clustering(
    S: np.ndarray,
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
        S: Input data of shape (n_samples, n_features)
        sigma: Bandwidth parameter for Gaussian kernel (default: 1.0)
        k: Number of clusters (default: 3)
        random_state: Random seed for KMeans reproducibility
        chunk_size: Number of rows to process per affinity-matrix chunk.

    Returns:
        Cluster labels of shape (n_samples,) with values in [0, k-1]
    """
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError(f"S must be a 2D array, got shape {S.shape}")
    if len(S) == 0:
        return np.array([], dtype=int)

    # Step 1: Affinity matrix
    A = matriz_afinidade(S, sigma, chunk_size=chunk_size)

    # Step 2: Normalized Laplacian
    L = matriz_L(A)

    # Step 3: Eigenvector matrix
    Y = matriz_Y(L, k)

    # Step 4: KMeans on eigenvectors
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(Y)

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
        S = np.array(feature_rows)
    else:
        S = feature_rows

    if len(S) == 0:
        return []

    # Run spectral clustering
    labels = spectral_clustering(
        S,
        sigma=sigma,
        k=k,
        random_state=random_state,
        chunk_size=chunk_size,
    )

    # Return as list of ints
    return labels.tolist()

