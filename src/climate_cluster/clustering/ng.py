"""Spectral clustering algorithm for climate data."""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.cluster import KMeans


def metrica(si: np.ndarray, sj: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors.

    Args:
        si: First vector
        sj: Second vector

    Returns:
        Euclidean distance (L2 norm of difference)
    """
    return np.linalg.norm(si - sj)


def matriz_afinidade(S: np.ndarray, sigma: float) -> np.ndarray:
    """Create affinity matrix using Gaussian kernel.

    Args:
        S: Data matrix of shape (n_samples, n_features)
        sigma: Bandwidth parameter for Gaussian kernel

    Returns:
        Affinity matrix A of shape (n_samples, n_samples)
        A[i,j] = exp(-||s_i - s_j||^2 / (2*sigma^2)) for i != j
    """
    n = len(S)
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                A[i, j] = np.exp(-metrica(S[i], S[j]) ** 2 / (2 * sigma ** 2))

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
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(L)

    # Get indices of k largest eigenvalues
    indices = np.argsort(eigenvalues)[-k:]
    X = eigenvectors[:, indices].real

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

    Returns:
        Cluster labels of shape (n_samples,) with values in [0, k-1]
    """
    # Step 1: Affinity matrix
    A = matriz_afinidade(S, sigma)

    # Step 2: Normalized Laplacian
    L = matriz_L(A)

    # Step 3: Eigenvector matrix
    Y = matriz_Y(L, k)

    # Step 4: KMeans on eigenvectors
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    labels = kmeans.fit_predict(Y)

    return labels


def fit_predict(
    feature_rows: np.ndarray | list,
    sigma: float = 1.0,
    k: int = 3,
    random_state: int = 42,
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
    labels = spectral_clustering(S, sigma=sigma, k=k, random_state=random_state)

    # Return as list of ints
    return labels.tolist()

