"""K-Shape clustering for univariate and multivariate time series."""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate


_NORMALIZATION_EPSILON = 1e-12


def _as_time_series_dataset(
    X: np.ndarray,
    *,
    allow_empty: bool = False,
) -> np.ndarray:
    """Return finite time series with shape ``(samples, timestamps, features)``."""
    dataset = np.asarray(X, dtype=float)
    if dataset.ndim == 2:
        dataset = dataset[:, :, np.newaxis]
    if dataset.ndim != 3:
        raise ValueError(
            "K-Shape input must be a 2D or 3D array with samples on axis 0."
        )
    if dataset.shape[0] == 0 and not allow_empty:
        raise ValueError("K-Shape input must contain at least one time series.")
    if dataset.shape[1] < 2:
        raise ValueError("K-Shape time series must contain at least two timestamps.")
    if dataset.shape[2] == 0:
        raise ValueError("K-Shape input must contain at least one feature.")
    if not np.all(np.isfinite(dataset)):
        raise ValueError("K-Shape input must contain only finite values.")
    return dataset


def _z_normalize(dataset: np.ndarray) -> np.ndarray:
    """Z-normalize every feature of every time series along the time axis."""
    means = dataset.mean(axis=1, keepdims=True)
    standard_deviations = dataset.std(axis=1, keepdims=True)
    normalized = np.zeros_like(dataset, dtype=float)
    np.divide(
        dataset - means,
        standard_deviations,
        out=normalized,
        where=standard_deviations > _NORMALIZATION_EPSILON,
    )
    return normalized


def _cross_correlation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return full cross-correlation summed over feature channels."""
    correlation = np.zeros(2 * first.shape[0] - 1, dtype=float)
    for feature_index in range(first.shape[1]):
        correlation += correlate(
            first[:, feature_index],
            second[:, feature_index],
            mode="full",
            method="fft",
        )
    return correlation


def _shape_based_distance_and_shift(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, int]:
    """Return shape-based distance and the shift aligning ``second`` to ``first``."""
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= _NORMALIZATION_EPSILON:
        both_constant = (
            np.linalg.norm(first) <= _NORMALIZATION_EPSILON
            and np.linalg.norm(second) <= _NORMALIZATION_EPSILON
        )
        return (0.0 if both_constant else 1.0), 0

    normalized_correlation = _cross_correlation(first, second) / denominator
    best_index = int(np.argmax(normalized_correlation))
    best_correlation = float(np.clip(normalized_correlation[best_index], -1.0, 1.0))
    shift = best_index - (second.shape[0] - 1)
    return 1.0 - best_correlation, shift


def shape_based_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the K-Shape distance between two equal-length time series."""
    first_dataset = _as_time_series_dataset(np.asarray(first)[np.newaxis, ...])
    second_dataset = _as_time_series_dataset(np.asarray(second)[np.newaxis, ...])
    if first_dataset.shape[1:] != second_dataset.shape[1:]:
        raise ValueError("K-Shape time series must have matching shapes.")
    first_normalized = _z_normalize(first_dataset)[0]
    second_normalized = _z_normalize(second_dataset)[0]
    distance, _ = _shape_based_distance_and_shift(
        first_normalized,
        second_normalized,
    )
    return distance


def _shift_time_series(series: np.ndarray, shift: int) -> np.ndarray:
    """Shift a time series without wrapping, padding uncovered values with zero."""
    shifted = np.zeros_like(series)
    if shift == 0:
        shifted[:] = series
    elif 0 < shift < len(series):
        shifted[shift:] = series[:-shift]
    elif -len(series) < shift < 0:
        shifted[:shift] = series[-shift:]
    return shifted


def _distance_matrix(dataset: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return shape-based distances from every series to every centroid."""
    distances = np.empty((len(dataset), len(centroids)), dtype=float)
    for sample_index, sample in enumerate(dataset):
        for cluster_index, centroid in enumerate(centroids):
            distances[sample_index, cluster_index], _ = (
                _shape_based_distance_and_shift(centroid, sample)
            )
    return distances


def _extract_shape(
    members: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Extract one normalized shape centroid from aligned cluster members."""
    aligned = np.empty_like(members)
    for member_index, member in enumerate(members):
        _, shift = _shape_based_distance_and_shift(reference, member)
        aligned[member_index] = _shift_time_series(member, shift)
    aligned = _z_normalize(aligned)

    n_timestamps = members.shape[1]
    centering = np.eye(n_timestamps) - np.full(
        (n_timestamps, n_timestamps),
        1.0 / n_timestamps,
    )
    centroid = np.empty_like(reference)
    for feature_index in range(members.shape[2]):
        feature_members = aligned[:, :, feature_index]
        shape_matrix = centering @ feature_members.T @ feature_members @ centering
        eigenvalues, eigenvectors = np.linalg.eigh(shape_matrix)
        feature_centroid = eigenvectors[:, int(np.argmax(eigenvalues))]
        feature_reference = reference[:, feature_index]
        if np.linalg.norm(feature_centroid - feature_reference) > np.linalg.norm(
            -feature_centroid - feature_reference
        ):
            feature_centroid = -feature_centroid
        centroid[:, feature_index] = feature_centroid
    return _z_normalize(centroid[np.newaxis, ...])[0]


class KShape:
    """Cluster equal-length time series using the K-Shape algorithm.

    Two-dimensional input is interpreted as a collection of univariate series.
    Three-dimensional input uses the axes ``(samples, timestamps, features)``;
    all feature channels share the temporal shift used during alignment.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        max_iter: int = 100,
        n_init: int = 1,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state

    def _validate_parameters(self, n_samples: int) -> None:
        integer_parameters = {
            "n_clusters": self.n_clusters,
            "max_iter": self.max_iter,
            "n_init": self.n_init,
        }
        for name, value in integer_parameters.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(f"{name} must be a positive integer.")
            if int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.n_clusters > n_samples:
            raise ValueError(
                "n_clusters cannot exceed the number of input time series."
            )
        if isinstance(self.tol, (bool, np.bool_)) or not np.isfinite(self.tol):
            raise ValueError("tol must be a finite non-negative number.")
        if self.tol < 0:
            raise ValueError("tol must be a finite non-negative number.")

    def _initial_labels(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        labels = np.arange(n_samples, dtype=int) % self.n_clusters
        rng.shuffle(labels)
        return labels

    def _update_centroids(
        self,
        dataset: np.ndarray,
        labels: np.ndarray,
        previous_centroids: np.ndarray | None,
    ) -> np.ndarray:
        centroids = np.empty(
            (self.n_clusters, dataset.shape[1], dataset.shape[2]),
            dtype=float,
        )
        for cluster_index in range(self.n_clusters):
            members = dataset[labels == cluster_index]
            if len(members) == 0:
                raise RuntimeError("K-Shape encountered an empty cluster.")
            reference = (
                members[0]
                if previous_centroids is None
                else previous_centroids[cluster_index]
            )
            centroids[cluster_index] = _extract_shape(members, reference)
        return centroids

    def _restore_empty_clusters(
        self,
        labels: np.ndarray,
        distances: np.ndarray,
    ) -> np.ndarray:
        labels = labels.copy()
        counts = np.bincount(labels, minlength=self.n_clusters)
        assigned_distances = distances[np.arange(len(labels)), labels]
        for empty_cluster in np.flatnonzero(counts == 0):
            candidates = np.flatnonzero(counts[labels] > 1)
            if len(candidates) == 0:
                raise RuntimeError("K-Shape could not restore an empty cluster.")
            moved_sample = int(candidates[np.argmax(assigned_distances[candidates])])
            donor_cluster = int(labels[moved_sample])
            labels[moved_sample] = int(empty_cluster)
            counts[donor_cluster] -= 1
            counts[empty_cluster] += 1
        return labels

    def _fit_once(
        self,
        dataset: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        labels = self._initial_labels(len(dataset), rng)
        centroids: np.ndarray | None = None
        previous_inertia = np.inf

        for iteration in range(1, self.max_iter + 1):
            centroids = self._update_centroids(dataset, labels, centroids)
            distances = _distance_matrix(dataset, centroids)
            updated_labels = self._restore_empty_clusters(
                np.argmin(distances, axis=1),
                distances,
            )
            inertia = float(
                np.sum(distances[np.arange(len(dataset)), updated_labels] ** 2)
            )
            labels_unchanged = np.array_equal(updated_labels, labels)
            converged = abs(previous_inertia - inertia) <= self.tol
            labels = updated_labels
            if labels_unchanged or converged:
                break
            previous_inertia = inertia

        centroids = self._update_centroids(dataset, labels, centroids)
        distances = _distance_matrix(dataset, centroids)
        labels = self._restore_empty_clusters(np.argmin(distances, axis=1), distances)
        inertia = float(np.sum(distances[np.arange(len(dataset)), labels] ** 2))
        return labels, centroids, inertia, iteration

    def fit(self, X: np.ndarray) -> KShape:
        """Fit K-Shape and expose labels and learned shape centroids."""
        dataset = _as_time_series_dataset(X)
        self._validate_parameters(len(dataset))
        normalized_dataset = _z_normalize(dataset)
        root_rng = np.random.default_rng(self.random_state)

        best_result: tuple[np.ndarray, np.ndarray, float, int] | None = None
        for _ in range(self.n_init):
            run_rng = np.random.default_rng(
                root_rng.integers(0, np.iinfo(np.int64).max)
            )
            result = self._fit_once(normalized_dataset, run_rng)
            if best_result is None or result[2] < best_result[2]:
                best_result = result

        if best_result is None:  # pragma: no cover - n_init validation prevents it
            raise RuntimeError("K-Shape did not execute any initialization.")
        labels, centroids, inertia, n_iter = best_result
        self.labels_ = labels
        self.cluster_centers_ = centroids
        self.inertia_ = inertia
        self.n_iter_ = n_iter
        self.n_timestamps_ = dataset.shape[1]
        self.n_features_in_ = dataset.shape[2]
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit K-Shape and return training labels."""
        return self.fit(X).labels_.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign time series to the nearest learned shape centroid."""
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("KShape must be fitted before predict is called.")
        dataset = _as_time_series_dataset(X, allow_empty=True)
        expected_shape = (self.n_timestamps_, self.n_features_in_)
        if dataset.shape[1:] != expected_shape:
            raise ValueError(
                "K-Shape prediction input must match the fitted timestamp and "
                "feature dimensions."
            )
        if len(dataset) == 0:
            return np.array([], dtype=int)
        distances = _distance_matrix(_z_normalize(dataset), self.cluster_centers_)
        return np.argmin(distances, axis=1).astype(int)


def kshape_clustering(
    X: np.ndarray,
    k: int,
    random_state: int | None = None,
) -> np.ndarray:
    """Return K-Shape labels through the project's functional interface."""
    return KShape(n_clusters=k, random_state=random_state).fit_predict(X)
