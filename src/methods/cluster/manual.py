"""Horizon-rain-guided clustering for weather windows.

Cluster 0 is reserved for windows whose known horizon precipitation is zero.
Known rainy windows are ordered by precipitation and split into progressively
heavier rain groups. Windows without an observed horizon are assigned to the
nearest cluster centroid in window-feature space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from methods.tools.precipitation_utils import horizon_precipitation


@dataclass
class ManualRainClustering:
    """Cluster windows using known horizon rain and feature-space centroids.

    Labels are ordered by rain severity:

    - label 0: known zero-rain horizons;
    - labels 1 through ``n_clusters - 1``: low to heavy known rain;
    - unknown horizons: nearest learned window-feature centroid.
    """

    n_clusters: int
    zero_tolerance: float = 0.0

    def fit(self, feature_matrix: np.ndarray, horizon_rain: np.ndarray) -> "ManualRainClustering":
        """Learn ordered rain groups and their feature-space centroids."""
        features, rain = self._validated_inputs(feature_matrix, horizon_rain)
        known = np.isfinite(rain)
        zero = known & (np.abs(rain) <= self.zero_tolerance)
        rainy = known & (rain > self.zero_tolerance)

        if not np.any(zero):
            raise ValueError("At least one known zero-rain horizon is required.")

        n_rain_clusters = self.n_clusters - 1
        rainy_indices = np.flatnonzero(rainy)
        if len(rainy_indices) < n_rain_clusters:
            raise ValueError(
                f"At least {n_rain_clusters} known rainy horizons are required "
                f"to create {self.n_clusters} clusters; got {len(rainy_indices)}."
            )

        labels = np.full(len(features), -1, dtype=int)
        labels[zero] = 0

        ordered_rainy_indices = rainy_indices[
            np.argsort(rain[rainy_indices], kind="stable")
        ]
        rainy_groups = np.array_split(ordered_rainy_indices, n_rain_clusters)
        for cluster_id, group_indices in enumerate(rainy_groups, start=1):
            labels[group_indices] = cluster_id

        centroids = np.vstack(
            [features[labels == cluster_id].mean(axis=0) for cluster_id in range(self.n_clusters)]
        )
        rain_ranges = {
            0: (
                float(rain[zero].min()),
                float(rain[zero].max()),
            ),
            **{
                cluster_id: (
                    float(rain[group_indices].min()),
                    float(rain[group_indices].max()),
                )
                for cluster_id, group_indices in enumerate(rainy_groups, start=1)
            },
        }

        self.centroids_ = centroids
        self.rain_ranges_ = rain_ranges
        self.known_labels_ = labels
        return self

    def predict(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Assign windows to the nearest learned feature-space centroid."""
        if not hasattr(self, "centroids_"):
            raise RuntimeError("fit must be called before predict.")

        features = _as_feature_matrix(feature_matrix)
        if features.shape[1] != self.centroids_.shape[1]:
            raise ValueError(
                f"Expected {self.centroids_.shape[1]} features, "
                f"got {features.shape[1]}."
            )
        distances_sq = np.sum(
            (features[:, None, :] - self.centroids_[None, :, :]) ** 2,
            axis=2,
        )
        return np.argmin(distances_sq, axis=1).astype(int)

    def fit_predict(
        self,
        feature_matrix: np.ndarray,
        horizon_rain: np.ndarray,
    ) -> np.ndarray:
        """Cluster known horizons by rain and infer labels for unknown ones."""
        features, rain = self._validated_inputs(feature_matrix, horizon_rain)
        self.fit(features, rain)

        labels = self.known_labels_.copy()
        unknown = ~np.isfinite(rain)
        if np.any(unknown):
            labels[unknown] = self.predict(features[unknown])
        self.labels_ = labels
        return labels

    def _validated_inputs(
        self,
        feature_matrix: np.ndarray,
        horizon_rain: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        if self.zero_tolerance < 0:
            raise ValueError("zero_tolerance cannot be negative.")

        features = _as_feature_matrix(feature_matrix)
        rain = np.asarray(horizon_rain, dtype=float)
        if rain.ndim != 1:
            raise ValueError(f"horizon_rain must be one-dimensional, got {rain.shape}.")
        if len(features) != len(rain):
            raise ValueError(
                f"feature_matrix has {len(features)} rows but horizon_rain has "
                f"{len(rain)} values."
            )
        if np.any(np.isfinite(rain) & (rain < 0)):
            raise ValueError("Known horizon precipitation cannot be negative.")
        return features, rain


def manual_clustering(
    feature_matrix: np.ndarray,
    horizon_rain: np.ndarray,
    k: int,
    *,
    zero_tolerance: float = 0.0,
) -> np.ndarray:
    """Return manual rain-cluster labels for one feature matrix."""
    return ManualRainClustering(
        n_clusters=k,
        zero_tolerance=zero_tolerance,
    ).fit_predict(feature_matrix, horizon_rain)


def _as_feature_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    """Return a finite two-dimensional floating-point feature matrix."""
    features = np.asarray(feature_matrix, dtype=float)
    if features.ndim != 2:
        raise ValueError(
            f"feature_matrix must be two-dimensional, got shape {features.shape}."
        )
    if len(features) == 0:
        raise ValueError("feature_matrix cannot be empty.")
    if not np.all(np.isfinite(features)):
        raise ValueError("feature_matrix must contain only finite values.")
    return features
