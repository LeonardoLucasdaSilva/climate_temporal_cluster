"""Rule-based manual clustering for weather windows.

The legacy method groups training windows from their known horizon rain. The
rain-level method instead groups them from equal-width bins of mean input-window
precipitation. Windows without a directly constructed label can be assigned to
the nearest cluster centroid in window-feature space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from methods.tools.precipitation_utils import horizon_precipitation


SUPPORTED_MANUAL_CLUSTERING_METHODS = ("legacy", "rain_level")


def normalize_manual_clustering_method(method: str) -> str:
    """Return a supported manual clustering method name."""
    normalized = str(method).strip().lower()
    if normalized in SUPPORTED_MANUAL_CLUSTERING_METHODS:
        return normalized

    supported = ", ".join(SUPPORTED_MANUAL_CLUSTERING_METHODS)
    raise ValueError(
        f"Unsupported manual_clustering_method: {method!r}. "
        f"Use one of: {supported}"
    )


@dataclass
class ManualRainClustering:
    """Cluster windows with a selected rain rule and feature-space centroids.

    ``method="legacy"`` preserves the original labels:

    - label 0: known zero-rain horizons;
    - labels 1 through ``n_clusters - 1``: low to heavy known rain;
    - unknown horizons: nearest learned window-feature centroid.

    ``method="rain_level"`` divides ``[0, max(window_mean_rain)]`` into
    ``n_clusters`` equal-width intervals. Internal upper bounds are open, so a
    value exactly on a boundary belongs to the next cluster. The training
    maximum belongs to the last cluster.
    """

    n_clusters: int
    zero_tolerance: float = 0.0
    method: str = "legacy"

    def fit(
        self,
        feature_matrix: np.ndarray,
        horizon_rain: np.ndarray | None = None,
        *,
        window_mean_rain: np.ndarray | None = None,
    ) -> "ManualRainClustering":
        """Learn rule-based training labels and feature-space centroids."""
        self._validate_configuration()
        features = _as_feature_matrix(feature_matrix)
        method = normalize_manual_clustering_method(self.method)

        if method == "legacy":
            if horizon_rain is None:
                raise ValueError(
                    "horizon_rain is required when manual_clustering_method='legacy'."
                )
            rain = _as_rain_vector(
                horizon_rain,
                len(features),
                value_name="horizon_rain",
                allow_nan=True,
            )
            labels = self._legacy_labels(rain)
            thresholds = None
        else:
            if window_mean_rain is None:
                raise ValueError(
                    "window_mean_rain is required when "
                    "manual_clustering_method='rain_level'."
                )
            rain = _as_rain_vector(
                window_mean_rain,
                len(features),
                value_name="window_mean_rain",
                allow_nan=False,
            )
            labels, thresholds = self._rain_level_labels(rain)

        empty_clusters = [
            cluster_id
            for cluster_id in range(self.n_clusters)
            if not np.any(labels == cluster_id)
        ]
        if empty_clusters:
            raise ValueError(
                "Manual clustering produced empty training clusters: "
                f"{empty_clusters}. Reduce n_clusters or change the method."
            )

        centroids = np.vstack(
            [
                features[labels == cluster_id].mean(axis=0)
                for cluster_id in range(self.n_clusters)
            ]
        )
        rain_ranges = {
            cluster_id: (
                float(rain[labels == cluster_id].min()),
                float(rain[labels == cluster_id].max()),
            )
            for cluster_id in range(self.n_clusters)
        }

        self.centroids_ = centroids
        self.rain_ranges_ = rain_ranges
        self.known_labels_ = labels
        self.thresholds_ = thresholds
        self.method_ = method
        self.rain_values_ = rain
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
        horizon_rain: np.ndarray | None = None,
        *,
        window_mean_rain: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fit the selected manual rule and return its training labels."""
        features = _as_feature_matrix(feature_matrix)
        self.fit(
            features,
            horizon_rain,
            window_mean_rain=window_mean_rain,
        )

        labels = self.known_labels_.copy()
        if self.method_ == "legacy":
            unknown = ~np.isfinite(self.rain_values_)
            if np.any(unknown):
                labels[unknown] = self.predict(features[unknown])
        self.labels_ = labels
        return labels

    def _validate_configuration(self) -> None:
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        if self.zero_tolerance < 0:
            raise ValueError("zero_tolerance cannot be negative.")

    def _legacy_labels(self, rain: np.ndarray) -> np.ndarray:
        known = np.isfinite(rain)
        zero = known & (np.abs(rain) <= self.zero_tolerance)
        rainy_indices = np.flatnonzero(known & (rain > self.zero_tolerance))

        if not np.any(zero):
            raise ValueError("At least one known zero-rain horizon is required.")

        n_rain_clusters = self.n_clusters - 1
        if len(rainy_indices) < n_rain_clusters:
            raise ValueError(
                f"At least {n_rain_clusters} known rainy horizons are required "
                f"to create {self.n_clusters} clusters; got {len(rainy_indices)}."
            )

        labels = np.full(len(rain), -1, dtype=int)
        labels[zero] = 0
        ordered_rainy_indices = rainy_indices[
            np.argsort(rain[rainy_indices], kind="stable")
        ]
        for cluster_id, group_indices in enumerate(
            np.array_split(ordered_rainy_indices, n_rain_clusters),
            start=1,
        ):
            labels[group_indices] = cluster_id
        return labels

    def _rain_level_labels(
        self,
        window_mean_rain: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        max_rain = float(window_mean_rain.max())
        if max_rain <= 0:
            raise ValueError(
                "rain_level manual clustering requires at least one training "
                "window with positive mean precipitation."
            )

        thresholds = np.linspace(0.0, max_rain, self.n_clusters + 1)
        labels = np.searchsorted(
            thresholds[1:-1],
            window_mean_rain,
            side="right",
        ).astype(int)
        return labels, thresholds


def manual_clustering(
    feature_matrix: np.ndarray,
    horizon_rain: np.ndarray | None,
    k: int,
    *,
    zero_tolerance: float = 0.0,
    method: str = "legacy",
    window_mean_rain: np.ndarray | None = None,
) -> np.ndarray:
    """Return manual rain-cluster labels for one feature matrix."""
    return ManualRainClustering(
        n_clusters=k,
        zero_tolerance=zero_tolerance,
        method=method,
    ).fit_predict(
        feature_matrix,
        horizon_rain,
        window_mean_rain=window_mean_rain,
    )


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


def _as_rain_vector(
    values: np.ndarray,
    expected_length: int,
    *,
    value_name: str,
    allow_nan: bool,
) -> np.ndarray:
    """Return a validated one-dimensional non-negative rain vector."""
    rain = np.asarray(values, dtype=float)
    if rain.ndim != 1:
        raise ValueError(f"{value_name} must be one-dimensional, got {rain.shape}.")
    if len(rain) != expected_length:
        raise ValueError(
            f"feature_matrix has {expected_length} rows but {value_name} has "
            f"{len(rain)} values."
        )
    if np.any(np.isfinite(rain) & (rain < 0)):
        raise ValueError(f"{value_name} cannot contain negative precipitation.")
    if np.any(np.isinf(rain)) or (not allow_nan and np.any(np.isnan(rain))):
        raise ValueError(f"{value_name} must contain only finite values.")
    return rain
