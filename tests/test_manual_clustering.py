"""Tests for horizon-rain-guided manual clustering."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from methods.cluster.manual import (
    ManualRainClustering,
    manual_clustering,
    normalize_manual_clustering_method,
)
from methods.cluster.cluster_pipeline import cluster_feature_matrix
from methods.tools.precipitation_utils import horizon_precipitation


class ManualRainClusteringTests(unittest.TestCase):
    def test_known_horizons_are_grouped_from_zero_to_heavy_rain(self) -> None:
        features = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [2.0, 2.0],
                [2.2, 2.0],
                [8.0, 8.0],
                [8.2, 8.0],
            ]
        )
        rain = np.array([0.0, 0.0, 1.0, 2.0, 20.0, 30.0])

        model = ManualRainClustering(n_clusters=3)
        labels = model.fit_predict(features, rain)

        np.testing.assert_array_equal(labels, [0, 0, 1, 1, 2, 2])
        self.assertEqual(model.rain_ranges_[0], (0.0, 0.0))
        self.assertEqual(model.rain_ranges_[1], (1.0, 2.0))
        self.assertEqual(model.rain_ranges_[2], (20.0, 30.0))

    def test_missing_horizons_use_nearest_feature_centroid(self) -> None:
        features = np.array(
            [
                [0.0],
                [0.2],
                [5.0],
                [5.2],
                [10.0],
                [10.2],
                [0.1],
                [9.9],
            ]
        )
        rain = np.array([0.0, 0.0, 1.0, 2.0, 20.0, 30.0, np.nan, np.nan])

        labels = manual_clustering(features, rain, k=3)

        self.assertEqual(labels[-2], 0)
        self.assertEqual(labels[-1], 2)

    def test_rain_level_uses_equal_width_half_open_intervals(self) -> None:
        features = np.arange(8, dtype=float).reshape(-1, 1)
        window_mean_rain = np.array([0.0, 1.0, 2.0, 3.5, 4.0, 5.5, 6.0, 8.0])

        model = ManualRainClustering(n_clusters=4, method="rain_level")
        labels = model.fit_predict(
            features,
            None,
            window_mean_rain=window_mean_rain,
        )

        np.testing.assert_array_equal(labels, [0, 0, 1, 1, 2, 2, 3, 3])
        np.testing.assert_allclose(model.thresholds_, [0.0, 2.0, 4.0, 6.0, 8.0])
        self.assertEqual(model.rain_ranges_[0], (0.0, 1.0))
        self.assertEqual(model.rain_ranges_[3], (6.0, 8.0))

    def test_rain_level_requires_every_equal_width_bin_to_be_nonempty(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty training clusters"):
            ManualRainClustering(n_clusters=3, method="rain_level").fit_predict(
                np.arange(4, dtype=float).reshape(-1, 1),
                None,
                window_mean_rain=np.array([0.0, 0.1, 0.2, 10.0]),
            )

    def test_manual_clustering_method_is_validated(self) -> None:
        self.assertEqual(normalize_manual_clustering_method(" RAIN_LEVEL "), "rain_level")
        with self.assertRaisesRegex(ValueError, "manual_clustering_method"):
            normalize_manual_clustering_method("unsupported")

    def test_horizon_precipitation_supports_future_horizons(self) -> None:
        df = pd.DataFrame({"PRECIPITACAO_TOTAL": [0.0, 1.0, 2.0, 3.0, 4.0]})

        horizon_one = horizon_precipitation(df, window_size=2, horizon=1)
        horizon_two = horizon_precipitation(df, window_size=2, horizon=2)

        np.testing.assert_allclose(horizon_one[:3], [2.0, 3.0, 4.0])
        self.assertTrue(np.isnan(horizon_one[3]))
        np.testing.assert_allclose(horizon_two[:2], [3.0, 4.0])
        self.assertTrue(np.all(np.isnan(horizon_two[2:])))

    def test_requires_enough_known_rain_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "known rainy horizons"):
            manual_clustering(
                np.array([[0.0], [1.0], [2.0]]),
                np.array([0.0, 1.0, np.nan]),
                k=3,
            )

    def test_common_cluster_dispatch_supports_manual_algorithm(self) -> None:
        features = np.array([[0.0], [0.2], [5.0], [5.2], [10.0], [10.2]])
        rain = np.array([0.0, 0.0, 1.0, 2.0, 20.0, 30.0])

        labels = cluster_feature_matrix(
            features,
            n_clusters=3,
            algorithm="manual",
            horizon_rain=rain,
        )

        np.testing.assert_array_equal(labels, [0, 0, 1, 1, 2, 2])

    def test_common_cluster_dispatch_requires_manual_horizon_rain(self) -> None:
        with self.assertRaisesRegex(ValueError, "horizon_rain"):
            cluster_feature_matrix(
                np.array([[0.0], [1.0]]),
                n_clusters=2,
                algorithm="manual",
            )


if __name__ == "__main__":
    unittest.main()
