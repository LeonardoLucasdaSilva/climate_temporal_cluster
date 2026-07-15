"""Test precipitation-threshold cluster assignment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from NEW.precipitation_threshold_clusters import (  # noqa: E402
    PrecipitationThresholdConfig,
    assign_threshold_clusters,
    create_precipitation_threshold_clusters,
    create_precipitation_threshold_clusters_from_dataframe,
    window_precipitation_levels,
)


class PrecipitationThresholdClustersTest(unittest.TestCase):
    def test_assigns_clusters_from_existing_windows(self) -> None:
        windows = np.array(
            [
                [[0.0, 20.0], [0.0, 21.0]],
                [[1.0, 22.0], [2.0, 23.0]],
                [[5.0, 24.0], [7.0, 25.0]],
                [[20.0, 26.0], [15.0, 27.0]],
            ]
        )
        feature_columns = ["PRECIPITACAO_TOTAL", "TEMPERATURA_MAXIMA"]
        config = PrecipitationThresholdConfig(
            n_clusters=4,
            thresholds_mm=[0, 10, 30],
        )

        result = create_precipitation_threshold_clusters(
            windows,
            feature_columns,
            config,
        )

        np.testing.assert_array_equal(result.precipitation_mm, [0.0, 3.0, 12.0, 35.0])
        np.testing.assert_array_equal(result.labels, [0, 1, 2, 3])
        self.assertEqual(result.summary["n_windows"].tolist(), [1, 1, 1, 1])

    def test_supports_max_aggregation(self) -> None:
        windows = np.array([[[2.0], [8.0]], [[15.0], [1.0]]])
        levels = window_precipitation_levels(
            windows,
            ["PRECIPITACAO_TOTAL"],
            aggregation="max",
        )

        np.testing.assert_array_equal(levels, [8.0, 15.0])

    def test_assign_threshold_clusters_uses_last_cluster_above_final_limit(self) -> None:
        config = PrecipitationThresholdConfig(
            n_clusters=3,
            thresholds_mm=[0, 5],
        )

        labels = assign_threshold_clusters([0, 0.1, 5, 5.1], config)

        np.testing.assert_array_equal(labels, [0, 1, 1, 2])

    def test_requires_zero_as_first_threshold(self) -> None:
        config = PrecipitationThresholdConfig(
            n_clusters=3,
            thresholds_mm=[1, 5],
        )

        with self.assertRaises(ValueError):
            config.validate()

    def test_can_create_raw_precipitation_windows_from_dataframe(self) -> None:
        df = pd.DataFrame(
            {
                "Data": pd.date_range("2026-01-01", periods=5),
                "PRECIPITACAO_TOTAL": [0.0, 0.0, 2.0, 3.0, 9.0],
            }
        )
        config = PrecipitationThresholdConfig(
            n_clusters=3,
            thresholds_mm=[0, 5],
        )

        windows, result = create_precipitation_threshold_clusters_from_dataframe(
            df,
            window_size=2,
            config=config,
        )

        self.assertEqual(windows.shape, (4, 2, 1))
        np.testing.assert_array_equal(result.precipitation_mm, [0.0, 2.0, 5.0, 12.0])
        np.testing.assert_array_equal(result.labels, [0, 1, 1, 2])


if __name__ == "__main__":
    unittest.main()
