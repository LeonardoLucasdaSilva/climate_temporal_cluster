"""Tests for K-Shape clustering and pipeline dispatch."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from methods.cluster.cluster_pipeline import (  # noqa: E402
    SUPPORTED_CLUSTERING_ALGORITHMS,
    cluster_feature_matrix,
)
from methods.cluster.kshape import KShape, shape_based_distance  # noqa: E402
from methods.lstm_cluster.pipeline import (  # noqa: E402
    ExperimentConfig,
    _cluster_window_splits,
    create_window_split_data,
)


class KShapeTest(unittest.TestCase):
    def test_shape_distance_ignores_offset_and_amplitude(self) -> None:
        first = np.array([0.0, 1.0, 3.0, 1.0, 0.0])
        transformed = 4.0 * first + 12.0

        self.assertAlmostEqual(shape_based_distance(first, transformed), 0.0)

    def test_kshape_separates_shifted_waveforms(self) -> None:
        timestamps = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        first_shape = np.sin(timestamps)
        second_shape = np.sin(2.0 * timestamps)
        series = []
        expected = []
        for cluster_id, shape in enumerate((first_shape, second_shape)):
            for shift in (-2, -1, 0, 1, 2):
                series.append((cluster_id + 1.0) * np.roll(shape, shift) + 5.0)
                expected.append(cluster_id)

        model = KShape(
            n_clusters=2,
            n_init=5,
            max_iter=50,
            random_state=7,
        )
        labels = model.fit_predict(np.asarray(series))

        self.assertEqual(adjusted_rand_score(expected, labels), 1.0)
        np.testing.assert_array_equal(model.predict(np.asarray(series)), labels)
        self.assertEqual(model.cluster_centers_.shape, (2, 24, 1))

    def test_multivariate_series_are_supported(self) -> None:
        first = np.array(
            [
                [[0.0, 2.0], [1.0, 1.0], [2.0, 0.0], [1.0, 1.0]],
                [[0.0, 4.0], [2.0, 2.0], [4.0, 0.0], [2.0, 2.0]],
                [[2.0, 0.0], [1.0, 1.0], [0.0, 2.0], [1.0, 1.0]],
                [[4.0, 0.0], [2.0, 2.0], [0.0, 4.0], [2.0, 2.0]],
            ]
        )

        labels = KShape(n_clusters=2, n_init=3, random_state=3).fit_predict(first)

        self.assertEqual(labels.shape, (4,))
        self.assertEqual(len(np.unique(labels)), 2)

    def test_dispatcher_accepts_kshape_alias(self) -> None:
        series = np.array(
            [
                [0.0, 1.0, 0.0, -1.0],
                [0.0, 2.0, 0.0, -2.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 2.0, 0.0, 2.0],
            ]
        )

        labels = cluster_feature_matrix(
            series,
            n_clusters=2,
            algorithm="KSHAPE",
            random_state=11,
        )

        self.assertIn("kshape", SUPPORTED_CLUSTERING_ALGORITHMS)
        self.assertEqual(labels.shape, (4,))

    def test_lstm_pipeline_fits_and_predicts_from_window_tensors(self) -> None:
        config = ExperimentConfig(
            state="RS",
            station_id="A801",
            window_size=3,
            n_clusters=2,
            algorithm="kshape",
            sigma=None,
        )
        train_windows = np.arange(24.0).reshape(4, 3, 2)
        val_windows = np.arange(6.0).reshape(1, 3, 2)
        test_windows = np.arange(6.0, 12.0).reshape(1, 3, 2)
        fitted_model = Mock()
        fitted_model.labels_ = np.array([0, 0, 1, 1])
        fitted_model.predict.side_effect = [np.array([0]), np.array([1])]
        fitted_model.fit.return_value = fitted_model

        with patch(
            "methods.lstm_cluster.pipeline.KShape",
            return_value=fitted_model,
        ) as kshape_class:
            labels = _cluster_window_splits(
                config,
                train_windows.reshape(4, -1),
                val_windows.reshape(1, -1),
                test_windows.reshape(1, -1),
                np.arange(4.0),
                random_state=42,
                manual_zero_tolerance=0.0,
                train_windows=train_windows,
                val_windows=val_windows,
                test_windows=test_windows,
            )

        kshape_class.assert_called_once_with(n_clusters=2, random_state=42)
        fitted_model.fit.assert_called_once_with(train_windows)
        np.testing.assert_array_equal(labels[0], [0, 0, 1, 1])
        np.testing.assert_array_equal(labels[1], [0])
        np.testing.assert_array_equal(labels[2], [1])

    def test_lstm_window_pipeline_runs_with_kshape(self) -> None:
        day = np.arange(90, dtype=float)
        dataframe = pd.DataFrame(
            {
                "Data": pd.date_range("2025-01-01", periods=len(day)),
                "TEMPERATURA_MAXIMA": np.sin(day / 3.0),
                "PRECIPITACAO_TOTAL": 2.0 + np.cos(day / 5.0),
            }
        )
        config = ExperimentConfig(
            state="RS",
            station_id="A801",
            window_size=6,
            n_clusters=2,
            algorithm="kshape",
            sigma=None,
        )

        split_data, _ = create_window_split_data(
            dataframe,
            config,
            ["TEMPERATURA_MAXIMA", "PRECIPITACAO_TOTAL"],
            clustering_feature_normalize="standard",
            clustering_precipitation_normalize="standard",
            variance_threshold=None,
            train_ratio=0.6,
            val_ratio=0.2,
            run_only_cluster=True,
        )

        self.assertEqual(len(np.unique(split_data.c_train)), 2)
        self.assertEqual(len(split_data.c_train), len(split_data.i_train))
        self.assertEqual(len(split_data.c_val), len(split_data.i_val))
        self.assertEqual(len(split_data.c_test), len(split_data.i_test))

    def test_invalid_cluster_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            KShape(n_clusters=3).fit(np.ones((2, 5)))


if __name__ == "__main__":
    unittest.main()
