"""Tests for dataframe-level LSTM split window creation."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


stub_lstm = types.ModuleType("models.lstm")


class LSTMPrecipitationPredictor:
    """Minimal import stub; these tests do not train an LSTM."""


stub_lstm.LSTMPrecipitationPredictor = LSTMPrecipitationPredictor
sys.modules["models.lstm"] = stub_lstm

from methods.lstm_cluster.pipeline import (  # noqa: E402
    ExperimentConfig,
    _cluster_window_splits,
    _normalize_precipitation_scaler_type,
    create_window_split_data,
    quantile_weighted_mse_config,
    split_daily_dataframe,
)


class LSTMDataframeSplitsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "Data": pd.date_range("2025-01-01", periods=30),
                "TEMPERATURA_MAXIMA": np.arange(30, dtype=float),
                "TEMPERATURA_MIN": np.arange(30, dtype=float) / 2,
                "PRECIPITACAO_TOTAL": np.arange(30, dtype=float) % 5,
            }
        )

    def test_split_daily_dataframe_is_chronological(self) -> None:
        splits = split_daily_dataframe(self.df, train_ratio=0.5, val_ratio=0.2)

        self.assertEqual(len(splits.train), 15)
        self.assertEqual(len(splits.val), 6)
        self.assertEqual(len(splits.test), 9)
        self.assertEqual(splits.train_offset, 0)
        self.assertEqual(splits.val_offset, 15)
        self.assertEqual(splits.test_offset, 21)
        self.assertEqual(splits.train["Data"].iloc[-1], pd.Timestamp("2025-01-15"))
        self.assertEqual(splits.val["Data"].iloc[0], pd.Timestamp("2025-01-16"))
        self.assertEqual(splits.test["Data"].iloc[0], pd.Timestamp("2025-01-22"))

    def test_window_splits_do_not_share_raw_rows(self) -> None:
        config = ExperimentConfig(
            state="RS",
            station_id="A801",
            window_size=4,
            n_clusters=2,
            algorithm="kmeans",
            sigma=None,
        )
        split_data, _splits = create_window_split_data(
            self.df,
            config,
            ["TEMPERATURA_MAXIMA", "TEMPERATURA_MIN", "PRECIPITACAO_TOTAL"],
            clustering_feature_normalize="standard",
            clustering_precipitation_normalize="standard",
            lstm_feature_normalize="standard",
            lstm_precipitation_normalize="standard",
            variance_threshold=None,
            forecast_horizon=1,
            train_ratio=0.5,
            val_ratio=0.2,
            random_state=42,
            manual_zero_tolerance=0.0,
        )

        self.assertEqual(len(split_data.y_train), 11)
        self.assertEqual(len(split_data.y_val), 2)
        self.assertEqual(len(split_data.y_test), 5)
        self.assertEqual(split_data.i_train.tolist(), list(range(0, 11)))
        self.assertEqual(split_data.i_val.tolist(), list(range(15, 17)))
        self.assertEqual(split_data.i_test.tolist(), list(range(21, 26)))

        def raw_rows(indices: np.ndarray) -> set[int]:
            rows = set()
            for index in indices:
                rows.update(range(int(index), int(index) + config.window_size))
            return rows

        train_rows = raw_rows(split_data.i_train)
        val_rows = raw_rows(split_data.i_val)
        test_rows = raw_rows(split_data.i_test)

        self.assertFalse(train_rows & val_rows)
        self.assertFalse(train_rows & test_rows)
        self.assertFalse(val_rows & test_rows)

    def test_minmax_scaler_is_fit_only_on_training_rows(self) -> None:
        config = ExperimentConfig(
            state="RS",
            station_id="A801",
            window_size=4,
            n_clusters=2,
            algorithm="kmeans",
            sigma=None,
        )
        split_data, _splits = create_window_split_data(
            self.df,
            config,
            ["TEMPERATURA_MAXIMA", "TEMPERATURA_MIN"],
            clustering_feature_normalize="minmax",
            clustering_precipitation_normalize="standard",
            lstm_feature_normalize="minmax",
            lstm_precipitation_normalize="standard",
            variance_threshold=None,
            forecast_horizon=1,
            train_ratio=0.5,
            val_ratio=0.2,
            random_state=42,
            manual_zero_tolerance=0.0,
        )

        self.assertGreater(float(split_data.X_val.max()), 1.0)
        self.assertGreater(float(split_data.X_test.max()), 1.0)
        self.assertLessEqual(float(split_data.X_train.max()), 1.0)

    def test_precipitation_scaler_is_separate_from_covariate_scaler(self) -> None:
        config = ExperimentConfig(
            state="RS",
            station_id="A801",
            window_size=4,
            n_clusters=2,
            algorithm="kmeans",
            sigma=None,
        )
        split_data, _splits = create_window_split_data(
            self.df,
            config,
            ["TEMPERATURA_MAXIMA", "TEMPERATURA_MIN", "PRECIPITACAO_TOTAL"],
            clustering_feature_normalize="standard",
            clustering_precipitation_normalize="minmax",
            lstm_feature_normalize="standard",
            lstm_precipitation_normalize="minmax",
            variance_threshold=None,
            forecast_horizon=2,
            train_ratio=0.5,
            val_ratio=0.2,
            random_state=42,
            manual_zero_tolerance=0.0,
        )

        precipitation_positions = np.arange(2, split_data.X_train.shape[1], 3)
        covariate_positions = np.setdiff1d(
            np.arange(split_data.X_train.shape[1]),
            precipitation_positions,
        )
        self.assertGreaterEqual(float(split_data.X_train[:, precipitation_positions].min()), 0.0)
        self.assertLessEqual(float(split_data.X_train[:, precipitation_positions].max()), 1.0)
        self.assertLess(float(split_data.X_train[:, covariate_positions].min()), 0.0)

        np.testing.assert_allclose(
            split_data.y_train_by_lead_day_scaled,
            split_data.y_train_by_lead_day / 4.0,
        )
        np.testing.assert_allclose(
            split_data.y_val_by_lead_day_scaled,
            split_data.y_val_by_lead_day / 4.0,
        )
        np.testing.assert_allclose(
            split_data.y_train,
            split_data.y_train_by_lead_day[:, -1],
        )
        self.assertIsNotNone(split_data.target_scaler)

    def test_none_precipitation_scaler_keeps_precipitation_and_targets_in_mm(self) -> None:
        config = ExperimentConfig(
            state="RS",
            station_id="A801",
            window_size=4,
            n_clusters=2,
            algorithm="kmeans",
            sigma=None,
        )
        split_data, _splits = create_window_split_data(
            self.df,
            config,
            ["TEMPERATURA_MAXIMA", "TEMPERATURA_MIN", "PRECIPITACAO_TOTAL"],
            clustering_feature_normalize="standard",
            clustering_precipitation_normalize=None,
            lstm_feature_normalize="standard",
            lstm_precipitation_normalize=None,
            variance_threshold=None,
            forecast_horizon=2,
            train_ratio=0.5,
            val_ratio=0.2,
            random_state=42,
            manual_zero_tolerance=0.0,
        )

        precipitation_positions = np.arange(2, split_data.X_train.shape[1], 3)
        covariate_positions = np.setdiff1d(
            np.arange(split_data.X_train.shape[1]),
            precipitation_positions,
        )
        expected_train_precipitation = np.array(
            [
                self.df["PRECIPITACAO_TOTAL"]
                .iloc[int(start) : int(start) + config.window_size]
                .to_numpy(dtype=float)
                for start in split_data.i_train
            ]
        )

        np.testing.assert_allclose(
            split_data.X_train[:, precipitation_positions],
            expected_train_precipitation,
        )
        self.assertLess(float(split_data.X_train[:, covariate_positions].min()), 0.0)
        self.assertIsNone(split_data.target_scaler)
        np.testing.assert_allclose(
            split_data.y_train_by_lead_day_scaled,
            split_data.y_train_by_lead_day,
        )
        np.testing.assert_allclose(
            split_data.y_val_by_lead_day_scaled,
            split_data.y_val_by_lead_day,
        )

    def test_clustering_scalers_do_not_change_lstm_feature_space(self) -> None:
        config = ExperimentConfig(
            state="RS",
            station_id="A801",
            window_size=4,
            n_clusters=2,
            algorithm="kmeans",
            sigma=None,
        )
        feature_columns = [
            "TEMPERATURA_MAXIMA",
            "TEMPERATURA_MIN",
            "PRECIPITACAO_TOTAL",
        ]
        split_data, _splits = create_window_split_data(
            self.df,
            config,
            feature_columns,
            clustering_feature_normalize="standard",
            clustering_precipitation_normalize="minmax",
            lstm_feature_normalize=None,
            lstm_precipitation_normalize=None,
            variance_threshold=None,
            forecast_horizon=1,
            train_ratio=0.5,
            val_ratio=0.2,
            random_state=42,
            manual_zero_tolerance=0.0,
        )
        expected_train_windows = np.array(
            [
                self.df[feature_columns]
                .iloc[int(start) : int(start) + config.window_size]
                .to_numpy(dtype=float)
                .reshape(-1)
                for start in split_data.i_train
            ]
        )

        np.testing.assert_allclose(split_data.X_train, expected_train_windows)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                split_data.X_cluster_train,
                expected_train_windows,
            )
        np.testing.assert_allclose(
            split_data.y_train_by_lead_day_scaled,
            split_data.y_train_by_lead_day,
        )
        self.assertIsNone(split_data.target_scaler)

    def test_precipitation_scaler_type_accepts_disabled_values(self) -> None:
        self.assertIsNone(_normalize_precipitation_scaler_type(None))
        self.assertIsNone(_normalize_precipitation_scaler_type("none"))
        self.assertEqual(_normalize_precipitation_scaler_type("STANDARD"), "standard")

        with self.assertRaisesRegex(ValueError, "Unsupported precipitation_scaler_type"):
            _normalize_precipitation_scaler_type("robust")

    def test_forecast_horizon_controls_target_day_inside_each_split(self) -> None:
        config = ExperimentConfig(
            state="RS",
            station_id="A801",
            window_size=4,
            n_clusters=2,
            algorithm="kmeans",
            sigma=None,
        )
        split_data, _splits = create_window_split_data(
            self.df,
            config,
            ["TEMPERATURA_MAXIMA", "TEMPERATURA_MIN", "PRECIPITACAO_TOTAL"],
            clustering_feature_normalize=None,
            clustering_precipitation_normalize=None,
            lstm_feature_normalize=None,
            lstm_precipitation_normalize=None,
            variance_threshold=None,
            forecast_horizon=2,
            train_ratio=0.5,
            val_ratio=0.2,
            random_state=42,
            manual_zero_tolerance=0.0,
        )

        self.assertEqual(len(split_data.y_train), 10)
        self.assertEqual(len(split_data.y_val), 1)
        self.assertEqual(len(split_data.y_test), 4)
        self.assertEqual(split_data.i_train.tolist(), list(range(0, 10)))
        self.assertEqual(split_data.i_val.tolist(), [15])
        self.assertEqual(split_data.i_test.tolist(), list(range(21, 25)))
        np.testing.assert_allclose(
            split_data.current_train,
            [3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0],
        )
        np.testing.assert_allclose(split_data.current_val, [3.0])
        np.testing.assert_allclose(split_data.current_test, [4.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(
            split_data.y_train,
            [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        )
        np.testing.assert_allclose(split_data.y_val, [0.0])
        np.testing.assert_allclose(split_data.y_test, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            split_data.y_train_by_lead_day,
            [
                [4.0, 0.0],
                [0.0, 1.0],
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 0.0],
                [0.0, 1.0],
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
            ],
        )
        np.testing.assert_allclose(split_data.y_val_by_lead_day, [[4.0, 0.0]])
        np.testing.assert_allclose(
            split_data.y_test_by_lead_day,
            [
                [0.0, 1.0],
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
            ],
        )
        np.testing.assert_array_equal(
            split_data.test_target_dates_by_lead_day,
            np.array(
                [
                    ["2025-01-26", "2025-01-27"],
                    ["2025-01-27", "2025-01-28"],
                    ["2025-01-28", "2025-01-29"],
                    ["2025-01-29", "2025-01-30"],
                ],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_allclose(
            split_data.all_current_precipitation,
            [3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0],
        )
        np.testing.assert_allclose(
            split_data.test_targets_by_lead_day,
            [
                [0.0, 1.0],
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
            ],
        )

    def test_forecast_horizon_drops_windows_with_any_missing_lead_day(self) -> None:
        df = self.df.copy()
        df.loc[4, "PRECIPITACAO_TOTAL"] = np.nan
        config = ExperimentConfig(
            state="RS",
            station_id="A801",
            window_size=4,
            n_clusters=2,
            algorithm="kmeans",
            sigma=None,
        )

        split_data, _splits = create_window_split_data(
            df,
            config,
            ["TEMPERATURA_MAXIMA", "TEMPERATURA_MIN"],
            clustering_feature_normalize=None,
            clustering_precipitation_normalize=None,
            lstm_feature_normalize=None,
            lstm_precipitation_normalize=None,
            variance_threshold=None,
            forecast_horizon=2,
            train_ratio=0.5,
            val_ratio=0.2,
            random_state=42,
            manual_zero_tolerance=0.0,
        )

        self.assertEqual(split_data.i_train.tolist(), list(range(1, 10)))
        self.assertTrue(np.isfinite(split_data.y_train_by_lead_day).all())

    def test_held_out_windows_use_nearest_training_centroid(self) -> None:
        config = ExperimentConfig(
            state="RS",
            station_id="A801",
            window_size=2,
            n_clusters=2,
            algorithm="kmeans",
            sigma=None,
        )
        X_train = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [10.0, 0.0],
                [10.2, 0.0],
            ]
        )
        X_val = np.array([[0.1, 0.0], [10.4, 0.0]])
        X_test = np.array([[9.9, 0.0], [0.3, 0.0]])
        y_train = np.array([0.0, 0.0, 5.0, 5.0])

        c_train, c_val, c_test = _cluster_window_splits(
            config,
            X_train,
            X_val,
            X_test,
            y_train,
            random_state=42,
            manual_zero_tolerance=0.0,
        )
        centroids = np.vstack(
            [X_train[c_train == cluster_id].mean(axis=0) for cluster_id in range(2)]
        )

        def nearest_labels(samples: np.ndarray) -> np.ndarray:
            distances_sq = np.sum(
                (samples[:, None, :] - centroids[None, :, :]) ** 2,
                axis=2,
            )
            return np.argmin(distances_sq, axis=1)

        np.testing.assert_array_equal(c_val, nearest_labels(X_val))
        np.testing.assert_array_equal(c_test, nearest_labels(X_test))

    def test_quantile_weighted_mse_config_uses_cluster_rain_values(self) -> None:
        thresholds, weights = quantile_weighted_mse_config(
            np.array([0.0, 0.0, 1.0, 5.0, 20.0, 50.0]),
            quantiles=[0.5, 0.75],
            weights="auto",
        )

        self.assertEqual(thresholds, [3.0, 16.25])
        self.assertEqual(len(weights), len(thresholds) + 1)
        self.assertTrue(all(weight > 0 for weight in weights))
        self.assertGreater(weights[-1], weights[0])


if __name__ == "__main__":
    unittest.main()
