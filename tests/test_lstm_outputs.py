"""Tests for LSTM experiment output helpers."""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from data.lstm_outputs import (
    compressed_time_positions,
    save_cluster_prediction_scatters,
    save_forecast_horizon_diagnostics,
    save_forecast_lead_day_diagnostics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LstmOutputTests(unittest.TestCase):
    def test_compressed_time_positions_preserves_small_gaps(self) -> None:
        positions, compressed = compressed_time_positions(
            np.array([2, 3, 6, 9]),
            max_gap=4,
        )
        np.testing.assert_allclose(positions, [0.0, 1.0, 4.0, 7.0])
        np.testing.assert_array_equal(compressed, [False, False, False])

    def test_compressed_time_positions_shortens_large_gaps(self) -> None:
        positions, compressed = compressed_time_positions(
            np.array([2, 3, 30, 32, 100]),
            max_gap=5,
        )
        np.testing.assert_allclose(positions, [0.0, 1.0, 6.0, 8.0, 13.0])
        np.testing.assert_array_equal(compressed, [False, True, False, True])

    def test_forecast_horizon_diagnostics_compare_current_and_target(self) -> None:
        output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_forecast_horizon_diagnostics_test_{uuid.uuid4().hex}"
        )
        output_dir.mkdir()

        def fake_savefig(
            _figure: object,
            path: object,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            Path(path).write_bytes(b"plot")

        try:
            with patch("matplotlib.figure.Figure.savefig", fake_savefig):
                summary = save_forecast_horizon_diagnostics(
                    y_train=np.array([1.0, 3.0]),
                    y_val=np.array([2.0]),
                    y_test=np.array([4.0, 0.0, 3.0]),
                    current_train=np.array([0.0, 2.0]),
                    current_val=np.array([1.0]),
                    current_test=np.array([2.0, 1.0, 3.0]),
                    y_pred_test=np.array([3.5, 0.2, 2.5]),
                    c_test=np.array([0, 1, 0]),
                    train_indices=np.array([0, 1]),
                    val_indices=np.array([10]),
                    test_indices=np.array([20, 21, 22]),
                    output_dir=output_dir,
                    forecast_horizon=2,
                )

            diag_dir = output_dir / "forecast_horizon_diagnostics"
            self.assertTrue(
                (diag_dir / "forecast_horizon_behavior_report.txt").exists()
            )
            self.assertTrue(
                (diag_dir / "09_current_vs_forecast_horizon_by_split.png").exists()
            )
            self.assertTrue(
                (diag_dir / "10_test_current_target_prediction_timeseries.png").exists()
            )
            self.assertTrue(
                (diag_dir / "11_test_current_vs_horizon_by_cluster.png").exists()
            )

            all_df = pd.read_csv(
                diag_dir / "current_vs_forecast_horizon_precipitation.csv"
            )
            test_df = pd.read_csv(diag_dir / "test_forecast_horizon_behavior.csv")
            self.assertIn("current_window_precipitation_mm", all_df.columns)
            self.assertIn("forecast_horizon_precipitation_mm", test_df.columns)
            self.assertEqual(summary["forecast_horizon"], 2.0)
            self.assertGreater(
                summary["lstm_rmse_improvement_vs_persistence"],
                0.0,
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_cluster_prediction_scatters_write_one_plot_per_cluster(self) -> None:
        output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_cluster_prediction_scatter_test_{uuid.uuid4().hex}"
        )
        output_dir.mkdir()

        def fake_savefig(
            _figure: object,
            path: object,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            Path(path).write_bytes(b"plot")

        try:
            with patch("matplotlib.figure.Figure.savefig", fake_savefig):
                save_cluster_prediction_scatters(
                    y_test=np.array([0.2, 7.0, 5.0]),
                    y_pred_test=np.array([0.3, 6.6, 5.4]),
                    c_test=np.array([0, 1, 1]),
                    output_dir=output_dir,
                )

            scatter_dir = output_dir / "cluster_prediction_scatter"
            self.assertTrue(
                (scatter_dir / "cluster_0_predicted_vs_actual_scatter.png").exists()
            )
            self.assertTrue(
                (scatter_dir / "cluster_1_predicted_vs_actual_scatter.png").exists()
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_forecast_lead_day_diagnostics_writes_per_day_outputs(self) -> None:
        output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_forecast_lead_day_diagnostics_test_{uuid.uuid4().hex}"
        )
        output_dir.mkdir()

        def fake_savefig(_figure: object, path: object, *_args: object, **_kwargs: object) -> None:
            Path(path).write_bytes(b"plot")

        try:
            with patch("matplotlib.figure.Figure.savefig", fake_savefig):
                metrics_df = save_forecast_lead_day_diagnostics(
                    y_pred_test=np.array([0.5, 1.5, 2.5]),
                    c_test=np.array([0, 1, 0]),
                    test_indices=np.array([20, 21, 22]),
                    test_targets_by_lead_day=np.array(
                        [
                            [0.0, 1.0, 2.0],
                            [1.0, 2.0, 3.0],
                            [2.0, 3.0, 4.0],
                        ]
                    ),
                    output_dir=output_dir,
                    forecast_horizon=3,
                )

            diag_dir = output_dir / "forecast_horizon_diagnostics"
            self.assertTrue((diag_dir / "test_prediction_by_lead_day.csv").exists())
            self.assertTrue(
                (diag_dir / "test_prediction_metrics_by_lead_day.csv").exists()
            )
            self.assertTrue(
                (diag_dir / "12_prediction_error_by_lead_day.png").exists()
            )
            self.assertTrue(
                (diag_dir / "13_true_vs_predicted_by_lead_day.png").exists()
            )
            self.assertTrue(
                (diag_dir / "14_prediction_vs_actual_timeseries_by_lead_day.png").exists()
            )
            self.assertTrue(
                (
                    diag_dir
                    / "true_vs_predicted_by_lead_day"
                    / "true_vs_predicted_lead_day_03.png"
                ).exists()
            )
            self.assertEqual(metrics_df["lead_day"].tolist(), [1, 2, 3])
            self.assertTrue(bool(metrics_df.iloc[-1]["is_trained_target_day"]))
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
