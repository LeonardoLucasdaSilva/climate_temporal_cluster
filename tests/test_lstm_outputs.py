"""Tests for LSTM experiment output helpers."""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from data.lstm_outputs import (
    compressed_time_positions,
    pca_mode_label,
    save_cluster_silhouette_plot,
    save_cluster_prediction_scatters,
    save_cluster_prediction_timeseries,
    save_forecast_horizon_diagnostics,
    save_forecast_lead_day_diagnostics,
    save_prediction_timeseries_splits,
    save_sweep_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LstmOutputTests(unittest.TestCase):
    def test_sweep_summary_records_pca_choice(self) -> None:
        output_dir = PROJECT_ROOT / "tests" / f"_pca_summary_test_{uuid.uuid4().hex}"
        output_dir.mkdir()
        try:
            with (
                patch("data.lstm_outputs.latex_table", return_value="table"),
                patch(
                    "data.lstm_outputs.cluster_metric_latex_tables",
                    return_value="cluster tables",
                ),
            ):
                save_sweep_outputs(
                    results=[
                        {
                            "run_name": "pca_run",
                            "test_rmse": 1.0,
                            "test_mae": 0.5,
                            "pca_variance_threshold": 0.9,
                            "pca_for_clustering_only": True,
                            "pca_mode": "clustering only",
                        }
                    ],
                    sweep_dir=output_dir,
                    state="RS",
                    station_id="A801",
                    window_sizes=[15],
                    n_clusters_list=[3],
                    clustering_algorithm="kmeans",
                    pca_variance_threshold=0.9,
                    pca_for_clustering_only=True,
                    quantitative_metrics=["MSE"],
                )

            summary = (output_dir / "sweep_summary.txt").read_text(encoding="utf-8")
            self.assertIn("PCA variance threshold: 0.90", summary)
            self.assertIn("PCA mode: clustering only", summary)
            self.assertIn("pca_for_clustering_only", summary)
            self.assertIn("pca_mode", summary)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_pca_mode_label_covers_all_modes(self) -> None:
        self.assertEqual(pca_mode_label(None, False), "disabled")
        self.assertEqual(pca_mode_label(None, True), "disabled")
        self.assertEqual(pca_mode_label(0.9, True), "clustering only")
        self.assertEqual(pca_mode_label(0.9, False), "clustering and LSTM")

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

        try:
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
            self.assertFalse(
                (diag_dir / "09_current_vs_forecast_horizon_by_split.png").exists()
            )
            self.assertFalse(
                (diag_dir / "10_test_current_target_prediction_timeseries.png").exists()
            )
            self.assertFalse(
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

    def test_cluster_silhouette_plot_writes_summary_and_figure(self) -> None:
        output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_cluster_silhouette_test_{uuid.uuid4().hex}"
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
                summary = save_cluster_silhouette_plot(
                    {
                        "Training": (
                            np.array(
                                [
                                    [0.0, 0.0],
                                    [0.0, 0.2],
                                    [5.0, 5.0],
                                    [5.0, 5.2],
                                ]
                            ),
                            np.array([0, 0, 1, 1]),
                        ),
                        "Validation": (
                            np.array([[1.0, 1.0], [1.2, 1.2]]),
                            np.array([0, 0]),
                        ),
                    },
                    output_dir,
                )

            diag_dir = output_dir / "cluster_diagnostics"
            self.assertTrue((diag_dir / "08_silhouette_analysis.png").exists())
            self.assertTrue((diag_dir / "silhouette_scores.csv").exists())
            self.assertIn("mean_silhouette", summary.columns)
            self.assertIn("Training", summary["split"].tolist())
            self.assertIn("Validation", summary["split"].tolist())
            training_overall = summary[
                (summary["split"] == "Training") & (summary["cluster"] == "overall")
            ]["mean_silhouette"].iloc[0]
            self.assertGreater(training_overall, 0.0)
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
                    y_pred_test_by_lead_day=np.array(
                        [
                            [0.1, 1.1, 2.1],
                            [1.1, 2.1, 3.1],
                            [2.1, 3.1, 4.1],
                        ]
                    ),
                )

            diag_dir = output_dir / "forecast_horizon_diagnostics"
            lead_df = pd.read_csv(diag_dir / "test_prediction_by_lead_day.csv")
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
            self.assertTrue(metrics_df["is_trained_target_day"].all())
            predicted_day_2 = lead_df[
                (lead_df["window_index"] == 20) & (lead_df["lead_day"] == 2)
            ]["predicted_mm"].iloc[0]
            self.assertAlmostEqual(predicted_day_2, 1.1)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_prediction_timeseries_splits_writes_one_folder_per_lead_day(self) -> None:
        output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_prediction_timeseries_splits_test_{uuid.uuid4().hex}"
        )
        output_dir.mkdir()

        def fake_savefig(_figure: object, path: object, *_args: object, **_kwargs: object) -> None:
            Path(path).write_bytes(b"plot")

        try:
            actual_by_lead_day = np.array(
                [
                    [0.0, 1.0, 2.0],
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                    [3.0, 4.0, 5.0],
                    [4.0, 5.0, 6.0],
                    [5.0, 6.0, 7.0],
                    [6.0, 7.0, 8.0],
                    [7.0, 8.0, 9.0],
                ]
            )
            predicted_by_lead_day = actual_by_lead_day + 0.25

            with patch("matplotlib.figure.Figure.savefig", fake_savefig):
                save_prediction_timeseries_splits(
                    actual_by_lead_day,
                    predicted_by_lead_day,
                    output_dir,
                    n_splits=4,
                    forecast_horizon=3,
                )

            plot_dir = output_dir / "prediction_timeseries_splits"
            for lead_day in range(1, 4):
                lead_dir = plot_dir / f"lead_day_{lead_day:02d}"
                self.assertTrue(lead_dir.is_dir())
                for split_index in range(1, 5):
                    self.assertTrue(
                        (
                            lead_dir
                            / f"02_predictions_timeseries_split_{split_index:02d}_of_04.png"
                        ).exists()
                    )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_prediction_timeseries_splits_uses_target_dates_for_x_axis(self) -> None:
        output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_prediction_timeseries_splits_test_{uuid.uuid4().hex}"
        )
        output_dir.mkdir()
        captured: dict[str, dict[str, object]] = {}

        def fake_savefig(
            figure: object,
            path: object,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            plot_path = Path(path)
            ax = figure.axes[0]
            formatter = ax.xaxis.get_major_formatter()
            captured[plot_path.parent.name] = {
                "xlabel": ax.get_xlabel(),
                "xdata": ax.lines[0].get_xdata(),
                "formatted": formatter(
                    mdates.date2num(pd.Timestamp("2025-01-05").to_pydatetime())
                ),
            }
            plot_path.write_bytes(b"plot")

        try:
            actual_by_lead_day = np.array(
                [
                    [0.0, 1.0],
                    [1.0, 2.0],
                ]
            )
            predicted_by_lead_day = actual_by_lead_day + 0.25
            target_dates_by_lead_day = np.array(
                [
                    ["2025-01-05", "2025-01-06"],
                    ["2025-01-07", "2025-01-08"],
                ],
                dtype="datetime64[ns]",
            )

            with patch("matplotlib.figure.Figure.savefig", fake_savefig):
                save_prediction_timeseries_splits(
                    actual_by_lead_day,
                    predicted_by_lead_day,
                    output_dir,
                    n_splits=1,
                    forecast_horizon=2,
                    test_dates_by_lead_day=target_dates_by_lead_day,
                )

            self.assertEqual(captured["lead_day_01"]["xlabel"], "Target Date")
            self.assertEqual(captured["lead_day_01"]["formatted"], "05/01/2025")
            self.assertEqual(
                str(captured["lead_day_01"]["xdata"][0])[:10],
                "2025-01-05",
            )
            self.assertEqual(
                str(captured["lead_day_02"]["xdata"][0])[:10],
                "2025-01-06",
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_cluster_prediction_timeseries_uses_target_dates_for_x_axis(self) -> None:
        output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_cluster_prediction_timeseries_test_{uuid.uuid4().hex}"
        )
        output_dir.mkdir()
        captured: dict[str, object] = {}

        def fake_savefig(
            figure: object,
            path: object,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            plot_path = Path(path)
            actual_axis, residual_axis = figure.axes
            formatter = residual_axis.xaxis.get_major_formatter()
            captured["xlabel"] = residual_axis.get_xlabel()
            captured["formatted"] = formatter(
                mdates.date2num(pd.Timestamp("2025-01-05").to_pydatetime())
            )
            captured["xdata"] = actual_axis.lines[0].get_xdata()
            plot_path.write_bytes(b"plot")

        try:
            target_dates = np.array(
                ["2025-01-05", "2025-01-12", "2025-01-20"],
                dtype="datetime64[ns]",
            )

            with patch("matplotlib.figure.Figure.savefig", fake_savefig):
                save_cluster_prediction_timeseries(
                    y_test=np.array([1.0, 2.0, 4.0]),
                    y_pred_test=np.array([0.8, 2.2, 3.7]),
                    c_test=np.array([0, 0, 0]),
                    test_indices=np.array([10, 20, 30]),
                    output_dir=output_dir,
                    test_dates=target_dates,
                )

            plot_path = (
                output_dir
                / "cluster_prediction_timeseries"
                / "cluster_0_prediction_timeseries.png"
            )
            self.assertTrue(plot_path.exists())
            self.assertEqual(captured["xlabel"], "Target Date")
            self.assertEqual(captured["formatted"], "05/01/2025")
            self.assertEqual(str(captured["xdata"][0])[:10], "2025-01-05")
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
