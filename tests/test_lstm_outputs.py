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
    save_cluster_distribution_plot,
    save_cluster_silhouette_plot,
    save_cluster_prediction_scatters,
    save_cluster_prediction_timeseries,
    save_forecast_horizon_diagnostics,
    save_forecast_lead_day_diagnostics,
    save_oracle_model_visualizations,
    save_oracle_transfer_diagnostics,
    oracle_routing_diagnostic_tables,
    save_prediction_timeseries_splits,
    save_test_model_selection_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LstmOutputTests(unittest.TestCase):
    def test_oracle_model_mirrors_visualizations_only_with_valid_oracle_outputs(self) -> None:
        output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_oracle_model_plots_test_{uuid.uuid4().hex}"
        )
        output_dir.mkdir()
        test_model_selection = {
            "summary": {"primary_metric": "RMSE"},
            "selected_prediction_by_metric": {"RMSE": np.array([1.0, 2.0])},
            "selected_prediction_by_lead_day": np.array(
                [[0.5, 1.0], [1.5, 2.0]]
            ),
        }

        try:
            with patch("data.lstm_outputs.save_visualizations") as save_plots:
                written = save_oracle_model_visualizations(
                    test_model_selection,
                    y_test=np.array([1.0, 2.0]),
                    c_test=np.array([0, 1]),
                    test_indices=np.array([10, 11]),
                    forecast_horizon_precipitation=np.array([1.0, 2.0]),
                    input_cluster_labels=np.array([0, 1]),
                    histories_by_cluster={},
                    output_dir=output_dir,
                    forecast_horizon=2,
                    batch_size=4,
                    test_targets_by_lead_day=np.array(
                        [[0.5, 1.0], [1.5, 2.0]]
                    ),
                    regular_prediction_by_lead_day=np.array(
                        [[0.4, 0.9], [1.4, 1.9]]
                    ),
                    test_target_dates_by_lead_day=None,
                    cluster_feature_splits=None,
                )

            self.assertTrue(written)
            self.assertTrue((output_dir / "oracle_model").is_dir())
            self.assertEqual(save_plots.call_count, 1)
            args = save_plots.call_args.args
            np.testing.assert_allclose(args[1], [1.0, 2.0])
            self.assertEqual(args[7], output_dir / "oracle_model")
            np.testing.assert_array_equal(
                save_plots.call_args.kwargs["y_pred_test_by_lead_day"],
                [[0.5, 1.0], [1.5, 2.0]],
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_oracle_transfer_outputs_are_separate_from_same_cluster_result(self) -> None:
        output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_oracle_transfer_test_{uuid.uuid4().hex}"
        )
        output_dir.mkdir()
        test_model_selection = {
            "comparison_rows": [],
            "selection_rows": [
                {
                    "sample_index": 0,
                    "test_cluster": 0,
                    "metric": "RMSE",
                    "selected_model_cluster": 1,
                    "selected_is_same_cluster": False,
                    "actual": 10.0,
                    "selected_absolute_error": 1.0,
                    "same_cluster_absolute_error": 5.0,
                },
                {
                    "sample_index": 1,
                    "test_cluster": 0,
                    "metric": "RMSE",
                    "selected_model_cluster": 0,
                    "selected_is_same_cluster": True,
                    "actual": 2.0,
                    "selected_absolute_error": 0.5,
                    "same_cluster_absolute_error": 0.5,
                },
                {
                    "sample_index": 2,
                    "test_cluster": 1,
                    "metric": "RMSE",
                    "selected_model_cluster": 1,
                    "selected_is_same_cluster": True,
                    "actual": 3.0,
                    "selected_absolute_error": 0.25,
                    "same_cluster_absolute_error": 0.25,
                },
            ],
            "metric_summary_rows": [],
            "summary": {
                "primary_metric": "RMSE",
                "original_rmse": 3.0,
                "selected_rmse": 1.0,
                "rmse_improvement": 2.0,
                "original_mae": 2.0,
                "selected_mae": 0.5,
                "mae_improvement": 1.5,
                "original_mse": 9.0,
                "selected_mse": 1.0,
                "mse_improvement": 8.0,
                "mse_improvement_ci_low": 5.0,
                "mse_improvement_ci_high": 10.0,
                "mse_improvement_probability": 0.99,
                "switched_samples": 1,
                "n_test_samples": 3,
            },
            "selected_prediction_by_metric": {"RMSE": np.array([1.0, 2.0, 3.0])},
            "selected_model_by_metric": {"RMSE": np.array([1, 0, 1])},
        }

        def fake_savefig(_figure: object, path: object, *_args: object, **_kwargs: object) -> None:
            Path(path).write_bytes(b"plot")

        try:
            save_test_model_selection_report(output_dir, test_model_selection)
            with patch("matplotlib.figure.Figure.savefig", fake_savefig):
                save_oracle_transfer_diagnostics(
                    np.array([1.0, 2.0, 3.0]),
                    test_model_selection,
                    output_dir,
                )

            transfer_matrix = pd.read_csv(
                output_dir / "oracle_model_selection_matrix.csv"
            )
            self.assertEqual(
                transfer_matrix.columns.tolist(),
                ["assigned_test_cluster", "LSTM_0", "LSTM_1"],
            )
            self.assertEqual(transfer_matrix["LSTM_1"].tolist(), [1, 1])
            summary_text = (output_dir / "oracle_vs_same_cluster_summary.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("metricas, test_predictions.csv", summary_text)
            diagnostics_dir = output_dir / "oracle_model_selection_diagnostics"
            self.assertTrue(
                (diagnostics_dir / "01_oracle_predictions_vs_actual.png").exists()
            )
            self.assertTrue(
                (diagnostics_dir / "02_oracle_model_transfer_matrix.png").exists()
            )
            self.assertTrue(
                (
                    diagnostics_dir
                    / "03_oracle_switch_rate_by_assigned_cluster.png"
                ).exists()
            )
            self.assertTrue(
                (diagnostics_dir / "04_oracle_mae_by_assigned_cluster.png").exists()
            )
            self.assertTrue(
                (
                    diagnostics_dir
                    / "05_oracle_error_improvement_distribution.png"
                ).exists()
            )
            self.assertTrue((output_dir / "oracle_cluster_routing_summary.csv").exists())
            self.assertTrue((output_dir / "oracle_cluster_pair_summary.csv").exists())
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_oracle_routing_tables_quantify_cluster_switch_gains(self) -> None:
        selection_df = pd.DataFrame(
            [
                {
                    "test_cluster": 0,
                    "metric": "RMSE",
                    "selected_model_cluster": 1,
                    "selected_is_same_cluster": False,
                    "actual": 10.0,
                    "selected_absolute_error": 1.0,
                    "same_cluster_absolute_error": 5.0,
                },
                {
                    "test_cluster": 0,
                    "metric": "RMSE",
                    "selected_model_cluster": 0,
                    "selected_is_same_cluster": True,
                    "actual": 2.0,
                    "selected_absolute_error": 0.5,
                    "same_cluster_absolute_error": 0.5,
                },
                {
                    "test_cluster": 1,
                    "metric": "RMSE",
                    "selected_model_cluster": 1,
                    "selected_is_same_cluster": True,
                    "actual": 3.0,
                    "selected_absolute_error": 0.25,
                    "same_cluster_absolute_error": 0.25,
                },
            ]
        )

        cluster_summary, pair_summary = oracle_routing_diagnostic_tables(
            selection_df,
            "RMSE",
        )

        cluster_zero = cluster_summary.loc[
            cluster_summary["assigned_test_cluster"] == 0
        ].iloc[0]
        self.assertEqual(cluster_zero["n_test"], 2)
        self.assertEqual(cluster_zero["oracle_switched_samples"], 1)
        self.assertAlmostEqual(cluster_zero["oracle_switched_percent"], 50.0)
        self.assertAlmostEqual(cluster_zero["mae_improvement"], 2.0)
        transfer = pair_summary.loc[
            (pair_summary["assigned_test_cluster"] == 0)
            & (pair_summary["oracle_selected_model_cluster"] == 1)
        ].iloc[0]
        self.assertAlmostEqual(transfer["percent_of_assigned_cluster"], 50.0)
        self.assertAlmostEqual(transfer["mae_improvement"], 4.0)

    def test_cluster_distribution_includes_training_batch_statistics(self) -> None:
        output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_cluster_distribution_test_{uuid.uuid4().hex}"
        )
        output_dir.mkdir()
        captured: dict[str, object] = {}

        def fake_savefig(
            figure: object,
            path: object,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            captured["figure"] = figure
            captured["path"] = Path(path)
            Path(path).write_bytes(b"plot")

        try:
            with patch("matplotlib.figure.Figure.savefig", fake_savefig):
                statistics = save_cluster_distribution_plot(
                    c_test=np.array([0, 0, 1]),
                    output_dir=output_dir,
                    c_train=np.array([0, 0, 0, 1, 1]),
                    c_val=np.array([0, 1]),
                    batch_size=2,
                )

            self.assertIsNotNone(statistics)
            assert statistics is not None
            self.assertEqual(
                statistics["optimizer_steps_per_epoch"].tolist(),
                [2, 1],
            )
            csv_path = (
                output_dir
                / "cluster_diagnostics"
                / "cluster_training_batch_statistics.csv"
            )
            csv_df = pd.read_csv(csv_path)
            self.assertEqual(csv_df["n_train"].tolist(), [3, 2])
            self.assertEqual(csv_df["optimizer_steps_per_epoch"].tolist(), [2, 1])
            self.assertEqual(
                captured["path"],
                output_dir / "cluster_diagnostics" / "06_cluster_distribution.png",
            )

            figure = captured["figure"]
            table = figure.axes[1].tables[0]
            cells = table.get_celld()
            self.assertEqual(cells[(1, 1)].get_text().get_text(), "3")
            self.assertEqual(cells[(1, 2)].get_text().get_text(), "2")
            self.assertEqual(cells[(2, 1)].get_text().get_text(), "2")
            self.assertEqual(cells[(2, 2)].get_text().get_text(), "1")
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

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
