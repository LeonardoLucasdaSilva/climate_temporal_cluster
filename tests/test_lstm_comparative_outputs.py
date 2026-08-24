"""Tests for sweep-level LSTM comparative outputs."""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from data.lstm_comparative_outputs import (
    ComparativeRunData,
    _save_metric_comparison_plots,
    _save_timeseries_comparison_plots,
    _weighted_history_dataframe,
    comparative_overall_metrics_dataframe,
    normalize_pivot_parameter,
    render_report_compare,
    save_comparative_outputs,
    validate_comparative_pivot,
)
from methods.lstm_cluster.pipeline import (
    _execute_configuration_jobs,
    _normalize_learning_rate_values,
    build_configurations,
    run_experiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LstmComparativeOutputTests(unittest.TestCase):
    def test_parallel_configuration_jobs_preserve_sweep_order(self) -> None:
        configurations = build_configurations(
            [None],
            state="RS",
            station_id="A801",
            window_sizes=[5, 10],
            n_clusters_list=[2],
            clustering_algorithm="kmeans",
        )
        jobs = [
            ((None, config), {"marker": index})
            for index, config in enumerate(configurations)
        ]
        submitted: list[object] = []
        observed_worker_counts: list[int] = []

        class ImmediateFuture:
            def __init__(self, result: object) -> None:
                self._result = result

            def result(self) -> object:
                return self._result

        class ImmediateExecutor:
            def __init__(self, max_workers: int) -> None:
                observed_worker_counts.append(max_workers)

            def __enter__(self) -> ImmediateExecutor:
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def submit(self, function: object, *args: object) -> ImmediateFuture:
                submitted.append(args)
                return ImmediateFuture(function(*args))

        def fake_process(
            args: tuple[object, ...],
            kwargs: dict[str, object],
            _plot_style: object,
        ) -> dict[str, object]:
            config = args[1]
            return {
                "run_name": config.name,
                "marker": kwargs["marker"],
            }

        with (
            patch(
                "methods.lstm_cluster.pipeline.ProcessPoolExecutor",
                ImmediateExecutor,
            ),
            patch(
                "methods.lstm_cluster.pipeline.as_completed",
                side_effect=lambda futures: reversed(list(futures)),
            ),
            patch(
                "methods.lstm_cluster.pipeline._run_configuration_process",
                side_effect=fake_process,
            ),
            patch("methods.lstm_cluster.pipeline.os.cpu_count", return_value=8),
        ):
            results = _execute_configuration_jobs(
                jobs,
                parallel=True,
                show_console_info=False,
            )

        self.assertEqual(observed_worker_counts, [2])
        self.assertEqual(len(submitted), 2)
        self.assertEqual([result["marker"] for result in results], [0, 1])
        self.assertEqual(
            [result["run_name"] for result in results],
            [config.name for config in configurations],
        )

    def _run(
        self,
        *,
        name: str,
        window_size: int,
        n_clusters: int = 2,
        dates: list[str],
        actual: list[float],
        predicted: list[float],
        epochs: int,
    ) -> ComparativeRunData:
        history = {
            "loss": np.linspace(2.0, 1.0, epochs).tolist(),
            "val_loss": np.linspace(2.2, 1.2, max(epochs - 1, 1)).tolist(),
            "mse": np.linspace(2.0, 1.0, epochs).tolist(),
            "val_mse": np.linspace(2.2, 1.2, max(epochs - 1, 1)).tolist(),
            "mae": np.linspace(1.0, 0.5, epochs).tolist(),
            "val_mae": np.linspace(1.1, 0.6, max(epochs - 1, 1)).tolist(),
            "r2": np.linspace(0.1, 0.7, epochs).tolist(),
            "val_r2": np.linspace(0.0, 0.6, max(epochs - 1, 1)).tolist(),
        }
        return ComparativeRunData(
            run_name=name,
            parameters={
                "window_size": window_size,
                "n_clusters": n_clusters,
                "sigma": 1.0,
                "learning_rate": 0.001,
            },
            result_metrics={"test_rmse": 0.5},
            actual_by_lead_day=np.asarray(actual, dtype=float).reshape(-1, 1),
            predicted_by_lead_day=np.asarray(predicted, dtype=float).reshape(-1, 1),
            target_dates_by_lead_day=np.asarray(
                dates,
                dtype="datetime64[ns]",
            ).reshape(-1, 1),
            histories_by_cluster={0: history, 1: history},
            cluster_train_counts={0: 3, 1: 1},
        )

    def test_report_compare_cluster_step_skips_silhouette_for_single_cluster(
        self,
    ) -> None:
        output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_comparative_k1_cluster_step_test_{uuid.uuid4().hex}"
        )
        run = self._run(
            name="k1",
            window_size=10,
            n_clusters=1,
            dates=["2025-01-01", "2025-01-02"],
            actual=[0.0, 1.0],
            predicted=[0.1, 1.1],
            epochs=2,
        )
        metrics = pd.DataFrame(
            [
                {
                    "run_name": "k1",
                    "pivot_parameter": "window_size",
                    "pivot_value": 10,
                    "lead_day": 1,
                    "n_common_test_dates": 2,
                    "start_date": "2025-01-01",
                    "end_date": "2025-01-02",
                    "MSE": 0.1,
                    "RMSE": 0.316,
                    "MAE": 0.2,
                    "R2": 0.9,
                }
            ]
        )

        try:
            diagnostics_dir = output_dir / "k1" / "cluster_diagnostics"
            diagnostics_dir.mkdir(parents=True)
            (diagnostics_dir / "05_cluster_performance.png").write_bytes(b"plot")
            (diagnostics_dir / "08_silhouette_analysis.png").write_bytes(b"plot")

            report_tex = render_report_compare(
                output_dir / "comparative_analysis",
                output_dir,
                "window_size",
                metrics,
                [run],
            )

            self.assertIn("05_cluster_performance.png", report_tex)
            self.assertNotIn("08_silhouette_analysis.png", report_tex)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_report_compare_cluster_step_keeps_cluster_performance_per_run_when_cluster_count_is_fixed(
        self,
    ) -> None:
        output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_comparative_fixed_k_cluster_step_test_{uuid.uuid4().hex}"
        )
        run_a = self._run(
            name="w10",
            window_size=10,
            n_clusters=2,
            dates=["2025-01-01", "2025-01-02"],
            actual=[0.0, 1.0],
            predicted=[0.1, 1.1],
            epochs=2,
        )
        run_b = self._run(
            name="w20",
            window_size=20,
            n_clusters=2,
            dates=["2025-01-01", "2025-01-02"],
            actual=[0.0, 1.0],
            predicted=[0.2, 1.2],
            epochs=2,
        )
        metrics = pd.DataFrame(
            [
                {
                    "run_name": run.run_name,
                    "pivot_parameter": "window_size",
                    "pivot_value": run.parameters["window_size"],
                    "lead_day": 1,
                    "n_common_test_dates": 2,
                    "start_date": "2025-01-01",
                    "end_date": "2025-01-02",
                    "MSE": 0.1,
                    "RMSE": 0.316,
                    "MAE": 0.2,
                    "R2": 0.9,
                }
                for run in (run_a, run_b)
            ]
        )

        try:
            for run_name in ("w10", "w20"):
                diagnostics_dir = output_dir / run_name / "cluster_diagnostics"
                diagnostics_dir.mkdir(parents=True)
                (diagnostics_dir / "05_cluster_performance.png").write_bytes(b"plot")
                (
                    diagnostics_dir / "06_cluster_distribution.png"
                ).write_bytes(b"plot")

            report_tex = render_report_compare(
                output_dir / "comparative_analysis",
                output_dir,
                "window_size",
                metrics,
                [run_a, run_b],
            )

            self.assertIn(
                "../w10/cluster_diagnostics/05_cluster_performance.png",
                report_tex,
            )
            self.assertIn(
                "../w10/cluster_diagnostics/06_cluster_distribution.png",
                report_tex,
            )
            self.assertIn(
                "../w20/cluster_diagnostics/05_cluster_performance.png",
                report_tex,
            )
            self.assertNotIn(
                "../w20/cluster_diagnostics/06_cluster_distribution.png",
                report_tex,
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_pivot_aliases_and_constant_validation(self) -> None:
        self.assertEqual(normalize_pivot_parameter("K"), "n_clusters")
        self.assertEqual(normalize_pivot_parameter("learning rates"), "learning_rate")
        self.assertEqual(
            validate_comparative_pivot(
                [{"n_clusters": 2}, {"n_clusters": 3}],
                "K",
            ),
            "n_clusters",
        )
        with self.assertRaisesRegex(ValueError, "constant"):
            validate_comparative_pivot(
                [{"learning_rate": 0.001}, {"learning_rate": 0.001}],
                "learning_rate",
            )
        with self.assertRaisesRegex(ValueError, "Available parameters"):
            validate_comparative_pivot(
                [{"window_size": 10}],
                "not_a_parameter",
            )

    def test_outputs_align_runs_by_common_target_dates(self) -> None:
        output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_comparative_outputs_test_{uuid.uuid4().hex}"
        )
        run_a = self._run(
            name="w10",
            window_size=10,
            dates=["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            actual=[0.0, 1.0, 2.0, 3.0],
            predicted=[0.1, 1.1, 1.8, 2.9],
            epochs=4,
        )
        run_b = self._run(
            name="w20",
            window_size=20,
            dates=["2025-01-02", "2025-01-03", "2025-01-04"],
            actual=[1.0, 2.0, 3.0],
            predicted=[0.9, 2.2, 3.1],
            epochs=2,
        )

        def fake_savefig(
            _figure: object,
            path: object,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            Path(path).write_bytes(b"plot")

        try:
            output_dir.mkdir()
            for run_name in ("w10", "w20"):
                diagnostics_dir = (
                    output_dir / run_name / "cluster_diagnostics"
                )
                diagnostics_dir.mkdir(parents=True)
                (
                    diagnostics_dir / "05_cluster_performance.png"
                ).write_bytes(b"cluster plot")
            preexisting_comparison_dir = output_dir / "comparative_analysis"
            preexisting_comparison_dir.mkdir()
            stale_plot = (
                preexisting_comparison_dir
                / "04_test_metrics_vs_old_pivot_lead_day_01.png"
            )
            unrelated_file = preexisting_comparison_dir / "research_notes.txt"
            stale_plot.write_bytes(b"stale")
            unrelated_file.write_text("keep", encoding="utf-8")
            with patch("matplotlib.figure.Figure.savefig", fake_savefig):
                comparison_dir = save_comparative_outputs(
                    [run_a, run_b],
                    output_dir,
                    "window_size",
                )

            expected_files = (
                "01_test_timeseries_comparison_lead_day_01.png",
                "02_test_scatter_comparison_lead_day_01.png",
                "03_training_history_comparison.png",
                "04_test_metrics_vs_window_size_lead_day_01.png",
                "05_overall_test_metrics_vs_window_size.png",
                "test_predictions_comparison.csv",
                "aligned_test_predictions.csv",
                "training_history_comparison.csv",
                "comparative_metrics.csv",
                "comparison_manifest.csv",
                "comparison_summary.txt",
                "report_compare.tex",
            )
            for filename in expected_files:
                self.assertTrue((comparison_dir / filename).exists(), filename)

            aligned = pd.read_csv(comparison_dir / "aligned_test_predictions.csv")
            self.assertEqual(len(aligned), 6)
            self.assertEqual(
                sorted(aligned["target_date"].unique().tolist()),
                ["2025-01-02", "2025-01-03", "2025-01-04"],
            )
            metrics = pd.read_csv(comparison_dir / "comparative_metrics.csv")
            self.assertEqual(metrics["n_common_test_dates"].tolist(), [3, 3])
            self.assertEqual(metrics["pivot_value"].tolist(), [10, 20])
            np.testing.assert_allclose(metrics["MSE"].to_numpy(), [0.02, 0.02])
            histories = pd.read_csv(
                comparison_dir / "training_history_comparison.csv"
            )
            self.assertEqual(histories["epoch"].max(), 4)
            self.assertFalse(stale_plot.exists())
            self.assertTrue(unrelated_file.exists())
            self.assertIn(
                "Alignment: intersection",
                (comparison_dir / "comparison_summary.txt").read_text(
                    encoding="utf-8"
                ),
            )
            report_tex = (comparison_dir / "report_compare.tex").read_text(
                encoding="utf-8"
            )
            self.assertIn(r"\usepackage[hidelinks]{hyperref}", report_tex)
            self.assertIn(r"\tableofcontents", report_tex)
            self.assertIn(r"\section{Cluster Step}", report_tex)
            self.assertIn(r"\section{Prediction Time Series}", report_tex)
            self.assertIn(r"\subsection{day 1}", report_tex)
            self.assertIn(r"\section{Prediction Scatter plot}", report_tex)
            self.assertIn(r"\section{Training History}", report_tex)
            self.assertIn(r"\section{Test Metrics}", report_tex)
            self.assertIn(
                "01_test_timeseries_comparison_lead_day_01.png",
                report_tex,
            )
            self.assertIn(
                "02_test_scatter_comparison_lead_day_01.png",
                report_tex,
            )
            self.assertIn(
                "04_test_metrics_vs_window_size_lead_day_01.png",
                report_tex,
            )
            self.assertIn(
                "05_overall_test_metrics_vs_window_size.png",
                report_tex,
            )
            self.assertIn(
                r"\detokenize{01_test_timeseries_comparison_lead_day_01.png}",
                report_tex,
            )
            self.assertIn(
                r"\detokenize{../w10/cluster_diagnostics/05_cluster_performance.png}",
                report_tex,
            )
            self.assertIn(r"w10 \textendash{} 05 Cluster Performance", report_tex)
            self.assertNotIn(" -- ", report_tex)
            self.assertNotIn(" - ", report_tex)
            self.assertIn("Lead Day", report_tex)
            self.assertIn("WINDOW\\_SIZE", report_tex)
            self.assertNotIn("run\\_name", report_tex)
            self.assertIn(r"$t_0$", report_tex)
            self.assertIn(r"$t_f$", report_tex)
            self.assertNotIn("start\\_date", report_tex)
            self.assertNotIn("end\\_date", report_tex)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_timeseries_comparison_uses_subdued_seaborn_lines(self) -> None:
        output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_comparative_timeseries_style_test_{uuid.uuid4().hex}"
        )
        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        aligned_predictions = pd.DataFrame(
            [
                {
                    "run_name": run_name,
                    "pivot_value": window_size,
                    "lead_day": 1,
                    "target_date": target_date,
                    "actual_mm": actual_mm,
                    "predicted_mm": predicted_mm,
                }
                for run_name, window_size, predictions in (
                    ("w10", 10, [0.2, 1.2, 1.8]),
                    ("w20", 20, [0.1, 0.8, 2.1]),
                )
                for target_date, actual_mm, predicted_mm in zip(
                    dates,
                    [0.0, 1.0, 2.0],
                    predictions,
                )
            ]
        )

        def fake_lineplot(*_args: object, **kwargs: object) -> object:
            return kwargs["ax"]

        def fake_savefig(
            _figure: object,
            path: object,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            Path(path).write_bytes(b"plot")

        try:
            output_dir.mkdir()
            with (
                patch(
                    "data.lstm_comparative_outputs.sns.lineplot",
                    side_effect=fake_lineplot,
                ) as lineplot,
                patch(
                    "matplotlib.axes.Axes.plot",
                    side_effect=AssertionError(
                        "Comparative time-series curves must use seaborn."
                    ),
                ),
                patch("matplotlib.figure.Figure.savefig", fake_savefig),
            ):
                _save_timeseries_comparison_plots(
                    aligned_predictions,
                    output_dir,
                    "window_size",
                    n_splits=1,
                )

            self.assertEqual(lineplot.call_count, 2)
            calls_by_value = {
                call.kwargs["y"]: call.kwargs
                for call in lineplot.call_args_list
            }
            observed = calls_by_value["actual_mm"]
            predicted = calls_by_value["predicted_mm"]

            self.assertEqual(len(observed["data"]), 3)
            self.assertEqual(observed["data"]["target_date"].nunique(), 3)
            self.assertEqual(len(predicted["data"]), 6)
            self.assertEqual(predicted["hue"], "run_name")
            self.assertEqual(predicted["hue_order"], ["w10", "w20"])
            self.assertIsNone(observed["estimator"])
            self.assertIsNone(predicted["estimator"])
            self.assertFalse(observed["sort"])
            self.assertFalse(predicted["sort"])
            self.assertNotIn(
                str(observed["color"]).lower(),
                {"black", "#000000"},
            )
            self.assertLess(observed["linewidth"], 2.2)
            self.assertLess(predicted["linewidth"], 1.5)
            self.assertGreater(observed["alpha"], 0.0)
            self.assertLess(observed["alpha"], 1.0)
            self.assertGreater(predicted["alpha"], observed["alpha"])
            self.assertGreater(predicted["zorder"], observed["zorder"])
            self.assertTrue(
                (
                    output_dir
                    / "01_test_timeseries_comparison_lead_day_01.png"
                ).exists()
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_report_compare_metrics_table_uses_readable_pivot_header(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "run_name": "k2",
                    "pivot_parameter": "n_clusters",
                    "pivot_value": 2,
                    "lead_day": 1,
                    "n_common_test_dates": 3,
                    "start_date": "2025-01-01",
                    "end_date": "2025-01-03",
                    "MSE": 0.1,
                    "RMSE": 0.316,
                    "MAE": 0.2,
                    "R2": 0.9,
                },
                {
                    "run_name": "k3",
                    "pivot_parameter": "n_clusters",
                    "pivot_value": 3,
                    "lead_day": 2,
                    "n_common_test_dates": 3,
                    "start_date": "2025-01-02",
                    "end_date": "2025-01-04",
                    "MSE": 0.2,
                    "RMSE": 0.447,
                    "MAE": 0.3,
                    "R2": -0.1,
                }
            ]
        )

        report_tex = render_report_compare(
            PROJECT_ROOT / "tests" / "comparative_analysis",
            PROJECT_ROOT / "tests",
            "n_clusters",
            metrics,
            [],
        )

        self.assertIn(
            r"Lead Day & K & N Common Test Dates & $t_0$ & $t_f$",
            report_tex,
        )
        self.assertIn("01/01/2025 & 03/01/2025", report_tex)
        self.assertIn(r"\textbf{0.9}", report_tex)
        self.assertNotIn("run\\_name", report_tex)
        self.assertNotIn("start\\_date", report_tex)
        self.assertNotIn("end\\_date", report_tex)

    def test_report_compare_adds_compact_lead_day_metric_summary(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "run_name": run_name,
                    "pivot_parameter": "window_size",
                    "pivot_value": pivot_value,
                    "lead_day": lead_day,
                    "n_common_test_dates": 3,
                    "start_date": "2025-01-01",
                    "end_date": "2025-01-03",
                    "MSE": float(lead_day),
                    "RMSE": lead_day + pivot_value / 100.0,
                    "MAE": lead_day / 10.0,
                    "R2": 1.0 - lead_day / 10.0,
                }
                for run_name, pivot_value in (("w10", 10), ("w20", 20))
                for lead_day in range(1, 6)
            ]
        )

        report_tex = render_report_compare(
            PROJECT_ROOT / "tests" / "comparative_analysis",
            PROJECT_ROOT / "tests",
            "window_size",
            metrics,
            [],
        )

        self.assertIn(r"\caption*{Lead-Day Metric Summary}", report_tex)
        self.assertIn(
            r"\caption*{Overall Test Metrics Across Forecast Horizon}",
            report_tex,
        )
        self.assertIn(r"\multicolumn{3}{c}{D+5}", report_tex)
        self.assertIn("Run & WINDOW\\_SIZE", report_tex)
        self.assertIn("RMSE & MAE & R2", report_tex)
        self.assertIn(
            r"w10 & 10 & \textbf{1.1} & \textbf{0.1} & \textbf{0.9}",
            report_tex,
        )
        self.assertIn("w20 & 20", report_tex)
        self.assertNotIn(" & MSE", report_tex)

    def test_overall_metrics_pool_all_forecast_days(self) -> None:
        aligned_predictions = pd.DataFrame(
            [
                {
                    "run_name": "w10",
                    "pivot_value": 10,
                    "lead_day": lead_day,
                    "actual_mm": actual,
                    "predicted_mm": predicted,
                }
                for lead_day, actual, predicted in (
                    (1, 1.0, 2.0),
                    (2, 1.0, 1.0),
                )
            ]
        )

        overall = comparative_overall_metrics_dataframe(
            aligned_predictions,
            "window_size",
        )

        self.assertEqual(overall["forecast_days"].tolist(), [2])
        self.assertEqual(overall["n_common_test_points"].tolist(), [2])
        self.assertAlmostEqual(overall["RMSE"].iloc[0], 2**-0.5)
        self.assertAlmostEqual(overall["MAE"].iloc[0], 0.5)

    def test_metric_comparison_plot_has_three_side_by_side_panels(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "lead_day": 1,
                    "pivot_value": pivot_value,
                    "RMSE": 1.0,
                    "MAE": 0.5,
                    "R2": 0.2,
                }
                for pivot_value in (10, 20)
            ]
        )
        saved_figures: list[object] = []

        def fake_savefig(figure: object, *_args: object, **_kwargs: object) -> None:
            saved_figures.append(figure)

        with patch("matplotlib.figure.Figure.savefig", fake_savefig):
            _save_metric_comparison_plots(
                metrics,
                PROJECT_ROOT / "tests",
                "window_size",
            )

        self.assertEqual(len(saved_figures), 1)
        self.assertEqual(len(saved_figures[0].axes), 3)

    def test_history_aggregation_uses_fixed_clusters_and_common_epochs(self) -> None:
        histories = pd.DataFrame(
            [
                {
                    "run_name": "run",
                    "pivot_parameter": "window_size",
                    "pivot_value": 10,
                    "cluster": cluster,
                    "cluster_train_count": weight,
                    "epoch": epoch,
                    "metric": "loss",
                    "split": "train",
                    "value": value,
                }
                for cluster, weight, values in (
                    (0, 3, [10.0, 8.0, 6.0]),
                    (1, 1, [2.0, 4.0]),
                )
                for epoch, value in enumerate(values, start=1)
            ]
        )

        aggregated = _weighted_history_dataframe(histories)

        self.assertEqual(aggregated["epoch"].tolist(), [1, 2])
        np.testing.assert_allclose(aggregated["value"].to_numpy(), [8.0, 7.0])
        self.assertEqual(aggregated["contributing_clusters"].tolist(), [2, 2])

    def test_conflicting_actual_values_on_same_date_are_rejected(self) -> None:
        run_a = self._run(
            name="w10",
            window_size=10,
            dates=["2025-01-01", "2025-01-02"],
            actual=[0.0, 1.0],
            predicted=[0.0, 1.0],
            epochs=2,
        )
        run_b = self._run(
            name="w20",
            window_size=20,
            dates=["2025-01-01", "2025-01-02"],
            actual=[0.5, 1.0],
            predicted=[0.5, 1.0],
            epochs=2,
        )
        with self.assertRaisesRegex(ValueError, "Conflicting real precipitation"):
            save_comparative_outputs(
                [run_a, run_b],
                PROJECT_ROOT / "tests" / "unused_comparative_output",
                "window_size",
            )

    def test_runs_without_common_dates_are_rejected(self) -> None:
        run_a = self._run(
            name="w10",
            window_size=10,
            dates=["2025-01-01", "2025-01-02"],
            actual=[0.0, 1.0],
            predicted=[0.0, 1.0],
            epochs=2,
        )
        run_b = self._run(
            name="w20",
            window_size=20,
            dates=["2025-02-01", "2025-02-02"],
            actual=[0.0, 1.0],
            predicted=[0.0, 1.0],
            epochs=2,
        )
        with self.assertRaisesRegex(ValueError, "no common target dates"):
            save_comparative_outputs(
                [run_a, run_b],
                PROJECT_ROOT / "tests" / "unused_comparative_output",
                "window_size",
            )

    def test_multi_output_horizon_writes_each_lead_day(self) -> None:
        output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_comparative_multi_lead_test_{uuid.uuid4().hex}"
        )
        base_a = self._run(
            name="w10",
            window_size=10,
            dates=["2025-01-01", "2025-01-02", "2025-01-03"],
            actual=[0.0, 1.0, 2.0],
            predicted=[0.1, 0.9, 2.1],
            epochs=2,
        )
        base_b = self._run(
            name="w20",
            window_size=20,
            dates=["2025-01-01", "2025-01-02", "2025-01-03"],
            actual=[0.0, 1.0, 2.0],
            predicted=[0.2, 1.1, 1.9],
            epochs=2,
        )
        actual = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        dates = np.array(
            [
                ["2025-01-01", "2025-01-02"],
                ["2025-01-02", "2025-01-03"],
                ["2025-01-03", "2025-01-04"],
            ],
            dtype="datetime64[ns]",
        )

        def with_lead_days(
            base: ComparativeRunData,
            predicted: np.ndarray,
        ) -> ComparativeRunData:
            return ComparativeRunData(
                base.run_name,
                base.parameters,
                base.result_metrics,
                actual,
                predicted,
                dates,
                base.histories_by_cluster,
                base.cluster_train_counts,
            )

        def fake_savefig(
            _figure: object,
            path: object,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            Path(path).write_bytes(b"plot")

        try:
            output_dir.mkdir()
            with patch("matplotlib.figure.Figure.savefig", fake_savefig):
                comparison_dir = save_comparative_outputs(
                    [
                        with_lead_days(base_a, actual + 0.1),
                        with_lead_days(base_b, actual - 0.1),
                    ],
                    output_dir,
                    "window_size",
                )
            for prefix in (
                "01_test_timeseries_comparison",
                "02_test_scatter_comparison",
                "04_test_metrics_vs_window_size",
            ):
                self.assertTrue(
                    (comparison_dir / f"{prefix}_lead_day_02.png").exists()
                )
            metrics = pd.read_csv(comparison_dir / "comparative_metrics.csv")
            self.assertEqual(sorted(metrics["lead_day"].unique().tolist()), [1, 2])
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_single_test_still_produces_a_comparative_artifact_set(self) -> None:
        output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_comparative_single_test_{uuid.uuid4().hex}"
        )
        run = self._run(
            name="w10",
            window_size=10,
            dates=["2025-01-01", "2025-01-02", "2025-01-03"],
            actual=[0.0, 1.0, 2.0],
            predicted=[0.1, 0.9, 2.1],
            epochs=2,
        )

        def fake_savefig(
            _figure: object,
            path: object,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            Path(path).write_bytes(b"plot")

        try:
            output_dir.mkdir()
            with patch("matplotlib.figure.Figure.savefig", fake_savefig):
                comparison_dir = save_comparative_outputs(
                    [run],
                    output_dir,
                    "window_size",
                )
            self.assertTrue((comparison_dir / "comparison_manifest.csv").exists())
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_learning_rate_list_expands_configuration_names(self) -> None:
        configurations = build_configurations(
            [1.0],
            state="RS",
            station_id="A801",
            window_sizes=[15],
            n_clusters_list=[3],
            clustering_algorithm="spectral",
            training_parameter_values={
                "learning_rate": [0.001, 0.0001],
            },
        )
        self.assertEqual(
            [
                dict(configuration.variant_parameters)["learning_rate"]
                for configuration in configurations
            ],
            [0.001, 0.0001],
        )
        self.assertEqual(len({configuration.name for configuration in configurations}), 2)
        self.assertIn("lr_0p001", configurations[0].name)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _normalize_learning_rate_values([0.001, 0.001])

        integer_sigma = build_configurations(
            [20.0],
            state="RS",
            station_id="A801",
            window_sizes=[15],
            n_clusters_list=[3],
            clustering_algorithm="spectral",
        )[0]
        self.assertIn("sigma_20", integer_sigma.name)
        self.assertNotIn("sigma_20p0", integer_sigma.name)

    def test_sigma_slug_is_only_used_for_spectral_configuration_names(self) -> None:
        kshape_config = build_configurations(
            [None],
            state="RS",
            station_id="A801",
            window_sizes=[15],
            n_clusters_list=[3],
            clustering_algorithm="kshape",
        )[0]
        kmeans_config = build_configurations(
            [None],
            state="RS",
            station_id="A801",
            window_sizes=[15],
            n_clusters_list=[3],
            clustering_algorithm="kmeans",
        )[0]
        spectral_config = build_configurations(
            [1.0],
            state="RS",
            station_id="A801",
            window_sizes=[15],
            n_clusters_list=[3],
            clustering_algorithm="spectral",
        )[0]

        self.assertEqual(kshape_config.name, "RS_A801_w15_k03_kshape")
        self.assertEqual(kmeans_config.name, "RS_A801_w15_k03_kmeans")
        self.assertEqual(spectral_config.name, "RS_A801_w15_k03_spectral_sigma_1")

    def test_clustering_algorithm_list_expands_configuration_grid(self) -> None:
        configurations = build_configurations(
            [None],
            state="RS",
            station_id="A801",
            window_sizes=[15],
            n_clusters_list=[3],
            clustering_algorithm=["kshape", "kmeans"],
        )

        self.assertEqual(
            [configuration.algorithm for configuration in configurations],
            ["kshape", "kmeans"],
        )
        self.assertEqual(
            [configuration.name for configuration in configurations],
            ["RS_A801_w15_k03_kshape", "RS_A801_w15_k03_kmeans"],
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_configurations(
                [None],
                state="RS",
                station_id="A801",
                window_sizes=[15],
                n_clusters_list=[3],
                clustering_algorithm=["kmeans", "KMEANS"],
            )

    def test_clustering_algorithm_pivot_alias_is_supported(self) -> None:
        self.assertEqual(
            normalize_pivot_parameter("CLUSTERING_ALGORITHM"),
            "clustering_algorithm",
        )
        self.assertEqual(
            validate_comparative_pivot(
                [
                    {"clustering_algorithm": "kshape"},
                    {"clustering_algorithm": "kmeans"},
                ],
                "CLUSTERING_ALGORITHM",
            ),
            "clustering_algorithm",
        )

    def test_dtw_alias_is_canonicalized_in_configurations(self) -> None:
        config = build_configurations(
            [1.0],
            state="RS",
            station_id="A801",
            window_sizes=[15],
            n_clusters_list=[3],
            clustering_algorithm="spectral",
            cluster_dissimilarity_metric="DWT",
            cluster_assignment_method="knn",
        )[0]

        self.assertEqual(config.cluster_dissimilarity_metric, "dtw")
        self.assertIn("_dtw_", config.name)

    def test_cluster_timeseries_plot_setting_is_preserved_in_configuration(self) -> None:
        config = build_configurations(
            [None],
            state="RS",
            station_id="A801",
            window_sizes=[15],
            n_clusters_list=[3],
            clustering_algorithm="kmeans",
            plot_cluster_timeseries=False,
        )[0]

        self.assertFalse(config.plot_cluster_timeseries)

    def test_numeric_training_grids_use_the_cartesian_product(self) -> None:
        configurations = build_configurations(
            [None],
            state="RS",
            station_id="A801",
            window_sizes=[15],
            n_clusters_list=[3],
            clustering_algorithm="kmeans",
            training_parameter_values={
                "learning_rate": [0.001, 0.0001],
                "dropout_rate": [0.1, 0.3],
                "batch_size": [16],
            },
        )

        self.assertEqual(len(configurations), 4)
        variants = [dict(config.variant_parameters) for config in configurations]
        self.assertEqual(
            {
                (variant["learning_rate"], variant["dropout_rate"])
                for variant in variants
            },
            {(0.001, 0.1), (0.001, 0.3), (0.0001, 0.1), (0.0001, 0.3)},
        )
        self.assertTrue(all("batch" not in config.name for config in configurations))
        self.assertEqual(len({config.name for config in configurations}), 4)
        with self.assertRaisesRegex(ValueError, "Unsupported training sweep"):
            build_configurations(
                [None],
                state="RS",
                station_id="A801",
                window_sizes=[15],
                n_clusters_list=[3],
                clustering_algorithm="kmeans",
                training_parameter_values={"unknown_parameter": [1, 2]},
            )

    def test_lstm_units_2_accepts_none_in_configuration_grid(self) -> None:
        configurations = build_configurations(
            [None],
            state="RS",
            station_id="A801",
            window_sizes=[15],
            n_clusters_list=[3],
            clustering_algorithm="kmeans",
            training_parameter_values={"lstm_units_2": [None, 16]},
        )

        self.assertEqual(len(configurations), 2)
        variants = [dict(config.variant_parameters) for config in configurations]
        self.assertEqual(
            [variant["lstm_units_2"] for variant in variants],
            [None, 16],
        )
        self.assertEqual(
            [config.name for config in configurations],
            [
                "RS_A801_w15_k03_kmeans_u2_none",
                "RS_A801_w15_k03_kmeans_u2_16",
            ],
        )

    def test_pipeline_calls_comparative_writer_only_when_enabled(self) -> None:
        base_output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_comparative_pipeline_test_{uuid.uuid4().hex}"
        )
        dataframe = pd.DataFrame(
            {
                "Data": pd.date_range("2025-01-01", periods=8, freq="D"),
                "PRECIPITACAO_TOTAL": np.arange(8, dtype=float),
                "TEMPERATURA_MAXIMA": np.arange(8, dtype=float) + 20,
            }
        )
        observed_parameters: list[dict[str, object]] = []
        observed_create_report: list[bool] = []
        observed_train_info: list[bool] = []
        observed_silhouette_info: list[bool] = []

        def fake_run_configuration(*args: object, **kwargs: object) -> dict[str, object]:
            config = args[1]
            self.assertEqual(kwargs["loss_alpha"], 0.25)
            observed_create_report.append(bool(kwargs["create_report"]))
            observed_train_info.append(bool(kwargs["train_info"]))
            observed_silhouette_info.append(bool(kwargs["silhouette_info"]))
            collector = kwargs["comparative_runs"]
            if collector is not None:
                collector.append(config)
                parameters = kwargs["run_parameters"]
                observed_parameters.append(dict(parameters))
                self.assertEqual(parameters["loss_alpha"], 0.25)
                for field in (
                    "lstm_units",
                    "lstm_units_2",
                    "dropout_rate",
                    "learning_rate",
                    "weight_decay",
                    "epochs",
                    "batch_size",
                    "patience",
                    "warm_up",
                ):
                    self.assertTrue(np.isscalar(parameters[field]), field)
            else:
                self.assertIsNone(kwargs["run_parameters"])
            return {
                "run_name": config.name,
                "test_rmse": 1.0,
                "test_mae": 0.5,
            }

        common_arguments = {
            "state": "RS",
            "station_id": "A801",
            "window_sizes": [2, 3],
            "clustering_feature_normalize": None,
            "clustering_precipitation_normalize": None,
            "lstm_feature_normalize": None,
            "lstm_precipitation_normalize": None,
            "variance_threshold": None,
            "n_clusters_list": [2],
            "clustering_algorithm": "kmeans",
            "n_sigma_values": 1,
            "use_all_features": True,
            "quantitative_metrics": ["MSE"],
            "lstm_units": 4,
            "lstm_units_2": 2,
            "dropout_rate": 0.1,
            "learning_rate": 0.001,
            "epochs": 1,
            "batch_size": 2,
            "early_stopping": False,
            "patience": 1,
            "warm_up": 2,
            "early_stopping_metric": "loss",
            "lstm_loss_function": "mse",
            "loss_alpha": 0.25,
            "loss_quantiles": [0.9],
            "loss_quantile_weights": "auto",
            "verbose_training": 0,
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "random_state": 42,
            "data_root": PROJECT_ROOT / "data",
            "output_root": base_output_dir,
            "sweep_name": "mock_sweep",
            "show_console_info": False,
            "test_all_models": False,
            "train_info": False,
            "silhouette_info": False,
        }

        try:
            with (
                patch(
                    "methods.lstm_cluster.pipeline.load_station_daily_data",
                    return_value=dataframe,
                ),
                patch(
                    "methods.lstm_cluster.pipeline.run_configuration",
                    side_effect=fake_run_configuration,
                ) as run_configuration_mock,
                patch("methods.lstm_cluster.pipeline.save_sweep_outputs"),
                patch(
                    "methods.lstm_cluster.pipeline.save_comparative_outputs",
                    return_value=base_output_dir
                    / "mock_sweep"
                    / "comparative_analysis",
                ) as comparative_writer,
                patch("methods.lstm_cluster.pipeline.save_cluster_sweep_outputs"),
            ):
                run_experiment(
                    **common_arguments,
                    comparative_run=True,
                    pivot_parameter="window_size",
                )

                self.assertEqual(run_configuration_mock.call_count, 2)
                comparative_writer.assert_called_once()
                writer_args = comparative_writer.call_args.args
                self.assertEqual(len(writer_args[0]), 2)
                self.assertEqual(writer_args[2], "window_size")
                self.assertEqual(
                    [row["learning_rate"] for row in observed_parameters],
                    [0.001, 0.001],
                )
                self.assertEqual(
                    [row["warm_up"] for row in observed_parameters],
                    [2, 2],
                )
                self.assertEqual(observed_create_report, [True, True])
                self.assertEqual(observed_train_info, [False, False])
                self.assertEqual(observed_silhouette_info, [False, False])

                run_configuration_mock.reset_mock()
                comparative_writer.reset_mock()
                observed_parameters.clear()
                observed_create_report.clear()
                observed_train_info.clear()
                observed_silhouette_info.clear()
                run_experiment(
                    **{
                        **common_arguments,
                        "window_sizes": [2],
                        "learning_rate": [0.001, 0.0001],
                        "sweep_name": "mock_learning_rate_sweep",
                    },
                    comparative_run=True,
                    pivot_parameter="learning_rate",
                )
                self.assertEqual(run_configuration_mock.call_count, 2)
                self.assertEqual(
                    [row["learning_rate"] for row in observed_parameters],
                    [0.001, 0.0001],
                )
                self.assertEqual(observed_create_report, [True, True])
                self.assertEqual(observed_train_info, [False, False])
                self.assertEqual(observed_silhouette_info, [False, False])
                comparative_writer.assert_called_once()
                self.assertEqual(
                    comparative_writer.call_args.args[2],
                    "learning_rate",
                )

                run_configuration_mock.reset_mock()
                comparative_writer.reset_mock()
                observed_parameters.clear()
                observed_create_report.clear()
                observed_train_info.clear()
                observed_silhouette_info.clear()
                run_experiment(
                    **{
                        **common_arguments,
                        "window_sizes": [2],
                        "clustering_algorithm": ["kshape", "kmeans"],
                        "sweep_name": "mock_algorithm_sweep",
                    },
                    comparative_run=True,
                    pivot_parameter="CLUSTERING_ALGORITHM",
                )
                self.assertEqual(run_configuration_mock.call_count, 2)
                self.assertEqual(
                    [row["clustering_algorithm"] for row in observed_parameters],
                    ["kshape", "kmeans"],
                )
                comparative_writer.assert_called_once()
                self.assertEqual(
                    comparative_writer.call_args.args[2],
                    "clustering_algorithm",
                )

                run_configuration_mock.reset_mock()
                observed_parameters.clear()
                observed_create_report.clear()
                observed_train_info.clear()
                observed_silhouette_info.clear()
                run_experiment(
                    **{
                        **common_arguments,
                        "window_sizes": [2],
                        "sweep_name": "mock_tex_only_sweep",
                    },
                    comparative_run=False,
                    create_report=False,
                )
                self.assertEqual(run_configuration_mock.call_count, 1)
                self.assertEqual(observed_create_report, [False])
                self.assertEqual(observed_train_info, [False])
                self.assertEqual(observed_silhouette_info, [False])

                run_configuration_mock.reset_mock()
                observed_parameters.clear()
                observed_create_report.clear()
                observed_train_info.clear()
                observed_silhouette_info.clear()
                run_experiment(
                    **{
                        **common_arguments,
                        "window_sizes": [2],
                        "learning_rate": [0.001, 0.0001],
                        "sweep_name": "mock_cluster_only_sweep",
                    },
                    comparative_run=False,
                    run_only_cluster=True,
                )
                self.assertEqual(run_configuration_mock.call_count, 1)
                self.assertEqual(observed_parameters, [])
                self.assertEqual(observed_train_info, [False])
                self.assertEqual(observed_silhouette_info, [False])

                with patch(
                    "methods.lstm_cluster.pipeline._execute_configuration_jobs",
                    return_value=[
                        {"run_name": "first"},
                        {"run_name": "second"},
                    ],
                ) as configuration_executor:
                    run_experiment(
                        **{
                            **common_arguments,
                            "window_sizes": [2, 3],
                            "sweep_name": "mock_parallel_cluster_only_sweep",
                            "show_console_info": True,
                        },
                        comparative_run=False,
                        run_only_cluster=True,
                        parallel_training=True,
                    )
                self.assertTrue(configuration_executor.call_args.kwargs["parallel"])
                parallel_jobs = configuration_executor.call_args.args[0]
                self.assertEqual(len(parallel_jobs), 2)
                self.assertTrue(
                    all(job_kwargs["run_only_cluster"] for _, job_kwargs in parallel_jobs)
                )
                self.assertTrue(
                    all(
                        not job_kwargs["show_console_info"]
                        for _, job_kwargs in parallel_jobs
                    )
                )

                with self.assertRaisesRegex(ValueError, "unique output names"):
                    run_experiment(
                        **{
                            **common_arguments,
                            "window_sizes": [2, 2],
                            "sweep_name": "mock_duplicate_sweep",
                        },
                        comparative_run=False,
                    )

            with (
                patch(
                    "methods.lstm_cluster.pipeline.load_station_daily_data",
                    return_value=dataframe,
                ),
                patch(
                    "methods.lstm_cluster.pipeline.run_configuration",
                    side_effect=fake_run_configuration,
                ),
                patch("methods.lstm_cluster.pipeline.save_sweep_outputs"),
                patch(
                    "methods.lstm_cluster.pipeline.save_comparative_outputs"
                ) as disabled_writer,
            ):
                run_experiment(
                    **common_arguments,
                    comparative_run=False,
                    pivot_parameter="window_size",
                )
            disabled_writer.assert_not_called()

            with self.assertRaisesRegex(ValueError, "RUN_ONLY_CLUSTER=False"):
                run_experiment(
                    **common_arguments,
                    comparative_run=True,
                    pivot_parameter="window_size",
                    run_only_cluster=True,
                )

            with self.assertRaisesRegex(ValueError, "loss_alpha.*positive"):
                run_experiment(
                    **{
                        **common_arguments,
                        "lstm_loss_function": "weighted_mse_loss",
                        "loss_alpha": 0.0,
                    },
                    comparative_run=False,
                )
        finally:
            shutil.rmtree(base_output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
