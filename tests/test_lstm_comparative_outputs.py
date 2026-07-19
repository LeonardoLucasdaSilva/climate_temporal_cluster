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
    _weighted_history_dataframe,
    normalize_pivot_parameter,
    save_comparative_outputs,
    validate_comparative_pivot,
)
from methods.lstm_cluster.pipeline import (
    _normalize_learning_rate_values,
    build_configurations,
    run_experiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LstmComparativeOutputTests(unittest.TestCase):
    def _run(
        self,
        *,
        name: str,
        window_size: int,
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
                "n_clusters": 2,
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
                "test_predictions_comparison.csv",
                "aligned_test_predictions.csv",
                "training_history_comparison.csv",
                "comparative_metrics.csv",
                "comparison_manifest.csv",
                "comparison_summary.txt",
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
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

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

        def fake_run_configuration(*args: object, **kwargs: object) -> dict[str, object]:
            config = args[1]
            self.assertEqual(kwargs["loss_alpha"], 0.25)
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

                run_configuration_mock.reset_mock()
                comparative_writer.reset_mock()
                observed_parameters.clear()
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
                comparative_writer.assert_called_once()
                self.assertEqual(
                    comparative_writer.call_args.args[2],
                    "learning_rate",
                )

                run_configuration_mock.reset_mock()
                observed_parameters.clear()
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
