"""Tests for cross-experiment LaTeX meta-analysis reports."""

from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import data.meta_analysis_report as meta_report
from data.meta_analysis_report import (
    create_meta_analysis_report,
    load_meta_analysis_runs,
    render_meta_analysis_report,
)
from experiments import create_meta_analysis_report as runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MetaAnalysisReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_meta_analysis_report_test_{uuid.uuid4().hex}"
        )
        self.work_dir.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def _write_run(
        self,
        sweep_dir: Path,
        run_name: str,
        *,
        lead_one: tuple[float, float, float],
        lead_two: tuple[float, float, float],
        n_test: int = 20,
    ) -> None:
        run_dir = sweep_dir / run_name
        diagnostics_dir = run_dir / "forecast_horizon_diagnostics"
        diagnostics_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "lead_day": 1,
                    "forecast_horizon": 2,
                    "n_test": n_test,
                    "MSE": lead_one[0],
                    "MAE": lead_one[1],
                    "R2": lead_one[2],
                },
                {
                    "lead_day": 2,
                    "forecast_horizon": 2,
                    "n_test": n_test,
                    "MSE": lead_two[0],
                    "MAE": lead_two[1],
                    "R2": lead_two[2],
                },
            ]
        ).to_csv(
            diagnostics_dir / "test_prediction_metrics_by_lead_day.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "split": "Train",
                    "MSE": 1.0,
                    "MAE": 1.0,
                    "R2": 0.0,
                },
                {
                    "split": "Test",
                    "MSE": lead_two[0],
                    "MAE": lead_two[1],
                    "R2": lead_two[2],
                },
            ]
        ).to_csv(run_dir / "metrics_summary.csv", index=False)
        prediction_rows = []
        for sample_index, target_date in enumerate(
            pd.date_range("2020-01-01", periods=4),
        ):
            for lead_day in (1, 2):
                actual = float(sample_index + lead_day)
                prediction_rows.append(
                    {
                        "sample_index": sample_index,
                        "window_index": 100 + sample_index,
                        "lead_day": lead_day,
                        "target_date": target_date.strftime("%Y-%m-%d"),
                        "actual_mm": actual,
                        "predicted_mm": actual + 0.1 * len(run_name),
                    }
                )
        pd.DataFrame(prediction_rows).to_csv(
            diagnostics_dir / "test_prediction_by_lead_day.csv",
            index=False,
        )

    def _write_sweep(
        self,
        sweep_name: str,
        rows: list[dict[str, object]],
    ) -> Path:
        sweep_dir = self.work_dir / sweep_name
        sweep_dir.mkdir()
        pd.DataFrame(rows).to_csv(sweep_dir / "sweep_results.csv", index=False)
        return sweep_dir

    def test_loads_all_physical_runs_and_deduplicates_overlapping_sources(
        self,
    ) -> None:
        sweep_a = self._write_sweep(
            "sweep_a",
            [
                {
                    "run_name": "shared&run",
                    "forecast_horizon": 2,
                    "n_test": 20,
                    "test_mse": 4.0,
                    "test_mae": 1.4,
                    "test_r2": 0.5,
                    "lead_day_metrics_path": (
                        "forecast_horizon_diagnostics/"
                        "test_prediction_metrics_by_lead_day.csv"
                    ),
                }
            ],
        )
        self._write_run(
            sweep_a,
            "shared&run",
            lead_one=(3.0, 1.0, 0.4),
            lead_two=(4.0, 1.4, 0.5),
        )
        self._write_run(
            sweep_a,
            "unaggregated_run",
            lead_one=(7.0, 1.8, 0.2),
            lead_two=(9.0, 2.0, 0.1),
        )

        sweep_b = self._write_sweep(
            "sweep_b",
            [
                {
                    "run_name": "shared&run",
                    "forecast_horizon": 2,
                    "n_test": 30,
                    "test_mse": 5.0,
                    "test_mae": 1.2,
                    "test_r2": 0.4,
                    "lead_day_metrics_path": (
                        "forecast_horizon_diagnostics/"
                        "test_prediction_metrics_by_lead_day.csv"
                    ),
                }
            ],
        )
        self._write_run(
            sweep_b,
            "shared&run",
            lead_one=(4.5, 1.1, 0.3),
            lead_two=(5.0, 1.2, 0.4),
            n_test=30,
        )

        runs = load_meta_analysis_runs(
            [self.work_dir, sweep_a, sweep_a / "sweep_results.csv"]
        )

        self.assertEqual(len(runs), 3)
        self.assertEqual(
            [(run.experiment_name, run.run_name) for run in runs],
            [
                ("sweep_a", "shared&run"),
                ("sweep_a", "unaggregated_run"),
                ("sweep_b", "shared&run"),
            ],
        )
        self.assertEqual(runs[0].overall_metrics["MSE"], 4.0)
        self.assertEqual(runs[0].lead_day_metrics[1]["MAE"], 1.0)
        self.assertEqual(runs[2].n_test, 30)
        self.assertEqual(runs[2].forecast_horizon, 2)
        self.assertTrue(
            all(
                run.source_path.name
                == "test_prediction_metrics_by_lead_day.csv"
                for run in runs
            )
        )

    def test_overall_metrics_use_configured_forecast_horizon(self) -> None:
        run_dir = self.work_dir / "horizon_sweep" / "selected_horizon"
        diagnostics_dir = run_dir / "forecast_horizon_diagnostics"
        diagnostics_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "lead_day": 1,
                    "forecast_horizon": 2,
                    "n_test": 20,
                    "MSE": 1.0,
                    "MAE": 1.0,
                    "R2": 0.1,
                },
                {
                    "lead_day": 2,
                    "forecast_horizon": 2,
                    "n_test": 19,
                    "MSE": 2.0,
                    "MAE": 2.0,
                    "R2": 0.2,
                },
                {
                    "lead_day": 3,
                    "forecast_horizon": 2,
                    "n_test": 18,
                    "MSE": 99.0,
                    "MAE": 99.0,
                    "R2": -9.0,
                },
            ]
        ).to_csv(
            diagnostics_dir / "test_prediction_metrics_by_lead_day.csv",
            index=False,
        )

        runs = load_meta_analysis_runs([run_dir])

        self.assertEqual(runs[0].forecast_horizon, 2)
        self.assertEqual(runs[0].overall_metrics["MSE"], 2.0)
        self.assertIn(3, runs[0].lead_day_metrics)
        self.assertEqual(runs[0].n_test, 19)

        missing_dir = self.work_dir / "horizon_sweep" / "missing_horizon"
        missing_diagnostics = (
            missing_dir / "forecast_horizon_diagnostics"
        )
        missing_diagnostics.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "lead_day": 1,
                    "forecast_horizon": 2,
                    "MSE": 1.0,
                    "MAE": 1.0,
                    "R2": 0.1,
                },
                {
                    "lead_day": 3,
                    "forecast_horizon": 2,
                    "MSE": 3.0,
                    "MAE": 3.0,
                    "R2": 0.3,
                },
            ]
        ).to_csv(
            missing_diagnostics
            / "test_prediction_metrics_by_lead_day.csv",
            index=False,
        )
        with self.assertRaisesRegex(
            ValueError,
            r"forecast_horizon D\+2 is missing",
        ):
            load_meta_analysis_runs([missing_dir])

    def test_report_uses_global_run_columns_chunks_and_best_values(self) -> None:
        sweep_a = self.work_dir / "sweep_a"
        sweep_a.mkdir()
        self._write_run(
            sweep_a,
            "shared&run",
            lead_one=(3.0, 1.0, 0.4),
            lead_two=(4.0, 1.4, 0.5),
        )
        self._write_run(
            sweep_a,
            "second_run",
            lead_one=(8.0, 1.9, 0.0),
            lead_two=(9.0, 2.0, 0.1),
        )
        sweep_b = self.work_dir / "sweep_b"
        sweep_b.mkdir()
        self._write_run(
            sweep_b,
            "shared&run",
            lead_one=(4.5, 1.1, 0.3),
            lead_two=(5.0, 1.2, 0.4),
            n_test=30,
        )

        runs = load_meta_analysis_runs([sweep_a, sweep_b])
        tex = render_meta_analysis_report(
            runs,
            title="Meta & Analysis",
            runs_per_table=2,
            digits=3,
        )

        self.assertIn(r"\title{Meta \& Analysis}", tex)
        self.assertIn(r"& \multicolumn{2}{c}{Run} \\", tex)
        self.assertIn(r"Metric & 1 & 2 \\", tex)
        self.assertIn(r"& \multicolumn{1}{c}{Run} \\", tex)
        self.assertIn(r"Metric & 3 \\", tex)
        self.assertIn("Columns are numbered globally from Run 1 through Run 3.", tex)
        self.assertIn(r"shared\&run", tex)
        self.assertIn(r"\textbf{4.000}", tex)
        self.assertIn(r"\textbf{1.200}", tex)
        self.assertIn(r"\textbf{0.500}", tex)
        self.assertIn(r"MSE ($\mathrm{mm}^2$) $\downarrow$", tex)
        self.assertIn(r"RMSE ($\mathrm{mm}$) $\downarrow$", tex)
        self.assertIn(r"\textbf{1.732}", tex)
        self.assertIn(r"\section{Metrics by Lead Day}", tex)
        self.assertNotIn(r"\subsection{Lead day D+1}", tex)
        self.assertNotIn(r"\subsection{Lead day D+2}", tex)
        self.assertIn(r"\caption{Test metrics for lead day D+1", tex)
        self.assertIn(r"\caption{Test metrics for lead day D+2", tex)
        self.assertIn(
            "\\section{Metrics by Lead Day}\n"
            "\\begin{table}[!htbp]",
            tex,
        )
        self.assertIn(r"\begin{table}[!htbp]", tex)

    def test_registry_disambiguates_equal_experiment_folder_names(self) -> None:
        first_sweep = self.work_dir / "family_a" / "same_sweep"
        second_sweep = self.work_dir / "family_b" / "same_sweep"
        first_sweep.mkdir(parents=True)
        second_sweep.mkdir(parents=True)
        self._write_run(
            first_sweep,
            "run_one",
            lead_one=(1.0, 1.0, 0.1),
            lead_two=(2.0, 2.0, 0.2),
        )
        self._write_run(
            second_sweep,
            "run_two",
            lead_one=(3.0, 3.0, 0.3),
            lead_two=(4.0, 4.0, 0.4),
        )

        runs = load_meta_analysis_runs([first_sweep, second_sweep])
        tex = render_meta_analysis_report(runs)

        self.assertIn("from 2 experiment folders.", tex)
        self.assertIn(r"family\_\allowbreak{}a", tex)
        self.assertIn(r"family\_\allowbreak{}b", tex)

    def test_report_creates_period_timeseries_plots(self) -> None:
        sweep_dir = self.work_dir / "timeseries_sweep"
        sweep_dir.mkdir()
        self._write_run(
            sweep_dir,
            "run_one",
            lead_one=(1.0, 1.0, 0.1),
            lead_two=(2.0, 2.0, 0.2),
        )
        self._write_run(
            sweep_dir,
            "run_two",
            lead_one=(1.5, 1.5, 0.3),
            lead_two=(2.5, 2.5, 0.4),
        )

        report_path, runs = create_meta_analysis_report(
            [sweep_dir],
            self.work_dir / "reports" / "meta.tex",
            start_date="2020-01-02",
            end_date="2020-01-03",
        )

        self.assertEqual(len(runs), 2)
        plot_dir = report_path.parent / "meta_analysis_timeseries"
        self.assertTrue(
            (plot_dir / "01_meta_timeseries_comparison_lead_day_01.png").exists()
        )
        self.assertTrue(
            (plot_dir / "01_meta_timeseries_comparison_lead_day_02.png").exists()
        )
        tex = report_path.read_text(encoding="utf-8")
        self.assertIn(r"\section{Prediction Time Series Comparison}", tex)
        self.assertIn("Run 1 through Run 2", tex)
        self.assertIn("2020-01-02 to 2020-01-03", tex)
        self.assertIn(
            r"\detokenize{meta_analysis_timeseries/01_meta_timeseries_comparison_lead_day_01.png}",
            tex,
        )

        cli_report = self.work_dir / "reports" / "cli_timeseries.tex"
        exit_code = runner.main(
            [
                str(sweep_dir),
                "--output",
                str(cli_report),
                "--start-date",
                "2020-01-02",
                "--end-date",
                "2020-01-03",
            ]
        )
        self.assertEqual(exit_code, 0)
        self.assertTrue(
            (
                cli_report.parent
                / "meta_analysis_timeseries"
                / "01_meta_timeseries_comparison_lead_day_01.png"
            ).exists()
        )

    def test_timeseries_plots_derive_dates_for_legacy_prediction_rows(self) -> None:
        sweep_dir = self.work_dir / "lstm_cluster_sweep_RS_A801_2026_07_29_09h43"
        sweep_dir.mkdir()
        for run_name, offset in (("RS_A801_w03_k01", 0.1), ("RS_A801_w03_k02", 0.2)):
            run_dir = sweep_dir / run_name
            diagnostics_dir = run_dir / "forecast_horizon_diagnostics"
            diagnostics_dir.mkdir(parents=True)
            (run_dir / "summary.txt").write_text(
                "Station: RS/A801\nWindow size: 3\nForecast horizon: +2 day(s)\n",
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {"lead_day": 1, "forecast_horizon": 2, "MSE": 1.0, "MAE": 1.0, "R2": 0.1},
                    {"lead_day": 2, "forecast_horizon": 2, "MSE": 2.0, "MAE": 2.0, "R2": 0.2},
                ]
            ).to_csv(
                diagnostics_dir / "test_prediction_metrics_by_lead_day.csv",
                index=False,
            )
            rows = []
            for window_index in (0, 1):
                for lead_day in (1, 2):
                    actual = float(window_index + lead_day)
                    rows.append(
                        {
                            "lead_day": lead_day,
                            "window_index": window_index,
                            "actual_mm": actual,
                            "predicted_mm": actual + offset,
                        }
                    )
            pd.DataFrame(rows).to_csv(
                diagnostics_dir / "test_prediction_by_lead_day.csv",
                index=False,
            )

        fake_data_root = self.work_dir / "inmet"
        station_dir = fake_data_root / "RS" / "A801"
        station_dir.mkdir(parents=True)
        daily = pd.DataFrame(
            {
                "DATA": pd.date_range("2020-01-01", periods=10).strftime("%Y-%m-%d"),
                "TEMPERATURA_MAXIMA": 1.0,
                "TEMPERATURA_MIN": 1.0,
                "UMIDADE_MAX": 1.0,
                "UMIDADE_MIN": 1.0,
                "PRESSAO_MAX": 1.0,
                "PRESSAO_MIN": 1.0,
                "VELOCIDADE_VENTO": 1.0,
                "DIRECAO_VENTO_SIN": 1.0,
                "DIRECAO_VENTO_COS": 1.0,
                "RAJADA_VENTO": 1.0,
                "PRECIPITACAO_TOTAL": 1.0,
                "RADIACAO": 1.0,
            }
        )
        daily.to_csv(station_dir / "A801_2000_2026_daily.csv", sep=";", index=False)

        with patch.object(meta_report, "DATA_ROOT", fake_data_root):
            report_path, _runs = create_meta_analysis_report(
                [sweep_dir],
                self.work_dir / "reports" / "legacy_dates.tex",
                start_date="2020-01-04",
                end_date="2020-01-06",
            )

        tex = report_path.read_text(encoding="utf-8")
        self.assertIn("2020-01-04 to 2020-01-05", tex)
        self.assertTrue(
            (
                report_path.parent
                / "meta_analysis_timeseries"
                / "01_meta_timeseries_comparison_lead_day_01.png"
            ).exists()
        )

    def test_metrics_summary_fallback_and_runner_write_tex(self) -> None:
        run_dir = self.work_dir / "standalone_sweep" / "run_with_underscore"
        run_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {"split": "Train", "MSE": 2.0, "MAE": 1.0, "R2": 0.2},
                {"split": "Test", "MSE": 6.0, "MAE": 1.5, "R2": 0.3},
            ]
        ).to_csv(run_dir / "metrics_summary.csv", index=False)

        output_without_suffix = self.work_dir / "reports" / "meta"
        report_path, runs = create_meta_analysis_report(
            [run_dir],
            output_without_suffix,
        )

        self.assertEqual(report_path.suffix, ".tex")
        self.assertTrue(report_path.exists())
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].overall_metrics["MAE"], 1.5)
        self.assertEqual(runs[0].lead_day_metrics, {})
        self.assertIn(
            r"run\_\allowbreak{}with\_\allowbreak{}underscore",
            report_path.read_text(encoding="utf-8"),
        )

        cli_output = self.work_dir / "reports" / "cli_meta.tex"
        exit_code = runner.main(
            [
                str(run_dir),
                "--output",
                str(cli_output),
                "--title",
                "CLI report",
            ]
        )
        self.assertEqual(exit_code, 0)
        self.assertTrue(cli_output.exists())
        self.assertIn(
            r"\title{CLI report}",
            cli_output.read_text(encoding="utf-8"),
        )

    def test_runner_reports_configuration_errors_without_traceback(self) -> None:
        original_paths = runner.EXPERIMENT_PATHS
        runner.EXPERIMENT_PATHS = []
        stderr = StringIO()
        try:
            with redirect_stderr(stderr):
                exit_code = runner.main([])
        finally:
            runner.EXPERIMENT_PATHS = original_paths

        self.assertEqual(exit_code, 2)
        self.assertIn("Error: Configure EXPERIMENT_PATHS", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_rejects_duplicate_sweep_run_names(self) -> None:
        sweep_dir = self._write_sweep(
            "duplicate_sweep",
            [
                {
                    "run_name": "Same_Run",
                    "test_mse": 1.0,
                    "test_mae": 1.0,
                    "test_r2": 0.1,
                },
                {
                    "run_name": "same_run",
                    "test_mse": 2.0,
                    "test_mae": 2.0,
                    "test_r2": 0.2,
                },
            ],
        )

        with self.assertRaisesRegex(ValueError, "Duplicate run_name"):
            load_meta_analysis_runs([sweep_dir])

    def test_validates_comparative_names_and_alignment_contexts(self) -> None:
        missing_name_dir = (
            self.work_dir / "missing_name" / "comparative_analysis"
        )
        missing_name_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "run_name": None,
                    "lead_day": 1,
                    "n_common_test_dates": 10,
                    "MSE": 1.0,
                    "MAE": 1.0,
                    "R2": 0.1,
                }
            ]
        ).to_csv(
            missing_name_dir / "comparative_metrics.csv",
            index=False,
        )
        with self.assertRaisesRegex(ValueError, "Missing run name"):
            load_meta_analysis_runs([missing_name_dir])

        comparison_paths = []
        for index, start_date in enumerate(
            ("2020-01-01", "2020-02-01"),
            start=1,
        ):
            comparison_dir = (
                self.work_dir
                / f"aligned_{index}"
                / "comparative_analysis"
            )
            comparison_dir.mkdir(parents=True)
            comparison_path = comparison_dir / "comparative_metrics.csv"
            pd.DataFrame(
                [
                    {
                        "run_name": f"aligned_run_{index}",
                        "lead_day": 1,
                        "n_common_test_dates": 10,
                        "start_date": start_date,
                        "end_date": "2020-12-31",
                        "MSE": float(index),
                        "MAE": float(index),
                        "R2": index / 10.0,
                    }
                ]
            ).to_csv(comparison_path, index=False)
            comparison_paths.append(comparison_path)

        with self.assertRaisesRegex(ValueError, "alignment context"):
            load_meta_analysis_runs(comparison_paths)

    def test_rejects_mixing_raw_and_aligned_metric_scopes(self) -> None:
        raw_run_dir = self.work_dir / "raw_sweep" / "raw_run"
        raw_run_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {"split": "Test", "MSE": 2.0, "MAE": 1.0, "R2": 0.2},
            ]
        ).to_csv(raw_run_dir / "metrics_summary.csv", index=False)

        comparison_dir = self.work_dir / "aligned_sweep" / "comparative_analysis"
        comparison_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "run_name": "aligned_run",
                    "lead_day": 1,
                    "n_common_test_dates": 10,
                    "MSE": 1.0,
                    "MAE": 0.8,
                    "R2": 0.4,
                }
            ]
        ).to_csv(comparison_dir / "comparative_metrics.csv", index=False)

        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            load_meta_analysis_runs(
                [
                    raw_run_dir,
                    comparison_dir / "comparative_metrics.csv",
                ]
            )


if __name__ == "__main__":
    unittest.main()
