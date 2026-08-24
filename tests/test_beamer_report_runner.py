"""Tests for the experiment Beamer report runner."""

from __future__ import annotations

import shutil
import subprocess
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from experiments import create_beamer_report as runner
from experiments.create_beamer_report import (
    render_beamer,
    resolve_selected_plots,
    selected_run_parameters,
    write_beamer,
)


class BeamerReportRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work_dir = (
            Path(__file__).resolve().parents[1]
            / ".tmp"
            / f"_beamer_runner_test_{uuid.uuid4().hex}"
        )
        self.run_dir = self.work_dir / "run"
        self.run_dir.mkdir(parents=True)
        self.prediction_plot = self._write_plot(
            "prediction_overview/02_predictions_vs_actual.png"
        )
        self.scatter_plot = self._write_plot(
            "cluster_prediction_scatter/cluster_1_predicted_vs_actual_scatter.png"
        )
        self.residual_plot = self._write_plot(
            "residual_diagnostics/03_residuals_analysis.png"
        )
        self._write_plot("experiment_report.pdf")
        (self.run_dir / "summary.txt").write_text(
            "\n".join(
                [
                    "Run folder: run",
                    "Station: RS/A801",
                    "Window size: 10",
                    "Forecast horizon: +5 day(s)",
                    "Number of clusters: 2",
                    "Clustering algorithm: kshape",
                ]
            ),
            encoding="utf-8",
        )
        (self.work_dir / "sweep_results.csv").write_text(
            "run_name,learning_rate,epochs,lstm_units,lstm_units_2,test_rmse\n"
            "run,0.001,40,32,,17.06\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_resolve_selected_plots_accepts_path_glob_and_substring(self) -> None:
        selected = resolve_selected_plots(
            self.run_dir,
            [
                "prediction_overview/02_predictions_vs_actual.png",
                "cluster_prediction_scatter/*.png",
                "residuals_analysis",
            ],
        )

        self.assertEqual(
            [path.resolve() for path in selected],
            [
                self.prediction_plot.resolve(),
                self.scatter_plot.resolve(),
                self.residual_plot.resolve(),
            ],
        )

    def test_render_beamer_has_clickable_overview_and_plot_titles(self) -> None:
        tex = render_beamer(
            self.run_dir,
            [self.prediction_plot, self.scatter_plot],
            title="Demo Run",
        )

        self.assertIn(r"\usetheme{Madrid}", tex)
        self.assertNotIn(r"\usecolortheme{dove}", tex)
        self.assertIn(r"\setbeamercolor{section in toc}{fg=black}", tex)
        self.assertIn(r"\setbeamertemplate{itemize item}[ball]", tex)
        self.assertIn(r"\setbeamertemplate{section in toc}", tex)
        self.assertIn(r"\tableofcontents", tex)
        self.assertIn(r"\hypersetup{hidelinks}", tex)
        self.assertIn(r"\hypertarget{sec-prediction-overview}", tex)
        self.assertIn("Predicted vs Actual on Test Set", tex)
        self.assertIn("Cluster 1 - Test Actual vs Predicted Scatter", tex)
        self.assertIn(
            r"\detokenize{prediction_overview/02_predictions_vs_actual.png}",
            tex,
        )

    def test_render_beamer_adds_requested_run_parameter_section(self) -> None:
        tex = render_beamer(
            self.run_dir,
            [self.prediction_plot],
            parameters=[
                "learning_rate",
                "epochs",
                "LSTM_UNITS_1",
                "window_size",
                "CLUSTERING_ALGORITHM",
            ],
        )

        self.assertIn(r"\section{Run Parameters}", tex)
        self.assertIn(r"\begin{frame}{Run Parameters}", tex)
        self.assertIn(r"\begin{tabular}{|l|l|}", tex)
        self.assertIn(r"\textbf{Parameter} & \textbf{Value} \\", tex)
        self.assertIn(r"learning\_rate & 0.001 \\ \hline", tex)
        self.assertIn(r"epochs & 40 \\ \hline", tex)
        self.assertIn(r"LSTM\_UNITS\_1 & 32 \\ \hline", tex)
        self.assertIn(r"window\_size & 10 \\ \hline", tex)
        self.assertIn(r"CLUSTERING\_ALGORITHM & kshape \\ \hline", tex)

    def test_selected_run_parameters_accepts_runner_constant_names(self) -> None:
        rows = selected_run_parameters(
            self.run_dir,
            [
                "STATE",
                "STATION_ID",
                "WINDOW_SIZES",
                "N_CLUSTERS_LIST",
                "FORECAST_HORIZON",
                "missing_param",
            ],
        )

        self.assertEqual(
            rows,
            [
                ("STATE", "RS"),
                ("STATION_ID", "A801"),
                ("WINDOW_SIZES", "10"),
                ("N_CLUSTERS_LIST", "2"),
                ("FORECAST_HORIZON", "+5 day(s)"),
                ("missing_param", "not found"),
            ],
        )

    def test_write_beamer_uses_paths_relative_to_output_directory(self) -> None:
        output_path = self.work_dir / "slides" / "custom_beamer.tex"

        written_path = write_beamer(
            self.run_dir,
            [self.prediction_plot],
            output_path=output_path,
        )

        tex = written_path.read_text(encoding="utf-8")
        self.assertEqual(written_path, output_path)
        self.assertIn(
            r"\detokenize{../run/prediction_overview/02_predictions_vs_actual.png}",
            tex,
        )

    def test_main_without_arguments_uses_editable_runner_config(self) -> None:
        output_path = self.work_dir / "configured_beamer.tex"
        old_config = {
            "RUN_DIR": runner.RUN_DIR,
            "SELECTED_PLOTS": runner.SELECTED_PLOTS,
            "PLOTS_FILE": runner.PLOTS_FILE,
            "OUTPUT_PATH": runner.OUTPUT_PATH,
            "TITLE": runner.TITLE,
            "PARAMS": runner.PARAMS,
            "LIST_PLOTS": runner.LIST_PLOTS,
            "COMPILE_PDF": runner.COMPILE_PDF,
            "PDFLATEX_RUNS": runner.PDFLATEX_RUNS,
        }
        self.addCleanup(lambda: _restore_runner_config(old_config))

        runner.RUN_DIR = self.run_dir
        runner.SELECTED_PLOTS = ["prediction_overview/02_predictions_vs_actual.png"]
        runner.PLOTS_FILE = None
        runner.OUTPUT_PATH = output_path
        runner.TITLE = "Configured Run"
        runner.PARAMS = ["learning_rate"]
        runner.LIST_PLOTS = False
        runner.COMPILE_PDF = False

        self.assertEqual(runner.main([]), 0)
        tex = output_path.read_text(encoding="utf-8")
        self.assertIn(r"\title{Configured Run}", tex)
        self.assertIn(r"learning\_rate & 0.001 \\ \hline", tex)
        self.assertIn(
            r"\detokenize{run/prediction_overview/02_predictions_vs_actual.png}",
            tex,
        )

    def test_compile_beamer_pdf_runs_pdflatex_and_returns_pdf(self) -> None:
        tex_path = self.work_dir / "beamer.tex"
        tex_path.write_text(r"\documentclass{beamer}\begin{document}\end{document}", encoding="utf-8")

        def fake_run(command: list[str], cwd: Path, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            (Path(cwd) / Path(command[-1]).with_suffix(".pdf")).write_bytes(b"pdf")
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

        with patch("data.beamer_report.subprocess.run", side_effect=fake_run) as run:
            pdf_path = runner.compile_beamer_pdf(tex_path, runs=2)

        self.assertEqual(pdf_path, tex_path.with_suffix(".pdf").resolve())
        self.assertTrue(pdf_path.exists())
        self.assertEqual(run.call_count, 2)
        self.assertTrue(tex_path.with_name("beamer_compile.log").exists())

    def _write_plot(self, relative_path: str) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"plot")
        return path


def _restore_runner_config(old_config: dict[str, object]) -> None:
    for name, value in old_config.items():
        setattr(runner, name, value)


if __name__ == "__main__":
    unittest.main()
