"""Tests for persistent backend-specific experiment rankings."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import data.experiment_rankings as experiment_rankings
from data.experiment_rankings import (
    HORIZON_RANKING_COLUMNS,
    RANKING_COLUMNS,
    _ranking_table,
    initialize_experiment_progress,
    initialize_experiment_rankings,
    loss_display_name,
    render_architecture_threshold_triptych_rankings,
    render_cluster_comparison_rankings,
    render_horizon_threshold_triptych_rankings,
    synchronize_local_experiment_rankings,
    update_experiment_rankings,
)


class ExperimentRankingsTests(unittest.TestCase):
    def test_pipeline_ranking_calls_forward_active_training_metadata(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        for relative_path in (
            Path("src/methods/lstm_cluster/pipeline.py"),
            Path("src/methods/lstm_cluster_pytorch/pipeline.py"),
        ):
            tree = ast.parse(
                (project_root / relative_path).read_text(encoding="utf-8")
            )
            calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "update_experiment_rankings"
            ]
            self.assertEqual(len(calls), 1, relative_path.as_posix())
            keyword_names = {keyword.arg for keyword in calls[0].keywords}
            self.assertIn("patience_metric", keyword_names)
            self.assertIn("lstm_units", keyword_names)
            self.assertIn("lstm_units_2", keyword_names)
            self.assertIn("precipitation_standardized", keyword_names)
            self.assertIn("actual_by_lead_day", keyword_names)
            self.assertIn("predicted_by_lead_day", keyword_names)
            self.assertIn("forecast_horizon", keyword_names)
            sync_calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "synchronize_local_experiment_rankings"
            ]
            self.assertEqual(len(sync_calls), 1, relative_path.as_posix())

    def test_ranking_tables_group_matched_settings_and_order_k(self) -> None:
        rows = []
        for run_id, k_value, overall_rank, threshold_rank in (
            ("sweep/k3", 3, 1, 2),
            ("sweep/k1", 1, 2, 1),
        ):
            row = {column: pd.NA for column in RANKING_COLUMNS}
            row.update(
                {
                    "run_id": run_id,
                    "date": f"2026-01-0{overall_rank}",
                    "window_size": 15,
                    "n_clusters": k_value,
                    "clustering_method": "-" if k_value == 1 else "kmeans",
                    "loss_function": "MSE",
                    "n": 10,
                    "mae": float(overall_rank),
                    "mae_rank": overall_rank,
                    "mse": float(overall_rank),
                    "mse_rank": overall_rank,
                    "r2": float(3 - overall_rank),
                    "r2_rank": overall_rank,
                    "gt_0mm_threshold_mm": 0.0,
                    "gt_0mm_n": 5,
                    "gt_0mm_mae": float(threshold_rank),
                    "gt_0mm_mae_rank": threshold_rank,
                    "gt_0mm_mse": float(threshold_rank),
                    "gt_0mm_mse_rank": threshold_rank,
                    "gt_0mm_r2": float(3 - threshold_rank),
                    "gt_0mm_r2_rank": threshold_rank,
                    "updated_at": f"2026-01-0{overall_rank}T00:00:00",
                }
            )
            rows.append(row)

        rankings = pd.DataFrame(rows, columns=RANKING_COLUMNS)
        overall_tex = "\n".join(_ranking_table(rankings, "Overall", ""))
        threshold_tex = "\n".join(
            _ranking_table(rankings, "Wet days", "gt_0mm_")
        )

        self.assertLess(
            overall_tex.index("2026-01-02"),
            overall_tex.index("2026-01-01"),
        )
        self.assertLess(
            threshold_tex.index("2026-01-02"),
            threshold_tex.index("2026-01-01"),
        )

    def test_table_does_not_use_other_metrics_to_break_mae_ties(self) -> None:
        rows = []
        for run_id, mse_rank, updated_at in (
            ("sweep/older", 2, "2026-01-01T00:00:00"),
            ("sweep/newer", 1, "2026-01-02T00:00:00"),
        ):
            row = {column: pd.NA for column in RANKING_COLUMNS}
            row.update(
                {
                    "run_id": run_id,
                    "date": updated_at[:10],
                    "window_size": 15,
                    "n_clusters": 1,
                    "clustering_method": "kmeans",
                    "loss_function": "MSE",
                    "n": 10,
                    "mae": 1.0,
                    "mae_rank": 1,
                    "mse": float(mse_rank),
                    "mse_rank": mse_rank,
                    "r2": 0.5,
                    "r2_rank": 1,
                    "updated_at": updated_at,
                }
            )
            rows.append(row)

        table_tex = "\n".join(
            _ranking_table(
                pd.DataFrame(rows, columns=RANKING_COLUMNS),
                "Overall",
                "",
            )
        )

        self.assertLess(table_tex.index("2026-01-01"), table_tex.index("2026-01-02"))

    def test_initialize_creates_empty_backend_files(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            csv_path, tex_path, pdf_path = initialize_experiment_rankings(
                "tensorflow",
                compile_pdf=False,
                ranking_dir=Path(temporary_directory),
            )
            self.assertTrue(csv_path.exists())
            self.assertTrue(
                (Path(temporary_directory) / "experiment_horizon_rankings.csv").exists()
            )
            self.assertTrue(tex_path.exists())
            self.assertIn("TensorFlow Experiment Rankings", tex_path.read_text())
            self.assertIsNone(pdf_path)

    def test_progress_log_reports_completed_count(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            temporary_path = Path(temporary_directory)
            progress_path = initialize_experiment_progress(
                temporary_path / "progress.log",
                run_token="batch_123",
                total=126,
            )
            self.assertIn("processed: 0/126", progress_path.read_text())

            output_dir = temporary_path / "batch_123" / "configuration"
            output_dir.mkdir(parents=True)
            update_experiment_rankings(
                backend="pytorch",
                output_dir=output_dir,
                window_size=15,
                n_clusters=1,
                clustering_method="kmeans",
                loss_function="mse",
                mae=1.0,
                mse=2.0,
                r2=0.5,
                actual=[0.0, 5.0, 25.0, 50.0],
                predicted=[0.0, 4.0, 20.0, 40.0],
                compile_pdf=False,
                ranking_dir=temporary_path / "rankings",
                progress_log=progress_path,
                progress_run_token="batch_123",
                progress_total=126,
            )
            progress_text = progress_path.read_text()
            self.assertIn("processed: 1/126", progress_text)
            self.assertIn("percentage: 0.79%", progress_text)

    def test_loss_names_include_weighting_parameters(self) -> None:
        self.assertEqual(loss_display_name("mean_squared_error"), "MSE")
        self.assertEqual(loss_display_name("mae"), "MAE")
        self.assertEqual(loss_display_name("huber"), "HUB")
        self.assertEqual(
            loss_display_name("weighted_mse_loss", loss_alpha=0.01),
            "WMSE(a=.01)",
        )
        self.assertEqual(
            loss_display_name("quantile_weighted_mse", loss_quantiles=[0.9]),
            "QWMSE(.9)",
        )

    def test_update_upserts_and_ranks_each_metric(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            ranking_dir = Path(temporary_directory)
            first_output = ranking_dir / "sweep" / "first"
            second_output = ranking_dir / "sweep" / "second"
            first_output.mkdir(parents=True)
            second_output.mkdir(parents=True)

            update_experiment_rankings(
                backend="pytorch",
                output_dir=first_output,
                window_size=15,
                n_clusters=3,
                clustering_method="kmeans",
                loss_function="mean_squared_error",
                mae=2.0,
                mse=8.0,
                r2=0.4,
                actual=[0.0, 5.0, 15.0, 25.0, 35.0, 100.0, 200.0],
                predicted=[1.0, 4.0, 14.0, 20.0, 30.0, 90.0, 180.0],
                compile_pdf=False,
                ranking_dir=ranking_dir,
            )
            csv_path, tex_path, pdf_path = update_experiment_rankings(
                backend="pytorch",
                output_dir=second_output,
                window_size=30,
                n_clusters=5,
                clustering_method="kmeans",
                loss_function="quantile_weighted_mse",
                loss_quantiles=[0.9],
                mae=1.0,
                mse=10.0,
                r2=0.7,
                actual=[0.0, 5.0, 15.0, 25.0, 35.0, 100.0, 200.0],
                predicted=[0.0, 5.0, 15.0, 25.0, 35.0, 100.0, 200.0],
                compile_pdf=False,
                ranking_dir=ranking_dir,
            )

            rankings = pd.read_csv(csv_path).set_index("run_id")
            self.assertEqual(rankings.loc["sweep/second", "mae_rank"], 1)
            self.assertEqual(rankings.loc["sweep/first", "mse_rank"], 1)
            self.assertEqual(rankings.loc["sweep/second", "r2_rank"], 1)
            self.assertEqual(rankings.loc["sweep/second", "gt_0mm_mae_rank"], 1)
            self.assertEqual(rankings.loc["sweep/second", "gt_30mm_n"], 3)
            self.assertAlmostEqual(
                rankings.loc["sweep/second", "gt_q95_threshold_mm"],
                170.0,
            )
            self.assertFalse(
                bool(rankings.loc["sweep/first", "precipitation_standardized"])
            )
            self.assertIn("QWMSE(.9)", tex_path.read_text(encoding="utf-8"))
            self.assertIsNone(pdf_path)

            update_experiment_rankings(
                backend="pytorch",
                output_dir=first_output,
                window_size=15,
                n_clusters=3,
                clustering_method="kmeans",
                loss_function="huber",
                mae=0.5,
                mse=5.0,
                r2=0.8,
                actual=[0.0, 5.0, 15.0, 25.0, 35.0, 100.0, 200.0],
                predicted=[0.0, 5.0, 15.0, 25.0, 35.0, 100.0, 200.0],
                compile_pdf=False,
                ranking_dir=ranking_dir,
            )
            updated = pd.read_csv(csv_path).set_index("run_id")
            self.assertEqual(len(updated), 2)
            self.assertEqual(updated.loc["sweep/first", "loss_function"], "HUB")
            self.assertEqual(updated.loc["sweep/first", "mae_rank"], 1)

    def test_lead_days_are_ranked_independently_and_rendered(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            ranking_dir = Path(temporary_directory)
            outputs = [ranking_dir / "sweep" / name for name in ("a", "b")]
            for output in outputs:
                output.mkdir(parents=True)

            common = {
                "backend": "pytorch",
                "window_size": 15,
                "n_clusters": 1,
                "clustering_method": "-",
                "loss_function": "mse",
                "mae": 1.0,
                "mse": 1.0,
                "r2": 0.5,
                "forecast_horizon": 2,
                "compile_pdf": False,
                "ranking_dir": ranking_dir,
            }
            update_experiment_rankings(
                output_dir=outputs[0],
                actual=[1.0, 2.0, 3.0],
                predicted=[1.0, 2.0, 3.0],
                actual_by_lead_day=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                predicted_by_lead_day=[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                **common,
            )
            _, tex_path, _ = update_experiment_rankings(
                output_dir=outputs[1],
                actual=[1.0, 2.0, 3.0],
                predicted=[1.0, 2.0, 3.0],
                actual_by_lead_day=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                predicted_by_lead_day=[[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
                **common,
            )

            horizon_path = ranking_dir / "experiment_horizon_rankings.csv"
            horizons = pd.read_csv(horizon_path)
            self.assertEqual(list(horizons.columns), HORIZON_RANKING_COLUMNS)
            indexed = horizons.set_index(["lead_day", "run_id"])
            self.assertEqual(indexed.loc[(1, "sweep/a"), "mae_rank"], 1)
            self.assertEqual(indexed.loc[(2, "sweep/b"), "mae_rank"], 1)
            self.assertEqual(indexed.loc[(1, "sweep/a"), "gt_0mm_mae_rank"], 1)
            self.assertEqual(indexed.loc[(2, "sweep/b"), "gt_0mm_mae_rank"], 1)
            tex = tex_path.read_text(encoding="utf-8")
            self.assertIn("Configured target day (D+H)", tex)
            self.assertNotIn("Forecast lead day D+1", tex)
            for lead_day in (1, 2):
                stem = f"experiment_rankings_horizon_d{lead_day:02d}"
                self.assertTrue((ranking_dir / f"{stem}.csv").exists())
                horizon_tex = (ranking_dir / f"{stem}.tex").read_text()
                self.assertIn(f"Forecast lead day D+{lead_day}", horizon_tex)
                self.assertIn(r"Actual precipitation $>30$ mm", horizon_tex)
                self.assertIn(
                    "Actual precipitation above the test-set 99th percentile",
                    horizon_tex,
                )

            generated = render_cluster_comparison_rankings(
                "pytorch",
                compile_pdf=False,
                ranking_dir=ranking_dir,
            )
            self.assertEqual(len(generated), 9)
            threshold_labels = {
                "gt_0mm": r"Actual precipitation $>0$ mm",
                "gt_10mm": r"Actual precipitation $>10$ mm",
                "gt_30mm": r"Actual precipitation $>30$ mm",
            }
            for lead_day in (1, 2):
                for group, expected_label in threshold_labels.items():
                    dedicated_path = ranking_dir / (
                        "cluster_comparison_rankings_"
                        f"horizon_d{lead_day:02d}_{group}.tex"
                    )
                    self.assertIn(dedicated_path, generated)
                    dedicated_tex = dedicated_path.read_text(encoding="utf-8")
                    self.assertIn(expected_label, dedicated_tex)
                    self.assertNotIn("Configured target day", dedicated_tex)
                    for other_label in threshold_labels.values():
                        if other_label != expected_label:
                            self.assertNotIn(other_label, dedicated_tex)

    def test_local_sync_backfills_horizons_for_an_existing_ranking(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            ranking_dir = Path(temporary_directory)
            output_dir = ranking_dir / "sweep" / "run"
            output_dir.mkdir(parents=True)
            update_experiment_rankings(
                backend="pytorch",
                output_dir=output_dir,
                window_size=15,
                n_clusters=1,
                clustering_method="-",
                loss_function="mse",
                mae=1.0,
                mse=2.0,
                r2=0.5,
                compile_pdf=False,
                ranking_dir=ranking_dir,
            )
            diagnostics = output_dir / "forecast_horizon_diagnostics"
            diagnostics.mkdir()
            pd.DataFrame(
                [
                    {
                        "lead_day": 1,
                        "forecast_horizon": 2,
                        "is_trained_target_day": True,
                        "n_test": 3,
                        "MSE": 2.0,
                        "MAE": 1.0,
                        "R2": 0.5,
                    },
                    {
                        "lead_day": 2,
                        "forecast_horizon": 2,
                        "is_trained_target_day": True,
                        "n_test": 3,
                        "MSE": 3.0,
                        "MAE": 1.5,
                        "R2": 0.25,
                    },
                ]
            ).to_csv(
                diagnostics / "test_prediction_metrics_by_lead_day.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {"lead_day": 1, "actual_mm": 0.0, "predicted_mm": 1.0},
                    {"lead_day": 1, "actual_mm": 20.0, "predicted_mm": 18.0},
                    {"lead_day": 2, "actual_mm": 5.0, "predicted_mm": 4.0},
                    {"lead_day": 2, "actual_mm": 40.0, "predicted_mm": 35.0},
                ]
            ).to_csv(
                diagnostics / "test_prediction_by_lead_day.csv",
                index=False,
            )

            _, horizon_path, tex_path, _ = synchronize_local_experiment_rankings(
                "pytorch",
                compile_pdf=False,
                ranking_dir=ranking_dir,
            )

            horizons = pd.read_csv(horizon_path)
            self.assertEqual(horizons["lead_day"].tolist(), [1, 2])
            self.assertEqual(horizons.loc[0, "gt_0mm_n"], 1)
            self.assertEqual(horizons.loc[1, "gt_0mm_n"], 2)
            self.assertNotIn("Forecast lead day D+2", tex_path.read_text())
            self.assertIn(
                "Forecast lead day D+2",
                (
                    ranking_dir / "experiment_rankings_horizon_d02.tex"
                ).read_text(),
            )

    def test_local_sync_adds_a_complete_unregistered_run(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            ranking_dir = Path(temporary_directory)
            output_dir = ranking_dir / "copied_sweep" / "station_w15_k03_kmeans"
            output_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"split": "Train", "MSE": 1.0, "MAE": 1.0, "R2": 0.8},
                    {"split": "Test", "MSE": 4.0, "MAE": 2.0, "R2": 0.2},
                ]
            ).to_csv(output_dir / "metrics_summary.csv", index=False)
            pd.DataFrame(
                {"actual": [0.0, 2.0, 4.0], "predicted": [1.0, 2.0, 3.0]}
            ).to_csv(output_dir / "test_predictions.csv", index=False)
            (output_dir / "experiment_report.tex").write_text(
                "\n".join(
                    [
                        r"\item \textbf{Window Size:} 15",
                        r"\item \textbf{Number of Clusters:} 3",
                        r"\item \textbf{Algorithm:} kmeans",
                        r"\item \textbf{Clustering Precipitation Scaler:} standard",
                        r"\item \textbf{LSTM Precipitation Scaler:} standard",
                        r"\item \textbf{LSTM layer 1 units:} 128",
                        r"\item \textbf{LSTM layer 2 units:} 32",
                        r"\item \textbf{Early stopping metric:} mae",
                        r"\item \textbf{Loss:} quantile\_weighted\_mse",
                        r"\item \textbf{Loss quantiles:} 0.9",
                    ]
                ),
                encoding="utf-8",
            )

            csv_path, _, _, _ = synchronize_local_experiment_rankings(
                "pytorch",
                compile_pdf=False,
                ranking_dir=ranking_dir,
            )

            rankings = pd.read_csv(csv_path)
            self.assertEqual(len(rankings), 1)
            self.assertEqual(rankings.loc[0, "loss_function"], "QWMSE(.9)")
            self.assertEqual(rankings.loc[0, "lstm_units"], 128)
            self.assertEqual(rankings.loc[0, "lstm_units_2"], 32)
            self.assertTrue(bool(rankings.loc[0, "precipitation_standardized"]))

    def test_table_replaces_run_with_date_and_shows_standardization(self) -> None:
        row = {column: pd.NA for column in RANKING_COLUMNS}
        row.update(
            {
                "run_id": "requested_quantile/example",
                "date": "2026-01-01",
                "window_size": 15,
                "n_clusters": 1,
                "clustering_method": "kmeans",
                "loss_function": "QWMSE(.9)",
                "patience_metric": "R2",
                "lstm_units": 128,
                "lstm_units_2": pd.NA,
                "precipitation_standardized": True,
                "n": 10,
                "mae": 1.0,
                "mae_rank": 1,
                "mse": 2.0,
                "mse_rank": 1,
                "r2": 0.5,
                "r2_rank": 1,
                "updated_at": "2026-01-01T00:00:00",
            }
        )

        table_tex = "\n".join(
            _ranking_table(pd.DataFrame([row]), "Overall", "")
        )

        self.assertIn("Date & Window", table_tex)
        self.assertIn("2026-01-01", table_tex)
        self.assertNotIn("quantile/example", table_tex)
        self.assertNotIn("Run & Window", table_tex)
        self.assertIn("$U_1$ & $U_2$", table_tex)
        self.assertIn("QWMSE(.9) & 128 & - & True", table_tex)
        self.assertIn(" & 1 & - & QWMSE(.9)", table_tex)
        self.assertNotIn(" & kmeans & ", table_tex)
        self.assertNotIn("Patience", table_tex)
        self.assertNotIn(r"$R^2$", table_tex)
        self.assertIn(r"\begin{longtable}{@{}lrrllrrc", table_tex)
        self.assertIn(r"\setlength{\tabcolsep}{3pt}", table_tex)
        self.assertNotIn(r"p{", table_tex)

    def test_cluster_comparison_varies_clustering_and_k_by_row(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            ranking_dir = Path(temporary_directory)
            common = {
                "backend": "pytorch",
                "window_size": 15,
                "loss_function": "quantile_weighted_mse",
                "loss_quantiles": [0.9],
                "patience_metric": "mae",
                "lstm_units": 128,
                "lstm_units_2": 32,
                "precipitation_standardized": True,
                "actual": [0.0, 10.0, 20.0],
                "predicted": [0.0, 9.0, 18.0],
                "compile_pdf": False,
                "ranking_dir": ranking_dir,
            }
            for k_value, method, mae in (
                (1, "-", 2.0),
                (3, "spectral(1.5)", 1.0),
                (5, "spectral(1.5)", 1.5),
            ):
                output_dir = ranking_dir / "sweep" / f"k{k_value}"
                output_dir.mkdir(parents=True)
                update_experiment_rankings(
                    output_dir=output_dir,
                    n_clusters=k_value,
                    clustering_method=method,
                    mae=mae,
                    mse=mae * mae,
                    r2=1.0 - mae / 10.0,
                    **common,
                )

            generated = render_cluster_comparison_rankings(
                "pytorch",
                compile_pdf=False,
                ranking_dir=ranking_dir,
            )

            self.assertEqual(generated, [ranking_dir / "cluster_comparison_rankings.tex"])
            tex = generated[0].read_text(encoding="utf-8")
            self.assertIn("Window 15; QWMSE(.9); patience MAE", tex)
            self.assertIn("Clustering & $K$ & MAE & MSE", tex)
            self.assertNotIn("Loss & Patience", tex)
            self.assertNotIn(r"MSE & $R^2$", tex)
            k1_row = "QWMSE(.9) & 128 & 32 & True & 15 & - & 1"
            k3_row = (
                "QWMSE(.9) & 128 & 32 & True & 15 & "
                "spectral(1.5) & 3"
            )
            k5_row = (
                "QWMSE(.9) & 128 & 32 & True & 15 & "
                "spectral(1.5) & 5"
            )
            self.assertIn(k1_row, tex)
            self.assertIn(k3_row, tex)
            self.assertIn(k5_row, tex)
            self.assertLess(tex.index(k1_row), tex.index(k3_row))
            self.assertLess(tex.index(k3_row), tex.index(k5_row))
            self.assertIn(r"\definecolor{rankingbestgreen}{HTML}{228B22}", tex)
            self.assertIn(
                r"\textcolor{rankingbestgreen}{\textbf{1.0000}}",
                tex,
            )

    def test_dedicated_comparison_uses_only_its_threshold_metrics(self) -> None:
        row = {column: pd.NA for column in RANKING_COLUMNS}
        row.update(
            {
                "run_id": "sweep/threshold-only",
                "date": "2026-01-01",
                "window_size": 15,
                "n_clusters": 2,
                "clustering_method": "kmeans",
                "loss_function": "MSE",
                "patience_metric": "MAE",
                "lstm_units": 64,
                "lstm_units_2": 32,
                "precipitation_standardized": False,
                "mae": 987.6543,
                "mse": 876.5432,
                "r2": -9.8765,
                "gt_10mm_mae": 1.2345,
                "gt_10mm_mse": 2.3456,
                "gt_10mm_r2": 0.4567,
            }
        )
        label = r"Actual precipitation $>10$ mm"

        tex = experiment_rankings._cluster_comparison_tex(
            pd.DataFrame([row], columns=RANKING_COLUMNS),
            "pytorch",
            lead_day=1,
            sections=[(label, "gt_10mm_")],
            scope_label=label,
        )

        self.assertIn(
            r"\textcolor{rankingbestgreen}{\textbf{1.2345}}",
            tex,
        )
        self.assertIn(
            r"\textcolor{rankingbestgreen}{\textbf{2.3456}}",
            tex,
        )
        self.assertNotIn("0.4567", tex)
        self.assertNotIn("987.6543", tex)
        self.assertNotIn("876.5432", tex)
        self.assertNotIn("-9.8765", tex)

    def test_window_loss_triptych_places_three_thresholds_per_horizon(self) -> None:
        rows = []
        for lead_day in range(1, 6):
            for n_clusters, method, error in (
                (1, "-", 2.0),
                (3, "kmeans", 1.0),
            ):
                row = {column: pd.NA for column in HORIZON_RANKING_COLUMNS}
                row.update(
                    {
                        "run_id": f"sweep/d{lead_day}_k{n_clusters}",
                        "run_path": f"sweep/d{lead_day}_k{n_clusters}",
                        "date": "2026-10-01",
                        "window_size": 30,
                        "n_clusters": n_clusters,
                        "clustering_method": method,
                        "loss_function": "QWMSE(.9)",
                        "patience_metric": "MAE",
                        "lstm_units": 64,
                        "lstm_units_2": pd.NA,
                        "precipitation_standardized": False,
                        "lead_day": lead_day,
                        "forecast_horizon": 5,
                        "is_trained_target_day": True,
                        "gt_0mm_mae": error,
                        "gt_0mm_mse": error**2,
                        "gt_10mm_mae": error + 0.1,
                        "gt_10mm_mse": (error + 0.1) ** 2,
                        "gt_30mm_mae": error + 0.3,
                        "gt_30mm_mse": (error + 0.3) ** 2,
                        "updated_at": "2026-10-01T00:00:00",
                    }
                )
                rows.append(row)
        excluded = rows[0].copy()
        excluded.update(
            {
                "run_id": "excluded/w15",
                "run_path": "excluded/w15",
                "window_size": 15,
                "gt_0mm_mae": 99.0,
            }
        )
        rows.append(excluded)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            ranking_dir = Path(temporary_directory)
            pd.DataFrame(rows, columns=HORIZON_RANKING_COLUMNS).to_csv(
                ranking_dir / "experiment_horizon_rankings.csv",
                index=False,
            )
            tex_path, pdf_path = render_horizon_threshold_triptych_rankings(
                "pytorch",
                window_size=30,
                loss_function="QWMSE(.9)",
                compile_pdf=False,
                ranking_dir=ranking_dir,
            )

            self.assertIsNone(pdf_path)
            self.assertEqual(
                tex_path.name,
                "cluster_comparison_rankings_w30_qwmse_0p9_rain_thresholds.tex",
            )
            tex = tex_path.read_text(encoding="utf-8")
            for lead_day in range(1, 6):
                self.assertIn(f"Rain-threshold Comparisons --- D+{lead_day}", tex)
            self.assertEqual(tex.count(r"\begin{minipage}"), 15)
            self.assertIn("window=30; loss=QWMSE(.9)", tex)
            self.assertIn("precipitation standardized=False", tex)
            self.assertNotIn("Std. & Clustering", tex)
            self.assertNotIn(r"Network ($U_1/U_2$)", tex)
            self.assertIn(r"Actual precipitation $>0$ mm", tex)
            self.assertIn(r"Actual precipitation $>10$ mm", tex)
            self.assertIn(r"Actual precipitation $>30$ mm", tex)
            self.assertIn(
                r"\textcolor{rankingbestgreen}{\textbf{1.0000}}",
                tex,
            )
            self.assertNotIn("99.0000", tex)

    def test_architecture_triptych_varies_window_in_one_report(self) -> None:
        rows = []
        for lead_day in range(1, 6):
            for window_size, error in ((15, 1.5), (30, 1.0)):
                row = {column: pd.NA for column in HORIZON_RANKING_COLUMNS}
                row.update(
                    {
                        "run_id": f"sweep/d{lead_day}_w{window_size}",
                        "run_path": f"sweep/d{lead_day}_w{window_size}",
                        "date": "2026-10-01",
                        "window_size": window_size,
                        "n_clusters": 1,
                        "clustering_method": "-",
                        "loss_function": "QWMSE(.9)",
                        "patience_metric": "MAE",
                        "lstm_units": 64,
                        "lstm_units_2": pd.NA,
                        "precipitation_standardized": False,
                        "lead_day": lead_day,
                        "forecast_horizon": 5,
                        "is_trained_target_day": True,
                        "gt_0mm_mae": error,
                        "gt_0mm_mse": error**2,
                        "gt_10mm_mae": error + 0.1,
                        "gt_10mm_mse": (error + 0.1) ** 2,
                        "gt_30mm_mae": error + 0.3,
                        "gt_30mm_mse": (error + 0.3) ** 2,
                        "updated_at": "2026-10-01T00:00:00",
                    }
                )
                rows.append(row)
        excluded = rows[0].copy()
        excluded.update(
            {
                "run_id": "excluded/u1_128",
                "run_path": "excluded/u1_128",
                "lstm_units": 128,
                "gt_0mm_mae": 99.0,
            }
        )
        rows.append(excluded)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            ranking_dir = Path(temporary_directory)
            pd.DataFrame(rows, columns=HORIZON_RANKING_COLUMNS).to_csv(
                ranking_dir / "experiment_horizon_rankings.csv",
                index=False,
            )
            tex_path, pdf_path = render_architecture_threshold_triptych_rankings(
                "pytorch",
                lstm_units=64,
                lstm_units_2=None,
                loss_function="QWMSE(.9)",
                compile_pdf=False,
                ranking_dir=ranking_dir,
            )

            self.assertIsNone(pdf_path)
            self.assertEqual(
                tex_path.name,
                "cluster_comparison_rankings_u1_64_u2_none_"
                "qwmse_0p9_rain_thresholds.tex",
            )
            tex = tex_path.read_text(encoding="utf-8")
            self.assertEqual(tex.count(r"\begin{minipage}"), 15)
            self.assertIn(r"$U_1$=64; $U_2$=-", tex)
            self.assertIn("Window & Clustering & $K$ & MAE & MSE", tex)
            self.assertNotIn(r"Network ($U_1/U_2$)", tex)
            self.assertIn("15 & - & 1", tex)
            self.assertIn("30 & - & 1", tex)
            self.assertNotIn("99.0000", tex)

    def test_each_table_keeps_only_its_best_spectral_sigma_per_k(self) -> None:
        rows = []
        for method, overall_mae, wet_mae in (
            ("spectral(1.0)", 1.0, 5.0),
            ("spectral(2.0)", 2.0, 1.0),
            ("kmeans", 3.0, 3.0),
        ):
            row = {column: pd.NA for column in RANKING_COLUMNS}
            row.update(
                {
                    "run_id": f"sweep/{method}",
                    "date": "2026-01-01",
                    "window_size": 15,
                    "n_clusters": 3,
                    "clustering_method": method,
                    "loss_function": "MSE",
                    "patience_metric": "MAE",
                    "lstm_units": 64,
                    "lstm_units_2": 32,
                    "precipitation_standardized": False,
                    "n": 100,
                    "mae": overall_mae,
                    "mse": overall_mae**2,
                    "r2": 0.5,
                    "gt_10mm_threshold_mm": 10.0,
                    "gt_10mm_n": 20,
                    "gt_10mm_mae": wet_mae,
                    "gt_10mm_mse": wet_mae**2,
                    "gt_10mm_r2": 0.4,
                    "updated_at": "2026-01-01T00:00:00",
                }
            )
            rows.append(row)
        rankings = pd.DataFrame(rows, columns=RANKING_COLUMNS)

        overall_tex = "\n".join(_ranking_table(rankings, "Overall", ""))
        wet_tex = "\n".join(
            _ranking_table(rankings, "Wet", "gt_10mm_")
        )
        comparison_tex = "\n".join(
            experiment_rankings._cluster_comparison_tables(
                rankings,
                "gt_10mm_",
            )
        )

        self.assertIn("spectral(1.0)", overall_tex)
        self.assertNotIn("spectral(2.0)", overall_tex)
        self.assertIn("spectral(2.0)", wet_tex)
        self.assertNotIn("spectral(1.0)", wet_tex)
        self.assertIn("spectral(2.0)", comparison_tex)
        self.assertNotIn("spectral(1.0)", comparison_tex)
        self.assertIn("kmeans", overall_tex)
        self.assertIn("kmeans", wet_tex)
        self.assertIn("kmeans", comparison_tex)

    def test_landscape_starts_before_ranking_heading(self) -> None:
        tex = experiment_rankings._ranking_tex(
            pd.DataFrame(columns=RANKING_COLUMNS), "pytorch"
        )

        self.assertLess(tex.index(r"\begin{landscape}"), tex.index(r"\section*"))

    def test_table_omits_pytorch_sweep_name(self) -> None:
        row = {column: pd.NA for column in RANKING_COLUMNS}
        row.update(
            {
                "run_id": "pytorch_lstm_cluster_sweep_RS_A801_2026/run",
                "date": "2026-01-01",
                "window_size": 15,
                "n_clusters": 1,
                "clustering_method": "spectral(10.7)",
                "loss_function": "MSE",
                "patience_metric": "MAE",
                "lstm_units": 128,
                "precipitation_standardized": False,
                "n": 10,
                "mae": 1.0,
                "mae_rank": 1,
                "mse": 2.0,
                "mse_rank": 1,
                "r2": 0.5,
                "r2_rank": 1,
                "updated_at": "2026-01-01T00:00:00",
            }
        )

        table_tex = "\n".join(
            _ranking_table(pd.DataFrame([row]), "Overall", "")
        )

        self.assertNotIn("RS\\_A801\\_2026/run", table_tex)
        self.assertNotIn("pytorch\\_lstm\\_cluster\\_sweep", table_tex)

    def test_legacy_rows_default_to_unstandardized_precipitation(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            csv_path = Path(temporary_directory) / "experiment_rankings.csv"
            pd.DataFrame(
                [
                    {
                        "run_id": "legacy/run_u1_128_u2_32",
                        "updated_at": "2025-12-31T10:00:00",
                    }
                ]
            ).to_csv(csv_path, index=False)

            rankings = experiment_rankings._load_rankings(csv_path)

            self.assertEqual(rankings.loc[0, "date"], "2025-12-31")
            self.assertEqual(rankings.loc[0, "lstm_units_2"], 32)
            self.assertFalse(bool(rankings.loc[0, "precipitation_standardized"]))

    def test_locked_tex_file_does_not_abort_ranking_update(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            target = Path(temporary_directory) / "rankings.tex"
            target.write_text("old", encoding="utf-8")

            with patch.object(
                Path,
                "replace",
                side_effect=PermissionError("locked for test"),
            ):
                experiment_rankings._write_text_atomically("new", target)

            self.assertEqual(target.read_text(encoding="utf-8"), "old")
            self.assertEqual(
                target.with_suffix(".tex.tmp").read_text(encoding="utf-8"),
                "new",
            )

    def test_locked_csv_file_does_not_abort_ranking_update(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            target = Path(temporary_directory) / "rankings.csv"
            pd.DataFrame({"value": [1]}).to_csv(target, index=False)

            with patch.object(
                Path,
                "replace",
                side_effect=PermissionError("locked for test"),
            ):
                experiment_rankings._write_csv_atomically(
                    pd.DataFrame({"value": [2]}),
                    target,
                )

            self.assertEqual(pd.read_csv(target).loc[0, "value"], 1)
            self.assertEqual(
                pd.read_csv(target.with_suffix(".csv.tmp")).loc[0, "value"],
                2,
            )

    def test_concurrent_updates_keep_and_rerank_every_experiment(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            ranking_dir = Path(temporary_directory)
            outputs = [ranking_dir / "sweep" / f"run-{index}" for index in range(6)]
            for output in outputs:
                output.mkdir(parents=True)

            original_load = experiment_rankings._load_rankings

            def delayed_load(csv_path: Path) -> pd.DataFrame:
                time.sleep(0.02)
                return original_load(csv_path)

            def add_run(index: int) -> None:
                update_experiment_rankings(
                    backend="pytorch",
                    output_dir=outputs[index],
                    window_size=15,
                    n_clusters=3,
                    clustering_method="kmeans",
                    loss_function="mse",
                    mae=float(6 - index),
                    mse=float(12 - index),
                    r2=float(index) / 10.0,
                    actual=[0.0, 10.0, 20.0, 40.0],
                    predicted=[0.0, 9.0, 18.0, 36.0],
                    compile_pdf=False,
                    ranking_dir=ranking_dir,
                )

            with patch.object(
                experiment_rankings,
                "_load_rankings",
                side_effect=delayed_load,
            ):
                with ThreadPoolExecutor(max_workers=len(outputs)) as executor:
                    list(executor.map(add_run, range(len(outputs))))

            rankings = pd.read_csv(
                ranking_dir / "experiment_rankings.csv"
            ).set_index("run_id")
            self.assertEqual(len(rankings), len(outputs))
            for index in range(len(outputs)):
                run_id = f"sweep/run-{index}"
                expected_rank = 6 - index
                self.assertEqual(rankings.loc[run_id, "mae_rank"], expected_rank)
                self.assertEqual(rankings.loc[run_id, "mse_rank"], expected_rank)
                self.assertEqual(rankings.loc[run_id, "r2_rank"], expected_rank)


if __name__ == "__main__":
    unittest.main()
