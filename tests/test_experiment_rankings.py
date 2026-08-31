"""Tests for persistent backend-specific experiment rankings."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import data.experiment_rankings as experiment_rankings
from data.experiment_rankings import (
    RANKING_COLUMNS,
    _ranking_table,
    initialize_experiment_progress,
    initialize_experiment_rankings,
    loss_display_name,
    update_experiment_rankings,
)


class ExperimentRankingsTests(unittest.TestCase):
    def test_threshold_table_uses_its_own_rank_order(self) -> None:
        rows = []
        for run_id, overall_rank, threshold_rank in (
            ("sweep/overall-first", 1, 2),
            ("sweep/threshold-first", 2, 1),
        ):
            row = {column: pd.NA for column in RANKING_COLUMNS}
            row.update(
                {
                    "run_id": run_id,
                    "window_size": 15,
                    "n_clusters": 1,
                    "clustering_method": "kmeans",
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
            overall_tex.index("overall-first"),
            overall_tex.index("threshold-first"),
        )
        self.assertLess(
            threshold_tex.index("threshold-first"),
            threshold_tex.index("overall-first"),
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

        self.assertLess(table_tex.index("older"), table_tex.index("newer"))

    def test_initialize_creates_empty_backend_files(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            csv_path, tex_path, pdf_path = initialize_experiment_rankings(
                "tensorflow",
                compile_pdf=False,
                ranking_dir=Path(temporary_directory),
            )
            self.assertTrue(csv_path.exists())
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

    def test_table_compacts_run_name_and_shows_training_metadata(self) -> None:
        row = {column: pd.NA for column in RANKING_COLUMNS}
        row.update(
            {
                "run_id": "requested_quantile/example",
                "window_size": 15,
                "n_clusters": 1,
                "clustering_method": "kmeans",
                "loss_function": "QWMSE(.9)",
                "patience_metric": "R2",
                "lstm_units": 128,
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

        self.assertIn("quantile/example", table_tex)
        self.assertNotIn("requested", table_tex)
        self.assertIn("QWMSE(.9) & R2 & 128", table_tex)

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
