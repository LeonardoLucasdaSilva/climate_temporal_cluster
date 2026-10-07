"""Tests for compact final-experiment persistence."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.final_experiment_storage import (  # noqa: E402
    compact_result_arrays,
    compact_result_complete,
    decode_histories,
    load_metadata,
    save_compact_result,
    save_shared_assignment,
    update_compact_ranking,
    write_npz_atomic,
)


class FinalExperimentStorageTests(unittest.TestCase):
    def test_compact_result_has_two_committed_configuration_files(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            run_dir = Path(directory) / "run"
            arrays, history_index = compact_result_arrays(
                y_pred_train=np.array([1.0, 2.0]),
                y_pred_train_by_lead_day=np.array([[1.0], [2.0]]),
                y_pred_val=np.array([3.0]),
                y_pred_test=np.array([4.0]),
                y_pred_test_by_lead_day=np.array([[4.0]]),
                histories_by_cluster={0: {"loss": [3.0, 2.0]}},
            )

            save_compact_result(
                run_dir,
                {"parameters": {}, "history_index": history_index},
                arrays,
            )

            self.assertTrue(compact_result_complete(run_dir))
            self.assertEqual(
                sorted(path.name for path in run_dir.iterdir()),
                ["metadata.json", "result.npz"],
            )
            self.assertEqual(load_metadata(run_dir)["status"], "completed")
            with np.load(run_dir / "result.npz", allow_pickle=False) as archive:
                self.assertEqual(
                    decode_histories(archive, history_index),
                    {0: {"loss": [3.0, 2.0]}},
                )

    def test_cluster_assignments_are_shared_and_not_rewritten(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            final_dir = Path(directory)
            path = save_shared_assignment(
                final_dir,
                "labels",
                np.array([0, 1]),
                np.array([1]),
                np.array([0]),
                {"n_clusters": 2},
            )
            original_timestamp = path.stat().st_mtime_ns

            repeated = save_shared_assignment(
                final_dir,
                "labels",
                np.array([1, 1]),
                np.array([0]),
                np.array([1]),
                {"n_clusters": 2},
            )

            self.assertEqual(path, repeated)
            self.assertEqual(path.stat().st_mtime_ns, original_timestamp)
            with np.load(path, allow_pickle=False) as labels:
                np.testing.assert_array_equal(labels["c_train"], [0, 1])

    def test_compact_result_updates_overall_and_horizon_rankings(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            final_dir = Path(directory) / "final_experiment_sweep_1"
            run_dir = final_dir / "compact_results" / "batch" / "run"
            targets_path = final_dir / "_shared" / "targets" / "base.npz"
            write_npz_atomic(
                targets_path,
                {
                    "y_test": np.array([1.0, 2.0, 3.0]),
                    "y_test_by_lead_day": np.array(
                        [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
                    ),
                },
            )
            arrays, history_index = compact_result_arrays(
                y_pred_train=np.array([0.0]),
                y_pred_train_by_lead_day=np.array([[0.0, 0.0]]),
                y_pred_val=np.array([0.0]),
                y_pred_test=np.array([1.0, 2.0, 2.0]),
                y_pred_test_by_lead_day=np.array(
                    [[0.0, 1.0], [1.0, 2.0], [2.0, 2.0]]
                ),
                histories_by_cluster={0: {}},
            )
            save_compact_result(
                run_dir,
                {
                    "parameters": {
                        "window_size": 15,
                        "n_clusters": 1,
                        "clustering_method": "-",
                        "lstm_loss_function": "quantile_weighted_mse",
                        "loss_quantiles": [0.9],
                        "forecast_horizon": 2,
                        "early_stopping_metric": "mae",
                        "lstm_units": 64,
                        "lstm_units_2": 32,
                        "precipitation_standardized": False,
                    },
                    "metrics": {
                        "test_mae": 1 / 3,
                        "test_mse": 1 / 3,
                        "test_r2": 0.5,
                    },
                    "history_index": history_index,
                    "shared_targets": targets_path.relative_to(final_dir).as_posix(),
                },
                arrays,
            )

            update_compact_ranking(final_dir, run_dir, load_metadata(run_dir))

            ranking = np.genfromtxt(
                final_dir / "experiment_rankings.csv",
                delimiter=",",
                names=True,
                dtype=None,
                encoding="utf-8",
            )
            self.assertEqual(int(ranking["n_clusters"]), 1)
            self.assertEqual(int(ranking["lstm_units_2"]), 32)
            self.assertTrue(
                (final_dir / "experiment_rankings_horizon_d01.csv").is_file()
            )
            self.assertTrue(
                (final_dir / "experiment_rankings_horizon_d02.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
