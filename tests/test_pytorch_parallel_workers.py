"""Tests for bounded PyTorch cluster-training parallelism."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from methods.lstm_cluster_pytorch.pipeline import (
    ClusterTrainingResult,
    _parallel_worker_count,
    train_cluster_models,
)


class PytorchParallelWorkerTests(unittest.TestCase):
    @patch("methods.lstm_cluster_pytorch.pipeline.os.cpu_count", return_value=12)
    def test_worker_count_honors_explicit_cap(self, _cpu_count: object) -> None:
        self.assertEqual(_parallel_worker_count(9, max_workers=2), 2)
        self.assertEqual(_parallel_worker_count(1, max_workers=2), 1)

    def test_worker_count_rejects_invalid_cap(self) -> None:
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    _parallel_worker_count(5, max_workers=value)

    @patch("methods.lstm_cluster_pytorch.pipeline._train_single_cluster_model")
    def test_validation_lead_predictions_are_optionally_returned(
        self,
        train_single: object,
    ) -> None:
        train_single.return_value = ClusterTrainingResult(
            cluster_id=0,
            train_mask=np.array([True, True]),
            val_mask=np.array([True, True]),
            test_mask=np.array([True, True]),
            train_predictions_scaled=np.array([[1.0, 2.0], [3.0, 4.0]]),
            val_predictions_scaled=np.array([[5.0, 6.0], [7.0, 8.0]]),
            test_predictions_scaled=None,
            all_test_predictions_scaled=None,
            history={"loss": [1.0]},
            test_metrics=None,
        )
        common = {
            "X_train": np.ones((2, 2)),
            "X_val": np.ones((2, 2)),
            "X_test": np.ones((2, 2)),
            "y_train": np.array([2.0, 4.0]),
            "y_val": np.array([6.0, 8.0]),
            "y_test": np.array([0.0, 0.0]),
            "y_train_by_lead_day": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "y_val_by_lead_day": np.array([[5.0, 6.0], [7.0, 8.0]]),
            "y_test_by_lead_day": np.zeros((2, 2)),
            "y_train_by_lead_day_scaled": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "y_val_by_lead_day_scaled": np.array([[5.0, 6.0], [7.0, 8.0]]),
            "c_train": np.zeros(2, dtype=int),
            "c_val": np.zeros(2, dtype=int),
            "c_test": np.zeros(2, dtype=int),
            "sequence_length": 1,
            "lstm_units": 4,
            "lstm_units_2": None,
            "dropout_rate": 0.0,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "epochs": 1,
            "batch_size": 1,
            "early_stopping": False,
            "patience": 1,
            "warm_up": 0,
            "early_stopping_metric": "mae",
            "lstm_loss_function": "mse",
            "loss_quantiles": [],
            "loss_quantile_weights": "auto",
            "verbose_training": 0,
            "random_state": 42,
            "show_console_info": False,
            "test_all_models": False,
            "evaluate_test": False,
        }
        legacy = train_cluster_models(**common)
        extended = train_cluster_models(
            **common,
            return_validation_by_lead_day=True,
        )

        self.assertEqual(len(legacy), 8)
        self.assertEqual(len(extended), 9)
        np.testing.assert_allclose(extended[2], [6.0, 8.0])
        np.testing.assert_allclose(extended[3], [[5.0, 6.0], [7.0, 8.0]])


if __name__ == "__main__":
    unittest.main()
