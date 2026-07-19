"""Tests for custom LSTM loss functions without requiring TensorFlow."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LSTM_MODULE_PATH = PROJECT_ROOT / "src" / "models" / "lstm.py"
RUNNER_MODULE_PATH = (
    PROJECT_ROOT / "src" / "methods" / "lstm_cluster" / "run_experiment.py"
)


def _load_lstm_module_with_numpy_backend():
    """Load models.lstm with the TensorFlow operations needed by loss tests."""
    tensorflow_stub = types.ModuleType("tensorflow")
    tensorflow_stub.float32 = np.float32
    tensorflow_stub.cast = lambda value, dtype: np.asarray(value, dtype=dtype)
    tensorflow_stub.square = np.square
    tensorflow_stub.reduce_mean = np.mean
    tensorflow_stub.keras = types.SimpleNamespace(
        layers=types.SimpleNamespace(),
    )

    spec = importlib.util.spec_from_file_location(
        "_weighted_mse_lstm_under_test",
        LSTM_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load src/models/lstm.py for testing.")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"tensorflow": tensorflow_stub}):
        spec.loader.exec_module(module)
    return module


class WeightedMseLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lstm = _load_lstm_module_with_numpy_backend()

    def test_weighted_mse_loss_matches_requested_formula(self) -> None:
        y_real = np.array([0.0, 2.0])
        y_pred = np.array([1.0, 1.0])

        loss = self.lstm.weighted_mse_loss(y_real, y_pred, alpha=0.5)

        self.assertAlmostEqual(float(loss), 1.5)

    def test_weighted_mse_loss_reduces_horizon_per_batch_sample(self) -> None:
        y_real = np.array([[0.0, 2.0], [4.0, 1.0]], dtype=np.float64)
        y_pred = np.zeros_like(y_real)
        alpha = 0.25
        expected = np.mean(
            (1.0 + alpha * y_real) * np.square(y_real - y_pred),
            axis=-1,
        )

        loss = self.lstm.weighted_mse_loss(y_real, y_pred, alpha)

        np.testing.assert_allclose(loss, expected)

    def test_weighted_mse_loss_is_zero_for_exact_predictions(self) -> None:
        y_real = np.array([0, 1, 5])

        loss = self.lstm.weighted_mse_loss(y_real, y_real, alpha=2.0)

        self.assertEqual(float(loss), 0.0)

    def test_weighted_mse_loss_rejects_non_positive_or_non_finite_alpha(self) -> None:
        for alpha in (0.0, -1.0, np.nan, np.inf, -np.inf, True, "invalid"):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(ValueError, "alpha.*positive"):
                    self.lstm.weighted_mse_loss([1.0], [0.0], alpha)

    def test_resolver_binds_alpha_for_keras(self) -> None:
        loss_function = self.lstm.resolve_loss_function(
            "WEIGHTED_MSE_LOSS",
            weighted_mse_alpha=0.5,
        )

        self.assertTrue(callable(loss_function))
        self.assertEqual(loss_function.__name__, "weighted_mse_loss")
        self.assertAlmostEqual(float(loss_function([0.0, 2.0], [1.0, 1.0])), 1.5)

    def test_resolver_requires_alpha_for_weighted_mse(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires alpha"):
            self.lstm.resolve_loss_function("weighted_mse_loss")

    def test_experiment_runner_uses_weighted_mse_as_default(self) -> None:
        module_tree = ast.parse(
            RUNNER_MODULE_PATH.read_text(encoding="utf-8"),
            filename=str(RUNNER_MODULE_PATH),
        )
        constants = {}
        for node in module_tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    constants[target.id] = ast.literal_eval(node.value)
                except (TypeError, ValueError):
                    continue

        self.assertEqual(constants["LSTM_LOSS_FUNCTION"], "weighted_mse_loss")
        self.assertGreater(constants["LOSS_ALPHA"], 0)


if __name__ == "__main__":
    unittest.main()
