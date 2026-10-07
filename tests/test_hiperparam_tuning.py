from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from methods.lstm_cluster.hiperparam_tuning import (  # noqa: E402
    candidate_count,
    generate_trial_hyperparameters,
    optimization_spec,
    run_hyperparameter_tuning,
    single_trial_base_kwargs,
)
from methods.lstm_cluster.pipeline import _train_single_cluster_model  # noqa: E402


def _base_experiment_kwargs() -> dict[str, object]:
    return {
        "window_sizes": [15, 30],
        "n_clusters_list": [3, 5],
        "lstm_units": [64, 128],
        "lstm_units_2": [None, 32],
        "dropout_rate": [0.1, 0.2],
        "learning_rate": [1e-4, 1e-3],
        "weight_decay": [0.0, 1e-4],
        "epochs": [10, 20],
        "batch_size": [4, 8],
        "patience": [2, 4],
        "warm_up": [0, 2],
        "clustering_algorithm": ["kmeans", "manual"],
        "sigma_values": [0.3, 0.5],
        "run_only_cluster": False,
        "comparative_run": True,
    }


class HyperparameterTuningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work_dir = (
            PROJECT_ROOT / "tests" / f"_hiperparam_tuning_test_{uuid4().hex}"
        )

    def tearDown(self) -> None:
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)

    def test_random_search_is_deterministic_and_budgeted(self) -> None:
        search_space = {
            "window_size": [10, 20, 30],
            "learning_rate": [1e-4, 1e-3],
        }

        first = generate_trial_hyperparameters(
            search_space,
            strategy="random",
            n_trials=4,
            random_state=7,
        )
        second = generate_trial_hyperparameters(
            search_space,
            strategy="random",
            n_trials=4,
            random_state=7,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(len({tuple(row.items()) for row in first}), 4)
        self.assertEqual(candidate_count(search_space), 6)

    def test_search_space_accepts_a_supported_subset(self) -> None:
        trials = generate_trial_hyperparameters(
            {"dropout_rate": [0.1, 0.3]},
            strategy="grid",
            n_trials=None,
            random_state=42,
        )

        self.assertEqual(trials, [{"dropout_rate": 0.1}, {"dropout_rate": 0.3}])

    def test_optimization_spec_always_uses_validation(self) -> None:
        self.assertEqual(optimization_spec("RMSE"), ("RMSE", "val_rmse", "minimize"))
        self.assertEqual(optimization_spec("val_r2"), ("R2", "val_r2", "maximize"))

    def test_single_trial_base_collapses_existing_grids(self) -> None:
        parameters = single_trial_base_kwargs(_base_experiment_kwargs())

        self.assertEqual(parameters["window_sizes"], [15])
        self.assertEqual(parameters["n_clusters_list"], [3])
        self.assertEqual(parameters["lstm_units"], 64)
        self.assertIsNone(parameters["lstm_units_2"])
        self.assertEqual(parameters["clustering_algorithm"], "kmeans")
        self.assertEqual(parameters["sigma_values"], [0.3])
        self.assertFalse(parameters["comparative_run"])
        self.assertFalse(parameters["test_all_models"])
        self.assertFalse(parameters["create_report"])
        self.assertFalse(parameters["train_info"])
        self.assertFalse(parameters["silhouette_info"])
        self.assertTrue(parameters["validation_only"])

    def test_tuning_selects_validation_best_and_writes_checkpoints(self) -> None:
        runner_calls: list[dict[str, object]] = []

        def fake_runner(**kwargs: object) -> Path:
            runner_calls.append(dict(kwargs))
            sweep_dir = Path(kwargs["output_root"]) / str(kwargs["sweep_name"])
            sweep_dir.mkdir(parents=True)
            window_size = int(kwargs["window_sizes"][0])
            pd.DataFrame(
                [
                    {
                        "run_name": f"window_{window_size}",
                        "window_size": window_size,
                        "val_rmse": float(window_size),
                        "test_rmse": float(100 - window_size),
                    }
                ]
            ).to_csv(sweep_dir / "sweep_results.csv", index=False)
            return sweep_dir

        ranking_dir = self.work_dir.parent / f"_rankings_{uuid4().hex}"
        self.addCleanup(shutil.rmtree, ranking_dir, True)
        output_dir = run_hyperparameter_tuning(
            base_experiment_kwargs=_base_experiment_kwargs(),
            search_space={"window_size": [20, 10]},
            output_dir=self.work_dir,
            strategy="grid",
            n_trials=None,
            optimization_metric="RMSE",
            random_state=42,
            show_progress=False,
            experiment_runner=fake_runner,
            ranking_dir=ranking_dir,
            search_id="unit_search",
            compile_ranking_pdf=False,
        )

        results = pd.read_csv(output_dir / "tuning_results.csv")
        rankings = pd.read_csv(
            ranking_dir / "random_search_validation_rankings.csv"
        )
        best = json.loads(
            (output_dir / "best_hyperparameters.json").read_text(encoding="utf-8")
        )
        self.assertEqual(results.iloc[0]["window_size"], 10)
        self.assertNotIn("test_rmse", results.columns)
        self.assertNotIn("test_rmse", rankings.columns)
        self.assertEqual(rankings.iloc[0]["validation_rank"], 1)
        self.assertTrue(
            (ranking_dir / "random_search_validation_rankings.tex").is_file()
        )
        ranking_tex = (
            ranking_dir / "random_search_validation_rankings.tex"
        ).read_text(encoding="utf-8")
        self.assertIn("Validation-only Rankings", ranking_tex)
        self.assertIn(r"\textcolor{green!50!black}", ranking_tex)
        self.assertEqual(best["hyperparameters"]["window_size"], 10)
        self.assertEqual(best["objective_column"], "val_rmse")
        self.assertEqual(len(runner_calls), 2)
        for call in runner_calls:
            self.assertEqual(len(call["window_sizes"]), 1)
            self.assertEqual(len(call["n_clusters_list"]), 1)
            self.assertFalse(call["comparative_run"])
            self.assertFalse(call["test_all_models"])
            self.assertTrue(call["validation_only"])

    def test_validation_only_training_never_predicts_test_rows(self) -> None:
        model = MagicMock()
        model.fit.return_value = {"loss": [1.0]}
        model.predict.side_effect = lambda values: np.zeros((len(values), 1))
        predictor = MagicMock(return_value=model)
        fake_lstm_module = ModuleType("models.lstm")
        fake_lstm_module.LSTMPrecipitationPredictor = predictor

        with patch.dict(sys.modules, {"models.lstm": fake_lstm_module}), patch(
            "methods.lstm_cluster.pipeline._set_tensorflow_worker_threads"
        ):
            result = _train_single_cluster_model(
                cluster_id=0,
                X_train_lstm=np.zeros((2, 3, 1)),
                X_val_lstm=np.zeros((1, 3, 1)),
                X_test_lstm=np.zeros((2, 3, 1)),
                y_train=np.zeros(2),
                y_test=np.zeros(2),
                y_train_by_lead_day_scaled=np.zeros((2, 1)),
                y_val_by_lead_day_scaled=np.zeros((1, 1)),
                c_train=np.zeros(2, dtype=int),
                c_val=np.zeros(1, dtype=int),
                c_test=np.zeros(2, dtype=int),
                lstm_units=4,
                lstm_units_2=None,
                dropout_rate=0.1,
                learning_rate=1e-3,
                weight_decay=0.0,
                epochs=1,
                batch_size=1,
                early_stopping=False,
                patience=1,
                warm_up=0,
                early_stopping_metric="mae",
                lstm_loss_function="mean_squared_error",
                loss_quantiles=[0.9],
                loss_quantile_weights="auto",
                verbose_training=0,
                random_state=42,
                test_all_models=False,
                evaluate_test=False,
                loss_alpha=1.0,
                target_scaler=None,
                lstm_precipitation_transform=False,
            )

        self.assertEqual(model.predict.call_count, 2)
        self.assertIsNone(result.test_predictions_scaled)
        self.assertIsNone(result.all_test_predictions_scaled)
        self.assertIsNone(result.test_metrics)


if __name__ == "__main__":
    unittest.main()
