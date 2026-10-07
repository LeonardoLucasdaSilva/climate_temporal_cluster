from __future__ import annotations

import sys
from pathlib import Path
import shutil
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd

from data.random_search_outputs import (
    _final_ranking_tex,
    aggregate_confirmation_results,
    rank_screening_results,
    retain_best_spectral_sigma,
)
from experiments.random_search.run_random_search import (
    _apply_hyperparameters,
    _candidate_id,
    _confirmation_parameters,
)
from experiments.random_search import run_random_search as framework
from methods.lstm_cluster_pytorch.pipeline import _train_single_cluster_model
from methods.tools.random_search import generate_parameter_configurations


class RandomSearchFrameworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work_dir = Path(__file__).resolve().parent / (
            f"_random_search_framework_test_{uuid4().hex}"
        )

    def tearDown(self) -> None:
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)

    def test_random_sampling_is_unique_and_reproducible(self) -> None:
        space = {"units": [32, 64], "dropout": [0.1, 0.2, 0.3]}
        first = generate_parameter_configurations(
            space,
            strategy="random",
            n_trials=4,
            random_state=7,
        )
        second = generate_parameter_configurations(
            space,
            strategy="random",
            n_trials=4,
            random_state=7,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(len({tuple(row.items()) for row in first}), 4)

    def test_only_lowest_mae_spectral_sigma_is_retained_per_k(self) -> None:
        frame = pd.DataFrame(
            [
                {"algorithm": "kmeans", "n_clusters": 3, "val_mae": 4.0},
                {
                    "algorithm": "spectral",
                    "n_clusters": 3,
                    "sigma": 0.1,
                    "val_mae": 3.0,
                },
                {
                    "algorithm": "spectral",
                    "n_clusters": 3,
                    "sigma": 0.2,
                    "val_mae": 2.0,
                },
                {
                    "algorithm": "spectral",
                    "n_clusters": 5,
                    "sigma": 0.3,
                    "val_mae": 5.0,
                },
            ]
        )

        retained = retain_best_spectral_sigma(frame)

        self.assertEqual(len(retained), 3)
        best_k3 = retained.loc[
            retained["algorithm"].eq("spectral")
            & retained["n_clusters"].eq(3)
        ].iloc[0]
        self.assertEqual(best_k3["sigma"], 0.2)

    def test_screening_uses_ceiling_for_top_ten_percent(self) -> None:
        frame = pd.DataFrame(
            {
                "candidate_id": [f"candidate_{index}" for index in range(21)],
                "val_mae": np.arange(21, dtype=float),
            }
        )

        ranked, retained = rank_screening_results(frame, retain_fraction=0.10)

        self.assertEqual(len(retained), 3)
        self.assertEqual(ranked.iloc[0]["screening_rank"], 1)
        self.assertTrue(retained["retained_for_confirmation"].all())

    def test_final_ranking_requires_all_seeds_and_uses_mean_mae(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "candidate_id": "a",
                    "confirmation_seed": 43,
                    "val_mae": 3.0,
                },
                {
                    "candidate_id": "a",
                    "confirmation_seed": 44,
                    "val_mae": 1.0,
                },
                {
                    "candidate_id": "b",
                    "confirmation_seed": 43,
                    "val_mae": 0.5,
                },
            ]
        )

        rankings = aggregate_confirmation_results(frame, expected_seeds=[43, 44])

        self.assertEqual(rankings["candidate_id"].tolist(), ["a"])
        self.assertEqual(rankings.iloc[0]["mean_val_mae"], 2.0)
        self.assertAlmostEqual(rankings.iloc[0]["std_val_mae"], np.sqrt(2.0))
        self.assertEqual(rankings.iloc[0]["final_rank"], 1)
        report = _final_ranking_tex(rankings)
        self.assertIn("Multi-seed Validation Ranking", report)
        self.assertIn(r"\textcolor{green!50!black}", report)
        self.assertIn("test split is not predicted or evaluated", report)

    def test_confirmation_kwargs_are_exact_and_validation_only(self) -> None:
        base = {
            "window_sizes": [15],
            "n_clusters_list": [1, 3],
            "clustering_algorithm": ["kmeans", "spectral"],
            "test_all_models": False,
            "validation_only": True,
            "save_validation_history": False,
        }
        configuration = {
            "window_size": 30,
            "lstm_units": 128,
            "loss_quantiles": (0.9,),
        }
        row = {
            "algorithm": "spectral",
            "n_clusters": 5,
            "sigma": 0.25,
        }

        parameters = _confirmation_parameters(
            row,
            configuration,
            seed=43,
            candidate_dir=Path("unused"),
            base=base,
        )

        self.assertEqual(parameters["window_sizes"], [30])
        self.assertEqual(parameters["n_clusters_list"], [5])
        self.assertEqual(parameters["clustering_algorithm"], "spectral")
        self.assertEqual(parameters["sigma_values"], [0.25])
        self.assertEqual(parameters["random_state"], 43)
        self.assertTrue(parameters["validation_only"])
        self.assertTrue(parameters["save_validation_history"])
        self.assertFalse(parameters["test_all_models"])

    def test_candidate_id_includes_selected_spectral_sigma(self) -> None:
        candidate_id = _candidate_id(
            {
                "random_configuration_id": "random_0001",
                "n_clusters": 5,
                "algorithm": "spectral",
                "sigma": 0.25,
            }
        )
        self.assertEqual(candidate_id, "random_0001__k05_spectral_s0p25")

    def test_framework_runs_both_stages_and_writes_final_ranking(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_runner(**kwargs: object) -> Path:
            calls.append(dict(kwargs))
            sweep_dir = Path(kwargs["output_root"]) / str(kwargs["sweep_name"])
            sweep_dir.mkdir(parents=True)
            if isinstance(kwargs["clustering_algorithm"], list):
                rows = [
                    {
                        "run_name": "baseline",
                        "algorithm": "kmeans",
                        "n_clusters": 1,
                        "sigma": None,
                        "val_mae": 3.0,
                    },
                    {
                        "run_name": "kmeans_k3",
                        "algorithm": "kmeans",
                        "n_clusters": 3,
                        "sigma": None,
                        "val_mae": 2.0,
                    },
                    {
                        "run_name": "spectral_k3_s01",
                        "algorithm": "spectral",
                        "n_clusters": 3,
                        "sigma": 0.1,
                        "val_mae": 4.0,
                    },
                    {
                        "run_name": "spectral_k3_s02",
                        "algorithm": "spectral",
                        "n_clusters": 3,
                        "sigma": 0.2,
                        "val_mae": 1.0,
                    },
                ]
                for row in rows:
                    (sweep_dir / row["run_name"]).mkdir()
            else:
                seed = int(kwargs["random_state"])
                rows = [
                    {
                        "run_name": "confirmed",
                        "algorithm": kwargs["clustering_algorithm"],
                        "n_clusters": kwargs["n_clusters_list"][0],
                        "sigma": (
                            kwargs["sigma_values"][0]
                            if kwargs["sigma_values"] is not None
                            else None
                        ),
                        "val_mae": 1.0 + (seed - 43) * 0.2,
                    }
                ]
                (sweep_dir / "confirmed" / "model_fit").mkdir(parents=True)
            pd.DataFrame(rows).to_csv(sweep_dir / "sweep_results.csv", index=False)
            return sweep_dir

        base = {
            "n_clusters_list": [1, 3, 5, 7, 9],
            "clustering_algorithm": ["kmeans", "spectral"],
            "sigma_values": None,
            "test_all_models": False,
            "validation_only": True,
            "save_validation_history": False,
        }
        with patch.object(framework, "OUTPUT_ROOT", self.work_dir), patch.object(
            framework,
            "N_RANDOM_CONFIGURATIONS",
            1,
        ), patch.object(
            framework,
            "CONFIRMATION_SEEDS",
            [43, 44],
        ), patch.object(
            framework,
            "COMPILE_FINAL_RANKING_PDF",
            False,
        ), patch.object(
            framework,
            "_base_kwargs",
            return_value=base,
        ), patch.object(
            framework,
            "_automatic_sigma_grids",
            return_value={15: [0.1, 0.2], 30: [0.1, 0.2], 45: [0.1, 0.2]},
        ):
            output_dir = framework.run_framework(
                experiment_runner=fake_runner,
                search_id="unit_search",
            )

        final_rankings = pd.read_csv(
            output_dir / "final_validation_rankings.csv"
        )
        self.assertEqual(len(final_rankings), 1)
        self.assertEqual(final_rankings.iloc[0]["final_rank"], 1)
        confirmation_calls = [
            call for call in calls if isinstance(call["clustering_algorithm"], str)
        ]
        self.assertEqual(len(confirmation_calls), 2)
        self.assertTrue(
            all(call["save_validation_history"] for call in confirmation_calls)
        )
        self.assertTrue(all(call["validation_only"] for call in calls))
        screening_dir = output_dir / "stage1_screening" / "random_0001"
        self.assertTrue((screening_dir / "spectral_k3_s02").is_dir())
        self.assertFalse((screening_dir / "spectral_k3_s01").exists())
        self.assertFalse((screening_dir / "kmeans_k3").exists())
        self.assertFalse((screening_dir / "baseline").exists())
        self.assertTrue(
            (
                output_dir
                / "stage2_confirmation"
                / final_rankings.iloc[0]["candidate_id"]
                / "seed_43"
                / "confirmed"
                / "model_fit"
            ).is_dir()
        )

    def test_pytorch_validation_only_never_predicts_test_rows(self) -> None:
        model = MagicMock()
        model.fit.return_value = {"loss": [1.0]}
        model.predict.side_effect = lambda values: np.zeros((len(values), 1))
        predictor = MagicMock(return_value=model)
        fake_model_module = ModuleType("methods.lstm_cluster_pytorch.model")
        fake_model_module.LSTMPrecipitationPredictor = predictor

        with patch.dict(
            sys.modules,
            {"methods.lstm_cluster_pytorch.model": fake_model_module},
        ), patch(
            "methods.lstm_cluster_pytorch.pipeline._set_pytorch_worker_threads"
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
