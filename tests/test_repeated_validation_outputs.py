"""Tests for repeated validation metrics and report rendering."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from data.repeated_validation_outputs import (
    repeated_validation_tex,
    summarize_validation_repeats,
    validation_mae_by_horizon,
    validation_mae_by_horizon_and_rain_regime,
)


class RepeatedValidationOutputTests(unittest.TestCase):
    def test_validation_mae_is_calculated_by_horizon_and_threshold(self) -> None:
        metrics = validation_mae_by_horizon(
            np.array([[0.0, 12.0], [2.0, 30.0]]),
            np.array([[1.0, 10.0], [4.0, 24.0]]),
            thresholds_mm=[0.0, 10.0],
        )

        first = metrics.loc[
            metrics["horizon"].eq(1) & metrics["threshold_mm"].eq(0.0)
        ].iloc[0]
        second = metrics.loc[
            metrics["horizon"].eq(2) & metrics["threshold_mm"].eq(10.0)
        ].iloc[0]
        self.assertEqual(int(first["sample_count"]), 1)
        self.assertAlmostEqual(float(first["val_mae"]), 2.0)
        self.assertEqual(int(second["sample_count"]), 2)
        self.assertAlmostEqual(float(second["val_mae"]), 4.0)

    def test_summary_uses_sample_standard_deviation(self) -> None:
        seed_metrics = pd.DataFrame(
            [
                {
                    "config_id": "baseline",
                    "clustering": "-",
                    "n_clusters": 1,
                    "sigma": np.nan,
                    "horizon": 1,
                    "threshold_mm": 0.0,
                    "seed": seed,
                    "sample_count": 10,
                    "val_mae": float(seed),
                }
                for seed in (1, 2, 3, 4, 5)
            ]
        )
        summary = summarize_validation_repeats(
            seed_metrics,
            expected_seeds=[1, 2, 3, 4, 5],
        )

        self.assertAlmostEqual(float(summary.iloc[0]["mean_val_mae"]), 3.0)
        self.assertAlmostEqual(
            float(summary.iloc[0]["std_val_mae"]),
            float(np.std([1, 2, 3, 4, 5], ddof=1)),
        )
        self.assertTrue(bool(summary.iloc[0]["all_seeds_completed"]))
        tex = repeated_validation_tex(
            summary,
            seeds=[1, 2, 3, 4, 5],
        )
        self.assertIn("Mean Val. MAE", tex)
        self.assertIn("3.0000", tex)
        self.assertIn("1.5811", tex)

    def test_rain_regime_metrics_include_overall_and_nonoverlapping_bands(self) -> None:
        actual = np.array([[0.0], [5.0], [10.0], [20.0], [30.0], [40.0]])
        predicted = actual + np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])

        metrics = validation_mae_by_horizon_and_rain_regime(actual, predicted)

        by_regime = metrics.set_index("rain_regime")
        self.assertEqual(int(by_regime.loc["overall", "sample_count"]), 6)
        self.assertEqual(int(by_regime.loc["0_to_10mm", "sample_count"]), 3)
        self.assertEqual(int(by_regime.loc["10_to_30mm", "sample_count"]), 2)
        self.assertEqual(int(by_regime.loc["gt_30mm", "sample_count"]), 1)
        self.assertAlmostEqual(float(by_regime.loc["overall", "val_mae"]), 2.5)
        self.assertAlmostEqual(float(by_regime.loc["0_to_10mm", "val_mae"]), 1.0)
        self.assertAlmostEqual(float(by_regime.loc["10_to_30mm", "val_mae"]), 3.5)

    def test_report_keeps_lowest_mean_spectral_sigma_per_k(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "config_id": config_id,
                    "clustering": clustering,
                    "n_clusters": n_clusters,
                    "sigma": sigma,
                    "horizon": 1,
                    "threshold_mm": 0.0,
                    "sample_count": 10,
                    "mean_val_mae": mean,
                    "std_val_mae": 0.1,
                    "completed_seed_count": 5,
                    "expected_seed_count": 5,
                    "all_seeds_completed": True,
                }
                for config_id, clustering, n_clusters, sigma, mean in (
                    ("baseline", "-", 1, np.nan, 4.0),
                    ("k02_kmeans", "kmeans", 2, np.nan, 3.0),
                    ("k02_spectral_a", "spectral(1.0000)", 2, 1.0, 2.0),
                    ("k02_spectral_b", "spectral(2.0000)", 2, 2.0, 2.5),
                )
            ]
        )

        tex = repeated_validation_tex(summary, seeds=[1, 2, 3, 4, 5])

        self.assertIn("spectral(1.0000)", tex)
        self.assertNotIn("spectral(2.0000)", tex)


if __name__ == "__main__":
    unittest.main()
