"""Tests for spectral eigengap analysis."""

from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from methods.cluster.eigengap import (  # noqa: E402
    calculate_eigengaps,
    combine_sigma_values,
    create_pipeline_clustering_features,
    optimize_sigma_for_eigengap,
    plot_eigengaps,
    run_eigengap_analysis,
)


class EigengapTest(unittest.TestCase):
    def test_combines_and_deduplicates_additional_sigma_values(self) -> None:
        combined = combine_sigma_values([0.1, 1.0], [1.0, 2.0, 0.1])

        np.testing.assert_allclose(combined, [0.1, 1.0, 2.0])

    def test_feature_preprocessing_matches_pipeline_scaling(self) -> None:
        df = pd.DataFrame(
            {
                "Data": pd.date_range("2025-01-01", periods=10),
                "TEMPERATURA_MAXIMA": np.arange(10, dtype=float),
                "PRECIPITACAO_TOTAL": np.arange(10, 20, dtype=float),
            }
        )

        features, columns = create_pipeline_clustering_features(
            df,
            window_size=2,
            columns=["TEMPERATURA_MAXIMA", "PRECIPITACAO_TOTAL"],
            normalize=True,
            scaler_type="standard",
            precipitation_scaler_type=None,
            train_ratio=0.6,
            pca_variance_threshold=None,
        )

        train_temperature = np.arange(6, dtype=float)
        scaled_temperature = (
            train_temperature - train_temperature.mean()
        ) / train_temperature.std()
        expected_first_window = np.array(
            [scaled_temperature[0], 10.0, scaled_temperature[1], 11.0]
        )
        self.assertEqual(columns, ["TEMPERATURA_MAXIMA", "PRECIPITACAO_TOTAL"])
        self.assertEqual(features.shape, (5, 4))
        np.testing.assert_allclose(features[0], expected_first_window)

    def test_two_disconnected_groups_recommend_two_clusters(self) -> None:
        samples = np.array([[0.0], [0.01], [0.02], [10.0], [10.01], [10.02]])

        result = calculate_eigengaps(
            samples,
            sigma=0.1,
            window_size=4,
            max_gaps=4,
        )

        self.assertEqual(result.cluster_counts.tolist(), [1, 2, 3, 4])
        self.assertEqual(result.sigma, 0.1)
        self.assertEqual(result.best_n_clusters, 2)
        self.assertEqual(len(result.eigenvalues), 5)
        self.assertEqual(len(result.gaps), 4)
        self.assertGreater(result.gaps[1], result.gaps[0])
        self.assertGreater(result.gaps[1], result.gaps[2])

    def test_multi_window_run_saves_plots_and_prints_summary(self) -> None:
        output_dir = PROJECT_ROOT / "tests" / f"_eigengap_run_{uuid.uuid4().hex}"
        samples = np.array(
            [[0.0], [0.01], [0.02], [10.0], [10.01], [10.02]]
        )
        try:
            with (
                patch(
                    "methods.cluster.eigengap.load_station_daily_data",
                    return_value=pd.DataFrame(
                        {
                            "Data": pd.date_range("2025-01-01", periods=8),
                            "feature": np.arange(8, dtype=float),
                        }
                    ),
                ),
                patch(
                    "methods.cluster.eigengap.create_pipeline_clustering_features",
                    return_value=(samples, ["feature"]),
                ),
            ):
                console = StringIO()
                with redirect_stdout(console):
                    results = run_eigengap_analysis(
                        state="RS",
                        station_id="A801",
                        window_sizes=[2, 3],
                        output_dir=output_dir,
                        max_gaps=4,
                        sigma_bounds=(0.01, 2.0),
                        max_sigma_evaluations=5,
                    )

            self.assertEqual(
                [(result.window_size, result.sigma) for result in results],
                [(2, results[0].sigma), (3, results[1].sigma)],
            )
            for window_size in (2, 3):
                self.assertTrue(
                    (output_dir / f"eigengaps_window_{window_size:03d}_optimal.png").is_file()
                )
            self.assertIn("Eigengap summary", console.getvalue())
            self.assertIn("Suggested clusters", console.getvalue())
            self.assertIn("Maximum sigma evaluations per window: 5", console.getvalue())
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_plot_highlights_and_saves_eigengaps(self) -> None:
        output_dir = PROJECT_ROOT / "tests" / f"_eigengap_test_{uuid.uuid4().hex}"
        output_path = output_dir / "eigengaps.png"
        try:
            result = calculate_eigengaps(
                np.array([[0.0], [0.01], [10.0], [10.01]]),
                sigma=0.1,
                window_size=4,
                max_gaps=3,
            )

            saved_path = plot_eigengaps(result, output_path)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_degenerate_affinity_is_reported_without_arpack(self) -> None:
        result = calculate_eigengaps(
            np.array([[0.0], [10.0], [20.0]]),
            sigma=1e-12,
            window_size=30,
            max_gaps=2,
        )

        self.assertIsNone(result.best_n_clusters)
        self.assertIsNone(result.best_gap)
        self.assertIsNotNone(result.warning)
        self.assertIn("Affinity matrix is degenerate", result.warning)
        self.assertIn("Increase sigma", result.warning)

    def test_multi_window_run_prints_degenerate_warning_and_continues(self) -> None:
        output_dir = PROJECT_ROOT / "tests" / f"_eigengap_degenerate_{uuid.uuid4().hex}"
        samples = np.array([[0.0], [10.0], [20.0]])
        try:
            with (
                patch(
                    "methods.cluster.eigengap.load_station_daily_data",
                    return_value=pd.DataFrame(
                        {
                            "Data": pd.date_range("2025-01-01", periods=100),
                            "feature": np.arange(100, dtype=float),
                        }
                    ),
                ),
                patch(
                    "methods.cluster.eigengap.create_pipeline_clustering_features",
                    return_value=(samples, ["feature"]),
                ),
            ):
                console = StringIO()
                with redirect_stdout(console):
                    results = run_eigengap_analysis(
                        state="RS",
                        station_id="A801",
                        window_sizes=[30, 45],
                        output_dir=output_dir,
                        max_gaps=2,
                        sigma_bounds=(1e-12, 2e-12),
                        max_sigma_evaluations=3,
                    )

            self.assertEqual(len(results), 2)
            self.assertTrue(all(result.warning is not None for result in results))
            self.assertIn("No usable sigma found", console.getvalue())
            self.assertTrue(
                (output_dir / "eigengaps_window_030_optimal.png").is_file()
            )
            self.assertTrue(
                (output_dir / "eigengaps_window_045_optimal.png").is_file()
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least two samples"):
            calculate_eigengaps(
                np.array([[1.0]]),
                sigma=1.0,
                window_size=2,
            )
        with self.assertRaisesRegex(ValueError, "max_gaps must be positive"):
            calculate_eigengaps(
                np.array([[0.0], [1.0]]),
                sigma=1.0,
                window_size=2,
                max_gaps=0,
            )
        with self.assertRaisesRegex(ValueError, "at least 3"):
            run_eigengap_analysis(
                state="RS",
                station_id="A801",
                window_sizes=[2],
                output_dir=Path("unused"),
                max_sigma_evaluations=2,
            )

    def test_sigma_optimizer_ignores_first_gap_and_respects_budget(self) -> None:
        samples = np.array([[0.0], [0.01], [0.02], [10.0], [10.01], [10.02]])

        best, evaluations = optimize_sigma_for_eigengap(
            samples,
            window_size=3,
            sigma_bounds=(0.01, 2.0),
            max_sigma_evaluations=6,
            max_gaps=4,
        )

        self.assertLessEqual(len(evaluations), 6)
        self.assertEqual(best.best_n_clusters, 2)
        self.assertEqual(best.best_gap, max(result.best_gap for result in evaluations))


if __name__ == "__main__":
    unittest.main()
