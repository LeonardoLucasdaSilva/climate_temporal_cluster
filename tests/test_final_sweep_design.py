"""Tests for the standalone final experiment design."""

from __future__ import annotations

from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from experiments import run_final_experiment_sweeps as sweep


class FinalSweepDesignTests(unittest.TestCase):
    def test_architecture_and_loss_order(self) -> None:
        sigma_grid = {
            window: [1.0, 2.0, 3.0, 4.0, 5.0]
            for window in sweep.WINDOW_SIZES
        }

        configurations = sweep._configuration_grid(sigma_grid)

        self.assertEqual(len(configurations), 225)
        self.assertEqual(
            [dict(config.variant_parameters) for config in configurations[:3]],
            [
                {"lstm_units": 64, "lstm_units_2": None},
                {"lstm_units": 128, "lstm_units_2": None},
                {"lstm_units": 64, "lstm_units_2": 32},
            ],
        )
        self.assertEqual(
            [name for name, _quantiles in sweep.LOSS_CONFIGURATIONS],
            ["q090", "q080", "mixed_q075_q090_q095_q099"],
        )
        self.assertEqual(sweep.LEARNING_RATE, 1e-3)

    def test_progress_dashboard_separates_trained_and_invalid_counts(self) -> None:
        test_root = Path(__file__).parent / f"_progress_test_{uuid.uuid4().hex}"
        final_dir = test_root / "final_experiment_sweep_1"
        dashboard = test_root / "SWEEP_PROGRESS.md"
        final_dir.mkdir(parents=True)
        (final_dir / "configuration_manifest.csv").write_text(
            "configuration_id,status\n"
            "first,completed\n"
            "second,invalid_clustering\n"
            "third,completed\n",
            encoding="utf-8",
        )
        try:
            with patch.object(sweep, "PROGRESS_DASHBOARD", dashboard):
                processed = sweep._progress(final_dir, "third")

            self.assertEqual(processed, 3)
            dashboard_text = dashboard.read_text(encoding="utf-8")
            self.assertIn("| Processed | 3 / 675 (0.4%) |", dashboard_text)
            self.assertIn("| Successfully trained | 2 |", dashboard_text)
            self.assertIn("| Invalid/skipped | 1 |", dashboard_text)
            self.assertIn("- Latest configuration: `third`", dashboard_text)
            progress_text = (final_dir / "progress.txt").read_text(encoding="utf-8")
            self.assertIn("completed=3", progress_text)
            self.assertIn("trained=2", progress_text)
            self.assertIn("invalid_or_skipped=1", progress_text)
        finally:
            shutil.rmtree(test_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
