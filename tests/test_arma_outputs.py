from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import shutil
import unittest
import uuid

import numpy as np
import pandas as pd

from data.arma_outputs import save_arma_run_outputs


class ARMAOutputTests(unittest.TestCase):
    def test_save_arma_run_outputs_writes_comparable_diagnostics(self) -> None:
        output_dir = (
            Path(__file__).parent
            / f"_arma_outputs_test_{uuid.uuid4().hex}"
        )
        output_dir.mkdir()
        try:
            config = SimpleNamespace(
                name="RS_A801_w03_arma_p01_q00",
                window_size=3,
                p=1,
                q=0,
            )
            y_train_by_lead_day = np.array(
                [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
            )
            y_val_by_lead_day = np.array([[4.0, 5.0], [5.0, 6.0]])
            y_test_by_lead_day = np.array(
                [
                    [6.0, 7.0],
                    [7.0, 8.0],
                    [8.0, 9.0],
                    [9.0, 10.0],
                    [10.0, 11.0],
                    [11.0, 12.0],
                ]
            )
            y_pred_train_by_lead_day = y_train_by_lead_day + 0.2
            y_pred_val_by_lead_day = y_val_by_lead_day + 0.2
            y_pred_test_by_lead_day = y_test_by_lead_day + 0.2
            dates = np.column_stack(
                [
                    pd.date_range("2025-02-01", periods=6, freq="D").to_numpy(
                        dtype="datetime64[ns]"
                    ),
                    pd.date_range("2025-02-02", periods=6, freq="D").to_numpy(
                        dtype="datetime64[ns]"
                    ),
                ]
            )

            result = save_arma_run_outputs(
                config=config,
                output_dir=output_dir,
                y_train=y_train_by_lead_day[:, -1],
                y_val=y_val_by_lead_day[:, -1],
                y_test=y_test_by_lead_day[:, -1],
                y_train_by_lead_day=y_train_by_lead_day,
                y_val_by_lead_day=y_val_by_lead_day,
                y_test_by_lead_day=y_test_by_lead_day,
                current_train=np.array([0.0, 1.0, 2.0, 3.0]),
                current_val=np.array([4.0, 5.0]),
                current_test=np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0]),
                y_pred_train_by_lead_day=y_pred_train_by_lead_day,
                y_pred_val_by_lead_day=y_pred_val_by_lead_day,
                y_pred_test_by_lead_day=y_pred_test_by_lead_day,
                train_indices=np.arange(4),
                val_indices=np.arange(10, 12),
                test_indices=np.arange(20, 26),
                test_target_dates_by_lead_day=dates,
                state="RS",
                station_id="A801",
                forecast_horizon=2,
                train_ratio=0.6,
                val_ratio=0.1,
                trend="c",
                aic=100.0,
                bic=110.0,
                hqic=105.0,
                model_summary="ARMA summary",
                clip_negative_predictions=True,
            )

            self.assertEqual(result["run_name"], "RS_A801_w03_arma_p01_q00")
            self.assertTrue((output_dir / "metrics_summary.csv").exists())
            self.assertTrue((output_dir / "test_predictions.csv").exists())
            self.assertTrue((output_dir / "summary.txt").exists())
            self.assertTrue((output_dir / "arma_model_summary.txt").exists())
            self.assertTrue(
                (output_dir / "prediction_overview" / "02_predictions_vs_actual.png").exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "prediction_timeseries_splits"
                    / "lead_day_01"
                    / "02_predictions_timeseries_split_01_of_04.png"
                ).exists()
            )
            diag_dir = output_dir / "forecast_horizon_diagnostics"
            self.assertTrue((diag_dir / "test_prediction_by_lead_day.csv").exists())
            self.assertTrue(
                (diag_dir / "test_prediction_metrics_by_lead_day.csv").exists()
            )
            self.assertTrue((diag_dir / "12_prediction_error_by_lead_day.png").exists())
            self.assertTrue((diag_dir / "13_true_vs_predicted_by_lead_day.png").exists())
            self.assertTrue(
                (diag_dir / "14_prediction_vs_actual_timeseries_by_lead_day.png").exists()
            )
            self.assertTrue(
                (
                    diag_dir
                    / "true_vs_predicted_by_lead_day"
                    / "true_vs_predicted_lead_day_02.png"
                ).exists()
            )
            predictions_df = pd.read_csv(output_dir / "test_predictions.csv")
            self.assertIn("predicted_lead_day_2", predictions_df.columns)
            lead_df = pd.read_csv(diag_dir / "test_prediction_by_lead_day.csv")
            self.assertEqual(lead_df.loc[0, "target_date"], "2025-02-01")
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

