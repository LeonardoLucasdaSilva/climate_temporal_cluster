"""Tests for LSTM experiment LaTeX reports."""

from __future__ import annotations

import sys
import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from methods.lstm_cluster.report import config_summary_list, render_report  # noqa: E402


class LstmReportTests(unittest.TestCase):
    def test_report_configuration_includes_selected_scalers(self) -> None:
        tex = render_report(
            PROJECT_ROOT / "tests" / "_missing_report_dir",
            {
                "state": "RS",
                "station_id": "A801",
                "window_size": 8,
                "n_clusters": 3,
                "algorithm": "kmeans",
                "forecast_horizon": 2,
                "normalize": True,
                "scaler_type": "standard",
                "precipitation_scaler_type": "minmax",
                "target_scale": "normalized",
                "optimizer": "AdamW",
                "weight_decay": 1e-4,
                "pca_variance_threshold": 0.9,
                "pca_for_clustering_only": True,
            },
        )

        self.assertIn(r"\section*{Configuration}", tex)
        self.assertIn(r"\item \textbf{Covariate Scaler:} standard", tex)
        self.assertIn(r"\item \textbf{Precipitation Scaler:} minmax", tex)
        self.assertIn(r"\item \textbf{LSTM Target Scale:} normalized", tex)
        self.assertIn(r"\item \textbf{Optimizer:} AdamW", tex)
        self.assertIn(r"\item \textbf{Weight decay:} 0.0001", tex)
        self.assertIn(r"\item \textbf{PCA Variance Threshold:} 0.9", tex)
        self.assertIn(r"\item \textbf{PCA Mode:} clustering only", tex)

    def test_config_summary_reports_disabled_and_shared_pca_modes(self) -> None:
        disabled_tex = config_summary_list(
            {
                "window_size": 8,
                "pca_variance_threshold": None,
                "pca_for_clustering_only": False,
            }
        )
        shared_tex = config_summary_list(
            {
                "window_size": 8,
                "pca_variance_threshold": 0.95,
                "pca_for_clustering_only": False,
            }
        )

        self.assertIn(
            r"\item \textbf{PCA Variance Threshold:} not used",
            disabled_tex,
        )
        self.assertIn(r"\item \textbf{PCA Mode:} disabled", disabled_tex)
        self.assertIn(r"\item \textbf{PCA Variance Threshold:} 0.95", shared_tex)
        self.assertIn(
            r"\item \textbf{PCA Mode:} clustering and LSTM",
            shared_tex,
        )

    def test_config_summary_marks_scalers_disabled(self) -> None:
        tex = config_summary_list(
            {
                "window_size": 8,
                "normalize": False,
                "scaler_type": "standard",
                "precipitation_scaler_type": "minmax",
                "target_scale": "mm",
            }
        )

        self.assertIn(r"\item \textbf{Covariate Scaler:} none", tex)
        self.assertIn(r"\item \textbf{Precipitation Scaler:} none", tex)
        self.assertIn(r"\item \textbf{LSTM Target Scale:} mm", tex)
        self.assertNotIn("standard", tex)
        self.assertNotIn("minmax", tex)

    def test_report_includes_silhouette_plot_when_present(self) -> None:
        output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_report_silhouette_test_{uuid.uuid4().hex}"
        )
        figure_dir = output_dir / "cluster_diagnostics"
        figure_dir.mkdir(parents=True)
        (figure_dir / "08_silhouette_analysis.png").write_bytes(b"plot")

        try:
            tex = render_report(output_dir, {"window_size": 8, "n_clusters": 2})

            self.assertIn(r"\section*{Cluster Diagnostics}", tex)
            self.assertIn("cluster_diagnostics/08_silhouette_analysis.png", tex)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_report_labels_oracle_transfer_as_a_separate_section(self) -> None:
        output_dir = (
            PROJECT_ROOT
            / "tests"
            / f"_report_oracle_transfer_test_{uuid.uuid4().hex}"
        )
        output_dir.mkdir()
        pd.DataFrame(
            [
                {
                    "metric_selection": "RMSE",
                    "switched_samples": 2,
                    "n_test": 4,
                    "switched_samples_percent": 50.0,
                    "rmse_improvement": 1.0,
                    "rmse_improvement_percent": 25.0,
                    "mae_improvement": 0.5,
                    "mae_improvement_percent": 20.0,
                    "r2_improvement": 0.1,
                }
            ]
        ).to_csv(output_dir / "test_model_metric_summary.csv", index=False)
        pd.DataFrame(
            {
                "assigned_test_cluster": [0, 1],
                "LSTM_0": [1, 0],
                "LSTM_1": [1, 2],
            }
        ).to_csv(output_dir / "oracle_model_selection_matrix.csv", index=False)
        pd.DataFrame(
            [
                {
                    "assigned_test_cluster": 0,
                    "n_test": 2,
                    "oracle_switched_percent": 50.0,
                    "same_cluster_mae": 2.0,
                    "oracle_mae": 1.0,
                    "mae_improvement": 1.0,
                    "rmse_improvement": 0.75,
                }
            ]
        ).to_csv(output_dir / "oracle_cluster_routing_summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "assigned_test_cluster": 0,
                    "oracle_selected_model_cluster": 1,
                    "n_test": 1,
                    "percent_of_assigned_cluster": 50.0,
                    "mae_improvement": 4.0,
                    "mean_squared_error_improvement": 24.0,
                }
            ]
        ).to_csv(output_dir / "oracle_cluster_pair_summary.csv", index=False)

        try:
            tex = render_report(output_dir, {"window_size": 8, "n_clusters": 2})

            self.assertIn(
                r"\section*{Análise de transferência entre clusters}",
                tex,
            )
            self.assertIn("Oracle Transfer Matrix", tex)
            self.assertIn("Oracle Routing Summary by Assigned Cluster", tex)
            self.assertIn("Oracle Routing Summary by Transfer Pair", tex)
            self.assertIn("main test metrics, predictions, and plots", tex)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
