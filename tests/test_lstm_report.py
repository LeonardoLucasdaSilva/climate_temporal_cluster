"""Tests for LSTM experiment LaTeX reports."""

from __future__ import annotations

import sys
import shutil
import unittest
import uuid
from pathlib import Path


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
            },
        )

        self.assertIn(r"\section*{Configuration}", tex)
        self.assertIn(r"\item \textbf{Covariate Scaler:} standard", tex)
        self.assertIn(r"\item \textbf{Precipitation Scaler:} minmax", tex)
        self.assertIn(r"\item \textbf{LSTM Target Scale:} normalized", tex)

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


if __name__ == "__main__":
    unittest.main()
