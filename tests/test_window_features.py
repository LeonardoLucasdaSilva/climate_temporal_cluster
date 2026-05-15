"""Test window feature creation."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class WindowFeaturesTest(unittest.TestCase):
    def setUp(self) -> None:
        """Create sample data for testing."""
        self.df = pd.DataFrame(
            {
                "Data": pd.date_range("2025-01-01", periods=20),
                "TEMPERATURA_MAXIMA": np.arange(20, 40, 1.0),
                "TEMPERATURA_MIN": np.arange(10, 30, 1.0),
                "PRECIPITACAO_TOTAL": np.random.rand(20) * 10,
                "UMIDADE_MAX": np.random.rand(20) * 100,
                "VELOCIDADE_VENTO": np.random.rand(20) * 5,
            }
        )

    def test_create_windows_basic(self) -> None:
        """Test basic window creation."""
        import importlib

        window_features = importlib.import_module("climate_cluster.features.window_features")
        create_windows = window_features.create_windows

        windows, scaler = create_windows(
            self.df,
            window_size=4,
            columns=["TEMPERATURA_MAXIMA", "TEMPERATURA_MIN"],
            normalize=False,
        )

        # Check shape
        self.assertEqual(windows.shape[0], 20 - 4 + 1)  # 17 windows
        self.assertEqual(windows.shape[1], 4)  # window size
        self.assertEqual(windows.shape[2], 2)  # 2 features

        # Scaler should be None when normalize=False
        self.assertIsNone(scaler)

    def test_create_windows_normalized(self) -> None:
        """Test normalized window creation."""
        import importlib

        window_features = importlib.import_module("climate_cluster.features.window_features")
        create_windows = window_features.create_windows

        windows, scaler = create_windows(
            self.df,
            window_size=4,
            columns=["TEMPERATURA_MAXIMA", "TEMPERATURA_MIN"],
            normalize=True,
        )

        # Scaler should be present
        self.assertIsNotNone(scaler)

        # Data should be normalized (approximately zero mean)
        mean_val = np.mean(windows.reshape(-1, windows.shape[2]), axis=0)
        np.testing.assert_array_almost_equal(mean_val, [0, 0], decimal=1)

    def test_create_windows_all_columns(self) -> None:
        """Test with automatic column selection."""
        import importlib

        window_features = importlib.import_module("climate_cluster.features.window_features")
        create_windows = window_features.create_windows

        windows, scaler = create_windows(
            self.df,
            window_size=3,
            columns=None,  # Auto-select numeric columns
            normalize=True,
        )

        # Should have 5 numeric columns (all except 'Data')
        self.assertEqual(windows.shape[2], 5)
        self.assertEqual(windows.shape[1], 3)
        self.assertEqual(windows.shape[0], 20 - 3 + 1)  # 18 windows

    def test_windows_to_dataframe(self) -> None:
        """Test converting windows to dataframe."""
        import importlib

        window_features = importlib.import_module("climate_cluster.features.window_features")
        create_windows = window_features.create_windows
        windows_to_dataframe = window_features.windows_to_dataframe

        cols = ["TEMPERATURA_MAXIMA", "TEMPERATURA_MIN"]
        windows, scaler = create_windows(
            self.df,
            window_size=4,
            columns=cols,
            normalize=True,
        )

        df_out = windows_to_dataframe(windows, columns=cols, scaler=scaler, window_size=4)

        # Check shape
        self.assertEqual(df_out.shape[0], windows.shape[0])
        self.assertEqual(df_out.shape[1], 4 * 2)  # 4 days × 2 columns

        # Check column names
        expected_cols = [
            "TEMPERATURA_MAXIMA_day0",
            "TEMPERATURA_MAXIMA_day1",
            "TEMPERATURA_MAXIMA_day2",
            "TEMPERATURA_MAXIMA_day3",
            "TEMPERATURA_MIN_day0",
            "TEMPERATURA_MIN_day1",
            "TEMPERATURA_MIN_day2",
            "TEMPERATURA_MIN_day3",
        ]
        self.assertEqual(list(df_out.columns), expected_cols)

    def test_different_window_sizes(self) -> None:
        """Test different window sizes."""
        import importlib

        window_features = importlib.import_module("climate_cluster.features.window_features")
        create_windows = window_features.create_windows

        for ws in [2, 3, 5, 7]:
            windows, _ = create_windows(
                self.df,
                window_size=ws,
                columns=["TEMPERATURA_MAXIMA"],
                normalize=False,
            )
            self.assertEqual(windows.shape[1], ws)
            self.assertEqual(windows.shape[0], 20 - ws + 1)

    def test_window_size_larger_than_data(self) -> None:
        """Test error when window size > data size."""
        import importlib

        window_features = importlib.import_module("climate_cluster.features.window_features")
        create_windows = window_features.create_windows

        with self.assertRaises(ValueError):
            create_windows(
                self.df,
                window_size=100,
                columns=["TEMPERATURA_MAXIMA"],
                normalize=False,
            )

    def test_flattened_windows(self) -> None:
        """Test flattening windows for ML algorithms."""
        import importlib

        window_features = importlib.import_module("climate_cluster.features.window_features")
        create_windows = window_features.create_windows

        windows, _ = create_windows(
            self.df,
            window_size=4,
            columns=["TEMPERATURA_MAXIMA", "TEMPERATURA_MIN", "PRECIPITACAO_TOTAL"],
            normalize=False,
        )

        # Flatten
        windows_flat = windows.reshape(windows.shape[0], -1)

        # Check shape
        self.assertEqual(windows_flat.shape[0], 17)  # 20 - 4 + 1
        self.assertEqual(windows_flat.shape[1], 4 * 3)  # 4 days × 3 features


if __name__ == "__main__":
    unittest.main()

