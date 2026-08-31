"""Test single-station data loading."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class SingleStationLoadTest(unittest.TestCase):
    def test_load_single_station(self) -> None:
        """Test loading a single station with daily data."""
        import importlib

        load_data = importlib.import_module("data.load_data")
        load_station_daily_data = load_data.load_station_daily_data

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = tmp_path / "inmet"
            station_dir = data_root / "TO" / "A999"
            station_dir.mkdir(parents=True, exist_ok=True)

            # Create a sample station file with realistic data
            daily_file = station_dir / "A999_2000_2026_daily.csv"
            with daily_file.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp, delimiter=";")
                writer.writerow(
                    [
                        "DATA",
                        "DATA_SIN",
                        "DATA_COS",
                        "TEMPERATURA_MAXIMA",
                        "TEMPERATURA_MIN",
                        "UMIDADE_MAX",
                        "UMIDADE_MIN",
                        "PRESSAO_MAX",
                        "PRESSAO_MIN",
                        "VELOCIDADE_VENTO",
                        "DIRECAO_VENTO",
                        "DIRECAO_VENTO_SIN",
                        "DIRECAO_VENTO_COS",
                        "RAJADA_VENTO",
                        "PRECIPITACAO_TOTAL",
                        "RADIACAO",
                    ]
                )
                writer.writerow(
                    [
                        "2025-01-01",
                        "0.5",
                        "0.8660254",
                        "33",
                        "21",
                        "98",
                        "40",
                        "1010",
                        "1008",
                        "1.2",
                        "180",
                        "0",
                        "-1",
                        "2.5",
                        "10",
                        "18000",
                    ]
                )
                writer.writerow(
                    [
                        "2025-01-02",
                        "0.6",
                        "0.8",
                        "32",
                        "20",
                        "96",
                        "42",
                        "1012",
                        "1009",
                        "1.0",
                        "175",
                        "0.0871557",
                        "-0.9961947",
                        "2.3",
                        "0",
                        "17500",
                    ]
                )

            # Load the station
            df = load_station_daily_data(
                state="TO",
                station_id="A999",
                data_root=data_root,
            )

            # Assertions
            self.assertEqual(len(df), 2)
            self.assertIn("Data", df.columns)
            self.assertIn("TEMPERATURA_MAXIMA", df.columns)
            self.assertIn("PRECIPITACAO_TOTAL", df.columns)
            self.assertIn("DATA_SIN", df.columns)
            self.assertIn("DATA_COS", df.columns)
            self.assertIn("DIRECAO_VENTO_SIN", df.columns)
            self.assertIn("DIRECAO_VENTO_COS", df.columns)
            self.assertNotIn("DIRECAO_VENTO", df.columns)
            self.assertEqual(df["TEMPERATURA_MAXIMA"].iloc[0], 33.0)
            self.assertEqual(df["PRECIPITACAO_TOTAL"].iloc[0], 10.0)
            self.assertAlmostEqual(df["DATA_SIN"].iloc[0], 0.5)
            self.assertAlmostEqual(df["DATA_COS"].iloc[0], 0.8660254)

    def test_daily_extrema_and_encoded_wind_direction(self) -> None:
        """Use extrema aggregations and exclude raw wind direction."""
        from data.load_data import load_station_daily_data

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "inmet"
            station_dir = data_root / "RS" / "A801"
            station_dir.mkdir(parents=True, exist_ok=True)
            daily_file = station_dir / "A801_2000_2026_daily.csv"
            with daily_file.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp, delimiter=";")
                writer.writerow(
                    [
                        "DATA",
                        "TEMPERATURA_MAXIMA",
                        "TEMPERATURA_MIN",
                        "UMIDADE_MAX",
                        "UMIDADE_MIN",
                        "PRESSAO_MAX",
                        "PRESSAO_MIN",
                        "DIRECAO_VENTO",
                        "DIRECAO_VENTO_SIN",
                        "DIRECAO_VENTO_COS",
                    ]
                )
                writer.writerow(
                    ["2025-01-01", 30, "", 90, 45, 1010, 1000, 90, 1, 0]
                )
                writer.writerow(
                    ["2025-01-01", 35, 17, 98, 35, 1014, 997, 180, 0, -1]
                )

            df = load_station_daily_data("RS", "A801", data_root)

            self.assertEqual(len(df), 1)
            self.assertEqual(df.loc[0, "TEMPERATURA_MAXIMA"], 35)
            self.assertEqual(df.loc[0, "TEMPERATURA_MIN"], 17)
            self.assertEqual(df.loc[0, "UMIDADE_MAX"], 98)
            self.assertEqual(df.loc[0, "UMIDADE_MIN"], 35)
            self.assertEqual(df.loc[0, "PRESSAO_MAX"], 1014)
            self.assertEqual(df.loc[0, "PRESSAO_MIN"], 997)
            self.assertAlmostEqual(df.loc[0, "DIRECAO_VENTO_SIN"], 0.5)
            self.assertAlmostEqual(df.loc[0, "DIRECAO_VENTO_COS"], -0.5)
            self.assertNotIn("DIRECAO_VENTO", df.columns)

    def test_raw_wind_direction_is_not_an_automatic_numeric_feature(self) -> None:
        """Exclude raw circular degrees even when a dataframe contains them."""
        import pandas as pd

        from methods.cluster.cluster_pipeline import numeric_feature_columns

        df = pd.DataFrame(
            {
                "Data": pd.to_datetime(["2025-01-01"]),
                "DIRECAO_VENTO": [180.0],
                "DIRECAO_VENTO_SIN": [0.0],
                "DIRECAO_VENTO_COS": [-1.0],
            }
        )

        self.assertEqual(
            numeric_feature_columns(df),
            ["DIRECAO_VENTO_SIN", "DIRECAO_VENTO_COS"],
        )

    def test_load_single_station_custom_cols(self) -> None:
        """Test loading with custom columns."""
        import importlib

        load_data = importlib.import_module("data.load_data")
        load_station_daily_data = load_data.load_station_daily_data

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = tmp_path / "inmet"
            station_dir = data_root / "SP" / "A001"
            station_dir.mkdir(parents=True, exist_ok=True)

            daily_file = station_dir / "A001_2000_2026_daily.csv"
            with daily_file.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp, delimiter=";")
                writer.writerow(
                    [
                        "DATA",
                        "TEMPERATURA_MAXIMA",
                        "PRECIPITACAO_TOTAL",
                        "VELOCIDADE_VENTO",
                    ]
                )
                writer.writerow(["2025-02-01", "30", "5.5", "0.8"])

            df = load_station_daily_data(
                state="SP",
                station_id="A001",
                data_root=data_root,
                cols=["DATA", "TEMPERATURA_MAXIMA", "PRECIPITACAO_TOTAL"],
            )

            self.assertEqual(len(df), 1)
            self.assertIn("TEMPERATURA_MAXIMA", df.columns)
            self.assertIn("PRECIPITACAO_TOTAL", df.columns)


if __name__ == "__main__":
    unittest.main()

