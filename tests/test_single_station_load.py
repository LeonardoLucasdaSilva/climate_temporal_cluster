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

        config_data = importlib.import_module("climate_cluster.config_data")
        load_single_station = config_data.load_single_station

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = tmp_path / "inmet"
            station_dir = data_root / "TO" / "A999"
            station_dir.mkdir(parents=True, exist_ok=True)

            # Create a sample station file with realistic data
            daily_file = station_dir / "A999_2000_2025_daily.csv"
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
                        "VELOCIDADE_VENTO",
                        "DIRECAO_VENTO",
                        "RAJADA_VENTO",
                        "PRECIPITACAO_TOTAL",
                        "RADIACAO",
                    ]
                )
                writer.writerow(
                    [
                        "2025-01-01",
                        "33",
                        "21",
                        "98",
                        "40",
                        "1010",
                        "1008",
                        "1.2",
                        "180",
                        "2.5",
                        "10",
                        "18000",
                    ]
                )
                writer.writerow(
                    [
                        "2025-01-02",
                        "32",
                        "20",
                        "96",
                        "42",
                        "1012",
                        "1009",
                        "1.0",
                        "175",
                        "2.3",
                        "0",
                        "17500",
                    ]
                )

            # Load the station
            df = load_single_station(
                state="TO",
                station_id="A999",
                data_root=data_root,
            )

            # Assertions
            self.assertEqual(len(df), 2)
            self.assertIn("Data", df.columns)
            self.assertIn("TEMPERATURA_MAXIMA", df.columns)
            self.assertIn("PRECIPITACAO_TOTAL", df.columns)
            self.assertEqual(df["TEMPERATURA_MAXIMA"].iloc[0], 33.0)
            self.assertEqual(df["PRECIPITACAO_TOTAL"].iloc[0], 10.0)

    def test_load_single_station_custom_cols(self) -> None:
        """Test loading with custom columns."""
        import importlib

        config_data = importlib.import_module("climate_cluster.config_data")
        load_single_station = config_data.load_single_station

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = tmp_path / "inmet"
            station_dir = data_root / "SP" / "A001"
            station_dir.mkdir(parents=True, exist_ok=True)

            daily_file = station_dir / "A001_2000_2025_daily.csv"
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

            df = load_single_station(
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

