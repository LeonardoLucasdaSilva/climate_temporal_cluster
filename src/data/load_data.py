from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from data.clean_data import normalize_decimal_columns

DAILY_FILE_SUFFIX = "_daily.csv"


def iter_station_daily_files(data_root: Path) -> List[Path]:
    """Yield all station daily CSV files from the INMET data tree."""
    files = []
    if not data_root.exists():
        return files

    for csv_path in data_root.glob("*/*/*_daily.csv"):
        if csv_path.is_file():
            files.append(csv_path)
    return files


def station_info_from_path(file_path: Path, data_root: Path) -> Dict[str, str]:
    """Extract state and station ids from a station daily file path."""
    rel = file_path.relative_to(data_root)
    state = rel.parts[0]
    station_id = rel.parts[1]
    return {
        "state": state,
        "station_id": station_id,
        "station_key": f"{state}_{station_id}",
    }


def load_station_daily_data(
    state: str,
    station_id: str,
    data_root: Path,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load a single INMET station's daily data and group by day.

    Args:
        state: State code (e.g., 'SP', 'TO')
        station_id: Station code (e.g., 'A701', 'A055')
        data_root: Root path to INMET data (data/inmet/)
        cols: Columns to select. If None, uses all available columns.

    Returns:
        DataFrame with daily aggregated data, indexed by date.

    Raises:
        FileNotFoundError: If station file not found.
    """
    file_path = data_root / state / station_id / f"{station_id}_2000_2026_daily.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Station file not found: {file_path}")

    # Default columns matching INMET structure
    if cols is None:
        cols = [
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

    # Read CSV with semicolon delimiter
    df = pd.read_csv(file_path, delimiter=";", usecols=cols, na_values=[""])

    # Convert date column to datetime
    df["DATA"] = pd.to_datetime(df["DATA"], format="%Y-%m-%d", errors="coerce")

    # Drop rows with invalid dates
    df = df.dropna(subset=["DATA"])

    # Convert string columns to numeric (handle commas as decimal separators)
    df = normalize_decimal_columns(df, exclude=("DATA",))
    df = df.fillna(0)

    # Set date as index for daily grouping
    df.set_index("DATA", inplace=True)

    # Define aggregation: mean for temperatures/humidity/pressure, sum for rain/radiation
    agg_dict = {
        "TEMPERATURA_MAXIMA": "mean",
        "TEMPERATURA_MIN": "mean",
        "UMIDADE_MAX": "mean",
        "UMIDADE_MIN": "mean",
        "PRESSAO_MAX": "mean",
        "PRESSAO_MIN": "mean",
        "VELOCIDADE_VENTO": "mean",
        "DIRECAO_VENTO": "mean",
        "RAJADA_VENTO": "mean",
        "PRECIPITACAO_TOTAL": "sum",
        "RADIACAO": "sum",
    }

    # Keep only columns that exist in the dataframe
    agg_dict = {col: func for col, func in agg_dict.items() if col in df.columns}

    # Resample daily (already daily granularity, but ensures consistency)
    df_daily = df.resample("D").agg(agg_dict)

    # Reset index to make date a column again
    df_daily.reset_index(inplace=True)
    df_daily.rename(columns={"DATA": "Data"}, inplace=True)

    return df_daily

