from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from data.clean_data import normalize_decimal_columns

DAILY_FILE_SUFFIX = "_daily.csv"

NON_FEATURE_COLUMNS = frozenset({"DIRECAO_VENTO"})

DEFAULT_DAILY_COLUMNS = [
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
    "DIRECAO_VENTO_SIN",
    "DIRECAO_VENTO_COS",
    "RAJADA_VENTO",
    "PRECIPITACAO_TOTAL",
    "RADIACAO",
]

DAILY_AGGREGATIONS = {
    "DATA_SIN": "mean",
    "DATA_COS": "mean",
    "TEMPERATURA_MAXIMA": "max",
    "TEMPERATURA_MIN": "min",
    "UMIDADE_MAX": "max",
    "UMIDADE_MIN": "min",
    "PRESSAO_MAX": "max",
    "PRESSAO_MIN": "min",
    "VELOCIDADE_VENTO": "mean",
    "DIRECAO_VENTO_SIN": "mean",
    "DIRECAO_VENTO_COS": "mean",
    "RAJADA_VENTO": "mean",
    "PRECIPITACAO_TOTAL": "sum",
    "RADIACAO": "sum",
}


def daily_aggregation_for_column(column: str) -> str | None:
    """Return the daily aggregation for one weather column."""
    if column in NON_FEATURE_COLUMNS:
        return None
    if column.endswith(("_MAXIMA", "_MAX")):
        return "max"
    if column.endswith(("_MINIMA", "_MIN")):
        return "min"
    return DAILY_AGGREGATIONS.get(column)


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
        cols: Columns to select. If None, uses the available columns from the
            default weather and cyclic-date feature set.

    Returns:
        DataFrame with daily aggregated data, indexed by date.

    Raises:
        FileNotFoundError: If station file not found.
    """
    file_path = data_root / state / station_id / f"{station_id}_2000_2026_daily.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Station file not found: {file_path}")

    # Default features are optional because station files can have different
    # schemas. Explicitly requested columns retain pandas' strict validation.
    if cols is None:
        available_columns = pd.read_csv(file_path, delimiter=";", nrows=0).columns
        cols = [col for col in DEFAULT_DAILY_COLUMNS if col in available_columns]

    # Read CSV with semicolon delimiter
    df = pd.read_csv(file_path, delimiter=";", usecols=cols, na_values=[""])

    # Convert date column to datetime
    df["DATA"] = pd.to_datetime(df["DATA"], format="%Y-%m-%d", errors="coerce")

    # Drop rows with invalid dates
    df = df.dropna(subset=["DATA"])

    # Convert string columns to numeric (handle commas as decimal separators)
    df = normalize_decimal_columns(df, exclude=("DATA",))

    # Set date as index for daily grouping
    df.set_index("DATA", inplace=True)

    # Keep only columns that exist in the dataframe
    agg_dict = {
        col: aggregation
        for col in df.columns
        if (aggregation := daily_aggregation_for_column(col)) is not None
    }

    # Resample daily (already daily granularity, but ensures consistency)
    df_daily = df.resample("D").agg(agg_dict)
    # Aggregate valid observations first so missing extrema do not become false
    # zero-valued minima. Preserve the loader's zero-filled output afterward.
    df_daily = df_daily.fillna(0)

    # Reset index to make date a column again
    df_daily.reset_index(inplace=True)
    df_daily.rename(columns={"DATA": "Data"}, inplace=True)

    return df_daily

