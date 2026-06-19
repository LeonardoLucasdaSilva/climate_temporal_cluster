"""Reusable cleaning helpers for climate station dataframes."""

from __future__ import annotations

import pandas as pd


def normalize_decimal_columns(
    df: pd.DataFrame,
    exclude: tuple[str, ...] = ("DATA", "Data"),
) -> pd.DataFrame:
    """Convert comma-decimal numeric columns to floats where possible."""
    cleaned = df.copy()
    for col in cleaned.columns:
        if col in exclude:
            continue
        cleaned[col] = pd.to_numeric(
            cleaned[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
    return cleaned


def fill_missing_numeric(df: pd.DataFrame, value: float = 0.0) -> pd.DataFrame:
    """Fill missing numeric values without mutating the input dataframe."""
    cleaned = df.copy()
    numeric_cols = cleaned.select_dtypes(include="number").columns
    cleaned[numeric_cols] = cleaned[numeric_cols].fillna(value)
    return cleaned

