"""Shared weather-feature scaling used by clustering pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from methods.tools.precipitation_utils import DEFAULT_PRECIPITATION_COLUMN


SUPPORTED_SCALER_TYPES = ("standard", "minmax")
DISABLED_PRECIPITATION_SCALER_VALUES = {"", "none", "null"}
FeatureScaler = StandardScaler | MinMaxScaler


@dataclass(frozen=True)
class FeatureScalingState:
    """Fitted scalers for covariates and precipitation features."""

    covariate_scaler: FeatureScaler | None = None
    precipitation_scaler: FeatureScaler | None = None


def create_feature_scaler(scaler_type: str) -> FeatureScaler:
    """Return the configured scaler for weather feature normalization."""
    scaler_type = scaler_type.lower()
    if scaler_type == "standard":
        return StandardScaler()
    if scaler_type == "minmax":
        return MinMaxScaler()
    supported = ", ".join(SUPPORTED_SCALER_TYPES)
    raise ValueError(
        f"Unsupported scaler_type: {scaler_type!r}. Use one of: {supported}"
    )


def normalize_precipitation_scaler_type(scaler_type: str | None) -> str | None:
    """Return a precipitation scaler name, or None to keep rain in mm."""
    if scaler_type is None:
        return None

    normalized = str(scaler_type).strip().lower()
    if normalized in DISABLED_PRECIPITATION_SCALER_VALUES:
        return None
    if normalized in SUPPORTED_SCALER_TYPES:
        return normalized

    supported = ", ".join((*SUPPORTED_SCALER_TYPES, "None"))
    raise ValueError(
        "Unsupported precipitation_scaler_type: "
        f"{scaler_type!r}. Use one of: {supported}"
    )


def scale_weather_features(
    df: pd.DataFrame,
    columns: list[str],
    scalers: FeatureScalingState,
    covariate_scaler_type: str,
    precipitation_scaler_type: str | None,
    fit_scalers: bool,
) -> tuple[pd.DataFrame, FeatureScalingState]:
    """Scale covariates and precipitation using separate fitted transforms."""
    values_df = df[columns].astype(float).copy()
    covariate_columns = [
        column for column in columns if column != DEFAULT_PRECIPITATION_COLUMN
    ]
    precipitation_columns = [
        column for column in columns if column == DEFAULT_PRECIPITATION_COLUMN
    ]
    covariate_scaler = scalers.covariate_scaler
    precipitation_scaler = scalers.precipitation_scaler

    if fit_scalers:
        covariate_scaler = (
            create_feature_scaler(covariate_scaler_type).fit(
                values_df[covariate_columns].to_numpy(dtype=float)
            )
            if covariate_columns
            else None
        )
        precipitation_scaler = (
            create_feature_scaler(precipitation_scaler_type).fit(
                values_df[precipitation_columns].to_numpy(dtype=float)
            )
            if precipitation_columns and precipitation_scaler_type is not None
            else None
        )

    if covariate_scaler is not None and covariate_columns:
        values_df.loc[:, covariate_columns] = covariate_scaler.transform(
            values_df[covariate_columns].to_numpy(dtype=float)
        )
    if precipitation_scaler is not None and precipitation_columns:
        values_df.loc[:, precipitation_columns] = precipitation_scaler.transform(
            values_df[precipitation_columns].to_numpy(dtype=float)
        )

    return (
        values_df,
        FeatureScalingState(
            covariate_scaler=covariate_scaler,
            precipitation_scaler=precipitation_scaler,
        ),
    )
