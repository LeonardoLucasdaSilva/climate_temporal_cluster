"""Run ARMA precipitation baselines for comparison with LSTM+cluster."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data.arma_outputs import save_arma_run_outputs, save_arma_sweep_outputs
from data.load_data import load_station_daily_data
from methods.tools.precipitation_utils import DEFAULT_PRECIPITATION_COLUMN


DEFAULT_PLOT_STYLE: dict[str, object] = {
    "seaborn": {
        "style": "whitegrid",
        "palette": "deep",
    },
    "rc_params": {
        "figure.facecolor": "white",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    },
}


@dataclass(frozen=True)
class ARMAConfig:
    """One ARMA baseline configuration."""

    state: str
    station_id: str
    window_size: int
    p: int
    q: int

    @property
    def name(self) -> str:
        return (
            f"{self.state}_{self.station_id}_w{self.window_size:02d}_"
            f"arma_p{self.p:02d}_q{self.q:02d}"
        )


@dataclass(frozen=True)
class ARMASplitTargets:
    """Aligned precipitation targets and forecast origins for one split."""

    y: np.ndarray
    y_by_lead_day: np.ndarray
    current_precipitation: np.ndarray
    window_indices: np.ndarray
    origin_indices: np.ndarray
    target_dates_by_lead_day: np.ndarray


@dataclass(frozen=True)
class ARMAFitResult:
    """Small wrapper around a fitted statsmodels ARIMA result."""

    result: object
    aic: float
    bic: float
    hqic: float
    summary_text: str


@dataclass(frozen=True)
class DailyDataSplits:
    """Chronological train, validation, and test dataframe splits."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_offset: int
    val_offset: int
    test_offset: int


def _mapping_from_config(value: object) -> dict[str, object]:
    """Return nested config dictionaries safely."""
    return value if isinstance(value, dict) else {}


def setup_styling(plot_style: dict[str, object] | None = None) -> None:
    """Apply shared plotting defaults for generated figures."""
    plot_style = _mapping_from_config(plot_style)
    default_seaborn = _mapping_from_config(DEFAULT_PLOT_STYLE["seaborn"])
    default_rc_params = _mapping_from_config(DEFAULT_PLOT_STYLE["rc_params"])
    seaborn_style = {
        **default_seaborn,
        **_mapping_from_config(plot_style.get("seaborn")),
    }
    rc_params = {
        **default_rc_params,
        **_mapping_from_config(plot_style.get("rc_params")),
    }

    sns.set_theme(
        style=str(seaborn_style["style"]),
        palette=seaborn_style.get("palette"),
    )
    plt.rcParams.update(rc_params)


def split_daily_dataframe(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> DailyDataSplits:
    """Split daily observations chronologically before forecast alignment."""
    test_ratio = 1 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("TRAIN_RATIO + VAL_RATIO must be smaller than 1.")
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("TRAIN_RATIO and VAL_RATIO must be positive.")

    n_rows = len(df)
    train_end = int(np.floor(n_rows * train_ratio))
    val_end = train_end + int(np.floor(n_rows * val_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= n_rows:
        raise ValueError(
            "Split ratios leave at least one empty dataframe. "
            f"Got {n_rows} rows, train_end={train_end}, val_end={val_end}."
        )

    return DailyDataSplits(
        train=df.iloc[:train_end].reset_index(drop=True),
        val=df.iloc[train_end:val_end].reset_index(drop=True),
        test=df.iloc[val_end:].reset_index(drop=True),
        train_offset=0,
        val_offset=train_end,
        test_offset=val_end,
    )


def build_arma_configurations(
    state: str,
    station_id: str,
    window_sizes: Sequence[int],
    arma_orders: Sequence[tuple[int, int]],
) -> list[ARMAConfig]:
    """Return every requested window-size and ARMA(p, q) configuration."""
    configs: list[ARMAConfig] = []
    for window_size in window_sizes:
        if window_size <= 0:
            raise ValueError("window_sizes must contain only positive integers.")
        for p, q in arma_orders:
            if p < 0 or q < 0:
                raise ValueError("ARMA orders must be non-negative.")
            configs.append(
                ARMAConfig(
                    state=state,
                    station_id=station_id,
                    window_size=int(window_size),
                    p=int(p),
                    q=int(q),
                )
            )
    return configs


def prepare_arma_dataframe(
    df: pd.DataFrame,
    precipitation_column: str = DEFAULT_PRECIPITATION_COLUMN,
) -> pd.DataFrame:
    """Return a date-sorted univariate precipitation dataframe for ARMA."""
    if precipitation_column not in df.columns:
        raise ValueError(
            f"Dataframe does not contain precipitation column {precipitation_column!r}."
        )

    arma_df = df.copy()
    if "Data" in arma_df.columns:
        arma_df["Data"] = pd.to_datetime(arma_df["Data"], errors="coerce")
        arma_df = arma_df.sort_values("Data")
    else:
        arma_df["Data"] = pd.RangeIndex(len(arma_df))

    arma_df[precipitation_column] = pd.to_numeric(
        arma_df[precipitation_column],
        errors="coerce",
    )
    arma_df = arma_df.dropna(subset=[precipitation_column, "Data"]).reset_index(
        drop=True
    )
    if len(arma_df) < 10:
        raise ValueError("ARMA baseline needs at least 10 finite daily observations.")
    return arma_df


def create_arma_split_targets(
    df: pd.DataFrame,
    window_size: int,
    forecast_horizon: int,
    offset: int,
    precipitation_column: str = DEFAULT_PRECIPITATION_COLUMN,
) -> ARMASplitTargets:
    """Create split-local forecast targets aligned to LSTM window origins."""
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive.")
    if len(df) < window_size + forecast_horizon:
        empty_dates = np.empty((0, forecast_horizon), dtype="datetime64[ns]")
        return ARMASplitTargets(
            y=np.array([], dtype=float),
            y_by_lead_day=np.empty((0, forecast_horizon), dtype=float),
            current_precipitation=np.array([], dtype=float),
            window_indices=np.array([], dtype=int),
            origin_indices=np.array([], dtype=int),
            target_dates_by_lead_day=empty_dates,
        )

    n_valid = len(df) - window_size - forecast_horizon + 1
    valid_indices = np.arange(n_valid, dtype=int)
    origin_local_indices = valid_indices + window_size - 1
    current_precipitation = pd.to_numeric(
        df.iloc[origin_local_indices][precipitation_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    targets_by_lead_day: list[np.ndarray] = []
    dates_by_lead_day: list[np.ndarray] = []
    for lead_day in range(1, forecast_horizon + 1):
        target_indices = origin_local_indices + lead_day
        targets_by_lead_day.append(
            pd.to_numeric(
                df.iloc[target_indices][precipitation_column],
                errors="coerce",
            ).to_numpy(dtype=float)
        )
        dates_by_lead_day.append(
            pd.to_datetime(
                df.iloc[target_indices]["Data"],
                errors="coerce",
            ).to_numpy(dtype="datetime64[ns]")
        )

    target_matrix = np.column_stack(targets_by_lead_day)
    date_matrix = np.column_stack(dates_by_lead_day)
    finite_rows = np.isfinite(target_matrix).all(axis=1) & np.isfinite(
        current_precipitation
    )

    return ARMASplitTargets(
        y=target_matrix[finite_rows, -1],
        y_by_lead_day=target_matrix[finite_rows],
        current_precipitation=current_precipitation[finite_rows],
        window_indices=valid_indices[finite_rows] + offset,
        origin_indices=origin_local_indices[finite_rows] + offset,
        target_dates_by_lead_day=date_matrix[finite_rows],
    )


def fit_arma_model(
    train_series: pd.Series,
    p: int,
    q: int,
    trend: str = "c",
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
) -> ARMAFitResult:
    """Fit an ARMA(p, q) model using statsmodels ARIMA(order=(p, 0, q))."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for the ARMA baseline. Install project "
            "dependencies with `pip install -r requirements.txt` or "
            "`pip install -e .`."
        ) from exc

    if len(train_series) <= max(p, q) + 1:
        raise ValueError(
            f"Not enough training observations ({len(train_series)}) for "
            f"ARMA({p}, {q})."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(
            train_series.astype(float),
            order=(p, 0, q),
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )
        result = model.fit()

    return ARMAFitResult(
        result=result,
        aic=float(getattr(result, "aic", np.nan)),
        bic=float(getattr(result, "bic", np.nan)),
        hqic=float(getattr(result, "hqic", np.nan)),
        summary_text=str(result.summary()),
    )


def ensure_statsmodels_available() -> None:
    """Raise a clear error before creating output folders when statsmodels is absent."""
    try:
        import statsmodels  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for the ARMA baseline. Install project "
            "dependencies in the same Python that runs the script.\n"
            f"Current Python: {sys.executable}\n"
            f"Install command: \"{sys.executable}\" -m pip install statsmodels"
        ) from exc


def _fitted_arma_components(
    fitted_result: object,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return mean, AR coefficients, and MA coefficients from statsmodels."""
    param_names = list(getattr(fitted_result, "param_names", []))
    params = np.asarray(getattr(fitted_result, "params", []), dtype=float)
    param_by_name = dict(zip(param_names, params))
    mean = float(param_by_name.get("const", param_by_name.get("intercept", 0.0)))
    ar_params = np.asarray(getattr(fitted_result, "arparams", []), dtype=float)
    ma_params = np.asarray(getattr(fitted_result, "maparams", []), dtype=float)
    return mean, ar_params, ma_params


def _arma_conditional_mean(
    target_index: int,
    observed_values: np.ndarray,
    residuals: np.ndarray,
    forecast_values: dict[int, float],
    mean: float,
    ar_params: np.ndarray,
    ma_params: np.ndarray,
    origin_index: int,
) -> float:
    """Return one conditional ARMA forecast using observed or forecast lags."""
    forecast = mean
    for lag, coefficient in enumerate(ar_params, start=1):
        lag_index = target_index - lag
        if lag_index < 0:
            lag_value = mean
        elif lag_index <= origin_index:
            lag_value = float(observed_values[lag_index])
        else:
            lag_value = float(forecast_values.get(lag_index, mean))
        forecast += float(coefficient) * (lag_value - mean)

    for lag, coefficient in enumerate(ma_params, start=1):
        lag_index = target_index - lag
        lag_residual = (
            float(residuals[lag_index])
            if 0 <= lag_index <= origin_index
            else 0.0
        )
        forecast += float(coefficient) * lag_residual
    return float(forecast)


def _one_step_arma_residuals(
    observed_values: np.ndarray,
    mean: float,
    ar_params: np.ndarray,
    ma_params: np.ndarray,
) -> np.ndarray:
    """Compute fixed-parameter one-step residuals for the observed series."""
    residuals = np.zeros(len(observed_values), dtype=float)
    for index, observed in enumerate(observed_values):
        forecast = _arma_conditional_mean(
            target_index=index,
            observed_values=observed_values,
            residuals=residuals,
            forecast_values={},
            mean=mean,
            ar_params=ar_params,
            ma_params=ma_params,
            origin_index=index - 1,
        )
        residuals[index] = float(observed) - forecast
    return residuals


def rolling_forecast_by_origin(
    fitted_result: object,
    full_series: pd.Series,
    origin_indices: np.ndarray,
    forecast_horizon: int,
    clip_negative_predictions: bool,
) -> np.ndarray:
    """Forecast D+1..D+horizon for each origin without using future targets."""
    observed_values = full_series.to_numpy(dtype=float)
    mean, ar_params, ma_params = _fitted_arma_components(fitted_result)
    residuals = _one_step_arma_residuals(
        observed_values,
        mean=mean,
        ar_params=ar_params,
        ma_params=ma_params,
    )
    predictions = np.full((len(origin_indices), forecast_horizon), np.nan, dtype=float)
    for row, origin_index in enumerate(origin_indices.astype(int)):
        if origin_index < 0 or origin_index >= len(observed_values):
            continue
        forecast_values: dict[int, float] = {}
        for lead_day in range(1, forecast_horizon + 1):
            target_index = int(origin_index) + lead_day
            forecast = _arma_conditional_mean(
                target_index=target_index,
                observed_values=observed_values,
                residuals=residuals,
                forecast_values=forecast_values,
                mean=mean,
                ar_params=ar_params,
                ma_params=ma_params,
                origin_index=int(origin_index),
            )
            forecast_values[target_index] = forecast
            predictions[row, lead_day - 1] = forecast

    if clip_negative_predictions:
        predictions = np.maximum(predictions, 0.0)
    return predictions


def _validate_forecast_matrix(values: np.ndarray, name: str) -> None:
    """Fail the configuration when statsmodels returns unusable forecasts."""
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite ARMA forecasts.")


def run_arma_configuration(
    df: pd.DataFrame,
    config: ARMAConfig,
    output_dir: Path,
    forecast_horizon: int,
    train_ratio: float,
    val_ratio: float,
    trend: str,
    clip_negative_predictions: bool,
    precipitation_column: str,
) -> dict[str, float | int | str | None]:
    """Fit one ARMA configuration, save artifacts, and return a result row."""
    splits = split_daily_dataframe(df, train_ratio=train_ratio, val_ratio=val_ratio)
    train_targets = create_arma_split_targets(
        splits.train,
        config.window_size,
        forecast_horizon,
        splits.train_offset,
        precipitation_column=precipitation_column,
    )
    val_targets = create_arma_split_targets(
        splits.val,
        config.window_size,
        forecast_horizon,
        splits.val_offset,
        precipitation_column=precipitation_column,
    )
    test_targets = create_arma_split_targets(
        splits.test,
        config.window_size,
        forecast_horizon,
        splits.test_offset,
        precipitation_column=precipitation_column,
    )
    if len(train_targets.y) == 0 or len(val_targets.y) == 0 or len(test_targets.y) == 0:
        raise ValueError(
            "At least one split has no valid ARMA forecast targets. "
            "Reduce WINDOW_SIZE or FORECAST_HORIZON, or adjust split ratios."
        )

    train_series = splits.train[precipitation_column].astype(float)
    full_series = pd.concat(
        [
            splits.train[precipitation_column],
            splits.val[precipitation_column],
            splits.test[precipitation_column],
        ],
        ignore_index=True,
    ).astype(float)
    fit = fit_arma_model(train_series, config.p, config.q, trend=trend)

    y_pred_train_by_lead_day = rolling_forecast_by_origin(
        fit.result,
        full_series,
        train_targets.origin_indices,
        forecast_horizon,
        clip_negative_predictions=clip_negative_predictions,
    )
    y_pred_val_by_lead_day = rolling_forecast_by_origin(
        fit.result,
        full_series,
        val_targets.origin_indices,
        forecast_horizon,
        clip_negative_predictions=clip_negative_predictions,
    )
    y_pred_test_by_lead_day = rolling_forecast_by_origin(
        fit.result,
        full_series,
        test_targets.origin_indices,
        forecast_horizon,
        clip_negative_predictions=clip_negative_predictions,
    )
    _validate_forecast_matrix(y_pred_train_by_lead_day, "train predictions")
    _validate_forecast_matrix(y_pred_val_by_lead_day, "validation predictions")
    _validate_forecast_matrix(y_pred_test_by_lead_day, "test predictions")

    return save_arma_run_outputs(
        config=config,
        output_dir=output_dir,
        y_train=train_targets.y,
        y_val=val_targets.y,
        y_test=test_targets.y,
        y_train_by_lead_day=train_targets.y_by_lead_day,
        y_val_by_lead_day=val_targets.y_by_lead_day,
        y_test_by_lead_day=test_targets.y_by_lead_day,
        current_train=train_targets.current_precipitation,
        current_val=val_targets.current_precipitation,
        current_test=test_targets.current_precipitation,
        y_pred_train_by_lead_day=y_pred_train_by_lead_day,
        y_pred_val_by_lead_day=y_pred_val_by_lead_day,
        y_pred_test_by_lead_day=y_pred_test_by_lead_day,
        train_indices=train_targets.window_indices,
        val_indices=val_targets.window_indices,
        test_indices=test_targets.window_indices,
        test_target_dates_by_lead_day=test_targets.target_dates_by_lead_day,
        state=config.state,
        station_id=config.station_id,
        forecast_horizon=forecast_horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        trend=trend,
        aic=fit.aic,
        bic=fit.bic,
        hqic=fit.hqic,
        model_summary=fit.summary_text,
        clip_negative_predictions=clip_negative_predictions,
    )


def run_arma_experiment(
    state: str,
    station_id: str,
    window_sizes: Sequence[int],
    arma_orders: Sequence[tuple[int, int]],
    forecast_horizon: int,
    train_ratio: float,
    val_ratio: float,
    data_root: Path,
    output_root: Path,
    sweep_name: str | None = None,
    sweep_name_prefix: str = "arma_sweep",
    timestamp_format: str = "%Y_%m_%d_%Hh%M",
    plot_style: dict[str, object] | None = None,
    trend: str = "c",
    clip_negative_predictions: bool = True,
    precipitation_column: str = DEFAULT_PRECIPITATION_COLUMN,
    continue_on_error: bool = True,
) -> Path:
    """Run an ARMA sweep and save comparison artifacts under the output root."""
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive.")

    ensure_statsmodels_available()
    setup_styling(plot_style or DEFAULT_PLOT_STYLE)
    df = load_station_daily_data(state=state, station_id=station_id, data_root=data_root)
    df = prepare_arma_dataframe(df, precipitation_column=precipitation_column)
    configs = build_arma_configurations(
        state=state,
        station_id=station_id,
        window_sizes=window_sizes,
        arma_orders=arma_orders,
    )

    arma_root = output_root / "ARMA"
    timestamp = datetime.now().strftime(timestamp_format)
    sweep_folder_name = (
        sweep_name
        if sweep_name
        else f"{sweep_name_prefix}_{state}_{station_id}_{timestamp}"
    )
    sweep_dir = arma_root / sweep_folder_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str | None]] = []
    failures: list[dict[str, str]] = []
    for config in configs:
        output_dir = sweep_dir / config.name
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            rows.append(
                run_arma_configuration(
                    df=df,
                    config=config,
                    output_dir=output_dir,
                    forecast_horizon=forecast_horizon,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    trend=trend,
                    clip_negative_predictions=clip_negative_predictions,
                    precipitation_column=precipitation_column,
                )
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            failures.append({"run_name": config.name, "error": str(exc)})
            (output_dir / "error.txt").write_text(str(exc), encoding="utf-8")

    save_arma_sweep_outputs(
        sweep_dir=sweep_dir,
        result_rows=rows,
        failure_rows=failures,
        state=state,
        station_id=station_id,
        forecast_horizon=forecast_horizon,
        arma_orders=arma_orders,
        window_sizes=window_sizes,
    )
    return sweep_dir


def parse_arma_orders(raw_orders: Iterable[Sequence[int]]) -> list[tuple[int, int]]:
    """Normalize configured ARMA order pairs."""
    orders = [(int(order[0]), int(order[1])) for order in raw_orders]
    if not orders:
        raise ValueError("At least one ARMA order must be configured.")
    return orders
