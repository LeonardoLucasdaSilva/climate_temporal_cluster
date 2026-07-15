"""Run the LSTM-by-cluster precipitation sweep."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data.load_data import load_station_daily_data
from data.lstm_outputs import save_run_outputs, save_sweep_outputs
from evaluation.metrics import calculate_regression_metrics
from models.lstm import LSTMPrecipitationPredictor
from methods.cluster.manual import ManualRainClustering
from methods.cluster.ng import spectral_clustering
from methods.cluster.cluster_pipeline import (
    KMEANS_N_INIT,
    PCA_VARIANCE_THRESHOLD,
    SUPPORTED_CLUSTERING_ALGORITHMS,
    numeric_feature_columns,
)
from methods.lstm_cluster.console import print_info, print_section
from methods.lstm_cluster.report import generate_config_report
from methods.tools.dimensionality_reduction_tools import (
    determine_pca_components,
    flatten_windows,
)
from methods.tools.precipitation_utils import (
    DEFAULT_PRECIPITATION_COLUMN,
    precipitation_targets,
)
from methods.tools.sliding_windows import create_windows
from methods.tools.sigma_choosing import calculate_sigma_values


@dataclass(frozen=True)
class ExperimentConfig:
    """One sweep configuration."""

    state: str
    station_id: str
    window_size: int
    n_clusters: int
    algorithm: str
    sigma: float | None

    @property
    def name(self) -> str:
        sigma_part = "sigma_na" if self.sigma is None else f"sigma_{self.sigma:g}"
        return (
            f"{self.state}_{self.station_id}_w{self.window_size:02d}_"
            f"k{self.n_clusters:02d}_{self.algorithm}_{sigma_part}"
        ).replace(".", "p")


@dataclass(frozen=True)
class DailyDataSplits:
    """Chronological train, validation, and test dataframe splits."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_offset: int
    val_offset: int
    test_offset: int


@dataclass(frozen=True)
class WindowSplitData:
    """Window features, targets, labels, and original indices by split."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    y_train_by_lead_day: np.ndarray
    y_val_by_lead_day: np.ndarray
    y_test_by_lead_day: np.ndarray
    y_train_by_lead_day_scaled: np.ndarray
    y_val_by_lead_day_scaled: np.ndarray
    current_train: np.ndarray
    current_val: np.ndarray
    current_test: np.ndarray
    c_train: np.ndarray
    c_val: np.ndarray
    c_test: np.ndarray
    i_train: np.ndarray
    i_val: np.ndarray
    i_test: np.ndarray
    all_targets: np.ndarray
    all_current_precipitation: np.ndarray
    all_cluster_labels: np.ndarray
    test_targets_by_lead_day: np.ndarray
    test_target_dates_by_lead_day: np.ndarray
    target_scaler: FeatureScaler | None
    n_windows: int


@dataclass(frozen=True)
class FeatureScalingState:
    """Fitted scalers for covariates and precipitation features."""

    covariate_scaler: FeatureScaler | None = None
    precipitation_scaler: FeatureScaler | None = None


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

SUPPORTED_SCALER_TYPES = ("standard", "minmax")
DISABLED_PRECIPITATION_SCALER_VALUES = {"", "none", "null"}
SUPPORTED_LSTM_LOSS_FUNCTIONS = (
    "mean_squared_error",
    "mse",
    "mean_absolute_error",
    "mae",
    "huber",
    "quantile_weighted_mse",
)
FeatureScaler = StandardScaler | MinMaxScaler


def _mapping_from_config(value: object) -> Mapping[str, object]:
    """Return nested config dictionaries safely."""
    return value if isinstance(value, Mapping) else {}


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


def _normalize_precipitation_scaler_type(scaler_type: str | None) -> str | None:
    """Return a precipitation scaler name, or None to keep precipitation in mm."""
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


def _transform_precipitation_values(
    values: np.ndarray,
    scaler: FeatureScaler | None,
) -> np.ndarray:
    """Transform precipitation arrays with a one-column fitted scaler."""
    array = np.asarray(values, dtype=float)
    if scaler is None or array.size == 0:
        return array.copy()
    return scaler.transform(array.reshape(-1, 1)).reshape(array.shape)


def _inverse_transform_precipitation_values(
    values: np.ndarray,
    scaler: FeatureScaler | None,
) -> np.ndarray:
    """Inverse-transform precipitation arrays from the training target scale."""
    array = np.asarray(values, dtype=float)
    if scaler is None or array.size == 0:
        return array.copy()
    return scaler.inverse_transform(array.reshape(-1, 1)).reshape(array.shape)


def validate_loss_function(loss_function: str) -> str:
    """Return a normalized LSTM loss name after validation."""
    loss_function = loss_function.lower()
    if loss_function not in SUPPORTED_LSTM_LOSS_FUNCTIONS:
        supported = ", ".join(SUPPORTED_LSTM_LOSS_FUNCTIONS)
        raise ValueError(
            f"Unsupported lstm_loss_function: {loss_function!r}. "
            f"Use one of: {supported}"
        )
    return loss_function


def quantile_weighted_mse_config(
    values: np.ndarray,
    quantiles: Sequence[float],
    weights: Sequence[float] | str = "auto",
) -> tuple[list[float], list[float]]:
    """Return rain thresholds and bin weights for quantile-weighted MSE.

    Thresholds are calculated from the cluster's own training targets in the
    active target scale. In automatic mode, each bin receives inverse-frequency
    weight, normalized so the most common non-empty bin has weight 1.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [], [1.0]

    quantile_array = np.asarray(quantiles, dtype=float)
    if quantile_array.ndim != 1 or quantile_array.size == 0:
        raise ValueError("loss_quantiles must be a non-empty one-dimensional list.")
    if np.any((quantile_array <= 0) | (quantile_array >= 1)):
        raise ValueError("loss_quantiles must contain values between 0 and 1.")
    if np.any(np.diff(quantile_array) <= 0):
        raise ValueError("loss_quantiles must be strictly increasing.")

    thresholds = np.quantile(values, quantile_array)
    thresholds = np.unique(thresholds[np.isfinite(thresholds)])
    thresholds = thresholds.tolist()

    if isinstance(weights, str):
        weights = weights.lower()
    if weights != "auto":
        configured_weights = [float(weight) for weight in weights]
        if len(configured_weights) != len(thresholds) + 1:
            raise ValueError(
                "loss_quantile_weights must contain one more value than the "
                "number of unique cluster quantile thresholds."
            )
        if any(weight <= 0 for weight in configured_weights):
            raise ValueError("loss_quantile_weights must be positive.")
        return [float(threshold) for threshold in thresholds], configured_weights

    bin_indices = np.digitize(values, thresholds, right=True)
    counts = np.bincount(bin_indices, minlength=len(thresholds) + 1).astype(float)
    positive_counts = counts[counts > 0]
    if positive_counts.size == 0:
        return [float(threshold) for threshold in thresholds], [1.0] * (len(thresholds) + 1)

    min_count = positive_counts.min()
    auto_weights = np.ones_like(counts, dtype=float)
    nonempty = counts > 0
    auto_weights[nonempty] = min_count / counts[nonempty]
    auto_weights = auto_weights / auto_weights[auto_weights > 0].min()
    return (
        [float(threshold) for threshold in thresholds],
        [float(weight) for weight in auto_weights],
    )


def setup_styling(plot_style: Mapping[str, object] | None = None) -> None:
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


def build_configurations(
    sigmas: list[float | None],
    state: str,
    station_id: str,
    window_sizes: list[int],
    n_clusters_list: list[int],
    clustering_algorithm: str,
) -> list[ExperimentConfig]:
    """Return every window, cluster-count, and sigma combination."""
    return [
        ExperimentConfig(
            state=state,
            station_id=station_id,
            window_size=window_size,
            n_clusters=n_clusters,
            algorithm=clustering_algorithm.lower(),
            sigma=sigma,
        )
        for window_size in window_sizes
        for n_clusters in n_clusters_list
        for sigma in sigmas
    ]


def split_daily_dataframe(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> DailyDataSplits:
    """Split daily observations chronologically before window creation."""
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


def _create_window_tensor_from_features(
    features: np.ndarray,
    columns: list[str],
    window_size: int,
) -> np.ndarray:
    feature_df = pd.DataFrame(features, columns=columns)
    windows, _ = create_windows(
        feature_df,
        window_size=window_size,
        columns=columns,
        normalize=False,
        variance_threshold=None,
        verbose=False,
    )
    return windows


def _split_feature_matrix(
    df: pd.DataFrame,
    columns: list[str],
    window_size: int,
    scalers: FeatureScalingState,
    covariate_scaler_type: str,
    precipitation_scaler_type: str | None,
    pca: PCA | None,
    fit_scaler: bool,
    fit_pca: bool,
    variance_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray, FeatureScalingState, PCA | None]:
    values_df = df[columns].astype(float).copy()
    covariate_columns = [
        column for column in columns if column != DEFAULT_PRECIPITATION_COLUMN
    ]
    precipitation_columns = [
        column for column in columns if column == DEFAULT_PRECIPITATION_COLUMN
    ]
    covariate_scaler = scalers.covariate_scaler
    precipitation_scaler = scalers.precipitation_scaler

    if fit_scaler:
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

    values = values_df.to_numpy(dtype=float)
    windows = _create_window_tensor_from_features(values, columns, window_size)
    windows_flat = flatten_windows(windows)

    if fit_pca and variance_threshold is not None:
        n_components = determine_pca_components(windows_flat, variance_threshold)
        pca = PCA(n_components=n_components).fit(windows_flat)
    if pca is not None:
        windows_flat = pca.transform(windows_flat)

    return (
        windows,
        windows_flat,
        FeatureScalingState(
            covariate_scaler=covariate_scaler,
            precipitation_scaler=precipitation_scaler,
        ),
        pca,
    )


def _valid_supervised_windows(
    df: pd.DataFrame,
    windows_flat: np.ndarray,
    window_size: int,
    horizon: int,
    offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_indices, _targets = precipitation_targets(
        df,
        window_size,
        len(windows_flat),
        horizon=horizon,
    )
    targets_by_lead_day = _lead_day_targets(
        df,
        valid_indices,
        window_size,
        horizon,
    )
    target_dates_by_lead_day = _lead_day_dates(
        df,
        valid_indices,
        window_size,
        horizon,
    )
    finite_lead_days = np.all(np.isfinite(targets_by_lead_day), axis=1)
    valid_indices = valid_indices[finite_lead_days]
    targets_by_lead_day = targets_by_lead_day[finite_lead_days]
    target_dates_by_lead_day = target_dates_by_lead_day[finite_lead_days]
    targets = targets_by_lead_day[:, -1]
    current_indices = valid_indices + window_size - 1
    current_precipitation = pd.to_numeric(
        df.iloc[current_indices][DEFAULT_PRECIPITATION_COLUMN],
        errors="coerce",
    ).to_numpy(dtype=float)
    return (
        windows_flat[valid_indices],
        targets,
        targets_by_lead_day,
        current_precipitation,
        valid_indices + offset,
        target_dates_by_lead_day,
    )


def _lead_day_targets(
    df: pd.DataFrame,
    valid_indices: np.ndarray,
    window_size: int,
    max_horizon: int,
) -> np.ndarray:
    """Return target precipitation for lead days 1..max_horizon."""
    lead_targets = []
    for lead_day in range(1, max_horizon + 1):
        target_indices = valid_indices + window_size - 1 + lead_day
        lead_targets.append(
            pd.to_numeric(
                df.iloc[target_indices][DEFAULT_PRECIPITATION_COLUMN],
                errors="coerce",
            ).to_numpy(dtype=float)
        )
    if not lead_targets:
        return np.empty((len(valid_indices), 0), dtype=float)
    return np.column_stack(lead_targets)


def _lead_day_dates(
    df: pd.DataFrame,
    valid_indices: np.ndarray,
    window_size: int,
    max_horizon: int,
) -> np.ndarray:
    """Return target dates for lead days 1..max_horizon."""
    if "Data" not in df.columns:
        raise ValueError("Dataframe does not contain date column 'Data'.")

    lead_dates = []
    for lead_day in range(1, max_horizon + 1):
        target_indices = valid_indices + window_size - 1 + lead_day
        lead_dates.append(
            pd.to_datetime(
                df.iloc[target_indices]["Data"],
                errors="coerce",
            ).to_numpy(dtype="datetime64[ns]")
        )
    if not lead_dates:
        return np.empty((len(valid_indices), 0), dtype="datetime64[ns]")
    return np.column_stack(lead_dates)


def _nearest_centroid_labels(
    feature_matrix: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    if len(feature_matrix) == 0:
        return np.array([], dtype=int)
    distances_sq = np.sum(
        (feature_matrix[:, None, :] - centroids[None, :, :]) ** 2,
        axis=2,
    )
    return np.argmin(distances_sq, axis=1).astype(int)


def _cluster_window_splits(
    config: ExperimentConfig,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    manual_zero_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit cluster assignments on training windows and infer held-out labels."""
    if len(X_train) < config.n_clusters:
        raise ValueError(
            f"Need at least {config.n_clusters} training windows for clustering; "
            f"got {len(X_train)}."
        )

    if config.algorithm == "kmeans":
        model = KMeans(
            n_clusters=config.n_clusters,
            random_state=random_state,
            n_init=KMEANS_N_INIT,
        ).fit(X_train)
        return (
            model.labels_,
            _nearest_centroid_labels(X_val, model.cluster_centers_),
            _nearest_centroid_labels(X_test, model.cluster_centers_),
        )

    if config.algorithm == "manual":
        model = ManualRainClustering(
            n_clusters=config.n_clusters,
            zero_tolerance=manual_zero_tolerance,
        )
        c_train = model.fit_predict(X_train, y_train)
        return c_train, model.predict(X_val), model.predict(X_test)

    if config.algorithm == "spectral":
        if config.sigma is None:
            raise ValueError("sigma must be provided when algorithm='spectral'")
        c_train = spectral_clustering(
            X_train,
            sigma=config.sigma,
            k=config.n_clusters,
            random_state=random_state,
        )
        centroids = np.vstack(
            [
                X_train[c_train == cluster_id].mean(axis=0)
                for cluster_id in range(config.n_clusters)
            ]
        )
        if not np.all(np.isfinite(centroids)):
            raise ValueError("Spectral clustering produced an empty training cluster.")
        return (
            c_train,
            _nearest_centroid_labels(X_val, centroids),
            _nearest_centroid_labels(X_test, centroids),
        )

    supported = ", ".join(SUPPORTED_CLUSTERING_ALGORITHMS)
    raise ValueError(
        f"Unsupported clustering algorithm: {config.algorithm!r}. "
        f"Use one of: {supported}"
    )


def create_window_split_data(
    df: pd.DataFrame,
    config: ExperimentConfig,
    feature_columns: list[str],
    normalize: bool,
    scaler_type: str,
    precipitation_scaler_type: str | None,
    variance_threshold: float | None,
    forecast_horizon: int,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
    manual_zero_tolerance: float,
) -> tuple[WindowSplitData, DailyDataSplits]:
    """Create independent split windows from chronological dataframe blocks."""
    precipitation_scaler_type = _normalize_precipitation_scaler_type(
        precipitation_scaler_type
    )
    splits = split_daily_dataframe(df, train_ratio=train_ratio, val_ratio=val_ratio)

    empty_scalers = FeatureScalingState()
    train_windows, train_flat, feature_scalers, pca = _split_feature_matrix(
        splits.train,
        feature_columns,
        config.window_size,
        scalers=empty_scalers,
        covariate_scaler_type=scaler_type,
        precipitation_scaler_type=precipitation_scaler_type,
        pca=None,
        fit_scaler=normalize,
        fit_pca=True,
        variance_threshold=variance_threshold,
    )
    val_windows, val_flat, _, _ = _split_feature_matrix(
        splits.val,
        feature_columns,
        config.window_size,
        scalers=feature_scalers,
        covariate_scaler_type=scaler_type,
        precipitation_scaler_type=precipitation_scaler_type,
        pca=pca,
        fit_scaler=False,
        fit_pca=False,
        variance_threshold=variance_threshold,
    )
    test_windows, test_flat, _, _ = _split_feature_matrix(
        splits.test,
        feature_columns,
        config.window_size,
        scalers=feature_scalers,
        covariate_scaler_type=scaler_type,
        precipitation_scaler_type=precipitation_scaler_type,
        pca=pca,
        fit_scaler=False,
        fit_pca=False,
        variance_threshold=variance_threshold,
    )

    (
        X_train,
        y_train,
        y_train_by_lead_day,
        current_train,
        i_train,
        _train_target_dates_by_lead_day,
    ) = _valid_supervised_windows(
        splits.train,
        train_flat,
        config.window_size,
        forecast_horizon,
        splits.train_offset,
    )
    (
        X_val,
        y_val,
        y_val_by_lead_day,
        current_val,
        i_val,
        _val_target_dates_by_lead_day,
    ) = _valid_supervised_windows(
        splits.val,
        val_flat,
        config.window_size,
        forecast_horizon,
        splits.val_offset,
    )
    (
        X_test,
        y_test,
        y_test_by_lead_day,
        current_test,
        i_test,
        test_target_dates_by_lead_day,
    ) = _valid_supervised_windows(
        splits.test,
        test_flat,
        config.window_size,
        forecast_horizon,
        splits.test_offset,
    )
    target_scaler = (
        create_feature_scaler(precipitation_scaler_type).fit(
            y_train_by_lead_day.reshape(-1, 1)
        )
        if normalize and precipitation_scaler_type is not None
        else None
    )
    y_train_by_lead_day_scaled = _transform_precipitation_values(
        y_train_by_lead_day,
        target_scaler,
    )
    y_val_by_lead_day_scaled = _transform_precipitation_values(
        y_val_by_lead_day,
        target_scaler,
    )

    c_train, c_val, c_test = _cluster_window_splits(
        config,
        X_train,
        X_val,
        X_test,
        y_train,
        random_state=random_state,
        manual_zero_tolerance=manual_zero_tolerance,
    )
    return (
        WindowSplitData(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            y_train_by_lead_day=y_train_by_lead_day,
            y_val_by_lead_day=y_val_by_lead_day,
            y_test_by_lead_day=y_test_by_lead_day,
            y_train_by_lead_day_scaled=y_train_by_lead_day_scaled,
            y_val_by_lead_day_scaled=y_val_by_lead_day_scaled,
            current_train=current_train,
            current_val=current_val,
            current_test=current_test,
            c_train=c_train,
            c_val=c_val,
            c_test=c_test,
            i_train=i_train,
            i_val=i_val,
            i_test=i_test,
            all_targets=np.concatenate([y_train, y_val, y_test]),
            all_current_precipitation=np.concatenate(
                [current_train, current_val, current_test]
            ),
            all_cluster_labels=np.concatenate([c_train, c_val, c_test]),
            test_targets_by_lead_day=y_test_by_lead_day,
            test_target_dates_by_lead_day=test_target_dates_by_lead_day,
            target_scaler=target_scaler,
            n_windows=len(train_windows) + len(val_windows) + len(test_windows),
        ),
        splits,
    )


def to_lstm_shape(X: np.ndarray) -> np.ndarray:
    """Represent each flattened window as a one-step LSTM sequence."""
    return X.reshape(X.shape[0], 1, X.shape[1])


def clipped_predictions(
    model: LSTMPrecipitationPredictor,
    X: np.ndarray,
    target_scaler: FeatureScaler | None = None,
) -> np.ndarray:
    """Predict precipitation in millimeters and clip impossible negatives."""
    predictions = np.asarray(model.predict(X), dtype=float)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    predictions = _inverse_transform_precipitation_values(predictions, target_scaler)
    return np.maximum(predictions, 0.0)


def _lead_day_matrix(values: np.ndarray, name: str) -> np.ndarray:
    """Return target or prediction values as a two-dimensional lead-day matrix."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional.")
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one lead-day column.")
    return matrix


def bootstrap_mean_ci(
    values: np.ndarray,
    random_state: int,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Return a bootstrap CI and probability that the mean is positive."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(random_state)
    sample_indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    bootstrap_means = values[sample_indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(bootstrap_means, alpha)),
        float(np.quantile(bootstrap_means, 1.0 - alpha)),
        float(np.mean(bootstrap_means > 0.0)),
    )


def evaluate_test_samples_with_all_models(
    models_by_cluster: dict[int, LSTMPrecipitationPredictor],
    X_test_lstm: np.ndarray,
    y_test: np.ndarray,
    c_test: np.ndarray,
    original_y_pred_test: np.ndarray,
    random_state: int,
    target_scaler: FeatureScaler | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[int, dict[str, float]],
    dict[str, object],
]:
    """Select the best trained LSTM per test sample and metric."""
    model_items = sorted(models_by_cluster.items())
    model_cluster_ids = np.array(
        [cluster_id for cluster_id, _model in model_items],
        dtype=int,
    )
    y_pred_by_model_by_lead_day = np.stack(
        [
            clipped_predictions(model, X_test_lstm, target_scaler)
            for _cluster_id, model in model_items
        ],
        axis=1,
    )
    y_pred_by_model = y_pred_by_model_by_lead_day[:, :, -1]
    original_model_by_sample = np.asarray(c_test, dtype=int)
    primary_metric = "RMSE"

    per_metric_errors = {
        "MSE": (y_test[:, None] - y_pred_by_model) ** 2,
        "RMSE": (y_test[:, None] - y_pred_by_model) ** 2,
        "MAE": np.abs(y_test[:, None] - y_pred_by_model),
        "RMSLE": (
            np.log1p(np.maximum(y_test[:, None], 0.0))
            - np.log1p(np.maximum(y_pred_by_model, 0.0))
        )
        ** 2,
    }
    mape_report_errors = np.full_like(y_pred_by_model, np.nan, dtype=float)
    mape_selection_errors = per_metric_errors["MAE"].copy()
    nonzero_target = y_test != 0
    mape_report_errors[nonzero_target] = (
        np.abs(
            (y_test[nonzero_target, None] - y_pred_by_model[nonzero_target])
            / y_test[nonzero_target, None]
        )
        * 100.0
    )
    mape_selection_errors[nonzero_target] = mape_report_errors[nonzero_target]
    per_metric_errors["MAPE"] = mape_selection_errors

    y_pred_selected_by_metric: dict[str, np.ndarray] = {}
    y_pred_selected_by_lead_day_by_metric: dict[str, np.ndarray] = {}
    selected_model_by_metric: dict[str, np.ndarray] = {}
    for metric_name, errors in per_metric_errors.items():
        comparable_errors = np.where(np.isfinite(errors), errors, np.inf)
        best_offsets = np.argmin(comparable_errors, axis=1)
        y_pred_selected_by_metric[metric_name] = y_pred_by_model[
            np.arange(len(y_test)),
            best_offsets,
        ]
        y_pred_selected_by_lead_day_by_metric[metric_name] = (
            y_pred_by_model_by_lead_day[np.arange(len(y_test)), best_offsets, :]
        )
        selected_model_by_metric[metric_name] = model_cluster_ids[best_offsets]

    y_pred_selected = y_pred_selected_by_metric[primary_metric]
    y_pred_selected_by_lead_day = y_pred_selected_by_lead_day_by_metric[
        primary_metric
    ]
    selected_model_by_sample = selected_model_by_metric[primary_metric].astype(float)
    comparison_rows: list[dict[str, float | int | bool]] = []
    selection_rows: list[dict[str, float | int | str | bool]] = []
    metric_summary_rows: list[dict[str, float | int | str]] = []
    metrics_by_test_cluster: dict[int, dict[str, float]] = {}

    for sample_index, (actual, test_cluster_id) in enumerate(zip(y_test, c_test)):
        for model_offset, model_cluster_id in enumerate(model_cluster_ids):
            comparison_rows.append(
                {
                    "sample_index": sample_index,
                    "test_cluster": int(test_cluster_id),
                    "model_cluster": int(model_cluster_id),
                    "is_same_cluster_model": int(model_cluster_id)
                    == int(test_cluster_id),
                    "actual": float(actual),
                    "predicted": float(y_pred_by_model[sample_index, model_offset]),
                    "squared_error": float(
                        per_metric_errors["MSE"][sample_index, model_offset]
                    ),
                    "absolute_error": float(
                        per_metric_errors["MAE"][sample_index, model_offset]
                    ),
                    "squared_log_error": float(
                        per_metric_errors["RMSLE"][sample_index, model_offset]
                    ),
                    "absolute_percentage_error": float(
                        mape_report_errors[sample_index, model_offset]
                    ),
                }
            )

        for metric_name in per_metric_errors:
            selected_model = int(selected_model_by_metric[metric_name][sample_index])
            selected_prediction = float(
                y_pred_selected_by_metric[metric_name][sample_index]
            )
            original_prediction = float(original_y_pred_test[sample_index])
            selection_rows.append(
                {
                    "sample_index": sample_index,
                    "test_cluster": int(test_cluster_id),
                    "metric": metric_name,
                    "selected_model_cluster": selected_model,
                    "selected_is_same_cluster": selected_model
                    == int(test_cluster_id),
                    "actual": float(actual),
                    "selected_prediction": selected_prediction,
                    "same_cluster_prediction": original_prediction,
                    "selected_absolute_error": abs(float(actual) - selected_prediction),
                    "same_cluster_absolute_error": abs(
                        float(actual) - original_prediction
                    ),
                }
            )

    original_metrics = calculate_regression_metrics(y_test, original_y_pred_test)
    for metric_name, selected_predictions in y_pred_selected_by_metric.items():
        selected_metrics = calculate_regression_metrics(y_test, selected_predictions)
        selected_models = selected_model_by_metric[metric_name]
        metric_summary_rows.append(
            {
                "metric_selection": metric_name,
                "original_mse": float(original_metrics["MSE"]),
                "selected_mse": float(selected_metrics["MSE"]),
                "mse_improvement": float(
                    original_metrics["MSE"] - selected_metrics["MSE"]
                ),
                "mse_improvement_percent": float(
                    (
                        (original_metrics["MSE"] - selected_metrics["MSE"])
                        / original_metrics["MSE"]
                        * 100.0
                    )
                    if original_metrics["MSE"] != 0
                    else np.nan
                ),
                "original_rmse": float(original_metrics["RMSE"]),
                "selected_rmse": float(selected_metrics["RMSE"]),
                "rmse_improvement": float(
                    original_metrics["RMSE"] - selected_metrics["RMSE"]
                ),
                "rmse_improvement_percent": float(
                    (
                        (original_metrics["RMSE"] - selected_metrics["RMSE"])
                        / original_metrics["RMSE"]
                        * 100.0
                    )
                    if original_metrics["RMSE"] != 0
                    else np.nan
                ),
                "original_mae": float(original_metrics["MAE"]),
                "selected_mae": float(selected_metrics["MAE"]),
                "mae_improvement": float(
                    original_metrics["MAE"] - selected_metrics["MAE"]
                ),
                "mae_improvement_percent": float(
                    (
                        (original_metrics["MAE"] - selected_metrics["MAE"])
                        / original_metrics["MAE"]
                        * 100.0
                    )
                    if original_metrics["MAE"] != 0
                    else np.nan
                ),
                "original_rmsle": float(original_metrics["RMSLE"]),
                "selected_rmsle": float(selected_metrics["RMSLE"]),
                "rmsle_improvement": float(
                    original_metrics["RMSLE"] - selected_metrics["RMSLE"]
                ),
                "rmsle_improvement_percent": float(
                    (
                        (original_metrics["RMSLE"] - selected_metrics["RMSLE"])
                        / original_metrics["RMSLE"]
                        * 100.0
                    )
                    if original_metrics["RMSLE"] != 0
                    else np.nan
                ),
                "original_r2": float(original_metrics["R2"]),
                "selected_r2": float(selected_metrics["R2"]),
                "r2_improvement": float(
                    selected_metrics["R2"] - original_metrics["R2"]
                ),
                "original_mape": float(original_metrics["MAPE"]),
                "selected_mape": float(selected_metrics["MAPE"]),
                "mape_improvement": float(
                    original_metrics["MAPE"] - selected_metrics["MAPE"]
                ),
                "mape_improvement_percent": float(
                    (
                        (original_metrics["MAPE"] - selected_metrics["MAPE"])
                        / original_metrics["MAPE"]
                        * 100.0
                    )
                    if np.isfinite(original_metrics["MAPE"])
                    and original_metrics["MAPE"] != 0
                    else np.nan
                ),
                "switched_samples": int(np.sum(selected_models != original_model_by_sample)),
                "n_test": int(len(y_test)),
                "switched_samples_percent": float(
                    np.mean(selected_models != original_model_by_sample) * 100.0
                ),
            }
        )

    for test_cluster_id in sorted(np.unique(c_test)):
        mask = c_test == test_cluster_id
        metrics_by_test_cluster[int(test_cluster_id)] = calculate_regression_metrics(
            y_test[mask],
            y_pred_selected[mask],
        )

    selected_metrics = calculate_regression_metrics(y_test, y_pred_selected)
    squared_error_improvement = (
        (y_test - original_y_pred_test) ** 2 - (y_test - y_pred_selected) ** 2
    )
    ci_low, ci_high, improvement_probability = bootstrap_mean_ci(
        squared_error_improvement,
        random_state=random_state,
    )
    switched_samples = int(selected_model_by_sample.astype(int).size) - int(
        np.sum(selected_model_by_sample.astype(int) == original_model_by_sample)
    )
    summary = {
        "primary_metric": primary_metric,
        "original_mse": float(original_metrics["MSE"]),
        "selected_mse": float(selected_metrics["MSE"]),
        "mse_improvement": float(original_metrics["MSE"] - selected_metrics["MSE"]),
        "original_rmse": float(original_metrics["RMSE"]),
        "selected_rmse": float(selected_metrics["RMSE"]),
        "rmse_improvement": float(original_metrics["RMSE"] - selected_metrics["RMSE"]),
        "original_mae": float(original_metrics["MAE"]),
        "selected_mae": float(selected_metrics["MAE"]),
        "mae_improvement": float(original_metrics["MAE"] - selected_metrics["MAE"]),
        "mse_improvement_ci_low": ci_low,
        "mse_improvement_ci_high": ci_high,
        "mse_improvement_probability": improvement_probability,
        "n_test_clusters": int(len(np.unique(c_test))),
        "n_test_samples": int(len(y_test)),
        "switched_samples": switched_samples,
    }
    test_model_selection = {
        "comparison_rows": comparison_rows,
        "selection_rows": selection_rows,
        "metric_summary_rows": metric_summary_rows,
        "summary": summary,
        "selected_model_by_sample": selected_model_by_sample,
        "original_prediction_by_sample": original_y_pred_test,
        "selected_prediction_by_metric": y_pred_selected_by_metric,
        "selected_prediction_by_lead_day": y_pred_selected_by_lead_day,
        "selected_prediction_by_lead_day_by_metric": (
            y_pred_selected_by_lead_day_by_metric
        ),
        "selected_model_by_metric": selected_model_by_metric,
    }
    return (
        y_pred_selected,
        y_pred_selected_by_lead_day,
        metrics_by_test_cluster,
        test_model_selection,
    )


def train_cluster_models(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    y_train_by_lead_day: np.ndarray,
    y_val_by_lead_day: np.ndarray,
    y_test_by_lead_day: np.ndarray,
    y_train_by_lead_day_scaled: np.ndarray,
    y_val_by_lead_day_scaled: np.ndarray,
    c_train: np.ndarray,
    c_val: np.ndarray,
    c_test: np.ndarray,
    lstm_units: int,
    lstm_units_2: int,
    dropout_rate: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    early_stopping: bool,
    patience: int,
    lstm_loss_function: str,
    loss_quantiles: Sequence[float],
    loss_quantile_weights: Sequence[float] | str,
    verbose_training: int,
    random_state: int,
    show_console_info: bool,
    test_all_models: bool,
    target_scaler: FeatureScaler | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, object],
    dict[int, dict[str, float]],
    dict[str, object] | None,
]:
    """Train cluster-specific LSTMs and merge their predictions."""
    X_train_lstm = to_lstm_shape(X_train)
    X_val_lstm = to_lstm_shape(X_val)
    X_test_lstm = to_lstm_shape(X_test)
    y_train_by_lead_day = _lead_day_matrix(
        y_train_by_lead_day,
        "y_train_by_lead_day",
    )
    y_val_by_lead_day = _lead_day_matrix(y_val_by_lead_day, "y_val_by_lead_day")
    y_test_by_lead_day = _lead_day_matrix(y_test_by_lead_day, "y_test_by_lead_day")
    y_train_by_lead_day_scaled = _lead_day_matrix(
        y_train_by_lead_day_scaled,
        "y_train_by_lead_day_scaled",
    )
    y_val_by_lead_day_scaled = _lead_day_matrix(
        y_val_by_lead_day_scaled,
        "y_val_by_lead_day_scaled",
    )
    if (
        len(y_train_by_lead_day) != len(y_train)
        or len(y_val_by_lead_day) != len(y_val)
        or len(y_test_by_lead_day) != len(y_test)
        or len(y_train_by_lead_day_scaled) != len(y_train)
        or len(y_val_by_lead_day_scaled) != len(y_val)
    ):
        raise ValueError("Lead-day target matrices must match scalar target lengths.")
    n_outputs = int(y_train_by_lead_day.shape[1])
    if (
        y_val_by_lead_day.shape[1] != n_outputs
        or y_test_by_lead_day.shape[1] != n_outputs
        or y_train_by_lead_day_scaled.shape[1] != n_outputs
        or y_val_by_lead_day_scaled.shape[1] != n_outputs
    ):
        raise ValueError("Lead-day target matrices must have the same horizon width.")

    y_pred_train = np.zeros_like(y_train, dtype=float)
    y_pred_val = np.zeros_like(y_val, dtype=float)
    y_pred_test = np.zeros_like(y_test, dtype=float)
    y_pred_test_by_lead_day = np.zeros_like(y_test_by_lead_day, dtype=float)
    histories_by_cluster: dict[int, object] = {}
    metrics_by_cluster: dict[int, dict[str, float]] = {}
    models_by_cluster: dict[int, LSTMPrecipitationPredictor] = {}

    for cluster_id in sorted(np.unique(c_train)):
        tr_mask = c_train == cluster_id
        va_mask = c_val == cluster_id
        te_mask = c_test == cluster_id
        n_tr, n_va, n_te = tr_mask.sum(), va_mask.sum(), te_mask.sum()

        print_info(
            f"  Cluster {cluster_id}: train={n_tr}, val={n_va}, test={n_te}",
            show_console_info,
        )
        if n_tr == 0:
            continue

        loss_thresholds = None
        loss_weights = None
        if lstm_loss_function == "quantile_weighted_mse":
            loss_thresholds, loss_weights = quantile_weighted_mse_config(
                y_train_by_lead_day_scaled[tr_mask].ravel(),
                quantiles=loss_quantiles,
                weights=loss_quantile_weights,
            )
            print_info(
                "    Quantile-weighted MSE: "
                f"thresholds={loss_thresholds}, weights={loss_weights}",
                show_console_info,
            )

        model = LSTMPrecipitationPredictor(
            input_shape=(1, X_train_lstm.shape[2]),
            lstm_units=lstm_units,
            lstm_units_2=lstm_units_2,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            random_state=random_state,
            loss_function=lstm_loss_function,
            loss_quantile_thresholds_mm=loss_thresholds,
            loss_quantile_weights=loss_weights,
            output_units=n_outputs,
        )
        history = model.fit(
            X_train_lstm[tr_mask],
            y_train_by_lead_day_scaled[tr_mask],
            X_val=X_val_lstm[va_mask] if n_va > 0 else None,
            y_val=y_val_by_lead_day_scaled[va_mask] if n_va > 0 else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose_training if show_console_info else 0,
            early_stopping=early_stopping and n_va > 0,
            patience=patience,
        )

        y_pred_train_by_lead_day = clipped_predictions(
            model,
            X_train_lstm[tr_mask],
            target_scaler,
        )
        y_pred_train[tr_mask] = y_pred_train_by_lead_day[:, -1]
        if n_va > 0:
            y_pred_val_by_lead_day = clipped_predictions(
                model,
                X_val_lstm[va_mask],
                target_scaler,
            )
            y_pred_val[va_mask] = y_pred_val_by_lead_day[:, -1]
        if n_te > 0:
            cluster_y_pred_test_by_lead_day = clipped_predictions(
                model,
                X_test_lstm[te_mask],
                target_scaler,
            )
            y_pred_test_by_lead_day[te_mask] = cluster_y_pred_test_by_lead_day
            y_pred_test[te_mask] = cluster_y_pred_test_by_lead_day[:, -1]
            metrics_by_cluster[int(cluster_id)] = calculate_regression_metrics(
                y_test[te_mask],
                y_pred_test[te_mask],
            )

        histories_by_cluster[int(cluster_id)] = history
        models_by_cluster[int(cluster_id)] = model

    if not test_all_models:
        return (
            y_pred_train,
            y_pred_val,
            y_pred_test,
            y_pred_test_by_lead_day,
            histories_by_cluster,
            metrics_by_cluster,
            None,
        )

    (
        _y_pred_test_selected,
        _y_pred_test_by_lead_day_selected,
        _selected_metrics_by_cluster,
        test_model_selection,
    ) = evaluate_test_samples_with_all_models(
        models_by_cluster,
        X_test_lstm,
        y_test,
        c_test,
        original_y_pred_test=y_pred_test.copy(),
        random_state=random_state,
        target_scaler=target_scaler,
    )

    selection_summary = dict(test_model_selection["summary"])
    print_info(
        "  Oracle cross-cluster transfer diagnostic "
        f"({selection_summary['primary_metric']} primary): "
        f"{selection_summary['switched_samples']} of "
        f"{selection_summary['n_test_samples']} samples switched models",
        show_console_info,
    )

    return (
        y_pred_train,
        y_pred_val,
        y_pred_test,
        y_pred_test_by_lead_day,
        histories_by_cluster,
        metrics_by_cluster,
        test_model_selection,
    )


def run_configuration(
    df: pd.DataFrame,
    config: ExperimentConfig,
    numeric_cols: list[str],
    normalize: bool,
    scaler_type: str,
    precipitation_scaler_type: str | None,
    variance_threshold: float | None,
    output_dir: Path,
    use_all_features: bool,
    forecast_horizon: int,
    manual_zero_tolerance: float,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
    lstm_units: int,
    lstm_units_2: int,
    dropout_rate: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    early_stopping: bool,
    patience: int,
    lstm_loss_function: str,
    loss_quantiles: Sequence[float],
    loss_quantile_weights: Sequence[float] | str,
    verbose_training: int,
    show_console_info: bool,
    test_all_models: bool,
) -> dict[str, float | int | str | None]:
    """Run one sweep configuration and save its artifacts."""
    precipitation_scaler_type = _normalize_precipitation_scaler_type(
        precipitation_scaler_type
    )
    print_section(f"Running {config.name}", show_console_info)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = numeric_cols if use_all_features else numeric_feature_columns(df)
    split_data, daily_splits = create_window_split_data(
        df,
        config,
        feature_columns,
        normalize=normalize,
        scaler_type=scaler_type,
        precipitation_scaler_type=precipitation_scaler_type,
        variance_threshold=variance_threshold,
        forecast_horizon=forecast_horizon,
        random_state=random_state,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        manual_zero_tolerance=manual_zero_tolerance,
    )

    X_train = split_data.X_train
    X_val = split_data.X_val
    X_test = split_data.X_test
    y_train = split_data.y_train
    y_val = split_data.y_val
    y_test = split_data.y_test
    current_train = split_data.current_train
    current_val = split_data.current_val
    current_test = split_data.current_test
    c_train = split_data.c_train
    c_val = split_data.c_val
    c_test = split_data.c_test
    i_train = split_data.i_train
    i_val = split_data.i_val
    i_test = split_data.i_test

    print_info(
        f"  Daily rows: train={len(daily_splits.train)}, "
        f"val={len(daily_splits.val)}, test={len(daily_splits.test)}",
        show_console_info,
    )
    print_info(
        f"  Windows={split_data.n_windows}, "
        f"samples={len(split_data.all_targets)}, features={X_train.shape[1]}, "
        f"clusters={sorted(np.unique(split_data.all_cluster_labels).tolist())}",
        show_console_info,
    )
    print_info(
        f"  Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}",
        show_console_info,
    )
    n_samples = len(split_data.all_targets)
    report_config = {
        **asdict(config),
        "name": config.name,
        "dataset_start_date": df["Data"].min().date().isoformat(),
        "dataset_end_date": df["Data"].max().date().isoformat(),
        "features": feature_columns,
        "normalize": normalize,
        "scaler_type": scaler_type if normalize else "none",
        "precipitation_scaler_type": (
            precipitation_scaler_type if normalize else "none"
        ),
        "target_scale": "normalized" if split_data.target_scaler is not None else "mm",
        "n_samples": n_samples,
        "forecast_horizon": forecast_horizon,
        "manual_zero_tolerance": manual_zero_tolerance,
        "splits": {
            "Training": {
                "samples": len(y_train),
                "percent": len(y_train) / n_samples if n_samples else 0,
            },
            "Validation": {
                "samples": len(y_val),
                "percent": len(y_val) / n_samples if n_samples else 0,
            },
            "Test": {
                "samples": len(y_test),
                "percent": len(y_test) / n_samples if n_samples else 0,
            },
        },
        "lstm_units": lstm_units,
        "lstm_units_2": lstm_units_2,
        "dense_units": [16, 8],
        "output_units": forecast_horizon,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "batch_size": batch_size,
        "early_stopping": early_stopping,
        "patience": patience,
        "optimizer": "AdamW",
        "loss": lstm_loss_function,
        "loss_quantiles": list(loss_quantiles),
        "loss_quantile_weights": loss_quantile_weights,
        "metrics": ["mae", "mse"],
        "test_all_models": test_all_models,
    }

    (
        y_pred_train,
        y_pred_val,
        y_pred_test,
        y_pred_test_by_lead_day,
        histories_by_cluster,
        metrics_by_cluster,
        test_model_selection,
    ) = train_cluster_models(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        split_data.y_train_by_lead_day,
        split_data.y_val_by_lead_day,
        split_data.y_test_by_lead_day,
        split_data.y_train_by_lead_day_scaled,
        split_data.y_val_by_lead_day_scaled,
        c_train,
        c_val,
        c_test,
        lstm_units=lstm_units,
        lstm_units_2=lstm_units_2,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping=early_stopping,
        patience=patience,
        lstm_loss_function=lstm_loss_function,
        loss_quantiles=loss_quantiles,
        loss_quantile_weights=loss_quantile_weights,
        verbose_training=verbose_training,
        random_state=random_state,
        show_console_info=show_console_info,
        test_all_models=test_all_models,
        target_scaler=split_data.target_scaler,
    )

    result = save_run_outputs(
        config,
        output_dir,
        feature_columns,
        split_data.all_targets,
        split_data.all_current_precipitation,
        split_data.all_cluster_labels,
        y_train,
        y_val,
        y_test,
        split_data.test_targets_by_lead_day,
        current_train,
        current_val,
        current_test,
        y_pred_train,
        y_pred_val,
        y_pred_test,
        c_test,
        i_train,
        i_val,
        i_test,
        histories_by_cluster,
        metrics_by_cluster,
        state=config.state,
        station_id=config.station_id,
        pca_variance_threshold=variance_threshold,
        forecast_horizon=forecast_horizon,
        y_pred_test_by_lead_day=y_pred_test_by_lead_day,
        test_target_dates_by_lead_day=split_data.test_target_dates_by_lead_day,
        test_model_selection=test_model_selection,
        cluster_feature_splits={
            "Training": (X_train, c_train),
            "Validation": (X_val, c_val),
            "Test": (X_test, c_test),
        },
        batch_size=batch_size,
    )
    tex_path, pdf_path = generate_config_report(output_dir, report_config)
    print_info(f"  Report: {tex_path.name}", show_console_info)
    if pdf_path is not None:
        print_info(f"  Report PDF: {pdf_path.name}", show_console_info)
    print_info(
        f"  Test metrics: RMSE={result['test_rmse']:.4f}, "
        f"MAE={result['test_mae']:.4f}, R2={result['test_r2']:.4f}",
        show_console_info,
    )
    return result


def run_experiment(
    state: str,
    station_id: str,
    window_sizes: list[int],
    normalize: bool,
    scaler_type: str,
    precipitation_scaler_type: str | None,
    variance_threshold: float | None,
    n_clusters_list: list[int],
    clustering_algorithm: str,
    n_sigma_values: int,
    use_all_features: bool,
    quantitative_metrics: list[str],
    lstm_units: int,
    lstm_units_2: int,
    dropout_rate: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    early_stopping: bool,
    patience: int,
    lstm_loss_function: str,
    loss_quantiles: Sequence[float],
    loss_quantile_weights: Sequence[float] | str,
    verbose_training: int,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
    data_root: Path,
    output_root: Path,
    sweep_name: str | None = None,
    sweep_name_prefix: str = "lstm_cluster_sweep",
    timestamp_format: str = "%Y%m%d_%H%M%S",
    plot_style: Mapping[str, object] | None = None,
    show_console_info: bool = True,
    sigma_values: list[float] | None = None,
    forecast_horizon: int = 1,
    manual_zero_tolerance: float = 0.0,
    test_all_models: bool = True,
    weight_decay: float = 0.0,
) -> Path:
    """Run the configured sweep and return its output directory."""
    clustering_algorithm = clustering_algorithm.lower()
    scaler_type = scaler_type.lower()
    precipitation_scaler_type = _normalize_precipitation_scaler_type(
        precipitation_scaler_type
    )
    lstm_loss_function = validate_loss_function(lstm_loss_function)
    if clustering_algorithm not in SUPPORTED_CLUSTERING_ALGORITHMS:
        supported = ", ".join(SUPPORTED_CLUSTERING_ALGORITHMS)
        raise ValueError(
            f"Unsupported clustering algorithm: {clustering_algorithm!r}. "
            f"Use one of: {supported}"
        )
    if scaler_type not in SUPPORTED_SCALER_TYPES:
        supported = ", ".join(SUPPORTED_SCALER_TYPES)
        raise ValueError(
            f"Unsupported scaler_type: {scaler_type!r}. Use one of: {supported}"
        )
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive.")
    if manual_zero_tolerance < 0:
        raise ValueError("manual_zero_tolerance cannot be negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be a finite non-negative value.")

    timestamp = datetime.now().strftime(timestamp_format)
    if sweep_name is None:
        sweep_name = f"{sweep_name_prefix}_{state}_{station_id}_{timestamp}"
    output_root = Path(output_root)
    sweep_dir = output_root / sweep_name

    setup_styling(plot_style)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print_section("Loading Data", show_console_info)
    df = load_station_daily_data(state=state, station_id=station_id, data_root=data_root)
    numeric_cols = numeric_feature_columns(df)
    print_info(
        f"Loaded {len(df)} rows from {df['Data'].min().date()} to {df['Data'].max().date()}",
        show_console_info,
    )
    print_info(f"Numeric features: {numeric_cols}", show_console_info)

    if clustering_algorithm.lower() == "spectral":
        if sigma_values is None:
            print_section("Calculating Sigma Values", show_console_info)
            selected_sigma_values = calculate_sigma_values(
                df,
                n_values=n_sigma_values,
            ).tolist()
            print_info(
                f"Generated {len(selected_sigma_values)} sigma values: "
                f"{selected_sigma_values}",
                show_console_info,
            )
        else:
            selected_sigma_values = [float(sigma) for sigma in sigma_values]
            if not selected_sigma_values:
                raise ValueError(
                    "sigma_values must contain at least one value in manual mode."
                )
            valid_sigmas = all(
                np.isfinite(sigma) and sigma > 0
                for sigma in selected_sigma_values
            )
            if not valid_sigmas:
                raise ValueError(
                    "Every manual sigma value must be a positive, finite number."
                )
            print_section("Using Manual Sigma Values", show_console_info)
            print_info(
                f"Using {len(selected_sigma_values)} sigma values: "
                f"{selected_sigma_values}",
                show_console_info,
            )
    else:
        selected_sigma_values = [None]

    configurations = build_configurations(
        selected_sigma_values,
        state=state,
        station_id=station_id,
        window_sizes=window_sizes,
        n_clusters_list=n_clusters_list,
        clustering_algorithm=clustering_algorithm,
    )
    print_section("LSTM CLUSTER SWEEP", show_console_info)
    print_info(f"Station: {state}/{station_id}", show_console_info)
    print_info(f"Output directory: {sweep_dir}", show_console_info)
    print_info(
        "Normalization: off"
        if not normalize
        else (
            f"Normalization: covariates={scaler_type}, "
            f"precipitation/target={precipitation_scaler_type or 'none'}"
        ),
        show_console_info,
    )
    print_info(f"LSTM loss: {lstm_loss_function}", show_console_info)
    print_info(
        f"Optimizer: AdamW (learning_rate={learning_rate:g}, "
        f"weight_decay={weight_decay:g})",
        show_console_info,
    )
    print_info(f"Configurations: {len(configurations)}", show_console_info)
    for config in configurations:
        print_info(f"  - {config.name}: {asdict(config)}", show_console_info)

    results = []
    for index, config in enumerate(configurations, start=1):
        print_info(f"\nConfiguration {index}/{len(configurations)}", show_console_info)
        results.append(
            run_configuration(
                df,
                config,
                numeric_cols,
                normalize,
                scaler_type,
                precipitation_scaler_type,
                variance_threshold,
                sweep_dir / config.name,
                use_all_features=use_all_features,
                forecast_horizon=forecast_horizon,
                manual_zero_tolerance=manual_zero_tolerance,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                random_state=random_state,
                lstm_units=lstm_units,
                lstm_units_2=lstm_units_2,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                epochs=epochs,
                batch_size=batch_size,
                early_stopping=early_stopping,
                patience=patience,
                lstm_loss_function=lstm_loss_function,
                loss_quantiles=loss_quantiles,
                loss_quantile_weights=loss_quantile_weights,
                verbose_training=verbose_training,
                show_console_info=show_console_info,
                test_all_models=test_all_models,
            )
        )

    save_sweep_outputs(
        results,
        sweep_dir=sweep_dir,
        state=state,
        station_id=station_id,
        window_sizes=window_sizes,
        n_clusters_list=n_clusters_list,
        clustering_algorithm=clustering_algorithm,
        quantitative_metrics=quantitative_metrics,
    )

    print_section("Sweep Complete", show_console_info)
    print_info(f"Results folder: {sweep_dir}", show_console_info)
    print_info("Sweep-level files:", show_console_info)
    print_info("  - sweep_results.csv", show_console_info)
    print_info("  - sweep_summary.txt", show_console_info)
    print_info("  - overleaf_table.txt", show_console_info)
    print_info("  - overleaf_cluster_metric_tables.txt", show_console_info)
    print_info(
        "Each configuration folder contains metrics, reports, predictions, and plots.",
        show_console_info,
    )
    return sweep_dir
