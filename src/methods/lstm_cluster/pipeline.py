"""Run the LSTM-by-cluster precipitation sweep."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from data.load_data import load_station_daily_data
from data.lstm_outputs import save_run_outputs, save_sweep_outputs
from evaluation.metrics import calculate_regression_metrics
from models.lstm import LSTMPrecipitationPredictor
from methods.cluster.cluster_pipeline import (
    PCA_VARIANCE_THRESHOLD,
    SUPPORTED_CLUSTERING_ALGORITHMS,
    cluster_feature_matrix,
    create_cluster_feature_matrix,
    numeric_feature_columns,
)
from methods.lstm_cluster.console import print_info, print_section
from methods.lstm_cluster.report import generate_config_report
from methods.tools.precipitation_utils import (
    horizon_precipitation,
    precipitation_targets,
)
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


def _mapping_from_config(value: object) -> Mapping[str, object]:
    """Return nested config dictionaries safely."""
    return value if isinstance(value, Mapping) else {}


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


def split_by_cluster(
    X: np.ndarray,
    y: np.ndarray,
    cluster_labels: np.ndarray,
    sample_indices: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, ...]:
    """Split samples while preserving cluster balance when possible."""
    test_ratio = 1 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("TRAIN_RATIO + VAL_RATIO must be smaller than 1.")

    counts = pd.Series(cluster_labels).value_counts()
    stratify = cluster_labels if counts.min() >= 3 else None

    X_tv, X_test, y_tv, y_test, c_tv, c_test, i_tv, i_test = train_test_split(
        X,
        y,
        cluster_labels,
        sample_indices,
        test_size=test_ratio,
        stratify=stratify,
        random_state=random_state,
    )

    tv_counts = pd.Series(c_tv).value_counts()
    stratify_tv = c_tv if len(tv_counts) > 0 and tv_counts.min() >= 2 else None
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val, c_train, c_val, i_train, i_val = train_test_split(
        X_tv,
        y_tv,
        c_tv,
        i_tv,
        test_size=val_ratio_adjusted,
        stratify=stratify_tv,
        random_state=random_state,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        c_train,
        c_val,
        c_test,
        i_train,
        i_val,
        i_test,
    )


def to_lstm_shape(X: np.ndarray) -> np.ndarray:
    """Represent each flattened window as a one-step LSTM sequence."""
    return X.reshape(X.shape[0], 1, X.shape[1])


def clipped_predictions(model: LSTMPrecipitationPredictor, X: np.ndarray) -> np.ndarray:
    """Predict precipitation and clip impossible negative values."""
    return np.maximum(model.predict(X).ravel(), 0.0)


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
) -> tuple[np.ndarray, dict[int, dict[str, float]], dict[str, object]]:
    """Select the best trained LSTM per test sample and metric."""
    model_items = sorted(models_by_cluster.items())
    model_cluster_ids = np.array([cluster_id for cluster_id, _model in model_items], dtype=int)
    y_pred_by_model = np.column_stack(
        [clipped_predictions(model, X_test_lstm) for _cluster_id, model in model_items]
    )
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
    selected_model_by_metric: dict[str, np.ndarray] = {}
    for metric_name, errors in per_metric_errors.items():
        comparable_errors = np.where(np.isfinite(errors), errors, np.inf)
        best_offsets = np.argmin(comparable_errors, axis=1)
        y_pred_selected_by_metric[metric_name] = y_pred_by_model[
            np.arange(len(y_test)),
            best_offsets,
        ]
        selected_model_by_metric[metric_name] = model_cluster_ids[best_offsets]

    y_pred_selected = y_pred_selected_by_metric[primary_metric]
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
        "selected_model_by_metric": selected_model_by_metric,
    }
    return y_pred_selected, metrics_by_test_cluster, test_model_selection


def train_cluster_models(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    c_train: np.ndarray,
    c_val: np.ndarray,
    c_test: np.ndarray,
    lstm_units: int,
    lstm_units_2: int,
    dropout_rate: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    early_stopping: bool,
    patience: int,
    verbose_training: int,
    random_state: int,
    show_console_info: bool,
    test_all_models: bool,
) -> tuple[
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

    y_pred_train = np.zeros_like(y_train, dtype=float)
    y_pred_val = np.zeros_like(y_val, dtype=float)
    y_pred_test = np.zeros_like(y_test, dtype=float)
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

        model = LSTMPrecipitationPredictor(
            input_shape=(1, X_train_lstm.shape[2]),
            lstm_units=lstm_units,
            lstm_units_2=lstm_units_2,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        history = model.fit(
            X_train_lstm[tr_mask],
            y_train[tr_mask],
            X_val=X_val_lstm[va_mask] if n_va > 0 else None,
            y_val=y_val[va_mask] if n_va > 0 else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose_training if show_console_info else 0,
            early_stopping=early_stopping and n_va > 0,
            patience=patience,
        )

        y_pred_train[tr_mask] = clipped_predictions(model, X_train_lstm[tr_mask])
        if n_va > 0:
            y_pred_val[va_mask] = clipped_predictions(model, X_val_lstm[va_mask])
        if n_te > 0:
            y_pred_test[te_mask] = clipped_predictions(model, X_test_lstm[te_mask])
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
            histories_by_cluster,
            metrics_by_cluster,
            None,
        )

    y_pred_test_selected, selected_metrics_by_cluster, test_model_selection = (
        evaluate_test_samples_with_all_models(
            models_by_cluster,
            X_test_lstm,
            y_test,
            c_test,
            original_y_pred_test=y_pred_test.copy(),
            random_state=random_state,
        )
    )

    selection_summary = dict(test_model_selection["summary"])
    print_info(
        "  Per-sample test model selection "
        f"({selection_summary['primary_metric']} primary): "
        f"{selection_summary['switched_samples']} of "
        f"{selection_summary['n_test_samples']} samples switched models",
        show_console_info,
    )

    return (
        y_pred_train,
        y_pred_val,
        y_pred_test_selected,
        histories_by_cluster,
        selected_metrics_by_cluster,
        test_model_selection,
    )


def run_configuration(
    df: pd.DataFrame,
    config: ExperimentConfig,
    numeric_cols: list[str],
    normalize: bool,
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
    epochs: int,
    batch_size: int,
    early_stopping: bool,
    patience: int,
    verbose_training: int,
    show_console_info: bool,
    test_all_models: bool,
) -> dict[str, float | int | str | None]:
    """Run one sweep configuration and save its artifacts."""
    print_section(f"Running {config.name}", show_console_info)
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = numeric_cols if use_all_features else None
    windows, windows_flat, _, _, feature_columns = create_cluster_feature_matrix(
        df,
        window_size=config.window_size,
        columns=columns,
        normalize=normalize,
        variance_threshold=variance_threshold,
        verbose=show_console_info,
    )
    all_horizon_rain = horizon_precipitation(
        df,
        window_size=config.window_size,
        horizon=forecast_horizon,
    )[:len(windows)]
    labels = cluster_feature_matrix(
        windows_flat,
        n_clusters=config.n_clusters,
        algorithm=config.algorithm,
        sigma=config.sigma,
        random_state=random_state,
        horizon_rain=all_horizon_rain,
        zero_tolerance=manual_zero_tolerance,
    )
    valid_indices, targets = precipitation_targets(
        df,
        config.window_size,
        len(windows),
        horizon=forecast_horizon,
    )

    X = windows_flat[valid_indices]
    c = labels[valid_indices]
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        c_train,
        c_val,
        c_test,
        _i_train,
        _i_val,
        i_test,
    ) = split_by_cluster(
        X,
        targets,
        c,
        valid_indices,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_state=random_state,
    )

    print_info(
        f"  Windows={len(windows)}, samples={len(targets)}, "
        f"features={X.shape[1]}, clusters={sorted(np.unique(c).tolist())}",
        show_console_info,
    )
    print_info(
        f"  Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}",
        show_console_info,
    )
    n_samples = len(targets)
    report_config = {
        **asdict(config),
        "name": config.name,
        "dataset_start_date": df["Data"].min().date().isoformat(),
        "dataset_end_date": df["Data"].max().date().isoformat(),
        "features": feature_columns,
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
        "output_units": 1,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "early_stopping": early_stopping,
        "patience": patience,
        "optimizer": "Adam",
        "loss": "mean_squared_error",
        "metrics": ["mae", "mse"],
        "test_all_models": test_all_models,
    }

    (
        y_pred_train,
        y_pred_val,
        y_pred_test,
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
        c_train,
        c_val,
        c_test,
        lstm_units=lstm_units,
        lstm_units_2=lstm_units_2,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping=early_stopping,
        patience=patience,
        verbose_training=verbose_training,
        random_state=random_state,
        show_console_info=show_console_info,
        test_all_models=test_all_models,
    )

    result = save_run_outputs(
        config,
        output_dir,
        feature_columns,
        targets,
        c,
        y_train,
        y_val,
        y_test,
        y_pred_train,
        y_pred_val,
        y_pred_test,
        c_test,
        i_test,
        histories_by_cluster,
        metrics_by_cluster,
        state=config.state,
        station_id=config.station_id,
        pca_variance_threshold=variance_threshold,
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
) -> Path:
    """Run the configured sweep and return its output directory."""
    clustering_algorithm = clustering_algorithm.lower()
    if clustering_algorithm not in SUPPORTED_CLUSTERING_ALGORITHMS:
        supported = ", ".join(SUPPORTED_CLUSTERING_ALGORITHMS)
        raise ValueError(
            f"Unsupported clustering algorithm: {clustering_algorithm!r}. "
            f"Use one of: {supported}"
        )
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive.")
    if manual_zero_tolerance < 0:
        raise ValueError("manual_zero_tolerance cannot be negative.")

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
                epochs=epochs,
                batch_size=batch_size,
                early_stopping=early_stopping,
                patience=patience,
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
