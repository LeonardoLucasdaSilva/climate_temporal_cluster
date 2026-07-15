"""Output writers for ARMA precipitation baseline experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.lstm_outputs import save_prediction_timeseries_splits
from evaluation.evaluation_plot_tools import (
    plot_error_by_magnitude,
    plot_predictions_vs_actual,
    plot_residuals,
)
from evaluation.metrics import (
    calculate_regression_metrics,
    calculate_zero_precipitation_metrics,
)


def _lead_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain one or more lead-day columns.")
    return matrix


def _final_lead(values: np.ndarray, name: str) -> np.ndarray:
    return _lead_matrix(values, name)[:, -1]


def _finite_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return {
            "MSE": float("nan"),
            "RMSE": float("nan"),
            "MAE": float("nan"),
            "RMSLE": float("nan"),
            "R2": float("nan"),
            "MAPE": float("nan"),
        }
    return calculate_regression_metrics(y_true[mask], y_pred[mask])


def _safe_histogram_bins(values: np.ndarray, bins: int) -> int | np.ndarray:
    """Return histogram bins that work for nearly constant residuals."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return bins
    min_value = float(finite_values.min())
    max_value = float(finite_values.max())
    if not np.isclose(min_value, max_value, rtol=1e-12, atol=1e-12):
        return bins
    spread = max(abs(min_value) * 0.05, 0.5)
    return np.linspace(min_value - spread, max_value + spread, 2)


def metrics_dataframe(
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> pd.DataFrame:
    """Return split-level metrics as a tidy dataframe."""
    rows = []
    for split, metrics in (
        ("Train", train_metrics),
        ("Validation", val_metrics),
        ("Test", test_metrics),
    ):
        row = {"split": split}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def save_arma_lead_day_diagnostics(
    y_test_by_lead_day: np.ndarray,
    y_pred_test_by_lead_day: np.ndarray,
    current_test: np.ndarray,
    test_indices: np.ndarray,
    test_target_dates_by_lead_day: np.ndarray,
    output_dir: Path,
    forecast_horizon: int,
) -> pd.DataFrame:
    """Save per-lead-day ARMA metrics, tables, and diagnostic plots."""
    diag_dir = output_dir / "forecast_horizon_diagnostics"
    diag_dir.mkdir(exist_ok=True)

    actual_matrix = _lead_matrix(y_test_by_lead_day, "y_test_by_lead_day")
    predicted_matrix = _lead_matrix(
        y_pred_test_by_lead_day,
        "y_pred_test_by_lead_day",
    )
    if actual_matrix.shape != predicted_matrix.shape:
        raise ValueError("Actual and predicted lead-day matrices must match.")

    date_matrix = np.asarray(test_target_dates_by_lead_day)
    if date_matrix.ndim == 1:
        date_matrix = date_matrix.reshape(-1, 1)
    if date_matrix.shape != actual_matrix.shape:
        raise ValueError("Lead-day date matrix must match target matrix shape.")

    rows: list[dict[str, float | int | str]] = []
    for row_index in range(actual_matrix.shape[0]):
        for lead_offset in range(actual_matrix.shape[1]):
            actual = float(actual_matrix[row_index, lead_offset])
            predicted = float(predicted_matrix[row_index, lead_offset])
            rows.append(
                {
                    "sample_index": int(row_index),
                    "window_index": int(test_indices[row_index]),
                    "lead_day": int(lead_offset + 1),
                    "target_date": str(pd.Timestamp(date_matrix[row_index, lead_offset]).date()),
                    "current_window_precipitation_mm": float(current_test[row_index]),
                    "actual_mm": actual,
                    "predicted_mm": predicted,
                    "residual_mm": actual - predicted,
                    "absolute_error_mm": abs(actual - predicted),
                    "squared_error": (actual - predicted) ** 2,
                }
            )
    lead_df = pd.DataFrame(rows)
    lead_df.to_csv(diag_dir / "test_prediction_by_lead_day.csv", index=False)

    metrics_rows = []
    for lead_day, lead_values in lead_df.groupby("lead_day", sort=True):
        metrics = _finite_metrics(
            lead_values["actual_mm"].to_numpy(dtype=float),
            lead_values["predicted_mm"].to_numpy(dtype=float),
        )
        row = {"lead_day": int(lead_day), "n_test": int(len(lead_values))}
        row.update(metrics)
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(
        diag_dir / "test_prediction_metrics_by_lead_day.csv",
        index=False,
    )

    _save_lead_day_error_plot(metrics_df, diag_dir)
    _save_true_vs_predicted_grid(lead_df, metrics_df, diag_dir)
    _save_true_vs_predicted_by_lead(lead_df, metrics_df, diag_dir)
    _save_lead_day_timeseries(lead_df, diag_dir)
    return metrics_df


def _save_lead_day_error_plot(metrics_df: pd.DataFrame, diag_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(metrics_df["lead_day"], metrics_df["RMSE"], marker="o", linewidth=2)
    axes[0].set_xlabel("Lead day")
    axes[0].set_ylabel("RMSE (mm)")
    axes[0].set_title("ARMA Prediction Error by Lead Day - RMSE")
    axes[0].set_xticks(metrics_df["lead_day"])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(metrics_df["lead_day"], metrics_df["MAE"], marker="o", linewidth=2)
    axes[1].set_xlabel("Lead day")
    axes[1].set_ylabel("MAE (mm)")
    axes[1].set_title("ARMA Prediction Error by Lead Day - MAE")
    axes[1].set_xticks(metrics_df["lead_day"])
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(diag_dir / "12_prediction_error_by_lead_day.png")
    plt.close(fig)


def _save_true_vs_predicted_grid(
    lead_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    diag_dir: Path,
) -> None:
    n_leads = int(metrics_df["lead_day"].max()) if not metrics_df.empty else 0
    if n_leads == 0:
        return

    n_cols = min(3, n_leads)
    n_rows = int(np.ceil(n_leads / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    for ax in axes.ravel()[n_leads:]:
        ax.set_visible(False)

    for ax, lead_day in zip(axes.ravel(), range(1, n_leads + 1)):
        values = lead_df[lead_df["lead_day"] == lead_day]
        metrics = metrics_df.loc[metrics_df["lead_day"] == lead_day].iloc[0]
        actual = values["actual_mm"].to_numpy(dtype=float)
        predicted = values["predicted_mm"].to_numpy(dtype=float)
        ax.scatter(actual, predicted, alpha=0.55, s=22)
        finite = np.concatenate([actual[np.isfinite(actual)], predicted[np.isfinite(predicted)]])
        if finite.size:
            low = float(finite.min())
            high = float(finite.max())
            ax.plot([low, high], [low, high], "r--", linewidth=1.4)
        ax.set_xlabel(f"Actual D+{lead_day} (mm)")
        ax.set_ylabel(f"ARMA prediction D+{lead_day} (mm)")
        ax.set_title(
            f"D+{lead_day}: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}"
        )
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(diag_dir / "13_true_vs_predicted_by_lead_day.png")
    plt.close(fig)


def _save_true_vs_predicted_by_lead(
    lead_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    diag_dir: Path,
) -> None:
    by_lead_dir = diag_dir / "true_vs_predicted_by_lead_day"
    by_lead_dir.mkdir(exist_ok=True)

    for lead_day, values in lead_df.groupby("lead_day", sort=True):
        metrics = metrics_df.loc[metrics_df["lead_day"] == lead_day].iloc[0]
        actual = values["actual_mm"].to_numpy(dtype=float)
        predicted = values["predicted_mm"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(actual, predicted, alpha=0.6, s=28)
        finite = np.concatenate([actual[np.isfinite(actual)], predicted[np.isfinite(predicted)]])
        if finite.size:
            low = float(finite.min())
            high = float(finite.max())
            ax.plot([low, high], [low, high], "r--", linewidth=1.5)
        ax.set_xlabel(f"Actual precipitation at D+{int(lead_day)} (mm)")
        ax.set_ylabel(f"ARMA prediction at D+{int(lead_day)} (mm)")
        ax.set_title(
            f"True vs Predicted at D+{int(lead_day)}: "
            f"RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(by_lead_dir / f"true_vs_predicted_lead_day_{int(lead_day):02d}.png")
        plt.close(fig)


def _save_lead_day_timeseries(lead_df: pd.DataFrame, diag_dir: Path) -> None:
    lead_days = sorted(lead_df["lead_day"].unique())
    n_leads = len(lead_days)
    n_cols = min(2, n_leads)
    n_rows = int(np.ceil(n_leads / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.2 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )
    for ax in axes.ravel()[n_leads:]:
        ax.set_visible(False)

    for ax, lead_day in zip(axes.ravel(), lead_days):
        values = lead_df[lead_df["lead_day"] == lead_day].sort_values("target_date")
        x_values = pd.to_datetime(values["target_date"], errors="coerce")
        ax.plot(
            x_values,
            values["actual_mm"],
            label=f"Actual D+{int(lead_day)}",
            linewidth=1.5,
        )
        ax.plot(
            x_values,
            values["predicted_mm"],
            label=f"ARMA D+{int(lead_day)}",
            linewidth=1.5,
        )
        ax.set_xlabel("Target Date")
        ax.set_ylabel("Precipitation (mm)")
        ax.set_title(f"Lead day D+{int(lead_day)}")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(diag_dir / "14_prediction_vs_actual_timeseries_by_lead_day.png")
    plt.close(fig)


def save_arma_visualizations(
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    y_test_by_lead_day: np.ndarray,
    y_pred_test_by_lead_day: np.ndarray,
    test_target_dates_by_lead_day: np.ndarray,
    output_dir: Path,
    forecast_horizon: int,
) -> None:
    """Save ARMA plots equivalent to the main LSTM prediction diagnostics."""
    prediction_dir = output_dir / "prediction_overview"
    residual_dir = output_dir / "residual_diagnostics"
    prediction_dir.mkdir(exist_ok=True)
    residual_dir.mkdir(exist_ok=True)

    fig, _ = plot_predictions_vs_actual(
        y_test,
        y_pred_test,
        title="Test Set: ARMA Predictions vs Actual Precipitation",
    )
    fig.savefig(prediction_dir / "02_predictions_vs_actual.png")
    plt.close(fig)

    save_prediction_timeseries_splits(
        y_test_by_lead_day,
        y_pred_test_by_lead_day,
        output_dir,
        forecast_horizon=forecast_horizon,
        test_dates_by_lead_day=test_target_dates_by_lead_day,
    )

    fig, _ = plot_residuals(
        y_test,
        y_pred_test,
        title="Test Set: ARMA Residual Analysis",
    )
    fig.savefig(residual_dir / "03_residuals_analysis.png")
    plt.close(fig)

    fig, _ = plot_error_by_magnitude(
        y_test,
        y_pred_test,
        n_bins=10,
        title="Test Set: ARMA Error by Precipitation Magnitude",
    )
    fig.savefig(residual_dir / "04_error_by_magnitude.png")
    plt.close(fig)


def save_arma_model_fit_diagnostics(
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    output_dir: Path,
) -> None:
    """Save a compact training-residual diagnostic for the fitted ARMA model."""
    model_dir = output_dir / "model_fit"
    model_dir.mkdir(exist_ok=True)
    residuals = np.asarray(y_train, dtype=float) - np.asarray(y_pred_train, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(residuals, linewidth=1.3)
    axes[0].axhline(0.0, color="r", linestyle="--", linewidth=1.4)
    axes[0].set_xlabel("Training sample index")
    axes[0].set_ylabel("Residual (mm)")
    axes[0].set_title("ARMA Training Residuals")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(
        residuals[np.isfinite(residuals)],
        bins=_safe_histogram_bins(residuals, 35),
        edgecolor="black",
        alpha=0.75,
    )
    axes[1].set_xlabel("Residual (mm)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Training Residual Distribution")
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(model_dir / "01_arma_training_residuals.png")
    plt.close(fig)


def save_arma_summary(
    output_dir: Path,
    config: object,
    state: str,
    station_id: str,
    forecast_horizon: int,
    train_ratio: float,
    val_ratio: float,
    trend: str,
    split_sizes: dict[str, int],
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    zero_metrics: dict[str, float],
    aic: float,
    bic: float,
    hqic: float,
    clip_negative_predictions: bool,
) -> None:
    """Save a readable summary for one ARMA run."""
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("ARMA BASELINE CONFIGURATION\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Run folder: {config.name}\n")
        f.write(f"Station: {state}/{station_id}\n")
        f.write(f"Window size for target alignment: {config.window_size}\n")
        f.write(f"Forecast horizon: +{forecast_horizon} day(s)\n")
        f.write(f"ARMA order: p={config.p}, q={config.q}\n")
        f.write(f"Trend: {trend}\n")
        f.write(f"Clip negative predictions: {clip_negative_predictions}\n")
        f.write(f"Splits: {split_sizes}\n")
        f.write(f"Split ratios: train={train_ratio}, validation={val_ratio}\n")
        f.write(f"AIC={aic:.4f}, BIC={bic:.4f}, HQIC={hqic:.4f}\n\n")

        f.write("Metrics\n")
        f.write("-" * 72 + "\n")
        for split, metrics in (
            ("Train", train_metrics),
            ("Validation", val_metrics),
            ("Test", test_metrics),
        ):
            f.write(
                f"{split:<10} RMSE={metrics['RMSE']:.4f}  "
                f"MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}  "
                f"RMSLE={metrics['RMSLE']:.4f}\n"
            )

        f.write("\nZero vs rainy days on test set\n")
        f.write("-" * 72 + "\n")
        for key, value in zero_metrics.items():
            f.write(f"{key}: {value}\n")


def save_arma_evaluation_report(
    output_dir: Path,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    zero_metrics: dict[str, float],
) -> None:
    """Save a plain-text evaluation report for the ARMA baseline."""
    with open(output_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("ARMA PRECIPITATION BASELINE - EVALUATION REPORT\n")
        f.write("=" * 72 + "\n\n")
        for split, metrics in (
            ("TRAINING SET", train_metrics),
            ("VALIDATION SET", val_metrics),
            ("TEST SET", test_metrics),
        ):
            f.write(f"[{split} METRICS]\n")
            for metric_name, metric_value in metrics.items():
                suffix = "%" if metric_name == "MAPE" else ""
                f.write(f"{metric_name:8s}: {metric_value:.4f}{suffix}\n")
            f.write("\n")

        f.write("[ZERO vs RAINY DAYS ANALYSIS] - Test Set\n")
        for key, value in zero_metrics.items():
            f.write(f"{key}: {value}\n")


def save_arma_run_outputs(
    config: object,
    output_dir: Path,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    y_train_by_lead_day: np.ndarray,
    y_val_by_lead_day: np.ndarray,
    y_test_by_lead_day: np.ndarray,
    current_train: np.ndarray,
    current_val: np.ndarray,
    current_test: np.ndarray,
    y_pred_train_by_lead_day: np.ndarray,
    y_pred_val_by_lead_day: np.ndarray,
    y_pred_test_by_lead_day: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    test_target_dates_by_lead_day: np.ndarray,
    state: str,
    station_id: str,
    forecast_horizon: int,
    train_ratio: float,
    val_ratio: float,
    trend: str,
    aic: float,
    bic: float,
    hqic: float,
    model_summary: str,
    clip_negative_predictions: bool,
) -> dict[str, float | int | str | None]:
    """Save all artifacts for one ARMA run and return a sweep result row."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred_train = _final_lead(y_pred_train_by_lead_day, "y_pred_train_by_lead_day")
    y_pred_val = _final_lead(y_pred_val_by_lead_day, "y_pred_val_by_lead_day")
    y_pred_test = _final_lead(y_pred_test_by_lead_day, "y_pred_test_by_lead_day")

    train_metrics = _finite_metrics(y_train, y_pred_train)
    val_metrics = _finite_metrics(y_val, y_pred_val)
    test_metrics = _finite_metrics(y_test, y_pred_test)
    finite_test = np.isfinite(y_test) & np.isfinite(y_pred_test)
    zero_metrics = calculate_zero_precipitation_metrics(
        np.asarray(y_test)[finite_test],
        np.asarray(y_pred_test)[finite_test],
    )

    metrics_dataframe(train_metrics, val_metrics, test_metrics).to_csv(
        output_dir / "metrics_summary.csv",
        index=False,
    )

    predictions_df = pd.DataFrame(
        {
            "actual": y_test,
            "predicted": y_pred_test,
            "residual": np.asarray(y_test) - y_pred_test,
            "current_window_precipitation_mm": current_test,
            "forecast_horizon": forecast_horizon,
            "target_minus_current_mm": np.asarray(y_test) - np.asarray(current_test),
            "window_index": test_indices,
        }
    )
    y_test_matrix = _lead_matrix(y_test_by_lead_day, "y_test_by_lead_day")
    y_pred_matrix = _lead_matrix(y_pred_test_by_lead_day, "y_pred_test_by_lead_day")
    for lead_offset in range(y_test_matrix.shape[1]):
        lead_day = lead_offset + 1
        predictions_df[f"actual_lead_day_{lead_day}"] = y_test_matrix[:, lead_offset]
        predictions_df[f"predicted_lead_day_{lead_day}"] = y_pred_matrix[:, lead_offset]
        predictions_df[f"residual_lead_day_{lead_day}"] = (
            y_test_matrix[:, lead_offset] - y_pred_matrix[:, lead_offset]
        )
    predictions_df.sort_values("window_index").to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )

    lead_metrics = save_arma_lead_day_diagnostics(
        y_test_by_lead_day=y_test_by_lead_day,
        y_pred_test_by_lead_day=y_pred_test_by_lead_day,
        current_test=current_test,
        test_indices=test_indices,
        test_target_dates_by_lead_day=test_target_dates_by_lead_day,
        output_dir=output_dir,
        forecast_horizon=forecast_horizon,
    )
    save_arma_visualizations(
        y_test=y_test,
        y_pred_test=y_pred_test,
        y_test_by_lead_day=y_test_by_lead_day,
        y_pred_test_by_lead_day=y_pred_test_by_lead_day,
        test_target_dates_by_lead_day=test_target_dates_by_lead_day,
        output_dir=output_dir,
        forecast_horizon=forecast_horizon,
    )
    save_arma_model_fit_diagnostics(y_train, y_pred_train, output_dir)
    (output_dir / "arma_model_summary.txt").write_text(model_summary, encoding="utf-8")

    split_sizes = {
        "train": int(len(y_train)),
        "validation": int(len(y_val)),
        "test": int(len(y_test)),
    }
    save_arma_summary(
        output_dir=output_dir,
        config=config,
        state=state,
        station_id=station_id,
        forecast_horizon=forecast_horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        trend=trend,
        split_sizes=split_sizes,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        zero_metrics=zero_metrics,
        aic=aic,
        bic=bic,
        hqic=hqic,
        clip_negative_predictions=clip_negative_predictions,
    )
    save_arma_evaluation_report(
        output_dir=output_dir,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        zero_metrics=zero_metrics,
    )

    result = {
        "run_name": config.name,
        "window_size": int(config.window_size),
        "p": int(config.p),
        "q": int(config.q),
        "forecast_horizon": int(forecast_horizon),
        "aic": aic,
        "bic": bic,
        "hqic": hqic,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "test_mse": test_metrics["MSE"],
        "test_rmse": test_metrics["RMSE"],
        "test_mae": test_metrics["MAE"],
        "test_rmsle": test_metrics["RMSLE"],
        "test_r2": test_metrics["R2"],
        "test_mape": test_metrics["MAPE"],
        "zero_days_ratio": zero_metrics["zero_days_ratio"],
        "rainy_days_rmse": zero_metrics.get("rainy_days_rmse", np.nan),
        "lead_day_metrics_path": (
            "forecast_horizon_diagnostics/test_prediction_metrics_by_lead_day.csv"
            if not lead_metrics.empty
            else None
        ),
    }
    return result


def _format_float(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.4f}"


def save_arma_sweep_outputs(
    sweep_dir: Path,
    result_rows: Sequence[dict[str, float | int | str | None]],
    failure_rows: Sequence[dict[str, str]],
    state: str,
    station_id: str,
    forecast_horizon: int,
    arma_orders: Sequence[tuple[int, int]],
    window_sizes: Sequence[int],
) -> None:
    """Save sweep-level CSV and compact text summaries."""
    results_df = pd.DataFrame(result_rows)
    failures_df = pd.DataFrame(failure_rows)
    results_df.to_csv(sweep_dir / "sweep_results.csv", index=False)
    if not failures_df.empty:
        failures_df.to_csv(sweep_dir / "failed_runs.csv", index=False)

    with open(sweep_dir / "sweep_summary.txt", "w", encoding="utf-8") as f:
        f.write("ARMA BASELINE SWEEP SUMMARY\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Station: {state}/{station_id}\n")
        f.write(f"Forecast horizon: +{forecast_horizon} day(s)\n")
        f.write(f"Window sizes: {list(window_sizes)}\n")
        f.write(f"ARMA orders: {list(arma_orders)}\n")
        f.write(f"Successful runs: {len(results_df)}\n")
        f.write(f"Failed runs: {len(failures_df)}\n\n")

        if not results_df.empty:
            ranked = results_df.sort_values("test_rmse")
            f.write("Best runs by test RMSE\n")
            f.write("-" * 72 + "\n")
            for _, row in ranked.head(10).iterrows():
                f.write(
                    f"{row['run_name']}: RMSE={_format_float(row['test_rmse'])}, "
                    f"MAE={_format_float(row['test_mae'])}, "
                    f"AIC={_format_float(row['aic'])}\n"
                )

        if not failures_df.empty:
            f.write("\nFailed runs\n")
            f.write("-" * 72 + "\n")
            for _, row in failures_df.iterrows():
                f.write(f"{row['run_name']}: {row['error']}\n")

    with open(sweep_dir / "overleaf_table.txt", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\hline\n")
        f.write("Run & p & q & RMSE & MAE & R2 \\\\\n")
        f.write("\\hline\n")
        if not results_df.empty:
            ranked = results_df.sort_values("test_rmse")
            for _, row in ranked.iterrows():
                f.write(
                    f"{row['run_name']} & {int(row['p'])} & {int(row['q'])} & "
                    f"{_format_float(row['test_rmse'])} & "
                    f"{_format_float(row['test_mae'])} & "
                    f"{_format_float(row['test_r2'])} \\\\\n"
                )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
