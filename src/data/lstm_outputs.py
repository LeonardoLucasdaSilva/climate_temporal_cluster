"""Output writers for LSTM clustering experiments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping, Protocol

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score

from evaluation import (
    plot_cluster_performance,
    plot_error_by_magnitude,
    plot_predictions_vs_actual,
    plot_residuals,
)
from evaluation.metrics import (
    calculate_regression_metrics,
    calculate_zero_precipitation_metrics,
    create_evaluation_report,
)
from methods.tools.precipitation_utils import precipitation_bin_edges


class ExperimentConfigLike(Protocol):
    """Fields required from an experiment configuration."""

    window_size: int
    n_clusters: int
    algorithm: str
    sigma: float | None
    name: str


def save_training_history_plots(
    histories_by_cluster: dict[int, object],
    output_dir: Path,
) -> None:
    """Save loss and MAE curves for each cluster model."""
    plot_dir = output_dir / "model_fit"
    plot_dir.mkdir(exist_ok=True)

    for cluster_id, history in histories_by_cluster.items():
        hist = history.history
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        axes[0].plot(hist.get("loss", []), label="Train", linewidth=2)
        if "val_loss" in hist:
            axes[0].plot(hist["val_loss"], label="Validation", linewidth=2)
        axes[0].set_title(f"Cluster {cluster_id}: Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE")
        axes[0].legend()

        axes[1].plot(hist.get("mae", []), label="Train", linewidth=2)
        if "val_mae" in hist:
            axes[1].plot(hist["val_mae"], label="Validation", linewidth=2)
        axes[1].set_title(f"Cluster {cluster_id}: MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(plot_dir / f"01_training_history_cluster_{cluster_id}.png")
        plt.close(fig)


def save_cluster_precipitation_histograms(
    y_test: np.ndarray,
    c_test: np.ndarray,
    output_dir: Path,
) -> None:
    """Save one actual-precipitation histogram per cluster."""
    hist_dir = output_dir / "cluster_precipitation_histograms"
    hist_dir.mkdir(exist_ok=True)

    for cluster_id in sorted(np.unique(c_test)):
        mask = c_test == cluster_id
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(y_test[mask], bins=25, color="#4C78A8", edgecolor="black", alpha=0.8)
        ax.set_title(f"Cluster {int(cluster_id)}: Precipitation Occurrences")
        ax.set_xlabel("Precipitation (mm)")
        ax.set_ylabel("Number of occurrences")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(
            hist_dir / f"cluster_{int(cluster_id)}_precipitation_histogram.png",
        )
        plt.close(fig)


def save_input_precipitation_assignments(
    forecast_horizon_precipitation: np.ndarray,
    current_precipitation: np.ndarray,
    cluster_labels: np.ndarray,
    output_dir: Path,
    forecast_horizon: int,
) -> None:
    """Save the forecast-horizon precipitation target for each input window."""
    assignments = pd.DataFrame(
        {
            "input_index": np.arange(len(forecast_horizon_precipitation)),
            "cluster": cluster_labels.astype(int),
            "forecast_horizon": forecast_horizon,
            "current_window_precipitation_mm": current_precipitation,
            "forecast_horizon_precipitation_mm": forecast_horizon_precipitation,
            "target_minus_current_mm": (
                forecast_horizon_precipitation - current_precipitation
            ),
            "next_day_precipitation_mm": forecast_horizon_precipitation,
        }
    )
    assignments.to_csv(
        output_dir / "input_forecast_horizon_precipitation_by_cluster.csv",
        index=False,
    )
    assignments.to_csv(
        output_dir / "input_next_day_precipitation_by_cluster.csv",
        index=False,
    )


def save_input_precipitation_distribution_by_cluster(
    forecast_horizon_precipitation: np.ndarray,
    cluster_labels: np.ndarray,
    output_dir: Path,
    forecast_horizon: int,
) -> None:
    """Save horizontal histograms of horizon precipitation for each cluster."""
    hist_dir = output_dir / "input_precipitation_distribution_by_cluster"
    hist_dir.mkdir(exist_ok=True)

    bin_edges = precipitation_bin_edges(forecast_horizon_precipitation)
    cluster_ids = sorted(np.unique(cluster_labels))
    n_clusters = len(cluster_ids)
    n_cols = min(3, n_clusters)
    n_rows = int(np.ceil(n_clusters / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.2 * n_rows),
        sharey=True,
        squeeze=False,
    )

    for ax in axes.ravel()[n_clusters:]:
        ax.set_visible(False)

    for ax, cluster_id in zip(axes.ravel(), cluster_ids):
        mask = cluster_labels == cluster_id
        values = forecast_horizon_precipitation[mask]
        rainy_ratio = float(np.mean(values > 0)) if values.size else 0.0
        ax.hist(
            values,
            bins=bin_edges,
            orientation="horizontal",
            color="#4C78A8",
            edgecolor="white",
            alpha=0.9,
        )
        ax.set_title(f"Cluster {int(cluster_id)} | n={values.size} | rainy={rainy_ratio:.1%}")
        ax.set_xlabel("Samples")
        ax.set_ylabel(f"Horizon +{forecast_horizon} precipitation (mm)")
        ax.grid(True, alpha=0.25, axis="x")

        cluster_fig, cluster_ax = plt.subplots(figsize=(8, 6))
        cluster_ax.hist(
            values,
            bins=bin_edges,
            orientation="horizontal",
            color="#4C78A8",
            edgecolor="white",
            alpha=0.9,
        )
        cluster_ax.set_title(
            f"Input Windows in Cluster {int(cluster_id)}: "
            f"Horizon +{forecast_horizon} Precipitation"
        )
        cluster_ax.set_xlabel("Samples")
        cluster_ax.set_ylabel(f"Horizon +{forecast_horizon} precipitation (mm)")
        cluster_ax.grid(True, alpha=0.25, axis="x")
        cluster_fig.tight_layout()
        cluster_fig.savefig(
            hist_dir / f"cluster_{int(cluster_id)}_input_precipitation_distribution.png"
        )
        plt.close(cluster_fig)

    fig.suptitle(
        f"Input Window Horizon +{forecast_horizon} Precipitation Distribution by Cluster",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(hist_dir / "08_input_precipitation_distribution_by_cluster.png")
    plt.close(fig)


def _finite_correlation(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _safe_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
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


def _lead_day_prediction_matrix(
    y_pred_test: np.ndarray,
    y_pred_test_by_lead_day: np.ndarray | None,
    n_leads: int,
    n_rows: int | None = None,
) -> tuple[np.ndarray, bool]:
    """Return prediction columns aligned to lead-day targets."""
    if y_pred_test_by_lead_day is not None:
        predictions = np.asarray(y_pred_test_by_lead_day, dtype=float)
        uses_lead_day_outputs = True
    else:
        predictions = np.asarray(y_pred_test, dtype=float)
        uses_lead_day_outputs = (
            predictions.ndim == 2 and predictions.shape[1] == n_leads
        )

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if predictions.ndim != 2:
        raise ValueError("Lead-day predictions must be one- or two-dimensional.")

    if predictions.shape[1] == 1 and n_leads > 1 and not uses_lead_day_outputs:
        predictions = np.repeat(predictions, n_leads, axis=1)
    elif predictions.shape[1] != n_leads:
        raise ValueError(
            "Lead-day prediction columns must match test target lead-day columns."
        )
    if n_rows is not None and len(predictions) != n_rows:
        raise ValueError("Lead-day predictions must match test target row count.")

    return predictions, uses_lead_day_outputs


def save_forecast_horizon_diagnostics(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    current_train: np.ndarray,
    current_val: np.ndarray,
    current_test: np.ndarray,
    y_pred_test: np.ndarray,
    c_test: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    output_dir: Path,
    forecast_horizon: int,
) -> dict[str, float]:
    """Save plots and tables comparing current rain with horizon targets."""
    diag_dir = output_dir / "forecast_horizon_diagnostics"
    diag_dir.mkdir(exist_ok=True)

    split_frames = []
    for split_name, current, target, indices in (
        ("train", current_train, y_train, train_indices),
        ("validation", current_val, y_val, val_indices),
        ("test", current_test, y_test, test_indices),
    ):
        split_frames.append(
            pd.DataFrame(
                {
                    "split": split_name,
                    "window_index": indices,
                    "current_window_precipitation_mm": current,
                    "forecast_horizon_precipitation_mm": target,
                    "horizon_delta_mm": target - current,
                }
            )
        )
    all_df = pd.concat(split_frames, ignore_index=True)
    all_df.to_csv(
        diag_dir / "current_vs_forecast_horizon_precipitation.csv",
        index=False,
    )

    test_df = pd.DataFrame(
        {
            "window_index": test_indices,
            "cluster": c_test.astype(int),
            "current_window_precipitation_mm": current_test,
            "forecast_horizon_precipitation_mm": y_test,
            "lstm_prediction_mm": y_pred_test,
            "horizon_delta_mm": y_test - current_test,
            "persistence_residual_mm": y_test - current_test,
            "lstm_residual_mm": y_test - y_pred_test,
        }
    ).sort_values("window_index")
    test_df.to_csv(diag_dir / "test_forecast_horizon_behavior.csv", index=False)

    persistence_metrics = _safe_regression_metrics(y_test, current_test)
    lstm_metrics = _safe_regression_metrics(y_test, y_pred_test)
    summary = {
        "forecast_horizon": float(forecast_horizon),
        "current_target_correlation_train": _finite_correlation(current_train, y_train),
        "current_target_correlation_val": _finite_correlation(current_val, y_val),
        "current_target_correlation_test": _finite_correlation(current_test, y_test),
        "mean_horizon_delta_test": float(np.nanmean(y_test - current_test)),
        "median_horizon_delta_test": float(np.nanmedian(y_test - current_test)),
        "persistence_rmse_test": persistence_metrics["RMSE"],
        "persistence_mae_test": persistence_metrics["MAE"],
        "lstm_rmse_test": lstm_metrics["RMSE"],
        "lstm_mae_test": lstm_metrics["MAE"],
        "lstm_rmse_improvement_vs_persistence": (
            persistence_metrics["RMSE"] - lstm_metrics["RMSE"]
        ),
        "lstm_mae_improvement_vs_persistence": (
            persistence_metrics["MAE"] - lstm_metrics["MAE"]
        ),
    }

    with open(diag_dir / "forecast_horizon_behavior_report.txt", "w", encoding="utf-8") as f:
        f.write("FORECAST HORIZON BEHAVIOR REPORT\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Forecast horizon: +{forecast_horizon} day(s)\n")
        f.write(
            "Current precipitation is the precipitation observed on the final "
            "day inside each input window.\n"
        )
        f.write(
            "The forecast target is the precipitation observed the configured "
            "number of day(s) after that final input day.\n\n"
        )
        f.write("Current-vs-target correlation\n")
        f.write("-" * 72 + "\n")
        f.write(f"Train:      {summary['current_target_correlation_train']:.4f}\n")
        f.write(f"Validation: {summary['current_target_correlation_val']:.4f}\n")
        f.write(f"Test:       {summary['current_target_correlation_test']:.4f}\n\n")
        f.write("Test-set comparison against current-precipitation persistence\n")
        f.write("-" * 72 + "\n")
        f.write(
            f"Persistence RMSE: {summary['persistence_rmse_test']:.4f}; "
            f"LSTM RMSE: {summary['lstm_rmse_test']:.4f}; "
            f"improvement: {summary['lstm_rmse_improvement_vs_persistence']:.4f}\n"
        )
        f.write(
            f"Persistence MAE:  {summary['persistence_mae_test']:.4f}; "
            f"LSTM MAE:  {summary['lstm_mae_test']:.4f}; "
            f"improvement: {summary['lstm_mae_improvement_vs_persistence']:.4f}\n"
        )
        f.write(
            f"Mean target-current delta: "
            f"{summary['mean_horizon_delta_test']:.4f} mm\n"
        )
        f.write(
            f"Median target-current delta: "
            f"{summary['median_horizon_delta_test']:.4f} mm\n"
        )

    return summary


def save_forecast_lead_day_diagnostics(
    y_pred_test: np.ndarray,
    c_test: np.ndarray,
    test_indices: np.ndarray,
    test_targets_by_lead_day: np.ndarray,
    output_dir: Path,
    forecast_horizon: int,
    y_pred_test_by_lead_day: np.ndarray | None = None,
) -> pd.DataFrame:
    """Save metrics and plots comparing predictions against each lead day."""
    diag_dir = output_dir / "forecast_horizon_diagnostics"
    diag_dir.mkdir(exist_ok=True)

    test_targets_by_lead_day = np.asarray(test_targets_by_lead_day, dtype=float)
    if test_targets_by_lead_day.ndim == 1:
        test_targets_by_lead_day = test_targets_by_lead_day.reshape(-1, 1)
    y_pred_by_lead_day, uses_lead_day_outputs = _lead_day_prediction_matrix(
        y_pred_test,
        y_pred_test_by_lead_day,
        int(test_targets_by_lead_day.shape[1]),
        n_rows=len(test_targets_by_lead_day),
    )

    rows = []
    for lead_offset in range(test_targets_by_lead_day.shape[1]):
        lead_day = lead_offset + 1
        actual = test_targets_by_lead_day[:, lead_offset]
        predicted_values = y_pred_by_lead_day[:, lead_offset]
        for window_index, cluster, predicted, actual_value in zip(
            test_indices,
            c_test,
            predicted_values,
            actual,
        ):
            rows.append(
                {
                    "lead_day": lead_day,
                    "forecast_horizon": forecast_horizon,
                    "window_index": int(window_index),
                    "cluster": int(cluster),
                    "actual_mm": float(actual_value),
                    "predicted_mm": float(predicted),
                    "residual_mm": float(actual_value - predicted),
                    "absolute_error_mm": float(abs(actual_value - predicted)),
                    "squared_error_mm2": float((actual_value - predicted) ** 2),
                    "is_trained_target_day": (
                        uses_lead_day_outputs or lead_day == forecast_horizon
                    ),
                }
            )

    lead_df = pd.DataFrame(rows)
    lead_df.to_csv(diag_dir / "test_prediction_by_lead_day.csv", index=False)

    metric_rows = []
    for lead_day, lead_values in lead_df.groupby("lead_day", sort=True):
        metrics = _safe_regression_metrics(
            lead_values["actual_mm"].to_numpy(dtype=float),
            lead_values["predicted_mm"].to_numpy(dtype=float),
        )
        metric_rows.append(
            {
                "lead_day": int(lead_day),
                "forecast_horizon": forecast_horizon,
                "is_trained_target_day": (
                    uses_lead_day_outputs or int(lead_day) == forecast_horizon
                ),
                "n_test": len(lead_values),
                **metrics,
            }
        )
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(diag_dir / "test_prediction_metrics_by_lead_day.csv", index=False)

    if not metrics_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(
            metrics_df["lead_day"],
            metrics_df["RMSE"],
            marker="o",
            linewidth=2,
            label="RMSE",
            color="#D55E00",
        )
        ax.plot(
            metrics_df["lead_day"],
            metrics_df["MAE"],
            marker="s",
            linewidth=2,
            label="MAE",
            color="#0072B2",
        )
        if uses_lead_day_outputs:
            ax.plot([], [], color="black", label="All lead days trained")
        else:
            ax.axvline(
                forecast_horizon,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label=f"Trained horizon +{forecast_horizon}",
            )
        ax.set_xlabel("Lead day after input window")
        ax.set_ylabel("Error (mm)")
        ax.set_title("Test Error by Forecast Lead Day")
        ax.set_xticks(metrics_df["lead_day"])
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(diag_dir / "12_prediction_error_by_lead_day.png")
        plt.close(fig)

    n_leads = int(test_targets_by_lead_day.shape[1])
    if n_leads:
        n_cols = min(3, n_leads)
        n_rows = int(np.ceil(n_leads / n_cols))
        finite_actual = lead_df["actual_mm"].to_numpy(dtype=float)
        finite_predicted = lead_df["predicted_mm"].to_numpy(dtype=float)
        max_value = float(
            np.nanmax([finite_actual.max(), finite_predicted.max(), 1.0])
        )

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.1 * n_cols, 4.5 * n_rows),
            squeeze=False,
        )
        for ax in axes.ravel()[n_leads:]:
            ax.set_visible(False)
        for ax, lead_day in zip(axes.ravel(), range(1, n_leads + 1)):
            lead_values = lead_df[lead_df["lead_day"] == lead_day]
            ax.scatter(
                lead_values["actual_mm"],
                lead_values["predicted_mm"],
                c=lead_values["cluster"],
                cmap="tab20",
                s=30,
                alpha=0.75,
                edgecolors="black",
                linewidths=0.4,
            )
            ax.plot([0.0, max_value], [0.0, max_value], color="black", linestyle="--")
            metrics = metrics_df.loc[metrics_df["lead_day"] == lead_day].iloc[0]
            title_suffix = "" if uses_lead_day_outputs else (
                " target" if lead_day == forecast_horizon else ""
            )
            ax.set_title(
                f"D+{lead_day}{title_suffix}: "
                f"RMSE={metrics['RMSE']:.3f}, MAE={metrics['MAE']:.3f}"
            )
            ax.set_xlabel("Actual precipitation (mm)")
            ax.set_ylabel("Predicted precipitation (mm)")
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            (
                "True vs Predicted by Lead-Day Output"
                if uses_lead_day_outputs
                else "True vs Predicted Using Each Real Lead Day as Reference"
            ),
            y=1.0,
        )
        fig.tight_layout()
        fig.savefig(diag_dir / "13_true_vs_predicted_by_lead_day.png")
        plt.close(fig)

        by_lead_dir = diag_dir / "true_vs_predicted_by_lead_day"
        by_lead_dir.mkdir(exist_ok=True)
        for lead_day in range(1, n_leads + 1):
            lead_values = lead_df[lead_df["lead_day"] == lead_day].sort_values(
                "window_index"
            )
            metrics = metrics_df.loc[metrics_df["lead_day"] == lead_day].iloc[0]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(
                lead_values["actual_mm"],
                lead_values["predicted_mm"],
                c=lead_values["cluster"],
                cmap="tab20",
                s=38,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
            )
            ax.plot([0.0, max_value], [0.0, max_value], color="black", linestyle="--")
            ax.set_xlabel(f"Actual precipitation at D+{lead_day} (mm)")
            prediction_label = (
                f"LSTM prediction at D+{lead_day} (mm)"
                if uses_lead_day_outputs
                else "LSTM prediction (mm)"
            )
            ax.set_ylabel(prediction_label)
            ax.set_title(
                f"True vs Predicted at D+{lead_day}: "
                f"RMSE={metrics['RMSE']:.3f}, MAE={metrics['MAE']:.3f}"
            )
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                by_lead_dir / f"true_vs_predicted_lead_day_{lead_day:02d}.png"
            )
            plt.close(fig)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.6 * n_cols, 3.8 * n_rows),
            squeeze=False,
        )
        for ax in axes.ravel()[n_leads:]:
            ax.set_visible(False)
        for ax, lead_day in zip(axes.ravel(), range(1, n_leads + 1)):
            lead_values = lead_df[lead_df["lead_day"] == lead_day].sort_values(
                "window_index"
            )
            ax.plot(
                lead_values["window_index"],
                lead_values["actual_mm"],
                label=f"Actual D+{lead_day}",
                color="#009E73",
                linewidth=1.4,
            )
            ax.plot(
                lead_values["window_index"],
                lead_values["predicted_mm"],
                label=(
                    f"Prediction D+{lead_day}"
                    if uses_lead_day_outputs
                    else "Prediction"
                ),
                color="#D55E00",
                linewidth=1.4,
                alpha=0.85,
            )
            ax.set_title(f"Lead day D+{lead_day}")
            ax.set_xlabel("Original window index")
            ax.set_ylabel("Precipitation (mm)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.suptitle("Test Time Series: Prediction Compared With Each Lead Day", y=1.0)
        fig.tight_layout()
        fig.savefig(diag_dir / "14_prediction_vs_actual_timeseries_by_lead_day.png")
        plt.close(fig)

    return metrics_df


def save_cluster_prediction_histograms(
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    c_test: np.ndarray,
    output_dir: Path,
) -> None:
    """Save actual, predicted, and residual histograms for each test cluster."""
    hist_dir = output_dir / "cluster_prediction_histograms"
    hist_dir.mkdir(exist_ok=True)

    for cluster_id in sorted(np.unique(c_test)):
        mask = c_test == cluster_id
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        residuals = y_test[mask] - y_pred_test[mask]

        axes[0].hist(y_test[mask], bins=25, color="#4C78A8", alpha=0.8)
        axes[0].set_title(f"Cluster {cluster_id}: Actual")
        axes[0].set_xlabel("Precipitation (mm)")

        axes[1].hist(y_pred_test[mask], bins=25, color="#F58518", alpha=0.8)
        axes[1].set_title(f"Cluster {cluster_id}: Predicted")
        axes[1].set_xlabel("Precipitation (mm)")

        axes[2].hist(residuals, bins=25, color="#54A24B", alpha=0.8)
        axes[2].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[2].set_title(f"Cluster {cluster_id}: Residual")
        axes[2].set_xlabel("Actual - predicted (mm)")

        for ax in axes:
            ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(hist_dir / f"cluster_{int(cluster_id)}_prediction_histograms.png")
        plt.close(fig)


def _prediction_scatter_limits(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float]:
    """Return padded equal-axis limits for actual-vs-predicted scatter plots."""
    values = np.concatenate([actual, predicted])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0

    min_value = float(values.min())
    max_value = float(values.max())
    if np.isclose(min_value, max_value):
        padding = max(1.0, abs(max_value) * 0.05)
    else:
        padding = (max_value - min_value) * 0.05
    return min_value - padding, max_value + padding


def save_cluster_prediction_scatters(
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    c_test: np.ndarray,
    output_dir: Path,
) -> None:
    """Save test actual-versus-predicted scatter plots for each model cluster."""
    y_test = np.asarray(y_test, dtype=float)
    y_pred_test = np.asarray(y_pred_test, dtype=float)
    c_test = np.asarray(c_test)
    if len({len(y_test), len(y_pred_test), len(c_test)}) != 1:
        raise ValueError("Test actual, predicted, and cluster labels must align.")

    plot_dir = output_dir / "cluster_prediction_scatter"
    plot_dir.mkdir(exist_ok=True)
    if c_test.size == 0:
        return

    for cluster_id in sorted(np.unique(c_test)):
        mask = c_test == cluster_id
        cluster_actual = y_test[mask]
        cluster_predicted = y_pred_test[mask]
        finite_mask = np.isfinite(cluster_actual) & np.isfinite(cluster_predicted)
        if not np.any(finite_mask):
            continue
        cluster_actual = cluster_actual[finite_mask]
        cluster_predicted = cluster_predicted[finite_mask]

        min_value, max_value = _prediction_scatter_limits(
            cluster_actual,
            cluster_predicted,
        )
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        ax.scatter(
            cluster_actual,
            cluster_predicted,
            s=58,
            marker="x",
            color="#D62728",
            alpha=0.85,
            linewidths=1.4,
            label=f"Test (n={len(cluster_actual)})",
        )
        ax.plot(
            [min_value, max_value],
            [min_value, max_value],
            color="#666666",
            linestyle="--",
            linewidth=1.2,
            label="Perfect prediction",
        )
        ax.set_xlim(min_value, max_value)
        ax.set_ylim(min_value, max_value)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Cluster {int(cluster_id)}: Test Actual vs Predicted")
        ax.set_xlabel("Actual precipitation (mm)")
        ax.set_ylabel("Predicted precipitation (mm)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            plot_dir / f"cluster_{int(cluster_id)}_predicted_vs_actual_scatter.png"
        )
        plt.close(fig)


def compressed_time_positions(
    original_indices: np.ndarray,
    max_gap: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return chronological plot positions with only large gaps shortened."""
    indices = np.asarray(original_indices, dtype=int)
    if indices.ndim != 1:
        raise ValueError("original_indices must be one-dimensional.")
    if max_gap <= 0:
        raise ValueError("max_gap must be positive.")
    if len(indices) == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)
    if np.any(np.diff(indices) <= 0):
        raise ValueError("original_indices must be strictly increasing.")

    original_gaps = np.diff(indices)
    compressed_intervals = original_gaps > max_gap
    plotted_gaps = np.minimum(original_gaps, max_gap)
    positions = np.concatenate(([0.0], np.cumsum(plotted_gaps, dtype=float)))
    return positions, compressed_intervals


def save_cluster_prediction_timeseries(
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    c_test: np.ndarray,
    test_indices: np.ndarray,
    output_dir: Path,
    test_dates: np.ndarray | None = None,
    max_gap: int = 10,
) -> None:
    """Save actual, predicted, and residual time series for each test cluster."""
    y_test = np.asarray(y_test, dtype=float)
    y_pred_test = np.asarray(y_pred_test, dtype=float)
    c_test = np.asarray(c_test)
    test_indices = np.asarray(test_indices, dtype=int)
    lengths = {len(y_test), len(y_pred_test), len(c_test), len(test_indices)}
    if len(lengths) != 1:
        raise ValueError("Test values, labels, predictions, and indices must align.")
    date_labels = _prediction_timeseries_date_labels(
        test_dates,
        n_rows=len(y_test),
        n_leads=1,
    )
    if date_labels is not None:
        date_labels = date_labels.reshape(-1)

    plot_dir = output_dir / "cluster_prediction_timeseries"
    plot_dir.mkdir(exist_ok=True)

    for cluster_id in sorted(np.unique(c_test)):
        cluster_offsets = np.flatnonzero(c_test == cluster_id)
        chronological_order = np.argsort(test_indices[cluster_offsets], kind="stable")
        cluster_offsets = cluster_offsets[chronological_order]
        original_indices = test_indices[cluster_offsets]
        actual = y_test[cluster_offsets]
        predicted = y_pred_test[cluster_offsets]
        residuals = actual - predicted
        if date_labels is not None:
            x_values = date_labels[cluster_offsets]
            compressed_intervals = np.array([], dtype=bool)
        else:
            x_values, compressed_intervals = compressed_time_positions(
                original_indices,
                max_gap=max_gap,
            )
        metrics = calculate_regression_metrics(actual, predicted)

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(14, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0]},
        )
        axes[0].plot(
            x_values,
            actual,
            label="Actual",
            color="#4C78A8",
            linewidth=1.5,
            alpha=0.85,
        )
        axes[0].scatter(
            x_values,
            actual,
            color="#4C78A8",
            s=10,
            alpha=0.65,
            zorder=3,
        )
        axes[0].plot(
            x_values,
            predicted,
            label="Predicted",
            color="#F58518",
            linewidth=1.5,
            alpha=0.85,
        )
        axes[0].set_ylabel("Precipitation (mm)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(
            f"Cluster {int(cluster_id)} Test Performance | "
            f"n={len(actual)} | RMSE={metrics['RMSE']:.3f} | "
            f"MAE={metrics['MAE']:.3f} | R2={metrics['R2']:.3f}"
        )

        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
        axes[1].plot(
            x_values,
            residuals,
            color="#54A24B",
            linewidth=1.2,
            alpha=0.85,
        )
        axes[1].fill_between(
            x_values,
            residuals,
            0.0,
            color="#54A24B",
            alpha=0.18,
        )
        axes[1].set_ylabel("Residual (mm)")
        axes[1].grid(True, alpha=0.3)

        if date_labels is not None:
            axes[1].set_xlabel("Target Date")
            axes[1].xaxis.set_major_locator(mdates.AutoDateLocator())
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
            fig.autofmt_xdate(rotation=30, ha="right")
        else:
            positions = np.asarray(x_values, dtype=float)
            for interval_offset in np.flatnonzero(compressed_intervals):
                gap_marker = (
                    positions[interval_offset] + positions[interval_offset + 1]
                ) / 2
                for ax in axes:
                    ax.axvline(
                        gap_marker,
                        color="#888888",
                        linestyle=":",
                        linewidth=0.8,
                        alpha=0.55,
                    )

            tick_count = min(9, len(positions))
            tick_offsets = np.unique(
                np.linspace(0, len(positions) - 1, tick_count, dtype=int)
            )
            axes[1].set_xticks(positions[tick_offsets])
            axes[1].set_xticklabels(original_indices[tick_offsets])
            axes[1].set_xlabel(
                "Compressed test timeline (labels show original window index)"
            )
            compressed_count = int(compressed_intervals.sum())
            if compressed_count:
                fig.text(
                    0.5,
                    0.01,
                    f"{compressed_count} gaps larger than {max_gap} windows were "
                    f"displayed as {max_gap} windows.",
                    ha="center",
                    fontsize=9,
                )
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(
            plot_dir / f"cluster_{int(cluster_id)}_prediction_timeseries.png"
        )
        plt.close(fig)


def save_precipitation_by_cluster_plot(
    y_test: np.ndarray,
    c_test: np.ndarray,
    output_dir: Path,
) -> None:
    """Save a boxplot showing the target distribution by cluster."""
    plot_dir = output_dir / "cluster_diagnostics"
    plot_dir.mkdir(exist_ok=True)

    plot_df = pd.DataFrame({"cluster": c_test, "precipitation_mm": y_test})
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=plot_df, x="cluster", y="precipitation_mm", ax=ax)
    ax.set_title("Test Set: Precipitation Distribution by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Forecast-target precipitation (mm)")
    fig.tight_layout()
    fig.savefig(plot_dir / "07_precipitation_distribution_by_cluster.png")
    plt.close(fig)


def cluster_batch_statistics(
    c_train: np.ndarray,
    c_val: np.ndarray,
    c_test: np.ndarray,
    batch_size: int,
) -> pd.DataFrame:
    """Return split counts and optimizer steps per epoch for each cluster."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    labels_by_split = {
        "n_train": np.asarray(c_train).reshape(-1),
        "n_validation": np.asarray(c_val).reshape(-1),
        "n_test": np.asarray(c_test).reshape(-1),
    }
    nonempty_labels = [labels for labels in labels_by_split.values() if labels.size]
    if not nonempty_labels:
        return pd.DataFrame(
            columns=[
                "cluster",
                "n_train",
                "n_validation",
                "n_test",
                "batch_size",
                "optimizer_steps_per_epoch",
            ]
        )

    cluster_ids = np.unique(np.concatenate(nonempty_labels))
    rows = []
    for cluster_id in cluster_ids:
        counts = {
            name: int(np.count_nonzero(labels == cluster_id))
            for name, labels in labels_by_split.items()
        }
        rows.append(
            {
                "cluster": int(cluster_id),
                **counts,
                "batch_size": int(batch_size),
                "optimizer_steps_per_epoch": int(
                    np.ceil(counts["n_train"] / batch_size)
                ),
            }
        )
    return pd.DataFrame(rows)


def save_cluster_distribution_plot(
    c_test: np.ndarray,
    output_dir: Path,
    *,
    c_train: np.ndarray | None = None,
    c_val: np.ndarray | None = None,
    batch_size: int | None = None,
) -> pd.DataFrame | None:
    """Save cluster counts and, when available, per-cluster training workload."""
    plot_dir = output_dir / "cluster_diagnostics"
    plot_dir.mkdir(exist_ok=True)

    if c_train is not None and batch_size is not None:
        statistics = cluster_batch_statistics(
            c_train,
            np.asarray([]) if c_val is None else c_val,
            c_test,
            batch_size,
        )
        statistics.to_csv(
            plot_dir / "cluster_training_batch_statistics.csv",
            index=False,
        )

        figure_height = max(5.0, 2.4 + 0.32 * len(statistics))
        fig, (ax, table_ax) = plt.subplots(
            1,
            2,
            figsize=(16, figure_height),
            gridspec_kw={"width_ratios": [1.55, 1.0]},
        )
        cluster_positions = np.arange(len(statistics), dtype=float)
        bar_width = 0.25
        split_columns = (
            ("n_train", "Training", "#4C78A8"),
            ("n_validation", "Validation", "#F58518"),
            ("n_test", "Test", "#54A24B"),
        )
        for offset, (column, label, color) in zip(
            (-bar_width, 0.0, bar_width),
            split_columns,
        ):
            bars = ax.bar(
                cluster_positions + offset,
                statistics[column],
                width=bar_width,
                label=label,
                color=color,
                alpha=0.85,
            )
            ax.bar_label(bars, padding=2, fontsize=8)

        ax.set_title("Cluster Distribution by Split")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Samples")
        ax.set_xticks(cluster_positions)
        ax.set_xticklabels(statistics["cluster"].astype(str))
        ax.legend()
        ax.grid(True, axis="y", alpha=0.25)

        table_ax.axis("off")
        table_ax.set_title(
            f"Training Workload (batch_size={batch_size})",
            pad=12,
        )
        table = table_ax.table(
            cellText=[
                [
                    int(row.cluster),
                    int(row.n_train),
                    int(row.optimizer_steps_per_epoch),
                ]
                for row in statistics.itertuples(index=False)
            ],
            colLabels=["Cluster", "n_train", "ceil(n_train / batch)"],
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "06_cluster_distribution.png")
        plt.close(fig)
        return statistics

    unique_clusters, counts = np.unique(c_test, return_counts=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(unique_clusters, counts, color="#4C78A8", alpha=0.8)
    ax.set_title("Test Set: Cluster Distribution")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Samples")
    ax.bar_label(bars)
    fig.tight_layout()
    fig.savefig(plot_dir / "06_cluster_distribution.png")
    plt.close(fig)
    return None


def _prepare_silhouette_inputs(
    feature_matrix: np.ndarray,
    cluster_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Return finite silhouette inputs or a reason why they are invalid."""
    features = np.asarray(feature_matrix, dtype=float)
    labels = np.asarray(cluster_labels)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.ndim != 2:
        return features, labels, "feature matrix must be two-dimensional"
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if len(features) != len(labels):
        return features, labels, "feature and label counts do not match"

    finite_labels = np.isfinite(labels.astype(float, copy=False))
    finite_rows = np.all(np.isfinite(features), axis=1) & finite_labels
    features = features[finite_rows]
    labels = labels[finite_rows]

    unique_labels = np.unique(labels)
    if len(features) < 2:
        return features, labels, "at least two samples are required"
    if len(unique_labels) < 2:
        return features, labels, "at least two clusters are required"
    if len(unique_labels) >= len(features):
        return features, labels, "clusters must be fewer than samples"
    return features, labels, None


def _silhouette_unavailable_row(
    split_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    reason: str,
) -> dict[str, object]:
    return {
        "split": split_name,
        "cluster": "overall",
        "n_samples": int(len(features)),
        "n_clusters": int(len(np.unique(labels))) if labels.size else 0,
        "mean_silhouette": np.nan,
        "min_silhouette": np.nan,
        "max_silhouette": np.nan,
        "status": reason,
    }


def _draw_silhouette_axis(
    ax: plt.Axes,
    feature_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    split_name: str,
) -> list[dict[str, object]]:
    """Draw one silhouette plot panel and return its summary rows."""
    features, labels, reason = _prepare_silhouette_inputs(
        feature_matrix,
        cluster_labels,
    )
    if reason is not None:
        ax.text(
            0.5,
            0.5,
            f"Silhouette unavailable:\n{reason}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(split_name)
        ax.set_xticks([])
        ax.set_yticks([])
        return [_silhouette_unavailable_row(split_name, features, labels, reason)]

    values = silhouette_samples(features, labels)
    mean_value = float(silhouette_score(features, labels))
    unique_labels = sorted(np.unique(labels))
    colors = sns.color_palette("tab10", n_colors=max(len(unique_labels), 1))
    x_min = max(-1.0, min(-0.1, float(values.min()) - 0.05))
    y_lower = 10
    rows: list[dict[str, object]] = [
        {
            "split": split_name,
            "cluster": "overall",
            "n_samples": int(len(features)),
            "n_clusters": int(len(unique_labels)),
            "mean_silhouette": mean_value,
            "min_silhouette": float(values.min()),
            "max_silhouette": float(values.max()),
            "status": "ok",
        }
    ]

    for cluster_index, cluster_id in enumerate(unique_labels):
        cluster_values = np.sort(values[labels == cluster_id])
        y_upper = y_lower + len(cluster_values)
        color = colors[cluster_index % len(colors)]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.78,
        )
        ax.text(
            x_min + 0.02,
            y_lower + 0.5 * len(cluster_values),
            str(int(cluster_id)),
            va="center",
            fontsize=9,
        )
        rows.append(
            {
                "split": split_name,
                "cluster": int(cluster_id),
                "n_samples": int(len(cluster_values)),
                "n_clusters": int(len(unique_labels)),
                "mean_silhouette": float(cluster_values.mean()),
                "min_silhouette": float(cluster_values.min()),
                "max_silhouette": float(cluster_values.max()),
                "status": "ok",
            }
        )
        y_lower = y_upper + 10

    ax.axvline(
        mean_value,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean silhouette = {mean_value:.3f}",
    )
    ax.set_xlim([x_min, 1.0])
    ax.set_ylim([0, y_lower])
    ax.set_title(f"{split_name} | mean={mean_value:.3f}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.set_yticks([])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25, axis="x")
    return rows


def save_cluster_silhouette_plot(
    cluster_feature_splits: Mapping[str, tuple[np.ndarray, np.ndarray]] | None,
    output_dir: Path,
) -> pd.DataFrame:
    """Save train/validation/test silhouette diagnostics for cluster features."""
    plot_dir = output_dir / "cluster_diagnostics"
    plot_dir.mkdir(exist_ok=True)
    if not cluster_feature_splits:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "Silhouette unavailable:\nno feature splits were provided",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Cluster Silhouette Analysis")
        fig.tight_layout()
        fig.savefig(plot_dir / "08_silhouette_analysis.png")
        plt.close(fig)
        summary = pd.DataFrame(
            [
                {
                    "split": "All",
                    "cluster": "overall",
                    "n_samples": 0,
                    "n_clusters": 0,
                    "mean_silhouette": np.nan,
                    "min_silhouette": np.nan,
                    "max_silhouette": np.nan,
                    "status": "no feature splits were provided",
                }
            ]
        )
        summary.to_csv(plot_dir / "silhouette_scores.csv", index=False)
        return summary

    split_items = list(cluster_feature_splits.items())
    fig, axes = plt.subplots(
        1,
        len(split_items),
        figsize=(5.6 * len(split_items), 6.2),
        squeeze=False,
    )
    rows: list[dict[str, object]] = []
    for ax, (split_name, (feature_matrix, labels)) in zip(axes.ravel(), split_items):
        rows.extend(
            _draw_silhouette_axis(
                ax,
                feature_matrix,
                labels,
                split_name,
            )
        )

    fig.suptitle("Cluster Silhouette Analysis", y=1.02)
    fig.tight_layout()
    fig.savefig(plot_dir / "08_silhouette_analysis.png")
    plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(plot_dir / "silhouette_scores.csv", index=False)
    return summary


def save_prediction_timeseries_splits(
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    output_dir: Path,
    n_splits: int = 4,
    forecast_horizon: int | None = None,
    test_dates_by_lead_day: np.ndarray | None = None,
) -> None:
    """Save prediction time-series splits for each forecast lead day."""
    plot_dir = output_dir / "prediction_timeseries_splits"
    plot_dir.mkdir(exist_ok=True)

    actual_by_lead_day = np.asarray(y_test, dtype=float)
    predicted_by_lead_day = np.asarray(y_pred_test, dtype=float)
    if actual_by_lead_day.ndim == 1:
        actual_by_lead_day = actual_by_lead_day.reshape(-1, 1)
    if predicted_by_lead_day.ndim == 1:
        predicted_by_lead_day = predicted_by_lead_day.reshape(-1, 1)
    if actual_by_lead_day.ndim != 2 or predicted_by_lead_day.ndim != 2:
        raise ValueError(
            "Prediction time-series inputs must be one- or two-dimensional."
        )
    if actual_by_lead_day.shape != predicted_by_lead_day.shape:
        raise ValueError("Prediction time-series actual and predicted matrices must match.")

    n_leads = int(actual_by_lead_day.shape[1])
    if forecast_horizon is None:
        forecast_horizon = n_leads
    if int(forecast_horizon) != n_leads:
        raise ValueError("forecast_horizon must match the number of lead-day columns.")

    date_labels_by_lead_day = _prediction_timeseries_date_labels(
        test_dates_by_lead_day,
        n_rows=len(actual_by_lead_day),
        n_leads=n_leads,
    )
    indices = np.arange(len(actual_by_lead_day))
    for lead_offset in range(n_leads):
        lead_day = lead_offset + 1
        lead_dir = plot_dir / f"lead_day_{lead_day:02d}"
        lead_dir.mkdir(exist_ok=True)

        for split_index, split_indices in enumerate(
            np.array_split(indices, n_splits),
            start=1,
        ):
            fig, ax = plt.subplots(figsize=(14, 5.6))
            if split_indices.size > 0:
                x_values = (
                    date_labels_by_lead_day[split_indices, lead_offset]
                    if date_labels_by_lead_day is not None
                    else split_indices
                )
                ax.plot(
                    x_values,
                    actual_by_lead_day[split_indices, lead_offset],
                    label=f"Actual D+{lead_day}",
                    alpha=0.8,
                    linewidth=1.6,
                )
                ax.plot(
                    x_values,
                    predicted_by_lead_day[split_indices, lead_offset],
                    label=f"Predicted D+{lead_day}",
                    alpha=0.8,
                    linewidth=1.6,
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No samples in this split",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.set_title(
                f"D+{lead_day}: Predictions vs Actual - "
                f"Split {split_index} of {n_splits}"
            )
            if date_labels_by_lead_day is not None:
                ax.set_xlabel("Target Date")
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
                fig.autofmt_xdate(rotation=30, ha="right")
            else:
                ax.set_xlabel("Test Sample Index")
            ax.set_ylabel("Precipitation (mm)")
            if split_indices.size > 0:
                ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                lead_dir
                / f"02_predictions_timeseries_split_{split_index:02d}_of_{n_splits:02d}.png"
            )
            plt.close(fig)


def _prediction_timeseries_date_labels(
    test_dates_by_lead_day: np.ndarray | None,
    n_rows: int,
    n_leads: int,
) -> np.ndarray | None:
    """Return validated date labels for split prediction time-series plots."""
    if test_dates_by_lead_day is None:
        return None

    date_labels = np.asarray(test_dates_by_lead_day)
    if date_labels.ndim == 1:
        date_labels = date_labels.reshape(-1, 1)
    if date_labels.ndim != 2:
        raise ValueError(
            "Prediction time-series date labels must be one- or two-dimensional."
        )
    if date_labels.shape[0] != n_rows:
        raise ValueError("Prediction time-series date labels must match row count.")
    if date_labels.shape[1] == 1 and n_leads > 1:
        date_labels = np.repeat(date_labels, n_leads, axis=1)
    elif date_labels.shape[1] != n_leads:
        raise ValueError("Prediction time-series date columns must match lead days.")

    parsed_dates = pd.to_datetime(date_labels.reshape(-1), errors="coerce")
    if pd.isna(parsed_dates).any():
        raise ValueError("Prediction time-series date labels must contain valid dates.")
    return parsed_dates.to_numpy(dtype="datetime64[ns]").reshape(date_labels.shape)


def save_visualizations(
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    c_test: np.ndarray,
    test_indices: np.ndarray,
    forecast_horizon_precipitation: np.ndarray,
    input_cluster_labels: np.ndarray,
    histories_by_cluster: dict[int, object],
    output_dir: Path,
    forecast_horizon: int,
    batch_size: int | None = None,
    test_targets_by_lead_day: np.ndarray | None = None,
    y_pred_test_by_lead_day: np.ndarray | None = None,
    test_target_dates_by_lead_day: np.ndarray | None = None,
    cluster_feature_splits: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Save the diagnostic plots for one configuration."""
    prediction_dir = output_dir / "prediction_overview_same_cluster"
    residual_dir = output_dir / "residual_diagnostics"
    cluster_dir = output_dir / "cluster_diagnostics"
    prediction_dir.mkdir(exist_ok=True)
    residual_dir.mkdir(exist_ok=True)
    cluster_dir.mkdir(exist_ok=True)

    save_training_history_plots(histories_by_cluster, output_dir)

    fig, _ = plot_predictions_vs_actual(
        y_test,
        y_pred_test,
        cluster_labels=c_test,
        title="Test Set: Predictions vs Actual Precipitation",
    )
    fig.savefig(prediction_dir / "02_predictions_vs_actual.png")
    plt.close(fig)
    save_prediction_timeseries_splits(
        test_targets_by_lead_day if test_targets_by_lead_day is not None else y_test,
        y_pred_test_by_lead_day if y_pred_test_by_lead_day is not None else y_pred_test,
        output_dir,
        forecast_horizon=(
            forecast_horizon
            if test_targets_by_lead_day is not None
            or y_pred_test_by_lead_day is not None
            else None
        ),
        test_dates_by_lead_day=test_target_dates_by_lead_day,
    )
    cluster_timeseries_dates = None
    if test_target_dates_by_lead_day is not None:
        date_labels_by_lead_day = _prediction_timeseries_date_labels(
            test_target_dates_by_lead_day,
            n_rows=len(y_test),
            n_leads=int(forecast_horizon),
        )
        cluster_timeseries_dates = date_labels_by_lead_day[:, -1]
    save_cluster_prediction_timeseries(
        y_test,
        y_pred_test,
        c_test,
        test_indices,
        output_dir,
        test_dates=cluster_timeseries_dates,
    )

    fig, _ = plot_residuals(
        y_test,
        y_pred_test,
        cluster_labels=c_test,
        title="Test Set: Residual Analysis",
    )
    fig.savefig(residual_dir / "03_residuals_analysis.png")
    plt.close(fig)

    fig, _ = plot_error_by_magnitude(
        y_test,
        y_pred_test,
        n_bins=10,
        title="Test Set: Error by Precipitation Magnitude",
    )
    fig.savefig(residual_dir / "04_error_by_magnitude.png")
    plt.close(fig)

    fig, _ = plot_cluster_performance(
        c_test,
        y_test,
        y_pred_test,
        title="Test Set: LSTM Performance by Cluster",
    )
    fig.savefig(cluster_dir / "05_cluster_performance.png")
    plt.close(fig)

    training_cluster_labels = None
    validation_cluster_labels = None
    if cluster_feature_splits is not None:
        training_split = cluster_feature_splits.get("Training")
        validation_split = cluster_feature_splits.get("Validation")
        if training_split is not None:
            training_cluster_labels = training_split[1]
        if validation_split is not None:
            validation_cluster_labels = validation_split[1]
    save_cluster_distribution_plot(
        c_test,
        output_dir,
        c_train=training_cluster_labels,
        c_val=validation_cluster_labels,
        batch_size=batch_size,
    )
    save_precipitation_by_cluster_plot(y_test, c_test, output_dir)
    save_cluster_silhouette_plot(cluster_feature_splits, output_dir)
    save_cluster_precipitation_histograms(y_test, c_test, output_dir)
    save_input_precipitation_distribution_by_cluster(
        forecast_horizon_precipitation,
        input_cluster_labels,
        output_dir,
        forecast_horizon=forecast_horizon,
    )
    save_cluster_prediction_histograms(y_test, y_pred_test, c_test, output_dir)
    save_cluster_prediction_scatters(
        y_test,
        y_pred_test,
        c_test,
        output_dir,
    )


def save_oracle_model_visualizations(
    test_model_selection: dict[str, object],
    y_test: np.ndarray,
    c_test: np.ndarray,
    test_indices: np.ndarray,
    forecast_horizon_precipitation: np.ndarray,
    input_cluster_labels: np.ndarray,
    histories_by_cluster: dict[int, object],
    output_dir: Path,
    *,
    forecast_horizon: int,
    batch_size: int | None,
    test_targets_by_lead_day: np.ndarray,
    regular_prediction_by_lead_day: np.ndarray,
    test_target_dates_by_lead_day: np.ndarray | None,
    cluster_feature_splits: Mapping[str, tuple[np.ndarray, np.ndarray]] | None,
) -> bool:
    """Mirror normal plots using only the post-hoc oracle prediction."""
    selection_summary = dict(test_model_selection.get("summary", {}))
    primary_metric = str(selection_summary.get("primary_metric", "RMSE"))
    oracle_predictions = np.asarray(
        dict(test_model_selection.get("selected_prediction_by_metric", {})).get(
            primary_metric,
            [],
        ),
        dtype=float,
    )
    oracle_predictions_by_lead_day = np.asarray(
        test_model_selection.get("selected_prediction_by_lead_day", []),
        dtype=float,
    )
    if (
        len(oracle_predictions) != len(y_test)
        or oracle_predictions_by_lead_day.shape != regular_prediction_by_lead_day.shape
    ):
        return False

    oracle_model_dir = output_dir / "oracle_model"
    oracle_model_dir.mkdir(exist_ok=True)
    save_visualizations(
        y_test,
        oracle_predictions,
        c_test,
        test_indices,
        forecast_horizon_precipitation,
        input_cluster_labels,
        histories_by_cluster,
        oracle_model_dir,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        test_targets_by_lead_day=test_targets_by_lead_day,
        y_pred_test_by_lead_day=oracle_predictions_by_lead_day,
        test_target_dates_by_lead_day=test_target_dates_by_lead_day,
        cluster_feature_splits=cluster_feature_splits,
    )
    return True


def metrics_dataframe(
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> pd.DataFrame:
    """Return split-level metrics as a tidy dataframe."""
    rows = []
    for split, metrics in [
        ("Train", train_metrics),
        ("Validation", val_metrics),
        ("Test", test_metrics),
    ]:
        row = {"split": split}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_metrics_dataframe(metrics_by_cluster: dict[int, dict[str, float]]) -> pd.DataFrame:
    """Return test metrics for each cluster-specific model."""
    rows = []
    for cluster_id, metrics in sorted(metrics_by_cluster.items()):
        row = {"cluster": cluster_id}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def oracle_transfer_matrix(
    selection_df: pd.DataFrame,
    primary_metric: str,
) -> pd.DataFrame:
    """Count oracle-selected LSTMs for windows assigned to each test cluster."""
    required_columns = {
        "test_cluster",
        "metric",
        "selected_model_cluster",
    }
    if selection_df.empty or not required_columns.issubset(selection_df.columns):
        return pd.DataFrame()

    primary_rows = selection_df.loc[
        selection_df["metric"].astype(str).str.upper()
        == str(primary_metric).upper()
    ]
    if primary_rows.empty:
        return pd.DataFrame()

    matrix = pd.crosstab(
        primary_rows["test_cluster"].astype(int),
        primary_rows["selected_model_cluster"].astype(int),
    ).sort_index(axis=0).sort_index(axis=1)
    matrix.index.name = "assigned_test_cluster"
    matrix.columns = [f"LSTM_{int(cluster_id)}" for cluster_id in matrix.columns]
    return matrix


def oracle_routing_diagnostic_tables(
    selection_df: pd.DataFrame,
    primary_metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize oracle routing gains by assigned cluster and transfer pair."""
    required_columns = {
        "test_cluster",
        "metric",
        "selected_model_cluster",
        "selected_is_same_cluster",
        "actual",
        "selected_absolute_error",
        "same_cluster_absolute_error",
    }
    if selection_df.empty or not required_columns.issubset(selection_df.columns):
        return pd.DataFrame(), pd.DataFrame()

    primary_rows = selection_df.loc[
        selection_df["metric"].astype(str).str.upper()
        == str(primary_metric).upper()
    ].copy()
    if primary_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    primary_rows["test_cluster"] = primary_rows["test_cluster"].astype(int)
    primary_rows["selected_model_cluster"] = primary_rows[
        "selected_model_cluster"
    ].astype(int)
    primary_rows["switched_model"] = ~primary_rows[
        "selected_is_same_cluster"
    ].astype(bool)
    primary_rows["absolute_error_improvement"] = (
        primary_rows["same_cluster_absolute_error"].astype(float)
        - primary_rows["selected_absolute_error"].astype(float)
    )
    primary_rows["same_cluster_squared_error"] = (
        primary_rows["same_cluster_absolute_error"].astype(float) ** 2
    )
    primary_rows["selected_squared_error"] = (
        primary_rows["selected_absolute_error"].astype(float) ** 2
    )
    primary_rows["squared_error_improvement"] = (
        primary_rows["same_cluster_squared_error"]
        - primary_rows["selected_squared_error"]
    )

    cluster_rows: list[dict[str, float | int]] = []
    for assigned_cluster, values in primary_rows.groupby("test_cluster", sort=True):
        n_samples = int(len(values))
        switched = int(values["switched_model"].sum())
        same_rmse = float(np.sqrt(values["same_cluster_squared_error"].mean()))
        oracle_rmse = float(np.sqrt(values["selected_squared_error"].mean()))
        cluster_rows.append(
            {
                "assigned_test_cluster": int(assigned_cluster),
                "n_test": n_samples,
                "oracle_switched_samples": switched,
                "oracle_switched_percent": (
                    switched / n_samples * 100.0 if n_samples else np.nan
                ),
                "same_cluster_rmse": same_rmse,
                "oracle_rmse": oracle_rmse,
                "rmse_improvement": same_rmse - oracle_rmse,
                "same_cluster_mae": float(
                    values["same_cluster_absolute_error"].mean()
                ),
                "oracle_mae": float(values["selected_absolute_error"].mean()),
                "mae_improvement": float(
                    values["absolute_error_improvement"].mean()
                ),
                "mean_squared_error_improvement": float(
                    values["squared_error_improvement"].mean()
                ),
                "median_absolute_error_improvement": float(
                    values["absolute_error_improvement"].median()
                ),
                "mean_actual_precipitation_mm": float(values["actual"].mean()),
            }
        )

    pair_rows: list[dict[str, float | int]] = []
    assigned_sizes = primary_rows.groupby("test_cluster").size()
    for (assigned_cluster, oracle_cluster), values in primary_rows.groupby(
        ["test_cluster", "selected_model_cluster"],
        sort=True,
    ):
        n_samples = int(len(values))
        assigned_total = int(assigned_sizes.loc[assigned_cluster])
        pair_rows.append(
            {
                "assigned_test_cluster": int(assigned_cluster),
                "oracle_selected_model_cluster": int(oracle_cluster),
                "n_test": n_samples,
                "percent_of_assigned_cluster": (
                    n_samples / assigned_total * 100.0
                    if assigned_total
                    else np.nan
                ),
                "mean_actual_precipitation_mm": float(values["actual"].mean()),
                "same_cluster_mae": float(
                    values["same_cluster_absolute_error"].mean()
                ),
                "oracle_mae": float(values["selected_absolute_error"].mean()),
                "mae_improvement": float(
                    values["absolute_error_improvement"].mean()
                ),
                "mean_squared_error_improvement": float(
                    values["squared_error_improvement"].mean()
                ),
            }
        )

    return pd.DataFrame(cluster_rows), pd.DataFrame(pair_rows)


def save_oracle_transfer_diagnostics(
    y_test: np.ndarray,
    test_model_selection: dict[str, object],
    output_dir: Path,
) -> None:
    """Save clearly labelled oracle-only plots for cross-cluster transfer."""
    summary = dict(test_model_selection.get("summary", {}))
    primary_metric = str(summary.get("primary_metric", "RMSE"))
    selected_predictions = np.asarray(
        dict(test_model_selection.get("selected_prediction_by_metric", {})).get(
            primary_metric,
            [],
        ),
        dtype=float,
    )
    selected_models = np.asarray(
        dict(test_model_selection.get("selected_model_by_metric", {})).get(
            primary_metric,
            [],
        ),
        dtype=float,
    )
    if len(selected_predictions) != len(y_test) or len(selected_models) != len(y_test):
        return

    diagnostics_dir = output_dir / "oracle_model_selection_diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)
    fig, _ = plot_predictions_vs_actual(
        y_test,
        selected_predictions,
        cluster_labels=selected_models.astype(int),
        title=(
            "Oracle Diagnostic: Predictions by Post-hoc Selected Cluster LSTM"
        ),
    )
    fig.savefig(diagnostics_dir / "01_oracle_predictions_vs_actual.png")
    plt.close(fig)

    selection_df = pd.DataFrame(test_model_selection.get("selection_rows", []))
    transfer_matrix = oracle_transfer_matrix(selection_df, primary_metric)
    cluster_summary, pair_summary = oracle_routing_diagnostic_tables(
        selection_df,
        primary_metric,
    )
    if not cluster_summary.empty:
        cluster_summary.to_csv(
            output_dir / "oracle_cluster_routing_summary.csv",
            index=False,
        )
    if not pair_summary.empty:
        pair_summary.to_csv(
            output_dir / "oracle_cluster_pair_summary.csv",
            index=False,
        )
    if transfer_matrix.empty:
        return

    figure_height = max(4.0, 2.5 + 0.55 * len(transfer_matrix.index))
    figure_width = max(6.0, 3.5 + 0.95 * len(transfer_matrix.columns))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    sns.heatmap(
        transfer_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Test windows"},
        ax=ax,
    )
    ax.set_title(
        "Oracle Cross-cluster Transfer Matrix\n"
        "Rows: assigned cluster; columns: LSTM selected after observing test target"
    )
    ax.set_xlabel("Oracle-selected model")
    ax.set_ylabel("Assigned test cluster")
    fig.tight_layout()
    fig.savefig(diagnostics_dir / "02_oracle_model_transfer_matrix.png")
    plt.close(fig)

    if cluster_summary.empty:
        return

    cluster_plot_df = cluster_summary.sort_values("assigned_test_cluster").copy()
    cluster_plot_df["assigned_test_cluster"] = cluster_plot_df[
        "assigned_test_cluster"
    ].astype(str)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=cluster_plot_df,
        x="assigned_test_cluster",
        y="oracle_switched_percent",
        color="#4C78A8",
        ax=ax,
    )
    ax.set_title("Oracle Model Switch Rate by Assigned Test Cluster")
    ax.set_xlabel("Assigned test cluster")
    ax.set_ylabel("Windows where oracle selected another LSTM (%)")
    ax.set_ylim(0, max(100.0, float(cluster_plot_df["oracle_switched_percent"].max()) * 1.1))
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(diagnostics_dir / "03_oracle_switch_rate_by_assigned_cluster.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    melted_errors = cluster_plot_df.melt(
        id_vars="assigned_test_cluster",
        value_vars=["same_cluster_mae", "oracle_mae"],
        var_name="routing",
        value_name="MAE",
    )
    melted_errors["routing"] = melted_errors["routing"].map(
        {
            "same_cluster_mae": "Assigned-cluster LSTM",
            "oracle_mae": "Oracle-selected LSTM",
        }
    )
    sns.barplot(
        data=melted_errors,
        x="assigned_test_cluster",
        y="MAE",
        hue="routing",
        ax=ax,
    )
    ax.set_title("Same-cluster vs Oracle Error by Assigned Test Cluster")
    ax.set_xlabel("Assigned test cluster")
    ax.set_ylabel("Mean absolute error (mm)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(diagnostics_dir / "04_oracle_mae_by_assigned_cluster.png")
    plt.close(fig)

    primary_rows = selection_df.loc[
        selection_df["metric"].astype(str).str.upper()
        == str(primary_metric).upper()
    ].copy()
    primary_rows["assigned_test_cluster"] = primary_rows["test_cluster"].astype(str)
    primary_rows["absolute_error_improvement"] = (
        primary_rows["same_cluster_absolute_error"].astype(float)
        - primary_rows["selected_absolute_error"].astype(float)
    )
    primary_rows["oracle_changed_model"] = np.where(
        primary_rows["selected_is_same_cluster"].astype(bool),
        "Same LSTM",
        "Different LSTM",
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.boxplot(
        data=primary_rows,
        x="assigned_test_cluster",
        y="absolute_error_improvement",
        hue="oracle_changed_model",
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title("Per-window Oracle Absolute-error Improvement")
    ax.set_xlabel("Assigned test cluster")
    ax.set_ylabel("Same-cluster MAE minus oracle MAE (mm)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(diagnostics_dir / "05_oracle_error_improvement_distribution.png")
    plt.close(fig)


def save_test_model_selection_report(
    output_dir: Path,
    test_model_selection: dict[str, object],
) -> None:
    """Save cross-cluster test model selection details and a text summary."""
    comparison_df = pd.DataFrame(test_model_selection.get("comparison_rows", []))
    selection_df = pd.DataFrame(test_model_selection.get("selection_rows", []))
    metric_summary_df = pd.DataFrame(
        test_model_selection.get("metric_summary_rows", [])
    )
    summary = dict(test_model_selection.get("summary", {}))
    primary_metric = str(summary.get("primary_metric", "RMSE"))
    transfer_matrix = oracle_transfer_matrix(selection_df, primary_metric)
    cluster_summary, pair_summary = oracle_routing_diagnostic_tables(
        selection_df,
        primary_metric,
    )

    comparison_df.to_csv(output_dir / "test_model_comparison.csv", index=False)
    selection_df.to_csv(output_dir / "test_model_selection.csv", index=False)
    metric_summary_df.to_csv(output_dir / "test_model_metric_summary.csv", index=False)
    transfer_matrix.to_csv(output_dir / "oracle_model_selection_matrix.csv")
    cluster_summary.to_csv(output_dir / "oracle_cluster_routing_summary.csv", index=False)
    pair_summary.to_csv(output_dir / "oracle_cluster_pair_summary.csv", index=False)

    with open(output_dir / "test_model_selection_report.txt", "w", encoding="utf-8") as f:
        f.write("TEST CLUSTER MODEL SELECTION REPORT\n")
        f.write("=" * 72 + "\n\n")
        f.write(
            "Each test sample was evaluated with every trained cluster LSTM. "
            "A winning model is selected independently for each sample and "
            "metric. These selections are diagnostic only: the main metrics, "
            "predictions, and plots keep the LSTM of the assigned cluster.\n\n"
        )
        f.write(
            "Important interpretation note: model selection is performed on the "
            "test set itself, so the reported improvement is descriptive and "
            "strongly optimistic at sample level. Treat it as an oracle-style "
            "diagnostic for cross-cluster transfer, not as an unbiased "
            "generalization estimate.\n\n"
        )
        f.write(
            "Metric-selection note: because selection is done one scalar sample "
            "at a time, MSE, RMSE, MAE, and MAPE usually select the same model; "
            "they are monotonic with absolute prediction error for a fixed "
            "target. RMSLE can differ because it ranks log-scale distance.\n\n"
        )

        f.write("Overall comparison\n")
        f.write("-" * 72 + "\n")
        f.write(f"Primary selected metric: {summary.get('primary_metric', 'RMSE')}\n")
        f.write(f"Original same-cluster RMSE: {summary.get('original_rmse', float('nan')):.4f}\n")
        f.write(f"Selected-model RMSE:       {summary.get('selected_rmse', float('nan')):.4f}\n")
        f.write(f"RMSE improvement:          {summary.get('rmse_improvement', float('nan')):.4f}\n")
        f.write(f"Original same-cluster MAE:  {summary.get('original_mae', float('nan')):.4f}\n")
        f.write(f"Selected-model MAE:        {summary.get('selected_mae', float('nan')):.4f}\n")
        f.write(f"MAE improvement:           {summary.get('mae_improvement', float('nan')):.4f}\n")
        f.write(f"Original same-cluster MSE:  {summary.get('original_mse', float('nan')):.4f}\n")
        f.write(f"Selected-model MSE:        {summary.get('selected_mse', float('nan')):.4f}\n")
        f.write(f"MSE improvement:           {summary.get('mse_improvement', float('nan')):.4f}\n")
        f.write(
            "Bootstrap 95% CI for mean squared-error improvement: "
            f"[{summary.get('mse_improvement_ci_low', float('nan')):.4f}, "
            f"{summary.get('mse_improvement_ci_high', float('nan')):.4f}]\n"
        )
        f.write(
            f"Bootstrap probability improvement > 0: "
            f"{summary.get('mse_improvement_probability', float('nan')):.3f}\n"
        )
        f.write(
            f"Samples switched to a different model by primary metric: "
            f"{summary.get('switched_samples', 0)} of {summary.get('n_test_samples', 0)}\n\n"
        )

        f.write("Selected-model aggregate metrics by selection metric\n")
        f.write("-" * 72 + "\n")
        if metric_summary_df.empty:
            f.write("No test samples were available for model selection.\n")
        else:
            for _, row in metric_summary_df.sort_values("metric_selection").iterrows():
                f.write(
                    f"{row['metric_selection']} selection: "
                    f"switched {int(row['switched_samples'])}/"
                    f"{int(row['n_test'])} samples "
                    f"({row['switched_samples_percent']:.1f}%).\n"
                )
                f.write(
                    f"  MSE:   {row['original_mse']:.4f} -> "
                    f"{row['selected_mse']:.4f} "
                    f"(change {row['mse_improvement']:.4f}, "
                    f"{row['mse_improvement_percent']:.2f}%)\n"
                )
                f.write(
                    f"  RMSE:  {row['original_rmse']:.4f} -> "
                    f"{row['selected_rmse']:.4f} "
                    f"(change {row['rmse_improvement']:.4f}, "
                    f"{row['rmse_improvement_percent']:.2f}%)\n"
                )
                f.write(
                    f"  MAE:   {row['original_mae']:.4f} -> "
                    f"{row['selected_mae']:.4f} "
                    f"(change {row['mae_improvement']:.4f}, "
                    f"{row['mae_improvement_percent']:.2f}%)\n"
                )
                f.write(
                    f"  RMSLE: {row['original_rmsle']:.4f} -> "
                    f"{row['selected_rmsle']:.4f} "
                    f"(change {row['rmsle_improvement']:.4f}, "
                    f"{row['rmsle_improvement_percent']:.2f}%)\n"
                )
                f.write(
                    f"  R2:    {row['original_r2']:.4f} -> "
                    f"{row['selected_r2']:.4f} "
                    f"(changed {row['r2_improvement']:.4f})\n"
                )
                f.write(
                    f"  MAPE:  {row['original_mape']:.4f} -> "
                    f"{row['selected_mape']:.4f} "
                    f"(change {row['mape_improvement']:.4f}, "
                    f"{row['mape_improvement_percent']:.2f}%)\n"
                )

        f.write("\nFull model-by-sample errors are in test_model_comparison.csv.\n")
        f.write("Per-sample selected models by metric are in test_model_selection.csv.\n")
        f.write("Metric-level selected prediction summaries are in test_model_metric_summary.csv.\n")
        f.write("Assigned-cluster routing summaries are in oracle_cluster_routing_summary.csv.\n")
        f.write("Assigned-to-oracle pair summaries are in oracle_cluster_pair_summary.csv.\n")

    with open(
        output_dir / "oracle_vs_same_cluster_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("ANALISE DE TRANSFERENCIA ENTRE CLUSTERS\n")
        f.write("=" * 72 + "\n\n")
        f.write(
            "Resultado principal: cada janela de teste usa a LSTM do cluster "
            "ao qual foi atribuida. Portanto, metricas, test_predictions.csv "
            "e os plots principais nao usam a selecao abaixo.\n\n"
        )
        f.write(
            "Diagnostico-oraculo: para analisar transferencia, cada janela foi "
            "tambem avaliada por todas as LSTMs e a vencedora foi escolhida apos "
            "observar o alvo real no teste. Isto nao e uma estimativa valida de "
            "desempenho futuro; indica apenas o potencial caso o roteamento entre "
            "clusters fosse melhor.\n\n"
        )
        f.write(f"Metrica de selecao: {primary_metric}\n")
        f.write(
            f"RMSE mesma LSTM do cluster: {summary.get('original_rmse', float('nan')):.4f}\n"
        )
        f.write(
            f"RMSE oraculo: {summary.get('selected_rmse', float('nan')):.4f}\n"
        )
        f.write(
            f"Janelas cujo modelo mudou: {summary.get('switched_samples', 0)} de "
            f"{summary.get('n_test_samples', 0)}\n\n"
        )
        f.write("Matriz de transferencia (linhas=cluster atribuido; colunas=LSTM oraculo)\n")
        f.write("-" * 72 + "\n")
        if transfer_matrix.empty:
            f.write("Sem janelas elegiveis para a analise.\n")
        else:
            f.write(transfer_matrix.to_string())
            f.write("\n")
        if not cluster_summary.empty:
            f.write("\nResumo por cluster atribuido\n")
            f.write("-" * 72 + "\n")
            f.write(cluster_summary.to_string(index=False))
            f.write("\n")


def save_config_summary(
    config: ExperimentConfigLike,
    output_dir: Path,
    feature_columns: list[str],
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    zero_metrics: dict[str, float],
    metrics_by_cluster: dict[int, dict[str, float]],
    test_model_selection: dict[str, object] | None,
    split_sizes: dict[str, int],
    state: str,
    station_id: str,
    pca_variance_threshold: float,
    forecast_horizon: int,
) -> None:
    """Save a compact human-readable summary for one configuration."""
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("LSTM CLUSTER SWEEP CONFIGURATION\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Run folder: {config.name}\n")
        f.write(f"Station: {state}/{station_id}\n")
        f.write(f"Window size: {config.window_size}\n")
        f.write(f"Forecast horizon: +{forecast_horizon} day(s)\n")
        f.write(f"Number of clusters: {config.n_clusters}\n")
        f.write(f"Clustering algorithm: {config.algorithm}\n")
        f.write(f"Sigma: {config.sigma if config.sigma is not None else 'not used'}\n")
        f.write(f"PCA variance threshold: {pca_variance_threshold:.2f}\n" if pca_variance_threshold is not None else "PCA variance threshold: not used\n")
        f.write(f"Features ({len(feature_columns)}): {', '.join(feature_columns)}\n")
        f.write(f"Splits: {split_sizes}\n\n")

        f.write("Metrics\n")
        f.write("-" * 72 + "\n")
        for split, metrics in [
            ("Train", train_metrics),
            ("Validation", val_metrics),
            ("Test", test_metrics),
        ]:
            f.write(
                f"{split:<10} RMSE={metrics['RMSE']:.4f}  "
                f"MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}  "
                f"RMSLE={metrics['RMSLE']:.4f}\n"
            )

        f.write("\nZero vs rainy days on test set\n")
        f.write("-" * 72 + "\n")
        for key, value in zero_metrics.items():
            f.write(f"{key}: {value}\n")

        f.write("\nTest metrics by cluster\n")
        f.write("-" * 72 + "\n")
        for cluster_id, metrics in sorted(metrics_by_cluster.items()):
            f.write(
                f"Cluster {cluster_id}: MSE={metrics['MSE']:.4f}, "
                f"RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, "
                f"R2={metrics['R2']:.4f}\n"
            )

        if test_model_selection is not None:
            summary = dict(test_model_selection.get("summary", {}))
            f.write("\nAnalise de transferencia entre clusters (diagnostico-oraculo)\n")
            f.write("-" * 72 + "\n")
            f.write(
                "As metricas acima e os plots principais usam a LSTM do cluster "
                "atribuido. A comparacao abaixo usa o alvo do teste e serve apenas "
                "para diagnosticar o potencial de melhor roteamento.\n"
            )
            f.write(
                f"RMSE mesma LSTM do cluster={summary.get('original_rmse', float('nan')):.4f}, "
                f"RMSE oraculo={summary.get('selected_rmse', float('nan')):.4f}, "
                f"ganho descritivo={summary.get('rmse_improvement', float('nan')):.4f}\n"
            )
            f.write(
                f"Primary metric: {summary.get('primary_metric', 'RMSE')}; "
                f"switched samples: {summary.get('switched_samples', 0)} of "
                f"{summary.get('n_test_samples', 0)}\n"
            )


def save_run_outputs(
    config: ExperimentConfigLike,
    output_dir: Path,
    feature_columns: list[str],
    next_day_precipitation: np.ndarray,
    current_precipitation: np.ndarray,
    input_cluster_labels: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    test_targets_by_lead_day: np.ndarray,
    current_train: np.ndarray,
    current_val: np.ndarray,
    current_test: np.ndarray,
    y_pred_train: np.ndarray,
    y_pred_val: np.ndarray,
    y_pred_test: np.ndarray,
    c_test: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    histories_by_cluster: dict[int, object],
    metrics_by_cluster: dict[int, dict[str, float]],
    state: str,
    station_id: str,
    pca_variance_threshold: float,
    forecast_horizon: int,
    test_model_selection: dict[str, object] | None = None,
    y_pred_test_by_lead_day: np.ndarray | None = None,
    test_target_dates_by_lead_day: np.ndarray | None = None,
    cluster_feature_splits: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    batch_size: int | None = None,
) -> dict[str, float | int | str | None]:
    """Save all artifacts for one run and return one sweep-level result row."""
    train_metrics = calculate_regression_metrics(y_train, y_pred_train)
    val_metrics = calculate_regression_metrics(y_val, y_pred_val)
    test_metrics = calculate_regression_metrics(y_test, y_pred_test)
    zero_metrics = calculate_zero_precipitation_metrics(y_test, y_pred_test)
    split_sizes = {
        "train": len(y_train),
        "validation": len(y_val),
        "test": len(y_test),
    }
    metrics_dataframe(train_metrics, val_metrics, test_metrics).to_csv(
        output_dir / "metrics_summary.csv",
        index=False,
    )
    cluster_metrics_dataframe(metrics_by_cluster).to_csv(
        output_dir / "cluster_model_metrics.csv",
        index=False,
    )
    save_input_precipitation_assignments(
        next_day_precipitation,
        current_precipitation,
        input_cluster_labels,
        output_dir,
        forecast_horizon=forecast_horizon,
    )
    predictions_df = pd.DataFrame(
        {
            "actual": y_test,
            "predicted": y_pred_test,
            "residual": y_test - y_pred_test,
            "current_window_precipitation_mm": current_test,
            "forecast_horizon": forecast_horizon,
            "target_minus_current_mm": y_test - current_test,
            "cluster": c_test,
            "window_index": test_indices,
        }
    )
    test_targets_by_lead_day = np.asarray(test_targets_by_lead_day, dtype=float)
    if test_targets_by_lead_day.ndim == 1:
        test_targets_by_lead_day = test_targets_by_lead_day.reshape(-1, 1)
    prediction_by_lead_day, _uses_lead_day_outputs = _lead_day_prediction_matrix(
        y_pred_test,
        y_pred_test_by_lead_day,
        int(test_targets_by_lead_day.shape[1]),
        n_rows=len(test_targets_by_lead_day),
    )
    for lead_offset in range(test_targets_by_lead_day.shape[1]):
        lead_day = lead_offset + 1
        predictions_df[f"actual_lead_day_{lead_offset + 1}"] = (
            test_targets_by_lead_day[:, lead_offset]
        )
        predictions_df[f"predicted_lead_day_{lead_day}"] = (
            prediction_by_lead_day[:, lead_offset]
        )
        predictions_df[f"residual_lead_day_{lead_day}"] = (
            test_targets_by_lead_day[:, lead_offset]
            - prediction_by_lead_day[:, lead_offset]
        )
    predictions_df = predictions_df.sort_values("window_index")
    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)
    predictions_df.to_csv(output_dir / "test_predictions_same_cluster.csv", index=False)

    if test_model_selection is not None:
        selection_summary = dict(test_model_selection.get("summary", {}))
        primary_metric = str(selection_summary.get("primary_metric", "RMSE"))
        selected_prediction_by_metric = dict(
            test_model_selection.get("selected_prediction_by_metric", {})
        )
        selected_model_by_metric = dict(
            test_model_selection.get("selected_model_by_metric", {})
        )
        oracle_predictions = np.asarray(
            selected_prediction_by_metric.get(primary_metric, []),
            dtype=float,
        )
        oracle_models = np.asarray(
            selected_model_by_metric.get(primary_metric, []),
            dtype=float,
        )
        if len(oracle_predictions) == len(predictions_df) and len(oracle_models) == len(
            predictions_df
        ):
            output_order = np.argsort(np.asarray(test_indices))
            oracle_predictions_df = predictions_df.rename(
                columns={
                    "predicted": "same_cluster_prediction",
                    "residual": "same_cluster_residual",
                }
            ).copy()
            oracle_predictions_df["oracle_selection_metric"] = primary_metric
            oracle_predictions_df["oracle_selected_model_cluster"] = oracle_models[
                output_order
            ]
            oracle_predictions_df["oracle_prediction"] = oracle_predictions[output_order]
            oracle_predictions_df["oracle_residual"] = (
                y_test[output_order] - oracle_predictions[output_order]
            )

            selected_by_lead_day = np.asarray(
                test_model_selection.get("selected_prediction_by_lead_day", []),
                dtype=float,
            )
            if selected_by_lead_day.shape == prediction_by_lead_day.shape:
                for lead_offset in range(selected_by_lead_day.shape[1]):
                    lead_day = lead_offset + 1
                    oracle_predictions_df[
                        f"oracle_prediction_lead_day_{lead_day}"
                    ] = selected_by_lead_day[output_order, lead_offset]
                    oracle_predictions_df[
                        f"oracle_residual_lead_day_{lead_day}"
                    ] = (
                        test_targets_by_lead_day[output_order, lead_offset]
                        - selected_by_lead_day[output_order, lead_offset]
                    )
            oracle_predictions_df.to_csv(
                output_dir / "test_predictions_oracle_selection.csv",
                index=False,
            )

    horizon_summary = save_forecast_horizon_diagnostics(
        y_train,
        y_val,
        y_test,
        current_train,
        current_val,
        current_test,
        y_pred_test,
        c_test,
        train_indices,
        val_indices,
        test_indices,
        output_dir,
        forecast_horizon=forecast_horizon,
    )
    lead_day_metrics = save_forecast_lead_day_diagnostics(
        y_pred_test,
        c_test,
        test_indices,
        test_targets_by_lead_day,
        output_dir,
        forecast_horizon=forecast_horizon,
        y_pred_test_by_lead_day=prediction_by_lead_day,
    )

    if test_model_selection is not None:
        save_test_model_selection_report(output_dir, test_model_selection)

    with open(output_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(
            create_evaluation_report(
                y_train,
                y_val,
                y_test,
                y_pred_train,
                y_pred_val,
                y_pred_test,
                c_test,
            )
        )

    save_config_summary(
        config=config,
        output_dir=output_dir,
        feature_columns=feature_columns,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        zero_metrics=zero_metrics,
        metrics_by_cluster=metrics_by_cluster,
        test_model_selection=test_model_selection,
        split_sizes=split_sizes,
        state=state,
        station_id=station_id,
        pca_variance_threshold=pca_variance_threshold,
        forecast_horizon=forecast_horizon,
    )
    save_visualizations(
        y_test,
        y_pred_test,
        c_test,
        test_indices,
        next_day_precipitation,
        input_cluster_labels,
        histories_by_cluster,
        output_dir,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        test_targets_by_lead_day=test_targets_by_lead_day,
        y_pred_test_by_lead_day=prediction_by_lead_day,
        test_target_dates_by_lead_day=test_target_dates_by_lead_day,
        cluster_feature_splits=cluster_feature_splits,
    )
    if test_model_selection is not None:
        save_oracle_model_visualizations(
            test_model_selection,
            y_test,
            c_test,
            test_indices,
            next_day_precipitation,
            input_cluster_labels,
            histories_by_cluster,
            output_dir,
            forecast_horizon=forecast_horizon,
            batch_size=batch_size,
            test_targets_by_lead_day=test_targets_by_lead_day,
            regular_prediction_by_lead_day=prediction_by_lead_day,
            test_target_dates_by_lead_day=test_target_dates_by_lead_day,
            cluster_feature_splits=cluster_feature_splits,
        )
        save_oracle_transfer_diagnostics(
            y_test,
            test_model_selection,
            output_dir,
        )

    result = {
        "run_name": config.name,
        "window_size": config.window_size,
        "n_clusters": config.n_clusters,
        "algorithm": config.algorithm,
        "sigma": config.sigma,
        "forecast_horizon": forecast_horizon,
        "zero_days_ratio": zero_metrics["zero_days_ratio"],
        "rainy_days_rmse": zero_metrics.get("rainy_days_rmse", np.nan),
        "lead_day_metrics_path": (
            "forecast_horizon_diagnostics/test_prediction_metrics_by_lead_day.csv"
            if not lead_day_metrics.empty
            else None
        ),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
    }

    for split, metrics in [
        ("train", train_metrics),
        ("val", val_metrics),
        ("test", test_metrics),
    ]:
        for metric_name, value in metrics.items():
            result[f"{split}_{metric_name.lower()}"] = value

    for cluster_id, metrics in sorted(metrics_by_cluster.items()):
        for metric_name, value in metrics.items():
            result[f"cluster_{cluster_id}_{metric_name.lower()}"] = value

    for key, value in horizon_summary.items():
        result[f"horizon_{key}"] = value

    if test_model_selection is not None:
        selection_summary = dict(test_model_selection.get("summary", {}))
        for key, value in selection_summary.items():
            result[f"test_selection_{key}"] = value

    return result


def latex_table(results_df: pd.DataFrame, quantitative_metrics: list[str]) -> str:
    """Format sweep results as a LaTeX table."""
    table_df = results_df.sort_values(["test_rmse", "test_mae"]).copy()
    metrics = [metric.upper() for metric in quantitative_metrics]
    alignment = "rrr" + "r" * len(metrics) + "r"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{LSTM cluster sweep results for RS A801.}",
        r"\label{tab:lstm_cluster_sweep_rs_a801}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        "Window & $K$ & Sigma & "
        + " & ".join(metric_latex_label(metric, "test") for metric in metrics)
        + r" & $N_{test}$ \\",
        r"\hline",
    ]

    for _, row in table_df.iterrows():
        sigma = "N/A" if pd.isna(row["sigma"]) else f"{row['sigma']:.4g}"
        metric_values = [
            format_latex_number(row.get(f"test_{metric_key(metric)}"))
            for metric in metrics
        ]
        lines.append(
            f"{int(row['window_size'])} & "
            f"{int(row['n_clusters'])} & "
            f"{sigma} & "
            + " & ".join(metric_values)
            + " & "
            f"{int(row['n_test'])} \\\\"
        )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def format_latex_number(value: object, digits: int = 3) -> str:
    """Format numeric values for compact LaTeX tables."""
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def metric_key(metric_name: str) -> str:
    """Return the dataframe key used for a metric name."""
    return metric_name.lower()


def metric_latex_label(metric_name: str, suffix: str) -> str:
    """Return a compact LaTeX-safe metric column label."""
    label = "R^2" if metric_name.upper() == "R2" else metric_name.upper()
    return rf"${label}_{{{suffix}}}$"


def cluster_metric_latex_table(
    results_df: pd.DataFrame,
    n_clusters: int,
    quantitative_metrics: list[str],
) -> str:
    """Format one per-cluster metric table for a fixed number of clusters."""
    table_df = results_df[results_df["n_clusters"] == n_clusters].copy()
    table_df = table_df.sort_values(["window_size", "sigma"])

    metrics = [metric.upper() for metric in quantitative_metrics]
    overall_columns = [(metric, metric_key(metric)) for metric in metrics]
    cluster_columns = [
        (metric, metric_key(metric), cluster_id)
        for metric in metrics
        for cluster_id in range(n_clusters)
    ]
    alignment = "rr" + "r" * (len(overall_columns) + len(cluster_columns))
    metric_headers = [
        metric_latex_label(metric, "all")
        for metric, _ in overall_columns
    ] + [
        metric_latex_label(metric, str(cluster_id))
        for metric, _, cluster_id in cluster_columns
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        "Window & Sigma & " + " & ".join(metric_headers) + r" \\",
        r"\hline",
    ]

    for _, row in table_df.iterrows():
        sigma = "N/A" if pd.isna(row["sigma"]) else f"{row['sigma']:.4g}"
        overall_values = [
            format_latex_number(row.get(f"test_{metric_key_name}"))
            for _, metric_key_name in overall_columns
        ]
        cluster_values = [
            format_latex_number(row.get(f"cluster_{cluster_id}_{metric_key}"))
            for _, metric_key, cluster_id in cluster_columns
        ]
        lines.append(
            f"{int(row['window_size'])} & {sigma} & "
            + " & ".join(overall_values + cluster_values)
            + r" \\"
        )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            rf"\caption{{Overall and cluster-specific test metrics for $K={n_clusters}$ models.}}",
            rf"\label{{tab:lstm_cluster_metrics_k{n_clusters}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def cluster_metric_latex_tables(
    results_df: pd.DataFrame,
    quantitative_metrics: list[str],
) -> str:
    """Format one cluster-metric LaTeX table for each tested K."""
    tables = [
        cluster_metric_latex_table(results_df, int(n_clusters), quantitative_metrics)
        for n_clusters in sorted(results_df["n_clusters"].unique())
    ]
    return "\n".join(tables)


def save_sweep_outputs(
    results: list[dict[str, float | int | str | None]],
    sweep_dir: Path,
    state: str,
    station_id: str,
    window_sizes: list[int],
    n_clusters_list: list[int],
    clustering_algorithm: str,
    quantitative_metrics: list[str],
) -> None:
    """Save sweep-level CSV, text summary, and LaTeX table."""
    results_df = pd.DataFrame(results).sort_values(["test_rmse", "test_mae"])
    results_df.to_csv(sweep_dir / "sweep_results.csv", index=False)

    with open(sweep_dir / "overleaf_table.txt", "w", encoding="utf-8") as f:
        f.write(latex_table(results_df, quantitative_metrics))
    with open(sweep_dir / "overleaf_cluster_metric_tables.txt", "w", encoding="utf-8") as f:
        f.write(cluster_metric_latex_tables(results_df, quantitative_metrics))

    best = results_df.iloc[0]
    with open(sweep_dir / "sweep_summary.txt", "w", encoding="utf-8") as f:
        f.write("LSTM CLUSTER SWEEP SUMMARY\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Station: {state}/{station_id}\n")
        f.write(f"Configurations: {len(results_df)}\n")
        f.write(f"Window sizes: {window_sizes}\n")
        f.write(f"Cluster counts: {n_clusters_list}\n")
        f.write(f"Algorithm: {clustering_algorithm}\n\n")
        f.write("Best configuration by test RMSE\n")
        f.write("-" * 72 + "\n")
        f.write(best.to_string())
        f.write("\n\nFull results are in sweep_results.csv.\n")
