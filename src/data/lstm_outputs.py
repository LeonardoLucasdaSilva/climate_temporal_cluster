"""Output writers for LSTM clustering experiments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
        axes[1].set_ylabel("MAE (mm)")
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


def precipitation_bin_edges(values: np.ndarray) -> np.ndarray:
    """Return readable shared precipitation bins for cluster histograms."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([0.0, 1.0])

    max_value = float(values.max())
    if max_value <= 0:
        return np.array([0.0, 1.0])

    raw_edges = np.histogram_bin_edges(values, bins="fd")
    n_bins = max(8, min(len(raw_edges) - 1, 35))
    return np.linspace(0.0, max_value, n_bins + 1)


def save_input_precipitation_assignments(
    next_day_precipitation: np.ndarray,
    cluster_labels: np.ndarray,
    output_dir: Path,
) -> None:
    """Save the next-day precipitation target assigned to each input window."""
    pd.DataFrame(
        {
            "input_index": np.arange(len(next_day_precipitation)),
            "cluster": cluster_labels.astype(int),
            "next_day_precipitation_mm": next_day_precipitation,
        }
    ).to_csv(output_dir / "input_next_day_precipitation_by_cluster.csv", index=False)


def save_input_precipitation_distribution_by_cluster(
    next_day_precipitation: np.ndarray,
    cluster_labels: np.ndarray,
    output_dir: Path,
) -> None:
    """Save horizontal histograms of next-day precipitation for each input cluster."""
    hist_dir = output_dir / "input_precipitation_distribution_by_cluster"
    hist_dir.mkdir(exist_ok=True)

    bin_edges = precipitation_bin_edges(next_day_precipitation)
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
        values = next_day_precipitation[mask]
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
        ax.set_ylabel("Next-day precipitation (mm)")
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
            f"Input Windows in Cluster {int(cluster_id)}: Next-day Precipitation"
        )
        cluster_ax.set_xlabel("Samples")
        cluster_ax.set_ylabel("Next-day precipitation (mm)")
        cluster_ax.grid(True, alpha=0.25, axis="x")
        cluster_fig.tight_layout()
        cluster_fig.savefig(
            hist_dir / f"cluster_{int(cluster_id)}_input_precipitation_distribution.png"
        )
        plt.close(cluster_fig)

    fig.suptitle("Input Window Next-day Precipitation Distribution by Cluster", y=1.0)
    fig.tight_layout()
    fig.savefig(hist_dir / "08_input_precipitation_distribution_by_cluster.png")
    plt.close(fig)


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
        positions, compressed_intervals = compressed_time_positions(
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
            positions,
            actual,
            label="Actual",
            color="#4C78A8",
            linewidth=1.5,
            alpha=0.85,
        )
        axes[0].scatter(
            positions,
            actual,
            color="#4C78A8",
            s=10,
            alpha=0.65,
            zorder=3,
        )
        axes[0].plot(
            positions,
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
            positions,
            residuals,
            color="#54A24B",
            linewidth=1.2,
            alpha=0.85,
        )
        axes[1].fill_between(
            positions,
            residuals,
            0.0,
            color="#54A24B",
            alpha=0.18,
        )
        axes[1].set_ylabel("Residual (mm)")
        axes[1].grid(True, alpha=0.3)

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
    ax.set_ylabel("Next-day precipitation (mm)")
    fig.tight_layout()
    fig.savefig(plot_dir / "07_precipitation_distribution_by_cluster.png")
    plt.close(fig)


def save_cluster_distribution_plot(c_test: np.ndarray, output_dir: Path) -> None:
    """Save cluster sample counts for the test set."""
    plot_dir = output_dir / "cluster_diagnostics"
    plot_dir.mkdir(exist_ok=True)

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


def save_prediction_timeseries_splits(
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    output_dir: Path,
    n_splits: int = 4,
) -> None:
    """Save the prediction time series in readable sequential splits."""
    plot_dir = output_dir / "prediction_timeseries_splits"
    plot_dir.mkdir(exist_ok=True)

    indices = np.arange(len(y_test))
    for split_index, split_indices in enumerate(np.array_split(indices, n_splits), start=1):
        if split_indices.size == 0:
            continue

        fig, ax = plt.subplots(figsize=(14, 5.6))
        ax.plot(
            split_indices,
            y_test[split_indices],
            label="Actual",
            alpha=0.8,
            linewidth=1.6,
        )
        ax.plot(
            split_indices,
            y_pred_test[split_indices],
            label="Predicted",
            alpha=0.8,
            linewidth=1.6,
        )
        ax.set_title(f"Predictions vs Actual - Split {split_index} of {n_splits}")
        ax.set_xlabel("Test Sample Index")
        ax.set_ylabel("Precipitation (mm)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            plot_dir / f"02_predictions_timeseries_split_{split_index:02d}_of_{n_splits:02d}.png"
        )
        plt.close(fig)

def save_visualizations(
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    c_test: np.ndarray,
    test_indices: np.ndarray,
    next_day_precipitation: np.ndarray,
    input_cluster_labels: np.ndarray,
    histories_by_cluster: dict[int, object],
    output_dir: Path,
) -> None:
    """Save the diagnostic plots for one configuration."""
    prediction_dir = output_dir / "prediction_overview"
    residual_dir = output_dir / "residual_diagnostics"
    cluster_dir = output_dir / "cluster_diagnostics"
    prediction_dir.mkdir(exist_ok=True)
    residual_dir.mkdir(exist_ok=True)
    cluster_dir.mkdir(exist_ok=True)

    save_training_history_plots(histories_by_cluster, output_dir)

    fig, _ = plot_predictions_vs_actual(
        y_test,
        y_pred_test,
        title="Test Set: Predictions vs Actual Precipitation",
    )
    fig.savefig(prediction_dir / "02_predictions_vs_actual.png")
    plt.close(fig)
    save_prediction_timeseries_splits(y_test, y_pred_test, output_dir)
    save_cluster_prediction_timeseries(
        y_test,
        y_pred_test,
        c_test,
        test_indices,
        output_dir,
    )

    fig, _ = plot_residuals(y_test, y_pred_test, title="Test Set: Residual Analysis")
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

    save_cluster_distribution_plot(c_test, output_dir)
    save_precipitation_by_cluster_plot(y_test, c_test, output_dir)
    save_cluster_precipitation_histograms(y_test, c_test, output_dir)
    save_input_precipitation_distribution_by_cluster(
        next_day_precipitation,
        input_cluster_labels,
        output_dir,
    )
    save_cluster_prediction_histograms(y_test, y_pred_test, c_test, output_dir)


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

    comparison_df.to_csv(output_dir / "test_model_comparison.csv", index=False)
    selection_df.to_csv(output_dir / "test_model_selection.csv", index=False)
    metric_summary_df.to_csv(output_dir / "test_model_metric_summary.csv", index=False)

    with open(output_dir / "test_model_selection_report.txt", "w", encoding="utf-8") as f:
        f.write("TEST CLUSTER MODEL SELECTION REPORT\n")
        f.write("=" * 72 + "\n\n")
        f.write(
            "Each test sample was evaluated with every trained cluster LSTM. "
            "A winning model is selected independently for each sample and "
            "metric. The main prediction columns use the RMSE/MSE-equivalent "
            "per-sample squared-error choice.\n\n"
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
) -> None:
    """Save a compact human-readable summary for one configuration."""
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("LSTM CLUSTER SWEEP CONFIGURATION\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Run folder: {config.name}\n")
        f.write(f"Station: {state}/{station_id}\n")
        f.write(f"Window size: {config.window_size}\n")
        f.write(f"Number of clusters: {config.n_clusters}\n")
        f.write(f"Clustering algorithm: {config.algorithm}\n")
        f.write(f"Sigma: {config.sigma if config.sigma is not None else 'not used'}\n")
        f.write(f"PCA variance threshold: {pca_variance_threshold:.2f}\n")
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
            f.write("\nCross-cluster test model selection\n")
            f.write("-" * 72 + "\n")
            f.write(
                f"Same-cluster RMSE={summary.get('original_rmse', float('nan')):.4f}, "
                f"selected-model RMSE={summary.get('selected_rmse', float('nan')):.4f}, "
                f"improvement={summary.get('rmse_improvement', float('nan')):.4f}\n"
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
    input_cluster_labels: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    y_pred_train: np.ndarray,
    y_pred_val: np.ndarray,
    y_pred_test: np.ndarray,
    c_test: np.ndarray,
    test_indices: np.ndarray,
    histories_by_cluster: dict[int, object],
    metrics_by_cluster: dict[int, dict[str, float]],
    state: str,
    station_id: str,
    pca_variance_threshold: float,
    test_model_selection: dict[str, object] | None = None,
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
        input_cluster_labels,
        output_dir,
    )
    predictions_df = pd.DataFrame(
        {
            "actual": y_test,
            "predicted": y_pred_test,
            "residual": y_test - y_pred_test,
            "cluster": c_test,
            "window_index": test_indices,
        }
    )
    if test_model_selection is not None:
        selected_model_by_sample = np.asarray(
            test_model_selection.get("selected_model_by_sample", []),
            dtype=float,
        )
        original_prediction_by_sample = np.asarray(
            test_model_selection.get("original_prediction_by_sample", []),
            dtype=float,
        )
        if len(selected_model_by_sample) == len(predictions_df):
            predictions_df["selected_model_cluster"] = selected_model_by_sample
        if len(original_prediction_by_sample) == len(predictions_df):
            predictions_df["same_cluster_prediction"] = original_prediction_by_sample
            predictions_df["same_cluster_residual"] = y_test - original_prediction_by_sample
        selected_prediction_by_metric = dict(
            test_model_selection.get("selected_prediction_by_metric", {})
        )
        selected_model_by_metric = dict(
            test_model_selection.get("selected_model_by_metric", {})
        )
        for metric_name, metric_predictions in selected_prediction_by_metric.items():
            metric_key = str(metric_name).lower()
            metric_predictions = np.asarray(metric_predictions, dtype=float)
            if len(metric_predictions) == len(predictions_df):
                predictions_df[f"selected_prediction_{metric_key}"] = metric_predictions
                predictions_df[f"selected_residual_{metric_key}"] = (
                    y_test - metric_predictions
                )
        for metric_name, metric_models in selected_model_by_metric.items():
            metric_key = str(metric_name).lower()
            metric_models = np.asarray(metric_models, dtype=float)
            if len(metric_models) == len(predictions_df):
                predictions_df[f"selected_model_cluster_{metric_key}"] = metric_models
    predictions_df = predictions_df.sort_values("window_index")
    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)

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
    )

    result = {
        "run_name": config.name,
        "window_size": config.window_size,
        "n_clusters": config.n_clusters,
        "algorithm": config.algorithm,
        "sigma": config.sigma,
        "zero_days_ratio": zero_metrics["zero_days_ratio"],
        "rainy_days_rmse": zero_metrics.get("rainy_days_rmse", np.nan),
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
