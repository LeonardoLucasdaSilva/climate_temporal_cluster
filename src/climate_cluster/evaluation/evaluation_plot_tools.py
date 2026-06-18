"""Plot helpers for regression and cluster-level model diagnostics.

The functions in this module create matplotlib figures but do not save or show
them. Callers are responsible for saving with `fig.savefig(...)`, displaying
with `plt.show()`, and closing figures with `plt.close(fig)` when needed.
"""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual Precipitation",
    figsize: Tuple[int, int] = (12, 4),
) -> Tuple[plt.Figure, np.ndarray]:
    """Create time-series and scatter plots for predicted vs. actual values.

    The left subplot compares actual and predicted precipitation across sample
    order. The right subplot compares the same values as a scatter plot with a
    red dashed perfect-prediction reference line.

    Args:
        y_true: One-dimensional array with observed precipitation values.
            Expected shape is `(n_samples,)`.
        y_pred: One-dimensional array with predicted precipitation values.
            Must have the same length as `y_true`.
        title: Base title used for both subplots.
        figsize: Matplotlib figure size as `(width, height)` in inches.

    Returns:
        A tuple `(fig, axes)` where `fig` is the created matplotlib figure and
        `axes` is a two-element array of subplot axes: time series first,
        scatter plot second.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    ax1.plot(y_true, label="Actual", alpha=0.7, linewidth=1.5)
    ax1.plot(y_pred, label="Predicted", alpha=0.7, linewidth=1.5)
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Precipitation (mm)")
    ax1.set_title(f"{title} - Time Series")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(y_true, y_pred, alpha=0.5, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    ax2.set_xlabel("Actual Precipitation (mm)")
    ax2.set_ylabel("Predicted Precipitation (mm)")
    ax2.set_title(f"{title} - Scatter Plot")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis",
    figsize: Tuple[int, int] = (12, 4),
) -> Tuple[plt.Figure, np.ndarray]:
    """Create residual-over-time and residual-distribution plots.

    Residuals are calculated as `y_true - y_pred`. The left subplot shows each
    residual by sample index with a zero-reference line. The right subplot shows
    the residual distribution as a histogram.

    Args:
        y_true: One-dimensional array with observed precipitation values.
            Expected shape is `(n_samples,)`.
        y_pred: One-dimensional array with predicted precipitation values.
            Must have the same length as `y_true`.
        title: Base title used for both subplots.
        figsize: Matplotlib figure size as `(width, height)` in inches.

    Returns:
        A tuple `(fig, axes)` where `fig` is the created matplotlib figure and
        `axes` is a two-element array of subplot axes: residuals over time
        first, residual histogram second.
    """
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    ax1.scatter(range(len(residuals)), residuals, alpha=0.5, s=20)
    ax1.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Residuals (mm)")
    ax1.set_title(f"{title} - Over Time")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Residuals (mm)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"{title} - Distribution")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig, axes


def plot_error_by_magnitude(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    title: str = "Error Analysis by Precipitation Magnitude",
    figsize: Tuple[int, int] = (12, 4),
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot MAE and RMSE grouped by precipitation magnitude bins.

    The function bins samples by percentiles of `y_true`, then computes error
    statistics inside each non-empty bin. This is useful for checking whether a
    model behaves differently on low-, medium-, and high-precipitation days.

    Args:
        y_true: One-dimensional array with observed precipitation values.
            Expected shape is `(n_samples,)`.
        y_pred: One-dimensional array with predicted precipitation values.
            Must have the same length as `y_true`.
        n_bins: Number of percentile bins to create from `y_true`.
        title: Base title used for both subplots.
        figsize: Matplotlib figure size as `(width, height)` in inches.

    Returns:
        A tuple `(fig, axes)` where `fig` is the created matplotlib figure and
        `axes` is a two-element array of subplot axes: MAE by bin first, RMSE by
        bin second.
    """
    bins = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(y_true, bins) - 1
    errors = np.abs(y_true - y_pred)
    squared_errors = (y_true - y_pred) ** 2

    mae_by_bin = []
    rmse_by_bin = []
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            mae_by_bin.append(np.mean(errors[mask]))
            rmse_by_bin.append(np.sqrt(np.mean(squared_errors[mask])))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].bar(range(len(mae_by_bin)), mae_by_bin, color="steelblue", alpha=0.7)
    axes[0].set_xlabel("Precipitation Bin (ordered by magnitude)")
    axes[0].set_ylabel("Mean Absolute Error (mm)")
    axes[0].set_title(f"{title} - MAE")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(range(len(rmse_by_bin)), rmse_by_bin, color="coral", alpha=0.7)
    axes[1].set_xlabel("Precipitation Bin (ordered by magnitude)")
    axes[1].set_ylabel("Root Mean Squared Error (mm)")
    axes[1].set_title(f"{title} - RMSE")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig, axes


def plot_cluster_performance(
    cluster_labels: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "LSTM Performance by Cluster",
    figsize: Tuple[int, int] = (14, 5),
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot MAE and RMSE for each cluster.

    For every cluster id present in `cluster_labels`, this function filters the
    matching samples, computes MAE and RMSE, and annotates each bar with the
    number of samples in that cluster.

    Args:
        cluster_labels: One-dimensional array of cluster assignments. Must have
            the same length as `y_true` and `y_pred`.
        y_true: One-dimensional array with observed precipitation values.
            Expected shape is `(n_samples,)`.
        y_pred: One-dimensional array with predicted precipitation values.
            Must have the same length as `y_true`.
        title: Base title used for both subplots.
        figsize: Matplotlib figure size as `(width, height)` in inches.

    Returns:
        A tuple `(fig, axes)` where `fig` is the created matplotlib figure and
        `axes` is a two-element array of subplot axes: cluster MAE first,
        cluster RMSE second.
    """
    unique_clusters = np.unique(cluster_labels)
    mae_by_cluster = []
    rmse_by_cluster = []
    cluster_ids = []
    cluster_sizes = []

    for cluster_id in sorted(unique_clusters):
        mask = cluster_labels == cluster_id
        if np.sum(mask) > 0:
            y_true_cluster = y_true[mask]
            y_pred_cluster = y_pred[mask]
            mae_by_cluster.append(mean_absolute_error(y_true_cluster, y_pred_cluster))
            rmse_by_cluster.append(np.sqrt(mean_squared_error(y_true_cluster, y_pred_cluster)))
            cluster_ids.append(int(cluster_id))
            cluster_sizes.append(np.sum(mask))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_ids)))

    bars1 = axes[0].bar(range(len(mae_by_cluster)), mae_by_cluster, color=colors, alpha=0.7)
    axes[0].set_xlabel("Cluster ID")
    axes[0].set_ylabel("Mean Absolute Error (mm)")
    axes[0].set_title(f"{title} - MAE by Cluster")
    axes[0].set_xticks(range(len(cluster_ids)))
    axes[0].set_xticklabels(cluster_ids)
    axes[0].grid(True, alpha=0.3, axis="y")

    bars2 = axes[1].bar(range(len(rmse_by_cluster)), rmse_by_cluster, color=colors, alpha=0.7)
    axes[1].set_xlabel("Cluster ID")
    axes[1].set_ylabel("Root Mean Squared Error (mm)")
    axes[1].set_title(f"{title} - RMSE by Cluster")
    axes[1].set_xticks(range(len(cluster_ids)))
    axes[1].set_xticklabels(cluster_ids)
    axes[1].grid(True, alpha=0.3, axis="y")

    for ax, bars in [(axes[0], bars1), (axes[1], bars2)]:
        for bar, count in zip(bars, cluster_sizes):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    return fig, axes
