"""Evaluation metrics and visualization functions for LSTM models."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Calculate comprehensive regression evaluation metrics.

    Args:
        y_true: True precipitation values
        y_pred: Predicted precipitation values

    Returns:
        Dictionary with metrics:
        - MSE: Mean Squared Error
        - RMSE: Root Mean Squared Error
        - MAE: Mean Absolute Error
        - RMSLE: Root Mean Squared Logarithmic Error (for non-negative values)
        - R²: R-squared (coefficient of determination)
        - MAPE: Mean Absolute Percentage Error
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # RMSLE (Root Mean Squared Logarithmic Error)
    # Safe for zero values
    rmsle = np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))

    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = y_true != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'RMSLE': float(rmsle),
        'R2': float(r2),
        'MAPE': float(mape),
    }


def calculate_zero_precipitation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 1.0,
) -> Dict[str, float]:
    """Calculate metrics for zero vs. non-zero precipitation prediction.

    This is important for precipitation forecasting since many days have zero rain.

    Args:
        y_true: True precipitation values
        y_pred: Predicted precipitation values
        threshold: Precipitation threshold for "rainy day" (default: 1.0 mm)

    Returns:
        Dictionary with metrics on zero/non-zero days
    """
    zero_mask = y_true < threshold
    nonzero_mask = y_true >= threshold

    metrics = {
        'zero_days_count': int(np.sum(zero_mask)),
        'rainy_days_count': int(np.sum(nonzero_mask)),
        'zero_days_ratio': float(np.sum(zero_mask) / len(y_true)) if len(y_true) > 0 else 0,
    }

    # Metrics for zero days only
    if np.sum(zero_mask) > 0:
        y_zero_true = y_true[zero_mask]
        y_zero_pred = y_pred[zero_mask]
        metrics['zero_days_mae'] = float(mean_absolute_error(y_zero_true, y_zero_pred))
        metrics['zero_days_rmse'] = float(np.sqrt(mean_squared_error(y_zero_true, y_zero_pred)))

    # Metrics for rainy days only
    if np.sum(nonzero_mask) > 0:
        y_nonzero_true = y_true[nonzero_mask]
        y_nonzero_pred = y_pred[nonzero_mask]
        metrics['rainy_days_mae'] = float(mean_absolute_error(y_nonzero_true, y_nonzero_pred))
        metrics['rainy_days_rmse'] = float(np.sqrt(mean_squared_error(y_nonzero_true, y_nonzero_pred)))
        metrics['rainy_days_r2'] = float(r2_score(y_nonzero_true, y_nonzero_pred))

    return metrics


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual Precipitation",
    figsize: Tuple[int, int] = (12, 4),
) -> Tuple[plt.Figure, np.ndarray]:
    """Create comparison plots for predictions vs actual values.

    Creates a 1x2 subplot grid:
    - Left: Time series comparison
    - Right: Scatter plot with perfect prediction line

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size

    Returns:
        Tuple of (figure, axes)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Time series plot
    ax1 = axes[0]
    ax1.plot(y_true, label='Actual', alpha=0.7, linewidth=1.5)
    ax1.plot(y_pred, label='Predicted', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Precipitation (mm)')
    ax1.set_title(f'{title} - Time Series')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter plot
    ax2 = axes[1]
    ax2.scatter(y_true, y_pred, alpha=0.5, s=20)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax2.set_xlabel('Actual Precipitation (mm)')
    ax2.set_ylabel('Predicted Precipitation (mm)')
    ax2.set_title(f'{title} - Scatter Plot')
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
    """Create residual analysis plots.

    Creates a 1x2 subplot grid:
    - Left: Residuals over time
    - Right: Residuals distribution histogram

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size

    Returns:
        Tuple of (figure, axes)
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Residuals over time
    ax1 = axes[0]
    ax1.scatter(range(len(residuals)), residuals, alpha=0.5, s=20)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Residuals (mm)')
    ax1.set_title(f'{title} - Over Time')
    ax1.grid(True, alpha=0.3)

    # Residuals distribution
    ax2 = axes[1]
    ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Residuals (mm)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{title} - Distribution')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig, axes


def plot_error_by_magnitude(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    title: str = "Error Analysis by Precipitation Magnitude",
    figsize: Tuple[int, int] = (12, 4),
) -> Tuple[plt.Figure, np.ndarray]:
    """Analyze error metrics grouped by precipitation magnitude.

    Creates a 1x2 subplot grid:
    - Left: Mean Absolute Error by precipitation bin
    - Right: Root Mean Squared Error by precipitation bin

    Args:
        y_true: True values
        y_pred: Predicted values
        n_bins: Number of bins for grouping
        title: Plot title
        figsize: Figure size

    Returns:
        Tuple of (figure, axes)
    """
    # Create bins based on actual precipitation values
    bins = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(y_true, bins) - 1

    errors = np.abs(y_true - y_pred)
    squared_errors = (y_true - y_pred) ** 2

    mae_by_bin = []
    rmse_by_bin = []
    bin_centers = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            mae_by_bin.append(np.mean(errors[mask]))
            rmse_by_bin.append(np.sqrt(np.mean(squared_errors[mask])))
            bin_center = (bins[i] + bins[i + 1]) / 2
            bin_centers.append(bin_center)
            bin_counts.append(np.sum(mask))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # MAE by bin
    ax1 = axes[0]
    ax1.bar(range(len(mae_by_bin)), mae_by_bin, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Precipitation Bin (ordered by magnitude)')
    ax1.set_ylabel('Mean Absolute Error (mm)')
    ax1.set_title(f'{title} - MAE')
    ax1.grid(True, alpha=0.3, axis='y')

    # RMSE by bin
    ax2 = axes[1]
    ax2.bar(range(len(rmse_by_bin)), rmse_by_bin, color='coral', alpha=0.7)
    ax2.set_xlabel('Precipitation Bin (ordered by magnitude)')
    ax2.set_ylabel('Root Mean Squared Error (mm)')
    ax2.set_title(f'{title} - RMSE')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig, axes


def plot_cluster_performance(
    cluster_labels: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "LSTM Performance by Cluster",
    figsize: Tuple[int, int] = (14, 5),
) -> Tuple[plt.Figure, np.ndarray]:
    """Analyze and visualize LSTM performance for each cluster separately.

    Creates a 1x2 subplot grid:
    - Left: MAE for each cluster
    - Right: RMSE for each cluster

    Args:
        cluster_labels: Cluster assignment for each sample
        y_true: True precipitation values
        y_pred: Predicted precipitation values
        title: Plot title
        figsize: Figure size

    Returns:
        Tuple of (figure, axes)
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

            mae = mean_absolute_error(y_true_cluster, y_pred_cluster)
            rmse = np.sqrt(mean_squared_error(y_true_cluster, y_pred_cluster))

            mae_by_cluster.append(mae)
            rmse_by_cluster.append(rmse)
            cluster_ids.append(int(cluster_id))
            cluster_sizes.append(np.sum(mask))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # MAE by cluster
    ax1 = axes[0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_ids)))
    bars1 = ax1.bar(range(len(mae_by_cluster)), mae_by_cluster, color=colors, alpha=0.7)
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Mean Absolute Error (mm)')
    ax1.set_title(f'{title} - MAE by Cluster')
    ax1.set_xticks(range(len(cluster_ids)))
    ax1.set_xticklabels(cluster_ids)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add sample count labels
    for i, (bar, count) in enumerate(zip(bars1, cluster_sizes)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'n={count}', ha='center', va='bottom', fontsize=9)

    # RMSE by cluster
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(rmse_by_cluster)), rmse_by_cluster, color=colors, alpha=0.7)
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Root Mean Squared Error (mm)')
    ax2.set_title(f'{title} - RMSE by Cluster')
    ax2.set_xticks(range(len(cluster_ids)))
    ax2.set_xticklabels(cluster_ids)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add sample count labels
    for i, (bar, count) in enumerate(zip(bars2, cluster_sizes)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'n={count}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig, axes


def create_evaluation_report(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    y_pred_train: np.ndarray,
    y_pred_val: np.ndarray,
    y_pred_test: np.ndarray,
    cluster_labels: np.ndarray,
) -> str:
    """Create a comprehensive text evaluation report.

    Args:
        y_train: Training true values
        y_val: Validation true values
        y_test: Test true values
        y_pred_train: Training predictions
        y_pred_val: Validation predictions
        y_pred_test: Test predictions
        cluster_labels: Cluster assignments for test data

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("LSTM PRECIPITATION PREDICTION - COMPREHENSIVE EVALUATION REPORT")
    report.append("=" * 80)

    # Data splits overview
    report.append("\n[DATA SPLITS OVERVIEW]")
    report.append(f"Training set size: {len(y_train)} samples")
    report.append(f"Validation set size: {len(y_val)} samples")
    report.append(f"Test set size: {len(y_test)} samples")
    report.append(f"Total: {len(y_train) + len(y_val) + len(y_test)} samples")

    # Training metrics
    report.append("\n[TRAINING SET METRICS]")
    metrics_train = calculate_regression_metrics(y_train, y_pred_train)
    for metric_name, metric_value in metrics_train.items():
        if metric_name == 'MAPE':
            report.append(f"{metric_name:8s}: {metric_value:.2f}%")
        else:
            report.append(f"{metric_name:8s}: {metric_value:.4f}")

    # Validation metrics
    report.append("\n[VALIDATION SET METRICS]")
    metrics_val = calculate_regression_metrics(y_val, y_pred_val)
    for metric_name, metric_value in metrics_val.items():
        if metric_name == 'MAPE':
            report.append(f"{metric_name:8s}: {metric_value:.2f}%")
        else:
            report.append(f"{metric_name:8s}: {metric_value:.4f}")

    # Test metrics (main evaluation)
    report.append("\n[TEST SET METRICS] - PRIMARY EVALUATION")
    metrics_test = calculate_regression_metrics(y_test, y_pred_test)
    for metric_name, metric_value in metrics_test.items():
        if metric_name == 'MAPE':
            report.append(f"{metric_name:8s}: {metric_value:.2f}%")
        else:
            report.append(f"{metric_name:8s}: {metric_value:.4f}")

    # Zero vs Rainy day metrics
    report.append("\n[ZERO vs RAINY DAYS ANALYSIS] - Test Set")
    zero_metrics = calculate_zero_precipitation_metrics(y_test, y_pred_test)
    report.append(f"Zero precipitation days: {zero_metrics['zero_days_count']} ({zero_metrics['zero_days_ratio']*100:.1f}%)")
    report.append(f"Rainy days (≥1mm): {zero_metrics['rainy_days_count']}")
    if 'zero_days_mae' in zero_metrics:
        report.append(f"  Zero days MAE: {zero_metrics['zero_days_mae']:.4f} mm")
        report.append(f"  Zero days RMSE: {zero_metrics['zero_days_rmse']:.4f} mm")
    if 'rainy_days_mae' in zero_metrics:
        report.append(f"  Rainy days MAE: {zero_metrics['rainy_days_mae']:.4f} mm")
        report.append(f"  Rainy days RMSE: {zero_metrics['rainy_days_rmse']:.4f} mm")
        report.append(f"  Rainy days R²: {zero_metrics['rainy_days_r2']:.4f}")

    # Cluster-wise performance
    if cluster_labels is not None and len(cluster_labels) == len(y_test):
        report.append("\n[PERFORMANCE BY CLUSTER] - Test Set")
        unique_clusters = sorted(np.unique(cluster_labels))

        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            if np.sum(mask) > 0:
                y_true_cluster = y_test[mask]
                y_pred_cluster = y_pred_test[mask]

                mae = mean_absolute_error(y_true_cluster, y_pred_cluster)
                rmse = np.sqrt(mean_squared_error(y_true_cluster, y_pred_cluster))
                r2 = r2_score(y_true_cluster, y_pred_cluster)

                report.append(f"\n  Cluster {cluster_id} ({np.sum(mask)} samples):")
                report.append(f"    MAE:  {mae:.4f} mm")
                report.append(f"    RMSE: {rmse:.4f} mm")
                report.append(f"    R²:   {r2:.4f}")

    # Precipitation statistics
    report.append("\n[PRECIPITATION STATISTICS] - Test Set")
    report.append(f"Actual min:     {y_test.min():.2f} mm")
    report.append(f"Actual max:     {y_test.max():.2f} mm")
    report.append(f"Actual mean:    {y_test.mean():.2f} mm")
    report.append(f"Actual std:     {y_test.std():.2f} mm")
    report.append(f"Predicted min:  {y_pred_test.min():.2f} mm")
    report.append(f"Predicted max:  {y_pred_test.max():.2f} mm")
    report.append(f"Predicted mean: {y_pred_test.mean():.2f} mm")
    report.append(f"Predicted std:  {y_pred_test.std():.2f} mm")

    report.append("\n" + "=" * 80)

    return '\n'.join(report)

