"""Evaluation metrics and text reports for LSTM models."""

from __future__ import annotations

from typing import Dict

import numpy as np
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

