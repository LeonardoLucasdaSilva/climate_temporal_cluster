"""LSTM Clustering Experiment for RS A801 Station.

This script implements a complete machine learning pipeline for precipitation prediction:

1. **Data Loading & Preparation**: Loads RS A801 station data
2. **Feature Extraction**: Creates sliding windows from daily climate data
3. **Clustering**: Applies the configured clustering algorithm to segment weather patterns
4. **Train/Val/Test Split**: Stratifies data by cluster (70% train, 10% val, 20% test)
5. **LSTM Training**: Trains separate LSTM models for each cluster
6. **Evaluation**: Comprehensive quantitative and visual analysis

The model predicts next-day precipitation from window features within each cluster,
allowing for cluster-specific weather pattern prediction.

Author: Climate Cluster Project
Date: 2024
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from climate_cluster.config import DATA_ROOT, OUTPUTS_DIR
from climate_cluster.config_data import load_single_station
from clustering_protocol import (
    PCA_VARIANCE_THRESHOLD,
    cluster_feature_matrix,
    create_cluster_feature_matrix,
    numeric_feature_columns,
)
from climate_cluster.pipeline.lstm import LSTMPrecipitationPredictor
from climate_cluster.evaluation.metrics import (
    calculate_regression_metrics,
    calculate_zero_precipitation_metrics,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_error_by_magnitude,
    plot_cluster_performance,
    create_evaluation_report,
)


# ==================== CONFIGURATION ====================

# Station configuration
STATE = "RS"
STATION_ID = "A801"

# Clustering parameters
WINDOW_SIZE = 16  # Days per window
N_CLUSTERS = 7  # Number of clusters
CLUSTERING_ALGORITHM = "kmeans"  # Options: "kmeans", "spectral"
SIGMA = 0.1  # Used only when CLUSTERING_ALGORITHM = "spectral"
USE_ALL_FEATURES = True  # Use all available features or selected subset

# LSTM parameters
LSTM_UNITS = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001

# Training parameters
EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOPPING = True
PATIENCE = 10

# Data split (train, val, test)
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
TEST_RATIO = 0.3

# Random seed for reproducibility
RANDOM_STATE = 42

# Output directory
EXPERIMENT_NAME = f"lstm_cluster_{STATE}_{STATION_ID}_w{WINDOW_SIZE}_k{N_CLUSTERS}"
OUTPUT_DIR = OUTPUTS_DIR / EXPERIMENT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== HELPER FUNCTIONS ====================

def setup_styling():
    """Configure matplotlib and seaborn styling."""
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9


def split_by_cluster(X, y, cluster_labels, train_ratio=0.7, val_ratio=0.1, random_state=42):
    """Split data maintaining cluster stratification.

    Ensures each cluster has samples in train, validation, and test sets.

    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,)
        cluster_labels: Cluster assignment for each sample
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test,
                  cluster_train, cluster_val, cluster_test)
    """
    # First split: train+val vs test
    X_tv, X_test, y_tv, y_test, c_tv, c_test = train_test_split(
        X, y, cluster_labels,
        test_size=1 - train_ratio - val_ratio,
        stratify=cluster_labels,
        random_state=random_state,
    )

    # Second split: train vs val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val, c_train, c_val = train_test_split(
        X_tv, y_tv, c_tv,
        test_size=val_ratio_adjusted,
        stratify=c_tv,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, c_train, c_val, c_test


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    """Run the complete LSTM clustering experiment."""

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + f"  LSTM CLUSTERING EXPERIMENT - RS A801 STATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    setup_styling()

    # ==================== STEP 1: DATA LOADING ====================
    print_section("STEP 1: LOADING STATION DATA")

    print(f"\nLoading {STATE}/{STATION_ID} station data...")
    df = load_single_station(state=STATE, station_id=STATION_ID, data_root=DATA_ROOT)
    print(f"✓ Loaded {len(df)} days of data")
    print(f"  Date range: {df['Data'].min().date()} to {df['Data'].max().date()}")

    # Identify available numeric features
    numeric_cols = numeric_feature_columns(df)
    print(f"  Available features: {numeric_cols}")

    # ==================== STEP 2: FEATURE EXTRACTION ====================
    print_section("STEP 2: CREATING WINDOWS & EXTRACTING FEATURES")

    print(f"\nCreating windows with size: {WINDOW_SIZE} days")
    print(f"Using features: {numeric_cols if USE_ALL_FEATURES else 'Selected subset'}")

    # Create windows using the same feature protocol as the parameter sweep.
    columns = numeric_cols if USE_ALL_FEATURES else None
    windows, windows_flat, scaler, pca, feature_columns = create_cluster_feature_matrix(
        df,
        window_size=WINDOW_SIZE,
        columns=columns,
        normalize=True,
        variance_threshold=PCA_VARIANCE_THRESHOLD,
    )

    print(f"✓ Created {len(windows)} windows")
    print(f"  Window shape: {windows.shape}")

    print(f"  Flattened shape: {windows_flat.shape}")

    # ==================== STEP 3: CLUSTERING ====================
    print_section("STEP 3: CLUSTERING")

    print(f"\nApplying {CLUSTERING_ALGORITHM.upper()} clustering...")
    print(f"  Number of clusters: {N_CLUSTERS}")
    print(f"  PCA variance threshold: {PCA_VARIANCE_THRESHOLD:.2f}")
    if CLUSTERING_ALGORITHM.lower() == "spectral":
        print(f"  Sigma: {SIGMA}")
    else:
        print(f"  Sigma: not used by {CLUSTERING_ALGORITHM.upper()}")

    cluster_labels = cluster_feature_matrix(
        windows_flat,
        n_clusters=N_CLUSTERS,
        algorithm=CLUSTERING_ALGORITHM,
        sigma=SIGMA,
        random_state=RANDOM_STATE,
    )

    print(f"✓ Clustering complete")
    print(f"  Cluster distribution:")
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique_clusters, counts):
        print(f"    Cluster {cluster_id}: {count} windows ({100*count/len(cluster_labels):.1f}%)")

    # ==================== STEP 4: PREPARE TARGET VALUES ====================
    print_section("STEP 4: PREPARING TARGET VALUES")

    print(f"\nExtracting next-day precipitation values...")

    # Target values: precipitation on the day after each window ends
    # Window i spans days [i, i+1, ..., i+window_size-1]
    # Target is precipitation on day i+window_size
    target_values = []
    valid_indices = []

    for window_idx in range(len(windows)):
        target_day_idx = window_idx + WINDOW_SIZE
        if target_day_idx < len(df):
            precip = df.iloc[target_day_idx]['PRECIPITACAO_TOTAL']
            target_values.append(precip)
            valid_indices.append(window_idx)

    target_values = np.array(target_values)
    valid_indices = np.array(valid_indices)

    # Keep only valid samples
    windows_valid = windows_flat[valid_indices]
    cluster_labels_valid = cluster_labels[valid_indices]

    print(f"✓ Extracted target values")
    print(f"  Valid samples: {len(target_values)} / {len(windows)}")
    print(f"  Precipitation statistics:")
    print(f"    Min:   {target_values.min():.2f} mm")
    print(f"    Max:   {target_values.max():.2f} mm")
    print(f"    Mean:  {target_values.mean():.2f} mm")
    print(f"    Std:   {target_values.std():.2f} mm")
    print(f"    Zero days: {np.sum(target_values == 0)} ({100*np.sum(target_values == 0)/len(target_values):.1f}%)")

    # ==================== STEP 5: DATA SPLITTING ====================
    print_section("STEP 5: TRAIN/VALIDATION/TEST SPLIT")

    print(f"\nSplitting data with stratification by cluster...")
    print(f"  Train: {TRAIN_RATIO*100:.0f}%, Val: {VAL_RATIO*100:.0f}%, Test: {TEST_RATIO*100:.0f}%")

    X_train, X_val, X_test, y_train, y_val, y_test, c_train, c_val, c_test = split_by_cluster(
        windows_valid,
        target_values,
        cluster_labels_valid,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        random_state=RANDOM_STATE,
    )

    print(f"✓ Data split complete")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Test set: {len(X_test)} samples")

    # Prepare sequences for LSTM (reshape to 3D: samples, timesteps, features)
    # For now, treat each flattened window as a single timestep with multiple features
    print(f"\nPreparing sequences for LSTM...")
    X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val_lstm = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    print(f"  Training sequence shape: {X_train_lstm.shape}")
    print(f"  Validation sequence shape: {X_val_lstm.shape}")
    print(f"  Test sequence shape: {X_test_lstm.shape}")

    # ==================== STEP 6: LSTM MODEL TRAINING ====================
    print_section("STEP 6: TRAINING LSTM MODEL (PER CLUSTER)")

    print(f"\nModel hyperparameters:")
    print(f"  LSTM units: {LSTM_UNITS}")
    print(f"  LSTM units (2nd layer): {LSTM_UNITS_2}")
    print(f"  Dropout rate: {DROPOUT_RATE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Early stopping: {EARLY_STOPPING} (patience: {PATIENCE})")

    unique_train_clusters = np.unique(c_train)
    print(f"\nClusters present in training set: {unique_train_clusters.tolist()}")

    # Prepare containers for aggregated predictions and artifacts
    y_pred_train = np.zeros_like(y_train, dtype=float)
    y_pred_val = np.zeros_like(y_val, dtype=float)
    y_pred_test = np.zeros_like(y_test, dtype=float)

    histories_by_cluster = {}
    models_by_cluster = {}
    metrics_by_cluster = {}

    for cluster_id in unique_train_clusters:
        print(f"\n--- Training model for Cluster {cluster_id} ---")

        # Masks for this cluster
        tr_mask = (c_train == cluster_id)
        va_mask = (c_val == cluster_id)
        te_mask = (c_test == cluster_id)

        n_tr, n_va, n_te = tr_mask.sum(), va_mask.sum(), te_mask.sum()
        print(f"  Samples -> train: {n_tr}, val: {n_va}, test: {n_te}")

        if n_tr == 0:
            print(f"  Skipping cluster {cluster_id}: no training samples.")
            continue

        # Build model for this cluster
        model_c = LSTMPrecipitationPredictor(
            input_shape=(1, X_train_lstm.shape[2]),
            lstm_units=LSTM_UNITS,
            lstm_units_2=LSTM_UNITS_2,
            dropout_rate=DROPOUT_RATE,
            learning_rate=LEARNING_RATE,
            random_state=RANDOM_STATE,
        )

        # Fit model on cluster-specific data
        history_c = model_c.fit(
            X_train_lstm[tr_mask],
            y_train[tr_mask],
            X_val=X_val_lstm[va_mask] if n_va > 0 else None,
            y_val=y_val[va_mask] if n_va > 0 else None,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            early_stopping=EARLY_STOPPING and (n_va > 0),
            patience=PATIENCE,
        )

        histories_by_cluster[int(cluster_id)] = history_c
        models_by_cluster[int(cluster_id)] = model_c

        # Predict and fill aggregated arrays
        y_pred_train[tr_mask] = model_c.predict(X_train_lstm[tr_mask]).flatten()
        if n_va > 0:
            y_pred_val[va_mask] = model_c.predict(X_val_lstm[va_mask]).flatten()
        if n_te > 0:
            y_pred_test[te_mask] = model_c.predict(X_test_lstm[te_mask]).flatten()

        # Compute test metrics for this cluster (if test samples exist)
        if n_te > 0:
            m_test_c = calculate_regression_metrics(y_test[te_mask], y_pred_test[te_mask])
            metrics_by_cluster[int(cluster_id)] = m_test_c
            print(f"  Cluster {cluster_id} Test Metrics: "
                  f"RMSE={m_test_c['RMSE']:.4f}, MAE={m_test_c['MAE']:.4f}, R2={m_test_c['R2']:.4f}")

    print(f"\n✓ Training complete for all clusters")

    # ==================== STEP 7: PREDICTIONS ====================
    print_section("STEP 7: GENERATING PREDICTIONS")

    print("Predictions per cluster have been generated during the training loop.")
    # y_pred_train, y_pred_val, y_pred_test are ready as 1D arrays

    # ==================== STEP 8: EVALUATION ====================
    print_section("STEP 8: MODEL EVALUATION")

    # Calculate metrics
    metrics_train = calculate_regression_metrics(y_train, y_pred_train)
    metrics_val = calculate_regression_metrics(y_val, y_pred_val)
    metrics_test = calculate_regression_metrics(y_test, y_pred_test)

    print(f"\nTraining Set Metrics:")
    for key, value in metrics_train.items():
        if key == 'MAPE':
            print(f"  {key:8s}: {value:.2f}%")
        else:
            print(f"  {key:8s}: {value:.6f}")

    print(f"\nValidation Set Metrics:")
    for key, value in metrics_val.items():
        if key == 'MAPE':
            print(f"  {key:8s}: {value:.2f}%")
        else:
            print(f"  {key:8s}: {value:.6f}")

    print(f"\nTest Set Metrics (PRIMARY EVALUATION):")
    for key, value in metrics_test.items():
        if key == 'MAPE':
            print(f"  {key:8s}: {value:.2f}%")
        else:
            print(f"  {key:8s}: {value:.6f}")

    # Zero vs Rainy day analysis
    print(f"\nZero vs Rainy Days Analysis (Test Set):")
    zero_metrics = calculate_zero_precipitation_metrics(y_test, y_pred_test)
    print(f"  Zero days: {zero_metrics['zero_days_count']} ({zero_metrics['zero_days_ratio']*100:.1f}%)")
    print(f"  Rainy days: {zero_metrics['rainy_days_count']}")
    if 'zero_days_mae' in zero_metrics:
        print(f"    Zero days MAE: {zero_metrics['zero_days_mae']:.4f} mm")
        print(f"    Zero days RMSE: {zero_metrics['zero_days_rmse']:.4f} mm")
    if 'rainy_days_mae' in zero_metrics:
        print(f"    Rainy days MAE: {zero_metrics['rainy_days_mae']:.4f} mm")
        print(f"    Rainy days RMSE: {zero_metrics['rainy_days_rmse']:.4f} mm")

    # ==================== STEP 9: VISUALIZATIONS ====================
    print_section("STEP 9: GENERATING VISUALIZATIONS")

    print(f"\nGenerating evaluation plots...")

    # 1. Training history (one figure per cluster)
    for cluster_id, history_c in histories_by_cluster.items():
        hist = history_c.history
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Loss
        axes[0].plot(hist.get('loss', []), label='Training Loss', linewidth=2)
        if 'val_loss' in hist:
            axes[0].plot(hist['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title(f'Cluster {cluster_id}: Training History - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE
        axes[1].plot(hist.get('mae', []), label='Training MAE', linewidth=2)
        if 'val_mae' in hist:
            axes[1].plot(hist['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE (mm)')
        axes[1].set_title(f'Cluster {cluster_id}: Training History - MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = OUTPUT_DIR / f"01_training_history_cluster_{cluster_id}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {fname.name}")
        plt.close()

    # 2. Predictions vs Actual (Test Set)
    fig, axes = plot_predictions_vs_actual(
        y_test, y_pred_test,
        title="Test Set: Predictions vs Actual Precipitation"
    )
    plt.savefig(OUTPUT_DIR / "02_predictions_vs_actual.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 02_predictions_vs_actual.png")
    plt.close()

    # 3. Residuals analysis
    fig, axes = plot_residuals(
        y_test, y_pred_test,
        title="Test Set: Residual Analysis"
    )
    plt.savefig(OUTPUT_DIR / "03_residuals_analysis.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 03_residuals_analysis.png")
    plt.close()

    # 4. Error by magnitude
    fig, axes = plot_error_by_magnitude(
        y_test, y_pred_test,
        n_bins=10,
        title="Test Set: Error Analysis by Precipitation Magnitude"
    )
    plt.savefig(OUTPUT_DIR / "04_error_by_magnitude.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 04_error_by_magnitude.png")
    plt.close()

    # 5. Performance by cluster
    fig, axes = plot_cluster_performance(
        c_test, y_test, y_pred_test,
        title="Test Set: LSTM Performance by Cluster"
    )
    plt.savefig(OUTPUT_DIR / "05_cluster_performance.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 05_cluster_performance.png")
    plt.close()

    # 6. Cluster distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    unique_clusters, counts = np.unique(c_test, return_counts=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
    ax.bar(unique_clusters, counts, color=colors, alpha=0.7)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Test Set: Cluster Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (cluster_id, count) in enumerate(zip(unique_clusters, counts)):
        ax.text(cluster_id, count, str(count), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_cluster_distribution.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: 06_cluster_distribution.png")
    plt.close()

    # ==================== STEP 10: SAVE RESULTS ====================
    print_section("STEP 10: SAVING RESULTS")

    # Save evaluation report
    report = create_evaluation_report(
        y_train, y_val, y_test,
        y_pred_train, y_pred_val, y_pred_test,
        c_test
    )

    report_path = OUTPUT_DIR / "evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✓ Saved evaluation report: {report_path.name}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Set': ['Train', 'Validation', 'Test'],
        'MSE': [metrics_train['MSE'], metrics_val['MSE'], metrics_test['MSE']],
        'RMSE': [metrics_train['RMSE'], metrics_val['RMSE'], metrics_test['RMSE']],
        'MAE': [metrics_train['MAE'], metrics_val['MAE'], metrics_test['MAE']],
        'RMSLE': [metrics_train['RMSLE'], metrics_val['RMSLE'], metrics_test['RMSLE']],
        'R2': [metrics_train['R2'], metrics_val['R2'], metrics_test['R2']],
    })

    metrics_path = OUTPUT_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Saved metrics: {metrics_path.name}")

    # Save predictions
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred_test,
        'residual': y_test - y_pred_test,
        'cluster': c_test,
    })

    predictions_path = OUTPUT_DIR / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Saved test predictions: {predictions_path.name}")

    # Save experiment configuration
    config_path = OUTPUT_DIR / "experiment_config.txt"
    with open(config_path, 'w') as f:
        f.write("LSTM CLUSTERING EXPERIMENT CONFIGURATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Station: {STATE}/{STATION_ID}\n")
        f.write(f"Window size: {WINDOW_SIZE} days\n")
        f.write(f"Number of clusters: {N_CLUSTERS}\n")
        f.write(f"Clustering method: {CLUSTERING_ALGORITHM.upper()}\n")
        if CLUSTERING_ALGORITHM.lower() == "spectral":
            f.write(f"Sigma: {SIGMA}\n")
        else:
            f.write(f"Sigma: not used by {CLUSTERING_ALGORITHM.upper()}\n")
        f.write(f"PCA: variance_threshold={PCA_VARIANCE_THRESHOLD:.2f} applied before clustering\n")
        f.write(f"Features: {len(feature_columns)} columns ({', '.join(feature_columns)})\n")
        f.write(f"Per-cluster LSTM training: Yes (one model per cluster)\n")
        f.write(f"\nLSTM Model (per cluster):\n")
        f.write(f"  LSTM units: {LSTM_UNITS}\n")
        f.write(f"  LSTM units (2nd): {LSTM_UNITS_2}\n")
        f.write(f"  Dropout rate: {DROPOUT_RATE}\n")
        f.write(f"  Learning rate: {LEARNING_RATE}\n")
        f.write(f"  Epochs: {EPOCHS}\n")
        f.write(f"  Batch size: {BATCH_SIZE}\n")
        f.write(f"  Early stopping: {EARLY_STOPPING}\n")
        f.write(f"\nData Split (final proportions):\n")
        f.write(f"  Training: {TRAIN_RATIO*100:.0f}% ({len(X_train)} samples)\n")
        f.write(f"  Validation: {VAL_RATIO*100:.0f}% ({len(X_val)} samples)\n")
        f.write(f"  Test: {TEST_RATIO*100:.0f}% ({len(X_test)} samples)\n")
        f.write(f"\nExperiment timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"✓ Saved experiment configuration: {config_path.name}")

    # ==================== FINAL SUMMARY ====================
    print_section("EXPERIMENT COMPLETE")

    print(f"\n✓ All results saved to: {OUTPUT_DIR}")
    print(f"\nKey Results (Test Set):")
    print(f"  RMSE: {metrics_test['RMSE']:.4f} mm")
    print(f"  MAE:  {metrics_test['MAE']:.4f} mm")
    print(f"  R²:   {metrics_test['R2']:.4f}")
    print(f"\nOutput Files:")
    print(f"  - 01_training_history.png: Model training curves")
    print(f"  - 02_predictions_vs_actual.png: Time series and scatter plots")
    print(f"  - 03_residuals_analysis.png: Residual analysis")
    print(f"  - 04_error_by_magnitude.png: Error by precipitation magnitude")
    print(f"  - 05_cluster_performance.png: Performance metrics by cluster")
    print(f"  - 06_cluster_distribution.png: Cluster distribution in test set")
    print(f"  - evaluation_report.txt: Detailed evaluation report")
    print(f"  - metrics.csv: Metrics for train/val/test sets")
    print(f"  - test_predictions.csv: Predictions with residuals")
    print(f"  - experiment_config.txt: Full experiment configuration")

    print("\n" + "=" * 80)
    print("Thank you for using LSTM Clustering Experiment!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

