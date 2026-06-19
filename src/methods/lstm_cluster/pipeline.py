"""Sweep LSTM-by-cluster precipitation experiments for RS A801.

For each configuration, this script:
1. Builds climate windows and clusters them.
2. Trains one LSTM model per cluster.
3. Saves metrics, predictions, summaries, and diagnostic plots.

After all configurations finish, it also writes a LaTeX table that can be
copied directly into Overleaf.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from config import DATA_ROOT, OUTPUTS_DIR
from data.load_data import load_station_daily_data
from data.lstm_outputs import save_run_outputs, save_sweep_outputs
from evaluation.metrics import calculate_regression_metrics
from models.lstm import LSTMPrecipitationPredictor
from methods.cluster.cluster_pipeline import (
    PCA_VARIANCE_THRESHOLD,
    cluster_feature_matrix,
    create_cluster_feature_matrix,
    numeric_feature_columns,
)
from methods.tools.sigma_choosing import calculate_sigma_values


# ==================== CONFIGURATION ====================

STATE = "RS"
STATION_ID = "A801"

# Edit these lists to choose the sweep. LSTM sweeps can become expensive quickly.
WINDOW_SIZES = [8, 12, 16, 20, 24, 28]
N_CLUSTERS_LIST = [3,4,5]
CLUSTERING_ALGORITHM = "spectral"  # Options: "kmeans", "spectral"
#SIGMA_VALUES = [0.1]
N_SIGMA_VALUES = 5  # Used only by spectral clustering.
USE_ALL_FEATURES = True

# Metrics included in the cluster-level Overleaf tables.
# Available options from calculate_regression_metrics: MSE, RMSE, MAE, RMSLE, R2, MAPE.
#QUANTITATIVE_METRICS = ["MSE", "RMSE", "MAE", "R2"]
QUANTITATIVE_METRICS = ["MSE"]

LSTM_UNITS = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001

EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOPPING = True
PATIENCE = 10
VERBOSE_TRAINING = 1

TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
TEST_RATIO = 0.3
RANDOM_STATE = 42

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SWEEP_NAME = f"lstm_cluster_sweep_{STATE}_{STATION_ID}_{RUN_TIMESTAMP}"
SWEEP_DIR = OUTPUTS_DIR / SWEEP_NAME


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters that uniquely define one experiment run."""

    window_size: int
    n_clusters: int
    algorithm: str
    sigma: float | None

    @property
    def name(self) -> str:
        sigma_part = "sigma_na" if self.sigma is None else f"sigma_{self.sigma:g}"
        return (
            f"{STATE}_{STATION_ID}_w{self.window_size:02d}_"
            f"k{self.n_clusters:02d}_{self.algorithm}_{sigma_part}"
        ).replace(".", "p")


def setup_styling() -> None:
    """Configure plotting defaults."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9


def print_section(title: str) -> None:
    """Print a compact section header."""
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def build_configurations(sigmas: list[float | None]) -> list[ExperimentConfig]:
    """Build the sweep grid."""
    #sigmas = SIGMA_VALUES if CLUSTERING_ALGORITHM.lower() == "spectral" else [None]
    return [
        ExperimentConfig(
            window_size=window_size,
            n_clusters=n_clusters,
            algorithm=CLUSTERING_ALGORITHM.lower(),
            sigma=sigma,
        )
        for window_size in WINDOW_SIZES
        for n_clusters in N_CLUSTERS_LIST
        for sigma in sigmas
    ]


def split_by_cluster(
    X: np.ndarray,
    y: np.ndarray,
    cluster_labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, ...]:
    """Split data, stratifying by cluster whenever every cluster has enough rows."""
    test_ratio = 1 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("TRAIN_RATIO + VAL_RATIO must be smaller than 1.")

    counts = pd.Series(cluster_labels).value_counts()
    stratify = cluster_labels if counts.min() >= 3 else None

    X_tv, X_test, y_tv, y_test, c_tv, c_test = train_test_split(
        X,
        y,
        cluster_labels,
        test_size=test_ratio,
        stratify=stratify,
        random_state=random_state,
    )

    tv_counts = pd.Series(c_tv).value_counts()
    stratify_tv = c_tv if len(tv_counts) > 0 and tv_counts.min() >= 2 else None
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val, c_train, c_val = train_test_split(
        X_tv,
        y_tv,
        c_tv,
        test_size=val_ratio_adjusted,
        stratify=stratify_tv,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, c_train, c_val, c_test


def next_day_precipitation_targets(
    df: pd.DataFrame,
    window_size: int,
    n_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid window indices and next-day precipitation targets."""
    valid_indices = np.arange(min(n_windows, len(df) - window_size))
    target_indices = valid_indices + window_size
    targets = df.iloc[target_indices]["PRECIPITACAO_TOTAL"].to_numpy(dtype=float)
    return valid_indices, targets


def to_lstm_shape(X: np.ndarray) -> np.ndarray:
    """Represent one flattened window as one LSTM timestep."""
    return X.reshape(X.shape[0], 1, X.shape[1])


def clipped_predictions(model: LSTMPrecipitationPredictor, X: np.ndarray) -> np.ndarray:
    """Predict precipitation and enforce the physical non-negative lower bound."""
    return np.maximum(model.predict(X).ravel(), 0.0)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, object], dict[int, dict[str, float]]]:
    """Train one LSTM per cluster and return aggregate predictions."""
    X_train_lstm = to_lstm_shape(X_train)
    X_val_lstm = to_lstm_shape(X_val)
    X_test_lstm = to_lstm_shape(X_test)

    y_pred_train = np.zeros_like(y_train, dtype=float)
    y_pred_val = np.zeros_like(y_val, dtype=float)
    y_pred_test = np.zeros_like(y_test, dtype=float)
    histories_by_cluster: dict[int, object] = {}
    metrics_by_cluster: dict[int, dict[str, float]] = {}

    for cluster_id in sorted(np.unique(c_train)):
        tr_mask = c_train == cluster_id
        va_mask = c_val == cluster_id
        te_mask = c_test == cluster_id
        n_tr, n_va, n_te = tr_mask.sum(), va_mask.sum(), te_mask.sum()

        print(f"  Cluster {cluster_id}: train={n_tr}, val={n_va}, test={n_te}")
        if n_tr == 0:
            continue

        model = LSTMPrecipitationPredictor(
            input_shape=(1, X_train_lstm.shape[2]),
            lstm_units=LSTM_UNITS,
            lstm_units_2=LSTM_UNITS_2,
            dropout_rate=DROPOUT_RATE,
            learning_rate=LEARNING_RATE,
            random_state=RANDOM_STATE,
        )
        history = model.fit(
            X_train_lstm[tr_mask],
            y_train[tr_mask],
            X_val=X_val_lstm[va_mask] if n_va > 0 else None,
            y_val=y_val[va_mask] if n_va > 0 else None,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=VERBOSE_TRAINING,
            early_stopping=EARLY_STOPPING and n_va > 0,
            patience=PATIENCE,
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

    return y_pred_train, y_pred_val, y_pred_test, histories_by_cluster, metrics_by_cluster


def run_configuration(
    df: pd.DataFrame,
    config: ExperimentConfig,
    numeric_cols: list[str],
    output_dir: Path,
) -> dict[str, float | int | str | None]:
    """Run one full LSTM-clustering experiment."""
    print_section(f"Running {config.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = numeric_cols if USE_ALL_FEATURES else None
    windows, windows_flat, _, _, feature_columns = create_cluster_feature_matrix(
        df,
        window_size=config.window_size,
        columns=columns,
        normalize=True,
        variance_threshold=PCA_VARIANCE_THRESHOLD,
    )
    labels = cluster_feature_matrix(
        windows_flat,
        n_clusters=config.n_clusters,
        algorithm=config.algorithm,
        sigma=config.sigma,
        random_state=RANDOM_STATE,
    )
    valid_indices, targets = next_day_precipitation_targets(
        df,
        config.window_size,
        len(windows),
    )

    X = windows_flat[valid_indices]
    c = labels[valid_indices]
    X_train, X_val, X_test, y_train, y_val, y_test, c_train, c_val, c_test = split_by_cluster(
        X,
        targets,
        c,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        random_state=RANDOM_STATE,
    )

    print(
        f"  Windows={len(windows)}, samples={len(targets)}, "
        f"features={X.shape[1]}, clusters={sorted(np.unique(c).tolist())}"
    )
    print(f"  Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    (
        y_pred_train,
        y_pred_val,
        y_pred_test,
        histories_by_cluster,
        metrics_by_cluster,
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
    )

    result = save_run_outputs(
        config,
        output_dir,
        feature_columns,
        y_train,
        y_val,
        y_test,
        y_pred_train,
        y_pred_val,
        y_pred_test,
        c_test,
        histories_by_cluster,
        metrics_by_cluster,
        state=STATE,
        station_id=STATION_ID,
        pca_variance_threshold=PCA_VARIANCE_THRESHOLD,
    )
    print(
        f"  Test metrics: RMSE={result['test_rmse']:.4f}, "
        f"MAE={result['test_mae']:.4f}, R2={result['test_r2']:.4f}"
    )
    return result


def main() -> None:
    """Run the full sweep."""
    setup_styling()
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    print_section("Loading Data")
    df = load_station_daily_data(state=STATE, station_id=STATION_ID, data_root=DATA_ROOT)
    numeric_cols = numeric_feature_columns(df)
    print(f"Loaded {len(df)} rows from {df['Data'].min().date()} to {df['Data'].max().date()}")
    print(f"Numeric features: {numeric_cols}")

    if CLUSTERING_ALGORITHM.lower() == "spectral":
        print_section("Calculating Sigma Values")
        sigma_values = calculate_sigma_values(df, n_values=N_SIGMA_VALUES).tolist()
        print(f"Generated {len(sigma_values)} sigma values: {sigma_values}")
    else:
        sigma_values = [None]

    configurations = build_configurations(sigma_values)
    print_section("LSTM CLUSTER SWEEP")
    print(f"Station: {STATE}/{STATION_ID}")
    print(f"Output directory: {SWEEP_DIR}")
    print(f"Configurations: {len(configurations)}")
    for config in configurations:
        print(f"  - {config.name}: {asdict(config)}")


    results = []
    for index, config in enumerate(configurations, start=1):
        print(f"\nConfiguration {index}/{len(configurations)}")
        results.append(run_configuration(df, config, numeric_cols, SWEEP_DIR / config.name))

    save_sweep_outputs(
        results,
        sweep_dir=SWEEP_DIR,
        state=STATE,
        station_id=STATION_ID,
        window_sizes=WINDOW_SIZES,
        n_clusters_list=N_CLUSTERS_LIST,
        clustering_algorithm=CLUSTERING_ALGORITHM,
        quantitative_metrics=QUANTITATIVE_METRICS,
    )

    print_section("Sweep Complete")
    print(f"Results folder: {SWEEP_DIR}")
    print("Sweep-level files:")
    print("  - sweep_results.csv")
    print("  - sweep_summary.txt")
    print("  - overleaf_table.txt")
    print("  - overleaf_cluster_metric_tables.txt")
    print("Each configuration folder contains metrics, reports, predictions, and plots.")


if __name__ == "__main__":
    main()
