"""Small entry point for the LSTM cluster experiment."""

from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from config import DATA_ROOT, load_output_config, output_root_from_config
from methods.lstm_cluster.pipeline import run_experiment


# Station and data selection
STATE = "RS"
STATION_ID = "A801"

# Data setting
WINDOW_SIZES = [25]
WINDOW_STRIDE = 10  # Days between consecutive window starts;
FORECAST_HORIZON = 5
USE_ALL_FEATURES = True


#PCA settings
PCA_VARIANCE_THRESHOLD = None
PCA_FOR_CLUSTERING_ONLY = True  # Keep pre-PCA window features as LSTM inputs


#Normalization settings
CLUSTERING_FEATURE_NORMALIZE = 'minmax'  # "standard", "minmax", or None
CLUSTERING_PRECIPITATION_NORMALIZE = None  # "standard", "minmax", or None
LSTM_FEATURE_NORMALIZE = "minmax"  # "standard", "minmax", or None
LSTM_PRECIPITATION_NORMALIZE = None  # None keeps PRECIPITACAO_TOTAL and LSTM targets in mm



# Clustering setting
N_CLUSTERS_LIST = [3]
CLUSTERING_ALGORITHM = "spectral"  # "kmeans", "spectral", or "manual"
MANUAL_CLUSTERING_METHOD = "rain_level"  # "legacy" or "rain_level"
MANUAL_ZERO_TOLERANCE = 0.0  # Used only by legacy manual clustering
CLUSTER_ASSIGNMENT_METHOD = "knn"  # "centroid" or "knn"
CLUSTER_ASSIGNMENT_NEIGHBORS = 5  # Used only when assignment method is "knn"
N_SIGMA_VALUES = 5
SIGMA_MODE = "manual"  # "auto" or "manual"
MANUAL_SIGMA_VALUES = [0.1,0.3,0.5,1,5,10,100]  # Only used if SIGMA_MODE is "manual"



# Metrics exported to compact comparison tables
QUANTITATIVE_METRICS = ["MSE"]

# Sweep-level comparison between the tests produced by this run.
COMPARATIVE_RUN = False
PIVOT_PARAMETER = "LSTM_UNITS"  # e.g. "window_size", "learning_rate", "K", "sigma"

# Optional oracle diagnostic: evaluates every test window with every cluster LSTM.
# It never replaces the same-cluster test metrics or the main plots.
TEST_ALL_MODELS = True

# Model hyperparameters. Each numeric setting accepts one scalar or a sweep list.
LSTM_UNITS: int | list[int] = 64
LSTM_UNITS_2: int | list[int] = 32
DROPOUT_RATE: float | list[float] = 0.3
LEARNING_RATE: float | list[float] = 1e-3
WEIGHT_DECAY: float | list[float] = 1e-4  # Decoupled weight decay used by AdamW
# Supported: "mean_squared_error", "mae", "huber", "weighted_mse_loss",
# or "quantile_weighted_mse".
LSTM_LOSS_FUNCTION = "weighted_mse_loss"
LOSS_ALPHA = 1e-2  # Positive coefficient used only by weighted_mse_loss
LOSS_QUANTILES = [0.95]
LOSS_QUANTILE_WEIGHTS = "auto"  # "auto" or one positive weight per quantile bin

# Training settings. Numeric settings may also be lists in a comparative grid.
EPOCHS: int | list[int] = 150
BATCH_SIZE: int | list[int] = 12
EARLY_STOPPING = True
PATIENCE: int | list[int] = 20
EARLY_STOPPING_METRIC = "loss"  # "loss", "mse", "mae", or "r2"
VERBOSE_TRAINING = 1
SHOW_CONSOLE_INFO = True
RUN_ONLY_CLUSTER = True  # Only cluster windows; skip all LSTM training/output.

# Train/validation/test split
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
RANDOM_STATE = 42

# Output settings
OUTPUT_CONFIG = Path(__file__).with_name("config_output.yaml")


def main() -> None:
    """Run the experiment using the variables above and config_output.yaml."""
    sigma_mode = SIGMA_MODE.lower()
    if sigma_mode not in {"auto", "manual"}:
        raise ValueError("SIGMA_MODE must be either 'auto' or 'manual'.")

    output_config = load_output_config(OUTPUT_CONFIG)
    run_experiment(
        state=STATE,
        station_id=STATION_ID,
        window_sizes=WINDOW_SIZES,
        window_stride=WINDOW_STRIDE,
        clustering_feature_normalize=CLUSTERING_FEATURE_NORMALIZE,
        clustering_precipitation_normalize=CLUSTERING_PRECIPITATION_NORMALIZE,
        lstm_feature_normalize=LSTM_FEATURE_NORMALIZE,
        lstm_precipitation_normalize=LSTM_PRECIPITATION_NORMALIZE,
        variance_threshold=PCA_VARIANCE_THRESHOLD,
        pca_for_clustering_only=PCA_FOR_CLUSTERING_ONLY,
        run_only_cluster=RUN_ONLY_CLUSTER,
        n_clusters_list=N_CLUSTERS_LIST,
        clustering_algorithm=CLUSTERING_ALGORITHM,
        manual_clustering_method=MANUAL_CLUSTERING_METHOD,
        manual_zero_tolerance=MANUAL_ZERO_TOLERANCE,
        cluster_assignment_method=CLUSTER_ASSIGNMENT_METHOD,
        cluster_assignment_neighbors=CLUSTER_ASSIGNMENT_NEIGHBORS,
        n_sigma_values=N_SIGMA_VALUES,
        sigma_values=MANUAL_SIGMA_VALUES if sigma_mode == "manual" else None,
        use_all_features=USE_ALL_FEATURES,
        forecast_horizon=FORECAST_HORIZON,
        quantitative_metrics=QUANTITATIVE_METRICS,
        lstm_units=LSTM_UNITS,
        lstm_units_2=LSTM_UNITS_2,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
        test_all_models=TEST_ALL_MODELS,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        early_stopping=EARLY_STOPPING,
        patience=PATIENCE,
        early_stopping_metric=EARLY_STOPPING_METRIC,
        lstm_loss_function=LSTM_LOSS_FUNCTION,
        loss_alpha=LOSS_ALPHA,
        loss_quantiles=LOSS_QUANTILES,
        loss_quantile_weights=LOSS_QUANTILE_WEIGHTS,
        verbose_training=VERBOSE_TRAINING,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        random_state=RANDOM_STATE,
        data_root=DATA_ROOT,
        output_root=output_root_from_config(output_config),
        sweep_name=output_config.get("sweep_name") or None,
        sweep_name_prefix=str(output_config.get("sweep_name_prefix", "lstm_cluster_sweep")),
        timestamp_format=str(output_config.get("timestamp_format", "%Y%m%d_%H%M%S")),
        plot_style=output_config.get("plot_style"),
        show_console_info=SHOW_CONSOLE_INFO,
        comparative_run=COMPARATIVE_RUN,
        pivot_parameter=PIVOT_PARAMETER,
    )


if __name__ == "__main__":
    main()
