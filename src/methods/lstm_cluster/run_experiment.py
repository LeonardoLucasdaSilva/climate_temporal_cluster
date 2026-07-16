"""Small entry point for the LSTM cluster experiment."""

from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from config import DATA_ROOT, load_output_config, output_root_from_config
from methods.lstm_cluster.pipeline import run_experiment


# Station and data
STATE = "RS"
STATION_ID = "A801"

# Clustering sweep
WINDOW_SIZES = [35]
N_CLUSTERS_LIST = [5]
PCA_VARIANCE_THRESHOLD = None
PCA_FOR_CLUSTERING_ONLY = True  # Keep pre-PCA window features as LSTM inputs
CLUSTERING_FEATURE_NORMALIZE = 'standard'  # "standard", "minmax", or None
CLUSTERING_PRECIPITATION_NORMALIZE = 'minmax'  # "standard", "minmax", or None
LSTM_FEATURE_NORMALIZE = "standard"  # "standard", "minmax", or None
LSTM_PRECIPITATION_NORMALIZE = None  # None keeps PRECIPITACAO_TOTAL and LSTM targets in mm



# Clustering setting
CLUSTERING_ALGORITHM = "spectral"  # "kmeans" or "spectral"
N_SIGMA_VALUES = 5
SIGMA_MODE = "manual"  # "auto" or "manual"
MANUAL_SIGMA_VALUES = [0.4,0.5,0.6,0.7,0.8,0.9]  # Only used if SIGMA_MODE is "manual"
USE_ALL_FEATURES = True
FORECAST_HORIZON = 5

# Metrics exported to compact comparison tables
QUANTITATIVE_METRICS = ["MSE"]

# Optional oracle diagnostic: evaluates every test window with every cluster LSTM.
# It never replaces the same-cluster test metrics or the main plots.
TEST_ALL_MODELS = True

# Model hyperparameters
LSTM_UNITS = 256
LSTM_UNITS_2 = 128
DROPOUT_RATE = 0.1
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0  # Decoupled weight decay used by AdamW
LSTM_LOSS_FUNCTION = "quantile_weighted_mse"  # "mean_squared_error", "mae", "huber", or "quantile_weighted_mse"
LOSS_QUANTILES = [0.9]
LOSS_QUANTILE_WEIGHTS = "auto"  # "auto" or one positive weight per quantile bin

# Training settings
EPOCHS = 250
BATCH_SIZE = 64
EARLY_STOPPING = True
PATIENCE = 251
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
        clustering_feature_normalize=CLUSTERING_FEATURE_NORMALIZE,
        clustering_precipitation_normalize=CLUSTERING_PRECIPITATION_NORMALIZE,
        lstm_feature_normalize=LSTM_FEATURE_NORMALIZE,
        lstm_precipitation_normalize=LSTM_PRECIPITATION_NORMALIZE,
        variance_threshold=PCA_VARIANCE_THRESHOLD,
        pca_for_clustering_only=PCA_FOR_CLUSTERING_ONLY,
        run_only_cluster=RUN_ONLY_CLUSTER,
        n_clusters_list=N_CLUSTERS_LIST,
        clustering_algorithm=CLUSTERING_ALGORITHM,
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
        lstm_loss_function=LSTM_LOSS_FUNCTION,
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
    )


if __name__ == "__main__":
    main()
