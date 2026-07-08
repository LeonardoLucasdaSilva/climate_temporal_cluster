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

WINDOW_SIZES = [15,30,45,60]
N_CLUSTERS_LIST = [3]
PCA_VARIANCE_THRESHOLD = None
NORMALIZE = True
SCALER_TYPE = "minmax"  # "standard" or "minmax"; ignored when NORMALIZE=False
CLUSTERING_ALGORITHM = "kmeans"
N_SIGMA_VALUES = 5
SIGMA_MODE = "manual"  # "auto" or "manual"
MANUAL_SIGMA_VALUES = [0.3,0.5,0.7]  # Only used if SIGMA_MODE is "manual"
USE_ALL_FEATURES = True
FORECAST_HORIZON = 5

# Metrics exported to compact comparison tables
QUANTITATIVE_METRICS = ["MSE"]

# Model hyperparameters
LSTM_UNITS = 64
LSTM_UNITS_2 = 64
DROPOUT_RATE = 0.1
LEARNING_RATE = 0.001
LSTM_LOSS_FUNCTION = "mae"  # "mean_squared_error", "mae", "huber", or "quantile_weighted_mse"
LOSS_QUANTILES = [0.5, 0.75, 0.9, 0.95]
LOSS_QUANTILE_WEIGHTS = "auto"  # "auto" or one positive weight per quantile bin

# Training settings
EPOCHS = 250
BATCH_SIZE = 32
EARLY_STOPPING = True
PATIENCE = 100
VERBOSE_TRAINING = 1
SHOW_CONSOLE_INFO = True

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
        normalize=NORMALIZE,
        scaler_type=SCALER_TYPE,
        variance_threshold=PCA_VARIANCE_THRESHOLD,
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
