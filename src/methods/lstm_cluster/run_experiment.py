"""Small entry point for the LSTM cluster experiment."""

from __future__ import annotations

from pathlib import Path

from config import DATA_ROOT, load_output_config, output_root_from_config
from methods.lstm_cluster.pipeline import run_experiment


# Station and data
STATE = "RS"
STATION_ID = "A801"

# Clustering sweep
WINDOW_SIZES = [8]
N_CLUSTERS_LIST = [3]
CLUSTERING_ALGORITHM = "spectral"
N_SIGMA_VALUES = 5
USE_ALL_FEATURES = True

# Metrics exported to compact comparison tables
QUANTITATIVE_METRICS = ["MSE"]

# Model hyperparameters
LSTM_UNITS = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001

# Training settings
EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOPPING = True
PATIENCE = 10
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
    output_config = load_output_config(OUTPUT_CONFIG)
    run_experiment(
        state=STATE,
        station_id=STATION_ID,
        window_sizes=WINDOW_SIZES,
        n_clusters_list=N_CLUSTERS_LIST,
        clustering_algorithm=CLUSTERING_ALGORITHM,
        n_sigma_values=N_SIGMA_VALUES,
        use_all_features=USE_ALL_FEATURES,
        quantitative_metrics=QUANTITATIVE_METRICS,
        lstm_units=LSTM_UNITS,
        lstm_units_2=LSTM_UNITS_2,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        early_stopping=EARLY_STOPPING,
        patience=PATIENCE,
        verbose_training=VERBOSE_TRAINING,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        random_state=RANDOM_STATE,
        data_root=DATA_ROOT,
        output_root=output_root_from_config(output_config),
        sweep_name=output_config.get("sweep_name") or None,
        sweep_name_prefix=str(output_config.get("sweep_name_prefix", "lstm_cluster_sweep")),
        timestamp_format=str(output_config.get("timestamp_format", "%Y%m%d_%H%M%S")),
        show_console_info=SHOW_CONSOLE_INFO,
    )


if __name__ == "__main__":
    main()
