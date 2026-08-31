"""Small entry point for the PyTorch LSTM cluster experiment."""

from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from config import DATA_ROOT, load_output_config, output_root_from_config
from methods.lstm_cluster_pytorch.pipeline import run_experiment


# Station and data selection
STATE = "RS"
STATION_ID = "A801"

# Data setting
WINDOW_SIZES = [15,30,45]
WINDOW_STRIDE = 1                                 # Days between consecutive window starts;
FORECAST_HORIZON = 5
USE_ALL_FEATURES = True


# PCA settings
PCA_VARIANCE_THRESHOLD = None
PCA_FOR_CLUSTERING_ONLY = True                   # Keep pre-PCA window features as LSTM inputs


# Normalization settings
CLUSTERING_FEATURE_NORMALIZE = 'standard'         # "standard", "minmax", or None
CLUSTERING_PRECIPITATION_NORMALIZE = None  # "standard", "minmax", or None
LSTM_FEATURE_NORMALIZE = 'standard'              # "standard", "minmax", or None
LSTM_PRECIPITATION_NORMALIZE = None             # None keeps PRECIPITACAO_TOTAL and LSTM targets in mm
LSTM_PRECIPITATION_TRANSFORM = False             # Apply log(1+x) before LSTM precipitation normalization


# Clustering parameters
N_CLUSTERS_LIST = [1,3,5,7,9,12]
CLUSTERING_ALGORITHM = ["kmeans"]                # "kmeans", "kshape", "spectral", "manual", or a list
CLUSTER_ONLY_PRECIPITATION = False               # Cluster on precipitation time series only
CLUSTER_DISSIMILARITY_METRIC = "euclidean"       # "euclidean" or "dtw"
MANUAL_CLUSTERING_METHOD = "rain_level"          # "legacy" or "rain_level"
MANUAL_ZERO_TOLERANCE = 0.0                      # Used only by legacy manual clustering
CLUSTER_ASSIGNMENT_METHOD = "centroid"           # "centroid" or "knn"
CLUSTER_ASSIGNMENT_NEIGHBORS = 3                 # Used only when assignment method is "knn"
N_SIGMA_VALUES = 5
SIGMA_MODE = "manual"                            # "auto" or "manual"
MANUAL_SIGMA_VALUES = [0.3]                      # Only used if SIGMA_MODE is "manual"
RUN_ONLY_CLUSTER = False
TRAIN_INFO = False                                # Save train_performance/ diagnostics
SILHOUETTE_INFO = False                           # Save cluster_diagnostics silhouette diagnostics
PLOT_CLUSTER_TIMESERIES = False                  # Save test-window precipitation diagnostics by cluster
CLUSTER_TIMESERIES_PLOT_LIMIT = 3                # None saves every test-window series

# Model hyperparameters. Use LSTM_UNITS_2=None for a single LSTM layer.
LSTM_UNITS: int | list[int] = [1024]
LSTM_UNITS_2: int | None | list[int | None] = None
DROPOUT_RATE: float | list[float] = 0.2
LEARNING_RATE: float | list[float] = 1e-3
WEIGHT_DECAY: float | list[float] = [1e-4]         # Decoupled weight decay used by AdamW

# Metrics exported to compact comparison tables
QUANTITATIVE_METRICS = ["MAE", "MSE", "R2"]

# Optional oracle diagnostic: evaluates every test window with every cluster LSTM.
TEST_ALL_MODELS = False

# LSTM Loss and metrics
LSTM_LOSS_FUNCTION = "quantile_weighted_mse"     # Supported: "mean_squared_error", "mae", "huber", "weighted_mse_loss", "quantile_weighted_mse"
LOSS_ALPHA = 1e-2                                # Positive coefficient used only by weighted_mse_loss
LOSS_QUANTILES = [0.9]
LOSS_QUANTILE_WEIGHTS = "auto"                   # "auto" or one positive weight per quantile bin

# Training settings. Numeric settings may also be lists in a comparative grid.
EPOCHS: int | list[int] = 300
BATCH_SIZE: int | list[int] = 4
EARLY_STOPPING = True
PATIENCE: int | list[int] = 50
WARM_UP: int | list[int] = 10
EARLY_STOPPING_METRIC = "mae"                   # "loss", "mse", "mae", or "r2"
VERBOSE_TRAINING = 1
SHOW_CONSOLE_INFO = True                         # Only cluster windows; skip all LSTM training/output.
PARALEL = False                                  # Keep one PyTorch trainer on the single GPU at a time.
REQUIRE_GPU = True                               # Fail unless PyTorch detects a CUDA GPU.
CREATE_REPORT = False                            # Keep per-config TeX without compiling a PDF.

# Sweep-level comparison between the tests produced by this run.
COMPARATIVE_RUN = False
PIVOT_PARAMETER = "WINDOW_SIZES"         # e.g. "window_size", "learning_rate", "K", "sigma", "CLUSTERING_ALGORITHM"

# Train/validation/test split
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
RANDOM_STATE = 42

# Output settings
OUTPUT_CONFIG = Path(__file__).with_name("config_output.yaml")


def verify_pytorch_runtime() -> None:
    """Fail before preprocessing when PyTorch or the requested GPU is unavailable."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed in the Python interpreter running this "
            f"experiment: {sys.executable}. Install it with that interpreter's "
            "`-m pip`, or run with the project .venv interpreter."
        ) from exc
    if REQUIRE_GPU and not torch.cuda.is_available():
        raise RuntimeError(
            "This runner requires a CUDA GPU, but this PyTorch installation "
            f"cannot access one. Interpreter: {sys.executable}; "
            f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}."
        )
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(
        f"PyTorch {torch.__version__} | device={device} | python={sys.executable}",
        flush=True,
    )


def experiment_kwargs(
    overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return pipeline arguments from this runner's current settings."""
    sigma_mode = SIGMA_MODE.lower()
    if sigma_mode not in {"auto", "manual"}:
        raise ValueError("SIGMA_MODE must be either 'auto' or 'manual'.")

    output_config = load_output_config(OUTPUT_CONFIG)
    parameters: dict[str, object] = {
        "state": STATE,
        "station_id": STATION_ID,
        "window_sizes": WINDOW_SIZES,
        "window_stride": WINDOW_STRIDE,
        "clustering_feature_normalize": CLUSTERING_FEATURE_NORMALIZE,
        "clustering_precipitation_normalize": CLUSTERING_PRECIPITATION_NORMALIZE,
        "lstm_feature_normalize": LSTM_FEATURE_NORMALIZE,
        "lstm_precipitation_normalize": LSTM_PRECIPITATION_NORMALIZE,
        "lstm_precipitation_transform": LSTM_PRECIPITATION_TRANSFORM,
        "variance_threshold": PCA_VARIANCE_THRESHOLD,
        "pca_for_clustering_only": PCA_FOR_CLUSTERING_ONLY,
        "run_only_cluster": RUN_ONLY_CLUSTER,
        "train_info": TRAIN_INFO,
        "silhouette_info": SILHOUETTE_INFO,
        "plot_cluster_timeseries": PLOT_CLUSTER_TIMESERIES,
        "cluster_timeseries_plot_limit": CLUSTER_TIMESERIES_PLOT_LIMIT,
        "n_clusters_list": N_CLUSTERS_LIST,
        "clustering_algorithm": CLUSTERING_ALGORITHM,
        "cluster_only_precipitation": CLUSTER_ONLY_PRECIPITATION,
        "cluster_dissimilarity_metric": CLUSTER_DISSIMILARITY_METRIC,
        "manual_clustering_method": MANUAL_CLUSTERING_METHOD,
        "manual_zero_tolerance": MANUAL_ZERO_TOLERANCE,
        "cluster_assignment_method": CLUSTER_ASSIGNMENT_METHOD,
        "cluster_assignment_neighbors": CLUSTER_ASSIGNMENT_NEIGHBORS,
        "n_sigma_values": N_SIGMA_VALUES,
        "sigma_values": MANUAL_SIGMA_VALUES if sigma_mode == "manual" else None,
        "use_all_features": USE_ALL_FEATURES,
        "forecast_horizon": FORECAST_HORIZON,
        "quantitative_metrics": QUANTITATIVE_METRICS,
        "lstm_units": LSTM_UNITS,
        "lstm_units_2": LSTM_UNITS_2,
        "dropout_rate": DROPOUT_RATE,
        "learning_rate": LEARNING_RATE,
        "test_all_models": TEST_ALL_MODELS,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "early_stopping": EARLY_STOPPING,
        "patience": PATIENCE,
        "warm_up": WARM_UP,
        "early_stopping_metric": EARLY_STOPPING_METRIC,
        "lstm_loss_function": LSTM_LOSS_FUNCTION,
        "loss_alpha": LOSS_ALPHA,
        "loss_quantiles": LOSS_QUANTILES,
        "loss_quantile_weights": LOSS_QUANTILE_WEIGHTS,
        "verbose_training": VERBOSE_TRAINING,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "random_state": RANDOM_STATE,
        "data_root": DATA_ROOT,
        "output_root": output_root_from_config(output_config),
        "sweep_name": output_config.get("sweep_name") or None,
        "sweep_name_prefix": str(
            output_config.get("sweep_name_prefix", "pytorch_lstm_cluster_sweep")
        ),
        "timestamp_format": str(
            output_config.get("timestamp_format", "%Y%m%d_%H%M%S")
        ),
        "plot_style": output_config.get("plot_style"),
        "show_console_info": SHOW_CONSOLE_INFO,
        "comparative_run": COMPARATIVE_RUN,
        "pivot_parameter": PIVOT_PARAMETER,
        "parallel_training": PARALEL,
        "require_gpu": REQUIRE_GPU,
        "create_report": CREATE_REPORT,
        "ranking_progress_log": None,
        "ranking_progress_run_token": None,
        "ranking_progress_total": None,
    }
    if overrides:
        unknown = sorted(set(overrides) - set(parameters))
        if unknown:
            raise ValueError(
                "Unknown run_experiment override(s): " + ", ".join(unknown)
            )
        parameters.update(overrides)
    return parameters


def main() -> None:
    """Run the experiment using the variables above and config_output.yaml."""
    verify_pytorch_runtime()
    run_experiment(**experiment_kwargs())


if __name__ == "__main__":
    main()
