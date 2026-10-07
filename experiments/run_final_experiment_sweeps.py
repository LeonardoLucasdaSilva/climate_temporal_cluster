"""Run five compact, resumable PyTorch reproducibility sweeps."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import os
from pathlib import Path
import sys
import time
from typing import Mapping

# Set CUDA/plot behavior before importing torch through project modules.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DATA_ROOT  # noqa: E402
from data.load_data import load_station_daily_data  # noqa: E402
from evaluation.metrics import calculate_regression_metrics  # noqa: E402
from methods.cluster.cluster_pipeline import KMEANS_N_INIT, numeric_feature_columns  # noqa: E402
from methods.cluster.ng import (  # noqa: E402
    SpectralClusteringModel,
    SpectralTrainingGraph,
    build_spectral_training_graph,
)
from methods.lstm_cluster_pytorch.pipeline import (  # noqa: E402
    ExperimentConfig,
    WindowSplitData,
    _training_sigma_values,
    build_configurations,
    create_window_split_data,
    train_cluster_models,
)
from data.final_experiment_storage import (  # noqa: E402
    COMPACT_DIR_NAME,
    SHARED_DIR_NAME,
    assignment_key,
    base_key,
    compact_result_arrays,
    compact_result_complete,
    config_as_json,
    ensure_source_snapshot,
    load_metadata,
    refresh_compact_rankings,
    save_compact_result,
    save_shared_assignment,
    save_shared_targets,
    update_compact_ranking,
    write_json_atomic,
)


STATE = "RS"
STATION_ID = "A801"
WINDOW_SIZES = [15, 30, 45]
N_CLUSTERS = [1, 3, 5, 7, 9]
CLUSTERING_ALGORITHMS = ["kmeans", "spectral"]
N_SIGMA_VALUES = 5
LSTM_ARCHITECTURES = [(64, None), (128, None), (64, 32)]
EARLY_STOPPING_METRICS = ["mae"]
PRECIPITATION_STANDARDIZATION = [False]
LOSS_CONFIGURATIONS = [
    ("q090", [0.9]),
    ("q080", [0.8]),
    ("mixed_q075_q090_q095_q099", [0.75, 0.9, 0.95, 0.99]),
]
REPETITIONS = [1, 2, 3, 4, 5]
RANDOM_STATE = 42
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "pytorch"
PROGRESS_DASHBOARD = PROJECT_ROOT / "SWEEP_PROGRESS.md"

WINDOW_STRIDE = 1
FORECAST_HORIZON = 5
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
CLUSTERING_FEATURE_NORMALIZE = "standard"
LSTM_FEATURE_NORMALIZE = "standard"
VARIANCE_THRESHOLD = None
PCA_FOR_CLUSTERING_ONLY = True
LSTM_PRECIPITATION_TRANSFORM = False
CLUSTER_ONLY_PRECIPITATION = False
MIN_TRAINING_CLUSTER_FRACTION = 0.05

DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 300
BATCH_SIZE = 8
EARLY_STOPPING = True
PATIENCE = 50
WARM_UP = 10
LSTM_LOSS_FUNCTION = "quantile_weighted_mse"
LOSS_QUANTILE_WEIGHTS = "auto"
LOSS_ALPHA = 1e-2
TEST_ALL_MODELS = False
REQUIRE_GPU = True
PARALLEL_TRAINING = True
PARALLEL_WORKER_COUNT = 2
TRAIN_INFO = False
SILHOUETTE_INFO = False

EXPECTED_CONFIGURATIONS_PER_BATCH = 225
EXPECTED_CONFIGURATIONS_PER_REPETITION = 675
RANKING_AUDIT_INTERVAL = 25


def _batch_name(loss_name: str, early_metric: str, precipitation_std: bool) -> str:
    std_label = "precip_std_true" if precipitation_std else "precip_std_false"
    return f"{loss_name}_patience_{early_metric}_{std_label}"


def _validate_design() -> None:
    sigma_grid = {window: list(range(1, 6)) for window in WINDOW_SIZES}
    configs = _configuration_grid(sigma_grid)
    if len(configs) != EXPECTED_CONFIGURATIONS_PER_BATCH:
        raise RuntimeError(
            f"Expected {EXPECTED_CONFIGURATIONS_PER_BATCH} configurations per "
            f"batch, found {len(configs)}."
        )
    batch_count = (
        len(LOSS_CONFIGURATIONS)
        * len(EARLY_STOPPING_METRICS)
        * len(PRECIPITATION_STANDARDIZATION)
    )
    if len(configs) * batch_count != EXPECTED_CONFIGURATIONS_PER_REPETITION:
        raise RuntimeError("Final sweep repetition count is inconsistent.")


def _verify_pytorch_runtime() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(f"PyTorch is unavailable in {sys.executable}.") from exc
    if REQUIRE_GPU and not torch.cuda.is_available():
        raise RuntimeError("The final experiment runner requires a CUDA GPU.")
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(
        f"PyTorch {torch.__version__} | deterministic=True | device={device}",
        flush=True,
    )


def _write_design_manifest(final_dir: Path, repetition: int) -> None:
    write_json_atomic(
        final_dir / "experiment_design.json",
        {
            "schema_version": 1,
            "repetition": repetition,
            "random_state": RANDOM_STATE,
            "deterministic_training": True,
            "window_sizes": WINDOW_SIZES,
            "cluster_counts": N_CLUSTERS,
            "clustering_algorithms": CLUSTERING_ALGORITHMS,
            "automatic_sigma_candidates_per_window": N_SIGMA_VALUES,
            "loss_configurations": LOSS_CONFIGURATIONS,
            "lstm_architectures": [
                {"lstm_units_1": units_1, "lstm_units_2": units_2}
                for units_1, units_2 in LSTM_ARCHITECTURES
            ],
            "early_stopping_metrics": EARLY_STOPPING_METRICS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "parallel_training": PARALLEL_TRAINING,
            "parallel_worker_count": PARALLEL_WORKER_COUNT,
            "precipitation_standardization": PRECIPITATION_STANDARDIZATION,
            "planned_configurations": EXPECTED_CONFIGURATIONS_PER_REPETITION,
            "storage": {
                "per_configuration": ["metadata.json", "result.npz"],
                "shared": ["source data", "targets", "cluster assignments"],
                "full_artifacts": "reconstruct on demand without retraining",
            },
            "notes": [
                "K=1 is trained once per otherwise-distinct LSTM configuration.",
                "Spectral affinity/eigensystem work is reused for K=3,5,7,9.",
                "Overall, D+1...D+5, and grouped K rankings are retained.",
            ],
        },
    )


def _read_manifest(final_dir: Path) -> pd.DataFrame:
    path = final_dir / "configuration_manifest.csv"
    return pd.read_csv(path, keep_default_na=False) if path.is_file() else pd.DataFrame()


def _write_manifest_row(final_dir: Path, row: Mapping[str, object]) -> None:
    path = final_dir / "configuration_manifest.csv"
    frame = _read_manifest(final_dir)
    if not frame.empty and "configuration_id" in frame:
        frame = frame.loc[frame["configuration_id"] != row["configuration_id"]]
    frame = pd.concat([frame, pd.DataFrame([dict(row)])], ignore_index=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _completed_ids(final_dir: Path) -> set[str]:
    manifest = _read_manifest(final_dir)
    if manifest.empty:
        return set()
    return set(
        manifest.loc[
            manifest["status"].isin(["completed", "invalid_clustering"]),
            "configuration_id",
        ].astype(str)
    )


def _progress(final_dir: Path, latest: str) -> int:
    manifest = _read_manifest(final_dir)
    if manifest.empty or "status" not in manifest:
        trained = 0
        invalid = 0
    else:
        trained = int(manifest["status"].eq("completed").sum())
        invalid = int(manifest["status"].eq("invalid_clustering").sum())
    processed = trained + invalid
    remaining = max(0, EXPECTED_CONFIGURATIONS_PER_REPETITION - processed)
    updated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    (final_dir / "progress.txt").write_text(
        "\n".join(
            [
                f"run={final_dir.name}",
                f"completed={processed}",
                f"trained={trained}",
                f"invalid_or_skipped={invalid}",
                f"total={EXPECTED_CONFIGURATIONS_PER_REPETITION}",
                f"remaining={remaining}",
                f"latest={latest}",
                f"updated_at={updated_at}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    percentage = (
        100.0 * processed / EXPECTED_CONFIGURATIONS_PER_REPETITION
        if EXPECTED_CONFIGURATIONS_PER_REPETITION
        else 100.0
    )
    dashboard = "\n".join(
        [
            "# Sweep Progress",
            "",
            "This file is updated automatically after every configuration checkpoint.",
            "",
            "| Status | Count |",
            "|---|---:|",
            f"| Processed | {processed} / {EXPECTED_CONFIGURATIONS_PER_REPETITION} ({percentage:.1f}%) |",
            f"| Successfully trained | {trained} |",
            f"| Invalid/skipped | {invalid} |",
            f"| Remaining | {remaining} |",
            "",
            f"- Latest configuration: `{latest}`",
            f"- Last updated: `{updated_at}`",
            f"- Results directory: `outputs/pytorch/{final_dir.name}`",
            "",
            "`Processed` includes both trained configurations and clusterings that were "
            "validated and skipped as invalid.",
            "",
        ]
    )
    try:
        PROGRESS_DASHBOARD.write_text(dashboard, encoding="utf-8")
    except OSError as exc:
        # A dashboard opened by another Windows program must not stop the sweep.
        print(f"Warning: could not update {PROGRESS_DASHBOARD.name}: {exc}", flush=True)
    return processed


def _base_split(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    precipitation_std: bool,
) -> WindowSplitData:
    precipitation_scaler = "standard" if precipitation_std else None
    config = ExperimentConfig(
        state=STATE,
        station_id=STATION_ID,
        window_size=window_size,
        n_clusters=1,
        algorithm="kmeans",
        sigma=None,
        window_stride=WINDOW_STRIDE,
        cluster_only_precipitation=CLUSTER_ONLY_PRECIPITATION,
        plot_cluster_timeseries=False,
    )
    split_data, _ = create_window_split_data(
        dataframe,
        config,
        feature_columns,
        clustering_feature_normalize=CLUSTERING_FEATURE_NORMALIZE,
        clustering_precipitation_normalize=precipitation_scaler,
        lstm_feature_normalize=LSTM_FEATURE_NORMALIZE,
        lstm_precipitation_normalize=precipitation_scaler,
        variance_threshold=VARIANCE_THRESHOLD,
        forecast_horizon=FORECAST_HORIZON,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        random_state=RANDOM_STATE,
        pca_for_clustering_only=PCA_FOR_CLUSTERING_ONLY,
        lstm_precipitation_transform=LSTM_PRECIPITATION_TRANSFORM,
    )
    return split_data


def _nearest_centroid_labels(
    X_train: np.ndarray,
    c_train: np.ndarray,
    X_held_out: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    centroids = np.vstack(
        [X_train[c_train == cluster].mean(axis=0) for cluster in range(n_clusters)]
    )
    if not np.all(np.isfinite(centroids)):
        raise ValueError("Clustering produced an empty training cluster.")
    distances = np.sum(
        (X_held_out[:, None, :] - centroids[None, :, :]) ** 2,
        axis=2,
    )
    return np.argmin(distances, axis=1).astype(int)


def _configuration_grid(sigmas: Mapping[int, list[float]]) -> list[ExperimentConfig]:
    base_configs = build_configurations(
        sigmas,
        state=STATE,
        station_id=STATION_ID,
        window_sizes=WINDOW_SIZES,
        n_clusters_list=N_CLUSTERS,
        clustering_algorithm=CLUSTERING_ALGORITHMS,
        window_stride=WINDOW_STRIDE,
        cluster_only_precipitation=CLUSTER_ONLY_PRECIPITATION,
        plot_cluster_timeseries=False,
    )
    return [
        replace(
            config,
            variant_parameters=(
                ("lstm_units", units_1),
                ("lstm_units_2", units_2),
            ),
        )
        for config in base_configs
        for units_1, units_2 in LSTM_ARCHITECTURES
    ]


def _cluster_labels(
    split_data: WindowSplitData,
    config: ExperimentConfig,
    graph_cache: dict[tuple[bool, int, float], SpectralTrainingGraph],
    precipitation_std: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if config.n_clusters == 1:
        return tuple(
            np.zeros(length, dtype=int)
            for length in (len(split_data.X_train), len(split_data.X_val), len(split_data.X_test))
        )
    if config.algorithm == "kmeans":
        model = KMeans(
            n_clusters=config.n_clusters,
            random_state=RANDOM_STATE,
            n_init=KMEANS_N_INIT,
        ).fit(split_data.cluster_X_train)
        c_train = np.asarray(model.labels_, dtype=int)
        return (
            c_train,
            _nearest_centroid_labels(
                split_data.cluster_X_train, c_train, split_data.cluster_X_val, config.n_clusters
            ),
            _nearest_centroid_labels(
                split_data.cluster_X_train, c_train, split_data.cluster_X_test, config.n_clusters
            ),
        )
    if config.algorithm == "spectral":
        if config.sigma is None:
            raise ValueError("Spectral configuration is missing sigma.")
        key = (precipitation_std, config.window_size, float(config.sigma))
        if key not in graph_cache:
            graph_cache[key] = build_spectral_training_graph(
                split_data.cluster_X_train,
                float(config.sigma),
                max(N_CLUSTERS),
            )
        model = SpectralClusteringModel(
            sigma=float(config.sigma),
            k=config.n_clusters,
            random_state=RANDOM_STATE,
        ).fit_from_training_graph(graph_cache[key])
        return (
            np.asarray(model.labels_, dtype=int),
            model.predict(split_data.cluster_X_val),
            model.predict(split_data.cluster_X_test),
        )
    raise ValueError(f"Unsupported algorithm: {config.algorithm}")


def _parameters(
    config: ExperimentConfig,
    quantiles: list[float],
    early_metric: str,
    precipitation_std: bool,
) -> dict[str, object]:
    variants = dict(config.variant_parameters)
    units = int(variants["lstm_units"])
    units_2 = variants["lstm_units_2"]
    clustering_method = "-"
    if config.n_clusters > 1:
        clustering_method = (
            f"spectral({config.sigma:g})"
            if config.algorithm == "spectral" and config.sigma is not None
            else config.algorithm
        )
    return {
        "window_size": config.window_size,
        "n_clusters": config.n_clusters,
        "algorithm": config.algorithm,
        "clustering_method": clustering_method,
        "sigma": config.sigma,
        "lstm_units": units,
        "lstm_units_2": None if units_2 is None else int(units_2),
        "dropout_rate": DROPOUT_RATE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "early_stopping": EARLY_STOPPING,
        "patience": PATIENCE,
        "warm_up": WARM_UP,
        "early_stopping_metric": early_metric,
        "lstm_loss_function": LSTM_LOSS_FUNCTION,
        "loss_alpha": LOSS_ALPHA,
        "loss_quantiles": quantiles,
        "loss_quantile_weights": LOSS_QUANTILE_WEIGHTS,
        "precipitation_standardized": precipitation_std,
        "clustering_feature_normalize": CLUSTERING_FEATURE_NORMALIZE,
        "lstm_feature_normalize": LSTM_FEATURE_NORMALIZE,
        "variance_threshold": VARIANCE_THRESHOLD,
        "pca_for_clustering_only": PCA_FOR_CLUSTERING_ONLY,
        "lstm_precipitation_transform": LSTM_PRECIPITATION_TRANSFORM,
        "forecast_horizon": FORECAST_HORIZON,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "random_state": RANDOM_STATE,
        "parallel_training": PARALLEL_TRAINING,
        "parallel_worker_count": PARALLEL_WORKER_COUNT,
        "train_info": TRAIN_INFO,
        "silhouette_info": SILHOUETTE_INFO,
    }


def _report_config(
    dataframe: pd.DataFrame,
    config: ExperimentConfig,
    features: list[str],
    split_data: WindowSplitData,
    parameters: Mapping[str, object],
) -> dict[str, object]:
    scaler = "standard" if parameters["precipitation_standardized"] else None
    n_samples = len(split_data.all_targets)
    return {
        **asdict(config),
        "name": config.name,
        "dataset_start_date": dataframe["Data"].min().date().isoformat(),
        "dataset_end_date": dataframe["Data"].max().date().isoformat(),
        "features": features,
        "clustering_features": features,
        "clustering_feature_normalize": CLUSTERING_FEATURE_NORMALIZE,
        "clustering_precipitation_normalize": scaler,
        "lstm_feature_normalize": LSTM_FEATURE_NORMALIZE,
        "lstm_precipitation_normalize": scaler,
        "lstm_precipitation_transform": LSTM_PRECIPITATION_TRANSFORM,
        "target_scale": "normalized" if scaler else "mm",
        "pca_variance_threshold": VARIANCE_THRESHOLD,
        "pca_for_clustering_only": PCA_FOR_CLUSTERING_ONLY,
        "n_samples": n_samples,
        "forecast_horizon": FORECAST_HORIZON,
        "splits": {
            "Training": {"samples": len(split_data.y_train), "percent": len(split_data.y_train) / n_samples},
            "Validation": {"samples": len(split_data.y_val), "percent": len(split_data.y_val) / n_samples},
            "Test": {"samples": len(split_data.y_test), "percent": len(split_data.y_test) / n_samples},
        },
        "lstm_units": parameters["lstm_units"],
        "lstm_units_2": parameters["lstm_units_2"],
        "dropout_rate": DROPOUT_RATE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "early_stopping": EARLY_STOPPING,
        "patience": PATIENCE,
        "warm_up": WARM_UP,
        "early_stopping_metric": parameters["early_stopping_metric"],
        "loss": LSTM_LOSS_FUNCTION,
        "loss_quantiles": parameters["loss_quantiles"],
        "loss_quantile_weights": LOSS_QUANTILE_WEIGHTS,
        "random_state": RANDOM_STATE,
        "test_all_models": TEST_ALL_MODELS,
        "run_only_cluster": False,
        "parallel_training": PARALLEL_TRAINING,
        "parallel_worker_count": PARALLEL_WORKER_COUNT,
        "train_info": TRAIN_INFO,
        "silhouette_info": SILHOUETTE_INFO,
        "deterministic_training": True,
    }


def _train_and_save(
    *,
    final_dir: Path,
    run_dir: Path,
    repetition: int,
    batch_name: str,
    dataframe: pd.DataFrame,
    features: list[str],
    split_data: WindowSplitData,
    config: ExperimentConfig,
    labels: tuple[np.ndarray, np.ndarray, np.ndarray],
    shared_targets: Path,
    shared_assignment: Path,
    parameters: Mapping[str, object],
) -> dict[str, object]:
    c_train, c_val, c_test = labels
    outputs = train_cluster_models(
        split_data.X_train,
        split_data.X_val,
        split_data.X_test,
        split_data.y_train,
        split_data.y_val,
        split_data.y_test,
        split_data.y_train_by_lead_day,
        split_data.y_val_by_lead_day,
        split_data.y_test_by_lead_day,
        split_data.y_train_by_lead_day_scaled,
        split_data.y_val_by_lead_day_scaled,
        c_train,
        c_val,
        c_test,
        sequence_length=config.window_size,
        lstm_units=int(parameters["lstm_units"]),
        lstm_units_2=(
            None
            if parameters["lstm_units_2"] is None
            else int(parameters["lstm_units_2"])
        ),
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        early_stopping=EARLY_STOPPING,
        patience=PATIENCE,
        warm_up=WARM_UP,
        early_stopping_metric=str(parameters["early_stopping_metric"]),
        lstm_loss_function=LSTM_LOSS_FUNCTION,
        loss_quantiles=[float(value) for value in parameters["loss_quantiles"]],
        loss_quantile_weights=LOSS_QUANTILE_WEIGHTS,
        verbose_training=0,
        random_state=RANDOM_STATE,
        show_console_info=False,
        test_all_models=TEST_ALL_MODELS,
        parallel_training=PARALLEL_TRAINING,
        parallel_worker_count=PARALLEL_WORKER_COUNT,
        target_scaler=split_data.target_scaler,
        loss_alpha=LOSS_ALPHA,
        lstm_precipitation_transform=LSTM_PRECIPITATION_TRANSFORM,
        require_gpu=REQUIRE_GPU,
    )
    (
        y_pred_train,
        y_pred_train_by_lead_day,
        y_pred_val,
        y_pred_test,
        y_pred_test_by_lead_day,
        histories,
        _,
        _,
    ) = outputs
    metrics: dict[str, float] = {}
    for split_name, actual, predicted in (
        ("train", split_data.y_train, y_pred_train),
        ("val", split_data.y_val, y_pred_val),
        ("test", split_data.y_test, y_pred_test),
    ):
        for metric, value in calculate_regression_metrics(actual, predicted).items():
            metrics[f"{split_name}_{metric.lower()}"] = float(value)
    arrays, history_index = compact_result_arrays(
        y_pred_train=y_pred_train,
        y_pred_train_by_lead_day=y_pred_train_by_lead_day,
        y_pred_val=y_pred_val,
        y_pred_test=y_pred_test,
        y_pred_test_by_lead_day=y_pred_test_by_lead_day,
        histories_by_cluster=histories,
    )
    save_compact_result(
        run_dir,
        {
            "repetition": repetition,
            "batch": batch_name,
            "run_name": config.name,
            "config": config_as_json(config),
            "parameters": dict(parameters),
            "feature_columns": features,
            "metrics": metrics,
            "history_index": history_index,
            "shared_targets": shared_targets.relative_to(final_dir).as_posix(),
            "shared_assignment": shared_assignment.relative_to(final_dir).as_posix(),
            "source_data": f"{SHARED_DIR_NAME}/source_data.csv.gz",
            "report_config": _report_config(
                dataframe, config, features, split_data, parameters
            ),
        },
        arrays,
    )
    return load_metadata(run_dir)


def _prepare_shared_data(
    final_dir: Path,
    dataframe: pd.DataFrame,
    features: list[str],
    base_cache: dict[tuple[bool, int], WindowSplitData],
    sigma_cache: dict[tuple[bool, int], list[float]],
) -> dict[bool, dict[int, list[float]]]:
    grids: dict[bool, dict[int, list[float]]] = {False: {}, True: {}}
    for precipitation_std in PRECIPITATION_STANDARDIZATION:
        scaler = "standard" if precipitation_std else None
        for window_size in WINDOW_SIZES:
            key = (precipitation_std, window_size)
            if key not in base_cache:
                base_cache[key] = _base_split(
                    dataframe, features, window_size, precipitation_std
                )
            if key not in sigma_cache:
                sigma_cache[key] = _training_sigma_values(
                    dataframe,
                    features,
                    window_size=window_size,
                    window_stride=WINDOW_STRIDE,
                    forecast_horizon=FORECAST_HORIZON,
                    train_ratio=TRAIN_RATIO,
                    val_ratio=VAL_RATIO,
                    clustering_feature_normalize=CLUSTERING_FEATURE_NORMALIZE,
                    clustering_precipitation_normalize=scaler,
                    variance_threshold=VARIANCE_THRESHOLD,
                    cluster_only_precipitation=CLUSTER_ONLY_PRECIPITATION,
                    n_values=N_SIGMA_VALUES,
                ).tolist()
            grids[precipitation_std][window_size] = sigma_cache[key]
            save_shared_targets(
                final_dir,
                base_key(precipitation_std, window_size),
                base_cache[key],
            )
    return grids


def run_repetition(
    repetition: int,
    dataframe: pd.DataFrame,
    features: list[str],
    base_cache: dict[tuple[bool, int], WindowSplitData],
    sigma_cache: dict[tuple[bool, int], list[float]],
    label_cache: dict[tuple[bool, int, str, int, float | None], tuple[np.ndarray, np.ndarray, np.ndarray]],
    graph_cache: dict[tuple[bool, int, float], SpectralTrainingGraph],
) -> None:
    final_dir = OUTPUT_ROOT / f"final_experiment_sweep_{repetition}"
    final_dir.mkdir(parents=True, exist_ok=True)
    _write_design_manifest(final_dir, repetition)
    ensure_source_snapshot(final_dir, dataframe)
    found, repaired = refresh_compact_rankings(final_dir, compile_pdf=False)
    print(f"Ranking audit: compact={found}, repaired={repaired}", flush=True)
    sigma_grids = _prepare_shared_data(
        final_dir, dataframe, features, base_cache, sigma_cache
    )
    completed_ids = _completed_ids(final_dir)
    existing_manifest = _read_manifest(final_dir)
    latest = (
        str(existing_manifest.iloc[-1]["configuration_id"])
        if not existing_manifest.empty
        else "none yet"
    )
    _progress(final_dir, latest)

    for loss_name, quantiles in LOSS_CONFIGURATIONS:
        for early_metric in EARLY_STOPPING_METRICS:
            for precipitation_std in PRECIPITATION_STANDARDIZATION:
                batch_name = _batch_name(loss_name, early_metric, precipitation_std)
                print(
                    f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
                    f"repetition={repetition} batch={batch_name}",
                    flush=True,
                )
                configs = _configuration_grid(sigma_grids[precipitation_std])
                for config in configs:
                    configuration_id = f"{batch_name}/{config.name}"
                    run_dir = final_dir / COMPACT_DIR_NAME / configuration_id
                    if compact_result_complete(run_dir):
                        if configuration_id not in completed_ids:
                            metadata = load_metadata(run_dir)
                            parameters = dict(metadata["parameters"])
                            _write_manifest_row(
                                final_dir,
                                {
                                    "configuration_id": configuration_id,
                                    "status": "completed",
                                    "window_size": config.window_size,
                                    "n_clusters": config.n_clusters,
                                    "algorithm": config.algorithm,
                                    "sigma": config.sigma,
                                    "lstm_units": parameters["lstm_units"],
                                    "lstm_units_2": parameters["lstm_units_2"],
                                    "loss": loss_name,
                                    "patience_metric": early_metric,
                                    "precipitation_standardized": precipitation_std,
                                    "elapsed_seconds": "recovered",
                                    "run_path": run_dir.relative_to(PROJECT_ROOT).as_posix(),
                                    "updated_at": datetime.now().astimezone().isoformat(
                                        timespec="seconds"
                                    ),
                                },
                            )
                            completed_ids.add(configuration_id)
                            _progress(final_dir, configuration_id)
                        continue
                    if configuration_id in completed_ids:
                        rows = _read_manifest(final_dir)
                        status = rows.loc[
                            rows["configuration_id"] == configuration_id, "status"
                        ]
                        if not status.empty and status.iloc[-1] == "invalid_clustering":
                            continue

                    split_data = base_cache[(precipitation_std, config.window_size)]
                    label_key = (
                        precipitation_std,
                        config.window_size,
                        config.algorithm,
                        config.n_clusters,
                        config.sigma,
                    )
                    shared_key = assignment_key(*label_key)
                    existing_assignment = (
                        final_dir
                        / SHARED_DIR_NAME
                        / "cluster_assignments"
                        / f"{shared_key}.npz"
                    )
                    if label_key not in label_cache and existing_assignment.is_file():
                        with np.load(existing_assignment, allow_pickle=False) as saved:
                            label_cache[label_key] = (
                                np.asarray(saved["c_train"], dtype=int),
                                np.asarray(saved["c_val"], dtype=int),
                                np.asarray(saved["c_test"], dtype=int),
                            )
                    if label_key not in label_cache:
                        label_cache[label_key] = _cluster_labels(
                            split_data, config, graph_cache, precipitation_std
                        )
                    labels = label_cache[label_key]
                    c_train, c_val, c_test = labels
                    assignment_path = save_shared_assignment(
                        final_dir,
                        shared_key,
                        c_train,
                        c_val,
                        c_test,
                        {
                            "precipitation_standardized": precipitation_std,
                            "window_size": config.window_size,
                            "algorithm": config.algorithm,
                            "n_clusters": config.n_clusters,
                            "sigma": config.sigma,
                        },
                    )
                    cluster_counts = np.bincount(c_train, minlength=config.n_clusters)
                    smallest_fraction = float(cluster_counts.min() / len(c_train))
                    if (
                        config.algorithm == "spectral"
                        and smallest_fraction < MIN_TRAINING_CLUSTER_FRACTION
                    ):
                        _write_manifest_row(
                            final_dir,
                            {
                                "configuration_id": configuration_id,
                                "status": "invalid_clustering",
                                "window_size": config.window_size,
                                "n_clusters": config.n_clusters,
                                "algorithm": config.algorithm,
                                "sigma": config.sigma,
                                "cluster_counts": ";".join(
                                    str(int(value)) for value in cluster_counts
                                ),
                                "smallest_cluster_fraction": smallest_fraction,
                                "reason": "smallest training cluster below 5%",
                                "updated_at": datetime.now().astimezone().isoformat(
                                    timespec="seconds"
                                ),
                            },
                        )
                        completed_ids.add(configuration_id)
                        _progress(final_dir, configuration_id)
                        continue

                    parameters = _parameters(
                        config, quantiles, early_metric, precipitation_std
                    )
                    started = time.perf_counter()
                    targets_path = (
                        final_dir
                        / SHARED_DIR_NAME
                        / "targets"
                        / f"{base_key(precipitation_std, config.window_size)}.npz"
                    )
                    metadata = _train_and_save(
                        final_dir=final_dir,
                        run_dir=run_dir,
                        repetition=repetition,
                        batch_name=batch_name,
                        dataframe=dataframe,
                        features=features,
                        split_data=split_data,
                        config=config,
                        labels=labels,
                        shared_targets=targets_path,
                        shared_assignment=assignment_path,
                        parameters=parameters,
                    )
                    update_compact_ranking(final_dir, run_dir, metadata)
                    elapsed = time.perf_counter() - started
                    _write_manifest_row(
                        final_dir,
                        {
                            "configuration_id": configuration_id,
                            "status": "completed",
                            "window_size": config.window_size,
                            "n_clusters": config.n_clusters,
                            "algorithm": config.algorithm,
                            "sigma": config.sigma,
                            "lstm_units": parameters["lstm_units"],
                            "lstm_units_2": parameters["lstm_units_2"],
                            "loss": loss_name,
                            "patience_metric": early_metric,
                            "precipitation_standardized": precipitation_std,
                            "elapsed_seconds": elapsed,
                            "run_path": run_dir.relative_to(PROJECT_ROOT).as_posix(),
                            "updated_at": datetime.now().astimezone().isoformat(
                                timespec="seconds"
                            ),
                        },
                    )
                    completed_ids.add(configuration_id)
                    completed = _progress(final_dir, configuration_id)
                    print(
                        f"completed={completed}/{EXPECTED_CONFIGURATIONS_PER_REPETITION} "
                        f"elapsed={elapsed / 60:.1f} min {configuration_id}",
                        flush=True,
                    )
                    if completed % RANKING_AUDIT_INTERVAL == 0:
                        found, repaired = refresh_compact_rankings(
                            final_dir, compile_pdf=False
                        )
                        print(
                            f"Periodic ranking audit: compact={found}, repaired={repaired}",
                            flush=True,
                        )
                found, repaired = refresh_compact_rankings(
                    final_dir, compile_pdf=True
                )
                print(
                    f"Batch ranking PDFs refreshed: compact={found}, repaired={repaired}",
                    flush=True,
                )

    (final_dir / "completed.txt").write_text(
        datetime.now().astimezone().isoformat(timespec="seconds") + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repetitions", type=int, nargs="+", default=REPETITIONS, choices=REPETITIONS
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Repair and render rankings from compact results without training.",
    )
    args = parser.parse_args()
    _validate_design()
    total = EXPECTED_CONFIGURATIONS_PER_REPETITION * len(args.repetitions)
    print(
        f"Validated {EXPECTED_CONFIGURATIONS_PER_REPETITION} configurations per "
        f"repetition; selected total={total}.",
        flush=True,
    )
    if args.dry_run:
        return
    if args.render_only:
        for repetition in args.repetitions:
            final_dir = OUTPUT_ROOT / f"final_experiment_sweep_{repetition}"
            found, repaired = refresh_compact_rankings(final_dir, compile_pdf=True)
            print(
                f"{final_dir.name}: compact={found}, ranking_rows_repaired={repaired}",
                flush=True,
            )
        return

    _verify_pytorch_runtime()
    dataframe = load_station_daily_data(
        state=STATE, station_id=STATION_ID, data_root=DATA_ROOT
    )
    features = numeric_feature_columns(dataframe)
    base_cache: dict[tuple[bool, int], WindowSplitData] = {}
    sigma_cache: dict[tuple[bool, int], list[float]] = {}
    label_cache: dict[
        tuple[bool, int, str, int, float | None],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    graph_cache: dict[tuple[bool, int, float], SpectralTrainingGraph] = {}
    for repetition in args.repetitions:
        run_repetition(
            repetition,
            dataframe,
            features,
            base_cache,
            sigma_cache,
            label_cache,
            graph_cache,
        )


if __name__ == "__main__":
    main()
