"""Rerun K=1..9 validation rankings over five seeds and spectral sigmas."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.final_experiment_storage import (  # noqa: E402
    read_source_snapshot,
    write_json_atomic,
    write_npz_atomic,
)
from data.repeated_validation_outputs import (  # noqa: E402
    save_repeated_validation_outputs,
    summarize_validation_repeats,
    validation_mae_by_horizon,
    write_csv_atomic,
)
from methods.cluster.cluster_pipeline import KMEANS_N_INIT  # noqa: E402
from methods.cluster.ng import (  # noqa: E402
    SpectralClusteringModel,
    SpectralTrainingGraph,
    build_spectral_training_graph,
)
from methods.lstm_cluster_pytorch.pipeline import (  # noqa: E402
    ExperimentConfig,
    WindowSplitData,
    create_window_split_data,
    train_cluster_models,
)


SOURCE_SWEEP_DIR = PROJECT_ROOT / "outputs" / "pytorch" / "final_experiment_sweep_1"
SOURCE_BATCH = "q090_patience_mae_precip_std_false"
SOURCE_BASELINE = "RS_A801_w30_k01_kmeans_u1_64_u2_none"
OUTPUT_DIR = SOURCE_SWEEP_DIR / "validation_repeats_w30_qwmse_0p9_64_none"
REPORT_TEX_PATH = (
    SOURCE_SWEEP_DIR
    / "cluster_comparison_rankings_w30_qwmse_0p9_rain_thresholds.tex"
)
SEEDS = (42, 43, 44, 45, 46)
THRESHOLDS_MM = (0.0, 10.0, 30.0)
HEURISTIC_SIGMAS = (
    32.42130602307423,
    38.24076285811172,
    44.06021969314921,
    49.8796765281867,
    55.69913336322419,
)
PARALLEL_TRAINING_WORKERS = 2


@dataclass(frozen=True)
class ValidationConfiguration:
    """One K-means, spectral, or unclustered validation configuration."""

    config_id: str
    n_clusters: int
    algorithm: str
    sigma: float | None = None

    @property
    def clustering(self) -> str:
        if self.n_clusters == 1:
            return "-"
        if self.algorithm == "spectral":
            return f"spectral({self.sigma:.4f})"
        return self.algorithm


def _sigma_token(sigma: float) -> str:
    return f"{sigma:.4f}".replace(".", "p")


def _configurations() -> tuple[ValidationConfiguration, ...]:
    configurations = [ValidationConfiguration("k01_baseline", 1, "kmeans")]
    for n_clusters in range(2, 10):
        configurations.append(
            ValidationConfiguration(f"k{n_clusters:02d}_kmeans", n_clusters, "kmeans")
        )
        configurations.extend(
            ValidationConfiguration(
                f"k{n_clusters:02d}_spectral_s{_sigma_token(sigma)}",
                n_clusters,
                "spectral",
                sigma,
            )
            for sigma in HEURISTIC_SIGMAS
        )
    return tuple(configurations)


CONFIGURATIONS = _configurations()


def _verify_runtime() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The repeated validation runner requires a CUDA GPU.")
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    print(
        f"PyTorch {torch.__version__} | device={torch.cuda.get_device_name(0)} | "
        "test evaluation disabled",
        flush=True,
    )


def _reference_metadata() -> dict[str, object]:
    path = (
        SOURCE_SWEEP_DIR
        / "compact_results"
        / SOURCE_BATCH
        / SOURCE_BASELINE
        / "metadata.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _split_data(metadata: dict[str, object]) -> WindowSplitData:
    parameters = dict(metadata["parameters"])
    config_values = dict(metadata["config"])
    dataframe = read_source_snapshot(SOURCE_SWEEP_DIR)
    config = ExperimentConfig(
        state=str(config_values["state"]),
        station_id=str(config_values["station_id"]),
        window_size=int(parameters["window_size"]),
        n_clusters=1,
        algorithm="kmeans",
        sigma=None,
        window_stride=int(config_values["window_stride"]),
        cluster_only_precipitation=bool(config_values["cluster_only_precipitation"]),
        plot_cluster_timeseries=False,
    )
    precipitation_scaler = (
        "standard" if parameters["precipitation_standardized"] else None
    )
    split_data, _ = create_window_split_data(
        dataframe,
        config,
        [str(value) for value in metadata["feature_columns"]],
        clustering_feature_normalize=str(parameters["clustering_feature_normalize"]),
        clustering_precipitation_normalize=precipitation_scaler,
        lstm_feature_normalize=str(parameters["lstm_feature_normalize"]),
        lstm_precipitation_normalize=precipitation_scaler,
        variance_threshold=parameters["variance_threshold"],
        forecast_horizon=int(parameters["forecast_horizon"]),
        train_ratio=float(parameters["train_ratio"]),
        val_ratio=float(parameters["val_ratio"]),
        random_state=int(parameters["random_state"]),
        pca_for_clustering_only=bool(parameters["pca_for_clustering_only"]),
        lstm_precipitation_transform=bool(parameters["lstm_precipitation_transform"]),
    )
    return split_data


def _nearest_centroid_labels(
    training: np.ndarray,
    training_labels: np.ndarray,
    held_out: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    centroids = np.vstack(
        [training[training_labels == cluster].mean(axis=0) for cluster in range(n_clusters)]
    )
    if not np.all(np.isfinite(centroids)):
        raise ValueError("Clustering produced an empty training cluster.")
    distances = np.sum((held_out[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1).astype(int)


def _cluster_labels(
    split_data: WindowSplitData,
    configuration: ValidationConfiguration,
    *,
    seed: int,
    graph_cache: dict[float, SpectralTrainingGraph],
) -> tuple[np.ndarray, np.ndarray]:
    if configuration.n_clusters == 1:
        return (
            np.zeros(len(split_data.X_train), dtype=int),
            np.zeros(len(split_data.X_val), dtype=int),
        )
    if configuration.algorithm == "kmeans":
        model = KMeans(
            n_clusters=configuration.n_clusters,
            random_state=seed,
            n_init=KMEANS_N_INIT,
        ).fit(split_data.cluster_X_train)
        c_train = np.asarray(model.labels_, dtype=int)
        return c_train, _nearest_centroid_labels(
            split_data.cluster_X_train,
            c_train,
            split_data.cluster_X_val,
            configuration.n_clusters,
        )
    if configuration.sigma is None:
        raise ValueError("Spectral configuration is missing sigma.")
    sigma = float(configuration.sigma)
    if sigma not in graph_cache:
        graph_cache[sigma] = build_spectral_training_graph(
            split_data.cluster_X_train,
            sigma,
            max(item.n_clusters for item in CONFIGURATIONS),
        )
    model = SpectralClusteringModel(
        sigma=sigma,
        k=configuration.n_clusters,
        random_state=seed,
    ).fit_from_training_graph(graph_cache[sigma])
    return np.asarray(model.labels_, dtype=int), model.predict(split_data.cluster_X_val)


def _run_directory(seed: int, config_id: str) -> Path:
    return OUTPUT_DIR / "runs" / f"seed_{seed}" / config_id


def _load_completed_metrics(seed: int, config_id: str) -> pd.DataFrame | None:
    run_dir = _run_directory(seed, config_id)
    metadata_path = run_dir / "metadata.json"
    metrics_path = run_dir / "validation_mae.csv"
    predictions_path = run_dir / "validation_predictions.npz"
    if not (metadata_path.is_file() and metrics_path.is_file() and predictions_path.is_file()):
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        return None
    return pd.read_csv(metrics_path)


def _all_completed_metrics() -> pd.DataFrame:
    frames = [
        metrics
        for seed in SEEDS
        for configuration in CONFIGURATIONS
        if (metrics := _load_completed_metrics(seed, configuration.config_id))
        is not None
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _checkpoint() -> tuple[int, pd.DataFrame]:
    metrics = _all_completed_metrics()
    completed = sum(
        _load_completed_metrics(seed, configuration.config_id) is not None
        for seed in SEEDS
        for configuration in CONFIGURATIONS
    )
    if not metrics.empty:
        write_csv_atomic(metrics, OUTPUT_DIR / "validation_mae_by_seed.csv")
        summary = summarize_validation_repeats(metrics, expected_seeds=SEEDS)
        write_csv_atomic(summary, OUTPUT_DIR / "validation_mae_summary.csv")
    write_json_atomic(
        OUTPUT_DIR / "progress.json",
        {
            "completed_runs": completed,
            "total_runs": len(SEEDS) * len(CONFIGURATIONS),
            "remaining_runs": len(SEEDS) * len(CONFIGURATIONS) - completed,
        },
    )
    return completed, metrics


def _train_one(
    split_data: WindowSplitData,
    metadata: dict[str, object],
    configuration: ValidationConfiguration,
    *,
    seed: int,
    graph_cache: dict[float, SpectralTrainingGraph],
) -> None:
    parameters = dict(metadata["parameters"])
    run_dir = _run_directory(seed, configuration.config_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    c_train, c_val = _cluster_labels(
        split_data,
        configuration,
        seed=seed,
        graph_cache=graph_cache,
    )
    cluster_counts = np.bincount(c_train, minlength=configuration.n_clusters)
    smallest_fraction = float(cluster_counts.min() / len(c_train))
    c_test = np.zeros(len(split_data.X_test), dtype=int)
    started = time.perf_counter()
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
        sequence_length=int(parameters["window_size"]),
        lstm_units=64,
        lstm_units_2=None,
        dropout_rate=float(parameters["dropout_rate"]),
        learning_rate=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
        epochs=int(parameters["epochs"]),
        batch_size=int(parameters["batch_size"]),
        early_stopping=bool(parameters["early_stopping"]),
        patience=int(parameters["patience"]),
        warm_up=int(parameters["warm_up"]),
        early_stopping_metric=str(parameters["early_stopping_metric"]),
        lstm_loss_function=str(parameters["lstm_loss_function"]),
        loss_quantiles=[float(value) for value in parameters["loss_quantiles"]],
        loss_quantile_weights=parameters["loss_quantile_weights"],
        verbose_training=0,
        random_state=seed,
        show_console_info=False,
        test_all_models=False,
        evaluate_test=False,
        parallel_training=bool(parameters["parallel_training"]),
        parallel_worker_count=PARALLEL_TRAINING_WORKERS,
        target_scaler=split_data.target_scaler,
        loss_alpha=float(parameters["loss_alpha"]),
        lstm_precipitation_transform=bool(parameters["lstm_precipitation_transform"]),
        require_gpu=True,
        return_validation_by_lead_day=True,
    )
    y_pred_val_by_lead_day = np.asarray(outputs[3], dtype=float)
    histories = dict(outputs[6])
    metrics = validation_mae_by_horizon(
        split_data.y_val_by_lead_day,
        y_pred_val_by_lead_day,
        thresholds_mm=THRESHOLDS_MM,
    )
    metrics.insert(0, "sigma", configuration.sigma)
    metrics.insert(0, "n_clusters", configuration.n_clusters)
    metrics.insert(0, "clustering", configuration.clustering)
    metrics.insert(0, "config_id", configuration.config_id)
    metrics.insert(0, "seed", seed)
    write_csv_atomic(metrics, run_dir / "validation_mae.csv")
    write_npz_atomic(
        run_dir / "validation_predictions.npz",
        {"y_pred_val_by_lead_day": y_pred_val_by_lead_day},
    )
    write_json_atomic(
        run_dir / "metadata.json",
        {
            "status": "completed",
            "seed": seed,
            "config_id": configuration.config_id,
            "n_clusters": configuration.n_clusters,
            "algorithm": configuration.algorithm,
            "sigma": configuration.sigma,
            "cluster_counts": cluster_counts,
            "smallest_cluster_fraction": smallest_fraction,
            "epochs_by_cluster": {
                str(cluster): len(dict(history).get("loss", []))
                for cluster, history in histories.items()
            },
            "elapsed_seconds": time.perf_counter() - started,
            "test_evaluation": False,
        },
    )


def run() -> Path:
    """Run or resume all 60 training jobs and write the final report."""
    _verify_runtime()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = _reference_metadata()
    parameters = dict(metadata["parameters"])
    if (
        int(parameters["window_size"]) != 30
        or int(parameters["lstm_units"]) != 64
        or parameters["lstm_units_2"] is not None
        or [float(value) for value in parameters["loss_quantiles"]] != [0.9]
    ):
        raise ValueError("The reference metadata does not match w30/QWMSE(.9)/64/-.")
    split_data = _split_data(metadata)
    write_json_atomic(
        OUTPUT_DIR / "experiment_design.json",
        {
            "seeds": SEEDS,
            "thresholds_mm": THRESHOLDS_MM,
            "forecast_horizons": list(range(1, split_data.y_val_by_lead_day.shape[1] + 1)),
            "configurations": [configuration.__dict__ for configuration in CONFIGURATIONS],
            "cluster_counts": list(range(1, 10)),
            "heuristic_sigmas": HEURISTIC_SIGMAS,
            "parallel_training_workers": PARALLEL_TRAINING_WORKERS,
            "architecture": "64/-",
            "window_size": 30,
            "loss": "QWMSE(.9)",
            "validation_only": True,
            "test_evaluation": False,
            "standard_deviation": "sample (ddof=1)",
        },
    )
    graph_cache: dict[float, SpectralTrainingGraph] = {}
    total = len(SEEDS) * len(CONFIGURATIONS)
    for seed in SEEDS:
        for configuration in CONFIGURATIONS:
            if _load_completed_metrics(seed, configuration.config_id) is not None:
                continue
            print(
                f"[{seed}] {configuration.config_id} "
                f"({configuration.clustering}, K={configuration.n_clusters})",
                flush=True,
            )
            _train_one(
                split_data,
                metadata,
                configuration,
                seed=seed,
                graph_cache=graph_cache,
            )
            completed, _ = _checkpoint()
            print(f"completed={completed}/{total}", flush=True)
            import torch

            torch.cuda.empty_cache()

    completed, metrics = _checkpoint()
    if completed != total:
        raise RuntimeError(f"Only {completed}/{total} validation reruns completed.")
    save_repeated_validation_outputs(
        metrics,
        OUTPUT_DIR,
        REPORT_TEX_PATH,
        seeds=SEEDS,
        compile_pdf=False,
    )
    print(f"Repeated-validation outputs: {OUTPUT_DIR}", flush=True)
    print(f"Updated report source: {REPORT_TEX_PATH}", flush=True)
    return OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Regenerate the summary and report from completed seed metrics.",
    )
    args = parser.parse_args()
    if args.summarize_only:
        completed, metrics = _checkpoint()
        expected = len(SEEDS) * len(CONFIGURATIONS)
        if completed != expected:
            raise RuntimeError(f"Only {completed}/{expected} validation reruns completed.")
        save_repeated_validation_outputs(
            metrics,
            OUTPUT_DIR,
            REPORT_TEX_PATH,
            seeds=SEEDS,
            compile_pdf=False,
        )
        return
    run()


if __name__ == "__main__":
    main()
