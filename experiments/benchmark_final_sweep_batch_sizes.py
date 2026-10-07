"""Benchmark final-sweep batch sizes on one completed configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.load_data import load_station_daily_data  # noqa: E402
from evaluation.metrics import calculate_regression_metrics  # noqa: E402
from experiments import run_final_experiment_sweeps as sweep  # noqa: E402
from methods.cluster.cluster_pipeline import numeric_feature_columns  # noqa: E402
from methods.lstm_cluster_pytorch.pipeline import train_cluster_models  # noqa: E402


DEFAULT_REFERENCE_RUN = (
    PROJECT_ROOT
    / "outputs"
    / "pytorch"
    / "final_experiment_sweep_1"
    / "compact_results"
    / "q090_patience_mae_precip_std_false"
    / "RS_A801_w30_k05_spectral_sigma_32p42130602307423_u1_64"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "pytorch" / "batch_size_benchmark.csv"
)


class GpuSampler:
    """Sample whole-device utilization while one benchmark case runs."""

    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.utilization: list[float] = []
        self.memory_mib: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> GpuSampler:
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_seconds * 2)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                utilization, memory = result.stdout.splitlines()[0].split(",")
                self.utilization.append(float(utilization.strip()))
                self.memory_mib.append(float(memory.strip()))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(self.interval_seconds)


def _load_reference(
    run_dir: Path,
) -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    final_dir = run_dir.parents[2]
    assignment_path = final_dir / str(metadata["shared_assignment"])
    with np.load(assignment_path, allow_pickle=False) as saved:
        labels = tuple(
            np.asarray(saved[name], dtype=int)
            for name in ("c_train", "c_val", "c_test")
        )
    return metadata, labels


def _history_epoch_summary(histories: dict[int, object]) -> tuple[int, float, int]:
    epochs = [len(history["loss"]) for history in histories.values()]
    return min(epochs), float(np.mean(epochs)), max(epochs)


def benchmark_batch_sizes(
    reference_run: Path,
    batch_sizes: list[int],
    output_path: Path,
) -> pd.DataFrame:
    """Benchmark batch sizes using a saved configuration and assignments."""
    import torch

    metadata, labels = _load_reference(reference_run)
    parameters = dict(metadata["parameters"])
    config = dict(metadata["config"])
    sweep._verify_pytorch_runtime()

    dataframe = load_station_daily_data(
        state=str(config["state"]),
        station_id=str(config["station_id"]),
        data_root=sweep.DATA_ROOT,
    )
    features = numeric_feature_columns(dataframe)
    split_data = sweep._base_split(
        dataframe,
        features,
        int(config["window_size"]),
        bool(parameters["precipitation_standardized"]),
    )
    c_train, c_val, c_test = labels
    expected_lengths = (
        len(split_data.X_train),
        len(split_data.X_val),
        len(split_data.X_test),
    )
    if tuple(len(values) for values in labels) != expected_lengths:
        raise RuntimeError("Saved cluster assignments do not match the rebuilt split.")

    rows: list[dict[str, object]] = []
    reference_metrics = dict(metadata["metrics"])
    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with GpuSampler() as gpu:
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
                sequence_length=int(config["window_size"]),
                lstm_units=int(parameters["lstm_units"]),
                lstm_units_2=parameters["lstm_units_2"],
                dropout_rate=float(parameters["dropout_rate"]),
                learning_rate=float(parameters["learning_rate"]),
                weight_decay=float(parameters["weight_decay"]),
                epochs=int(parameters["epochs"]),
                batch_size=batch_size,
                early_stopping=bool(parameters["early_stopping"]),
                patience=int(parameters["patience"]),
                warm_up=int(parameters["warm_up"]),
                early_stopping_metric=str(parameters["early_stopping_metric"]),
                lstm_loss_function=str(parameters["lstm_loss_function"]),
                loss_quantiles=[float(value) for value in parameters["loss_quantiles"]],
                loss_quantile_weights=parameters["loss_quantile_weights"],
                verbose_training=0,
                random_state=int(parameters["random_state"]),
                show_console_info=False,
                test_all_models=False,
                parallel_training=False,
                target_scaler=split_data.target_scaler,
                loss_alpha=float(parameters["loss_alpha"]),
                lstm_precipitation_transform=bool(
                    parameters["lstm_precipitation_transform"]
                ),
                require_gpu=True,
            )
            torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - started
        y_pred_val = outputs[2]
        y_pred_test = outputs[3]
        histories = outputs[5]
        val_metrics = calculate_regression_metrics(split_data.y_val, y_pred_val)
        test_metrics = calculate_regression_metrics(split_data.y_test, y_pred_test)
        min_epochs, mean_epochs, max_epochs = _history_epoch_summary(histories)
        utilization = np.asarray(gpu.utilization, dtype=float)
        row = {
            "batch_size": batch_size,
            "elapsed_seconds": elapsed_seconds,
            "speedup_vs_batch_4": np.nan,
            "gpu_util_mean_percent": (
                float(utilization.mean()) if len(utilization) else np.nan
            ),
            "gpu_util_p95_percent": (
                float(np.percentile(utilization, 95)) if len(utilization) else np.nan
            ),
            "gpu_memory_peak_mib": float(torch.cuda.max_memory_allocated() / 2**20),
            "gpu_memory_nvidia_smi_peak_mib": (
                max(gpu.memory_mib) if gpu.memory_mib else np.nan
            ),
            "epochs_min": min_epochs,
            "epochs_mean": mean_epochs,
            "epochs_max": max_epochs,
            "val_mae": val_metrics["MAE"],
            "val_mse": val_metrics["MSE"],
            "val_r2": val_metrics["R2"],
            "test_mae": test_metrics["MAE"],
            "test_mse": test_metrics["MSE"],
            "test_r2": test_metrics["R2"],
            "saved_batch_4_test_mae": reference_metrics["test_mae"],
        }
        rows.append(row)
        baseline = next(
            (float(item["elapsed_seconds"]) for item in rows if item["batch_size"] == 4),
            None,
        )
        if baseline is not None:
            for item in rows:
                item["speedup_vs_batch_4"] = baseline / float(item["elapsed_seconds"])
        frame = pd.DataFrame(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        print(frame.tail(1).to_string(index=False), flush=True)
        print(f"Updated {output_path}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE_RUN)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 16, 32, 64])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if any(value <= 0 for value in args.batch_sizes):
        parser.error("batch sizes must be positive")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("batch sizes must not contain duplicates")
    benchmark_batch_sizes(
        args.reference_run.resolve(),
        args.batch_sizes,
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
