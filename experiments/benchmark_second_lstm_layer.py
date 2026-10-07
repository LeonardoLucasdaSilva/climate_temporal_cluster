"""Compare a 32-unit second LSTM layer with saved one-layer configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


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


BATCH_NAME = "q090_patience_mae_precip_std_false"
REFERENCE_NAMES = (
    "RS_A801_w15_k01_kmeans_u1_64",
    "RS_A801_w15_k03_kmeans_u1_64",
    "RS_A801_w15_k05_kmeans_u1_64",
)
DEFAULT_SWEEP_DIR = PROJECT_ROOT / "outputs" / "pytorch" / "final_experiment_sweep_1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "pytorch" / "second_lstm_layer_benchmark.json"
SECOND_LAYER_UNITS = 32


def _load_labels(
    sweep_dir: Path,
    relative_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(sweep_dir / relative_path, allow_pickle=False) as saved:
        return tuple(
            np.asarray(saved[name], dtype=int)
            for name in ("c_train", "c_val", "c_test")
        )


def _metric_set(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        name.lower(): float(value)
        for name, value in calculate_regression_metrics(actual, predicted).items()
    }


def _percent_change(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return float("nan")
    return 100.0 * (candidate - baseline) / abs(baseline)


def benchmark(
    sweep_dir: Path,
    output_path: Path,
) -> dict[str, object]:
    """Run paired second-layer candidates for K=1, K=3, and K=5."""
    sweep._verify_pytorch_runtime()
    dataframe = load_station_daily_data(
        state=sweep.STATE,
        station_id=sweep.STATION_ID,
        data_root=sweep.DATA_ROOT,
    )
    features = numeric_feature_columns(dataframe)
    split_data = sweep._base_split(
        dataframe,
        features,
        window_size=15,
        precipitation_std=False,
    )

    result: dict[str, object] = {
        "changed_parameter": "lstm_units_2",
        "baseline_value": None,
        "candidate_value": SECOND_LAYER_UNITS,
        "controls": {
            "window_size": 15,
            "lstm_units_1": 64,
            "batch_size": sweep.BATCH_SIZE,
            "loss": sweep.LSTM_LOSS_FUNCTION,
            "loss_quantiles": [0.9],
            "early_stopping_metric": "mae",
            "precipitation_standardized": False,
            "random_state": sweep.RANDOM_STATE,
            "parallel_worker_count": sweep.PARALLEL_WORKER_COUNT,
        },
        "configurations": [],
    }
    configurations: list[dict[str, object]] = []

    for reference_name in REFERENCE_NAMES:
        reference_dir = sweep_dir / "compact_results" / BATCH_NAME / reference_name
        metadata = json.loads(
            (reference_dir / "metadata.json").read_text(encoding="utf-8")
        )
        parameters = dict(metadata["parameters"])
        config = dict(metadata["config"])
        c_train, c_val, c_test = _load_labels(
            sweep_dir,
            str(metadata["shared_assignment"]),
        )

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
            sequence_length=int(config["window_size"]),
            lstm_units=int(parameters["lstm_units"]),
            lstm_units_2=SECOND_LAYER_UNITS,
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
            random_state=int(parameters["random_state"]),
            show_console_info=False,
            test_all_models=False,
            parallel_training=sweep.PARALLEL_TRAINING,
            parallel_worker_count=sweep.PARALLEL_WORKER_COUNT,
            target_scaler=split_data.target_scaler,
            loss_alpha=float(parameters["loss_alpha"]),
            lstm_precipitation_transform=bool(
                parameters["lstm_precipitation_transform"]
            ),
            require_gpu=True,
        )
        elapsed_seconds = time.perf_counter() - started
        candidate_metrics = {
            "train": _metric_set(split_data.y_train, outputs[0]),
            "val": _metric_set(split_data.y_val, outputs[2]),
            "test": _metric_set(split_data.y_test, outputs[3]),
        }
        baseline_metrics = {
            split: {
                metric: float(metadata["metrics"][f"{split}_{metric}"])
                for metric in ("mae", "mse", "r2")
            }
            for split in ("train", "val", "test")
        }
        histories = outputs[5]
        epochs = [len(history["loss"]) for history in histories.values()]
        comparison = {
            "reference_run": reference_name,
            "n_clusters": int(config["n_clusters"]),
            "candidate_elapsed_seconds": elapsed_seconds,
            "candidate_epochs_min": min(epochs),
            "candidate_epochs_mean": float(np.mean(epochs)),
            "candidate_epochs_max": max(epochs),
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "change": {
                split: {
                    "mae_percent": _percent_change(
                        candidate_metrics[split]["mae"],
                        baseline_metrics[split]["mae"],
                    ),
                    "mse_percent": _percent_change(
                        candidate_metrics[split]["mse"],
                        baseline_metrics[split]["mse"],
                    ),
                    "r2_absolute": (
                        candidate_metrics[split]["r2"]
                        - baseline_metrics[split]["r2"]
                    ),
                }
                for split in ("train", "val", "test")
            },
        }
        configurations.append(comparison)
        result["configurations"] = configurations
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(
            f"K={config['n_clusters']} complete in {elapsed_seconds / 60:.2f} min | "
            f"val MAE {baseline_metrics['val']['mae']:.4f} -> "
            f"{candidate_metrics['val']['mae']:.4f} | test MAE "
            f"{baseline_metrics['test']['mae']:.4f} -> "
            f"{candidate_metrics['test']['mae']:.4f}",
            flush=True,
        )

    validation_changes = [
        float(item["change"]["val"]["mae_percent"])
        for item in configurations
    ]
    test_changes = [
        float(item["change"]["test"]["mae_percent"])
        for item in configurations
    ]
    result["summary"] = {
        "validation_mae_improved_count": sum(value < 0 for value in validation_changes),
        "test_mae_improved_count": sum(value < 0 for value in test_changes),
        "mean_validation_mae_percent_change": float(np.mean(validation_changes)),
        "mean_test_mae_percent_change": float(np.mean(test_changes)),
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"Saved {output_path}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    benchmark(args.sweep_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
