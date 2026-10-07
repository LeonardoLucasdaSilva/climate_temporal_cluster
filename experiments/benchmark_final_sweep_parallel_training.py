"""Benchmark two-worker LSTM training against one saved sequential run."""

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

from data.final_experiment_storage import compact_result_arrays  # noqa: E402
from data.load_data import load_station_daily_data  # noqa: E402
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
    / "RS_A801_w15_k03_kmeans_u1_64"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "pytorch" / "parallel_training_benchmark.json"
)


def _load_labels(
    final_dir: Path,
    relative_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(final_dir / relative_path, allow_pickle=False) as saved:
        return tuple(
            np.asarray(saved[name], dtype=int)
            for name in ("c_train", "c_val", "c_test")
        )


def benchmark(reference_run: Path, output_path: Path) -> dict[str, object]:
    """Run a two-worker reproduction and compare it with saved arrays."""
    import torch

    metadata = json.loads(
        (reference_run / "metadata.json").read_text(encoding="utf-8")
    )
    parameters = dict(metadata["parameters"])
    config = dict(metadata["config"])
    final_dir = reference_run.parents[2]
    labels = _load_labels(final_dir, str(metadata["shared_assignment"]))

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
        lstm_units_2=parameters["lstm_units_2"],
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
        parallel_training=True,
        parallel_worker_count=2,
        target_scaler=split_data.target_scaler,
        loss_alpha=float(parameters["loss_alpha"]),
        lstm_precipitation_transform=bool(
            parameters["lstm_precipitation_transform"]
        ),
        require_gpu=True,
    )
    elapsed_seconds = time.perf_counter() - started
    candidate_arrays, _history_index = compact_result_arrays(
        y_pred_train=outputs[0],
        y_pred_train_by_lead_day=outputs[1],
        y_pred_val=outputs[2],
        y_pred_test=outputs[3],
        y_pred_test_by_lead_day=outputs[4],
        histories_by_cluster=outputs[5],
    )

    comparisons: dict[str, dict[str, float | bool]] = {}
    with np.load(reference_run / "result.npz", allow_pickle=False) as reference:
        same_keys = set(candidate_arrays) == set(reference.files)
        for name, candidate in candidate_arrays.items():
            expected = np.asarray(reference[name])
            difference = (
                float(np.max(np.abs(candidate - expected)))
                if candidate.size
                else 0.0
            )
            comparisons[name] = {
                "exact": bool(np.array_equal(candidate, expected, equal_nan=True)),
                "max_abs_difference": difference,
            }

    exact = same_keys and all(item["exact"] for item in comparisons.values())
    sequential_seconds = float(
        next(
            row["elapsed_seconds"]
            for row in sweep._read_manifest(final_dir).to_dict("records")
            if row["configuration_id"]
            == f'{metadata["batch"]}/{metadata["run_name"]}'
        )
    )
    result: dict[str, object] = {
        "reference_run": str(reference_run),
        "workers": 2,
        "sequential_seconds": sequential_seconds,
        "parallel_seconds": elapsed_seconds,
        "speedup": sequential_seconds / elapsed_seconds,
        "same_array_keys": same_keys,
        "all_arrays_exact": exact,
        "comparisons": comparisons,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "comparisons"}, indent=2))
    print(f"Compared {len(comparisons)} arrays; max difference="
          f"{max(item['max_abs_difference'] for item in comparisons.values())}")
    print(f"Saved {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    benchmark(args.reference_run.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
