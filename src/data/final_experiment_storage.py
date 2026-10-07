"""Compact storage, ranking repair, and reconstruction for final sweeps."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.experiment_rankings import (  # noqa: E402
    render_cluster_comparison_rankings,
    update_experiment_rankings,
)
from evaluation.metrics import calculate_regression_metrics  # noqa: E402
from methods.lstm_cluster.report import generate_config_report  # noqa: E402
from methods.lstm_cluster_pytorch.pipeline import (  # noqa: E402
    ExperimentConfig,
    WindowSplitData,
    create_window_split_data,
)
from data.lstm_outputs import save_run_outputs  # noqa: E402


SCHEMA_VERSION = 1
SHARED_DIR_NAME = "_shared"
COMPACT_DIR_NAME = "compact_results"


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write one JSON object without exposing a partially written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write one compressed NumPy archive atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 checksum for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shared_source_path(final_dir: Path) -> Path:
    return Path(final_dir) / SHARED_DIR_NAME / "source_data.csv.gz"


def ensure_source_snapshot(final_dir: Path, dataframe: pd.DataFrame) -> Path:
    """Save one exact input snapshot per repetition for future reconstruction."""
    path = shared_source_path(final_dir)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        dataframe.to_csv(temporary, index=False, compression="gzip")
        temporary.replace(path)
        write_json_atomic(
            path.with_suffix(".json"),
            {
                "rows": len(dataframe),
                "columns": list(dataframe.columns),
                "sha256": sha256_file(path),
            },
        )
    return path


def base_key(precipitation_standardized: bool, window_size: int) -> str:
    return f"precip_std_{str(bool(precipitation_standardized)).lower()}_w{window_size:02d}"


def assignment_key(
    precipitation_standardized: bool,
    window_size: int,
    algorithm: str,
    n_clusters: int,
    sigma: float | None,
) -> str:
    """Return a stable short key for shared train/validation/test labels."""
    payload = json.dumps(
        {
            "precipitation_standardized": bool(precipitation_standardized),
            "window_size": int(window_size),
            "algorithm": str(algorithm),
            "n_clusters": int(n_clusters),
            "sigma": None if sigma is None else float(sigma),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def save_shared_targets(
    final_dir: Path,
    key: str,
    split_data: WindowSplitData,
) -> Path:
    """Persist observed targets once per preprocessing/window combination."""
    path = Path(final_dir) / SHARED_DIR_NAME / "targets" / f"{key}.npz"
    if not path.is_file():
        write_npz_atomic(
            path,
            {
                "y_test": np.asarray(split_data.y_test),
                "y_test_by_lead_day": np.asarray(split_data.y_test_by_lead_day),
            },
        )
    return path


def save_shared_assignment(
    final_dir: Path,
    key: str,
    c_train: np.ndarray,
    c_val: np.ndarray,
    c_test: np.ndarray,
    description: Mapping[str, object],
) -> Path:
    """Persist cluster assignments once, shared by every LSTM configuration."""
    path = Path(final_dir) / SHARED_DIR_NAME / "cluster_assignments" / f"{key}.npz"
    if not path.is_file():
        write_npz_atomic(
            path,
            {
                "c_train": np.asarray(c_train, dtype=np.int16),
                "c_val": np.asarray(c_val, dtype=np.int16),
                "c_test": np.asarray(c_test, dtype=np.int16),
            },
        )
        write_json_atomic(path.with_suffix(".json"), dict(description))
    return path


def compact_result_arrays(
    *,
    y_pred_train: np.ndarray,
    y_pred_train_by_lead_day: np.ndarray,
    y_pred_val: np.ndarray,
    y_pred_test: np.ndarray,
    y_pred_test_by_lead_day: np.ndarray,
    histories_by_cluster: Mapping[int, object],
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Encode predictions and training histories in one compressed archive."""
    arrays = {
        "y_pred_train": np.asarray(y_pred_train),
        "y_pred_train_by_lead_day": np.asarray(y_pred_train_by_lead_day),
        "y_pred_val": np.asarray(y_pred_val),
        "y_pred_test": np.asarray(y_pred_test),
        "y_pred_test_by_lead_day": np.asarray(y_pred_test_by_lead_day),
    }
    history_index: dict[str, list[str]] = {}
    for cluster_id, raw_history in sorted(histories_by_cluster.items()):
        history = (
            dict(raw_history)
            if isinstance(raw_history, Mapping)
            else dict(getattr(raw_history, "history", {}))
        )
        metrics: list[str] = []
        for metric, values in sorted(history.items()):
            metric_name = str(metric)
            arrays[f"history__{int(cluster_id)}__{metric_name}"] = np.asarray(
                values,
                dtype=np.float32,
            )
            metrics.append(metric_name)
        history_index[str(int(cluster_id))] = metrics
    return arrays, history_index


def decode_histories(
    archive: Mapping[str, np.ndarray],
    history_index: Mapping[str, Sequence[str]],
) -> dict[int, dict[str, list[float]]]:
    """Decode history arrays saved by :func:`compact_result_arrays`."""
    return {
        int(cluster_id): {
            metric: np.asarray(
                archive[f"history__{cluster_id}__{metric}"], dtype=float
            ).tolist()
            for metric in metrics
        }
        for cluster_id, metrics in history_index.items()
    }


def save_compact_result(
    run_dir: Path,
    metadata: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> tuple[Path, Path]:
    """Save the two per-configuration files that define a completed result."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.npz"
    metadata_path = run_dir / "metadata.json"
    write_npz_atomic(result_path, arrays)
    completed_metadata = dict(metadata)
    completed_metadata.update(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "result_file": result_path.name,
            "result_sha256": sha256_file(result_path),
        }
    )
    write_json_atomic(metadata_path, completed_metadata)
    return metadata_path, result_path


def load_metadata(run_dir: Path) -> dict[str, object]:
    return json.loads((Path(run_dir) / "metadata.json").read_text(encoding="utf-8"))


def compact_result_complete(run_dir: Path) -> bool:
    """Return whether a compact result has both committed files."""
    metadata_path = Path(run_dir) / "metadata.json"
    result_path = Path(run_dir) / "result.npz"
    if not metadata_path.is_file() or not result_path.is_file():
        return False
    try:
        return load_metadata(run_dir).get("status") == "completed"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def update_compact_ranking(
    final_dir: Path,
    run_dir: Path,
    metadata: Mapping[str, object],
    *,
    compile_pdf: bool = False,
) -> None:
    """Upsert one compact result into overall and per-horizon rankings."""
    parameters = dict(metadata["parameters"])
    metrics = dict(metadata["metrics"])
    shared_targets = Path(final_dir) / str(metadata["shared_targets"])
    result_path = Path(run_dir) / str(metadata.get("result_file", "result.npz"))
    with np.load(shared_targets, allow_pickle=False) as targets, np.load(
        result_path, allow_pickle=False
    ) as result:
        update_experiment_rankings(
            backend="pytorch",
            output_dir=Path(run_dir),
            window_size=int(parameters["window_size"]),
            n_clusters=int(parameters["n_clusters"]),
            clustering_method=str(parameters["clustering_method"]),
            loss_function=str(parameters["lstm_loss_function"]),
            loss_quantiles=[float(value) for value in parameters["loss_quantiles"]],
            mae=float(metrics["test_mae"]),
            mse=float(metrics["test_mse"]),
            r2=float(metrics["test_r2"]),
            actual=targets["y_test"],
            predicted=result["y_pred_test"],
            actual_by_lead_day=targets["y_test_by_lead_day"],
            predicted_by_lead_day=result["y_pred_test_by_lead_day"],
            forecast_horizon=int(parameters["forecast_horizon"]),
            patience_metric=str(parameters["early_stopping_metric"]),
            lstm_units=int(parameters["lstm_units"]),
            lstm_units_2=(
                None
                if parameters.get("lstm_units_2") is None
                else int(parameters["lstm_units_2"])
            ),
            precipitation_standardized=bool(
                parameters["precipitation_standardized"]
            ),
            compile_pdf=compile_pdf,
            ranking_dir=Path(final_dir),
        )


def refresh_compact_rankings(
    final_dir: Path,
    *,
    compile_pdf: bool,
) -> tuple[int, int]:
    """Add compact results missing from rankings and optionally render PDFs."""
    final_dir = Path(final_dir)
    ranking_path = final_dir / "experiment_rankings.csv"
    horizon_path = final_dir / "experiment_horizon_rankings.csv"
    ranked_paths: set[str] = set()
    horizon_paths: set[str] = set()
    if ranking_path.is_file():
        ranked_paths = set(pd.read_csv(ranking_path)["run_path"].astype(str))
    if horizon_path.is_file():
        horizon_paths = set(pd.read_csv(horizon_path)["run_path"].astype(str))

    found = 0
    repaired = 0
    compact_root = final_dir / COMPACT_DIR_NAME
    for metadata_path in sorted(compact_root.glob("*/*/metadata.json")):
        run_dir = metadata_path.parent
        if not compact_result_complete(run_dir):
            continue
        found += 1
        try:
            relative_run_path = run_dir.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            relative_run_path = run_dir.resolve().as_posix()
        if relative_run_path in ranked_paths and relative_run_path in horizon_paths:
            continue
        update_compact_ranking(final_dir, run_dir, load_metadata(run_dir))
        repaired += 1

    if compile_pdf:
        render_cluster_comparison_rankings(
            "pytorch",
            ranking_dir=final_dir,
            compile_pdf=True,
        )
        # Re-upserting the last row compiles all ordinary horizon ranking PDFs.
        completed = sorted(compact_root.glob("*/*/metadata.json"))
        if completed:
            last_dir = completed[-1].parent
            if compact_result_complete(last_dir):
                update_compact_ranking(
                    final_dir,
                    last_dir,
                    load_metadata(last_dir),
                    compile_pdf=True,
                )
    return found, repaired


def read_source_snapshot(final_dir: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(shared_source_path(final_dir))
    if "Data" in dataframe.columns:
        dataframe["Data"] = pd.to_datetime(dataframe["Data"])
    return dataframe


def reconstruct_artifacts(run_dir: Path, output_dir: Path | None = None) -> Path:
    """Recreate the ordinary artifact tree for one compact result, without training."""
    run_dir = Path(run_dir).resolve()
    metadata = load_metadata(run_dir)
    final_dir = run_dir.parents[2]
    parameters = dict(metadata["parameters"])
    config_values = dict(metadata["config"])
    config_values["variant_parameters"] = tuple(
        tuple(item) for item in config_values.get("variant_parameters", [])
    )
    config = ExperimentConfig(**config_values)
    dataframe = read_source_snapshot(final_dir)
    feature_columns = [str(value) for value in metadata["feature_columns"]]
    precipitation_scaler = (
        "standard" if parameters["precipitation_standardized"] else None
    )
    base_config = ExperimentConfig(
        state=config.state,
        station_id=config.station_id,
        window_size=config.window_size,
        n_clusters=1,
        algorithm="kmeans",
        sigma=None,
        window_stride=config.window_stride,
        cluster_only_precipitation=config.cluster_only_precipitation,
        plot_cluster_timeseries=config.plot_cluster_timeseries,
        cluster_timeseries_plot_limit=config.cluster_timeseries_plot_limit,
    )
    split_data, _daily_splits = create_window_split_data(
        dataframe,
        base_config,
        feature_columns,
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
        lstm_precipitation_transform=bool(
            parameters["lstm_precipitation_transform"]
        ),
    )
    assignment_path = final_dir / str(metadata["shared_assignment"])
    result_path = run_dir / str(metadata.get("result_file", "result.npz"))
    with np.load(assignment_path, allow_pickle=False) as labels, np.load(
        result_path, allow_pickle=False
    ) as result:
        c_train = np.asarray(labels["c_train"], dtype=int)
        c_val = np.asarray(labels["c_val"], dtype=int)
        c_test = np.asarray(labels["c_test"], dtype=int)
        histories = decode_histories(result, dict(metadata["history_index"]))
        y_pred_train = np.asarray(result["y_pred_train"], dtype=float)
        y_pred_train_by_lead_day = np.asarray(
            result["y_pred_train_by_lead_day"], dtype=float
        )
        y_pred_val = np.asarray(result["y_pred_val"], dtype=float)
        y_pred_test = np.asarray(result["y_pred_test"], dtype=float)
        y_pred_test_by_lead_day = np.asarray(
            result["y_pred_test_by_lead_day"], dtype=float
        )

    metrics_by_cluster = {
        cluster_id: calculate_regression_metrics(
            split_data.y_test[c_test == cluster_id],
            y_pred_test[c_test == cluster_id],
        )
        for cluster_id in sorted(int(value) for value in np.unique(c_train))
        if np.any(c_test == cluster_id)
    }
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else run_dir / "reconstructed_artifacts"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_labels = np.concatenate([c_train, c_val, c_test])
    save_run_outputs(
        config,
        output_dir,
        feature_columns,
        split_data.all_targets,
        split_data.all_current_precipitation,
        all_labels,
        split_data.y_train,
        split_data.y_val,
        split_data.y_test,
        split_data.y_test_by_lead_day,
        split_data.current_train,
        split_data.current_val,
        split_data.current_test,
        y_pred_train,
        y_pred_val,
        y_pred_test,
        c_test,
        split_data.i_train,
        split_data.i_val,
        split_data.i_test,
        histories,
        metrics_by_cluster,
        state=config.state,
        station_id=config.station_id,
        pca_variance_threshold=parameters["variance_threshold"],
        pca_for_clustering_only=bool(parameters["pca_for_clustering_only"]),
        forecast_horizon=int(parameters["forecast_horizon"]),
        y_pred_test_by_lead_day=y_pred_test_by_lead_day,
        test_target_dates_by_lead_day=split_data.test_target_dates_by_lead_day,
        cluster_feature_splits={
            "Training": (split_data.cluster_X_train, c_train),
            "Validation": (split_data.cluster_X_val, c_val),
            "Test": (split_data.cluster_X_test, c_test),
        },
        batch_size=int(parameters["batch_size"]),
        train_targets_by_lead_day=split_data.y_train_by_lead_day,
        y_pred_train_by_lead_day=y_pred_train_by_lead_day,
        train_target_dates_by_lead_day=split_data.train_target_dates_by_lead_day,
        train_cluster_labels=c_train,
        input_window_mean_precipitation_train=split_data.input_window_mean_train,
        input_window_mean_precipitation_val=split_data.input_window_mean_val,
        input_window_mean_precipitation_test=split_data.input_window_mean_test,
        test_input_precipitation_windows=split_data.test_input_precipitation_windows,
        train_info=bool(parameters["train_info"]),
        silhouette_info=bool(parameters["silhouette_info"]),
    )
    generate_config_report(
        output_dir,
        dict(metadata["report_config"]),
        compile_pdf=True,
    )
    return output_dir


def config_as_json(config: ExperimentConfig) -> dict[str, object]:
    """Return a JSON-compatible configuration mapping."""
    return dict(asdict(config))
