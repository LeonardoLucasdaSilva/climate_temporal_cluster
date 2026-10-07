"""Run the two-stage, validation-only PyTorch random-search framework."""

from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Callable, Mapping

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import OUTPUTS_BASE_DIR  # noqa: E402
from data.load_data import load_station_daily_data  # noqa: E402
from data.random_search_outputs import (  # noqa: E402
    aggregate_confirmation_results,
    rank_screening_results,
    retain_best_spectral_sigma,
    save_final_validation_rankings,
)
from methods.cluster.cluster_pipeline import numeric_feature_columns  # noqa: E402
from methods.lstm_cluster_pytorch.pipeline import (  # noqa: E402
    _training_sigma_values,
    run_experiment,
)
from methods.tools.random_search import (  # noqa: E402
    generate_parameter_configurations,
    parameter_combination_count,
)


# Add or remove values here. Clustering is intentionally controlled separately.
WINDOW_SIZES = [15, 30, 45]
LSTM_UNITS = [64, 128, 256]
LSTM_UNITS_2 = [None, 32, 64]
DROPOUT_RATES = [0.1, 0.2, 0.3]
LEARNING_RATES = [1e-4, 5e-4, 1e-3]
WEIGHT_DECAYS = [1e-5, 1e-4, 1e-3]
EPOCHS = [300]
BATCH_SIZES = [4, 8]
PATIENCE = [30, 50]
WARM_UP = [5, 10]
LOSS_QUANTILE_SETS = [(0.8,), (0.9,), (0.75, 0.9, 0.95, 0.99)]

SEARCH_SPACE: dict[str, list[object]] = {
    "window_size": WINDOW_SIZES,
    "lstm_units": LSTM_UNITS,
    "lstm_units_2": LSTM_UNITS_2,
    "dropout_rate": DROPOUT_RATES,
    "learning_rate": LEARNING_RATES,
    "weight_decay": WEIGHT_DECAYS,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZES,
    "patience": PATIENCE,
    "warm_up": WARM_UP,
    "loss_quantiles": LOSS_QUANTILE_SETS,
}

# Stage 1: random configurations and the fixed clustering expansion.
N_RANDOM_CONFIGURATIONS = 20
SCREENING_SEED = 42
CLUSTER_COUNTS = [1, 3, 5, 7, 9]
CLUSTERING_ALGORITHMS = ["kmeans", "spectral"]
N_AUTOMATIC_SIGMAS = 5
TOP_FRACTION = 0.10

# Stage 2: every retained exact configuration is rerun for all these seeds.
CONFIRMATION_SEEDS = [43, 44, 45, 46, 47]

# Runtime and output settings.
REQUIRE_GPU = True
PARALLEL_TRAINING = False
FAIL_FAST = False
SHOW_PROGRESS = True
COMPILE_FINAL_RANKING_PDF = True
SEARCH_NAME: str | None = None
SEARCH_NAME_PREFIX = "two_stage_random_search"
OUTPUT_ROOT = OUTPUTS_BASE_DIR / "pytorch" / "random_search"


PIPELINE_PARAMETER_MAP = {
    "window_size": "window_sizes",
    "lstm_units": "lstm_units",
    "lstm_units_2": "lstm_units_2",
    "dropout_rate": "dropout_rate",
    "learning_rate": "learning_rate",
    "weight_decay": "weight_decay",
    "epochs": "epochs",
    "batch_size": "batch_size",
    "patience": "patience",
    "warm_up": "warm_up",
    "loss_quantiles": "loss_quantiles",
}


def _validate_framework_settings() -> None:
    if CLUSTER_COUNTS != [1, 3, 5, 7, 9]:
        raise ValueError("CLUSTER_COUNTS must be exactly [1, 3, 5, 7, 9].")
    if CLUSTERING_ALGORITHMS != ["kmeans", "spectral"]:
        raise ValueError(
            "CLUSTERING_ALGORITHMS must be exactly ['kmeans', 'spectral']."
        )
    if isinstance(N_AUTOMATIC_SIGMAS, bool) or N_AUTOMATIC_SIGMAS <= 0:
        raise ValueError("N_AUTOMATIC_SIGMAS must be positive.")
    if not 0 < TOP_FRACTION <= 1:
        raise ValueError("TOP_FRACTION must be in the interval (0, 1].")
    if isinstance(SCREENING_SEED, bool) or not isinstance(SCREENING_SEED, int):
        raise ValueError("SCREENING_SEED must be an integer.")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int)
        for seed in CONFIRMATION_SEEDS
    ):
        raise ValueError("Every confirmation seed must be an integer.")
    if len(set(CONFIRMATION_SEEDS)) < 2:
        raise ValueError("CONFIRMATION_SEEDS must contain at least two seeds.")
    if SCREENING_SEED in CONFIRMATION_SEEDS:
        raise ValueError("Confirmation seeds must differ from SCREENING_SEED.")
    parameter_combination_count(SEARCH_SPACE)


def _verify_pytorch_runtime() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(f"PyTorch is unavailable in {sys.executable}.") from exc
    if REQUIRE_GPU and not torch.cuda.is_available():
        raise RuntimeError("The random-search framework requires a CUDA GPU.")
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(
        f"PyTorch {torch.__version__} | deterministic=True | device={device}",
        flush=True,
    )


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_remove_tree(path: Path, root: Path) -> None:
    resolved_path = Path(path).resolve()
    resolved_root = Path(root).resolve()
    if resolved_path == resolved_root or not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Refusing to remove path outside search output: {path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def _search_id() -> str:
    return SEARCH_NAME or (
        f"{SEARCH_NAME_PREFIX}_{datetime.now().strftime('%Y_%m_%d_%Hh%M%S')}"
    )


def _sampled_configurations() -> list[dict[str, object]]:
    sampled = generate_parameter_configurations(
        SEARCH_SPACE,
        strategy="random",
        n_trials=N_RANDOM_CONFIGURATIONS,
        random_state=SCREENING_SEED,
    )
    return [dict(configuration) for configuration in sampled]


def _configuration_id(index: int) -> str:
    return f"random_{index:04d}"


def _sigma_token(value: object) -> str:
    if pd.isna(value):
        return "none"
    return f"{float(value):.12g}".replace("-", "m").replace(".", "p")


def _candidate_id(row: Mapping[str, object]) -> str:
    random_id = str(row["random_configuration_id"])
    k = int(row["n_clusters"])
    if k == 1:
        return f"{random_id}__k01_baseline"
    algorithm = str(row["algorithm"])
    if algorithm == "spectral":
        return f"{random_id}__k{k:02d}_spectral_s{_sigma_token(row['sigma'])}"
    return f"{random_id}__k{k:02d}_{algorithm}"


def _base_kwargs() -> dict[str, object]:
    from methods.lstm_cluster_pytorch import run_experiment as settings

    parameters = settings.experiment_kwargs()
    parameters.update(
        {
            "n_clusters_list": list(CLUSTER_COUNTS),
            "clustering_algorithm": list(CLUSTERING_ALGORITHMS),
            "n_sigma_values": N_AUTOMATIC_SIGMAS,
            "test_all_models": False,
            "run_only_cluster": False,
            "comparative_run": False,
            "create_report": False,
            "train_info": False,
            "silhouette_info": False,
            "plot_cluster_timeseries": False,
            "parallel_training": PARALLEL_TRAINING,
            "require_gpu": REQUIRE_GPU,
            "validation_only": True,
            "save_validation_history": False,
            "verbose_training": 0,
            "show_console_info": False,
        }
    )
    return parameters


def _apply_hyperparameters(
    base: Mapping[str, object],
    configuration: Mapping[str, object],
) -> dict[str, object]:
    parameters = dict(base)
    unsupported = sorted(set(configuration) - set(PIPELINE_PARAMETER_MAP))
    if unsupported:
        raise ValueError("Unsupported random parameter(s): " + ", ".join(unsupported))
    for name, value in configuration.items():
        pipeline_name = PIPELINE_PARAMETER_MAP[name]
        if name == "window_size":
            parameters[pipeline_name] = [int(value)]
        elif name == "loss_quantiles":
            parameters[pipeline_name] = [float(item) for item in value]
        else:
            parameters[pipeline_name] = value
    return parameters


def _canonical_quantiles(value: object) -> str:
    if isinstance(value, str):
        return value
    return ",".join(f"{float(item):g}" for item in value)


def _read_results(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _automatic_sigma_grids(
    sampled: list[dict[str, object]],
    base: Mapping[str, object],
) -> dict[int, list[float]]:
    dataframe = load_station_daily_data(
        state=str(base["state"]),
        station_id=str(base["station_id"]),
        data_root=Path(base["data_root"]),
    )
    features = numeric_feature_columns(dataframe)
    grids: dict[int, list[float]] = {}
    for window_size in sorted({int(row["window_size"]) for row in sampled}):
        grids[window_size] = _training_sigma_values(
            dataframe,
            features,
            window_size=window_size,
            window_stride=int(base["window_stride"]),
            forecast_horizon=int(base["forecast_horizon"]),
            train_ratio=float(base["train_ratio"]),
            val_ratio=float(base["val_ratio"]),
            clustering_feature_normalize=base["clustering_feature_normalize"],
            clustering_precipitation_normalize=(
                base["clustering_precipitation_normalize"]
            ),
            variance_threshold=base["variance_threshold"],
            cluster_only_precipitation=bool(base["cluster_only_precipitation"]),
            n_values=N_AUTOMATIC_SIGMAS,
        ).tolist()
    return grids


def _annotate_screening_rows(
    frame: pd.DataFrame,
    *,
    random_id: str,
    configuration: Mapping[str, object],
    sweep_dir: Path,
    search_dir: Path,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["random_configuration_id"] = random_id
    frame["screening_seed"] = SCREENING_SEED
    for name, value in configuration.items():
        frame[name] = (
            _canonical_quantiles(value) if name == "loss_quantiles" else value
        )
    frame["candidate_id"] = [
        _candidate_id(row) for row in frame.to_dict(orient="records")
    ]
    frame["configuration_directory"] = [
        (sweep_dir / str(run_name)).relative_to(search_dir).as_posix()
        for run_name in frame["run_name"]
    ]
    return frame


def _run_stage1_configuration(
    *,
    index: int,
    configuration: Mapping[str, object],
    sigma_grids: Mapping[int, list[float]],
    search_dir: Path,
    base: Mapping[str, object],
    experiment_runner: Callable[..., Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    random_id = _configuration_id(index)
    stage_root = search_dir / "stage1_screening"
    sweep_dir = stage_root / random_id
    retained_path = sweep_dir / "retained_clustering_results.csv"
    audit_path = sweep_dir / "all_sigma_results.csv"
    marker = sweep_dir / "stage1_complete.json"
    if marker.is_file() and retained_path.is_file() and audit_path.is_file():
        return _read_results(retained_path), _read_results(audit_path)
    if sweep_dir.exists():
        _safe_remove_tree(sweep_dir, search_dir)

    parameters = _apply_hyperparameters(base, configuration)
    parameters.update(
        {
            "sigma_values": sigma_grids[int(configuration["window_size"])],
            "random_state": SCREENING_SEED,
            "output_root": stage_root,
            "sweep_name": random_id,
            "save_validation_history": False,
        }
    )
    completed_dir = Path(experiment_runner(**parameters))
    all_results = _read_results(completed_dir / "sweep_results.csv")
    if all_results.empty:
        raise RuntimeError(f"No valid screening results were produced for {random_id}.")
    all_results = _annotate_screening_rows(
        all_results,
        random_id=random_id,
        configuration=configuration,
        sweep_dir=completed_dir,
        search_dir=search_dir,
    )
    retained = retain_best_spectral_sigma(all_results)
    retained_names = set(retained["run_name"].astype(str))
    for run_name in all_results.loc[
        all_results["algorithm"].astype(str).eq("spectral"), "run_name"
    ].astype(str):
        if run_name not in retained_names:
            _safe_remove_tree(completed_dir / run_name, search_dir)
    all_results.to_csv(audit_path, index=False)
    retained.to_csv(retained_path, index=False)
    _write_json(
        marker,
        {
            "random_configuration_id": random_id,
            "retained_rows": len(retained),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return retained, all_results


def _confirmation_parameters(
    row: Mapping[str, object],
    configuration: Mapping[str, object],
    *,
    seed: int,
    candidate_dir: Path,
    base: Mapping[str, object],
) -> dict[str, object]:
    parameters = _apply_hyperparameters(base, configuration)
    algorithm = str(row["algorithm"])
    parameters.update(
        {
            "n_clusters_list": [int(row["n_clusters"])],
            "clustering_algorithm": algorithm,
            "sigma_values": (
                [float(row["sigma"])] if algorithm == "spectral" else None
            ),
            "random_state": int(seed),
            "output_root": candidate_dir,
            "sweep_name": f"seed_{seed}",
            "save_validation_history": True,
        }
    )
    return parameters


def _run_confirmation_seed(
    row: Mapping[str, object],
    configuration: Mapping[str, object],
    *,
    seed: int,
    search_dir: Path,
    base: Mapping[str, object],
    experiment_runner: Callable[..., Path],
) -> dict[str, object]:
    candidate_id = str(row["candidate_id"])
    candidate_dir = search_dir / "stage2_confirmation" / candidate_id
    sweep_dir = candidate_dir / f"seed_{seed}"
    marker = sweep_dir / "confirmation_complete.json"
    if marker.is_file() and (sweep_dir / "sweep_results.csv").is_file():
        result_frame = _read_results(sweep_dir / "sweep_results.csv")
    else:
        if sweep_dir.exists():
            _safe_remove_tree(sweep_dir, search_dir)
        parameters = _confirmation_parameters(
            row,
            configuration,
            seed=seed,
            candidate_dir=candidate_dir,
            base=base,
        )
        completed_dir = Path(experiment_runner(**parameters))
        result_frame = _read_results(completed_dir / "sweep_results.csv")
        if len(result_frame) != 1:
            raise RuntimeError(
                f"Expected one confirmation result for {candidate_id}/seed_{seed}; "
                f"found {len(result_frame)}."
            )
        _write_json(
            marker,
            {
                "candidate_id": candidate_id,
                "seed": seed,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
    result = result_frame.iloc[0].to_dict()
    for key, value in row.items():
        if key not in {"val_mae", "screening_rank", "retained_for_confirmation"}:
            result[key] = value
    result["confirmation_seed"] = int(seed)
    result["confirmation_directory"] = sweep_dir.relative_to(search_dir).as_posix()
    return result


def _prune_bottom_screening_artifacts(
    rankings: pd.DataFrame,
    retained: pd.DataFrame,
    search_dir: Path,
) -> None:
    keep_ids = set(retained["candidate_id"].astype(str))
    for row in rankings.to_dict(orient="records"):
        if str(row["candidate_id"]) in keep_ids:
            continue
        directory = search_dir / str(row["configuration_directory"])
        _safe_remove_tree(directory, search_dir)


def _design_payload(
    sampled: list[dict[str, object]],
) -> dict[str, object]:
    clustered_k_count = len(CLUSTER_COUNTS) - 1
    screening_runs_per_random = (
        1 + clustered_k_count + clustered_k_count * N_AUTOMATIC_SIGMAS
    )
    retained_rows_per_random = 1 + 2 * clustered_k_count
    expected_screening_rows = len(sampled) * retained_rows_per_random
    expected_top_count = max(1, math.ceil(expected_screening_rows * TOP_FRACTION))
    return {
        "schema_version": 1,
        "framework": "two_stage_pytorch_validation_random_search",
        "search_space": SEARCH_SPACE,
        "sampled_configurations": sampled,
        "n_random_configurations": N_RANDOM_CONFIGURATIONS,
        "screening_seed": SCREENING_SEED,
        "cluster_counts": CLUSTER_COUNTS,
        "clustering_algorithms": CLUSTERING_ALGORITHMS,
        "automatic_sigma_candidates": N_AUTOMATIC_SIGMAS,
        "top_fraction": TOP_FRACTION,
        "confirmation_seeds": CONFIRMATION_SEEDS,
        "planned_training_runs": {
            "screening": len(sampled) * screening_runs_per_random,
            "confirmation_if_all_valid": (
                expected_top_count * len(CONFIRMATION_SEEDS)
            ),
        },
        "validation_metric": "MAE",
        "test_evaluation": False,
    }


def _ensure_design(search_dir: Path, payload: Mapping[str, object]) -> None:
    path = search_dir / "framework_design.json"
    serialized = _json_safe(dict(payload))
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != serialized:
            raise ValueError(
                "The existing SEARCH_NAME uses a different framework design. "
                "Choose a new SEARCH_NAME or restore the original settings."
            )
        return
    _write_json(path, payload)


def run_framework(
    *,
    experiment_runner: Callable[..., Path] = run_experiment,
    search_id: str | None = None,
) -> Path:
    """Run or resume both validation-only search stages."""
    _validate_framework_settings()
    sampled = _sampled_configurations()
    search_dir = OUTPUT_ROOT / (search_id or _search_id())
    search_dir.mkdir(parents=True, exist_ok=True)
    _ensure_design(search_dir, _design_payload(sampled))

    sampled_rows = [
        {"random_configuration_id": _configuration_id(index), **configuration}
        for index, configuration in enumerate(sampled, start=1)
    ]
    pd.DataFrame(sampled_rows).to_csv(
        search_dir / "sampled_hyperparameter_configurations.csv",
        index=False,
    )
    base = _base_kwargs()
    sigma_path = search_dir / "automatic_sigma_values.json"
    if sigma_path.is_file():
        sigma_grids = {
            int(window): [float(value) for value in values]
            for window, values in json.loads(
                sigma_path.read_text(encoding="utf-8")
            ).items()
        }
    else:
        sigma_grids = _automatic_sigma_grids(sampled, base)
        _write_json(
            sigma_path,
            {str(window): values for window, values in sigma_grids.items()},
        )
    required_windows = {int(row["window_size"]) for row in sampled}
    missing_sigma_windows = sorted(required_windows - set(sigma_grids))
    if missing_sigma_windows:
        raise ValueError(
            "automatic_sigma_values.json is missing window(s): "
            + ", ".join(str(value) for value in missing_sigma_windows)
        )

    stage1_frames: list[pd.DataFrame] = []
    sigma_audits: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    for index, configuration in enumerate(sampled, start=1):
        random_id = _configuration_id(index)
        if SHOW_PROGRESS:
            print(
                f"Stage 1 random configuration {index}/{len(sampled)}: "
                f"{configuration}",
                flush=True,
            )
        try:
            retained, audit = _run_stage1_configuration(
                index=index,
                configuration=configuration,
                sigma_grids=sigma_grids,
                search_dir=search_dir,
                base=base,
                experiment_runner=experiment_runner,
            )
            stage1_frames.append(retained)
            sigma_audits.append(audit)
        except Exception as exc:
            failures.append(
                {
                    "stage": "screening",
                    "random_configuration_id": random_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if FAIL_FAST:
                raise
            print(f"  Failed: {failures[-1]['error']}", flush=True)

    stage1_results = (
        pd.concat(stage1_frames, ignore_index=True, sort=False)
        if stage1_frames
        else pd.DataFrame()
    )
    sigma_audit = (
        pd.concat(sigma_audits, ignore_index=True, sort=False)
        if sigma_audits
        else pd.DataFrame()
    )
    sigma_audit.to_csv(search_dir / "stage1_all_sigma_metrics.csv", index=False)
    if stage1_results.empty:
        pd.DataFrame(failures).to_csv(search_dir / "failures.csv", index=False)
        raise RuntimeError("Stage 1 finished without a valid configuration.")
    stage1_rankings, retained_top = rank_screening_results(
        stage1_results,
        retain_fraction=TOP_FRACTION,
    )
    stage1_rankings.to_csv(
        search_dir / "stage1_screening_rankings.csv",
        index=False,
    )
    retained_top.to_csv(
        search_dir / "stage1_retained_top_10_percent.csv",
        index=False,
    )
    _prune_bottom_screening_artifacts(stage1_rankings, retained_top, search_dir)

    sampled_by_id = {
        _configuration_id(index): configuration
        for index, configuration in enumerate(sampled, start=1)
    }
    seed_results: list[dict[str, object]] = []
    for candidate_index, row in enumerate(
        retained_top.to_dict(orient="records"),
        start=1,
    ):
        candidate_id = str(row["candidate_id"])
        random_id = str(row["random_configuration_id"])
        configuration = sampled_by_id[random_id]
        candidate_dir = search_dir / "stage2_confirmation" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(
            candidate_dir / "selected_screening_configuration.csv",
            index=False,
        )
        for seed in CONFIRMATION_SEEDS:
            if SHOW_PROGRESS:
                print(
                    f"Stage 2 candidate {candidate_index}/{len(retained_top)}, "
                    f"seed={seed}: {candidate_id}",
                    flush=True,
                )
            try:
                seed_results.append(
                    _run_confirmation_seed(
                        row,
                        configuration,
                        seed=seed,
                        search_dir=search_dir,
                        base=base,
                        experiment_runner=experiment_runner,
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "stage": "confirmation",
                        "candidate_id": candidate_id,
                        "seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if FAIL_FAST:
                    raise
                print(f"  Failed: {failures[-1]['error']}", flush=True)
            pd.DataFrame(seed_results).to_csv(
                search_dir / "confirmation_seed_metrics.csv",
                index=False,
            )

    seed_frame = pd.DataFrame(seed_results)
    seed_frame.to_csv(search_dir / "confirmation_seed_metrics.csv", index=False)
    final_rankings = aggregate_confirmation_results(
        seed_frame,
        expected_seeds=CONFIRMATION_SEEDS,
    )
    save_final_validation_rankings(
        final_rankings,
        search_dir,
        compile_pdf=COMPILE_FINAL_RANKING_PDF,
    )
    pd.DataFrame(failures).to_csv(search_dir / "failures.csv", index=False)
    _write_json(
        search_dir / "framework_complete.json",
        {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "screened_candidates": len(stage1_rankings),
            "retained_candidates": len(retained_top),
            "fully_confirmed_candidates": len(final_rankings),
            "failures": len(failures),
        },
    )
    print(f"Random-search results: {search_dir}", flush=True)
    return search_dir


def main() -> None:
    """Verify the runtime and run or resume the configured framework."""
    _verify_pytorch_runtime()
    run_framework()


if __name__ == "__main__":
    main()
