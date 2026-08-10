"""Budgeted hyperparameter tuning for the LSTM cluster experiment."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from itertools import product
import math
from pathlib import Path
import random
import sys
from time import monotonic

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from data.hyperparam_tuning_outputs import save_hyperparameter_tuning_outputs


# Candidate values. Remove entries from HYPERPARAMETER_SPACE to keep them fixed
# at the first value configured in run_experiment.py.
WINDOW_SIZES = [15, 30, 45]
N_CLUSTERS_LIST = [3, 5, 7]
LSTM_UNITS = [64, 128, 256,512]
LSTM_UNITS_2 = [None, 32, 64, 128]
DROPOUT_RATE = [0.1, 0.2, 0.3]
LEARNING_RATE = [1e-3]
WEIGHT_DECAY = [1e-3,1e-4,1e-5]

HYPERPARAMETER_SPACE: dict[str, Sequence[int | float | None]] = {
    "window_size": WINDOW_SIZES,
    "n_clusters": N_CLUSTERS_LIST,
    "lstm_units": LSTM_UNITS,
    "lstm_units_2": LSTM_UNITS_2,
    "dropout_rate": DROPOUT_RATE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
}

# Search and selection settings.
SEARCH_STRATEGY = "random"                       # "random" or "grid"
N_TRIALS: int | None = 20                       # None evaluates every combination
OPTIMIZATION_METRIC = "RMSE"                   # Validation MSE, RMSE, MAE, R2, RMSLE, or MAPE
RANDOM_STATE = 42
FAIL_FAST = False                               # Continue after an invalid/failed candidate
SHOW_TUNING_PROGRESS = True

# Output settings. Trial folders are created inside this tuning folder.
TUNING_NAME: str | None = None
TUNING_NAME_PREFIX = "lstm_cluster_tuning"


TUNABLE_PIPELINE_PARAMETERS = {
    "window_size": "window_sizes",
    "n_clusters": "n_clusters_list",
    "lstm_units": "lstm_units",
    "lstm_units_2": "lstm_units_2",
    "dropout_rate": "dropout_rate",
    "learning_rate": "learning_rate",
    "weight_decay": "weight_decay",
}
SUPPORTED_OPTIMIZATION_METRICS = {
    "MSE",
    "RMSE",
    "MAE",
    "R2",
    "RMSLE",
    "MAPE",
}
PIPELINE_GRID_PARAMETERS = (
    "window_sizes",
    "n_clusters_list",
    "lstm_units",
    "lstm_units_2",
    "dropout_rate",
    "learning_rate",
    "weight_decay",
    "epochs",
    "batch_size",
    "patience",
    "warm_up",
)
LIST_WRAPPED_PIPELINE_PARAMETERS = {"window_sizes", "n_clusters_list"}


def _first_configured_value(value: object, setting_name: str) -> object:
    """Return the first configured scalar for a single tuning trial."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError(f"{setting_name} must contain at least one value.")
        return value[0]
    return value


def _validated_search_space(
    search_space: Mapping[str, Sequence[int | float | None]],
) -> dict[str, list[int | float | None]]:
    """Return a validated, insertion-ordered tuning search space."""
    if not search_space:
        raise ValueError("HYPERPARAMETER_SPACE must contain at least one parameter.")
    unsupported = sorted(set(search_space) - set(TUNABLE_PIPELINE_PARAMETERS))
    if unsupported:
        raise ValueError(
            "Unsupported tuning hyperparameter(s): " + ", ".join(unsupported)
        )

    normalized: dict[str, list[int | float | None]] = {}
    for parameter, raw_values in search_space.items():
        if isinstance(raw_values, (str, bytes)):
            raise ValueError(f"{parameter} candidates must be a sequence of values.")
        values = list(raw_values)
        if not values:
            raise ValueError(f"{parameter} must contain at least one candidate.")
        if len(values) != len(set(values)):
            raise ValueError(f"{parameter} cannot contain duplicate candidates.")
        normalized[parameter] = values
    return normalized


def candidate_count(
    search_space: Mapping[str, Sequence[int | float | None]],
) -> int:
    """Return the size of the Cartesian candidate space."""
    validated = _validated_search_space(search_space)
    return math.prod(len(values) for values in validated.values())


def generate_trial_hyperparameters(
    search_space: Mapping[str, Sequence[int | float | None]],
    *,
    strategy: str,
    n_trials: int | None,
    random_state: int,
) -> list[dict[str, int | float | None]]:
    """Return deterministic grid or random-search trial configurations."""
    validated = _validated_search_space(search_space)
    normalized_strategy = str(strategy).strip().lower()
    if normalized_strategy not in {"grid", "random"}:
        raise ValueError("SEARCH_STRATEGY must be either 'grid' or 'random'.")
    if n_trials is not None:
        if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials <= 0:
            raise ValueError("N_TRIALS must be a positive integer or None.")

    parameter_names = list(validated)
    combinations = [
        dict(zip(parameter_names, values))
        for values in product(*(validated[name] for name in parameter_names))
    ]
    trial_count = len(combinations) if n_trials is None else min(
        n_trials,
        len(combinations),
    )
    if normalized_strategy == "grid":
        return combinations[:trial_count]
    return random.Random(random_state).sample(combinations, k=trial_count)


def optimization_spec(metric: str) -> tuple[str, str, str]:
    """Return canonical metric, validation result column, and direction."""
    normalized = str(metric).strip().upper().replace("VAL_", "")
    if normalized not in SUPPORTED_OPTIMIZATION_METRICS:
        supported = ", ".join(sorted(SUPPORTED_OPTIMIZATION_METRICS))
        raise ValueError(f"OPTIMIZATION_METRIC must be one of: {supported}.")
    return normalized, f"val_{normalized.lower()}", (
        "maximize" if normalized == "R2" else "minimize"
    )


def single_trial_base_kwargs(
    base_experiment_kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Collapse every existing experiment grid to one fixed base value."""
    parameters = dict(base_experiment_kwargs)
    if parameters.get("run_only_cluster"):
        raise ValueError("Hyperparameter tuning requires RUN_ONLY_CLUSTER=False.")

    for parameter in PIPELINE_GRID_PARAMETERS:
        if parameter not in parameters:
            raise ValueError(f"Missing base experiment parameter: {parameter}.")
        first_value = _first_configured_value(parameters[parameter], parameter)
        parameters[parameter] = (
            [first_value]
            if parameter in LIST_WRAPPED_PIPELINE_PARAMETERS
            else first_value
        )

    if "clustering_algorithm" in parameters:
        parameters["clustering_algorithm"] = _first_configured_value(
            parameters["clustering_algorithm"],
            "clustering_algorithm",
        )
    sigma_values = parameters.get("sigma_values")
    if sigma_values is not None:
        parameters["sigma_values"] = [
            _first_configured_value(sigma_values, "sigma_values")
        ]

    parameters["comparative_run"] = False
    parameters["run_only_cluster"] = False
    return parameters


def _trial_pipeline_kwargs(
    base_experiment_kwargs: Mapping[str, object],
    hyperparameters: Mapping[str, int | float | None],
    *,
    output_root: Path,
    trial_name: str,
) -> dict[str, object]:
    """Return one exact pipeline configuration for a tuning trial."""
    parameters = single_trial_base_kwargs(base_experiment_kwargs)
    for hyperparameter, value in hyperparameters.items():
        pipeline_parameter = TUNABLE_PIPELINE_PARAMETERS[hyperparameter]
        parameters[pipeline_parameter] = (
            [value]
            if pipeline_parameter in LIST_WRAPPED_PIPELINE_PARAMETERS
            else value
        )
    parameters["output_root"] = Path(output_root)
    parameters["sweep_name"] = trial_name
    return parameters


def _read_single_trial_result(sweep_dir: Path) -> dict[str, object]:
    """Read the unique result row written by a tuning trial."""
    results_path = Path(sweep_dir) / "sweep_results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Trial did not create {results_path}.")
    results = pd.read_csv(results_path)
    if len(results) != 1:
        raise ValueError(
            "Every tuning trial must produce exactly one configuration; "
            f"found {len(results)} in {results_path}."
        )
    return results.iloc[0].to_dict()


def run_hyperparameter_tuning(
    *,
    base_experiment_kwargs: Mapping[str, object],
    search_space: Mapping[str, Sequence[int | float | None]],
    output_dir: Path,
    strategy: str = "random",
    n_trials: int | None = 20,
    optimization_metric: str = "RMSE",
    random_state: int = 42,
    fail_fast: bool = False,
    show_progress: bool = True,
    experiment_runner: Callable[..., Path] | None = None,
) -> Path:
    """Run budgeted tuning and select the best validation configuration."""
    if experiment_runner is None:
        from methods.lstm_cluster.pipeline import run_experiment

        experiment_runner = run_experiment

    validated_space = _validated_search_space(search_space)
    trials = generate_trial_hyperparameters(
        validated_space,
        strategy=strategy,
        n_trials=n_trials,
        random_state=random_state,
    )
    metric, objective_column, direction = optimization_spec(optimization_metric)
    total_candidates = candidate_count(validated_space)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Tuning output directory already exists and is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    writer_kwargs = {
        "output_dir": output_dir,
        "hyperparameter_names": list(validated_space),
        "optimization_metric": metric,
        "objective_column": objective_column,
        "direction": direction,
        "search_strategy": str(strategy).strip().lower(),
        "total_candidates": total_candidates,
        "requested_trials": n_trials,
    }
    save_hyperparameter_tuning_outputs(results, **writer_kwargs)

    for trial_number, hyperparameters in enumerate(trials, start=1):
        trial_name = f"trial_{trial_number:04d}"
        if show_progress:
            print(
                f"Tuning trial {trial_number}/{len(trials)}: "
                f"{hyperparameters}"
            )
        started_at = monotonic()
        trial_kwargs = _trial_pipeline_kwargs(
            base_experiment_kwargs,
            hyperparameters,
            output_root=output_dir,
            trial_name=trial_name,
        )
        try:
            sweep_dir = Path(experiment_runner(**trial_kwargs))
            pipeline_result = _read_single_trial_result(sweep_dir)
            objective_value = float(pipeline_result[objective_column])
            if not math.isfinite(objective_value):
                raise ValueError(
                    f"Trial objective {objective_column} is not finite: "
                    f"{objective_value}."
                )
            trial_result = {
                **pipeline_result,
                **hyperparameters,
                "trial": trial_number,
                "trial_name": trial_name,
                "status": "completed",
                "objective_value": objective_value,
                "duration_seconds": monotonic() - started_at,
                "trial_directory": str(sweep_dir),
                "error": None,
            }
        except Exception as exc:
            trial_result = {
                **hyperparameters,
                "trial": trial_number,
                "trial_name": trial_name,
                "status": "failed",
                "objective_value": None,
                "duration_seconds": monotonic() - started_at,
                "trial_directory": str(output_dir / trial_name),
                "error": f"{type(exc).__name__}: {exc}",
            }
            results.append(trial_result)
            save_hyperparameter_tuning_outputs(results, **writer_kwargs)
            if fail_fast:
                raise
            if show_progress:
                print(f"  Failed: {trial_result['error']}")
            continue

        results.append(trial_result)
        save_hyperparameter_tuning_outputs(results, **writer_kwargs)
        if show_progress:
            print(f"  Validation {metric}: {objective_value:.6g}")

    best = save_hyperparameter_tuning_outputs(results, **writer_kwargs)
    if best is None:
        raise RuntimeError(
            "Hyperparameter tuning finished without a successful finite trial. "
            f"See {output_dir / 'tuning_results.csv'}."
        )
    if show_progress:
        print(
            f"Best trial: {best['trial_name']} "
            f"(validation {metric}={best['objective_value']:.6g})"
        )
        print(f"Tuning results: {output_dir}")
    return output_dir


def main() -> None:
    """Tune the configured search space using run_experiment.py as the base."""
    from methods.lstm_cluster import run_experiment as base_settings

    base_experiment_kwargs = base_settings.experiment_kwargs()
    timestamp_format = str(
        base_experiment_kwargs.get("timestamp_format", "%Y%m%d_%H%M%S")
    )
    tuning_name = TUNING_NAME or (
        f"{TUNING_NAME_PREFIX}_{base_experiment_kwargs['state']}_"
        f"{base_experiment_kwargs['station_id']}_"
        f"{datetime.now().strftime(timestamp_format)}"
    )
    run_hyperparameter_tuning(
        base_experiment_kwargs=base_experiment_kwargs,
        search_space=HYPERPARAMETER_SPACE,
        output_dir=Path(base_experiment_kwargs["output_root"]) / tuning_name,
        strategy=SEARCH_STRATEGY,
        n_trials=N_TRIALS,
        optimization_metric=OPTIMIZATION_METRIC,
        random_state=RANDOM_STATE,
        fail_fast=FAIL_FAST,
        show_progress=SHOW_TUNING_PROGRESS,
    )


if __name__ == "__main__":
    main()
