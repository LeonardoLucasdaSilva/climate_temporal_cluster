"""Output writers for LSTM hyperparameter tuning runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _json_value(value: object) -> object:
    """Return a JSON-safe scalar while preserving missing values as null."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def tuning_results_dataframe(
    results: Sequence[Mapping[str, object]],
    *,
    direction: str,
) -> pd.DataFrame:
    """Return trial results with completed trials ordered by objective."""
    frame = pd.DataFrame(results)
    if frame.empty:
        return frame

    frame = frame.copy()
    status_rank = frame.get(
        "status",
        pd.Series("failed", index=frame.index),
    ).map({"completed": 0, "failed": 1}).fillna(2)
    frame["_status_rank"] = status_rank
    sort_columns = ["_status_rank"]
    ascending = [True]
    if "objective_value" in frame:
        sort_columns.append("objective_value")
        ascending.append(direction == "minimize")
    if "trial" in frame:
        sort_columns.append("trial")
        ascending.append(True)
    return frame.sort_values(
        sort_columns,
        ascending=ascending,
        na_position="last",
    ).drop(columns="_status_rank")


def best_completed_trial(
    results: Sequence[Mapping[str, object]],
    *,
    direction: str,
) -> dict[str, object] | None:
    """Return the best completed trial with a finite objective value."""
    frame = tuning_results_dataframe(results, direction=direction)
    if frame.empty or "status" not in frame or "objective_value" not in frame:
        return None
    completed = frame.loc[frame["status"] == "completed"].copy()
    completed["objective_value"] = pd.to_numeric(
        completed["objective_value"],
        errors="coerce",
    )
    completed = completed.loc[np.isfinite(completed["objective_value"])]
    if completed.empty:
        return None
    return {
        str(key): _json_value(value)
        for key, value in completed.iloc[0].to_dict().items()
    }


def save_hyperparameter_tuning_outputs(
    results: Sequence[Mapping[str, object]],
    *,
    output_dir: Path,
    hyperparameter_names: Sequence[str],
    optimization_metric: str,
    objective_column: str,
    direction: str,
    search_strategy: str,
    total_candidates: int,
    requested_trials: int | None,
) -> dict[str, object] | None:
    """Checkpoint tuning results and return the best completed trial."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df = tuning_results_dataframe(results, direction=direction)
    results_df.to_csv(output_dir / "tuning_results.csv", index=False)

    best = best_completed_trial(results, direction=direction)
    if best is not None:
        best_payload = {
            "trial": best.get("trial"),
            "trial_name": best.get("trial_name"),
            "trial_directory": best.get("trial_directory"),
            "optimization_metric": optimization_metric,
            "objective_column": objective_column,
            "direction": direction,
            "objective_value": best.get("objective_value"),
            "hyperparameters": {
                name: best.get(name) for name in hyperparameter_names
            },
        }
        (output_dir / "best_hyperparameters.json").write_text(
            json.dumps(best_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    completed_count = sum(
        1 for result in results if result.get("status") == "completed"
    )
    failed_count = sum(1 for result in results if result.get("status") == "failed")
    requested_text = "all" if requested_trials is None else str(requested_trials)
    summary_lines = [
        "LSTM HYPERPARAMETER TUNING SUMMARY",
        "=" * 72,
        "",
        f"Updated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Search strategy: {search_strategy}",
        f"Candidate combinations: {total_candidates}",
        f"Requested trials: {requested_text}",
        f"Recorded trials: {len(results)}",
        f"Completed trials: {completed_count}",
        f"Failed trials: {failed_count}",
        f"Optimization metric: validation {optimization_metric}",
        f"Objective column: {objective_column}",
        f"Direction: {direction}",
        f"Tuned hyperparameters: {', '.join(hyperparameter_names)}",
        "",
    ]
    if best is None:
        summary_lines.append("Best trial: unavailable (no completed finite trial).")
    else:
        summary_lines.extend(
            [
                f"Best trial: {best.get('trial_name')}",
                f"Best objective: {best.get('objective_value')}",
                "Best hyperparameters:",
                *[
                    f"  {name}: {best.get(name)}"
                    for name in hyperparameter_names
                ],
            ]
        )
    (output_dir / "tuning_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    return best
