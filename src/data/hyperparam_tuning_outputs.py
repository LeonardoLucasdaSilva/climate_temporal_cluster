"""Output writers for LSTM hyperparameter tuning runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from methods.lstm_cluster.report import compile_tex, latex_escape


VALIDATION_RANKING_STEM = "random_search_validation_rankings"
TEST_DERIVED_COLUMNS = {
    "n_test",
    "zero_days_ratio",
    "rainy_days_rmse",
    "lead_day_metrics_path",
}
TEST_DERIVED_PREFIXES = ("test_", "cluster_", "horizon_")


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


def validation_only_results_dataframe(
    results: Sequence[Mapping[str, object]],
    *,
    direction: str,
) -> pd.DataFrame:
    """Return ordered tuning results with every test-derived field removed."""
    frame = tuning_results_dataframe(results, direction=direction)
    excluded = [
        column
        for column in frame.columns
        if column in TEST_DERIVED_COLUMNS
        or column.startswith(TEST_DERIVED_PREFIXES)
    ]
    return frame.drop(columns=excluded, errors="ignore")


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
    ranking_dir: Path | None = None,
    search_id: str | None = None,
    compile_ranking_pdf: bool = False,
) -> dict[str, object] | None:
    """Checkpoint tuning results and return the best completed trial."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df = validation_only_results_dataframe(results, direction=direction)
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
    if ranking_dir is not None:
        if not search_id:
            raise ValueError("search_id is required when ranking_dir is provided.")
        save_validation_search_rankings(
            results_df,
            ranking_dir=ranking_dir,
            search_id=search_id,
            hyperparameter_names=hyperparameter_names,
            optimization_metric=optimization_metric,
            objective_column=objective_column,
            direction=direction,
            compile_pdf=compile_ranking_pdf,
        )
    return best


def save_validation_trial_outputs(
    result: Mapping[str, object],
    output_dir: Path,
) -> Path:
    """Write the train/validation metrics for one validation-only trial."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "validation_metrics.csv"
    validation_only_results_dataframe([result], direction="minimize").to_csv(
        path,
        index=False,
    )
    return path


def save_validation_sweep_outputs(
    results: Sequence[Mapping[str, object]],
    sweep_dir: Path,
) -> Path:
    """Write a validation-only sweep table without any test-derived columns."""
    sweep_dir = Path(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    path = sweep_dir / "sweep_results.csv"
    validation_only_results_dataframe(results, direction="minimize").to_csv(
        path,
        index=False,
    )
    (sweep_dir / "validation_only.txt").write_text(
        "This sweep did not predict or evaluate the test split.\n",
        encoding="utf-8",
    )
    return path


def save_validation_search_rankings(
    current_results: pd.DataFrame,
    *,
    ranking_dir: Path,
    search_id: str,
    hyperparameter_names: Sequence[str],
    optimization_metric: str,
    objective_column: str,
    direction: str,
    compile_pdf: bool,
) -> tuple[Path, Path, Path | None]:
    """Upsert one search into the shared validation-only ranking files."""
    ranking_dir = Path(ranking_dir)
    ranking_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ranking_dir / f"{VALIDATION_RANKING_STEM}.csv"
    tex_path = ranking_dir / f"{VALIDATION_RANKING_STEM}.tex"

    current = current_results.copy()
    current.insert(0, "search_id", str(search_id))
    current["optimization_metric"] = str(optimization_metric)
    current["objective_column"] = str(objective_column)
    current["direction"] = str(direction)
    existing = pd.read_csv(csv_path) if csv_path.is_file() else pd.DataFrame()
    if not existing.empty and "search_id" in existing:
        existing = existing.loc[existing["search_id"].astype(str) != str(search_id)]
    rankings = pd.concat([existing, current], ignore_index=True, sort=False)
    rankings["validation_rank"] = pd.NA
    if not rankings.empty:
        for (_stored_search, stored_metric), group in rankings.groupby(
            ["search_id", "optimization_metric"],
            dropna=False,
        ):
            completed = group["status"].astype(str).eq("completed")
            objective = pd.to_numeric(group["objective_value"], errors="coerce")
            finite = completed & np.isfinite(objective)
            if finite.any():
                stored_direction = str(group["direction"].iloc[0])
                rankings.loc[group.index[finite], "validation_rank"] = objective.loc[
                    finite
                ].rank(
                    method="min",
                    ascending=stored_direction == "minimize",
                ).astype("Int64")
    sort_columns = [
        column
        for column in ("search_id", "validation_rank", "trial")
        if column in rankings
    ]
    rankings = rankings.sort_values(sort_columns, na_position="last", kind="stable")
    rankings.to_csv(csv_path, index=False)
    tex_path.write_text(
        _validation_rankings_tex(rankings, hyperparameter_names),
        encoding="utf-8",
    )
    pdf_path = compile_tex(tex_path) if compile_pdf else None
    return csv_path, tex_path, pdf_path


def _validation_rankings_tex(
    rankings: pd.DataFrame,
    hyperparameter_names: Sequence[str],
) -> str:
    """Return a compact, clearly labeled validation-only ranking report."""
    display_columns = [
        "search_id",
        "trial_name",
        *[name for name in hyperparameter_names if name in rankings],
        "optimization_metric",
        "objective_value",
        "validation_rank",
        "status",
    ]
    headers = [
        "Search",
        "Trial",
        *[
            name.replace("_", " ").title()
            for name in hyperparameter_names
            if name in rankings
        ],
        "Metric",
        "Validation value",
        "Validation rank",
        "Status",
    ]
    alignment = "@{}" + "l" * len(display_columns) + "@{}"
    lines = [
        r"\documentclass[9pt]{article}",
        r"\usepackage[margin=1cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{pdflscape}",
        r"\usepackage[table]{xcolor}",
        r"\begin{document}",
        r"\section*{Random Search --- Validation-only Rankings}",
        r"\textbf{Selection boundary:} Test predictions and test metrics are "
        r"not computed during this search.",
        r"\begin{landscape}",
        r"\scriptsize",
        rf"\begin{{longtable}}{{{alignment}}}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in rankings.iterrows():
        cells = []
        rank_value = row.get("validation_rank")
        is_best = not pd.isna(rank_value) and int(rank_value) == 1
        for column in display_columns:
            value = row.get(column)
            if pd.isna(value):
                cells.append("--")
            elif column == "objective_value":
                formatted = f"{float(value):.6g}"
                cells.append(
                    rf"\textcolor{{green!50!black}}{{\textbf{{{formatted}}}}}"
                    if is_best
                    else formatted
                )
            elif column == "validation_rank":
                formatted = str(int(value))
                cells.append(
                    rf"\textcolor{{green!50!black}}{{\textbf{{{formatted}}}}}"
                    if is_best
                    else formatted
                )
            else:
                cells.append(latex_escape(str(value)))
        lines.append(" & ".join(cells) + r" \\")
    if rankings.empty:
        lines.append(
            rf"\multicolumn{{{len(display_columns)}}}{{c}}{{No trials recorded.}} \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            r"\end{landscape}",
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)
