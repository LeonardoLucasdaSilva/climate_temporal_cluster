"""Persistent cross-run rankings for LSTM experiment configurations."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import errno
import os
from pathlib import Path
import re
import time
from typing import Sequence

import numpy as np
import pandas as pd

from config import OUTPUTS_BASE_DIR, PROJECT_ROOT
from methods.lstm_cluster.report import compile_tex, latex_escape


THRESHOLD_GROUPS: tuple[tuple[str, str], ...] = (
    ("gt_0mm", r"Actual precipitation $>0$ mm"),
    ("gt_10mm", r"Actual precipitation $>10$ mm"),
    ("gt_20mm", r"Actual precipitation $>20$ mm"),
    ("gt_30mm", r"Actual precipitation $>30$ mm"),
    ("gt_q95", r"Actual precipitation above the test-set 95th percentile"),
    ("gt_q99", r"Actual precipitation above the test-set 99th percentile"),
)
DEDICATED_RAIN_THRESHOLD_GROUPS = {"gt_0mm", "gt_10mm", "gt_30mm"}
TRIPTYCH_THRESHOLD_GROUPS = (
    ("gt_0mm", r"Actual precipitation $>0$ mm"),
    ("gt_10mm", r"Actual precipitation $>10$ mm"),
    ("gt_30mm", r"Actual precipitation $>30$ mm"),
)

IDENTIFICATION_COLUMNS = [
    "run_id",
    "run_path",
    "date",
    "window_size",
    "n_clusters",
    "clustering_method",
    "loss_function",
    "patience_metric",
    "lstm_units",
    "lstm_units_2",
    "precipitation_standardized",
]
OVERALL_COLUMNS = [
    "n",
    "mae",
    "mae_rank",
    "mse",
    "mse_rank",
    "r2",
    "r2_rank",
]
THRESHOLD_COLUMNS = [
    column
    for group, _label in THRESHOLD_GROUPS
    for column in (
        f"{group}_threshold_mm",
        f"{group}_n",
        f"{group}_mae",
        f"{group}_mae_rank",
        f"{group}_mse",
        f"{group}_mse_rank",
        f"{group}_r2",
        f"{group}_r2_rank",
    )
]
RANKING_COLUMNS = [
    *IDENTIFICATION_COLUMNS,
    *OVERALL_COLUMNS,
    *THRESHOLD_COLUMNS,
    "updated_at",
]
HORIZON_RANKING_COLUMNS = [
    *IDENTIFICATION_COLUMNS,
    "lead_day",
    "forecast_horizon",
    "is_trained_target_day",
    *OVERALL_COLUMNS,
    *THRESHOLD_COLUMNS,
    "updated_at",
]
MATCHED_CONFIGURATION_COLUMNS = [
    "loss_function",
    "patience_metric",
    "lstm_units",
    "lstm_units_2",
    "precipitation_standardized",
    "window_size",
]
RANKING_LOCK_TIMEOUT_SECONDS = 600.0
LOCAL_SYNC_INTERVAL_SECONDS = 300.0
LOCAL_SYNC_STAMP = ".experiment_rankings_local_sync"


def initialize_experiment_rankings(
    backend: str,
    *,
    compile_pdf: bool = True,
    ranking_dir: Path | None = None,
) -> tuple[Path, Path, Path | None]:
    """Create or refresh a backend leaderboard, including an empty one."""
    normalized_backend, resolved_ranking_dir = _ranking_location(
        backend,
        ranking_dir,
    )
    resolved_ranking_dir.mkdir(parents=True, exist_ok=True)
    csv_path = resolved_ranking_dir / "experiment_rankings.csv"
    horizon_csv_path = resolved_ranking_dir / "experiment_horizon_rankings.csv"
    tex_path = resolved_ranking_dir / "experiment_rankings.tex"
    with _ranking_lock(resolved_ranking_dir):
        rankings, horizon_rankings = _merge_local_completed_runs(
            _load_rankings(csv_path),
            _load_horizon_rankings(horizon_csv_path),
            resolved_ranking_dir,
        )
        rankings = _rank_experiments(rankings)
        horizon_rankings = _rank_horizon_experiments(horizon_rankings)
        _write_csv_atomically(rankings, csv_path)
        _write_csv_atomically(horizon_rankings, horizon_csv_path)
        _mark_local_sync(resolved_ranking_dir)
        _write_text_atomically(
            _ranking_tex(rankings, normalized_backend),
            tex_path,
        )
        pdf_path = compile_tex(tex_path) if compile_pdf else None
        _write_horizon_ranking_files(
            horizon_rankings,
            normalized_backend,
            resolved_ranking_dir,
            compile_pdf=compile_pdf,
        )
    return csv_path, tex_path, pdf_path


def synchronize_local_experiment_rankings(
    backend: str,
    *,
    compile_pdf: bool = True,
    ranking_dir: Path | None = None,
) -> tuple[Path, Path, Path, Path | None]:
    """Merge completed local runs and their lead-day metrics into rankings."""
    normalized_backend, resolved_ranking_dir = _ranking_location(
        backend,
        ranking_dir,
    )
    resolved_ranking_dir.mkdir(parents=True, exist_ok=True)
    csv_path = resolved_ranking_dir / "experiment_rankings.csv"
    horizon_csv_path = resolved_ranking_dir / "experiment_horizon_rankings.csv"
    tex_path = resolved_ranking_dir / "experiment_rankings.tex"
    with _ranking_lock(resolved_ranking_dir):
        rankings, horizon_rankings = _merge_local_completed_runs(
            _load_rankings(csv_path),
            _load_horizon_rankings(horizon_csv_path),
            resolved_ranking_dir,
        )
        rankings = _rank_experiments(rankings)
        horizon_rankings = _rank_horizon_experiments(horizon_rankings)
        _write_csv_atomically(rankings, csv_path)
        _write_csv_atomically(horizon_rankings, horizon_csv_path)
        _write_text_atomically(
            _ranking_tex(rankings, normalized_backend),
            tex_path,
        )
        _mark_local_sync(resolved_ranking_dir)
        pdf_path = compile_tex(tex_path) if compile_pdf else None
        _write_horizon_ranking_files(
            horizon_rankings,
            normalized_backend,
            resolved_ranking_dir,
            compile_pdf=compile_pdf,
        )
    return csv_path, horizon_csv_path, tex_path, pdf_path


def render_cluster_comparison_rankings(
    backend: str,
    *,
    compile_pdf: bool = True,
    ranking_dir: Path | None = None,
) -> list[Path]:
    """Render K-column comparison tables from one local ranking registry."""
    normalized_backend, resolved_ranking_dir = _ranking_location(
        backend,
        ranking_dir,
    )
    resolved_ranking_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    with _ranking_lock(resolved_ranking_dir):
        rankings = _rank_experiments(
            _load_rankings(resolved_ranking_dir / "experiment_rankings.csv")
        )
        horizon_rankings = _rank_horizon_experiments(
            _load_horizon_rankings(
                resolved_ranking_dir / "experiment_horizon_rankings.csv"
            )
        )
        main_tex = resolved_ranking_dir / "cluster_comparison_rankings.tex"
        _write_text_atomically(
            _cluster_comparison_tex(rankings, normalized_backend),
            main_tex,
        )
        generated.append(main_tex)
        if compile_pdf:
            main_pdf = compile_tex(main_tex)
            if main_pdf is not None:
                generated.append(main_pdf)

        trained = horizon_rankings.loc[
            horizon_rankings["is_trained_target_day"].map(_as_bool)
        ].copy()
        numeric_lead_days = pd.to_numeric(trained["lead_day"], errors="coerce")
        for lead_day in sorted(
            int(value) for value in numeric_lead_days.dropna().unique()
        ):
            rows = trained.loc[numeric_lead_days == lead_day].copy()
            tex_path = resolved_ranking_dir / (
                f"cluster_comparison_rankings_horizon_d{lead_day:02d}.tex"
            )
            _write_text_atomically(
                _cluster_comparison_tex(
                    rows,
                    normalized_backend,
                    lead_day=lead_day,
                ),
                tex_path,
            )
            generated.append(tex_path)
            if compile_pdf:
                horizon_pdf = compile_tex(tex_path)
                if horizon_pdf is not None:
                    generated.append(horizon_pdf)

            for group, label in THRESHOLD_GROUPS:
                if group not in DEDICATED_RAIN_THRESHOLD_GROUPS:
                    continue
                threshold_tex_path = resolved_ranking_dir / (
                    "cluster_comparison_rankings_"
                    f"horizon_d{lead_day:02d}_{group}.tex"
                )
                _write_text_atomically(
                    _cluster_comparison_tex(
                        rows,
                        normalized_backend,
                        lead_day=lead_day,
                        sections=[(label, f"{group}_")],
                        scope_label=label,
                    ),
                    threshold_tex_path,
                )
                generated.append(threshold_tex_path)
                if compile_pdf:
                    threshold_pdf = compile_tex(threshold_tex_path)
                    if threshold_pdf is not None:
                        generated.append(threshold_pdf)
    return generated


def render_horizon_threshold_triptych_rankings(
    backend: str,
    *,
    window_size: int,
    loss_function: str,
    compile_pdf: bool = True,
    ranking_dir: Path | None = None,
) -> tuple[Path, Path | None]:
    """Render three rain-threshold tables side by side for each lead day."""
    loss_token = _ranking_loss_token(loss_function)
    stem = (
        "cluster_comparison_rankings_"
        f"w{int(window_size)}_{loss_token}_rain_thresholds"
    )
    return _render_filtered_horizon_threshold_triptych(
        backend,
        ranking_dir=ranking_dir,
        stem=stem,
        compile_pdf=compile_pdf,
        loss_function=loss_function,
        window_size=int(window_size),
    )


def render_architecture_threshold_triptych_rankings(
    backend: str,
    *,
    lstm_units: int,
    lstm_units_2: int | None,
    loss_function: str,
    compile_pdf: bool = True,
    ranking_dir: Path | None = None,
) -> tuple[Path, Path | None]:
    """Render threshold triptychs with architecture fixed and window varying."""
    loss_token = _ranking_loss_token(loss_function)
    second_layer_token = "none" if lstm_units_2 is None else str(int(lstm_units_2))
    stem = (
        "cluster_comparison_rankings_"
        f"u1_{int(lstm_units)}_u2_{second_layer_token}_"
        f"{loss_token}_rain_thresholds"
    )
    return _render_filtered_horizon_threshold_triptych(
        backend,
        ranking_dir=ranking_dir,
        stem=stem,
        compile_pdf=compile_pdf,
        loss_function=loss_function,
        lstm_units=int(lstm_units),
        lstm_units_2=lstm_units_2,
        filter_architecture=True,
    )


def _render_filtered_horizon_threshold_triptych(
    backend: str,
    *,
    ranking_dir: Path | None,
    stem: str,
    compile_pdf: bool,
    loss_function: str,
    window_size: int | None = None,
    lstm_units: int | None = None,
    lstm_units_2: int | None = None,
    filter_architecture: bool = False,
) -> tuple[Path, Path | None]:
    """Render one filtered horizon-threshold triptych report."""
    normalized_backend, resolved_ranking_dir = _ranking_location(
        backend,
        ranking_dir,
    )
    resolved_ranking_dir.mkdir(parents=True, exist_ok=True)
    tex_path = resolved_ranking_dir / f"{stem}.tex"
    with _ranking_lock(resolved_ranking_dir):
        horizon_rankings = _rank_horizon_experiments(
            _load_horizon_rankings(
                resolved_ranking_dir / "experiment_horizon_rankings.csv"
            )
        )
        _write_text_atomically(
            _horizon_threshold_triptych_tex(
                horizon_rankings,
                normalized_backend,
                window_size=window_size,
                loss_function=str(loss_function),
                lstm_units=lstm_units,
                lstm_units_2=lstm_units_2,
                filter_architecture=filter_architecture,
            ),
            tex_path,
        )
        pdf_path = compile_tex(tex_path) if compile_pdf else None
    return tex_path, pdf_path


def _ranking_loss_token(loss_function: str) -> str:
    normalized = re.sub(
        r"\(\.(\d)",
        r"(0.\1",
        str(loss_function).lower(),
    ).replace(".", "p")
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def initialize_experiment_progress(
    log_path: Path,
    *,
    run_token: str,
    total: int,
) -> Path:
    """Initialize the concise progress log for a requested experiment batch."""
    return _write_progress_log(
        Path(log_path),
        run_token=run_token,
        completed=0,
        total=total,
        latest="none",
    )


def loss_display_name(
    loss_function: str,
    *,
    loss_alpha: float = 1.0,
    loss_quantiles: Sequence[float] = (),
) -> str:
    """Return an unambiguous display name for a configured loss."""
    normalized = str(loss_function).strip().lower()
    names = {
        "mean_squared_error": "MSE",
        "mse": "MSE",
        "mean_absolute_error": "MAE",
        "mae": "MAE",
        "huber": "HUB",
    }
    if normalized in names:
        return names[normalized]
    if normalized == "weighted_mse_loss":
        return f"WMSE(a={_compact_number(loss_alpha)})"
    if normalized == "quantile_weighted_mse":
        quantiles = "/".join(_compact_number(value) for value in loss_quantiles)
        return f"QWMSE({quantiles})"
    return str(loss_function)


def update_experiment_rankings(
    *,
    backend: str,
    output_dir: Path,
    window_size: int,
    n_clusters: int,
    clustering_method: str,
    loss_function: str,
    mae: float,
    mse: float,
    r2: float,
    actual: Sequence[float] | np.ndarray | None = None,
    predicted: Sequence[float] | np.ndarray | None = None,
    loss_alpha: float = 1.0,
    loss_quantiles: Sequence[float] = (),
    patience_metric: str = "mae",
    lstm_units: int = 1024,
    lstm_units_2: int | None = None,
    precipitation_standardized: bool = False,
    actual_by_lead_day: Sequence[Sequence[float]] | np.ndarray | None = None,
    predicted_by_lead_day: Sequence[Sequence[float]] | np.ndarray | None = None,
    forecast_horizon: int | None = None,
    compile_pdf: bool = True,
    ranking_dir: Path | None = None,
    progress_log: Path | None = None,
    progress_run_token: str | None = None,
    progress_total: int | None = None,
) -> tuple[Path, Path, Path | None]:
    """Upsert one completed configuration and rebuild its backend ranking."""
    normalized_backend, ranking_dir = _ranking_location(backend, ranking_dir)
    ranking_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ranking_dir / "experiment_rankings.csv"
    horizon_csv_path = ranking_dir / "experiment_horizon_rankings.csv"
    tex_path = ranking_dir / "experiment_rankings.tex"
    resolved_output = Path(output_dir).resolve()
    try:
        run_path = resolved_output.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        run_path = resolved_output.as_posix()

    threshold_metrics, sample_count = _threshold_metrics(actual, predicted)
    updated_at = datetime.now().astimezone()
    row = {
        "run_id": f"{resolved_output.parent.name}/{resolved_output.name}",
        "run_path": run_path,
        "date": updated_at.date().isoformat(),
        "window_size": int(window_size),
        "n_clusters": int(n_clusters),
        "clustering_method": str(clustering_method),
        "loss_function": loss_display_name(
            loss_function,
            loss_alpha=loss_alpha,
            loss_quantiles=loss_quantiles,
        ),
        "patience_metric": str(patience_metric).strip().upper(),
        "lstm_units": int(lstm_units),
        "lstm_units_2": pd.NA if lstm_units_2 is None else int(lstm_units_2),
        "precipitation_standardized": bool(precipitation_standardized),
        "n": sample_count,
        "mae": _finite_metric(mae, "mae"),
        "mse": _finite_metric(mse, "mse"),
        "r2": _finite_metric(r2, "r2"),
        **threshold_metrics,
        "updated_at": updated_at.isoformat(timespec="seconds"),
    }
    horizon_rows = _lead_day_metric_rows(
        row,
        actual_by_lead_day,
        predicted_by_lead_day,
        forecast_horizon=forecast_horizon,
    )
    with _ranking_lock(ranking_dir):
        # The complete read-modify-write transaction must be serialized. Sweep
        # configurations may finish in different worker processes, and locking
        # only the final replace would allow a stale frame to discard a run.
        rankings = _load_rankings(csv_path)
        horizon_rankings = _load_horizon_rankings(horizon_csv_path)
        if _local_sync_due(ranking_dir):
            rankings, horizon_rankings = _merge_local_completed_runs(
                rankings,
                horizon_rankings,
                ranking_dir,
            )
            _mark_local_sync(ranking_dir)
        rankings = rankings.loc[rankings["run_path"] != run_path].copy()
        rankings = pd.concat([rankings, pd.DataFrame([row])], ignore_index=True)
        rankings = _rank_experiments(rankings)
        if horizon_rows:
            horizon_rankings = horizon_rankings.loc[
                horizon_rankings["run_path"] != run_path
            ].copy()
            horizon_rankings = pd.concat(
                [horizon_rankings, pd.DataFrame(horizon_rows)],
                ignore_index=True,
            )
        horizon_rankings = _rank_horizon_experiments(horizon_rankings)
        _write_csv_atomically(rankings, csv_path)
        _write_csv_atomically(horizon_rankings, horizon_csv_path)
        if progress_log is not None:
            if not progress_run_token or progress_total is None:
                raise ValueError(
                    "progress_run_token and progress_total are required with "
                    "progress_log."
                )
            completed = int(
                rankings["run_path"]
                .astype(str)
                .str.contains(progress_run_token, regex=False)
                .sum()
            )
            _write_progress_log(
                Path(progress_log),
                run_token=progress_run_token,
                completed=completed,
                total=int(progress_total),
                latest=str(row["run_id"]),
            )
        _write_text_atomically(
            _ranking_tex(rankings, normalized_backend),
            tex_path,
        )
        # LaTeX uses shared auxiliary and output paths, so compilation belongs
        # to the same transaction as the CSV and TeX refresh.
        pdf_path = compile_tex(tex_path) if compile_pdf else None
        _write_horizon_ranking_files(
            horizon_rankings,
            normalized_backend,
            ranking_dir,
            compile_pdf=compile_pdf,
        )
    return csv_path, tex_path, pdf_path


def _write_progress_log(
    log_path: Path,
    *,
    run_token: str,
    completed: int,
    total: int,
    latest: str,
) -> Path:
    if total <= 0:
        raise ValueError("progress total must be positive.")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    percentage = min(completed / total, 1.0) * 100.0
    status = "complete" if completed >= total else "running"
    contents = "\n".join(
        [
            f"status: {status}",
            f"batch: {run_token}",
            f"processed: {completed}/{total}",
            f"percentage: {percentage:.2f}%",
            f"latest_completed: {latest}",
            "updated_at: "
            + datetime.now().astimezone().isoformat(timespec="seconds"),
            "",
        ]
    )
    temporary_path = log_path.with_suffix(log_path.suffix + ".tmp")
    temporary_path.write_text(contents, encoding="utf-8")
    temporary_path.replace(log_path)
    return log_path


def _ranking_location(
    backend: str,
    ranking_dir: Path | None,
) -> tuple[str, Path]:
    normalized_backend = str(backend).strip().lower()
    if normalized_backend not in {"tensorflow", "pytorch"}:
        raise ValueError("backend must be either 'tensorflow' or 'pytorch'.")
    return normalized_backend, Path(
        ranking_dir
        if ranking_dir is not None
        else OUTPUTS_BASE_DIR / normalized_backend
    )


def _local_sync_due(ranking_dir: Path) -> bool:
    stamp_path = Path(ranking_dir) / LOCAL_SYNC_STAMP
    if not stamp_path.exists():
        return True
    return time.time() - stamp_path.stat().st_mtime >= LOCAL_SYNC_INTERVAL_SECONDS


def _mark_local_sync(ranking_dir: Path) -> None:
    stamp_path = Path(ranking_dir) / LOCAL_SYNC_STAMP
    _write_text_atomically(
        datetime.now().astimezone().isoformat(timespec="seconds") + "\n",
        stamp_path,
    )


def _merge_local_completed_runs(
    rankings: pd.DataFrame,
    horizon_rankings: pd.DataFrame,
    ranking_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add completed runs found below this backend's local output root."""
    rankings = rankings.copy()
    horizon_rankings = horizon_rankings.copy()
    main_by_path = {
        _normalized_stored_path(row.run_path): row._asdict()
        for row in rankings.itertuples(index=False)
        if not pd.isna(row.run_path)
    }
    threshold_complete_paths: set[str] = set()
    if not horizon_rankings.empty:
        for stored_path, rows in horizon_rankings.groupby("run_path", dropna=True):
            if (
                rows["gt_0mm_threshold_mm"].notna().all()
                and rows["gt_q99_threshold_mm"].notna().all()
            ):
                threshold_complete_paths.add(_normalized_stored_path(stored_path))
    discovered_main: list[dict[str, object]] = []
    discovered_horizons: list[dict[str, object]] = []
    refreshed_horizon_paths: set[str] = set()

    run_dirs = {
        path.parent.resolve()
        for path in Path(ranking_dir).rglob("metrics_summary.csv")
    }
    run_dirs.update(
        path.parents[1].resolve()
        for path in Path(ranking_dir).rglob(
            "test_prediction_metrics_by_lead_day.csv"
        )
    )
    for run_dir in sorted(run_dirs):
        metrics_path = run_dir / "metrics_summary.csv"
        normalized_path = os.path.normcase(str(run_dir))
        identification = main_by_path.get(normalized_path)
        if identification is None:
            if not metrics_path.exists():
                continue
            identification = _saved_ranking_row(run_dir, metrics_path)
            if identification is None:
                continue
            discovered_main.append(identification)
            main_by_path[normalized_path] = identification
        if normalized_path in threshold_complete_paths:
            continue
        lead_rows = _saved_horizon_rows(run_dir, identification)
        if lead_rows:
            discovered_horizons.extend(lead_rows)
            refreshed_horizon_paths.add(normalized_path)

    if discovered_main:
        rankings = pd.concat(
            [rankings, pd.DataFrame(discovered_main)],
            ignore_index=True,
        )
    if discovered_horizons:
        normalized_existing_paths = horizon_rankings["run_path"].map(
            lambda value: (
                "" if pd.isna(value) else _normalized_stored_path(value)
            )
        )
        horizon_rankings = horizon_rankings.loc[
            ~normalized_existing_paths.isin(refreshed_horizon_paths)
        ].copy()
        horizon_rankings = pd.concat(
            [horizon_rankings, pd.DataFrame(discovered_horizons)],
            ignore_index=True,
        )
    return rankings, horizon_rankings


def _normalized_stored_path(value: object) -> str:
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return os.path.normcase(str(path.resolve()))


def _stored_run_path(run_dir: Path) -> str:
    try:
        return run_dir.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return run_dir.as_posix()


def _saved_ranking_row(
    run_dir: Path,
    metrics_path: Path,
) -> dict[str, object] | None:
    report_path = run_dir / "experiment_report.tex"
    if not report_path.exists():
        return None
    try:
        fields = _parse_report_fields(report_path)
        window_size = int(fields["Window Size"])
        n_clusters = int(fields["Number of Clusters"])
        algorithm = _latex_unescape(fields["Algorithm"]).strip().lower()
        metrics = pd.read_csv(metrics_path)
        test_rows = metrics.loc[
            metrics["split"].astype(str).str.strip().str.lower() == "test"
        ]
        if len(test_rows) != 1:
            return None
        test = test_rows.iloc[0]
        predictions_path = run_dir / "test_predictions.csv"
        if predictions_path.exists():
            predictions = pd.read_csv(
                predictions_path,
                usecols=lambda column: column in {"actual", "predicted"},
            )
            actual = predictions.get("actual")
            predicted = predictions.get("predicted")
        else:
            actual = predicted = None
        threshold_metrics, sample_count = _threshold_metrics(actual, predicted)
        if pd.isna(sample_count):
            lead_path = (
                run_dir
                / "forecast_horizon_diagnostics"
                / "test_prediction_metrics_by_lead_day.csv"
            )
            if lead_path.exists():
                lead_metrics = pd.read_csv(lead_path)
                sample_count = pd.to_numeric(
                    lead_metrics.get("n_test"), errors="coerce"
                ).max()

        loss = _latex_unescape(fields.get("Loss", "MSE"))
        quantiles = [
            float(value)
            for value in re.findall(
                r"[-+]?\d*\.?\d+",
                _latex_unescape(fields.get("Loss quantiles", "")),
            )
        ]
        alpha_values = re.findall(
            r"[-+]?\d*\.?\d+",
            _latex_unescape(fields.get("Loss alpha", "1")),
        )
        loss_alpha = float(alpha_values[0]) if alpha_values else 1.0
        updated_at = datetime.fromtimestamp(
            metrics_path.stat().st_mtime
        ).astimezone()
        run_path = _stored_run_path(run_dir)
        return {
            "run_id": f"{run_dir.parent.name}/{run_dir.name}",
            "run_path": run_path,
            "date": updated_at.date().isoformat(),
            "window_size": window_size,
            "n_clusters": n_clusters,
            "clustering_method": _saved_clustering_method(
                run_dir.name,
                n_clusters,
                algorithm,
                fields.get("Sigma"),
            ),
            "loss_function": loss_display_name(
                loss,
                loss_alpha=loss_alpha,
                loss_quantiles=quantiles,
            ),
            "patience_metric": _latex_unescape(
                fields.get("Early stopping metric", "MAE")
            ).strip().upper(),
            "lstm_units": int(fields.get("LSTM layer 1 units", 1024)),
            "lstm_units_2": _optional_layer_units(
                fields.get("LSTM layer 2 units")
            ),
            "precipitation_standardized": (
                _latex_unescape(
                    fields.get("Clustering Precipitation Scaler", "none")
                ).strip().lower()
                == "standard"
                and _latex_unescape(
                    fields.get("LSTM Precipitation Scaler", "none")
                ).strip().lower()
                == "standard"
            ),
            "n": sample_count,
            "mae": float(test["MAE"]),
            "mae_rank": pd.NA,
            "mse": float(test["MSE"]),
            "mse_rank": pd.NA,
            "r2": float(test["R2"]),
            "r2_rank": pd.NA,
            **threshold_metrics,
            "updated_at": updated_at.isoformat(timespec="seconds"),
        }
    except (KeyError, OSError, TypeError, ValueError, pd.errors.ParserError):
        return None


def _parse_report_fields(report_path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    pattern = re.compile(r"\\item\s+\\textbf\{([^{}]+):\}\s*(.*?)\s*$")
    for line in report_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            fields[match.group(1)] = match.group(2)
    return fields


def _latex_unescape(value: object) -> str:
    return str(value).replace(r"\_", "_").replace(r"\%", "%")


def _optional_layer_units(value: object | None) -> object:
    if value is None:
        return pd.NA
    text = _latex_unescape(value).strip().lower()
    if text in {"", "-", "--", "none", "null", "n/a"}:
        return pd.NA
    match = re.search(r"\d+", text)
    return int(match.group()) if match else pd.NA


def _saved_clustering_method(
    run_name: str,
    n_clusters: int,
    algorithm: str,
    report_sigma: object,
) -> str:
    if n_clusters == 1:
        return "-"
    if algorithm != "spectral":
        return algorithm
    sigma_match = re.search(r"_sigma_([^_]+)$", run_name)
    if sigma_match:
        try:
            sigma = float(sigma_match.group(1).replace("p", "."))
            return f"spectral({sigma:g})"
        except ValueError:
            pass
    sigma_values = re.findall(r"[-+]?\d*\.?\d+", str(report_sigma))
    return (
        f"spectral({float(sigma_values[0]):g})"
        if sigma_values
        else "spectral"
    )


def _saved_horizon_rows(
    run_dir: Path,
    identification: dict[str, object],
) -> list[dict[str, object]]:
    metrics_path = (
        run_dir
        / "forecast_horizon_diagnostics"
        / "test_prediction_metrics_by_lead_day.csv"
    )
    if not metrics_path.exists():
        return []
    try:
        metrics = pd.read_csv(metrics_path)
        required = {"lead_day", "n_test", "MAE", "MSE", "R2"}
        if not required.issubset(metrics.columns):
            return []
        prediction_path = (
            run_dir
            / "forecast_horizon_diagnostics"
            / "test_prediction_by_lead_day.csv"
        )
        thresholds_by_day: dict[int, dict[str, object]] = {}
        if prediction_path.exists():
            predictions = pd.read_csv(prediction_path)
            prediction_columns = {"lead_day", "actual_mm", "predicted_mm"}
            if prediction_columns.issubset(predictions.columns):
                for lead_value, lead_predictions in predictions.groupby(
                    "lead_day",
                    sort=True,
                ):
                    threshold_metrics, _sample_count = _threshold_metrics(
                        lead_predictions["actual_mm"],
                        lead_predictions["predicted_mm"],
                    )
                    thresholds_by_day[int(lead_value)] = threshold_metrics
        shared = {
            column: identification[column]
            for column in IDENTIFICATION_COLUMNS
        }
        rows: list[dict[str, object]] = []
        for metric in metrics.itertuples(index=False):
            values = metric._asdict()
            lead_day = int(values["lead_day"])
            forecast_horizon = int(values.get("forecast_horizon", lead_day))
            trained = values.get(
                "is_trained_target_day",
                lead_day == forecast_horizon,
            )
            threshold_metrics = thresholds_by_day.get(
                lead_day,
                {column: pd.NA for column in THRESHOLD_COLUMNS},
            )
            rows.append(
                {
                    **shared,
                    "lead_day": lead_day,
                    "forecast_horizon": forecast_horizon,
                    "is_trained_target_day": _as_bool(trained),
                    "n": int(values["n_test"]),
                    "mae": float(values["MAE"]),
                    "mae_rank": pd.NA,
                    "mse": float(values["MSE"]),
                    "mse_rank": pd.NA,
                    "r2": float(values["R2"]),
                    "r2_rank": pd.NA,
                    **threshold_metrics,
                    "updated_at": identification["updated_at"],
                }
            )
        return rows
    except (OSError, TypeError, ValueError, pd.errors.ParserError):
        return []


def _finite_metric(value: float, name: str) -> float:
    metric = float(value)
    if not np.isfinite(metric):
        raise ValueError(f"{name} must be finite, received {value!r}.")
    return metric


def _threshold_metrics(
    actual: Sequence[float] | np.ndarray | None,
    predicted: Sequence[float] | np.ndarray | None,
) -> tuple[dict[str, object], object]:
    if actual is None or predicted is None:
        return {
            column: pd.NA
            for column in THRESHOLD_COLUMNS
        }, pd.NA

    y_true = np.asarray(actual, dtype=float).reshape(-1)
    y_pred = np.asarray(predicted, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("actual and predicted must have the same shape.")
    if y_true.size == 0:
        raise ValueError("actual and predicted cannot be empty.")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("actual and predicted must contain only finite values.")

    cutoffs = {
        "gt_0mm": 0.0,
        "gt_10mm": 10.0,
        "gt_20mm": 20.0,
        "gt_30mm": 30.0,
        "gt_q95": float(np.quantile(y_true, 0.95)),
        "gt_q99": float(np.quantile(y_true, 0.99)),
    }
    result: dict[str, float | int] = {}
    for group, cutoff in cutoffs.items():
        mask = y_true > cutoff
        selected_true = y_true[mask]
        selected_pred = y_pred[mask]
        result[f"{group}_threshold_mm"] = cutoff
        result[f"{group}_n"] = int(mask.sum())
        if selected_true.size == 0:
            mae = mse = r2 = np.nan
        else:
            residual = selected_true - selected_pred
            mae = float(np.mean(np.abs(residual)))
            mse = float(np.mean(np.square(residual)))
            denominator = float(
                np.sum(np.square(selected_true - np.mean(selected_true)))
            )
            r2 = (
                float(1.0 - np.sum(np.square(residual)) / denominator)
                if selected_true.size >= 2 and denominator > 0.0
                else np.nan
            )
        result[f"{group}_mae"] = mae
        result[f"{group}_mse"] = mse
        result[f"{group}_r2"] = r2
    return result, int(y_true.size)


def _lead_day_metric_rows(
    identification: dict[str, object],
    actual_by_lead_day: Sequence[Sequence[float]] | np.ndarray | None,
    predicted_by_lead_day: Sequence[Sequence[float]] | np.ndarray | None,
    *,
    forecast_horizon: int | None,
) -> list[dict[str, object]]:
    if actual_by_lead_day is None or predicted_by_lead_day is None:
        return []
    actual = np.asarray(actual_by_lead_day, dtype=float)
    predicted = np.asarray(predicted_by_lead_day, dtype=float)
    if actual.ndim == 1:
        actual = actual.reshape(-1, 1)
    if predicted.ndim == 1:
        predicted = predicted.reshape(-1, 1)
    if actual.ndim != 2 or predicted.ndim != 2 or actual.shape != predicted.shape:
        raise ValueError(
            "actual_by_lead_day and predicted_by_lead_day must be matching "
            "one- or two-dimensional arrays."
        )
    if actual.size == 0:
        raise ValueError("lead-day actual and predicted arrays cannot be empty.")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted)):
        raise ValueError("lead-day actual and predicted arrays must be finite.")

    configured_horizon = int(forecast_horizon or actual.shape[1])
    if configured_horizon <= 0:
        raise ValueError("forecast_horizon must be positive.")
    shared = {
        column: identification[column]
        for column in IDENTIFICATION_COLUMNS
    }
    rows: list[dict[str, object]] = []
    for offset in range(actual.shape[1]):
        metrics = _regression_metrics(actual[:, offset], predicted[:, offset])
        threshold_metrics, _sample_count = _threshold_metrics(
            actual[:, offset],
            predicted[:, offset],
        )
        rows.append(
            {
                **shared,
                "lead_day": offset + 1,
                "forecast_horizon": configured_horizon,
                "is_trained_target_day": True,
                "n": int(actual.shape[0]),
                "mae": metrics["mae"],
                "mae_rank": pd.NA,
                "mse": metrics["mse"],
                "mse_rank": pd.NA,
                "r2": metrics["r2"],
                "r2_rank": pd.NA,
                **threshold_metrics,
                "updated_at": identification["updated_at"],
            }
        )
    return rows


def _regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    residual = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    mse = float(np.mean(np.square(residual)))
    denominator = float(np.sum(np.square(actual - np.mean(actual))))
    r2 = (
        float(1.0 - np.sum(np.square(residual)) / denominator)
        if len(actual) >= 2 and denominator > 0.0
        else np.nan
    )
    return {
        "mae": float(np.mean(np.abs(residual))),
        "mse": mse,
        "r2": r2,
    }


def _load_rankings(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=RANKING_COLUMNS)
    rankings = pd.read_csv(csv_path)
    run_ids = rankings.get("run_id", pd.Series("", index=rankings.index)).astype(str)
    if "patience_metric" not in rankings:
        is_new_r2_sweep = run_ids.str.startswith(
            "requested_multiquantile_q0p75_q0p9_q0p95_q0p99_"
        )
        rankings["patience_metric"] = np.where(is_new_r2_sweep, "R2", "MAE")
    if "lstm_units" not in rankings:
        parsed_units = pd.to_numeric(
            run_ids.str.extract(r"_u1_(\d+)(?:_|$)", expand=False),
            errors="coerce",
        )
        rankings["lstm_units"] = parsed_units.fillna(1024).astype("Int64")
    if "lstm_units_2" not in rankings:
        rankings["lstm_units_2"] = pd.to_numeric(
            run_ids.str.extract(r"_u2_(\d+)(?:_|$)", expand=False),
            errors="coerce",
        ).astype("Int64")
    if "date" not in rankings:
        rankings["date"] = rankings.get(
            "updated_at",
            pd.Series(pd.NA, index=rankings.index),
        ).astype("string").str.slice(0, 10)
    if "precipitation_standardized" not in rankings:
        rankings["precipitation_standardized"] = False
    if "loss_function" in rankings:
        rankings["loss_function"] = rankings["loss_function"].map(
            _abbreviate_saved_loss
        )
    for column in RANKING_COLUMNS:
        if column not in rankings:
            rankings[column] = pd.NA
    return rankings[RANKING_COLUMNS]


def _load_horizon_rankings(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=HORIZON_RANKING_COLUMNS)
    rankings = pd.read_csv(csv_path)
    if "lstm_units_2" not in rankings:
        run_ids = rankings.get(
            "run_id", pd.Series("", index=rankings.index)
        ).astype(str)
        rankings["lstm_units_2"] = pd.to_numeric(
            run_ids.str.extract(r"_u2_(\d+)(?:_|$)", expand=False),
            errors="coerce",
        ).astype("Int64")
    for column in HORIZON_RANKING_COLUMNS:
        if column not in rankings:
            rankings[column] = pd.NA
    return rankings[HORIZON_RANKING_COLUMNS]


def _rank_experiments(rankings: pd.DataFrame) -> pd.DataFrame:
    ranked = rankings.copy()
    for prefix in ("", *(f"{group}_" for group, _label in THRESHOLD_GROUPS)):
        for metric, ascending in (("mae", True), ("mse", True), ("r2", False)):
            metric_column = f"{prefix}{metric}"
            ranked[f"{metric_column}_rank"] = (
                pd.to_numeric(ranked[metric_column], errors="coerce")
                .rank(method="min", ascending=ascending)
                .astype("Int64")
            )
    ranked = ranked.sort_values(
        ["mae_rank", "mse_rank", "r2_rank", "updated_at"],
        kind="stable",
    ).reset_index(drop=True)
    return ranked[RANKING_COLUMNS]


def _rank_horizon_experiments(rankings: pd.DataFrame) -> pd.DataFrame:
    ranked = rankings.copy()
    if ranked.empty:
        return ranked.reindex(columns=HORIZON_RANKING_COLUMNS)
    lead_days = pd.to_numeric(ranked["lead_day"], errors="coerce")
    ranked["lead_day"] = lead_days.astype("Int64")
    for prefix in ("", *(f"{group}_" for group, _label in THRESHOLD_GROUPS)):
        for metric, ascending in (("mae", True), ("mse", True), ("r2", False)):
            metric_column = f"{prefix}{metric}"
            ranked[f"{metric_column}_rank"] = (
                pd.to_numeric(ranked[metric_column], errors="coerce")
                .groupby(ranked["lead_day"])
                .rank(method="min", ascending=ascending)
                .astype("Int64")
            )
    ranked = ranked.sort_values(
        ["lead_day", "mae_rank", "mse_rank", "r2_rank", "updated_at"],
        kind="stable",
    ).reset_index(drop=True)
    return ranked[HORIZON_RANKING_COLUMNS]


def _write_csv_atomically(rankings: pd.DataFrame, csv_path: Path) -> None:
    temporary_path = csv_path.with_suffix(".csv.tmp")
    try:
        rankings.to_csv(temporary_path, index=False)
        temporary_path.replace(csv_path)
    except PermissionError as exc:
        # Spreadsheet applications can lock an open CSV on Windows. Compact
        # experiment results remain authoritative, so retain the fresh
        # temporary table and let the next ranking audit retry the update.
        print(
            f"Warning: could not replace locked ranking file {csv_path}: {exc}",
            flush=True,
        )


def _write_text_atomically(contents: str, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary_path.write_text(contents, encoding="utf-8")
        temporary_path.replace(path)
    except PermissionError as exc:
        # Windows may lock a TeX file while it is open in an editor/viewer.
        # Rankings are derived artifacts, so retain the fresh temporary file
        # and allow the numerical experiment to continue. A later refresh
        # retries the same atomic replacement.
        print(
            f"Warning: could not replace locked ranking file {path}: {exc}",
            flush=True,
        )


def _compact_number(value: float) -> str:
    text = f"{float(value):g}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def _abbreviate_saved_loss(value: object) -> object:
    if pd.isna(value):
        return value
    label = str(value)
    if label == "HUBER":
        return "HUB"
    if label.startswith("WEIGHTED_MSE(alpha=") and label.endswith(")"):
        alpha = label.removeprefix("WEIGHTED_MSE(alpha=")[:-1]
        return f"WMSE(a={_compact_number(float(alpha))})"
    if label.startswith("QUANTILE(q=") and label.endswith(")"):
        quantiles = label.removeprefix("QUANTILE(q=")[:-1]
        compact = "/".join(
            _compact_number(float(value)) for value in quantiles.split(",")
        )
        return f"QWMSE({compact})"
    return label


@contextmanager
def _ranking_lock(ranking_dir: Path):
    """Serialize leaderboard transactions across processes."""
    lock_path = Path(ranking_dir) / ".experiment_rankings.lock"
    with lock_path.open("a+b") as lock_file:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        if os.name == "nt":
            import msvcrt

            deadline = time.monotonic() + RANKING_LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno not in {errno.EACCES, errno.EDEADLK}:
                        raise
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out waiting for ranking lock: {lock_path}"
                        ) from None
                    time.sleep(0.1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _ranking_tex(rankings: pd.DataFrame, backend: str) -> str:
    backend_label = "PyTorch" if backend == "pytorch" else "TensorFlow"
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=1.2cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{pdflscape}",
        r"\usepackage[table]{xcolor}",
        r"\begin{document}",
        r"\begin{landscape}",
        rf"\section*{{{backend_label} Experiment Rankings}}",
        rf"Generated: {latex_escape(generated_at)}. Lower MAE and MSE are better. Ties share the same rank. The configured-target table uses each run's final forecast day (D+5 for five-day runs).",
        *_ranking_table(rankings, "Configured target day (D+H)", ""),
    ]
    for group, label in THRESHOLD_GROUPS:
        lines.extend(_ranking_table(rankings, label, f"{group}_"))
    lines.extend(
        [
            r"\end{landscape}",
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)


def _cluster_comparison_tex(
    rankings: pd.DataFrame,
    backend: str,
    *,
    lead_day: int | None = None,
    sections: Sequence[tuple[str, str]] | None = None,
    scope_label: str | None = None,
) -> str:
    """Return a report whose tables compare only the cluster-count axis."""
    backend_label = "PyTorch" if backend == "pytorch" else "TensorFlow"
    suffix = "" if lead_day is None else f" --- D+{lead_day}"
    if scope_label is not None:
        suffix += f" --- {scope_label}"
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        r"\documentclass[9pt]{article}",
        r"\usepackage[margin=1.1cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{pdflscape}",
        r"\usepackage[table]{xcolor}",
        r"\definecolor{rankingbestgreen}{HTML}{228B22}",
        r"\begin{document}",
        r"\begin{landscape}",
        rf"\section*{{{backend_label} Cluster-count Comparisons{suffix}}}",
        rf"Generated: {latex_escape(generated_at)}. Each table fixes loss, patience, $U_1$, $U_2$, precipitation standardization, and window size. Clustering method and $K$ vary by row. For each $K$, only the spectral sigma with the lowest MAE for the current table is shown; non-spectral rows are retained. The best MAE and MSE within each matched table are bold green (lower is better).",
    ]
    selected_sections = sections
    if selected_sections is None:
        selected_sections = [("Configured target day (D+H)", "")]
        selected_sections.extend(
            (label, f"{group}_") for group, label in THRESHOLD_GROUPS
        )
    for title, prefix in selected_sections:
        lines.append(rf"\subsection*{{{title}}}")
        lines.extend(_cluster_comparison_tables(rankings, prefix))
    lines.extend([r"\end{landscape}", r"\end{document}", ""])
    return "\n".join(lines)


def _cluster_comparison_tables(
    rankings: pd.DataFrame,
    prefix: str,
) -> list[str]:
    """Return matched-configuration tables with clustering and K as row axes."""
    rankings = _select_best_spectral_sigma_rows(rankings, prefix)
    clustered = rankings.loc[
        pd.to_numeric(rankings["n_clusters"], errors="coerce") > 1
    ].copy()
    if clustered.empty:
        return [r"No completed clustered configurations yet.", r"\clearpage"]

    metric_columns = [
        f"{prefix}mae",
        f"{prefix}mse",
    ]
    lines: list[str] = []
    grouped = rankings.groupby(
        MATCHED_CONFIGURATION_COLUMNS,
        sort=True,
        dropna=False,
    )
    for matched_configuration, matched_rows in grouped:
        loss, patience, units_1, units_2, precipitation_std, window_size = (
            matched_configuration
        )
        matched_rows = matched_rows.sort_values(
            ["n_clusters", "clustering_method", metric_columns[0], "updated_at"],
            kind="stable",
            na_position="last",
        )
        best_values: list[float] = []
        for metric_column in metric_columns:
            values = pd.to_numeric(matched_rows[metric_column], errors="coerce")
            finite = [float(value) for value in values if pd.notna(value)]
            if not finite:
                best_values.append(np.nan)
            else:
                best_values.append(min(finite))

        lines.extend(
            [
                rf"\subsubsection*{{Window {int(window_size)}; {latex_escape(str(loss))}; patience {_latex_text(patience)}; $U_1={_latex_integer(units_1)}$; $U_2={_latex_optional_layer(units_2)}$; precip. std.={_latex_bool(precipitation_std)}}}",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{2.5pt}",
                r"\rowcolors{2}{gray!10}{white}",
                r"\begin{longtable}{@{}lrrcrlrrr@{}}",
                r"\toprule",
                r"Loss & $U_1$ & $U_2$ & Precip. std. & Window & Clustering & $K$ & MAE & MSE \\",
                r"\midrule",
                r"\endfirsthead",
                r"\toprule",
                r"Loss & $U_1$ & $U_2$ & Precip. std. & Window & Clustering & $K$ & MAE & MSE \\",
                r"\midrule",
                r"\endhead",
            ]
        )
        for row in matched_rows.itertuples(index=False):
            cells = [
                latex_escape(str(row.loss_function)),
                _latex_integer(row.lstm_units),
                _latex_optional_layer(row.lstm_units_2),
                _latex_bool(row.precipitation_standardized),
                _latex_integer(row.window_size),
                (
                    "-"
                    if int(row.n_clusters) == 1
                    else _latex_text(row.clustering_method)
                ),
                _latex_integer(row.n_clusters),
            ]
            for metric_column, best_value in zip(metric_columns, best_values):
                value = pd.to_numeric(getattr(row, metric_column), errors="coerce")
                text = _latex_metric(value)
                if pd.notna(value) and np.isclose(float(value), best_value):
                    text = _latex_bold_green(text)
                cells.append(text)
            lines.append(" & ".join(cells) + r" \\")
        lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    lines.append(r"\clearpage")
    return lines


def _horizon_threshold_triptych_tex(
    horizon_rankings: pd.DataFrame,
    backend: str,
    *,
    window_size: int | None,
    loss_function: str,
    lstm_units: int | None = None,
    lstm_units_2: int | None = None,
    filter_architecture: bool = False,
) -> str:
    """Return five landscape pages of side-by-side rain-threshold tables."""
    backend_label = "PyTorch" if backend == "pytorch" else "TensorFlow"
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    trained = horizon_rankings.loc[
        horizon_rankings["is_trained_target_day"].map(_as_bool)
    ].copy()
    lead_days = pd.to_numeric(trained["lead_day"], errors="coerce")
    windows = pd.to_numeric(trained["window_size"], errors="coerce")
    selection = lead_days.between(1, 5) & trained["loss_function"].astype(
        str
    ).eq(str(loss_function))
    if window_size is not None:
        selection &= windows.eq(int(window_size))
    if filter_architecture:
        selection &= pd.to_numeric(
            trained["lstm_units"], errors="coerce"
        ).eq(int(lstm_units))
        second_layer = pd.to_numeric(trained["lstm_units_2"], errors="coerce")
        if lstm_units_2 is None:
            selection &= second_layer.isna()
        else:
            selection &= second_layer.eq(int(lstm_units_2))
    selected = trained.loc[selection].copy()
    common_configuration, common_columns = _triptych_common_configuration(
        selected,
        loss_function=loss_function,
    )

    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.7cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{pdflscape}",
        r"\usepackage[table]{xcolor}",
        r"\definecolor{rankingbestgreen}{HTML}{228B22}",
        r"\begin{document}",
        r"\begin{landscape}",
    ]
    for lead_day in range(1, 6):
        rows = selected.loc[lead_days.loc[selected.index].eq(lead_day)].copy()
        lines.extend(
            [
                rf"\section*{{{backend_label} Rain-threshold Comparisons --- D+{lead_day}}}",
                rf"\textbf{{Common configuration:}} {common_configuration}.\\",
                rf"Generated: {latex_escape(generated_at)}. Bold green values are the best MAE and MSE within each matched architecture.\par",
                r"\vspace{0.4em}",
                r"\noindent",
            ]
        )
        for panel_index, (group, label) in enumerate(TRIPTYCH_THRESHOLD_GROUPS):
            lines.extend(
                [
                    r"\begin{minipage}[t]{0.31\linewidth}",
                    r"\centering",
                    rf"\textbf{{{label}}}\par\vspace{{0.25em}}",
                    *_horizon_threshold_panel(
                        rows,
                        f"{group}_",
                        common_columns=common_columns,
                    ),
                    r"\end{minipage}",
                ]
            )
            if panel_index < len(TRIPTYCH_THRESHOLD_GROUPS) - 1:
                lines.append(r"\hfill")
        if lead_day < 5:
            lines.append(r"\clearpage")
    lines.extend([r"\end{landscape}", r"\end{document}", ""])
    return "\n".join(lines)


def _triptych_common_configuration(
    rankings: pd.DataFrame,
    *,
    loss_function: str,
) -> tuple[str, set[str]]:
    """Return settings shared by all rows in a filtered triptych report."""
    settings: list[str] = []
    candidates = (
        ("window", "window_size", _latex_integer),
        ("patience", "patience_metric", _latex_text),
        (r"$U_1$", "lstm_units", _latex_integer),
        (r"$U_2$", "lstm_units_2", _latex_optional_layer),
        (
            "precipitation standardized",
            "precipitation_standardized",
            _latex_bool,
        ),
    )
    common_columns: set[str] = set()
    for label, column, formatter in candidates:
        if rankings.empty:
            continue
        normalized = rankings[column].map(
            lambda value: None if pd.isna(value) else value
        )
        if normalized.astype(str).nunique(dropna=False) == 1:
            common_columns.add(column)
            settings.append(rf"{label}={formatter(rankings.iloc[0][column])}")
    loss_setting = rf"loss={latex_escape(str(loss_function))}"
    loss_position = 1 if settings and settings[0].startswith("window=") else 0
    settings.insert(loss_position, loss_setting)
    return "; ".join(settings), common_columns


def _horizon_threshold_panel(
    rankings: pd.DataFrame,
    prefix: str,
    *,
    common_columns: set[str],
) -> list[str]:
    """Return one compact threshold table for the triptych report."""
    rows = _select_best_spectral_sigma_rows(rankings, prefix).sort_values(
        [
            *MATCHED_CONFIGURATION_COLUMNS,
            "n_clusters",
            "clustering_method",
            f"{prefix}mae",
            "updated_at",
        ],
        kind="stable",
        na_position="last",
    )
    show_window = "window_size" not in common_columns
    show_network = not {"lstm_units", "lstm_units_2"}.issubset(common_columns)
    show_standardization = "precipitation_standardized" not in common_columns
    headers: list[str] = []
    alignments: list[str] = []
    if show_window:
        headers.append("Window")
        alignments.append("r")
    if show_network:
        headers.append(r"Network ($U_1/U_2$)")
        alignments.append("l")
    if show_standardization:
        headers.append("Std.")
        alignments.append("c")
    headers.extend(["Clustering", "$K$", "MAE", "MSE"])
    alignments.extend(["l", "r", "r", "r"])
    column_count = len(headers)
    lines = [
        r"\tiny",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\rowcolors{2}{gray!10}{white}",
        rf"\begin{{tabular}}{{@{{}}{''.join(alignments)}@{{}}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    previous_group: tuple[object, ...] | None = None
    best_by_group: dict[tuple[object, ...], tuple[float, float]] = {}
    for matched_configuration, matched_rows in rows.groupby(
        MATCHED_CONFIGURATION_COLUMNS,
        sort=False,
        dropna=False,
    ):
        group_key = tuple(
            None if pd.isna(value) else value for value in matched_configuration
        )
        best_by_group[group_key] = tuple(
            float(pd.to_numeric(matched_rows[f"{prefix}{metric}"], errors="coerce").min())
            for metric in ("mae", "mse")
        )

    for row in rows.itertuples(index=False):
        current_group = tuple(
            None if pd.isna(getattr(row, column)) else getattr(row, column)
            for column in MATCHED_CONFIGURATION_COLUMNS
        )
        if previous_group is not None and current_group != previous_group:
            lines.append(r"\midrule")
        previous_group = current_group
        best_values = best_by_group[current_group]
        metrics: list[str] = []
        for metric, best_value in zip(("mae", "mse"), best_values):
            value = pd.to_numeric(getattr(row, f"{prefix}{metric}"), errors="coerce")
            text = _latex_metric(value)
            if pd.notna(value) and np.isclose(float(value), best_value):
                text = _latex_bold_green(text)
            metrics.append(text)
        network = (
            f"{_latex_integer(row.lstm_units)}/"
            f"{_latex_optional_layer(row.lstm_units_2)}"
        )
        clustering = (
            "-"
            if int(row.n_clusters) == 1
            else _latex_text(row.clustering_method)
        )
        cells: list[str] = []
        if show_window:
            cells.append(_latex_integer(row.window_size))
        if show_network:
            cells.append(network)
        if show_standardization:
            cells.append(_latex_bool(row.precipitation_standardized))
        cells.extend(
            [
                clustering,
                _latex_integer(row.n_clusters),
                *metrics,
            ]
        )
        lines.append(" & ".join(cells) + r" \\")
    if rows.empty:
        lines.append(
            rf"\multicolumn{{{column_count}}}{{c}}{{No completed experiments.}} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return lines


def _write_horizon_ranking_files(
    horizon_rankings: pd.DataFrame,
    backend: str,
    ranking_dir: Path,
    *,
    compile_pdf: bool,
) -> None:
    trained = horizon_rankings.loc[
        horizon_rankings["is_trained_target_day"].map(_as_bool)
    ].copy()
    numeric_lead_days = pd.to_numeric(trained["lead_day"], errors="coerce")
    lead_days = sorted(int(value) for value in numeric_lead_days.dropna().unique())
    for lead_day in lead_days:
        rows = trained.loc[numeric_lead_days == lead_day].copy()
        rows = rows.sort_values(
            ["mae", "updated_at"],
            kind="stable",
            na_position="last",
        )
        stem = f"experiment_rankings_horizon_d{lead_day:02d}"
        csv_path = Path(ranking_dir) / f"{stem}.csv"
        tex_path = Path(ranking_dir) / f"{stem}.tex"
        _write_csv_atomically(rows[HORIZON_RANKING_COLUMNS], csv_path)
        _write_text_atomically(
            _horizon_ranking_tex(rows, backend, lead_day),
            tex_path,
        )
        if compile_pdf:
            compile_tex(tex_path)


def _horizon_ranking_tex(
    rankings: pd.DataFrame,
    backend: str,
    lead_day: int,
) -> str:
    backend_label = "PyTorch" if backend == "pytorch" else "TensorFlow"
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=1.2cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{pdflscape}",
        r"\usepackage[table]{xcolor}",
        r"\begin{document}",
        r"\begin{landscape}",
        rf"\section*{{{backend_label} Experiment Rankings --- D+{lead_day}}}",
        rf"Generated: {latex_escape(generated_at)}. Lower MAE and MSE are better. Ties share the same rank.",
        *_ranking_table(rankings, rf"Forecast lead day D+{lead_day}", ""),
    ]
    for group, label in THRESHOLD_GROUPS:
        lines.extend(_ranking_table(rankings, label, f"{group}_"))
    lines.extend([r"\end{landscape}", r"\end{document}", ""])
    return "\n".join(lines)


def _ranking_table(
    rankings: pd.DataFrame,
    title: str,
    prefix: str,
) -> list[str]:
    table_rankings = _select_best_spectral_sigma_rows(rankings, prefix).sort_values(
        [
            *MATCHED_CONFIGURATION_COLUMNS,
            "n_clusters",
            "clustering_method",
            "updated_at",
        ],
        kind="stable",
        na_position="last",
    )
    n_column = f"{prefix}n" if prefix else "n"
    has_cutoff = bool(prefix)
    cutoff_header = " & Cutoff (mm)" if has_cutoff else ""
    alignment_columns = ["l", "r", "r", "l", "l", "r", "r", "c", "r"]
    if has_cutoff:
        alignment_columns.append("r")
    alignment_columns.extend(["r", "r", "r", "r"])
    alignment = "@{}" + "".join(alignment_columns) + "@{}"
    header = (
        r"Date & Window & $K$ & Clustering & Loss & $U_1$ & $U_2$ & "
        r"Precip. std. & $N$"
        + cutoff_header
        + r" & MAE & Rank & MSE & Rank \\"
    )
    lines = [
        rf"\subsection*{{{title}}}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\rowcolors{2}{gray!10}{white}",
        rf"\begin{{longtable}}{{{alignment}}}",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
    ]
    previous_group: tuple[object, ...] | None = None
    for row in table_rankings.itertuples(index=False):
        current_group = tuple(
            None if pd.isna(getattr(row, column)) else getattr(row, column)
            for column in MATCHED_CONFIGURATION_COLUMNS
        )
        if previous_group is not None and current_group != previous_group:
            lines.append(r"\midrule")
        previous_group = current_group
        row_values = row._asdict()
        cells = [
            _latex_text(row.date),
            str(int(row.window_size)),
            str(int(row.n_clusters)),
            (
                "-"
                if int(row.n_clusters) == 1
                else _latex_text(row.clustering_method)
            ),
            latex_escape(str(row.loss_function)),
            _latex_integer(row.lstm_units),
            _latex_optional_layer(row.lstm_units_2),
            _latex_bool(row.precipitation_standardized),
            _latex_integer(row_values[n_column]),
        ]
        if has_cutoff:
            cells.append(_latex_metric(row_values[f"{prefix}threshold_mm"]))
        cells.extend(
            [
                _latex_metric(row_values[f"{prefix}mae"]),
                _latex_integer(row_values[f"{prefix}mae_rank"]),
                _latex_metric(row_values[f"{prefix}mse"]),
                _latex_integer(row_values[f"{prefix}mse_rank"]),
            ]
        )
        lines.append(" & ".join(cells) + r" \\")
    if table_rankings.empty:
        column_count = 14 if has_cutoff else 13
        lines.append(
            rf"\multicolumn{{{column_count}}}{{c}}{{No completed experiments yet.}} \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\clearpage"])
    return lines


def _select_best_spectral_sigma_rows(
    rankings: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """Keep the lowest-MAE spectral sigma in every matched configuration/K."""
    if rankings.empty:
        return rankings.copy()

    working = rankings.copy()
    method = working["clustering_method"].astype("string")
    spectral_mask = method.str.lower().str.startswith("spectral", na=False)
    if not spectral_mask.any():
        return working

    order_column = "__spectral_selection_order"
    metric_column = "__spectral_selection_mae"
    working[order_column] = np.arange(len(working))
    working[metric_column] = pd.to_numeric(
        working[f"{prefix}mae"],
        errors="coerce",
    )
    spectral = working.loc[spectral_mask].sort_values(
        [
            *MATCHED_CONFIGURATION_COLUMNS,
            "n_clusters",
            metric_column,
            "updated_at",
            "clustering_method",
            order_column,
        ],
        kind="stable",
        na_position="last",
    )
    best_orders = set(
        spectral.drop_duplicates(
            [*MATCHED_CONFIGURATION_COLUMNS, "n_clusters"],
            keep="first",
        )[order_column]
    )
    keep = ~spectral_mask | working[order_column].isin(best_orders)
    return working.loc[keep].drop(columns=[order_column, metric_column])


def _latex_metric(value: object) -> str:
    return "--" if pd.isna(value) else f"{float(value):.4f}"


def _latex_bold_green(value: str) -> str:
    return rf"\textcolor{{rankingbestgreen}}{{\textbf{{{value}}}}}"


def _latex_integer(value: object) -> str:
    return "--" if pd.isna(value) else str(int(value))


def _latex_optional_layer(value: object) -> str:
    return "-" if pd.isna(value) else str(int(value))


def _latex_text(value: object) -> str:
    return "--" if pd.isna(value) else latex_escape(str(value))


def _latex_bool(value: object) -> str:
    if pd.isna(value):
        return "--"
    if isinstance(value, str):
        return "True" if value.strip().lower() == "true" else "False"
    return "True" if bool(value) else "False"


def _as_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)
