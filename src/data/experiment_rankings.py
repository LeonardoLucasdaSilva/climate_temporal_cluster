"""Persistent cross-run rankings for LSTM experiment configurations."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import errno
import os
from pathlib import Path
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

IDENTIFICATION_COLUMNS = [
    "run_id",
    "run_path",
    "window_size",
    "n_clusters",
    "clustering_method",
    "loss_function",
    "patience_metric",
    "lstm_units",
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
RANKING_LOCK_TIMEOUT_SECONDS = 600.0


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
    tex_path = resolved_ranking_dir / "experiment_rankings.tex"
    with _ranking_lock(resolved_ranking_dir):
        rankings = _rank_experiments(_load_rankings(csv_path))
        _write_csv_atomically(rankings, csv_path)
        _write_text_atomically(
            _ranking_tex(rankings, normalized_backend),
            tex_path,
        )
        pdf_path = compile_tex(tex_path) if compile_pdf else None
    return csv_path, tex_path, pdf_path


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
    tex_path = ranking_dir / "experiment_rankings.tex"
    resolved_output = Path(output_dir).resolve()
    try:
        run_path = resolved_output.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        run_path = resolved_output.as_posix()

    threshold_metrics, sample_count = _threshold_metrics(actual, predicted)
    row = {
        "run_id": f"{resolved_output.parent.name}/{resolved_output.name}",
        "run_path": run_path,
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
        "n": sample_count,
        "mae": _finite_metric(mae, "mae"),
        "mse": _finite_metric(mse, "mse"),
        "r2": _finite_metric(r2, "r2"),
        **threshold_metrics,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    with _ranking_lock(ranking_dir):
        # The complete read-modify-write transaction must be serialized. Sweep
        # configurations may finish in different worker processes, and locking
        # only the final replace would allow a stale frame to discard a run.
        rankings = _load_rankings(csv_path)
        rankings = rankings.loc[rankings["run_path"] != run_path].copy()
        rankings = pd.concat([rankings, pd.DataFrame([row])], ignore_index=True)
        rankings = _rank_experiments(rankings)
        _write_csv_atomically(rankings, csv_path)
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
    if "loss_function" in rankings:
        rankings["loss_function"] = rankings["loss_function"].map(
            _abbreviate_saved_loss
        )
    for column in RANKING_COLUMNS:
        if column not in rankings:
            rankings[column] = pd.NA
    return rankings[RANKING_COLUMNS]


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


def _write_csv_atomically(rankings: pd.DataFrame, csv_path: Path) -> None:
    temporary_path = csv_path.with_suffix(".csv.tmp")
    rankings.to_csv(temporary_path, index=False)
    temporary_path.replace(csv_path)


def _write_text_atomically(contents: str, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(contents, encoding="utf-8")
    temporary_path.replace(path)


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
        rf"\section*{{{backend_label} Experiment Rankings}}",
        rf"Generated: {latex_escape(generated_at)}. Lower MAE and MSE are better; higher $R^2$ is better. Ties share the same rank.",
        r"\begin{landscape}",
        *_ranking_table(rankings, "Overall test set", ""),
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


def _ranking_table(
    rankings: pd.DataFrame,
    title: str,
    prefix: str,
) -> list[str]:
    table_rankings = rankings.sort_values(
        [
            f"{prefix}mae",
            "updated_at",
        ],
        kind="stable",
        na_position="last",
    )
    n_column = f"{prefix}n" if prefix else "n"
    has_cutoff = bool(prefix)
    cutoff_header = " & Cutoff (mm)" if has_cutoff else ""
    alignment = (
        r"p{3.8cm}rrp{1.8cm}p{2.5cm}p{1.2cm}rrr rr rr rr"
        if has_cutoff
        else r"p{3.8cm}rrp{1.8cm}p{2.5cm}p{1.2cm}rr rr rr rr"
    )
    header = (
        r"Run & Window & $K$ & Clustering & Loss & Patience & $U_1$ & $N$"
        + cutoff_header
        + r" & MAE & Rank & MSE & Rank & $R^2$ & Rank \\"
    )
    lines = [
        rf"\subsection*{{{title}}}",
        r"\small",
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
    for row in table_rankings.itertuples(index=False):
        row_values = row._asdict()
        cells = [
            latex_escape(_display_run_id(str(row.run_id))),
            str(int(row.window_size)),
            str(int(row.n_clusters)),
            latex_escape(str(row.clustering_method)),
            latex_escape(str(row.loss_function)),
            _latex_text(row.patience_metric),
            _latex_integer(row.lstm_units),
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
                _latex_metric(row_values[f"{prefix}r2"]),
                _latex_integer(row_values[f"{prefix}r2_rank"]),
            ]
        )
        lines.append(" & ".join(cells) + r" \\")
    if table_rankings.empty:
        column_count = 15 if has_cutoff else 14
        lines.append(
            rf"\multicolumn{{{column_count}}}{{c}}{{No completed experiments yet.}} \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\clearpage"])
    return lines


def _latex_metric(value: object) -> str:
    return "--" if pd.isna(value) else f"{float(value):.4f}"


def _latex_integer(value: object) -> str:
    return "--" if pd.isna(value) else str(int(value))


def _latex_text(value: object) -> str:
    return "--" if pd.isna(value) else latex_escape(str(value))


def _display_run_id(run_id: str) -> str:
    parent, separator, name = run_id.partition("/")
    parent = parent.removeprefix("requested_")
    return f"{parent}{separator}{name}"
