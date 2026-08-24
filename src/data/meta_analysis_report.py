"""Cross-experiment metric loading and LaTeX meta-analysis reports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from config import DATA_ROOT
from data.beamer_report import latex_escape
from data.load_data import load_station_daily_data
from data.lstm_outputs import format_latex_number


META_ANALYSIS_METRICS = ("MSE", "MAE", "R2")
LEAD_DAY_TABLE_METRICS = ("RMSE", "MAE", "R2")
SUPPORTED_METRIC_FILES = {
    "sweep_results.csv",
    "metrics_summary.csv",
    "test_prediction_metrics_by_lead_day.csv",
    "comparative_metrics.csv",
}
DEFAULT_RUNS_PER_TABLE = 10
DEFAULT_DIGITS = 4
META_TIMESERIES_DIRNAME = "meta_analysis_timeseries"
_OBSERVED_TIMESERIES_COLOR = "#465362"
_OBSERVED_TIMESERIES_ALPHA = 0.52
_OBSERVED_TIMESERIES_LINEWIDTH = 1.05
_OBSERVED_TIMESERIES_ZORDER = 2
_PREDICTED_TIMESERIES_ZORDER = 3


@dataclass(frozen=True)
class MetaAnalysisRun:
    """Metrics and traceability retained for one physical experiment run."""

    run_name: str
    experiment_name: str
    experiment_path: Path
    run_path: Path
    source_path: Path
    metric_scope: str
    overall_metrics: Mapping[str, float | None]
    lead_day_metrics: Mapping[int, Mapping[str, float | None]]
    n_test: int | None = None
    forecast_horizon: int | None = None
    alignment_context: Mapping[
        int,
        tuple[str | None, str | None, int | None],
    ] = field(default_factory=dict)


@dataclass(frozen=True)
class _LeadMetricData:
    metrics: Mapping[int, Mapping[str, float | None]]
    n_test_by_day: Mapping[int, int | None]
    forecast_horizon: int


@dataclass(frozen=True)
class MetaAnalysisTimeseriesPlot:
    """A generated meta-analysis time-series comparison plot."""

    lead_day: int
    path: Path
    start_date: str
    end_date: str
    n_common_dates: int


def load_meta_analysis_runs(
    experiment_paths: Sequence[Path],
) -> list[MetaAnalysisRun]:
    """Load and deduplicate every run found under the selected inputs."""
    if not experiment_paths:
        raise ValueError("At least one experiment path is required.")

    sources: list[Path] = []
    seen_sources: set[str] = set()
    for input_path in experiment_paths:
        for source_path in _discover_metric_sources(Path(input_path)):
            source_key = _path_key(source_path)
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            sources.append(source_path)

    runs: list[MetaAnalysisRun] = []
    seen_runs: set[tuple[str, ...]] = set()
    for source_path in sources:
        for run in _load_metric_source(source_path):
            run_key = _run_key(run)
            if run_key in seen_runs:
                continue
            seen_runs.add(run_key)
            runs.append(run)

    if not runs:
        raise ValueError("No experiment runs with MSE, MAE, and R2 were found.")

    scopes = {run.metric_scope for run in runs}
    if len(scopes) > 1:
        raise ValueError(
            "Raw and common-date aligned metrics cannot be mixed in one report. "
            "Create separate reports for raw run artifacts and "
            "comparative_metrics.csv inputs."
        )
    if scopes == {"aligned"}:
        _validate_aligned_contexts(runs)
    return runs


def render_meta_analysis_report(
    runs: Sequence[MetaAnalysisRun],
    *,
    title: str = "Cross-Experiment Meta-Analysis",
    runs_per_table: int = DEFAULT_RUNS_PER_TABLE,
    digits: int = DEFAULT_DIGITS,
    timeseries_plots: Sequence[MetaAnalysisTimeseriesPlot] = (),
    figure_base_dir: Path | None = None,
) -> str:
    """Return a complete LaTeX report with runs arranged as columns."""
    runs = list(runs)
    if not runs:
        raise ValueError("At least one run is required to render the report.")
    if runs_per_table <= 0:
        raise ValueError("runs_per_table must be positive.")
    if digits < 0:
        raise ValueError("digits cannot be negative.")

    scopes = {run.metric_scope for run in runs}
    if len(scopes) != 1:
        raise ValueError("Every run in one report must use the same metric scope.")
    metric_scope = next(iter(scopes))
    if metric_scope == "aligned":
        _validate_aligned_contexts(runs)
    lead_days = sorted(
        {
            int(lead_day)
            for run in runs
            for lead_day in run.lead_day_metrics
        }
    )

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.7in]{geometry}",
        r"\usepackage{array}",
        r"\usepackage{booktabs}",
        r"\usepackage{caption}",
        r"\usepackage{float}",
        r"\usepackage{graphicx}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\usepackage{longtable}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\captionsetup{font=small,labelfont=bf}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{5pt}",
        r"\begin{document}",
        rf"\title{{{latex_escape(title)}}}",
        r"\author{}",
        r"\date{}",
        r"\maketitle",
        r"\vspace{-2em}",
        r"\tableofcontents",
        r"\clearpage",
        r"\section{Method}",
        (
            f"This report compares {len(runs)} runs from "
            f"{len({_path_key(run.experiment_path) for run in runs})} "
            "experiment folders. "
            "Columns are numbered globally from Run 1 through "
            f"Run {len(runs)}."
        ),
        (
            "MSE/RMSE and MAE are minimized; $R^2$ is maximized. "
            "The best available value across all runs for each metric is bold."
        ),
        (
            "Metrics are reproduced from saved experiment artifacts and are not "
            "pooled statistical estimates. Test sample counts, forecast horizons, "
            "stations, and date intervals may differ; verify the run registry "
            "before interpreting rankings."
        ),
        (
            "Metric scope: raw per-run test artifacts."
            if metric_scope == "raw"
            else (
                "Metric scope: common-date aligned comparative artifacts. "
                "When multiple comparative sources are included, their "
                "lead days, date intervals, and sample counts must match."
            )
        ),
        r"\section{Run Registry}",
        _run_registry_table(runs),
        *_timeseries_plot_section(timeseries_plots, figure_base_dir),
        r"\bigskip",
        r"\section{Overall Test Metrics}",
        (
            "Overall values use each raw run's configured forecast horizon "
            "when lead-day metrics are available, or its saved Test summary "
            "otherwise. Aligned comparative inputs use their greatest "
            "reported lead day."
        ),
        *_metric_table_blocks(
            runs,
            [run.overall_metrics for run in runs],
            caption="Overall test metrics",
            label_prefix="overall",
            runs_per_table=runs_per_table,
            digits=digits,
            metrics=META_ANALYSIS_METRICS,
        ),
    ]

    if lead_days:
        lines.extend([r"\bigskip", r"\section{Metrics by Lead Day}"])
        for index, lead_day in enumerate(lead_days):
            if index:
                lines.append(r"\medskip")
            lines.extend(
                _metric_table_blocks(
                    runs,
                    [
                        run.lead_day_metrics.get(lead_day, {})
                        for run in runs
                    ],
                    caption=f"Test metrics for lead day D+{lead_day}",
                    label_prefix=f"lead_day_{lead_day:02d}",
                    runs_per_table=runs_per_table,
                    digits=digits,
                    metrics=LEAD_DAY_TABLE_METRICS,
                )
            )

    lines.extend([r"\end{document}", ""])
    return "\n".join(lines)


def write_meta_analysis_report(
    runs: Sequence[MetaAnalysisRun],
    output_path: Path,
    *,
    title: str = "Cross-Experiment Meta-Analysis",
    runs_per_table: int = DEFAULT_RUNS_PER_TABLE,
    digits: int = DEFAULT_DIGITS,
    timeseries_plots: Sequence[MetaAnalysisTimeseriesPlot] = (),
) -> Path:
    """Write the rendered meta-analysis report and return its path."""
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".tex")
    elif output_path.suffix.lower() != ".tex":
        raise ValueError("The meta-analysis output must use the .tex extension.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_meta_analysis_report(
            runs,
            title=title,
            runs_per_table=runs_per_table,
            digits=digits,
            timeseries_plots=timeseries_plots,
            figure_base_dir=output_path.parent,
        ),
        encoding="utf-8",
    )
    return output_path


def create_meta_analysis_report(
    experiment_paths: Sequence[Path],
    output_path: Path,
    *,
    title: str = "Cross-Experiment Meta-Analysis",
    runs_per_table: int = DEFAULT_RUNS_PER_TABLE,
    digits: int = DEFAULT_DIGITS,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[Path, list[MetaAnalysisRun]]:
    """Load selected experiments and write their meta-analysis report."""
    runs = load_meta_analysis_runs(experiment_paths)
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".tex")
    timeseries_plots = (
        save_meta_analysis_timeseries_plots(
            runs,
            output_path.parent / META_TIMESERIES_DIRNAME,
            start_date=start_date,
            end_date=end_date,
        )
        if start_date is not None or end_date is not None
        else []
    )
    report_path = write_meta_analysis_report(
        runs,
        output_path,
        title=title,
        runs_per_table=runs_per_table,
        digits=digits,
        timeseries_plots=timeseries_plots,
    )
    return report_path, runs


def save_meta_analysis_timeseries_plots(
    runs: Sequence[MetaAnalysisRun],
    output_dir: Path,
    *,
    start_date: str | None,
    end_date: str | None,
) -> list[MetaAnalysisTimeseriesPlot]:
    """Save per-lead observed-vs-run prediction plots for the chosen period."""
    runs = list(runs)
    if not runs:
        raise ValueError("At least one run is required to plot time series.")

    start_bound = _optional_date_bound(start_date, "start_date")
    end_bound = _optional_date_bound(end_date, "end_date")
    if (
        start_bound is not None
        and end_bound is not None
        and start_bound > end_bound
    ):
        raise ValueError("start_date cannot be after end_date.")

    predictions = _meta_analysis_predictions_dataframe(runs)
    if start_bound is not None:
        predictions = predictions[predictions["target_date"] >= start_bound]
    if end_bound is not None:
        predictions = predictions[predictions["target_date"] <= end_bound]
    if predictions.empty:
        raise ValueError(
            "No prediction rows were found inside the selected date period."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_plot in output_dir.glob("01_meta_timeseries_comparison_lead_day_*.png"):
        stale_plot.unlink()
    palette = dict(
        zip(
            range(1, len(runs) + 1),
            sns.color_palette("colorblind", n_colors=max(len(runs), 1)),
        )
    )
    prediction_linewidth, prediction_alpha = (
        _timeseries_prediction_aesthetics(len(runs))
    )
    plots: list[MetaAnalysisTimeseriesPlot] = []
    incomplete_leads: list[int] = []
    for lead_day, lead_values in predictions.groupby("lead_day", sort=True):
        lead_day = int(lead_day)
        run_sets = {
            int(run_number): set(values["target_date"])
            for run_number, values in lead_values.groupby("run_number")
        }
        if set(run_sets) != set(range(1, len(runs) + 1)):
            incomplete_leads.append(lead_day)
            continue
        common_dates = sorted(set.intersection(*run_sets.values()))
        if not common_dates:
            incomplete_leads.append(lead_day)
            continue
        plot_values = lead_values[lead_values["target_date"].isin(common_dates)]
        actual = _common_actual_timeseries(plot_values, lead_day)
        plot_path = (
            output_dir
            / f"01_meta_timeseries_comparison_lead_day_{lead_day:02d}.png"
        )
        _save_meta_timeseries_plot(
            actual,
            plot_values,
            plot_path,
            lead_day=lead_day,
            palette=palette,
            prediction_linewidth=prediction_linewidth,
            prediction_alpha=prediction_alpha,
        )
        plots.append(
            MetaAnalysisTimeseriesPlot(
                lead_day=lead_day,
                path=plot_path,
                start_date=f"{actual['target_date'].iloc[0]:%Y-%m-%d}",
                end_date=f"{actual['target_date'].iloc[-1]:%Y-%m-%d}",
                n_common_dates=len(common_dates),
            )
        )

    if incomplete_leads:
        lead_text = ", ".join(f"D+{lead_day}" for lead_day in incomplete_leads)
        raise ValueError(
            "Every selected run must contain common target dates inside the "
            f"selected period for each plotted lead day; missing {lead_text}."
        )
    if not plots:
        raise ValueError(
            "No lead-day plots could be created for the selected date period."
        )
    return plots


def _meta_analysis_predictions_dataframe(
    runs: Sequence[MetaAnalysisRun],
) -> pd.DataFrame:
    frames = []
    for run_number, run in enumerate(runs, start=1):
        source_path = _prediction_source_path(run)
        frame = _read_csv(source_path)
        lead_column = _required_column(frame, "lead_day", source_path)
        date_column = _optional_column(frame, "target_date")
        actual_column = _required_column(frame, "actual_mm", source_path)
        predicted_column = _required_column(frame, "predicted_mm", source_path)
        if run.metric_scope == "aligned":
            run_column = _required_column(frame, "run_name", source_path)
            frame = frame[
                frame[run_column].astype(str).str.strip() == run.run_name
            ].copy()
            if frame.empty:
                raise ValueError(
                    f"Aligned prediction source {source_path} has no rows "
                    f"for run {run.run_name!r}."
                )

        target_dates = (
            pd.to_datetime(frame[date_column], errors="coerce")
            if date_column is not None
            else _derive_target_dates_from_window_indices(
                frame,
                run,
                source_path,
                lead_column,
            )
        )
        normalized = pd.DataFrame(
            {
                "run_number": run_number,
                "run_name": run.run_name,
                "run_label": f"Run {run_number}",
                "lead_day": pd.to_numeric(
                    frame[lead_column],
                    errors="coerce",
                ),
                "target_date": target_dates,
                "actual_mm": pd.to_numeric(
                    frame[actual_column],
                    errors="coerce",
                ),
                "predicted_mm": pd.to_numeric(
                    frame[predicted_column],
                    errors="coerce",
                ),
            }
        )
        invalid_columns = [
            column
            for column in ("lead_day", "target_date", "actual_mm", "predicted_mm")
            if normalized[column].isna().any()
        ]
        if invalid_columns:
            raise ValueError(
                f"{source_path} contains invalid prediction values in: "
                + ", ".join(invalid_columns)
            )
        normalized["lead_day"] = normalized["lead_day"].astype(int)
        duplicated = normalized.duplicated(
            ["run_number", "lead_day", "target_date"],
            keep=False,
        )
        if duplicated.any():
            duplicate = normalized[duplicated].iloc[0]
            raise ValueError(
                "Prediction rows must be unique by run, lead day, and target "
                f"date; duplicate found for Run {run_number}, "
                f"D+{int(duplicate['lead_day'])}, "
                f"{duplicate['target_date']:%Y-%m-%d} in {source_path}."
            )
        frames.append(normalized)

    if not frames:
        raise ValueError("No prediction sources were selected.")
    return pd.concat(frames, ignore_index=True)


def _derive_target_dates_from_window_indices(
    frame: pd.DataFrame,
    run: MetaAnalysisRun,
    source_path: Path,
    lead_column: str,
) -> pd.Series:
    window_column = _optional_column(frame, "window_index")
    if window_column is None:
        raise ValueError(
            f"{source_path} is missing required column 'target_date'. "
            "Legacy fallback also requires 'window_index'."
        )
    station_state, station_id = _run_station(run)
    window_size = _run_window_size(run)
    if station_state is None or station_id is None or window_size is None:
        raise ValueError(
            f"{source_path} is missing required column 'target_date', and "
            "the run summary does not contain enough context to derive dates "
            "(Station and Window size are required)."
        )

    station_frame = load_station_daily_data(station_state, station_id, DATA_ROOT)
    if "Data" not in station_frame.columns:
        raise ValueError(
            f"Cannot derive target dates for {source_path}: station data has "
            "no 'Data' column."
        )
    window_indices = pd.to_numeric(frame[window_column], errors="coerce")
    lead_days = pd.to_numeric(frame[lead_column], errors="coerce")
    target_indices = window_indices + int(window_size) - 1 + lead_days
    if (
        window_indices.isna().any()
        or lead_days.isna().any()
        or (target_indices < 0).any()
        or (target_indices >= len(station_frame)).any()
    ):
        raise ValueError(
            f"Cannot derive target dates for {source_path}: invalid "
            "window_index or lead_day values."
        )
    dates = pd.to_datetime(
        station_frame.iloc[target_indices.astype(int).to_numpy()]["Data"],
        errors="coerce",
    )
    if dates.isna().any():
        raise ValueError(
            f"Cannot derive target dates for {source_path}: station dates "
            "could not be parsed."
        )
    return pd.Series(dates.to_numpy(), index=frame.index)


def _run_station(run: MetaAnalysisRun) -> tuple[str | None, str | None]:
    summary = _run_summary_fields(run)
    station = summary.get("station")
    if station and "/" in station:
        state, station_id = [part.strip() for part in station.split("/", 1)]
        if state and station_id:
            return state, station_id
    match = re.search(r"(?:^|_)([A-Z]{2})_([A-Z]\d{3})(?:_|$)", run.experiment_name)
    if match:
        return match.group(1), match.group(2)
    return None, None


def _run_window_size(run: MetaAnalysisRun) -> int | None:
    summary = _run_summary_fields(run)
    value = summary.get("window size")
    if value is None:
        return _window_size_from_run_name(run.run_name)
    try:
        return int(float(value.split()[0]))
    except (TypeError, ValueError, IndexError):
        return _window_size_from_run_name(run.run_name)


def _run_summary_fields(run: MetaAnalysisRun) -> dict[str, str]:
    summary_path = run.run_path / "summary.txt"
    if not summary_path.is_file():
        return {}
    fields: dict[str, str] = {}
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().casefold()] = value.strip()
    return fields


def _window_size_from_run_name(run_name: str) -> int | None:
    match = re.search(r"(?:^|_)w(\d+)(?:_|$)", run_name)
    return int(match.group(1)) if match else None


def _prediction_source_path(run: MetaAnalysisRun) -> Path:
    if run.metric_scope == "aligned":
        source_path = run.source_path.parent / "aligned_test_predictions.csv"
    else:
        source_path = (
            run.run_path
            / "forecast_horizon_diagnostics"
            / "test_prediction_by_lead_day.csv"
        )
    if not source_path.is_file():
        raise FileNotFoundError(
            "Time-series plots require saved prediction rows for every "
            f"selected run. Missing: {source_path}"
        )
    return source_path


def _common_actual_timeseries(
    plot_values: pd.DataFrame,
    lead_day: int,
) -> pd.DataFrame:
    actual_ranges = (
        plot_values.groupby("target_date")["actual_mm"]
        .agg(["min", "max"])
        .reset_index()
    )
    conflicts = (
        (actual_ranges["max"] - actual_ranges["min"]).abs() > 1e-9
    )
    if conflicts.any():
        conflict_date = actual_ranges[conflicts]["target_date"].iloc[0]
        raise ValueError(
            "Selected runs disagree on observed precipitation for "
            f"D+{lead_day} at {conflict_date:%Y-%m-%d}."
        )
    return (
        plot_values.groupby("target_date", as_index=False)["actual_mm"]
        .first()
        .sort_values("target_date")
    )


def _save_meta_timeseries_plot(
    actual: pd.DataFrame,
    plot_values: pd.DataFrame,
    output_path: Path,
    *,
    lead_day: int,
    palette: Mapping[int, object],
    prediction_linewidth: float,
    prediction_alpha: float,
) -> None:
    run_numbers = sorted(int(value) for value in plot_values["run_number"].unique())
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=_OBSERVED_TIMESERIES_COLOR,
            linewidth=1.6,
            alpha=_OBSERVED_TIMESERIES_ALPHA,
            linestyle=(0, (5, 2)),
        ),
        *[
            Line2D(
                [0],
                [0],
                color=palette[run_number],
                linewidth=1.6,
                alpha=prediction_alpha,
            )
            for run_number in run_numbers
        ],
    ]
    legend_labels = ["Observed", *(f"Run {number}" for number in run_numbers)]
    legend_columns = min(len(legend_handles), 5)
    legend_rows = math.ceil(len(legend_handles) / legend_columns)
    legend_order = _legend_column_major_order(len(legend_handles), legend_columns)
    figure_height = 5.6 + 0.32 * legend_rows

    with sns.axes_style(
        "whitegrid",
        rc={
            "axes.edgecolor": "#C8D0D9",
            "axes.facecolor": "#FCFCFD",
            "grid.color": "#DCE2E8",
            "grid.linewidth": 0.65,
        },
    ):
        fig, axis = plt.subplots(figsize=(16, figure_height))
        sns.lineplot(
            data=actual,
            x="target_date",
            y="actual_mm",
            estimator=None,
            sort=False,
            color=_OBSERVED_TIMESERIES_COLOR,
            linewidth=_OBSERVED_TIMESERIES_LINEWIDTH,
            alpha=_OBSERVED_TIMESERIES_ALPHA,
            linestyle=(0, (5, 2)),
            zorder=_OBSERVED_TIMESERIES_ZORDER,
            legend=False,
            ax=axis,
        )
        sns.lineplot(
            data=plot_values.sort_values(["target_date", "run_number"]),
            x="target_date",
            y="predicted_mm",
            hue="run_number",
            hue_order=run_numbers,
            palette=palette,
            estimator=None,
            sort=False,
            linewidth=prediction_linewidth,
            alpha=prediction_alpha,
            zorder=_PREDICTED_TIMESERIES_ZORDER,
            legend=False,
            ax=axis,
        )
        start_date = pd.Timestamp(actual["target_date"].iloc[0])
        end_date = pd.Timestamp(actual["target_date"].iloc[-1])
        axis.set_title(
            "Meta-analysis test time-series comparison - "
            f"lead day D+{lead_day} | "
            f"{start_date:%d/%m/%Y} - {end_date:%d/%m/%Y}",
            fontsize=14,
            fontweight="semibold",
        )
        axis.set_xlabel("Target date")
        axis.set_ylabel("Precipitation (mm)")
        date_locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
        axis.xaxis.set_major_locator(date_locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(date_locator))
        axis.tick_params(axis="x", rotation=15)
        axis.margins(x=0.01)
        axis.set_axisbelow(True)
        axis.grid(False)
        axis.yaxis.grid(True, alpha=0.55)
        if plot_values[["actual_mm", "predicted_mm"]].min().min() >= 0.0:
            axis.set_ylim(bottom=0.0)
        sns.despine(ax=axis, top=True, right=True)
        fig.legend(
            [legend_handles[index] for index in legend_order],
            [legend_labels[index] for index in legend_order],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=legend_columns,
            frameon=False,
            handlelength=2.8,
            columnspacing=1.5,
            fontsize=9.5,
        )
        top_margin_inches = 0.58 + 0.32 * legend_rows
        fig.tight_layout(rect=(0, 0, 1, 1 - top_margin_inches / figure_height))
        fig.savefig(output_path, bbox_inches="tight", dpi=180)
        plt.close(fig)


def _timeseries_plot_section(
    plots: Sequence[MetaAnalysisTimeseriesPlot],
    figure_base_dir: Path | None,
) -> list[str]:
    if not plots:
        return []

    lines = [
        r"\clearpage",
        r"\section{Prediction Time Series Comparison}",
        (
            "These figures compare the observed precipitation series with the "
            "predicted series from every globally numbered run. Dates are "
            "restricted to the common target-date intersection available to "
            "all selected runs inside the requested period."
        ),
    ]
    for plot in sorted(plots, key=lambda item: item.lead_day):
        lines.extend(
            [
                r"\begin{figure}[H]",
                r"\centering",
                (
                    r"\includegraphics[width=0.98\textwidth,"
                    r"height=0.62\textheight,keepaspectratio]"
                    rf"{{{_latex_graphics_path(figure_base_dir, plot.path)}}}"
                ),
                (
                    r"\caption*{"
                    f"Lead day D+{plot.lead_day}: "
                    f"{latex_escape(plot.start_date)} to "
                    f"{latex_escape(plot.end_date)} "
                    f"({plot.n_common_dates} common target dates)."
                    r"}"
                ),
                r"\end{figure}",
            ]
        )
    return lines


def _latex_graphics_path(
    base_dir: Path | None,
    figure_path: Path,
) -> str:
    if base_dir is None:
        include_path = str(figure_path)
    else:
        include_path = os.path.relpath(figure_path, base_dir)
    return rf"\detokenize{{{include_path.replace('\\', '/')}}}"


def _legend_column_major_order(item_count: int, columns: int) -> list[int]:
    rows = math.ceil(item_count / columns)
    return [
        row_index * columns + column_index
        for column_index in range(columns)
        for row_index in range(rows)
        if row_index * columns + column_index < item_count
    ]


def _timeseries_prediction_aesthetics(run_count: int) -> tuple[float, float]:
    """Return line width and alpha that scale with comparison density."""
    if run_count <= 8:
        return 1.1, 0.82
    if run_count <= 16:
        return 0.95, 0.72
    return 0.8, 0.62


def _optional_date_bound(value: str | None, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{name} must be a valid date; got {value!r}.")
    return pd.Timestamp(parsed)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser used by the report runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a LaTeX MSE/MAE/R2 meta-analysis from saved experiment "
            "folders or metric CSV files."
        )
    )
    parser.add_argument(
        "experiment_paths",
        nargs="+",
        type=Path,
        help=(
            "Sweep folders, run folders, comparative_analysis folders, "
            "or supported metric CSV files."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("meta_analysis.tex"),
        help="Destination .tex file.",
    )
    parser.add_argument(
        "--title",
        default="Cross-Experiment Meta-Analysis",
        help="LaTeX report title.",
    )
    parser.add_argument(
        "--runs-per-table",
        type=_positive_cli_int,
        default=DEFAULT_RUNS_PER_TABLE,
        help="Maximum run columns per table block.",
    )
    parser.add_argument(
        "--digits",
        type=_nonnegative_cli_int,
        default=DEFAULT_DIGITS,
        help="Decimal places used for metric values.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help=(
            "Inclusive first target date for optional meta-analysis "
            "time-series plots (YYYY-MM-DD)."
        ),
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help=(
            "Inclusive last target date for optional meta-analysis "
            "time-series plots (YYYY-MM-DD)."
        ),
    )
    return parser


def _discover_metric_sources(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Experiment path not found: {input_path}")
    if input_path.is_file():
        if input_path.name.lower() not in SUPPORTED_METRIC_FILES:
            raise ValueError(
                f"Unsupported metric file {input_path.name!r}. Expected one of: "
                + ", ".join(sorted(SUPPORTED_METRIC_FILES))
            )
        return [input_path]

    direct_comparative = input_path / "comparative_metrics.csv"
    if (
        input_path.name.lower() == "comparative_analysis"
        and direct_comparative.is_file()
    ):
        return [direct_comparative]

    sources: list[Path] = []
    for filename in (
        "test_prediction_metrics_by_lead_day.csv",
        "metrics_summary.csv",
        "sweep_results.csv",
    ):
        candidates = sorted(
            (
                path
                for path in input_path.rglob(filename)
                if "comparative_analysis"
                not in {part.lower() for part in path.parts}
            ),
            key=lambda path: path.as_posix().casefold(),
        )
        sources.extend(candidates)

    if sources:
        return sources
    if direct_comparative.is_file():
        return [direct_comparative]
    raise ValueError(
        f"No supported metric artifacts were found under {input_path}."
    )


def _load_metric_source(source_path: Path) -> list[MetaAnalysisRun]:
    source_name = source_path.name.lower()
    if source_name == "test_prediction_metrics_by_lead_day.csv":
        return [_load_lead_day_run(source_path)]
    if source_name == "metrics_summary.csv":
        return [_load_metrics_summary_run(source_path)]
    if source_name == "sweep_results.csv":
        return _load_sweep_runs(source_path)
    if source_name == "comparative_metrics.csv":
        return _load_comparative_runs(source_path)
    raise ValueError(f"Unsupported metric source: {source_path}")


def _load_lead_day_run(source_path: Path) -> MetaAnalysisRun:
    run_dir = _run_dir_for_lead_source(source_path)
    experiment_dir = _experiment_dir(run_dir)
    lead_data = _read_lead_metric_data(source_path, sample_column="n_test")
    final_day = lead_data.forecast_horizon
    if final_day not in lead_data.metrics:
        raise ValueError(
            f"Configured forecast_horizon D+{final_day} is missing from "
            f"{source_path}."
        )
    return MetaAnalysisRun(
        run_name=run_dir.name,
        experiment_name=experiment_dir.name,
        experiment_path=experiment_dir,
        run_path=run_dir,
        source_path=source_path,
        metric_scope="raw",
        overall_metrics=lead_data.metrics[final_day],
        lead_day_metrics=lead_data.metrics,
        n_test=lead_data.n_test_by_day.get(final_day),
        forecast_horizon=lead_data.forecast_horizon,
    )


def _load_metrics_summary_run(source_path: Path) -> MetaAnalysisRun:
    run_dir = source_path.parent
    experiment_dir = _experiment_dir(run_dir)
    lead_path = (
        run_dir
        / "forecast_horizon_diagnostics"
        / "test_prediction_metrics_by_lead_day.csv"
    )
    if lead_path.is_file():
        return _load_lead_day_run(lead_path)

    frame = _read_csv(source_path)
    split_column = _required_column(frame, "split", source_path)
    test_rows = frame[
        frame[split_column].astype(str).str.strip().str.casefold() == "test"
    ]
    if len(test_rows) != 1:
        raise ValueError(
            f"{source_path} must contain exactly one Test row; found "
            f"{len(test_rows)}."
        )
    metric_columns = _metric_columns(frame, source_path)
    return MetaAnalysisRun(
        run_name=run_dir.name,
        experiment_name=experiment_dir.name,
        experiment_path=experiment_dir,
        run_path=run_dir,
        source_path=source_path,
        metric_scope="raw",
        overall_metrics=_metric_values(
            test_rows.iloc[0],
            metric_columns,
            source_path,
        ),
        lead_day_metrics={},
    )


def _load_sweep_runs(source_path: Path) -> list[MetaAnalysisRun]:
    frame = _read_csv(source_path)
    run_column = _required_column(frame, "run_name", source_path)
    metric_columns = {
        metric: _required_column(
            frame,
            f"test_{metric.lower()}",
            source_path,
        )
        for metric in META_ANALYSIS_METRICS
    }
    n_test_column = _optional_column(frame, "n_test")
    horizon_column = _optional_column(frame, "forecast_horizon")
    lead_path_column = _optional_column(frame, "lead_day_metrics_path")
    run_names = [
        _required_text(row[run_column], source_path, row_number)
        for row_number, (_, row) in enumerate(frame.iterrows(), start=1)
    ]
    normalized_names = pd.Series([name.casefold() for name in run_names])
    duplicate_names = normalized_names.duplicated(keep=False)
    if duplicate_names.any():
        duplicate_name = pd.Series(run_names)[duplicate_names].iloc[0]
        raise ValueError(
            f"Duplicate run_name {duplicate_name!r} in {source_path}."
        )

    runs = []
    for run_name, (_, row) in zip(run_names, frame.iterrows()):
        run_dir = source_path.parent / run_name
        lead_path = _sweep_lead_metric_path(
            run_dir,
            row,
            lead_path_column,
        )
        if lead_path is not None:
            runs.append(_load_lead_day_run(lead_path))
            continue

        runs.append(
            MetaAnalysisRun(
                run_name=run_name,
                experiment_name=source_path.parent.name,
                experiment_path=source_path.parent,
                run_path=run_dir,
                source_path=source_path,
                metric_scope="raw",
                overall_metrics=_metric_values(
                    row,
                    metric_columns,
                    source_path,
                ),
                lead_day_metrics={},
                n_test=_optional_integer(
                    row[n_test_column] if n_test_column else None,
                    source_path,
                    "n_test",
                ),
                forecast_horizon=_optional_integer(
                    row[horizon_column] if horizon_column else None,
                    source_path,
                    "forecast_horizon",
                ),
            )
        )
    return runs


def _load_comparative_runs(source_path: Path) -> list[MetaAnalysisRun]:
    frame = _read_csv(source_path)
    run_column = _required_column(frame, "run_name", source_path)
    lead_column = _required_column(frame, "lead_day", source_path)
    metric_columns = _metric_columns(frame, source_path)
    sample_column = _optional_column(frame, "n_common_test_dates")
    start_column = _optional_column(frame, "start_date")
    end_column = _optional_column(frame, "end_date")
    comparison_dir = source_path.parent
    sweep_dir = (
        comparison_dir.parent
        if comparison_dir.name.lower() == "comparative_analysis"
        else comparison_dir
    )

    validated_run_names = [
        _required_text(value, source_path, row_number)
        for row_number, value in enumerate(frame[run_column], start=1)
    ]
    frame = frame.copy()
    frame[run_column] = validated_run_names

    runs = []
    for run_name, run_values in frame.groupby(
        run_column,
        sort=False,
        dropna=False,
    ):
        lead_metrics: dict[int, Mapping[str, float | None]] = {}
        sample_counts: dict[int, int | None] = {}
        alignment_context: dict[
            int,
            tuple[str | None, str | None, int | None],
        ] = {}
        for _, row in run_values.iterrows():
            lead_day = _positive_integer(row[lead_column], source_path, "lead_day")
            if lead_day in lead_metrics:
                raise ValueError(
                    f"Duplicate lead day D+{lead_day} for {run_name!r} in "
                    f"{source_path}."
                )
            lead_metrics[lead_day] = _metric_values(
                row,
                metric_columns,
                source_path,
            )
            sample_counts[lead_day] = _optional_integer(
                row[sample_column] if sample_column else None,
                source_path,
                "n_common_test_dates",
            )
            alignment_context[lead_day] = (
                _optional_text(row[start_column] if start_column else None),
                _optional_text(row[end_column] if end_column else None),
                sample_counts[lead_day],
            )
        final_day = max(lead_metrics)
        runs.append(
            MetaAnalysisRun(
                run_name=run_name,
                experiment_name=sweep_dir.name,
                experiment_path=sweep_dir,
                run_path=sweep_dir / run_name,
                source_path=source_path,
                metric_scope="aligned",
                overall_metrics=lead_metrics[final_day],
                lead_day_metrics=lead_metrics,
                n_test=sample_counts.get(final_day),
                forecast_horizon=final_day,
                alignment_context=alignment_context,
            )
        )
    return runs


def _read_lead_metric_data(
    source_path: Path,
    *,
    sample_column: str,
) -> _LeadMetricData:
    frame = _read_csv(source_path)
    lead_column = _required_column(frame, "lead_day", source_path)
    metric_columns = _metric_columns(frame, source_path)
    samples_column = _optional_column(frame, sample_column)
    horizon_column = _optional_column(frame, "forecast_horizon")
    metrics: dict[int, Mapping[str, float | None]] = {}
    sample_counts: dict[int, int | None] = {}
    for _, row in frame.iterrows():
        lead_day = _positive_integer(row[lead_column], source_path, "lead_day")
        if lead_day in metrics:
            raise ValueError(f"Duplicate lead day D+{lead_day} in {source_path}.")
        metrics[lead_day] = _metric_values(row, metric_columns, source_path)
        sample_counts[lead_day] = _optional_integer(
            row[samples_column] if samples_column else None,
            source_path,
            sample_column,
        )
    if not metrics:
        raise ValueError(f"No lead-day metrics found in {source_path}.")

    configured_horizons = {
        value
        for value in (
            _optional_integer(
                row[horizon_column] if horizon_column else None,
                source_path,
                "forecast_horizon",
            )
            for _, row in frame.iterrows()
        )
        if value is not None
    }
    if len(configured_horizons) > 1:
        raise ValueError(
            f"Conflicting forecast_horizon values in {source_path}."
        )
    forecast_horizon = (
        next(iter(configured_horizons))
        if configured_horizons
        else max(metrics)
    )
    return _LeadMetricData(metrics, sample_counts, forecast_horizon)


def _metric_table_blocks(
    runs: Sequence[MetaAnalysisRun],
    metric_values: Sequence[Mapping[str, float | None]],
    *,
    caption: str,
    label_prefix: str,
    runs_per_table: int,
    digits: int,
    metrics: Sequence[str],
) -> list[str]:
    best_values = {
        metric: _best_metric_value(metric_values, metric)
        for metric in metrics
    }
    blocks: list[str] = []
    for chunk_start in range(0, len(runs), runs_per_table):
        chunk_end = min(chunk_start + runs_per_table, len(runs))
        chunk_metrics = metric_values[chunk_start:chunk_end]
        run_numbers = list(range(chunk_start + 1, chunk_end + 1))
        chunk_caption = (
            caption
            if len(runs) <= runs_per_table
            else f"{caption} (Runs {run_numbers[0]}--{run_numbers[-1]})"
        )
        blocks.extend(
            _metric_table(
                chunk_metrics,
                run_numbers,
                caption=chunk_caption,
                label=f"{label_prefix}_{chunk_start // runs_per_table + 1}",
                best_values=best_values,
                digits=digits,
                metrics=metrics,
            )
        )
    return blocks


def _metric_table(
    metric_values: Sequence[Mapping[str, float | None]],
    run_numbers: Sequence[int],
    *,
    caption: str,
    label: str,
    best_values: Mapping[str, float | None],
    digits: int,
    metrics: Sequence[str],
) -> list[str]:
    column_count = len(run_numbers)
    alignment = "l" + "r" * column_count
    use_resize = column_count > 6
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{tab:meta_{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
    ]
    if use_resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.extend(
        [
            rf"\begin{{tabular}}{{{alignment}}}",
            r"\toprule",
            rf"& \multicolumn{{{column_count}}}{{c}}{{Run}} \\",
            rf"\cmidrule(lr){{2-{column_count + 1}}}",
            "Metric & " + " & ".join(str(number) for number in run_numbers) + r" \\",
            r"\midrule",
        ]
    )

    metric_labels = {
        "MSE": r"MSE ($\mathrm{mm}^2$) $\downarrow$",
        "RMSE": r"RMSE ($\mathrm{mm}$) $\downarrow$",
        "MAE": r"MAE ($\mathrm{mm}$) $\downarrow$",
        "R2": r"$R^2$ $\uparrow$",
    }
    for metric in metrics:
        formatted_values = [
            _format_metric_value(
                values.get(metric),
                best_values[metric],
                digits,
            )
            for values in metric_values
        ]
        lines.append(
            metric_labels[metric]
            + " & "
            + " & ".join(formatted_values)
            + r" \\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if use_resize:
        lines.append(r"}")
    lines.extend([r"\end{table}", ""])
    return lines


def _run_registry_table(runs: Sequence[MetaAnalysisRun]) -> str:
    paths_by_name: dict[str, set[str]] = {}
    for run in runs:
        paths_by_name.setdefault(run.experiment_name.casefold(), set()).add(
            _path_key(run.experiment_path)
        )
    ambiguous_names = {
        name for name, paths in paths_by_name.items() if len(paths) > 1
    }

    lines = [
        r"\begingroup",
        r"\small",
        (
            r"\begin{longtable}{r "
            r">{\raggedright\arraybackslash}p{0.24\textwidth} "
            r">{\raggedright\arraybackslash}p{0.38\textwidth} r r}"
        ),
        r"\caption{Run numbering and source experiment metadata.}",
        r"\label{tab:meta_run_registry}\\",
        r"\toprule",
        r"Run & Experiment & Original run name & $N_{test}$ & Horizon \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Run & Experiment & Original run name & $N_{test}$ & Horizon \\",
        r"\midrule",
        r"\endhead",
    ]
    for run_number, run in enumerate(runs, start=1):
        n_test = "--" if run.n_test is None else str(run.n_test)
        horizon = (
            "--"
            if run.forecast_horizon is None
            else f"D+{run.forecast_horizon}"
        )
        experiment_label = (
            run.experiment_path.resolve().as_posix()
            if run.experiment_name.casefold() in ambiguous_names
            else run.experiment_name
        )
        lines.append(
            f"{run_number} & "
            f"{_latex_breakable_identifier(experiment_label)} & "
            f"{_latex_breakable_identifier(run.run_name)} & "
            f"{n_test} & {horizon} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup"])
    return "\n".join(lines)


def _format_metric_value(
    value: float | None,
    best_value: float | None,
    digits: int,
) -> str:
    formatted = format_latex_number(value, digits)
    if (
        value is not None
        and best_value is not None
        and math.isclose(value, best_value, rel_tol=1e-12, abs_tol=1e-12)
    ):
        return rf"\textbf{{{formatted}}}"
    return formatted


def _best_metric_value(
    metric_values: Sequence[Mapping[str, float | None]],
    metric: str,
) -> float | None:
    finite_values = [
        float(value)
        for values in metric_values
        for value in [values.get(metric)]
        if value is not None and math.isfinite(float(value))
    ]
    if not finite_values:
        return None
    return max(finite_values) if metric == "R2" else min(finite_values)


def _read_csv(source_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(source_path)
    if frame.empty:
        raise ValueError(f"Metric source is empty: {source_path}")
    return frame


def _metric_columns(
    frame: pd.DataFrame,
    source_path: Path,
) -> dict[str, str]:
    columns = {
        metric: _required_column(frame, metric, source_path)
        for metric in META_ANALYSIS_METRICS
    }
    rmse_column = _optional_column(frame, "RMSE")
    if rmse_column is not None:
        columns["RMSE"] = rmse_column
    return columns


def _metric_values(
    row: pd.Series,
    columns: Mapping[str, str],
    source_path: Path,
) -> dict[str, float | None]:
    values = {
        metric: _optional_metric(row[column], source_path, metric)
        for metric, column in columns.items()
    }
    if values.get("RMSE") is None and values.get("MSE") is not None:
        mse_value = values["MSE"]
        values["RMSE"] = (
            math.sqrt(mse_value)
            if mse_value is not None and mse_value >= 0.0
            else None
        )
    return values


def _required_column(
    frame: pd.DataFrame,
    requested: str,
    source_path: Path,
) -> str:
    column = _optional_column(frame, requested)
    if column is None:
        raise ValueError(
            f"{source_path} is missing required column {requested!r}."
        )
    return column


def _optional_column(frame: pd.DataFrame, requested: str) -> str | None:
    lookup = {
        str(column).strip().casefold(): str(column)
        for column in frame.columns
    }
    return lookup.get(requested.strip().casefold())


def _optional_metric(
    value: object,
    source_path: Path,
    metric: str,
) -> float | None:
    if pd.isna(value):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {metric} value {value!r} in {source_path}."
        ) from exc
    return numeric_value if math.isfinite(numeric_value) else None


def _positive_integer(
    value: object,
    source_path: Path,
    column: str,
) -> int:
    integer_value = _optional_integer(value, source_path, column)
    if integer_value is None or integer_value <= 0:
        raise ValueError(
            f"{column} must contain positive integers in {source_path}."
        )
    return integer_value


def _optional_integer(
    value: object,
    source_path: Path,
    column: str,
) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {column} value {value!r} in {source_path}."
        ) from exc
    if not numeric_value.is_integer():
        raise ValueError(
            f"{column} must contain integers in {source_path}; got {value!r}."
        )
    return int(numeric_value)


def _required_text(
    value: object,
    source_path: Path,
    row_number: int | None,
) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text:
        return text
    location = (
        f" row {row_number}" if row_number is not None else ""
    )
    raise ValueError(f"Missing run name in {source_path}{location}.")


def _optional_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _run_dir_for_lead_source(source_path: Path) -> Path:
    for ancestor in source_path.parents:
        if (ancestor / "metrics_summary.csv").is_file():
            return ancestor
    if source_path.parent.name.lower() == "forecast_horizon_diagnostics":
        return source_path.parent.parent
    return source_path.parent


def _experiment_dir(run_dir: Path) -> Path:
    for ancestor in run_dir.parents:
        if (ancestor / "sweep_results.csv").is_file():
            return ancestor
    return run_dir.parent


def _sweep_lead_metric_path(
    run_dir: Path,
    row: pd.Series,
    lead_path_column: str | None,
) -> Path | None:
    candidates = []
    if lead_path_column is not None and not pd.isna(row[lead_path_column]):
        configured_path = Path(str(row[lead_path_column]))
        candidates.append(
            configured_path
            if configured_path.is_absolute()
            else run_dir / configured_path
        )
    candidates.append(
        run_dir
        / "forecast_horizon_diagnostics"
        / "test_prediction_metrics_by_lead_day.csv"
    )
    return next((path for path in candidates if path.is_file()), None)


def _run_key(run: MetaAnalysisRun) -> tuple[str, ...]:
    if run.metric_scope == "aligned":
        return (
            run.metric_scope,
            _path_key(run.source_path),
            run.run_name.casefold(),
        )
    return (run.metric_scope, _path_key(run.run_path))


def _path_key(path: Path) -> str:
    return str(path.resolve()).casefold()


def _validate_aligned_contexts(runs: Sequence[MetaAnalysisRun]) -> None:
    """Ensure aligned artifacts describe one genuinely comparable test scope."""
    if not runs:
        return

    expected_leads = set(runs[0].lead_day_metrics)
    for run in runs:
        run_leads = set(run.lead_day_metrics)
        if run_leads != expected_leads:
            raise ValueError(
                "Aligned comparative sources must contain identical lead-day "
                "sets; found incompatible sets in "
                f"{runs[0].source_path} and {run.source_path}."
            )
        if set(run.alignment_context) != run_leads:
            raise ValueError(
                "Aligned comparative metrics are missing date/sample context "
                f"for one or more lead days in {run.source_path}."
            )

    source_count = len({_path_key(run.source_path) for run in runs})
    for lead_day in sorted(expected_leads):
        contexts = {
            run.alignment_context[lead_day]
            for run in runs
        }
        if len(contexts) != 1:
            raise ValueError(
                "Aligned comparative sources must use an identical alignment "
                f"context for D+{lead_day} (start_date, end_date, and "
                "n_common_test_dates)."
            )
        context = next(iter(contexts))
        if source_count > 1 and any(value is None for value in context):
            raise ValueError(
                "Combining multiple aligned comparative sources requires "
                "start_date, end_date, and n_common_test_dates for every "
                f"lead day; context is incomplete for D+{lead_day}."
            )


def _positive_cli_int(value: str) -> int:
    try:
        integer_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if integer_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return integer_value


def _nonnegative_cli_int(value: str) -> int:
    try:
        integer_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if integer_value < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return integer_value


def _latex_breakable_identifier(value: object) -> str:
    """Escape an identifier while allowing breaks at path separators."""
    return (
        latex_escape(value)
        .replace(r"\_", r"\_\allowbreak{}")
        .replace("/", r"/\allowbreak{}")
    )
