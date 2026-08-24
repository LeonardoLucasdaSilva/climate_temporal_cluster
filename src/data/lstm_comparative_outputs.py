"""Sweep-level comparative outputs for LSTM clustering experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import os
from pathlib import Path
import re
from typing import Mapping, Sequence
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation.metrics import calculate_regression_metrics


PIVOT_ALIASES = {
    "window": "window_size",
    "windows": "window_size",
    "window_sizes": "window_size",
    "j": "window_size",
    "k": "n_clusters",
    "cluster": "n_clusters",
    "clusters": "n_clusters",
    "cluster_count": "n_clusters",
    "number_of_clusters": "n_clusters",
    "algorithm": "clustering_algorithm",
    "algorithms": "clustering_algorithm",
    "clustering_algorithms": "clustering_algorithm",
    "lr": "learning_rate",
    "learning_rates": "learning_rate",
    "dropout": "dropout_rate",
}

PIVOT_LABELS = {
    "window_size": "Window size",
    "n_clusters": "Number of clusters (K)",
    "sigma": "Sigma",
    "learning_rate": "Learning rate",
    "dropout_rate": "Dropout rate",
    "weight_decay": "Weight decay",
    "lstm_units": "LSTM units (layer 1)",
    "lstm_units_2": "LSTM units (layer 2)",
    "batch_size": "Batch size",
    "epochs": "Maximum epochs",
    "patience": "Early-stopping patience",
    "warm_up": "Early-stopping warm-up epochs",
    "forecast_horizon": "Forecast horizon",
    "clustering_algorithm": "Clustering algorithm",
    "train_info": "Train-performance diagnostics",
    "silhouette_info": "Silhouette diagnostics",
}

COMPARATIVE_METRICS = ("RMSE", "MAE", "R2")
HISTORY_METRICS = ("loss", "mse", "mae", "r2")
REPORT_COMPARE_TEX_NAME = "report_compare.tex"
COMPARATIVE_ARTIFACT_NAMES = (
    "test_predictions_comparison.csv",
    "aligned_test_predictions.csv",
    "training_history_comparison.csv",
    "comparative_metrics.csv",
    "comparison_manifest.csv",
    "comparison_summary.txt",
    REPORT_COMPARE_TEX_NAME,
    "03_training_history_comparison.png",
)
COMPARATIVE_ARTIFACT_PATTERNS = (
    "01_test_timeseries_comparison_lead_day_*.png",
    "02_test_scatter_comparison_lead_day_*.png",
    "04_test_metrics_vs_*_lead_day_*.png",
    "05_overall_test_metrics_vs_*.png",
)
_OBSERVED_TIMESERIES_COLOR = "#465362"
_OBSERVED_TIMESERIES_ALPHA = 0.52
_OBSERVED_TIMESERIES_LINEWIDTH = 1.05
_PREDICTED_TIMESERIES_ZORDER = 3
_OBSERVED_TIMESERIES_ZORDER = 2


@dataclass(frozen=True)
class ComparativeRunData:
    """Predictions, histories, and metadata retained for one sweep test."""

    run_name: str
    parameters: Mapping[str, object]
    result_metrics: Mapping[str, object]
    actual_by_lead_day: np.ndarray
    predicted_by_lead_day: np.ndarray
    target_dates_by_lead_day: np.ndarray
    histories_by_cluster: Mapping[int, Mapping[str, Sequence[float]]]
    cluster_train_counts: Mapping[int, int]


def normalize_pivot_parameter(pivot_parameter: str) -> str:
    """Return the canonical snake-case name for a comparison pivot."""
    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(pivot_parameter).strip().lower(),
    ).strip("_")
    if not normalized:
        raise ValueError("pivot_parameter cannot be empty.")
    return PIVOT_ALIASES.get(normalized, normalized)


def validate_comparative_pivot(
    parameter_rows: Sequence[Mapping[str, object]],
    pivot_parameter: str,
) -> str:
    """Validate a pivot against the tests that will be compared."""
    if not parameter_rows:
        raise ValueError("Comparative analysis requires at least one test.")

    canonical = normalize_pivot_parameter(pivot_parameter)
    missing = [
        index
        for index, row in enumerate(parameter_rows, start=1)
        if canonical not in row
    ]
    if missing:
        available = sorted(
            set.intersection(*(set(row) for row in parameter_rows))
        )
        raise ValueError(
            f"Unsupported PIVOT_PARAMETER {pivot_parameter!r}. "
            f"Available parameters: {', '.join(available)}"
        )

    values = [row[canonical] for row in parameter_rows]
    if any(_is_missing(value) for value in values):
        raise ValueError(
            f"PIVOT_PARAMETER {canonical!r} contains a missing value in this sweep."
        )

    distinct_values = {_hashable_parameter_value(value) for value in values}
    if len(parameter_rows) > 1 and len(distinct_values) < 2:
        raise ValueError(
            f"PIVOT_PARAMETER {canonical!r} is constant in this sweep. "
            "Configure at least two distinct values before enabling COMPARATIVE_RUN."
        )
    return canonical


def build_comparative_run_data(
    *,
    run_name: str,
    parameters: Mapping[str, object],
    result_metrics: Mapping[str, object],
    actual_by_lead_day: np.ndarray,
    predicted_by_lead_day: np.ndarray,
    target_dates_by_lead_day: np.ndarray,
    histories_by_cluster: Mapping[int, object],
    cluster_train_labels: np.ndarray,
) -> ComparativeRunData:
    """Copy one completed run into a framework-independent comparison payload."""
    history_values: dict[int, dict[str, list[float]]] = {}
    for cluster_id, history_object in histories_by_cluster.items():
        raw_history = getattr(history_object, "history", history_object)
        if not isinstance(raw_history, Mapping):
            raise ValueError(
                f"Training history for cluster {cluster_id} is not a mapping."
            )
        history_values[int(cluster_id)] = {
            str(metric_name): np.asarray(values, dtype=float).reshape(-1).tolist()
            for metric_name, values in raw_history.items()
        }

    labels = np.asarray(cluster_train_labels, dtype=int).reshape(-1)
    cluster_counts = {
        int(cluster_id): int(np.sum(labels == cluster_id))
        for cluster_id in history_values
    }
    return ComparativeRunData(
        run_name=str(run_name),
        parameters=dict(parameters),
        result_metrics=dict(result_metrics),
        actual_by_lead_day=np.asarray(actual_by_lead_day, dtype=float).copy(),
        predicted_by_lead_day=np.asarray(predicted_by_lead_day, dtype=float).copy(),
        target_dates_by_lead_day=np.asarray(target_dates_by_lead_day).copy(),
        histories_by_cluster=history_values,
        cluster_train_counts=cluster_counts,
    )


def save_comparative_outputs(
    runs: Sequence[ComparativeRunData],
    sweep_dir: Path,
    pivot_parameter: str,
    *,
    n_timeseries_splits: int = 4,
) -> Path:
    """Save aligned predictions, histories, metrics, and comparative plots."""
    runs = list(runs)
    if n_timeseries_splits <= 0:
        raise ValueError("n_timeseries_splits must be positive.")
    run_names = [run.run_name for run in runs]
    if len(set(run_names)) != len(run_names):
        raise ValueError("Comparative run names must be unique.")

    pivot = validate_comparative_pivot(
        [run.parameters for run in runs],
        pivot_parameter,
    )
    predictions = comparative_predictions_dataframe(runs, pivot)
    aligned_predictions = align_predictions_on_common_dates(predictions)
    histories = comparative_histories_dataframe(runs, pivot)
    metrics = comparative_metrics_dataframe(aligned_predictions, pivot)
    overall_metrics = comparative_overall_metrics_dataframe(
        aligned_predictions,
        pivot,
    )
    manifest = comparative_manifest_dataframe(
        runs,
        aligned_predictions,
        histories,
        pivot,
    )

    comparison_dir = Path(sweep_dir) / "comparative_analysis"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    _clear_comparative_artifacts(comparison_dir)
    predictions.to_csv(
        comparison_dir / "test_predictions_comparison.csv",
        index=False,
    )
    aligned_predictions.to_csv(
        comparison_dir / "aligned_test_predictions.csv",
        index=False,
    )
    histories.to_csv(
        comparison_dir / "training_history_comparison.csv",
        index=False,
    )
    metrics.to_csv(
        comparison_dir / "comparative_metrics.csv",
        index=False,
    )
    manifest.to_csv(comparison_dir / "comparison_manifest.csv", index=False)

    _save_timeseries_comparison_plots(
        aligned_predictions,
        comparison_dir,
        pivot,
        n_splits=n_timeseries_splits,
    )
    _save_scatter_comparison_plots(
        aligned_predictions,
        comparison_dir,
        pivot,
    )
    _save_training_history_comparison_plot(
        histories,
        comparison_dir,
        pivot,
    )
    _save_metric_comparison_plots(metrics, comparison_dir, pivot)
    _save_overall_metric_comparison_plot(
        overall_metrics,
        comparison_dir,
        pivot,
    )
    _write_comparison_summary(
        comparison_dir,
        pivot,
        aligned_predictions,
        metrics,
        runs,
    )
    _write_report_compare(
        comparison_dir,
        Path(sweep_dir),
        pivot,
        metrics,
        runs,
        overall_metrics=overall_metrics,
    )
    return comparison_dir


def _clear_comparative_artifacts(output_dir: Path) -> None:
    """Remove only artifacts owned by this comparative-output writer."""
    owned_artifacts = {
        output_dir / artifact_name
        for artifact_name in COMPARATIVE_ARTIFACT_NAMES
    }
    for pattern in COMPARATIVE_ARTIFACT_PATTERNS:
        owned_artifacts.update(output_dir.glob(pattern))
    for artifact_path in owned_artifacts:
        if artifact_path.is_file() or artifact_path.is_symlink():
            artifact_path.unlink()


def comparative_predictions_dataframe(
    runs: Sequence[ComparativeRunData],
    pivot_parameter: str,
) -> pd.DataFrame:
    """Return tidy test predictions with an explicit target date per lead day."""
    rows: list[dict[str, object]] = []
    expected_lead_days: int | None = None
    for run in runs:
        actual = _lead_day_matrix(run.actual_by_lead_day, "actual_by_lead_day")
        predicted = _lead_day_matrix(
            run.predicted_by_lead_day,
            "predicted_by_lead_day",
        )
        dates = _date_matrix(run.target_dates_by_lead_day)
        if actual.shape != predicted.shape:
            raise ValueError(
                f"Actual and predicted matrices do not match for run {run.run_name!r}."
            )
        if dates.shape != actual.shape:
            raise ValueError(
                f"Target-date and prediction matrices do not match for run "
                f"{run.run_name!r}."
            )
        if expected_lead_days is None:
            expected_lead_days = actual.shape[1]
        elif actual.shape[1] != expected_lead_days:
            raise ValueError("All comparative runs must use the same forecast horizon.")
        if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
            raise ValueError(
                f"Comparative predictions contain non-finite values for {run.run_name!r}."
            )

        pivot_value = run.parameters[pivot_parameter]
        for lead_offset in range(actual.shape[1]):
            for sample_index in range(actual.shape[0]):
                rows.append(
                    {
                        "run_name": run.run_name,
                        "pivot_parameter": pivot_parameter,
                        "pivot_value": pivot_value,
                        "lead_day": lead_offset + 1,
                        "sample_index": sample_index,
                        "target_date": dates[sample_index, lead_offset],
                        "actual_mm": float(actual[sample_index, lead_offset]),
                        "predicted_mm": float(predicted[sample_index, lead_offset]),
                    }
                )

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise ValueError("Comparative prediction data is empty.")
    duplicate_mask = predictions.duplicated(
        ["run_name", "lead_day", "target_date"],
        keep=False,
    )
    if duplicate_mask.any():
        duplicate = predictions.loc[
            duplicate_mask,
            ["run_name", "lead_day", "target_date"],
        ].iloc[0]
        raise ValueError(
            "Duplicate target date in comparative predictions: "
            f"run={duplicate['run_name']}, lead_day={duplicate['lead_day']}, "
            f"date={duplicate['target_date']}."
        )

    for (lead_day, target_date), values in predictions.groupby(
        ["lead_day", "target_date"],
        sort=False,
    ):
        actual_values = values["actual_mm"].to_numpy(dtype=float)
        if actual_values.size > 1 and not np.allclose(
            actual_values,
            actual_values[0],
            rtol=1e-9,
            atol=1e-9,
        ):
            raise ValueError(
                "Conflicting real precipitation values for "
                f"D+{lead_day} on {pd.Timestamp(target_date).date()}."
            )
    return predictions.sort_values(
        ["lead_day", "target_date", "run_name"]
    ).reset_index(drop=True)


def align_predictions_on_common_dates(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep the intersection of target dates shared by every compared run."""
    run_count = predictions["run_name"].nunique()
    aligned_parts = []
    for lead_day, lead_values in predictions.groupby("lead_day", sort=True):
        date_counts = lead_values.groupby("target_date")["run_name"].nunique()
        common_dates = date_counts[date_counts == run_count].index
        if len(common_dates) == 0:
            raise ValueError(
                f"Comparative runs have no common target dates for D+{lead_day}."
            )
        aligned_parts.append(lead_values[lead_values["target_date"].isin(common_dates)])
    return pd.concat(aligned_parts, ignore_index=True).sort_values(
        ["lead_day", "target_date", "run_name"]
    ).reset_index(drop=True)


def comparative_histories_dataframe(
    runs: Sequence[ComparativeRunData],
    pivot_parameter: str,
) -> pd.DataFrame:
    """Return tidy per-cluster training histories for every compared test."""
    rows: list[dict[str, object]] = []
    for run in runs:
        pivot_value = run.parameters[pivot_parameter]
        for cluster_id, history in sorted(run.histories_by_cluster.items()):
            weight = int(run.cluster_train_counts.get(cluster_id, 0))
            if weight <= 0:
                weight = 1
            for history_key, values in history.items():
                split = "validation" if history_key.startswith("val_") else "train"
                metric = history_key[4:] if split == "validation" else history_key
                numeric_values = np.asarray(values, dtype=float).reshape(-1)
                for epoch, value in enumerate(numeric_values, start=1):
                    if not np.isfinite(value):
                        continue
                    rows.append(
                        {
                            "run_name": run.run_name,
                            "pivot_parameter": pivot_parameter,
                            "pivot_value": pivot_value,
                            "cluster": int(cluster_id),
                            "cluster_train_count": weight,
                            "epoch": epoch,
                            "metric": metric,
                            "split": split,
                            "value": float(value),
                        }
                    )
    history_df = pd.DataFrame(rows)
    if history_df.empty:
        raise ValueError("Comparative training-history data is empty.")
    return history_df.sort_values(
        ["metric", "run_name", "cluster", "split", "epoch"]
    ).reset_index(drop=True)


def comparative_metrics_dataframe(
    aligned_predictions: pd.DataFrame,
    pivot_parameter: str,
) -> pd.DataFrame:
    """Recalculate metrics on the common target-date interval."""
    rows = []
    for (run_name, lead_day), values in aligned_predictions.groupby(
        ["run_name", "lead_day"],
        sort=True,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = calculate_regression_metrics(
                values["actual_mm"].to_numpy(dtype=float),
                values["predicted_mm"].to_numpy(dtype=float),
            )
        rows.append(
            {
                "run_name": run_name,
                "pivot_parameter": pivot_parameter,
                "pivot_value": values["pivot_value"].iloc[0],
                "lead_day": int(lead_day),
                "n_common_test_dates": int(len(values)),
                "start_date": values["target_date"].min(),
                "end_date": values["target_date"].max(),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["lead_day", "pivot_value", "run_name"],
        key=lambda values: values.map(_pivot_sort_key)
        if values.name == "pivot_value"
        else values,
    ).reset_index(drop=True)


def comparative_overall_metrics_dataframe(
    aligned_predictions: pd.DataFrame,
    pivot_parameter: str,
) -> pd.DataFrame:
    """Recalculate metrics over every lead day in the common-date interval."""
    rows = []
    for run_name, values in aligned_predictions.groupby("run_name", sort=True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = calculate_regression_metrics(
                values["actual_mm"].to_numpy(dtype=float),
                values["predicted_mm"].to_numpy(dtype=float),
            )
        rows.append(
            {
                "run_name": run_name,
                "pivot_parameter": pivot_parameter,
                "pivot_value": values["pivot_value"].iloc[0],
                "n_common_test_points": int(len(values)),
                "forecast_days": int(values["lead_day"].nunique()),
                **metrics,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["pivot_value", "run_name"],
        key=lambda values: values.map(_pivot_sort_key)
        if values.name == "pivot_value"
        else values,
    ).reset_index(drop=True)


def comparative_manifest_dataframe(
    runs: Sequence[ComparativeRunData],
    aligned_predictions: pd.DataFrame,
    histories: pd.DataFrame,
    pivot_parameter: str,
) -> pd.DataFrame:
    """Return one traceability row per compared test."""
    rows = []
    for run in runs:
        run_predictions = aligned_predictions[
            aligned_predictions["run_name"] == run.run_name
        ]
        run_histories = histories[histories["run_name"] == run.run_name]
        row = {
            "run_name": run.run_name,
            "pivot_parameter": pivot_parameter,
            "pivot_value": run.parameters[pivot_parameter],
            "common_start_date": run_predictions["target_date"].min(),
            "common_end_date": run_predictions["target_date"].max(),
            "forecast_horizon": int(run_predictions["lead_day"].max()),
            "n_common_test_dates_per_lead": int(
                run_predictions.groupby("lead_day").size().min()
            ),
            "trained_clusters": len(run.histories_by_cluster),
            "minimum_trained_epochs": int(
                run_histories.groupby(["cluster", "metric", "split"])["epoch"]
                .max()
                .min()
            ),
            "maximum_trained_epochs": int(run_histories["epoch"].max()),
        }
        row.update(run.parameters)
        for key, value in run.result_metrics.items():
            row.setdefault(str(key), value)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "pivot_value",
        key=lambda values: values.map(_pivot_sort_key),
    ).reset_index(drop=True)


def _save_timeseries_comparison_plots(
    aligned_predictions: pd.DataFrame,
    output_dir: Path,
    pivot_parameter: str,
    *,
    n_splits: int,
) -> None:
    run_labels = _run_display_labels(aligned_predictions, pivot_parameter)
    palette = _run_palette(aligned_predictions)
    run_order = _ordered_run_names(aligned_predictions)
    prediction_linewidth, prediction_alpha = (
        _timeseries_prediction_aesthetics(len(run_order))
    )
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
                color=palette[run_name],
                linewidth=1.6,
                alpha=prediction_alpha,
            )
            for run_name in run_order
        ],
    ]
    legend_labels = [
        "Observed",
        *(run_labels[run_name] for run_name in run_order),
    ]
    legend_columns = min(len(legend_handles), 4)
    legend_rows = ceil(len(legend_handles) / legend_columns)
    legend_order = [
        row_index * legend_columns + column_index
        for column_index in range(legend_columns)
        for row_index in range(legend_rows)
        if row_index * legend_columns + column_index < len(legend_handles)
    ]

    for lead_day, lead_values in aligned_predictions.groupby("lead_day", sort=True):
        target_dates = np.array(
            sorted(lead_values["target_date"].unique()),
            dtype="datetime64[ns]",
        )
        effective_splits = min(n_splits, len(target_dates))
        date_splits = np.array_split(target_dates, effective_splits)
        n_columns = 2 if effective_splits > 1 else 1
        n_rows = ceil(effective_splits / n_columns)
        figure_height = 4.4 * n_rows + 0.8 + 0.32 * legend_rows
        with sns.axes_style(
            "whitegrid",
            rc={
                "axes.edgecolor": "#C8D0D9",
                "axes.facecolor": "#FCFCFD",
                "grid.color": "#DCE2E8",
                "grid.linewidth": 0.65,
            },
        ):
            fig, axes = plt.subplots(
                n_rows,
                n_columns,
                figsize=(16, figure_height),
                squeeze=False,
                sharey=True,
            )
            for split_index, (axis, split_dates) in enumerate(
                zip(axes.flat, date_splits),
                start=1,
            ):
                split_values = lead_values[
                    lead_values["target_date"].isin(split_dates)
                ].sort_values(["target_date", "run_name"])
                actual = (
                    split_values.groupby("target_date", as_index=False)["actual_mm"]
                    .first()
                    .sort_values("target_date")
                )
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
                    data=split_values,
                    x="target_date",
                    y="predicted_mm",
                    hue="run_name",
                    hue_order=run_order,
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
                    f"Period {split_index}/{effective_splits} | "
                    f"{start_date:%d/%m/%Y} - {end_date:%d/%m/%Y}",
                    fontsize=11,
                    fontweight="semibold",
                )
                axis.set_xlabel("Target date")
                axis.set_ylabel(
                    "Precipitation (mm)"
                    if (split_index - 1) % n_columns == 0
                    else ""
                )
                date_locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
                axis.xaxis.set_major_locator(date_locator)
                axis.xaxis.set_major_formatter(
                    mdates.ConciseDateFormatter(date_locator)
                )
                axis.tick_params(axis="x", rotation=15)
                axis.margins(x=0.01)
                axis.set_axisbelow(True)
                axis.grid(False)
                axis.yaxis.grid(True, alpha=0.55)
                sns.despine(ax=axis, top=True, right=True)

            if (
                lead_values[["actual_mm", "predicted_mm"]]
                .to_numpy(dtype=float)
                .min()
                >= 0.0
            ):
                axes.flat[0].set_ylim(bottom=0.0)
            for axis in axes.flat[len(date_splits) :]:
                axis.set_visible(False)

        fig.legend(
            [legend_handles[index] for index in legend_order],
            [legend_labels[index] for index in legend_order],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=legend_columns,
            frameon=False,
            handlelength=2.8,
            columnspacing=1.5,
            fontsize=9.5,
        )
        fig.suptitle(
            f"Test time-series comparison - lead day D+{int(lead_day)}",
            y=0.995,
            fontsize=15,
            fontweight="semibold",
        )
        top_margin_inches = 0.78 + 0.32 * legend_rows
        fig.tight_layout(
            rect=(0, 0, 1, 1 - top_margin_inches / figure_height)
        )
        fig.savefig(
            output_dir
            / f"01_test_timeseries_comparison_lead_day_{int(lead_day):02d}.png",
            bbox_inches="tight",
        )
        plt.close(fig)


def _save_scatter_comparison_plots(
    aligned_predictions: pd.DataFrame,
    output_dir: Path,
    pivot_parameter: str,
) -> None:
    run_labels = _run_display_labels(aligned_predictions, pivot_parameter)
    palette = _run_palette(aligned_predictions)
    run_order = _ordered_run_names(aligned_predictions)
    for lead_day, lead_values in aligned_predictions.groupby("lead_day", sort=True):
        n_columns = min(3, len(run_order))
        n_rows = ceil(len(run_order) / n_columns)
        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(5.3 * n_columns, 4.8 * n_rows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        plot_min = float(
            min(lead_values["actual_mm"].min(), lead_values["predicted_mm"].min())
        )
        plot_max = float(
            max(lead_values["actual_mm"].max(), lead_values["predicted_mm"].max())
        )
        padding = max((plot_max - plot_min) * 0.05, 0.5)
        limits = (max(0.0, plot_min - padding), plot_max + padding)
        metric_lookup = comparative_metrics_dataframe(
            lead_values,
            pivot_parameter,
        ).set_index("run_name")
        for axis, run_name in zip(axes.flat, run_order):
            run_values = lead_values[lead_values["run_name"] == run_name]
            axis.scatter(
                run_values["actual_mm"],
                run_values["predicted_mm"],
                s=22,
                alpha=0.55,
                color=palette[run_name],
                edgecolors="none",
                label=run_labels[run_name],
            )
            axis.plot(limits, limits, "--", color="black", linewidth=1.4, label="Ideal")
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("Actual precipitation (mm)")
            axis.set_ylabel("Predicted precipitation (mm)")
            metrics = metric_lookup.loc[run_name]
            axis.set_title(
                f"{run_labels[run_name]}\n"
                f"RMSE={metrics['RMSE']:.3f} | R2={metrics['R2']:.3f}"
            )
            axis.grid(True, alpha=0.3)
            axis.legend(loc="upper left")
        for axis in axes.flat[len(run_order) :]:
            axis.set_visible(False)
        fig.suptitle(
            f"Test Scatter Comparison - D+{int(lead_day)}",
            fontsize=14,
            y=1.01,
        )
        fig.tight_layout()
        fig.savefig(
            output_dir
            / f"02_test_scatter_comparison_lead_day_{int(lead_day):02d}.png",
            bbox_inches="tight",
        )
        plt.close(fig)


def _save_training_history_comparison_plot(
    histories: pd.DataFrame,
    output_dir: Path,
    pivot_parameter: str,
) -> None:
    aggregated = _weighted_history_dataframe(histories)
    run_labels = _run_display_labels(histories, pivot_parameter)
    palette = _run_palette(histories)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    metric_labels = {
        "loss": "Loss",
        "mse": "MSE",
        "mae": "MAE",
        "r2": "R2",
    }
    for axis, metric in zip(axes.flat, HISTORY_METRICS):
        metric_values = aggregated[aggregated["metric"] == metric]
        for run_name in _ordered_run_names(histories):
            for split, linestyle in (("train", "-"), ("validation", "--")):
                values = metric_values[
                    (metric_values["run_name"] == run_name)
                    & (metric_values["split"] == split)
                ].sort_values("epoch")
                if values.empty:
                    continue
                axis.plot(
                    values["epoch"],
                    values["value"],
                    color=palette[run_name],
                    linestyle=linestyle,
                    linewidth=2,
                    label=f"{run_labels[run_name]} - {split}",
                )
        axis.set_title(metric_labels[metric])
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric_labels[metric])
        axis.grid(True, alpha=0.3)
        if not metric_values.empty:
            axis.legend(fontsize=8)
    fig.suptitle(
        "Cluster-Weighted LSTM Training-History Comparison",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "03_training_history_comparison.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def _save_metric_comparison_plots(
    metrics: pd.DataFrame,
    output_dir: Path,
    pivot_parameter: str,
) -> None:
    for lead_day, lead_values in metrics.groupby("lead_day", sort=True):
        _save_metric_comparison_panels(
            lead_values,
            output_dir,
            pivot_parameter,
            title=(
                f"Common-Date Test Metrics vs "
                f"{PIVOT_LABELS.get(pivot_parameter, pivot_parameter.replace('_', ' ').title())} "
                f"- D+{int(lead_day)}"
            ),
            filename=(
                f"04_test_metrics_vs_{pivot_parameter}_"
                f"lead_day_{int(lead_day):02d}.png"
            ),
        )


def _save_overall_metric_comparison_plot(
    overall_metrics: pd.DataFrame,
    output_dir: Path,
    pivot_parameter: str,
) -> None:
    """Save the three-panel metric plot for the complete forecast horizon."""
    if overall_metrics.empty:
        return
    pivot_label = PIVOT_LABELS.get(
        pivot_parameter,
        pivot_parameter.replace("_", " ").title(),
    )
    _save_metric_comparison_panels(
        overall_metrics,
        output_dir,
        pivot_parameter,
        title=f"Overall Test Metrics Across Forecast Horizon vs {pivot_label}",
        filename=f"05_overall_test_metrics_vs_{pivot_parameter}.png",
    )


def _save_metric_comparison_panels(
    metrics: pd.DataFrame,
    output_dir: Path,
    pivot_parameter: str,
    *,
    title: str,
    filename: str,
) -> None:
    """Render one row of RMSE, MAE, and R2 panels for a metric dataframe."""
    pivot_label = PIVOT_LABELS.get(
        pivot_parameter,
        pivot_parameter.replace("_", " ").title(),
    )
    fig, axes = plt.subplots(
        1,
        len(COMPARATIVE_METRICS),
        figsize=(18, 5.5),
        squeeze=False,
    )
    pivot_values = sorted(metrics["pivot_value"].unique(), key=_pivot_sort_key)
    numeric_pivot = all(_is_number(value) for value in pivot_values)
    use_log_scale = (
        pivot_parameter == "learning_rate"
        and numeric_pivot
        and all(
            np.isfinite(float(value)) and float(value) > 0
            for value in pivot_values
        )
    )
    x_lookup = {
        value: float(value) if numeric_pivot else index
        for index, value in enumerate(pivot_values)
    }
    individual_x = metrics["pivot_value"].map(x_lookup).to_numpy(dtype=float)
    for axis, metric_name in zip(axes.flat, COMPARATIVE_METRICS):
        individual_y = metrics[metric_name].to_numpy(dtype=float)
        axis.scatter(
            individual_x,
            individual_y,
            s=58,
            color="#4C78A8",
            alpha=0.85,
            label="Compared test",
            zorder=3,
        )
        means = metrics.groupby("pivot_value", sort=False)[metric_name].mean()
        means = means.reindex(pivot_values)
        mean_x = np.array([x_lookup[value] for value in pivot_values], dtype=float)
        axis.plot(
            mean_x,
            means.to_numpy(dtype=float),
            color="#F58518",
            linewidth=2,
            marker="o",
            label="Mean by pivot",
            zorder=2,
        )
        finite_values = metrics[np.isfinite(metrics[metric_name])]
        if not finite_values.empty:
            best_index = (
                finite_values[metric_name].idxmax()
                if metric_name == "R2"
                else finite_values[metric_name].idxmin()
            )
            best = finite_values.loc[best_index]
            axis.scatter(
                [x_lookup[best["pivot_value"]]],
                [best[metric_name]],
                marker="*",
                s=180,
                color="#E45756",
                edgecolor="black",
                linewidth=0.6,
                label="Best",
                zorder=4,
            )
        axis.set_title(metric_name)
        axis.set_xlabel(pivot_label)
        axis.set_ylabel(_metric_axis_label(metric_name))
        if use_log_scale:
            axis.set_xscale("log")
        axis.set_xticks([x_lookup[value] for value in pivot_values])
        axis.set_xticklabels([_format_parameter_value(value) for value in pivot_values])
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / filename, bbox_inches="tight")
    plt.close(fig)


def _write_comparison_summary(
    output_dir: Path,
    pivot_parameter: str,
    aligned_predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    runs: Sequence[ComparativeRunData],
) -> None:
    other_varied_parameters = _other_varied_parameters(runs, pivot_parameter)
    lines = [
        "LSTM SWEEP COMPARATIVE ANALYSIS",
        "=" * 72,
        f"Pivot parameter: {pivot_parameter}",
        f"Compared tests: {aligned_predictions['run_name'].nunique()}",
        "Alignment: intersection of target dates shared by every test",
        "Metrics: recalculated on the common-date interval",
        (
            "Training history: cluster curves weighted by training sample count "
            "and truncated to epochs shared by every contributing cluster"
        ),
        (
            "WARNING: other parameters also vary: "
            + ", ".join(other_varied_parameters)
            + ". Metric changes cannot be attributed only to the selected pivot."
            if other_varied_parameters
            else "Other varied parameters: none"
        ),
        "",
    ]
    for lead_day, lead_metrics in metrics.groupby("lead_day", sort=True):
        start_date = pd.Timestamp(lead_metrics["start_date"].min()).date()
        end_date = pd.Timestamp(lead_metrics["end_date"].max()).date()
        lines.extend(
            [
                f"D+{int(lead_day)}",
                "-" * 72,
                f"Common target dates: {start_date} to {end_date}",
                f"Samples per test: {int(lead_metrics['n_common_test_dates'].min())}",
            ]
        )
        for metric_name in COMPARATIVE_METRICS:
            finite_values = lead_metrics[np.isfinite(lead_metrics[metric_name])]
            if finite_values.empty:
                continue
            best_index = (
                finite_values[metric_name].idxmax()
                if metric_name == "R2"
                else finite_values[metric_name].idxmin()
            )
            best = finite_values.loc[best_index]
            lines.append(
                f"Best {metric_name}: {best[metric_name]:.6g} "
                f"({pivot_parameter}={_format_parameter_value(best['pivot_value'])}, "
                f"run={best['run_name']})"
            )
        lines.append("")
    (output_dir / "comparison_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _write_report_compare(
    comparison_dir: Path,
    sweep_dir: Path,
    pivot_parameter: str,
    metrics: pd.DataFrame,
    runs: Sequence[ComparativeRunData],
    *,
    overall_metrics: pd.DataFrame | None = None,
) -> None:
    """Write a LaTeX report that gathers the comparative plots and tables."""
    (comparison_dir / REPORT_COMPARE_TEX_NAME).write_text(
        render_report_compare(
            comparison_dir,
            sweep_dir,
            pivot_parameter,
            metrics,
            runs,
            overall_metrics=overall_metrics,
        ),
        encoding="utf-8",
    )


def render_report_compare(
    comparison_dir: Path,
    sweep_dir: Path,
    pivot_parameter: str,
    metrics: pd.DataFrame,
    runs: Sequence[ComparativeRunData],
    *,
    overall_metrics: pd.DataFrame | None = None,
) -> str:
    """Return the LaTeX source for the sweep-level comparative report."""
    comparison_dir = Path(comparison_dir)
    sweep_dir = Path(sweep_dir)
    pivot_label = PIVOT_LABELS.get(
        pivot_parameter,
        pivot_parameter.replace("_", " ").title(),
    )
    lead_days = sorted(int(value) for value in metrics["lead_day"].unique())
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.7in]{geometry}",
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
        r"\setcounter{tocdepth}{2}",
        r"\begin{document}",
        rf"\title{{{_latex_escape('LSTM Sweep Comparative Report')}}}",
        r"\author{}",
        r"\date{}",
        r"\maketitle",
        r"\vspace{-2em}",
        r"\tableofcontents",
        r"\clearpage",
        r"\section{Summary}",
        _bullet_list(
            (
                ("Sweep folder", sweep_dir.name),
                ("Pivot parameter", pivot_parameter),
                ("Pivot label", pivot_label),
                ("Compared tests", len(runs)),
                ("Forecast lead days", ", ".join(f"D+{day}" for day in lead_days)),
            )
        ),
        r"\section{Cluster Step}",
    ]
    cluster_figures = _cluster_step_figures(sweep_dir, runs)
    if cluster_figures:
        for run_name, figure_path in cluster_figures:
            lines.extend(
                _figure_block(
                    comparison_dir,
                    figure_path,
                    _caption_join(run_name, _caption_from_path(figure_path)),
                )
            )
    else:
        lines.append(
            _unavailable_text("No cluster-step figures were found for this sweep.")
        )

    lines.append(r"\clearpage")
    lines.append(r"\section{Prediction Time Series}")
    for lead_day in lead_days:
        lines.append(rf"\subsection{{day {lead_day}}}")
        lines.extend(
            _figures_for_lead_day(
                comparison_dir,
                f"01_test_timeseries_comparison_lead_day_{lead_day:02d}.png",
            )
        )

    lines.append(r"\section{Prediction Scatter plot}")
    for lead_day in lead_days:
        lines.append(rf"\subsection{{day {lead_day}}}")
        lines.extend(
            _figures_for_lead_day(
                comparison_dir,
                f"02_test_scatter_comparison_lead_day_{lead_day:02d}.png",
            )
        )

    lines.append(r"\section{Training History}")
    training_history_path = comparison_dir / "03_training_history_comparison.png"
    if training_history_path.exists():
        lines.extend(_figure_block(comparison_dir, training_history_path))
    else:
        lines.append(_unavailable_text("Training-history comparison plot not found."))

    lines.append(r"\section{Test Metrics}")
    if overall_metrics is None:
        overall_metrics = _fallback_overall_metrics_dataframe(metrics)
    lines.append(_overall_metric_summary_table(overall_metrics, metrics))
    lines.append(r"\subsection{overall horizon}")
    overall_plot = comparison_dir / (
        f"05_overall_test_metrics_vs_{pivot_parameter}.png"
    )
    if overall_plot.exists():
        lines.extend(_figure_block(comparison_dir, overall_plot))
    else:
        lines.append(_unavailable_text("Overall horizon metric plot not found."))
    lines.append(_metrics_latex_table(metrics))
    lines.append(_lead_day_metric_summary_table(metrics))
    for lead_day in lead_days:
        lines.append(rf"\subsection{{day {lead_day}}}")
        lines.extend(
            _figures_for_lead_day(
                comparison_dir,
                f"04_test_metrics_vs_{pivot_parameter}_lead_day_{lead_day:02d}.png",
            )
        )

    lines.extend([r"\end{document}", ""])
    return "\n".join(line for line in lines if line is not None)


def _cluster_step_figures(
    sweep_dir: Path,
    runs: Sequence[ComparativeRunData],
) -> list[tuple[str, Path]]:
    patterns = (
        "cluster_diagnostics/05_cluster_performance.png",
        "cluster_diagnostics/06_cluster_distribution.png",
        "cluster_diagnostics/07_precipitation_distribution_by_cluster.png",
        "cluster_diagnostics/08_silhouette_analysis.png",
        "05_cluster_performance.png",
        "06_cluster_distribution.png",
        "07_precipitation_distribution_by_cluster.png",
        "08_silhouette_analysis.png",
    )
    fixed_cluster_count = _fixed_cluster_count(runs)
    shown_figure_names: set[str] = set()
    figures: list[tuple[str, Path]] = []
    for run in runs:
        run_dir = sweep_dir / run.run_name
        run_cluster_count = _run_cluster_count(run)
        seen: set[Path] = set()
        for pattern in patterns:
            for figure_path in sorted(run_dir.glob(pattern)):
                if _is_silhouette_figure(figure_path) and run_cluster_count == 1:
                    continue
                if (
                    fixed_cluster_count is not None
                    and not _is_cluster_performance_figure(figure_path)
                    and figure_path.name in shown_figure_names
                ):
                    continue
                if figure_path.is_file() and figure_path not in seen:
                    figures.append((run.run_name, figure_path))
                    seen.add(figure_path)
                    shown_figure_names.add(figure_path.name)
    return figures


def _fixed_cluster_count(runs: Sequence[ComparativeRunData]) -> int | None:
    cluster_counts = [
        cluster_count
        for run in runs
        for cluster_count in [_run_cluster_count(run)]
        if cluster_count is not None
    ]
    if cluster_counts and len(set(cluster_counts)) == 1:
        return cluster_counts[0]
    return None


def _run_cluster_count(run: ComparativeRunData) -> int | None:
    value = run.parameters.get("n_clusters")
    if _is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_silhouette_figure(figure_path: Path) -> bool:
    return figure_path.name == "08_silhouette_analysis.png"


def _is_cluster_performance_figure(figure_path: Path) -> bool:
    return figure_path.name == "05_cluster_performance.png"


def _figures_for_lead_day(
    comparison_dir: Path,
    filename: str,
) -> list[str]:
    figure_path = comparison_dir / filename
    if not figure_path.exists():
        return [_unavailable_text(f"Missing figure: {filename}.")]
    return _figure_block(comparison_dir, figure_path)


def _figure_block(
    output_dir: Path,
    figure_path: Path,
    caption: str | None = None,
) -> list[str]:
    relative_path = _latex_graphics_path(output_dir, figure_path)
    caption = caption or _caption_from_path(figure_path)
    return [
        r"\begin{figure}[H]",
        r"\centering",
        rf"\includegraphics[width=0.96\textwidth,height=0.42\textheight,keepaspectratio]{{{relative_path}}}",
        rf"\caption*{{{_latex_caption(caption)}}}",
        r"\end{figure}",
    ]


def _latex_graphics_path(output_dir: Path, figure_path: Path) -> str:
    relative_path = os.path.relpath(figure_path, output_dir).replace("\\", "/")
    return rf"\detokenize{{{relative_path}}}"


def _metrics_latex_table(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return _unavailable_text("Comparative metrics table is empty.")
    wanted = [
        column
        for column in (
            "lead_day",
            "pivot_value",
            "n_common_test_dates",
            "start_date",
            "end_date",
            "RMSE",
            "MAE",
            "R2",
        )
        if column in metrics.columns
    ]
    if not wanted:
        return _unavailable_text("Comparative metrics columns were not found.")
    table = metrics[wanted].copy()
    for column in ("start_date", "end_date"):
        if column in table.columns:
            table[column] = pd.to_datetime(table[column]).dt.strftime("%d/%m/%Y")
    alignment = "l" * len(table.columns)
    header_labels = [
        _test_metrics_column_label(column, metrics)
        for column in table.columns
    ]
    header = " & ".join(_latex_table_header(column) for column in header_labels) + r" \\"
    rows = []
    best_indices = {
        metric: _best_metric_indices(table, metric, ("lead_day",))
        for metric in COMPARATIVE_METRICS
        if metric in table.columns
    }
    previous_lead_day: object | None = None
    for _, row in table.iterrows():
        current_lead_day = row.get("lead_day")
        if (
            previous_lead_day is not None
            and current_lead_day != previous_lead_day
        ):
            rows.append(r"\midrule")
        row_values = []
        for column in table.columns:
            value = _latex_table_value(row[column])
            if column in best_indices and row.name in best_indices[column]:
                value = _latex_bold_value(value)
            row_values.append(value)
        rows.append(" & ".join(row_values) + r" \\")
        previous_lead_day = current_lead_day
    return "\n".join(
        [
            r"\begin{longtable}{" + alignment + r"}",
            r"\caption*{Common-Date Test Metrics}\\",
            r"\caption*{\textit{Bold values identify the best model for each metric.}}\\",
            r"\toprule",
            header,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{longtable}",
        ]
    )


def _overall_metric_summary_table(
    overall_metrics: pd.DataFrame,
    lead_day_metrics: pd.DataFrame,
) -> str:
    """Render one row per pivot value with metrics pooled across the horizon."""
    required = {"run_name", "pivot_value", "RMSE", "MAE", "R2"}
    if overall_metrics.empty or not required.issubset(overall_metrics.columns):
        return _unavailable_text(
            "Overall horizon metric summary requires run, pivot, RMSE, MAE, and R2."
        )
    pivot_parameter = _metrics_pivot_parameter(
        overall_metrics
        if "pivot_parameter" in overall_metrics.columns
        else lead_day_metrics
    )
    pivot_column = _pivot_table_column_label(pivot_parameter)
    table = (
        overall_metrics[["run_name", "pivot_value", "RMSE", "MAE", "R2"]]
        .drop_duplicates()
        .assign(_sort_key=lambda frame: frame["pivot_value"].map(_pivot_sort_key))
        .sort_values(["_sort_key", "run_name"])
    )
    best_indices = {
        metric: _best_metric_indices(table, metric)
        for metric in COMPARATIVE_METRICS
    }
    rows = []
    for _, row in table.iterrows():
        row_values = []
        for column in ("run_name", "pivot_value", "RMSE", "MAE", "R2"):
            value = _latex_table_value(row[column])
            if column in best_indices and row.name in best_indices[column]:
                value = _latex_bold_value(value)
            row_values.append(value)
        rows.append(
            " & ".join(row_values)
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\caption*{Overall Test Metrics Across Forecast Horizon}",
            r"\caption*{\textit{Bold values identify the best model for each metric.}}",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            " & ".join(
                _latex_table_header(value)
                for value in ("Run", pivot_column, "RMSE", "MAE", "R2")
            )
            + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def _best_metric_indices(
    frame: pd.DataFrame,
    metric: str,
    group_columns: Sequence[str] = (),
) -> set[object]:
    """Return row indices tied for best metric, optionally within groups."""
    if metric not in frame.columns:
        return set()
    groups = (
        frame.groupby(list(group_columns), dropna=False, sort=False)
        if group_columns
        else [(None, frame)]
    )
    best_indices: set[object] = set()
    for _group_key, values in groups:
        numeric_values = pd.to_numeric(values[metric], errors="coerce")
        finite_mask = np.isfinite(numeric_values.to_numpy(dtype=float))
        if not finite_mask.any():
            continue
        finite_values = numeric_values[finite_mask]
        target = (
            finite_values.max()
            if metric == "R2"
            else finite_values.min()
        )
        tied = values.index[
            finite_mask & np.isclose(numeric_values.to_numpy(dtype=float), target)
        ]
        best_indices.update(tied.tolist())
    return best_indices


def _latex_bold_value(value: str) -> str:
    return rf"\textbf{{{value}}}"


def _fallback_overall_metrics_dataframe(metrics: pd.DataFrame) -> pd.DataFrame:
    """Approximate pooled metrics for callers that only provide lead-day rows."""
    required = {"run_name", "pivot_value", "RMSE", "MAE", "R2"}
    if metrics.empty or not required.issubset(metrics.columns):
        return pd.DataFrame()
    rows = []
    for (run_name, pivot_value), values in metrics.groupby(
        ["run_name", "pivot_value"],
        sort=False,
    ):
        weights = (
            values["n_common_test_dates"].to_numpy(dtype=float)
            if "n_common_test_dates" in values
            else np.ones(len(values), dtype=float)
        )
        finite_weights = np.isfinite(weights) & (weights > 0)
        if not finite_weights.any():
            finite_weights = np.ones(len(values), dtype=bool)
            weights = np.ones(len(values), dtype=float)
        selected = values.loc[finite_weights]
        selected_weights = weights[finite_weights]
        mse_values = (
            selected["MSE"].to_numpy(dtype=float)
            if "MSE" in selected
            else selected["RMSE"].to_numpy(dtype=float) ** 2
        )
        rows.append(
            {
                "run_name": run_name,
                "pivot_parameter": _metrics_pivot_parameter(metrics),
                "pivot_value": pivot_value,
                "RMSE": float(
                    np.sqrt(np.average(mse_values, weights=selected_weights))
                ),
                "MAE": float(
                    np.average(
                        selected["MAE"].to_numpy(dtype=float),
                        weights=selected_weights,
                    )
                ),
                "R2": float(
                    np.average(
                        selected["R2"].to_numpy(dtype=float),
                        weights=selected_weights,
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _lead_day_metric_summary_table(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return _unavailable_text("Lead-day metric summary is empty.")
    required = {"run_name", "pivot_value", "lead_day", "RMSE", "MAE", "R2"}
    if not required.issubset(metrics.columns):
        return _unavailable_text(
            "Lead-day metric summary requires run_name, pivot_value, lead_day, "
            "RMSE, MAE, and R2."
        )

    pivot_parameter = _metrics_pivot_parameter(metrics)
    pivot_column = _pivot_table_column_label(pivot_parameter)
    lead_days = sorted(int(day) for day in metrics["lead_day"].dropna().unique())
    if not lead_days:
        return _unavailable_text("Lead-day metric summary has no lead days.")

    run_order = (
        metrics[["run_name", "pivot_value"]]
        .drop_duplicates()
        .assign(_sort_key=lambda frame: frame["pivot_value"].map(_pivot_sort_key))
        .sort_values(["_sort_key", "run_name"])
    )
    lookup = metrics.set_index(["run_name", "lead_day"])
    metric_names = ("RMSE", "MAE", "R2")
    best_indices = {
        metric: _best_metric_indices(metrics, metric, ("lead_day",))
        for metric in metric_names
    }
    alignment = "ll" + "r" * (len(lead_days) * len(metric_names))

    first_header = [
        "Run",
        pivot_column,
        *[
            rf"\multicolumn{{{len(metric_names)}}}{{c}}{{D+{lead_day}}}"
            for lead_day in lead_days
        ],
    ]
    second_header = [
        "",
        "",
        *[
            metric
            for _lead_day in lead_days
            for metric in metric_names
        ],
    ]
    rows = []
    for row in run_order.itertuples(index=False):
        row_values = [
            _latex_table_value(row.run_name),
            _latex_table_value(row.pivot_value),
        ]
        for lead_day in lead_days:
            if (row.run_name, lead_day) not in lookup.index:
                row_values.extend(["N/A"] * len(metric_names))
                continue
            metric_row = lookup.loc[(row.run_name, lead_day)]
            original_index = metrics[
                (metrics["run_name"] == row.run_name)
                & (metrics["lead_day"] == lead_day)
            ].index[0]
            for metric in metric_names:
                value = _latex_table_value(metric_row[metric])
                if original_index in best_indices[metric]:
                    value = _latex_bold_value(value)
                row_values.append(value)
        rows.append(" & ".join(row_values) + r" \\")

    return "\n".join(
        [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\caption*{Lead-Day Metric Summary}",
            r"\caption*{\textit{Bold values identify the best model for each metric.}}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{" + alignment + r"}",
            r"\toprule",
            " & ".join(
                _latex_table_header(value)
                if not str(value).startswith(r"\multicolumn")
                else str(value)
                for value in first_header
            )
            + r" \\",
            " & ".join(_latex_table_header(value) for value in second_header)
            + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
        ]
    )


def _test_metrics_column_label(column: str, metrics: pd.DataFrame) -> str:
    if column == "lead_day":
        return "Lead Day"
    if column == "pivot_value":
        pivot_parameter = _metrics_pivot_parameter(metrics)
        return _pivot_table_column_label(pivot_parameter)
    if column == "start_date":
        return _RawLatex(r"$t_0$")
    if column == "end_date":
        return _RawLatex(r"$t_f$")
    if column in {"RMSE", "MAE", "R2"}:
        return column
    return column.replace("_", " ").title()


def _latex_table_header(value: object) -> str:
    if isinstance(value, _RawLatex):
        return str(value)
    return _latex_escape(value)


def _metrics_pivot_parameter(metrics: pd.DataFrame) -> str:
    if "pivot_parameter" not in metrics.columns or metrics["pivot_parameter"].empty:
        return "pivot_value"
    pivot_values = metrics["pivot_parameter"].dropna()
    if pivot_values.empty:
        return "pivot_value"
    return str(pivot_values.iloc[0])


def _pivot_table_column_label(pivot_parameter: str) -> str:
    labels = {
        "n_clusters": "K",
        "lstm_units": "LSTM_UNIT",
        "lstm_units_2": "LSTM_UNITS_2",
        "learning_rate": "LEARNING_RATE",
        "dropout_rate": "DROPOUT_RATE",
        "weight_decay": "WEIGHT_DECAY",
        "window_size": "WINDOW_SIZE",
        "batch_size": "BATCH_SIZE",
        "epochs": "EPOCHS",
        "patience": "PATIENCE",
        "warm_up": "WARM_UP",
        "forecast_horizon": "FORECAST_HORIZON",
        "train_info": "TRAIN_INFO",
        "silhouette_info": "SILHOUETTE_INFO",
        "sigma": "SIGMA",
    }
    return labels.get(pivot_parameter, pivot_parameter.replace("_", " ").title())


def _bullet_list(rows: Sequence[tuple[str, object]]) -> str:
    items = "\n".join(
        rf"\item \textbf{{{_latex_escape(label)}:}} {_latex_escape(_format_report_value(value))}"
        for label, value in rows
    )
    return "\n".join(
        [
            r"\begin{itemize}",
            r"\setlength\itemsep{0.1em}",
            items,
            r"\end{itemize}",
        ]
    )


def _caption_from_path(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").title()


def _caption_join(left: str, right: str) -> str:
    return _RawLatex(
        rf"{_latex_escape(left)} \textendash{{}} {_latex_escape(right)}"
    )


class _RawLatex(str):
    """String that intentionally contains LaTeX markup."""


def _latex_table_value(value: object) -> str:
    formatted = _format_report_value(value)
    if isinstance(formatted, _RawLatex):
        return str(formatted)
    return _latex_escape(formatted)


def _latex_caption(caption: object) -> str:
    if isinstance(caption, _RawLatex):
        return str(caption)
    return _latex_escape(caption)


def _unavailable_text(message: str) -> str:
    return rf"\textit{{{_latex_escape(message)}}}"


def _format_report_value(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        pass
    if _is_number(value):
        numeric_value = float(value)
        formatted = f"{abs(numeric_value):g}" if numeric_value.is_integer() else f"{abs(numeric_value):.4g}"
        if numeric_value < 0:
            return _RawLatex(r"$-$" + formatted)
        return formatted
    return str(value)


def _latex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    text = str(value)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _weighted_history_dataframe(histories: pd.DataFrame) -> pd.DataFrame:
    rows = []
    history_columns = [
        "run_name",
        "pivot_parameter",
        "pivot_value",
        "metric",
        "split",
    ]
    for history_values, values in histories.groupby(history_columns, sort=False):
        last_epoch_by_cluster = values.groupby("cluster", sort=False)["epoch"].max()
        common_last_epoch = int(last_epoch_by_cluster.min())
        contributing_clusters = int(last_epoch_by_cluster.size)
        truncated = values[values["epoch"] <= common_last_epoch]
        for epoch, epoch_values in truncated.groupby("epoch", sort=True):
            if epoch_values["cluster"].nunique() != contributing_clusters:
                continue
            weights = epoch_values["cluster_train_count"].to_numpy(dtype=float)
            metric_values = epoch_values["value"].to_numpy(dtype=float)
            rows.append(
                {
                    **dict(zip(history_columns, history_values)),
                    "epoch": int(epoch),
                    "value": float(np.average(metric_values, weights=weights)),
                    "contributing_clusters": contributing_clusters,
                }
            )
    return pd.DataFrame(rows)


def _other_varied_parameters(
    runs: Sequence[ComparativeRunData],
    pivot_parameter: str,
) -> list[str]:
    common_parameters = set.intersection(
        *(set(run.parameters) for run in runs)
    )
    varied = []
    for parameter in sorted(common_parameters - {pivot_parameter}):
        values = {
            _hashable_parameter_value(run.parameters[parameter])
            for run in runs
            if not _is_missing(run.parameters[parameter])
        }
        if len(values) > 1:
            varied.append(parameter)
    return varied


def _lead_day_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty one- or two-dimensional array.")
    return matrix


def _date_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(
            "target_dates_by_lead_day must be a non-empty one- or "
            "two-dimensional array."
        )
    parsed = pd.to_datetime(matrix.reshape(-1), errors="coerce")
    if pd.isna(parsed).any():
        raise ValueError("Comparative target dates must contain valid dates.")
    return parsed.to_numpy(dtype="datetime64[ns]").reshape(matrix.shape)


def _run_display_labels(
    values: pd.DataFrame,
    pivot_parameter: str,
) -> dict[str, str]:
    run_values = values[["run_name", "pivot_value"]].drop_duplicates()
    pivot_counts = run_values.groupby("pivot_value")["run_name"].size().to_dict()
    labels = {}
    for row in run_values.itertuples(index=False):
        base = f"{pivot_parameter}={_format_parameter_value(row.pivot_value)}"
        labels[row.run_name] = (
            f"{base} | {row.run_name}"
            if pivot_counts[row.pivot_value] > 1
            else base
        )
    return labels


def _run_palette(values: pd.DataFrame) -> dict[str, object]:
    run_order = _ordered_run_names(values)
    colors = sns.color_palette("colorblind", n_colors=max(len(run_order), 1))
    return dict(zip(run_order, colors))


def _timeseries_prediction_aesthetics(run_count: int) -> tuple[float, float]:
    """Return line width and alpha that scale with comparison density."""
    if run_count <= 8:
        return 1.1, 0.82
    if run_count <= 16:
        return 0.95, 0.72
    return 0.8, 0.62


def _ordered_run_names(values: pd.DataFrame) -> list[str]:
    run_values = values[["run_name", "pivot_value"]].drop_duplicates()
    return (
        run_values.assign(
            _sort_key=run_values["pivot_value"].map(_pivot_sort_key)
        )
        .sort_values(["_sort_key", "run_name"])["run_name"]
        .tolist()
    )


def _metric_axis_label(metric_name: str) -> str:
    return {
        "MSE": "MSE (mm2)",
        "RMSE": "RMSE (mm)",
        "MAE": "MAE (mm)",
        "R2": "R2",
    }[metric_name]


def _format_parameter_value(value: object) -> str:
    if _is_number(value):
        return f"{float(value):g}"
    return str(value)


def _pivot_sort_key(value: object) -> tuple[int, object]:
    if _is_number(value):
        return (0, float(value))
    return (1, str(value))


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value,
        (bool, np.bool_),
    )


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _hashable_parameter_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(np.asarray(value, dtype=object).reshape(-1).tolist())
    return value
