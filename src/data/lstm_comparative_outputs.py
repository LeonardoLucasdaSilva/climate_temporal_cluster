"""Sweep-level comparative outputs for LSTM clustering experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
import re
from typing import Mapping, Sequence
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
    "forecast_horizon": "Forecast horizon",
}

COMPARATIVE_METRICS = ("MSE", "RMSE", "MAE", "R2")
HISTORY_METRICS = ("loss", "mse", "mae", "r2")
COMPARATIVE_ARTIFACT_NAMES = (
    "test_predictions_comparison.csv",
    "aligned_test_predictions.csv",
    "training_history_comparison.csv",
    "comparative_metrics.csv",
    "comparison_manifest.csv",
    "comparison_summary.txt",
    "03_training_history_comparison.png",
)
COMPARATIVE_ARTIFACT_PATTERNS = (
    "01_test_timeseries_comparison_lead_day_*.png",
    "02_test_scatter_comparison_lead_day_*.png",
    "04_test_metrics_vs_*_lead_day_*.png",
)


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
    _write_comparison_summary(
        comparison_dir,
        pivot,
        aligned_predictions,
        metrics,
        runs,
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
    for lead_day, lead_values in aligned_predictions.groupby("lead_day", sort=True):
        target_dates = np.array(
            sorted(lead_values["target_date"].unique()),
            dtype="datetime64[ns]",
        )
        effective_splits = min(n_splits, len(target_dates))
        date_splits = np.array_split(target_dates, effective_splits)
        n_columns = 2 if effective_splits > 1 else 1
        n_rows = ceil(effective_splits / n_columns)
        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(15, 4.6 * n_rows),
            squeeze=False,
            sharey=True,
        )
        for split_index, (axis, split_dates) in enumerate(
            zip(axes.flat, date_splits),
            start=1,
        ):
            split_values = lead_values[
                lead_values["target_date"].isin(split_dates)
            ]
            actual = (
                split_values.groupby("target_date", as_index=False)["actual_mm"]
                .first()
                .sort_values("target_date")
            )
            axis.plot(
                actual["target_date"],
                actual["actual_mm"],
                color="black",
                linewidth=2.2,
                label="Actual",
                zorder=5,
            )
            for run_name, run_values in split_values.groupby("run_name", sort=False):
                run_values = run_values.sort_values("target_date")
                axis.plot(
                    run_values["target_date"],
                    run_values["predicted_mm"],
                    linewidth=1.5,
                    alpha=0.9,
                    color=palette[run_name],
                    label=run_labels[run_name],
                )
            axis.set_title(
                f"Common test period - part {split_index} of {effective_splits}"
            )
            axis.set_xlabel("Target date")
            axis.set_ylabel("Precipitation (mm)")
            axis.xaxis.set_major_locator(mdates.AutoDateLocator())
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
            axis.tick_params(axis="x", rotation=30)
            axis.grid(True, alpha=0.3)
        for axis in axes.flat[len(date_splits) :]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=min(len(labels), 4),
        )
        fig.suptitle(
            f"Test Time-Series Comparison - D+{int(lead_day)}",
            y=1.02,
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
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
    pivot_label = PIVOT_LABELS.get(
        pivot_parameter,
        pivot_parameter.replace("_", " ").title(),
    )
    for lead_day, lead_values in metrics.groupby("lead_day", sort=True):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)
        pivot_values = sorted(
            lead_values["pivot_value"].unique(),
            key=_pivot_sort_key,
        )
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
        individual_x = lead_values["pivot_value"].map(x_lookup).to_numpy(dtype=float)
        for axis, metric_name in zip(axes.flat, COMPARATIVE_METRICS):
            individual_y = lead_values[metric_name].to_numpy(dtype=float)
            axis.scatter(
                individual_x,
                individual_y,
                s=58,
                color="#4C78A8",
                alpha=0.85,
                label="Compared test",
                zorder=3,
            )
            means = lead_values.groupby("pivot_value", sort=False)[metric_name].mean()
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
            finite_values = lead_values[np.isfinite(lead_values[metric_name])]
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
        fig.suptitle(
            f"Common-Date Test Metrics vs {pivot_label} - D+{int(lead_day)}",
            fontsize=14,
            y=1.01,
        )
        fig.tight_layout()
        fig.savefig(
            output_dir
            / (
                f"04_test_metrics_vs_{pivot_parameter}_"
                f"lead_day_{int(lead_day):02d}.png"
            ),
            bbox_inches="tight",
        )
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
