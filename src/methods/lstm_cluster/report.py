"""LaTeX report writer for LSTM cluster experiment folders."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


REPORT_TEX_NAME = "experiment_report.tex"

FIGURE_SECTIONS: tuple[tuple[str, Sequence[str]], ...] = (
    (
        "Cluster Diagnostics",
        (
            "cluster_diagnostics/05_cluster_performance.png",
            "cluster_diagnostics/06_cluster_distribution.png",
            "cluster_diagnostics/07_precipitation_distribution_by_cluster.png",
            "cluster_diagnostics/08_silhouette_analysis.png",
            "05_cluster_performance.png",
            "06_cluster_distribution.png",
            "07_precipitation_distribution_by_cluster.png",
            "08_silhouette_analysis.png",
        ),
    ),
    (
        "Input Precipitation",
        (
            "input_precipitation_distribution_by_cluster/08_input_precipitation_distribution_by_cluster.png",
            "08_input_precipitation_distribution_by_cluster.png",
            "input_precipitation_distribution_by_cluster/*.png",
        ),
    ),
    (
        "Forecast Horizon Diagnostics",
        (
            "forecast_horizon_diagnostics/12_prediction_error_by_lead_day.png",
            "forecast_horizon_diagnostics/13_true_vs_predicted_by_lead_day.png",
            "forecast_horizon_diagnostics/14_prediction_vs_actual_timeseries_by_lead_day.png",
        ),
    ),
    (
        "Cluster Histograms",
        (
            "cluster_precipitation_histograms/*.png",
            "cluster_prediction_histograms/*.png",
        ),
    ),
    ("Model Fit", ("model_fit/01_training_history_cluster_*.png", "01_training_history_cluster_*.png")),
    (
        "Predictions (Same-cluster Routing)",
        (
            "prediction_overview_same_cluster/02_predictions_vs_actual.png",
            "residual_diagnostics/03_residuals_analysis.png",
            "residual_diagnostics/04_error_by_magnitude.png",
            "prediction_overview/02_predictions_vs_actual.png",
            "02_predictions_vs_actual.png",
            "03_residuals_analysis.png",
            "04_error_by_magnitude.png",
        ),
    ),
    (
        "Oracle Transfer Diagnostics",
        (
            "oracle_model_selection_diagnostics/01_oracle_predictions_vs_actual.png",
            "oracle_model_selection_diagnostics/02_oracle_model_transfer_matrix.png",
            "oracle_model_selection_diagnostics/03_oracle_switch_rate_by_assigned_cluster.png",
            "oracle_model_selection_diagnostics/04_oracle_mae_by_assigned_cluster.png",
            "oracle_model_selection_diagnostics/05_oracle_error_improvement_distribution.png",
        ),
    ),
    (
        "Prediction Time Series Splits",
        (
            "prediction_timeseries_splits/lead_day_*/02_predictions_timeseries_split_*_of_04.png",
            "prediction_timeseries_splits/02_predictions_timeseries_split_*_of_04.png",
            "02_predictions_timeseries_split_*_of_04.png",
        ),
    ),
    (
        "Cluster Prediction Time Series",
        (
            "cluster_prediction_timeseries/*.png",
        ),
    ),
    (
        "Cluster Prediction Scatter",
        (
            "cluster_prediction_scatter/*.png",
        ),
    ),
)


def generate_config_report(
    output_dir: Path,
    config: object | Mapping[str, object] | None = None,
    *,
    compile_pdf: bool = True,
) -> tuple[Path, Path | None]:
    """Write a compact LaTeX report for one configuration output folder."""
    output_dir = Path(output_dir)
    tex_path = output_dir / REPORT_TEX_NAME
    tex_path.write_text(render_report(output_dir, config), encoding="utf-8")

    pdf_path = compile_tex(tex_path) if compile_pdf else None
    return tex_path, pdf_path


def render_report(
    output_dir: Path,
    config: object | Mapping[str, object] | None = None,
) -> str:
    """Return the LaTeX source for a configuration folder."""
    output_dir = Path(output_dir)
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.7in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{caption}",
        r"\usepackage{float}",
        r"\usepackage{graphicx}",
        r"\usepackage{longtable}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\graphicspath{{./}}",
        r"\captionsetup{font=small,labelfont=bf}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{5pt}",
        r"\begin{document}",
        rf"\title{{{latex_escape(report_title(output_dir, config))}}}",
        r"\author{}",
        r"\date{}",
        r"\maketitle",
        r"\vspace{-2em}",
        r"\section*{Dataset}",
        dataset_summary(config),
        r"\section*{Configuration}",
        config_summary_list(config),
        r"\section*{LSTM Configs}",
        lstm_configs_list(config),
        r"\section*{Metrics}",
        metrics_table(output_dir / "metrics_summary.csv"),
        cluster_metrics_table(output_dir / "cluster_model_metrics.csv"),
        forecast_horizon_section(output_dir),
        test_model_selection_section(output_dir),
    ]

    for section_title, patterns in FIGURE_SECTIONS:
        figures = collect_figures(output_dir, patterns)
        if figures:
            if section_title == "Prediction Time Series Splits":
                lines.append(r"\clearpage")
                lines.append(rf"\section*{{{latex_escape(section_title)}}}")
                lines.extend(timeseries_split_figure_blocks(output_dir, figures))
            else:
                lines.append(rf"\section*{{{latex_escape(section_title)}}}")
                for figure_path in figures:
                    lines.extend(figure_block(output_dir, figure_path))

    lines.extend([r"\end{document}", ""])
    return "\n".join(line for line in lines if line is not None)


def report_title(
    output_dir: Path,
    config: object | Mapping[str, object] | None = None,
) -> str:
    """Return a generic report title."""
    config_map = config_mapping(config)
    state = config_map.get("state")
    station_id = config_map.get("station_id")
    if state and station_id:
        return f"LSTM+Cluster Experiment - {state} - {station_id}"
    return f"LSTM+Cluster Experiment - {output_dir.name.replace('_', ' ')}"


def config_mapping(config: object | Mapping[str, object] | None) -> dict[str, object]:
    """Convert a config-like object to a plain dictionary."""
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if is_dataclass(config) and not isinstance(config, type):
        values = asdict(config)
    else:
        values = {
            key: getattr(config, key)
            for key in ("state", "station_id", "window_size", "n_clusters", "algorithm", "sigma")
            if hasattr(config, key)
        }
    if hasattr(config, "name"):
        values["name"] = getattr(config, "name")
    return values


def config_summary_list(config: object | Mapping[str, object] | None) -> str:
    """Return a minimal configuration summary as bullets."""
    config_map = config_mapping(config)
    leading_labels = (
        ("state", "State"),
        ("station_id", "Station ID"),
        ("window_size", "Window Size"),
        ("n_clusters", "Number of Clusters"),
        ("algorithm", "Algorithm"),
        ("sigma", "Sigma"),
    )
    trailing_labels = (
        ("forecast_horizon", "Forecast Horizon"),
        ("manual_zero_tolerance", "Manual Zero Tolerance"),
        ("test_all_models", "Test Samples on All Models"),
    )
    rows = [
        (label, config_map.get(key))
        for key, label in leading_labels
        if key in config_map
    ]
    clustering_feature_scaler = configured_scaler_summary(
    if "normalize" in config_map:
        rows.append(("Normalize", config_map.get("normalize")))
    if (
        "pca_variance_threshold" in config_map
        or "pca_for_clustering_only" in config_map
    ):
        pca_variance_threshold = config_map.get("pca_variance_threshold")
        rows.append(
            (
                "PCA Variance Threshold",
                pca_variance_threshold
                if pca_variance_threshold is not None
                else "not used",
            )
        )
        rows.append(
            (
                "PCA Mode",
                "disabled"
                if pca_variance_threshold is None
                else (
                    "clustering only"
                    if config_map.get("pca_for_clustering_only")
                    else "clustering and LSTM"
                ),
            )
        )
    covariate_scaler = configured_scaler_summary(config_map, "scaler_type")
    if covariate_scaler is not None:
        rows.append(("Covariate Scaler", covariate_scaler))
    precipitation_scaler = configured_scaler_summary(
        config_map,
        "clustering_feature_normalize",
    )
    if clustering_feature_scaler is not None:
        rows.append(("Clustering Feature Scaler", clustering_feature_scaler))
    clustering_precipitation_scaler = configured_scaler_summary(
        config_map,
        "clustering_precipitation_normalize",
    )
    if clustering_precipitation_scaler is not None:
        rows.append(
            ("Clustering Precipitation Scaler", clustering_precipitation_scaler)
        )
    lstm_feature_scaler = configured_scaler_summary(
        config_map,
        "lstm_feature_normalize",
    )
    if lstm_feature_scaler is not None:
        rows.append(("LSTM Feature Scaler", lstm_feature_scaler))
    lstm_precipitation_scaler = configured_scaler_summary(
        config_map,
        "lstm_precipitation_normalize",
    )
    if lstm_precipitation_scaler is not None:
        rows.append(("LSTM Precipitation Scaler", lstm_precipitation_scaler))
    if "target_scale" in config_map:
        rows.append(("LSTM Target Scale", config_map.get("target_scale")))
    rows.extend(
        (label, config_map.get(key))
        for key, label in trailing_labels
        if key in config_map
    )
    if not rows:
        return ""

    items = "\n".join(
        rf"\item \textbf{{{latex_escape(label)}:}} {latex_escape(format_value(value))}"
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


def configured_scaler_summary(
    config_map: Mapping[str, object],
    key: str,
) -> str | None:
    """Return a configured scaler name, honoring disabled normalization."""
    if key not in config_map:
        return None

    scaler_type = config_map.get(key)
    if scaler_type is None or str(scaler_type).strip().lower() in {"", "none"}:
        return "none"
    return str(scaler_type)


def lstm_configs_list(config: object | Mapping[str, object] | None) -> str:
    """Return LSTM architecture and training hyperparameters as bullets."""
    config_map = config_mapping(config)
    layer_rows = [
        ("LSTM layer 1 units", config_map.get("lstm_units")),
        ("LSTM layer 2 units", config_map.get("lstm_units_2")),
        ("Dense layer units", config_map.get("dense_units")),
        ("Output units", config_map.get("output_units")),
    ]
    hyperparameter_rows = [
        ("Dropout rate", config_map.get("dropout_rate")),
        ("Learning rate", config_map.get("learning_rate")),
        ("Weight decay", config_map.get("weight_decay")),
        ("Epochs", config_map.get("epochs")),
        ("Batch size", config_map.get("batch_size")),
        ("Early stopping", config_map.get("early_stopping")),
        ("Patience", config_map.get("patience")),
        ("Optimizer", config_map.get("optimizer")),
        ("Loss", config_map.get("loss")),
        ("Loss quantiles", config_map.get("loss_quantiles")),
        ("Loss quantile weights", config_map.get("loss_quantile_weights")),
        ("Metrics", config_map.get("metrics")),
    ]
    rows = [
        *[(label, value) for label, value in layer_rows if value is not None],
        *[(label, value) for label, value in hyperparameter_rows if value is not None],
    ]
    if not rows:
        return unavailable_text("LSTM configuration was not provided.")

    return bullet_list(rows)


def dataset_summary(config: object | Mapping[str, object] | None) -> str:
    """Return dataset feature, date range, and split details."""
    config_map = config_mapping(config)
    rows = []
    if config_map.get("dataset_start_date") is not None:
        rows.append(("Start date", config_map["dataset_start_date"]))
    if config_map.get("dataset_end_date") is not None:
        rows.append(("End date", config_map["dataset_end_date"]))
    if config_map.get("features") is not None:
        rows.append(("Features", config_map["features"]))
    if config_map.get("n_samples") is not None:
        rows.append(("Input samples", config_map["n_samples"]))

    splits = config_map.get("splits")
    if isinstance(splits, Mapping):
        for split_name, split_values in splits.items():
            if isinstance(split_values, Mapping):
                samples = split_values.get("samples")
                percent = split_values.get("percent")
                rows.append((f"{split_name} split", format_split(samples, percent)))
            else:
                rows.append((f"{split_name} split", split_values))

    if not rows:
        return unavailable_text("Dataset metadata was not provided.")

    return bullet_list(rows)


def bullet_list(rows: Sequence[tuple[str, object]]) -> str:
    """Return labeled rows as a compact LaTeX bullet list."""
    items = "\n".join(
        rf"\item \textbf{{{latex_escape(label)}:}} {latex_escape(format_value(value))}"
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


def unavailable_text(message: str) -> str:
    """Return a short LaTeX placeholder for missing report context."""
    return rf"\textit{{{latex_escape(message)}}}"


def metrics_table(csv_path: Path) -> str:
    """Return a compact split-level metric table when available."""
    if not csv_path.exists():
        return ""
    df = pd.read_csv(csv_path)
    wanted = [column for column in ("split", "MSE", "RMSE", "MAE", "R2") if column in df.columns]
    return dataframe_table(df[wanted], "General Metrics") if wanted else ""


def cluster_metrics_table(csv_path: Path) -> str:
    """Return a compact cluster metric table when available."""
    if not csv_path.exists():
        return ""
    df = pd.read_csv(csv_path)
    wanted = [column for column in ("cluster", "MSE", "RMSE", "MAE", "R2") if column in df.columns]
    return dataframe_table(df[wanted], "Cluster Metrics") if wanted else ""


def test_model_selection_section(output_dir: Path) -> str:
    """Return the optional oracle diagnostic for cross-cluster transfer."""
    summary_path = output_dir / "test_model_metric_summary.csv"
    if not summary_path.exists():
        return ""

    df = pd.read_csv(summary_path)
    if df.empty:
        return ""

    rows = []
    for _, row in df.sort_values("metric_selection").iterrows():
        rows.append(
            {
                "Selection": row["metric_selection"],
                "Changed Samples": (
                    f"{int(row['switched_samples'])}/{int(row['n_test'])} "
                    f"({row['switched_samples_percent']:.1f}%)"
                ),
                "RMSE Change": (
                    f"{row['rmse_improvement']:.4g} "
                    f"({row['rmse_improvement_percent']:.2f}%)"
                ),
                "MAE Change": (
                    f"{row['mae_improvement']:.4g} "
                    f"({row['mae_improvement_percent']:.2f}%)"
                ),
                "R2 Change": f"{row['r2_improvement']:.4g}",
            }
        )

    paragraph = (
        "The main test metrics, predictions, and plots use only the LSTM trained "
        "for the cluster assigned to each window. Separately, this oracle "
        "diagnostic evaluates every test window with every trained cluster model "
        "and selects the winner after observing the true test target. It therefore "
        "does not estimate future performance, but shows whether strong candidate "
        "predictions existed and routing between clusters may be the limiting step. "
        "Positive error changes mean lower error, except for R2 where positive "
        "means higher R2."
    )
    lines = [
        r"\section*{Análise de transferência entre clusters}",
        latex_escape(paragraph),
        dataframe_table(pd.DataFrame(rows), "Oracle vs. Same-cluster Summary"),
    ]
    matrix_path = output_dir / "oracle_model_selection_matrix.csv"
    if matrix_path.exists():
        transfer_matrix = pd.read_csv(matrix_path)
        if not transfer_matrix.empty:
            lines.append(
                dataframe_table(
                    transfer_matrix,
                    "Oracle Transfer Matrix (rows: assigned cluster; columns: selected LSTM)",
                )
            )
    cluster_summary_path = output_dir / "oracle_cluster_routing_summary.csv"
    if cluster_summary_path.exists():
        cluster_summary = pd.read_csv(cluster_summary_path)
        wanted = [
            column
            for column in (
                "assigned_test_cluster",
                "n_test",
                "oracle_switched_percent",
                "same_cluster_mae",
                "oracle_mae",
                "mae_improvement",
                "rmse_improvement",
            )
            if column in cluster_summary.columns
        ]
        if wanted and not cluster_summary.empty:
            lines.append(
                dataframe_table(
                    cluster_summary[wanted],
                    "Oracle Routing Summary by Assigned Cluster",
                )
            )
    pair_summary_path = output_dir / "oracle_cluster_pair_summary.csv"
    if pair_summary_path.exists():
        pair_summary = pd.read_csv(pair_summary_path)
        wanted = [
            column
            for column in (
                "assigned_test_cluster",
                "oracle_selected_model_cluster",
                "n_test",
                "percent_of_assigned_cluster",
                "mae_improvement",
                "mean_squared_error_improvement",
            )
            if column in pair_summary.columns
        ]
        if wanted and not pair_summary.empty:
            lines.append(
                dataframe_table(
                    pair_summary[wanted],
                    "Oracle Routing Summary by Transfer Pair",
                )
            )
    return "\n".join(lines)


def forecast_horizon_section(output_dir: Path) -> str:
    """Return the generated forecast-horizon behavior report when available."""
    report_path = (
        output_dir
        / "forecast_horizon_diagnostics"
        / "forecast_horizon_behavior_report.txt"
    )
    if not report_path.exists():
        return ""

    content = report_path.read_text(encoding="utf-8").strip()
    if not content:
        return ""

    return "\n".join(
        [
            r"\section*{Forecast Horizon Behavior}",
            r"\begin{verbatim}",
            content,
            r"\end{verbatim}",
        ]
    )


def dataframe_table(df: pd.DataFrame, caption: str) -> str:
    """Return a small LaTeX table from a dataframe."""
    alignment = "l" + "r" * max(len(df.columns) - 1, 0)
    header = " & ".join(latex_escape(str(column)) for column in df.columns) + r" \\"
    rows = []
    for _, row in df.iterrows():
        rows.append(
            " & ".join(latex_escape(format_value(row[column])) for column in df.columns)
            + r" \\"
        )

    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            rf"\caption*{{{latex_escape(caption)}}}",
            rf"\begin{{tabular}}{{{alignment}}}",
            r"\toprule",
            header,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def collect_figures(output_dir: Path, patterns: Sequence[str]) -> list[Path]:
    """Collect existing figures for a report section."""
    figures: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(output_dir.glob(pattern)):
            if path.is_file() and path not in seen:
                figures.append(path)
                seen.add(path)
    return figures


def timeseries_split_figure_blocks(output_dir: Path, figures: Sequence[Path]) -> list[str]:
    """Return large time-series split figures, two per page."""
    lines: list[str] = []
    for index, figure_path in enumerate(figures, start=1):
        relative_path = figure_path.relative_to(output_dir).as_posix()
        caption = figure_path.stem.replace("_", " ").replace("-", " ").title()
        lead_day = figure_path.parent.name.removeprefix("lead_day_")
        if lead_day != figure_path.parent.name and lead_day.isdigit():
            caption = f"D+{int(lead_day)} - {caption}"
        lines.extend(
            [
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.98\textwidth,height=0.38\textheight,keepaspectratio]{{{relative_path}}}",
                rf"\caption*{{{latex_escape(caption)}}}",
                r"\end{figure}",
            ]
        )
        if index % 2 == 0 and index < len(figures):
            lines.append(r"\clearpage")
    return lines

def figure_block(output_dir: Path, figure_path: Path) -> list[str]:
    """Return a LaTeX figure block for one image."""
    relative_path = figure_path.relative_to(output_dir).as_posix()
    caption = figure_path.stem.replace("_", " ").replace("-", " ").title()
    return [
        r"\begin{figure}[H]",
        r"\centering",
        rf"\includegraphics[width=0.94\textwidth]{{{relative_path}}}",
        rf"\caption*{{{latex_escape(caption)}}}",
        r"\end{figure}",
    ]


def compile_tex(tex_path: Path) -> Path | None:
    """Compile a LaTeX file with the first working local compiler."""
    tex_path = Path(tex_path).resolve()
    compiler_runs = available_compiler_runs()
    tex_env = tex_compile_environment(tex_path)
    bootstrap_miktex(tex_env, tex_path.parent)
    if not compiler_runs:
        return None

    failures: list[str] = []
    for compiler_name, commands in compiler_runs:
        cleanup_latex_files(tex_path)
        results = [
            subprocess.run(
                [*command, tex_path.name],
                cwd=tex_path.parent,
                check=False,
                capture_output=True,
                text=True,
                env=tex_env,
            )
            for command in commands
        ]
        if all(result.returncode == 0 for result in results):
            pdf_path = tex_path.with_suffix(".pdf")
            if pdf_path.exists():
                cleanup_latex_files(tex_path)
                return pdf_path

        failures.append(format_compile_failure(compiler_name, results))

    log_path = tex_path.with_name("experiment_report_compile.log")
    log_path.write_text("\n\n".join(failures), encoding="utf-8")
    return None


def tex_compile_environment(tex_path: Path) -> dict[str, str]:
    """Return an environment with writable MiKTeX user state."""
    env = os.environ.copy()
    state_root = tex_path.parent.parent / ".miktex"
    user_config = state_root / "config"
    user_data = state_root / "data"
    user_install = state_root / "install"
    for directory in (user_config, user_data, user_install):
        directory.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "MIKTEX_USERCONFIG": str(user_config),
            "MIKTEX_USERDATA": str(user_data),
            "MIKTEX_USERINSTALL": str(user_install),
        }
    )
    return env


def bootstrap_miktex(env: Mapping[str, str], cwd: Path) -> None:
    """Initialize a local MiKTeX package database when MiKTeX is available."""
    mpm = shutil.which("mpm")
    initexmf = shutil.which("initexmf")
    if not mpm or not initexmf:
        return

    manifest = Path(env["MIKTEX_USERINSTALL"]) / "miktex" / "config" / "package-manifests.ini"
    if manifest.exists():
        return

    for command in ([mpm, "--update-db"], [initexmf, "--update-fndb"]):
        subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            env=dict(env),
        )
def available_compiler_runs() -> list[tuple[str, list[list[str]]]]:
    """Return local LaTeX compilers in preferred order."""
    compilers: list[tuple[str, list[list[str]]]] = []
    if latexmk := shutil.which("latexmk"):
        compilers.append(
            (
                "latexmk",
                [[latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error"]],
            )
        )

    for engine_name in ("pdflatex", "xelatex", "lualatex"):
        if engine := shutil.which(engine_name):
            command = [engine, "-interaction=nonstopmode", "-halt-on-error"]
            compilers.append((engine_name, [command, command]))

    return compilers


def format_compile_failure(
    compiler_name: str,
    results: Sequence[subprocess.CompletedProcess[str]],
) -> str:
    """Return readable compiler output for troubleshooting."""
    lines = [f"Compiler failed: {compiler_name}"]
    for index, result in enumerate(results, start=1):
        lines.append(f"Run {index}, exit code {result.returncode}")
        lines.append(result.stdout)
        lines.append(result.stderr)
    return "\n".join(lines)


def cleanup_latex_files(tex_path: Path) -> None:
    """Remove auxiliary files after a successful compile."""
    for suffix in (".aux", ".fls", ".fdb_latexmk", ".log", ".out"):
        aux_path = tex_path.with_suffix(suffix)
        if aux_path.exists():
            aux_path.unlink()


def format_value(value: object) -> str:
    """Format values for compact report tables."""
    if value is None:
        return "N/A"
    if isinstance(value, Mapping):
        return ", ".join(f"{key}: {format_value(item)}" for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, str):
        return ", ".join(format_value(item) for item in value)
    if pd.isna(value):
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def format_split(samples: object, percent: object) -> str:
    """Format split sample counts and percentages."""
    if samples is None:
        return "N/A"
    if isinstance(percent, float):
        return f"{samples} samples ({percent:.1%})"
    return f"{samples} samples"


def latex_escape(value: object) -> str:
    """Escape text for LaTeX."""
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


def main() -> None:
    """Generate reports for one or more existing configuration folders."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dirs", nargs="+", type=Path)
    parser.add_argument("--no-pdf", action="store_true", help="Only write the .tex file.")
    args = parser.parse_args()

    for output_dir in args.output_dirs:
        tex_path, pdf_path = generate_config_report(output_dir, compile_pdf=not args.no_pdf)
        print(f"Wrote {tex_path}")
        if pdf_path:
            print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
