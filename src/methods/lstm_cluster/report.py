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
    ("Model Fit", ("01_training_history_cluster_*.png",)),
    (
        "Predictions",
        (
            "02_predictions_vs_actual.png",
            "03_residuals_analysis.png",
            "04_error_by_magnitude.png",
        ),
    ),
    (
        "Prediction Time Series Splits",
        (
            "02_predictions_timeseries_split_*_of_04.png",
        ),
    ),
    (
        "Cluster Diagnostics",
        (
            "05_cluster_performance.png",
            "06_cluster_distribution.png",
            "07_precipitation_distribution_by_cluster.png",
        ),
    ),
    (
        "Input Precipitation",
        (
            "08_input_precipitation_distribution_by_cluster.png",
            "input_precipitation_distribution_by_cluster/*.png",
        ),
    ),
    (
        "Cluster Histograms",
        (
            "cluster_precipitation_histograms/*.png",
            "cluster_prediction_histograms/*.png",
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
        r"\section*{Configuration}",
        config_summary_list(config),
        r"\section*{Metrics}",
        metrics_table(output_dir / "metrics_summary.csv"),
        cluster_metrics_table(output_dir / "cluster_model_metrics.csv"),
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
    labels = (
        ("state", "State"),
        ("station_id", "Station ID"),
        ("window_size", "Window Size"),
        ("n_clusters", "Number of Clusters"),
        ("algorithm", "Algorithm"),
        ("sigma", "Sigma"),
    )
    rows = [
        (label, config_map.get(key))
        for key, label in labels
        if key in config_map
    ]
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


def metrics_table(csv_path: Path) -> str:
    """Return a compact split-level metric table when available."""
    if not csv_path.exists():
        return ""
    df = pd.read_csv(csv_path)
    wanted = [column for column in ("Split", "MSE", "RMSE", "MAE", "R2") if column in df.columns]
    return dataframe_table(df[wanted], "General Metrics") if wanted else ""


def cluster_metrics_table(csv_path: Path) -> str:
    """Return a compact cluster metric table when available."""
    if not csv_path.exists():
        return ""
    df = pd.read_csv(csv_path)
    wanted = [column for column in ("Cluster", "MSE", "RMSE", "MAE", "R2") if column in df.columns]
    return dataframe_table(df[wanted], "Cluster Metrics") if wanted else ""


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
    for pattern in patterns:
        figures.extend(sorted(output_dir.glob(pattern)))
    return [path for path in figures if path.is_file()]


def timeseries_split_figure_blocks(output_dir: Path, figures: Sequence[Path]) -> list[str]:
    """Return large time-series split figures, two per page."""
    lines: list[str] = []
    for index, figure_path in enumerate(figures, start=1):
        relative_path = figure_path.relative_to(output_dir).as_posix()
        caption = figure_path.stem.replace("_", " ").replace("-", " ").title()
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
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


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
