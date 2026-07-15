"""Beamer report helpers for saved LSTM-cluster experiment plots."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
from typing import Sequence


IMAGE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
EXCLUDED_IMAGE_NAMES = {"beamer.pdf", "experiment_report.pdf"}

SECTION_RULES: tuple[tuple[str, str], ...] = (
    ("model_fit/", "Model Training"),
    ("prediction_overview/", "Prediction Overview"),
    ("cluster_prediction_scatter/", "Cluster Test Scatter"),
    ("prediction_timeseries_splits/", "Test Time Series"),
    ("cluster_prediction_timeseries/", "Cluster Time Series"),
    ("residual_diagnostics/", "Residual Diagnostics"),
    ("cluster_diagnostics/", "Cluster Diagnostics"),
    ("cluster_prediction_histograms/", "Cluster Distributions"),
    ("cluster_precipitation_histograms/", "Cluster Distributions"),
    ("input_precipitation_distribution_by_cluster/", "Input Horizon Distributions"),
    (
        "forecast_horizon_diagnostics/true_vs_predicted_by_lead_day/",
        "Lead-Day Diagnostics",
    ),
    ("forecast_horizon_diagnostics/", "Forecast Horizon Diagnostics"),
)

SECTION_DESCRIPTIONS = {
    "Model Training": "Training curves and convergence behavior for cluster models.",
    "Prediction Overview": "Global test prediction quality and actual-versus-predicted behavior.",
    "Cluster Test Scatter": "Per-cluster test scatter plots comparing actual and predicted rainfall.",
    "Test Time Series": "Chronological prediction behavior on test samples.",
    "Cluster Time Series": "Per-cluster test trajectories, predictions, and residuals.",
    "Residual Diagnostics": "Residual structure and error magnitude diagnostics.",
    "Cluster Diagnostics": "Cluster-level metrics, balance, and rainfall profile.",
    "Cluster Distributions": "Actual, predicted, residual, and rainfall distributions by cluster.",
    "Input Horizon Distributions": "Input-window rainfall and forecast-horizon target distributions.",
    "Forecast Horizon Diagnostics": "Current value, forecast target, and horizon-specific behavior.",
    "Lead-Day Diagnostics": "How one prediction compares against D+1 through the configured horizon.",
    "Other Selected Plots": "Selected figures that do not match a known experiment plot group.",
}


@dataclass(frozen=True)
class PlotSlide:
    """One selected plot and its Beamer metadata."""

    path: Path
    relative_to_run: str
    section: str
    title: str


def discover_plots(run_dir: Path) -> list[Path]:
    """Return image-like plot files under one experiment configuration folder."""
    run_dir = Path(run_dir)
    return sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.name.lower() not in EXCLUDED_IMAGE_NAMES
        and ".miktex" not in path.parts
    )


def resolve_selected_plots(run_dir: Path, selectors: Sequence[str] | None) -> list[Path]:
    """Resolve selector strings into existing plots under `run_dir`.

    Selectors can be exact relative paths, glob patterns, absolute paths inside
    the run directory, or case-insensitive substrings of a plot path.
    """
    run_dir = Path(run_dir).resolve()
    all_plots = discover_plots(run_dir)
    if not selectors:
        return all_plots

    selected: list[Path] = []
    seen: set[Path] = set()
    all_by_relative = {
        _relative_posix(path, run_dir).lower(): path
        for path in all_plots
    }

    for selector in selectors:
        matches = _matches_selector(run_dir, all_plots, all_by_relative, selector)
        if not matches:
            raise ValueError(f"No plots matched selector: {selector}")
        for match in sorted(matches):
            resolved = match.resolve()
            if resolved not in seen:
                selected.append(match)
                seen.add(resolved)
    return selected


def build_slides(run_dir: Path, plots: Sequence[Path]) -> list[PlotSlide]:
    """Attach section and title metadata to selected plot paths."""
    run_dir = Path(run_dir)
    slides = []
    for path in plots:
        relative = _relative_posix(path, run_dir)
        section = section_for_plot(relative)
        slides.append(
            PlotSlide(
                path=Path(path),
                relative_to_run=relative,
                section=section,
                title=plot_title(relative),
            )
        )
    return slides


def group_slides(slides: Sequence[PlotSlide]) -> OrderedDict[str, list[PlotSlide]]:
    """Group slides by analysis section in the standard model-review order."""
    sections = [section for _prefix, section in SECTION_RULES]
    ordered_names = [*dict.fromkeys(sections), "Other Selected Plots"]
    grouped: OrderedDict[str, list[PlotSlide]] = OrderedDict()
    for section in ordered_names:
        section_slides = [slide for slide in slides if slide.section == section]
        if section_slides:
            grouped[section] = section_slides
    return grouped


def render_beamer(
    run_dir: Path,
    plots: Sequence[Path],
    title: str | None = None,
    tex_dir: Path | None = None,
) -> str:
    """Render a complete Beamer document for selected experiment plots."""
    run_dir = Path(run_dir)
    tex_dir = Path(tex_dir) if tex_dir is not None else run_dir
    slides = build_slides(run_dir, plots)
    grouped = group_slides(slides)
    presentation_title = title or f"Experiment Run: {run_dir.name}"

    lines = [
        r"\documentclass[aspectratio=169,11pt]{beamer}",
        r"\usetheme{Madrid}",
        r"\usecolortheme{dove}",
        r"\setbeamertemplate{navigation symbols}{}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{graphicx}",
        r"\usepackage{hyperref}",
        r"\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}",
        rf"\title{{{latex_escape(presentation_title)}}}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\begin{frame}",
        r"\titlepage",
        r"\end{frame}",
        *_overview_frame(grouped),
    ]

    if not grouped:
        lines.extend(
            [
                r"\begin{frame}{No Plots Selected}",
                r"No matching plots were found for this experiment run.",
                r"\end{frame}",
                r"\end{document}",
                "",
            ]
        )
        return "\n".join(lines)

    for section, section_slides in grouped.items():
        lines.extend(_section_intro_frame(section, len(section_slides)))
        for slide in section_slides:
            lines.extend(_plot_frame(slide, tex_dir))

    lines.extend([r"\end{document}", ""])
    return "\n".join(lines)


def write_beamer(
    run_dir: Path,
    plots: Sequence[Path],
    output_path: Path | None = None,
    title: str | None = None,
) -> Path:
    """Write `beamer.tex` for a saved experiment run and return its path."""
    run_dir = Path(run_dir).resolve()
    output_path = Path(output_path) if output_path is not None else run_dir / "beamer.tex"
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_beamer(run_dir, plots, title=title, tex_dir=output_path.parent),
        encoding="utf-8",
    )
    return output_path


def compile_beamer_pdf(tex_path: Path, runs: int = 2) -> Path:
    """Compile a Beamer `.tex` file with pdflatex and return the PDF path."""
    tex_path = Path(tex_path).resolve()
    if not tex_path.exists():
        raise FileNotFoundError(f"Beamer TeX file not found: {tex_path}")
    if runs < 1:
        raise ValueError("runs must be at least 1.")

    log_path = tex_path.with_name(f"{tex_path.stem}_compile.log")
    combined_output: list[str] = []
    for run_index in range(1, runs + 1):
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name,
            ],
            cwd=tex_path.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        combined_output.extend(
            [
                f"=== pdflatex run {run_index} ===",
                result.stdout,
                result.stderr,
            ]
        )
        if result.returncode != 0:
            log_path.write_text("\n".join(combined_output), encoding="utf-8")
            raise RuntimeError(
                f"pdflatex failed for {tex_path}. See {log_path}."
            )

    log_path.write_text("\n".join(combined_output), encoding="utf-8")
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise RuntimeError(f"pdflatex finished but did not create {pdf_path}.")
    return pdf_path


def section_for_plot(relative_path: str) -> str:
    """Return the analysis section for a run-relative plot path."""
    normalized = relative_path.replace("\\", "/").lower()
    for prefix, section in SECTION_RULES:
        if normalized.startswith(prefix):
            return section
    return "Other Selected Plots"


def plot_title(relative_path: str) -> str:
    """Return a compact descriptive frame title for a plot path."""
    normalized = relative_path.replace("\\", "/")
    stem = Path(normalized).stem
    cluster_match = re.search(r"cluster[_-](\d+)", stem, flags=re.IGNORECASE)
    cluster_prefix = f"Cluster {cluster_match.group(1)} - " if cluster_match else ""
    lead_match = re.search(r"lead_day[_-](\d+)", normalized, flags=re.IGNORECASE)
    lead_prefix = f"D+{int(lead_match.group(1))} - " if lead_match else ""
    lower = normalized.lower()

    known_titles = (
        ("02_predictions_vs_actual", "Predicted vs Actual on Test Set"),
        ("03_residuals_analysis", "Residual Analysis on Test Set"),
        ("04_error_by_magnitude", "Error by Rainfall Magnitude"),
        ("05_cluster_performance", "Test Performance by Cluster"),
        ("06_cluster_distribution", "Cluster Distribution"),
        ("07_precipitation_distribution_by_cluster", "Rainfall Distribution by Cluster"),
        ("08_input_precipitation_distribution_by_cluster", "Input Rainfall by Cluster"),
        ("12_prediction_error_by_lead_day", "Prediction Error by Lead Day"),
        ("13_true_vs_predicted_by_lead_day", "True vs Predicted by Lead Day"),
        (
            "14_prediction_vs_actual_timeseries_by_lead_day",
            "Prediction vs Actual Time Series by Lead Day",
        ),
        ("predicted_vs_actual_scatter", "Test Actual vs Predicted Scatter"),
        ("prediction_timeseries", "Test Prediction Time Series"),
        ("prediction_histograms", "Prediction and Residual Histograms"),
        ("precipitation_histogram", "Rainfall Histogram"),
        ("training_history", "Training History"),
        ("true_vs_predicted_lead_day", "True vs Predicted for Lead Day"),
    )
    for token, title in known_titles:
        if token in lower:
            return f"{cluster_prefix}{lead_prefix}{title}"
    return f"{cluster_prefix}{lead_prefix}{_humanize(stem)}"


def latex_escape(value: object) -> str:
    """Escape common LaTeX special characters."""
    text = "" if value is None else str(value)
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
    return "".join(replacements.get(char, char) for char in text)


def _matches_selector(
    run_dir: Path,
    all_plots: Sequence[Path],
    all_by_relative: dict[str, Path],
    selector: str,
) -> list[Path]:
    selector = selector.strip()
    normalized = selector.replace("\\", "/")
    selector_path = Path(selector)

    if selector_path.is_absolute():
        try:
            resolved = selector_path.resolve()
            resolved.relative_to(run_dir)
        except ValueError:
            return []
        return [resolved] if resolved.exists() and resolved.is_file() else []

    if _has_glob(normalized):
        return [
            path
            for path in run_dir.glob(normalized)
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and path.name.lower() not in EXCLUDED_IMAGE_NAMES
        ]

    exact = all_by_relative.get(normalized.lower())
    if exact is not None:
        return [exact]

    return [
        path
        for path in all_plots
        if normalized.lower() in _relative_posix(path, run_dir).lower()
    ]


def _overview_frame(grouped: OrderedDict[str, list[PlotSlide]]) -> list[str]:
    lines = [r"\begin{frame}{Overview}", r"\small"]
    if not grouped:
        lines.append("No plot sections were selected.")
    else:
        lines.append(r"\begin{itemize}")
        for section, slides in grouped.items():
            target = _section_target(section)
            label = latex_escape(section)
            count = len(slides)
            lines.append(
                rf"\item \hyperlink{{{target}}}{{\beamergotobutton{{{label}}}}} "
                rf"\hfill {count} plot(s)"
            )
        lines.append(r"\end{itemize}")
    lines.extend([r"\end{frame}"])
    return lines


def _section_intro_frame(section: str, count: int) -> list[str]:
    description = SECTION_DESCRIPTIONS.get(section, "")
    target = _section_target(section)
    return [
        rf"\section{{{latex_escape(section)}}}",
        rf"\begin{{frame}}{{{latex_escape(section)}}}",
        rf"\hypertarget{{{target}}}{{}}",
        rf"\large {latex_escape(description)}",
        "",
        rf"\vfill\small {count} selected plot(s)",
        r"\end{frame}",
    ]


def _plot_frame(slide: PlotSlide, tex_dir: Path) -> list[str]:
    include_path = _tex_relative_path(slide.path, tex_dir)
    return [
        rf"\begin{{frame}}{{{latex_escape(slide.title)}}}",
        r"\centering",
        rf"\includegraphics[width=\textwidth,height=0.78\textheight,keepaspectratio]{{\detokenize{{{include_path}}}}}",
        r"\vfill",
        rf"{{\scriptsize\texttt{{\detokenize{{{slide.relative_to_run}}}}}}}",
        r"\end{frame}",
    ]


def _tex_relative_path(path: Path, tex_dir: Path) -> str:
    return Path(os.path.relpath(path, start=tex_dir)).as_posix()


def _relative_posix(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _section_target(section: str) -> str:
    return "sec-" + _slugify(section)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug or "section"


def _humanize(value: str) -> str:
    cleaned = re.sub(r"^\d+[_-]+", "", value)
    return cleaned.replace("_", " ").replace("-", " ").title()


def _has_glob(value: str) -> bool:
    return any(char in value for char in "*?[")
