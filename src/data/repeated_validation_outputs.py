"""Validation-by-horizon metrics and multi-seed ranking report outputs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from methods.lstm_cluster.report import compile_tex


IDENTITY_COLUMNS = [
    "config_id",
    "clustering",
    "n_clusters",
    "sigma",
    "horizon",
    "threshold_mm",
]


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV without exposing a partial checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def validation_mae_by_horizon(
    actual_by_lead_day: np.ndarray,
    predicted_by_lead_day: np.ndarray,
    *,
    thresholds_mm: Sequence[float],
) -> pd.DataFrame:
    """Return validation MAE for every lead day and rain threshold."""
    actual = np.asarray(actual_by_lead_day, dtype=float)
    predicted = np.asarray(predicted_by_lead_day, dtype=float)
    if actual.ndim != 2 or predicted.ndim != 2:
        raise ValueError("Validation targets and predictions must be 2D matrices.")
    if actual.shape != predicted.shape:
        raise ValueError("Validation targets and predictions must have equal shapes.")

    rows: list[dict[str, float | int]] = []
    for lead_offset in range(actual.shape[1]):
        lead_actual = actual[:, lead_offset]
        lead_predicted = predicted[:, lead_offset]
        finite = np.isfinite(lead_actual) & np.isfinite(lead_predicted)
        for threshold in thresholds_mm:
            mask = finite & (lead_actual > float(threshold))
            rows.append(
                {
                    "horizon": lead_offset + 1,
                    "threshold_mm": float(threshold),
                    "sample_count": int(mask.sum()),
                    "val_mae": (
                        float(np.mean(np.abs(lead_predicted[mask] - lead_actual[mask])))
                        if np.any(mask)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def validation_mae_by_horizon_and_rain_regime(
    actual_by_lead_day: np.ndarray,
    predicted_by_lead_day: np.ndarray,
) -> pd.DataFrame:
    """Return validation MAE for cumulative thresholds and disjoint rain bands."""
    actual = np.asarray(actual_by_lead_day, dtype=float)
    predicted = np.asarray(predicted_by_lead_day, dtype=float)
    if actual.ndim != 2 or predicted.ndim != 2:
        raise ValueError("Validation targets and predictions must be 2D matrices.")
    if actual.shape != predicted.shape:
        raise ValueError("Validation targets and predictions must have equal shapes.")

    regimes = (
        ("overall", "Overall", np.nan, np.nan, np.nan),
        ("gt_0mm", ">0 mm", 0.0, np.nan, np.nan),
        ("0_to_10mm", "0--10 mm", np.nan, 0.0, 10.0),
        ("gt_10mm", ">10 mm", 10.0, np.nan, np.nan),
        ("10_to_30mm", "10--30 mm", np.nan, 10.0, 30.0),
        ("gt_30mm", ">30 mm", 30.0, np.nan, np.nan),
    )
    rows: list[dict[str, float | int | str]] = []
    for lead_offset in range(actual.shape[1]):
        lead_actual = actual[:, lead_offset]
        lead_predicted = predicted[:, lead_offset]
        finite = np.isfinite(lead_actual) & np.isfinite(lead_predicted)
        for regime, label, threshold, lower, upper in regimes:
            if regime == "overall":
                mask = finite
            elif regime.startswith("gt_"):
                mask = finite & (lead_actual > float(threshold))
            elif regime == "0_to_10mm":
                mask = finite & (lead_actual >= 0.0) & (lead_actual <= 10.0)
            else:
                mask = finite & (lead_actual > 10.0) & (lead_actual <= 30.0)
            rows.append(
                {
                    "horizon": lead_offset + 1,
                    "rain_regime": regime,
                    "rain_regime_label": label,
                    "threshold_mm": threshold,
                    "lower_mm": lower,
                    "upper_mm": upper,
                    "sample_count": int(mask.sum()),
                    "val_mae": (
                        float(np.mean(np.abs(lead_predicted[mask] - lead_actual[mask])))
                        if np.any(mask)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_validation_repeats(
    seed_metrics: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    """Aggregate per-seed validation MAE with sample standard deviation."""
    required = {*IDENTITY_COLUMNS, "seed", "sample_count", "val_mae"}
    missing = sorted(required - set(seed_metrics.columns))
    if missing:
        raise ValueError("Missing validation metric columns: " + ", ".join(missing))

    frame = seed_metrics.copy()
    frame["val_mae"] = pd.to_numeric(frame["val_mae"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["val_mae"])].copy()
    expected_count = len(set(int(seed) for seed in expected_seeds))
    summary = frame.groupby(IDENTITY_COLUMNS, dropna=False, as_index=False).agg(
        sample_count=("sample_count", "first"),
        mean_val_mae=("val_mae", "mean"),
        std_val_mae=("val_mae", lambda values: float(np.std(values, ddof=1))),
        completed_seed_count=("seed", "nunique"),
    )
    summary["expected_seed_count"] = expected_count
    summary["all_seeds_completed"] = summary["completed_seed_count"].eq(
        expected_count
    )
    return summary.sort_values(
        ["horizon", "threshold_mm", "mean_val_mae"], kind="stable"
    ).reset_index(drop=True)


def repeated_validation_tex(
    summary: pd.DataFrame,
    *,
    seeds: Sequence[int],
) -> str:
    """Render the five-horizon, three-threshold repeated-validation report."""
    incomplete = summary.loc[~summary["all_seeds_completed"].astype(bool)]
    if not incomplete.empty:
        raise ValueError("Cannot render a final report with incomplete seed results.")

    seed_text = ", ".join(str(int(seed)) for seed in seeds)
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,landscape,margin=1cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage[table]{xcolor}",
        r"\definecolor{rankingbestgreen}{HTML}{228B22}",
        r"\setlength{\parindent}{0pt}",
        r"\pagestyle{empty}",
        r"\begin{document}",
        r"\section*{PyTorch Repeated Validation Rain-threshold Comparison}",
        r"\colorbox{gray!8}{\parbox{\dimexpr\linewidth-2\fboxsep\relax}{%",
        r"\textbf{Common configuration:} validation split only; architecture=64/--; "
        r"window=30; loss=QWMSE(.9); early-stopping metric=MAE; "
        rf"precipitation standardized=False; seeds={seed_text}.\par",
        r"The test set was not predicted or evaluated. Mean and sample standard "
        r"deviation use five independent seeded reruns. For every $K=2,\ldots,9$, "
        r"the displayed spectral row is the lowest-mean result among five heuristic "
        r"sigmas. Bold green marks the lowest mean validation MAE in each table.%",
        r"}}",
        r"\vspace{0.6em}",
    ]
    horizons = sorted(int(value) for value in summary["horizon"].unique())
    for horizon_index, horizon in enumerate(horizons):
        if horizon_index:
            lines.append(r"\clearpage")
        lines.extend(
            [
                rf"\subsection*{{Forecast horizon D+{horizon}}}",
                r"\noindent",
            ]
        )
        threshold_items = sorted(
            float(value) for value in summary["threshold_mm"].unique()
        )
        for threshold_index, threshold in enumerate(threshold_items):
            table = summary.loc[
                summary["horizon"].eq(horizon)
                & summary["threshold_mm"].eq(float(threshold))
            ].copy()
            is_spectral = table["clustering"].astype(str).str.startswith("spectral")
            non_spectral = table.loc[~is_spectral]
            spectral = table.loc[is_spectral]
            best_spectral = spectral.loc[
                spectral.groupby("n_clusters", dropna=False)["mean_val_mae"].idxmin()
            ]
            table = pd.concat([non_spectral, best_spectral], ignore_index=True)
            table["algorithm_order"] = table["clustering"].astype(str).str.startswith(
                "spectral"
            ).astype(int)
            table = table.sort_values(
                ["n_clusters", "algorithm_order"], kind="stable"
            ).reset_index(drop=True)
            best_mean = float(table["mean_val_mae"].min())
            sample_count = int(table["sample_count"].iloc[0])
            lines.extend(
                [
                    r"\begin{minipage}[t]{0.31\linewidth}",
                    r"\centering",
                    rf"\textbf{{Actual validation precipitation $>{float(threshold):g}$ mm ($n={sample_count}$)}}\par\vspace{{0.25em}}",
                    r"\footnotesize",
                    r"\setlength{\tabcolsep}{2.8pt}",
                    r"\renewcommand{\arraystretch}{1.12}",
                    r"\rowcolors{2}{gray!10}{white}",
                    r"\begin{tabular}{@{}lrrr@{}}",
                    r"\toprule",
                    r"Clustering & $K$ & Mean Val. MAE & Std. \\",
                    r"\midrule",
                ]
            )
            for row in table.to_dict(orient="records"):
                mean_text = f"{float(row['mean_val_mae']):.4f}"
                if np.isclose(float(row["mean_val_mae"]), best_mean):
                    mean_text = (
                        rf"\textcolor{{rankingbestgreen}}{{\textbf{{{mean_text}}}}}"
                    )
                lines.append(
                    f"{row['clustering']} & {int(row['n_clusters'])} & "
                    f"{mean_text} & {float(row['std_val_mae']):.4f} " + r"\\"
                )
            lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{minipage}"])
            if threshold_index < len(threshold_items) - 1:
                lines.append(r"\hfill")
    lines.extend([r"\end{document}", ""])
    return "\n".join(lines)


def save_repeated_validation_outputs(
    seed_metrics: pd.DataFrame,
    output_dir: Path,
    report_tex_path: Path,
    *,
    seeds: Sequence[int],
    compile_pdf: bool = False,
) -> tuple[Path, Path, Path, Path | None]:
    """Save seed metrics, mean/std summary, and the final TeX/PDF report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_dir / "validation_mae_by_seed.csv"
    summary_path = output_dir / "validation_mae_summary.csv"
    write_csv_atomic(seed_metrics, seed_path)
    summary = summarize_validation_repeats(seed_metrics, expected_seeds=seeds)
    write_csv_atomic(summary, summary_path)
    report_tex_path = Path(report_tex_path)
    report_tex_path.write_text(
        repeated_validation_tex(
            summary,
            seeds=seeds,
        ),
        encoding="utf-8",
    )
    pdf_path = compile_tex(report_tex_path) if compile_pdf else None
    return seed_path, summary_path, report_tex_path, pdf_path
