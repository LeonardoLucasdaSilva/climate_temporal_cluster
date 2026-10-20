"""Selection and ranking outputs for the two-stage PyTorch random search."""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path

import numpy as np
import pandas as pd

from methods.lstm_cluster.report import compile_tex, latex_escape


FINAL_RANKING_COLUMNS = [
    "final_rank",
    "candidate_id",
    "random_configuration_id",
    "window_size",
    "n_clusters",
    "algorithm",
    "sigma",
    "lstm_units",
    "lstm_units_2",
    "dropout_rate",
    "learning_rate",
    "weight_decay",
    "epochs",
    "batch_size",
    "patience",
    "warm_up",
    "lstm_loss_function",
    "loss_quantiles",
    "mean_val_mae",
    "std_val_mae",
    "completed_seed_count",
    "expected_seed_count",
    "all_seeds_completed",
]

HORIZON_THRESHOLD_IDENTITY_COLUMNS = [
    "candidate_id",
    "random_configuration_id",
    "window_size",
    "n_clusters",
    "algorithm",
    "sigma",
    "lstm_units",
    "lstm_units_2",
    "dropout_rate",
    "learning_rate",
    "weight_decay",
    "epochs",
    "batch_size",
    "patience",
    "warm_up",
    "lstm_loss_function",
    "loss_quantiles",
    "horizon",
    "threshold_mm",
]
RANKING_HORIZONS = (1, 2, 3, 4, 5)
RANKING_THRESHOLDS_MM = (0.0, 10.0, 30.0)
RAIN_REGIME_ORDER = (
    "overall",
    "gt_0mm",
    "0_to_10mm",
    "gt_10mm",
    "10_to_30mm",
    "gt_30mm",
)
RAIN_REGIME_LABELS = {
    "overall": "Overall",
    "gt_0mm": ">0 mm",
    "0_to_10mm": "0--10 mm",
    "gt_10mm": ">10 mm",
    "10_to_30mm": "10--30 mm",
    "gt_30mm": ">30 mm",
}


def retain_best_spectral_sigma(results: pd.DataFrame) -> pd.DataFrame:
    """Keep every non-spectral row and the lowest-MAE sigma for each spectral K."""
    if results.empty:
        return results.copy()
    required = {"algorithm", "n_clusters", "val_mae"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError("Missing screening result columns: " + ", ".join(missing))

    frame = results.copy()
    frame["val_mae"] = pd.to_numeric(frame["val_mae"], errors="coerce")
    non_spectral = frame.loc[frame["algorithm"].astype(str) != "spectral"]
    spectral = frame.loc[
        frame["algorithm"].astype(str).eq("spectral")
        & np.isfinite(frame["val_mae"])
    ]
    if spectral.empty:
        return non_spectral.reset_index(drop=True)
    grouping = [
        column
        for column in ("random_configuration_id", "n_clusters")
        if column in spectral
    ]
    best_indices = spectral.groupby(grouping, dropna=False)["val_mae"].idxmin()
    retained = pd.concat(
        [non_spectral, spectral.loc[best_indices]],
        ignore_index=True,
    )
    sort_columns = [
        column
        for column in ("random_configuration_id", "n_clusters", "algorithm")
        if column in retained
    ]
    return retained.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def rank_screening_results(
    results: pd.DataFrame,
    *,
    retain_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank finite validation MAE rows and return the top retained fraction."""
    if not 0 < retain_fraction <= 1:
        raise ValueError("retain_fraction must be in the interval (0, 1].")
    ranked = results.copy()
    ranked["val_mae"] = pd.to_numeric(ranked["val_mae"], errors="coerce")
    ranked = ranked.loc[np.isfinite(ranked["val_mae"])].copy()
    ranked = ranked.sort_values("val_mae", kind="stable").reset_index(drop=True)
    ranked["screening_rank"] = np.arange(1, len(ranked) + 1)
    keep_count = (
        max(1, math.ceil(len(ranked) * retain_fraction)) if len(ranked) else 0
    )
    ranked["retained_for_confirmation"] = ranked["screening_rank"] <= keep_count
    return ranked, ranked.head(keep_count).copy()


def aggregate_confirmation_results(
    seed_results: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    """Rank fully confirmed candidates by mean validation MAE."""
    if seed_results.empty:
        return pd.DataFrame(columns=FINAL_RANKING_COLUMNS)
    frame = seed_results.copy()
    frame["val_mae"] = pd.to_numeric(frame["val_mae"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["val_mae"])].copy()
    expected_count = len(set(int(seed) for seed in expected_seeds))
    identity_columns = [
        column
        for column in (
            "candidate_id",
            "random_configuration_id",
            "window_size",
            "n_clusters",
            "algorithm",
            "sigma",
            "lstm_units",
            "lstm_units_2",
            "dropout_rate",
            "learning_rate",
            "weight_decay",
            "epochs",
            "batch_size",
            "patience",
            "warm_up",
            "loss_quantiles",
        )
        if column in frame
    ]
    grouped = frame.groupby(
        identity_columns,
        dropna=False,
        as_index=False,
    ).agg(
        mean_val_mae=("val_mae", "mean"),
        std_val_mae=("val_mae", lambda values: float(np.std(values, ddof=1))),
        completed_seed_count=("confirmation_seed", "nunique"),
    )
    grouped["expected_seed_count"] = expected_count
    grouped["all_seeds_completed"] = grouped["completed_seed_count"].eq(
        expected_count
    )
    grouped = grouped.loc[grouped["all_seeds_completed"]].copy()
    grouped = grouped.sort_values("mean_val_mae", kind="stable").reset_index(
        drop=True
    )
    grouped["final_rank"] = np.arange(1, len(grouped) + 1)
    return grouped


def aggregate_confirmation_horizon_threshold_results(
    seed_results: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    """Rank fully confirmed candidates for every horizon/rain threshold."""
    required = {
        "candidate_id",
        "confirmation_seed",
        "horizon",
        "threshold_mm",
        "sample_count",
        "val_mae",
    }
    missing = sorted(required - set(seed_results.columns))
    if missing:
        if seed_results.empty:
            return pd.DataFrame()
        raise ValueError(
            "Missing horizon/threshold confirmation columns: " + ", ".join(missing)
        )

    frame = seed_results.copy()
    frame["val_mae"] = pd.to_numeric(frame["val_mae"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["val_mae"])].copy()
    expected_count = len(set(int(seed) for seed in expected_seeds))
    identity_columns = [
        column for column in HORIZON_THRESHOLD_IDENTITY_COLUMNS if column in frame
    ]
    grouped = frame.groupby(identity_columns, dropna=False, as_index=False).agg(
        sample_count=("sample_count", "first"),
        mean_val_mae=("val_mae", "mean"),
        std_val_mae=("val_mae", lambda values: float(np.std(values, ddof=1))),
        completed_seed_count=("confirmation_seed", "nunique"),
    )
    grouped["expected_seed_count"] = expected_count
    grouped["all_seeds_completed"] = grouped["completed_seed_count"].eq(
        expected_count
    )
    grouped = grouped.loc[grouped["all_seeds_completed"]].copy()
    grouped = grouped.sort_values(
        ["horizon", "threshold_mm", "mean_val_mae"], kind="stable"
    ).reset_index(drop=True)
    grouped["ranking_rank"] = (
        grouped.groupby(["horizon", "threshold_mm"], sort=False).cumcount() + 1
    )
    return grouped


def best_stage1_configuration_by_horizon_rain_regime(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Select the best K/algorithm/sigma within each sampled configuration."""
    required = {
        "random_configuration_id",
        "horizon",
        "rain_regime",
        "val_mae",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(
            "Missing stage-1 rain-regime columns: " + ", ".join(missing)
        )
    frame = metrics.copy()
    frame["val_mae"] = pd.to_numeric(frame["val_mae"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["val_mae"])].copy()
    if frame.empty:
        return frame
    grouping = ["random_configuration_id", "horizon", "rain_regime"]
    best_indices = frame.groupby(grouping, dropna=False)["val_mae"].idxmin()
    best = frame.loc[best_indices].copy()
    regime_order = {name: index for index, name in enumerate(RAIN_REGIME_ORDER)}
    best["_rain_regime_order"] = best["rain_regime"].map(regime_order)
    return best.sort_values(
        ["random_configuration_id", "horizon", "_rain_regime_order"],
        kind="stable",
    ).drop(columns="_rain_regime_order").reset_index(drop=True)


def aggregate_confirmation_horizon_rain_regime_results(
    seed_results: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    """Rank fully confirmed candidates for every horizon and rain regime."""
    required = {
        "candidate_id",
        "confirmation_seed",
        "horizon",
        "rain_regime",
        "sample_count",
        "val_mae",
    }
    missing = sorted(required - set(seed_results.columns))
    if missing:
        if seed_results.empty:
            return pd.DataFrame()
        raise ValueError(
            "Missing rain-regime confirmation columns: " + ", ".join(missing)
        )
    frame = seed_results.copy()
    frame["val_mae"] = pd.to_numeric(frame["val_mae"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["val_mae"])].copy()
    expected_count = len(set(int(seed) for seed in expected_seeds))
    identity_columns = [
        column
        for column in HORIZON_THRESHOLD_IDENTITY_COLUMNS
        if column in frame and column != "threshold_mm"
    ]
    for column in (
        "rain_regime",
        "rain_regime_label",
        "threshold_mm",
        "lower_mm",
        "upper_mm",
    ):
        if column in frame and column not in identity_columns:
            identity_columns.append(column)
    grouped = frame.groupby(identity_columns, dropna=False, as_index=False).agg(
        sample_count=("sample_count", "first"),
        mean_val_mae=("val_mae", "mean"),
        std_val_mae=("val_mae", lambda values: float(np.std(values, ddof=1))),
        completed_seed_count=("confirmation_seed", "nunique"),
    )
    grouped["expected_seed_count"] = expected_count
    grouped["all_seeds_completed"] = grouped["completed_seed_count"].eq(
        expected_count
    )
    grouped = grouped.loc[grouped["all_seeds_completed"]].copy()
    regime_order = {name: index for index, name in enumerate(RAIN_REGIME_ORDER)}
    grouped["_rain_regime_order"] = grouped["rain_regime"].map(regime_order)
    grouped = grouped.sort_values(
        ["horizon", "_rain_regime_order", "mean_val_mae"], kind="stable"
    ).reset_index(drop=True)
    grouped["ranking_rank"] = (
        grouped.groupby(["horizon", "rain_regime"], sort=False).cumcount() + 1
    )
    return grouped.drop(columns="_rain_regime_order")


def save_final_validation_rankings(
    rankings: pd.DataFrame,
    output_dir: Path,
    *,
    compile_pdf: bool,
) -> tuple[Path, Path, Path | None]:
    """Write final mean/std validation-MAE rankings as CSV, TeX, and PDF."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "final_validation_rankings.csv"
    tex_path = output_dir / "final_validation_rankings.tex"
    rankings.to_csv(csv_path, index=False)
    tex_path.write_text(_final_ranking_tex(rankings), encoding="utf-8")
    pdf_path = compile_tex(tex_path) if compile_pdf else None
    return csv_path, tex_path, pdf_path


def save_horizon_threshold_validation_rankings(
    rankings: pd.DataFrame,
    output_dir: Path,
    *,
    compile_pdf: bool,
) -> tuple[Path, Path, Path, Path | None]:
    """Save combined and per-table validation-MAE rankings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "final_validation_rankings_by_horizon_threshold.csv"
    tex_path = output_dir / "final_validation_rankings_by_horizon_threshold.tex"
    table_dir = output_dir / "final_rankings_by_horizon_threshold"
    table_dir.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(csv_path, index=False)
    if not rankings.empty:
        for horizon in RANKING_HORIZONS:
            for threshold in RANKING_THRESHOLDS_MM:
                table = rankings.loc[
                    rankings["horizon"].eq(horizon)
                    & rankings["threshold_mm"].eq(threshold)
                ]
                threshold_token = f"{float(threshold):g}".replace(".", "p")
                table.to_csv(
                    table_dir
                    / f"validation_mae_d{int(horizon):02d}_gt{threshold_token}mm.csv",
                    index=False,
                )
    tex_path.write_text(
        _horizon_threshold_ranking_tex(rankings),
        encoding="utf-8",
    )
    pdf_path = compile_tex(tex_path) if compile_pdf else None
    return csv_path, table_dir, tex_path, pdf_path


def _compact_random_configuration_id(value: object) -> str:
    text = str(value)
    return "R" + text.removeprefix("random_")


def _horizon_threshold_ranking_tex(rankings: pd.DataFrame) -> str:
    """Render 15 side-by-side horizon/rain-threshold ranking tables."""
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,landscape,margin=0.8cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage[table]{xcolor}",
        r"\definecolor{rankingbestgreen}{HTML}{228B22}",
        r"\setlength{\parindent}{0pt}",
        r"\pagestyle{empty}",
        r"\begin{document}",
        r"\section*{Random Search --- Validation MAE by Horizon and Rain Threshold}",
        r"\colorbox{gray!8}{\parbox{\dimexpr\linewidth-2\fboxsep\relax}{%",
        r"Each table independently ranks the fully confirmed candidates by mean "
        r"validation MAE across the confirmation seeds. Standard deviation is the "
        r"sample standard deviation. The test split was not predicted or evaluated.%",
        r"}}",
    ]
    if rankings.empty:
        lines.extend(
            [
                r"\vspace{1em}",
                r"No fully confirmed horizon/threshold results.",
                r"\end{document}",
                "",
            ]
        )
        return "\n".join(lines)

    horizons = RANKING_HORIZONS
    thresholds = RANKING_THRESHOLDS_MM
    for horizon_index, horizon in enumerate(horizons):
        if horizon_index:
            lines.append(r"\clearpage")
        lines.extend(
            [
                rf"\subsection*{{Forecast horizon D+{horizon}}}",
                r"\noindent",
            ]
        )
        for threshold_index, threshold in enumerate(thresholds):
            table = rankings.loc[
                rankings["horizon"].eq(horizon)
                & rankings["threshold_mm"].eq(threshold)
            ].sort_values("ranking_rank", kind="stable")
            sample_count = (
                int(table["sample_count"].iloc[0]) if not table.empty else 0
            )
            lines.extend(
                [
                    r"\begin{minipage}[t]{0.32\linewidth}",
                    r"\centering",
                    rf"\textbf{{Actual validation precipitation $>{threshold:g}$ mm ($n={sample_count}$)}}\par\vspace{{0.2em}}",
                    r"\scriptsize",
                    r"\setlength{\tabcolsep}{1.8pt}",
                    r"\renewcommand{\arraystretch}{1.08}",
                    r"\rowcolors{2}{gray!10}{white}",
                    r"\begin{tabular}{@{}rrrlrrr@{}}",
                    r"\toprule",
                    r"Rank & Config & $K$ & Algorithm & $\sigma$ & Mean & Std. \\",
                    r"\midrule",
                ]
            )
            for row in table.to_dict(orient="records"):
                rank_text = str(int(row["ranking_rank"]))
                mean_text = f"{float(row['mean_val_mae']):.4f}"
                std_text = f"{float(row['std_val_mae']):.4f}"
                if int(row["ranking_rank"]) == 1:
                    rank_text = rf"\textcolor{{rankingbestgreen}}{{\textbf{{{rank_text}}}}}"
                    mean_text = rf"\textcolor{{rankingbestgreen}}{{\textbf{{{mean_text}}}}}"
                    std_text = rf"\textcolor{{rankingbestgreen}}{{\textbf{{{std_text}}}}}"
                sigma_text = (
                    "--" if pd.isna(row.get("sigma")) else f"{float(row['sigma']):.3g}"
                )
                algorithm = str(row["algorithm"])
                if int(row["n_clusters"]) == 1:
                    algorithm = "baseline"
                lines.append(
                    f"{rank_text} & "
                    f"{latex_escape(_compact_random_configuration_id(row['random_configuration_id']))} & "
                    f"{int(row['n_clusters'])} & {latex_escape(algorithm)} & "
                    f"{sigma_text} & {mean_text} & {std_text} " + r"\\"
                )
            if table.empty:
                lines.append(r"\multicolumn{7}{c}{No completed candidates.} \\")
            lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{minipage}"])
            if threshold_index < len(thresholds) - 1:
                lines.append(r"\hfill")
    lines.extend([r"\end{document}", ""])
    return "\n".join(lines)


def save_horizon_rain_regime_validation_rankings(
    rankings: pd.DataFrame,
    output_dir: Path,
    *,
    compile_pdf: bool,
) -> tuple[Path, Path, Path, Path | None]:
    """Save overall, cumulative-threshold, and interval MAE rankings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "final_validation_rankings_by_horizon_rain_regime.csv"
    tex_path = output_dir / "final_validation_rankings_by_horizon_rain_regime.tex"
    table_dir = output_dir / "final_rankings_by_horizon_rain_regime"
    table_dir.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(csv_path, index=False)
    if not rankings.empty:
        for horizon in RANKING_HORIZONS:
            for regime in RAIN_REGIME_ORDER:
                table = rankings.loc[
                    rankings["horizon"].eq(horizon)
                    & rankings["rain_regime"].eq(regime)
                ]
                table.to_csv(
                    table_dir / f"validation_mae_d{horizon:02d}_{regime}.csv",
                    index=False,
                )
    tex_path.write_text(
        _horizon_rain_regime_ranking_tex(rankings),
        encoding="utf-8",
    )
    pdf_path = compile_tex(tex_path) if compile_pdf else None
    return csv_path, table_dir, tex_path, pdf_path


def _append_compact_ranking_table(
    lines: list[str],
    table: pd.DataFrame,
    *,
    title: str,
) -> None:
    sample_count = int(table["sample_count"].iloc[0]) if not table.empty else 0
    lines.extend(
        [
            r"\begin{minipage}[t]{0.32\linewidth}",
            r"\centering",
            rf"\textbf{{{title} ($n={sample_count}$)}}\par\vspace{{0.2em}}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{1.8pt}",
            r"\renewcommand{\arraystretch}{1.08}",
            r"\rowcolors{2}{gray!10}{white}",
            r"\begin{tabular}{@{}rrrlrrr@{}}",
            r"\toprule",
            r"Rank & Config & $K$ & Algorithm & $\sigma$ & Mean & Std. \\",
            r"\midrule",
        ]
    )
    for row in table.sort_values("ranking_rank", kind="stable").to_dict(
        orient="records"
    ):
        rank_text = str(int(row["ranking_rank"]))
        mean_text = f"{float(row['mean_val_mae']):.4f}"
        std_text = f"{float(row['std_val_mae']):.4f}"
        if int(row["ranking_rank"]) == 1:
            rank_text = rf"\textcolor{{rankingbestgreen}}{{\textbf{{{rank_text}}}}}"
            mean_text = rf"\textcolor{{rankingbestgreen}}{{\textbf{{{mean_text}}}}}"
            std_text = rf"\textcolor{{rankingbestgreen}}{{\textbf{{{std_text}}}}}"
        sigma_text = (
            "--" if pd.isna(row.get("sigma")) else f"{float(row['sigma']):.3g}"
        )
        algorithm = (
            "baseline"
            if int(row["n_clusters"]) == 1
            else str(row["algorithm"])
        )
        lines.append(
            f"{rank_text} & "
            f"{latex_escape(_compact_random_configuration_id(row['random_configuration_id']))} & "
            f"{int(row['n_clusters'])} & {latex_escape(algorithm)} & "
            f"{sigma_text} & {mean_text} & {std_text} " + r"\\"
        )
    if table.empty:
        lines.append(r"\multicolumn{7}{c}{No completed candidates.} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{minipage}"])


def _horizon_rain_regime_ranking_tex(rankings: pd.DataFrame) -> str:
    """Render all six rain-regime rankings on two pages per horizon."""
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,landscape,margin=0.8cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage[table]{xcolor}",
        r"\definecolor{rankingbestgreen}{HTML}{228B22}",
        r"\setlength{\parindent}{0pt}",
        r"\pagestyle{empty}",
        r"\begin{document}",
        r"\section*{Random Search --- Validation MAE by Horizon and Rain Regime}",
        r"\colorbox{gray!8}{\parbox{\dimexpr\linewidth-2\fboxsep\relax}{%",
        r"Each table independently ranks fully confirmed candidates by mean "
        r"validation MAE across confirmation seeds. Overall uses every finite "
        r"validation target. Bands are 0--10 mm (inclusive) and >10--30 mm. "
        r"The test split was not predicted or evaluated.%",
        r"}}",
    ]
    if rankings.empty:
        lines.extend(
            [
                r"\vspace{1em}",
                r"No fully confirmed horizon/rain-regime results.",
                r"\end{document}",
                "",
            ]
        )
        return "\n".join(lines)

    regime_pages = (RAIN_REGIME_ORDER[:3], RAIN_REGIME_ORDER[3:])
    page_index = 0
    for horizon in RANKING_HORIZONS:
        for regime_group in regime_pages:
            if page_index:
                lines.append(r"\clearpage")
            page_index += 1
            lines.extend(
                [
                    rf"\subsection*{{Forecast horizon D+{horizon}}}",
                    r"\noindent",
                ]
            )
            for regime_index, regime in enumerate(regime_group):
                table = rankings.loc[
                    rankings["horizon"].eq(horizon)
                    & rankings["rain_regime"].eq(regime)
                ]
                _append_compact_ranking_table(
                    lines,
                    table,
                    title=RAIN_REGIME_LABELS[regime],
                )
                if regime_index < len(regime_group) - 1:
                    lines.append(r"\hfill")
    lines.extend([r"\end{document}", ""])
    return "\n".join(lines)


def _final_ranking_tex(rankings: pd.DataFrame) -> str:
    available_columns = set(rankings.columns) or set(FINAL_RANKING_COLUMNS)
    columns = [
        column
        for column in (
            "final_rank",
            "candidate_id",
            "window_size",
            "n_clusters",
            "algorithm",
            "sigma",
            "lstm_units",
            "lstm_units_2",
            "dropout_rate",
            "learning_rate",
            "batch_size",
            "lstm_loss_function",
            "mean_val_mae",
            "std_val_mae",
            "completed_seed_count",
        )
        if column in available_columns
    ]
    header_labels = {"lstm_loss_function": "Loss"}
    headers = [
        header_labels.get(column, column.replace("_", " ").title())
        for column in columns
    ]
    lines = [
        r"\documentclass[9pt]{article}",
        r"\usepackage[margin=1cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{pdflscape}",
        r"\usepackage[table]{xcolor}",
        r"\begin{document}",
        r"\section*{PyTorch Random Search --- Multi-seed Validation Ranking}",
        r"\textbf{Selection boundary:} Ranking uses mean validation MAE only; "
        r"the test split is not predicted or evaluated.",
        r"\begin{landscape}",
        r"\scriptsize",
        rf"\begin{{longtable}}{{@{{}}{'l' * len(columns)}@{{}}}}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
        r"\endhead",
    ]
    integer_columns = {
        "final_rank",
        "window_size",
        "n_clusters",
        "batch_size",
        "completed_seed_count",
    }
    for _, row in rankings.iterrows():
        best = int(row["final_rank"]) == 1
        cells: list[str] = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                formatted = "--"
            elif column in {"mean_val_mae", "std_val_mae"}:
                formatted = f"{float(value):.6g}"
            elif column in integer_columns:
                formatted = str(int(value))
            elif column == "lstm_loss_function":
                formatted = latex_escape(
                    {
                        "quantile_weighted_mse": "QWMSE",
                        "mean_squared_error": "MSE",
                        "mae": "MAE",
                    }.get(str(value), str(value))
                )
            else:
                formatted = latex_escape(value)
            if best and column in {"final_rank", "mean_val_mae", "std_val_mae"}:
                formatted = (
                    rf"\textcolor{{green!50!black}}{{\textbf{{{formatted}}}}}"
                )
            cells.append(formatted)
        lines.append(" & ".join(cells) + r" \\")
    if rankings.empty:
        lines.append(
            rf"\multicolumn{{{len(columns)}}}{{c}}"
            r"{No fully confirmed candidates.} \\"
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
