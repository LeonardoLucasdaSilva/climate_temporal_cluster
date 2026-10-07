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
    "loss_quantiles",
    "mean_val_mae",
    "std_val_mae",
    "completed_seed_count",
    "expected_seed_count",
    "all_seeds_completed",
]


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
            "mean_val_mae",
            "std_val_mae",
            "completed_seed_count",
        )
        if column in available_columns
    ]
    headers = [column.replace("_", " ").title() for column in columns]
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
