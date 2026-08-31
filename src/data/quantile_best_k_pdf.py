"""Direct PDF tables for matched quantile K=1 and best higher-K runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


THRESHOLDS = (
    ("gt_10mm", "Actual precipitation >10 mm"),
    ("gt_20mm", "Actual precipitation >20 mm"),
    ("gt_30mm", "Actual precipitation >30 mm"),
    ("gt_q99", "99th percentile"),
)
LOSSES = ("QWMSE(.9)", "QWMSE(.85)")


def select_best_pairs(rankings: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Select one K=1/best-higher-K pair per loss at w=15."""
    selected = rankings[
        rankings["loss_function"].isin(LOSSES)
        & rankings["window_size"].eq(15)
    ].copy()
    selected["sweep"] = selected["run_id"].astype(str).str.split("/").str[0]
    pairs: list[tuple[pd.Series, pd.Series]] = []
    for loss in LOSSES:
        possible_pairs: list[tuple[pd.Series, pd.Series]] = []
        bases = selected[
            selected["loss_function"].eq(loss)
            & selected["n_clusters"].eq(1)
        ]
        for _, base in bases.iterrows():
            candidates = selected[selected["n_clusters"].gt(1)]
            for column in (
                "sweep",
                "window_size",
                "clustering_method",
                "loss_function",
                "patience_metric",
                "lstm_units",
            ):
                candidates = candidates[candidates[column].eq(base[column])]
            candidates = candidates.dropna(
                subset=[f"{prefix}_mae", f"{prefix}_mae_rank"]
            )
            if candidates.empty:
                continue
            best = candidates.sort_values(
                [f"{prefix}_mae_rank", f"{prefix}_mae", "n_clusters"],
                kind="stable",
            ).iloc[0]
            if float(best[f"{prefix}_mae"]) < float(base[f"{prefix}_mae"]):
                possible_pairs.append((base, best))
        if not possible_pairs:
            continue
        pairs.append(
            min(
                possible_pairs,
                key=lambda pair: (
                    int(pair[1][f"{prefix}_mae_rank"]),
                    float(pair[1][f"{prefix}_mae"]),
                ),
            )
        )
    return pd.DataFrame(
        [row for pair in pairs for row in pair]
    ).reset_index(drop=True)


def write_quantile_best_k_pdf(ranking_csv: Path, output_pdf: Path) -> Path:
    """Create one sectioned PDF table of matched quantile configurations."""
    rankings = pd.read_csv(ranking_csv)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        sections = [
            (title, prefix, select_best_pairs(rankings, prefix))
            for prefix, title in THRESHOLDS
        ]
        _write_table(pdf, sections)
    return output_pdf


def _write_table(
    pdf: PdfPages,
    sections: list[tuple[str, str, pd.DataFrame]],
) -> None:
    headers = [
        "Window",
        "K",
        "Loss",
        "Patience",
        "MAE",
        "MSE",
    ]
    values: list[list[str]] = []
    row_kinds: list[str] = []
    separator_titles: dict[int, str] = {}
    for title, prefix, rows in sections:
        separator = [""] * len(headers)
        values.append(separator)
        row_kinds.append("separator")
        separator_titles[len(values)] = title
        for row_index, row in enumerate(rows.itertuples(index=False)):
            data = row._asdict()
            values.append(
                [
                    str(int(row.window_size)),
                    str(int(row.n_clusters)),
                    str(row.loss_function),
                    str(row.patience_metric),
                    _metric(data[f"{prefix}_mae"]),
                    _metric(data[f"{prefix}_mse"]),
                ]
            )
            row_kinds.append("better" if row_index % 2 == 1 else "baseline")

    figure, axis = plt.subplots(figsize=(15.5, 9.0))
    axis.axis("off")
    axis.set_title(
        "K=1 versus best matched higher-K quantile configurations",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )
    table = axis.table(
        cellText=values,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.38)
    for (table_row, _column), cell in table.get_celld().items():
        if table_row == 0:
            cell.set_facecolor("#CFE2F3")
            cell.set_text_props(weight="bold")
            continue
        kind = row_kinds[table_row - 1]
        if kind == "separator":
            cell.set_facecolor("#3D6D8E")
            cell.set_edgecolor("#3D6D8E")
            cell.set_text_props(color="white", weight="bold", fontsize=10)
        elif kind == "better":
            cell.set_facecolor("#F7F7F7")
            cell.set_text_props(weight="bold")
            if _column in (headers.index("MAE"), headers.index("MSE")):
                cell.set_text_props(color="#228B22", weight="bold")
        else:
            cell.set_facecolor("#F7F7F7")

    figure.text(
        0.5,
        0.055,
        (
            "Bold rows are the lowest-MAE higher-K configurations matched to "
            "the K=1 row immediately above; all other settings are identical."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.02, 0.08, 0.98, 0.96))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    inverse_figure = figure.transFigure.inverted()
    for table_row, title in separator_titles.items():
        left = table[(table_row, 0)].get_window_extent(renderer)
        right = table[(table_row, len(headers) - 1)].get_window_extent(renderer)
        center = inverse_figure.transform(
            ((left.x0 + right.x1) / 2, (left.y0 + left.y1) / 2)
        )
        figure.text(
            center[0],
            center[1],
            title,
            ha="center",
            va="center",
            color="white",
            weight="bold",
            fontsize=10,
        )
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _metric(value: object) -> str:
    return "--" if pd.isna(value) else f"{float(value):.4f}"
