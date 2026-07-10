"""Command-line runner for creating Beamer decks from saved run plots."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.beamer_report import (  # noqa: E402
    compile_beamer_pdf,
    discover_plots,
    render_beamer,
    resolve_selected_plots,
    write_beamer,
)


# =============================================================================
# EDITE AQUI
# =============================================================================

# Pasta de uma configuracao ja salva dentro de outputs/.
# Exemplo:
# RUN_DIR = PROJECT_ROOT / "outputs" / "lstm_cluster_sweep_RS_A801_2026_07_09_12h26" / "RS_A801_w15_k03_kmeans_sigma_na"
RUN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "lstm_cluster_sweep_RS_A801_2026_07_09_16h37"
    / "RS_A801_w15_k01_kmeans_sigma_na"
)



# Lista de plots que entram no Beamer. Aceita caminho relativo da run, glob,
# caminho absoluto dentro da run, ou trecho do nome/caminho.
# Deixe [] para incluir todos os plots encontrados.
SELECTED_PLOTS = [
    "prediction_overview/02_predictions_vs_actual.png",
    "cluster_prediction_scatter/*.png",
    "residual_diagnostics/*.png",
    "cluster_diagnostics/*.png",
]

# Arquivo opcional com um seletor de plot por linha. Use None para ignorar.
PLOTS_FILE = None

# Saida opcional. Use None para salvar como RUN_DIR / "beamer.tex".
OUTPUT_PATH = None

# Titulo opcional. Use None para usar o nome da pasta da run.
TITLE = None

# Troque para True para apenas listar os plots disponiveis e nao criar o .tex.
LIST_PLOTS = False

# Troque para False se quiser criar apenas o .tex sem compilar o PDF.
COMPILE_PDF = True

# Numero de compilacoes do pdflatex. Duas passagens atualizam links/overview.
PDFLATEX_RUNS = 2


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the Beamer runner."""
    parser = argparse.ArgumentParser(
        description="Create beamer.tex from selected plots in one outputs run.",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Experiment configuration folder under outputs.",
    )
    parser.add_argument(
        "--plots",
        nargs="*",
        default=None,
        help=(
            "Plot selectors: relative paths, glob patterns, absolute paths inside "
            "the run, or case-insensitive substrings. Omit to include all plots."
        ),
    )
    parser.add_argument(
        "--plots-file",
        type=Path,
        help="Text file with one plot selector per line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .tex path. Defaults to RUN_DIR/beamer.tex.",
    )
    parser.add_argument(
        "--title",
        help="Presentation title. Defaults to the run folder name.",
    )
    parser.add_argument(
        "--list-plots",
        action="store_true",
        help="Print all available run-relative plot paths and exit.",
    )
    parser.add_argument(
        "--no-compile-pdf",
        action="store_true",
        help="Create only the .tex file and skip pdflatex.",
    )
    parser.add_argument(
        "--pdflatex-runs",
        type=int,
        default=PDFLATEX_RUNS,
        help="Number of pdflatex runs. Defaults to 2.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Beamer generator from the command line."""
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        return run_from_config()

    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    all_plots = discover_plots(run_dir)
    if args.list_plots:
        for plot_path in all_plots:
            print(_run_relative(plot_path, run_dir))
        return 0

    selectors: list[str] = []
    if args.plots:
        selectors.extend(args.plots)
    if args.plots_file is not None:
        selectors.extend(_read_plot_selector_file(args.plots_file))

    selected_plots = resolve_selected_plots(run_dir, selectors)
    output_path = write_beamer(
        run_dir,
        selected_plots,
        output_path=args.output,
        title=args.title,
    )
    print(f"Wrote {output_path}")
    print(f"Selected {len(selected_plots)} plot(s)")
    if not args.no_compile_pdf:
        pdf_path = compile_beamer_pdf(output_path, runs=args.pdflatex_runs)
        print(f"Wrote {pdf_path}")
    return 0


def run_from_config() -> int:
    """Create the Beamer file using the editable constants above."""
    run_dir = Path(RUN_DIR).resolve()
    if not run_dir.is_dir():
        raise SystemExit(
            "RUN_DIR does not exist. Edit RUN_DIR at the top of "
            f"{Path(__file__).name}: {run_dir}"
        )

    all_plots = discover_plots(run_dir)
    if LIST_PLOTS:
        for plot_path in all_plots:
            print(_run_relative(plot_path, run_dir))
        return 0

    selectors = list(SELECTED_PLOTS)
    if PLOTS_FILE is not None:
        selectors.extend(_read_plot_selector_file(Path(PLOTS_FILE)))

    selected_plots = resolve_selected_plots(run_dir, selectors)
    output_path = write_beamer(
        run_dir,
        selected_plots,
        output_path=Path(OUTPUT_PATH) if OUTPUT_PATH is not None else None,
        title=TITLE,
    )
    print(f"Wrote {output_path}")
    print(f"Selected {len(selected_plots)} plot(s)")
    if COMPILE_PDF:
        pdf_path = compile_beamer_pdf(output_path, runs=PDFLATEX_RUNS)
        print(f"Wrote {pdf_path}")
    return 0


def _read_plot_selector_file(path: Path) -> list[str]:
    selectors = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            selectors.append(stripped)
    return selectors


def _run_relative(plot_path: Path, run_dir: Path) -> str:
    return plot_path.resolve().relative_to(run_dir.resolve()).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
