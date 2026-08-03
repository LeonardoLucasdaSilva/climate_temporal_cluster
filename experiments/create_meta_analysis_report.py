"""Create a LaTeX meta-analysis across saved experiment runs."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.meta_analysis_report import (  # noqa: E402
    build_parser,
    create_meta_analysis_report,
)


# Editable runner configuration. Leave this empty when using command-line paths.
EXPERIMENT_PATHS: list[Path] = [
     PROJECT_ROOT / "outputs" / "31_07_26" / "lstm_cluster_sweep_RS_A801_2026_07_31_11h20",
     PROJECT_ROOT / "outputs" / "31_07_26" / "lstm_cluster_sweep_RS_A801_2026_07_31_10h52",
     #PROJECT_ROOT / "outputs" / "29_07_26" / "lstm_cluster_sweep_RS_A801_2026_07_29_09h43",     
    # PROJECT_ROOT / "outputs" / "29_07_26" / "ARMA" / "arma_sweep_RS_A801_2026_07_29_10h00",
     #PROJECT_ROOT / "outputs" / "29_07_26" / "lstm_cluster_sweep_RS_A801_2026_07_29_10h15",
     
]

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "META_ANALYSIS" / "META_16h07" / "meta_analysis.tex"
REPORT_TITLE = "Cross-Experiment Meta-Analysis"
RUNS_PER_TABLE = 10
DECIMAL_DIGITS = 3
START_DATE: str | None = None
END_DATE: str | None = None


def run_from_config() -> Path:
    """Create the report using the editable constants above."""
    if not EXPERIMENT_PATHS:
        raise ValueError(
            "Configure EXPERIMENT_PATHS or pass experiment paths on the "
            "command line."
        )
    report_path, runs = create_meta_analysis_report(
        EXPERIMENT_PATHS,
        OUTPUT_PATH,
        title=REPORT_TITLE,
        runs_per_table=RUNS_PER_TABLE,
        digits=DECIMAL_DIGITS,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    print(f"Meta-analysis runs: {len(runs)}")
    print(f"LaTeX report saved to: {report_path}")
    return report_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run configured mode without arguments or parse command-line inputs."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if not arguments:
            run_from_config()
            return 0

        args = build_parser().parse_args(arguments)
        report_path, runs = create_meta_analysis_report(
            args.experiment_paths,
            args.output,
            title=args.title,
            runs_per_table=args.runs_per_table,
            digits=args.digits,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        print(f"Meta-analysis runs: {len(runs)}")
        print(f"LaTeX report saved to: {report_path}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
