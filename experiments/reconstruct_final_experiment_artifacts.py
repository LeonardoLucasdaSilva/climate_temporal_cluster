"""Reconstruct ordinary plots, CSVs, TeX, and PDF from compact final output."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.final_experiment_storage import reconstruct_artifacts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Compact configuration folder containing metadata.json and result.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional artifact destination (default: RUN_DIR/reconstructed_artifacts).",
    )
    arguments = parser.parse_args()
    output = reconstruct_artifacts(arguments.run_dir, arguments.output_dir)
    print(f"Reconstructed artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
