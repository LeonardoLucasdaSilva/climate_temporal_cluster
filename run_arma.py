"""Run the ARMA precipitation baseline from the repository root."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from methods.arma.run_arma import main


if __name__ == "__main__":
    main()

