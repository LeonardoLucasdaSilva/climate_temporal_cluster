"""Compatibility launcher for the PyTorch two-stage random-search framework."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.random_search.run_random_search import main  # noqa: E402


if __name__ == "__main__":
    main()
