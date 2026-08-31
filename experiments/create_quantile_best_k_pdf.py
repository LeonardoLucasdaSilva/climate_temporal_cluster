"""Generate direct quantile K=1 versus best-higher-K PDF tables."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.quantile_best_k_pdf import write_quantile_best_k_pdf


def main() -> None:
    output = write_quantile_best_k_pdf(
        PROJECT_ROOT / "outputs" / "pytorch" / "experiment_rankings.csv",
        PROJECT_ROOT / "outputs" / "pytorch" / "quantile_best_k_comparison.pdf",
    )
    print(output)


if __name__ == "__main__":
    main()
