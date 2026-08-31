"""Run the requested PyTorch window/cluster/loss experiment matrix."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import OUTPUTS_BASE_DIR
from data.experiment_rankings import initialize_experiment_progress
from methods.lstm_cluster_pytorch.pipeline import run_experiment
from methods.lstm_cluster_pytorch.run_experiment import (
    experiment_kwargs,
    verify_pytorch_runtime,
)


WINDOW_SIZES = [15, 30, 45]
N_CLUSTERS = [1, 3, 5, 7, 9, 12]
LOSS_PROFILES: tuple[tuple[str, str, list[float]], ...] = (
    ("mse", "mean_squared_error", [0.9]),
    ("mae", "mae", [0.9]),
    ("huber", "huber", [0.9]),
    ("weighted_mse", "weighted_mse_loss", [0.9]),
    ("quantile_q0p9", "quantile_weighted_mse", [0.9]),
    ("quantile_q0p8", "quantile_weighted_mse", [0.8]),
    ("quantile_q0p7", "quantile_weighted_mse", [0.7]),
)


def main() -> None:
    """Run all 126 requested configurations serially on CUDA."""
    verify_pytorch_runtime()
    batch_id = datetime.now().strftime("%Y_%m_%d_%Hh%M%S")
    total_configurations = len(LOSS_PROFILES) * len(WINDOW_SIZES) * len(N_CLUSTERS)
    progress_log = OUTPUTS_BASE_DIR / "pytorch" / "requested_loss_sweep_progress.log"
    initialize_experiment_progress(
        progress_log,
        run_token=batch_id,
        total=total_configurations,
    )
    for index, (loss_slug, loss_function, quantiles) in enumerate(
        LOSS_PROFILES,
        start=1,
    ):
        print(
            f"\nRequested loss sweep {index}/{len(LOSS_PROFILES)}: {loss_slug}",
            flush=True,
        )
        parameters = experiment_kwargs(
            {
                "window_sizes": WINDOW_SIZES,
                "n_clusters_list": N_CLUSTERS,
                "clustering_algorithm": ["kmeans"],
                "lstm_loss_function": loss_function,
                "loss_quantiles": quantiles,
                "quantitative_metrics": ["MAE", "MSE", "R2"],
                "sweep_name": f"requested_{loss_slug}_{batch_id}",
                "parallel_training": False,
                "require_gpu": True,
                "create_report": False,
                "ranking_progress_log": progress_log,
                "ranking_progress_run_token": batch_id,
                "ranking_progress_total": total_configurations,
            }
        )
        run_experiment(**parameters)


if __name__ == "__main__":
    main()
