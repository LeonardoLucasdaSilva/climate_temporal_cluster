"""User-facing entry point for the ARMA precipitation baseline."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import DATA_ROOT, OUTPUTS_DIR
from methods.arma.pipeline import parse_arma_orders, run_arma_experiment


# Station and data
STATE = "RS"
STATION_ID = "A801"

# Forecast alignment. WINDOW_SIZES mirrors the LSTM runner's input-window
# alignment so test samples and lead-day plots can be compared directly.
WINDOW_SIZES = [5, 10, 15]
FORECAST_HORIZON = 5

# Traditional ARMA(p, q) baselines. ARMA is fit as ARIMA(order=(p, 0, q)).
ARMA_ORDERS = [(1, 0), (2, 1), (5, 1)]
TREND = "c"
CLIP_NEGATIVE_PREDICTIONS = True
CONTINUE_ON_ERROR = True

# Train/validation/test split
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1

# Outputs
OUTPUT_ROOT = OUTPUTS_DIR
SWEEP_NAME = None
SWEEP_NAME_PREFIX = "arma_sweep"
TIMESTAMP_FORMAT = "%Y_%m_%d_%Hh%M"


def main() -> None:
    """Run the configured ARMA baseline sweep."""
    sweep_dir = run_arma_experiment(
        state=STATE,
        station_id=STATION_ID,
        window_sizes=WINDOW_SIZES,
        arma_orders=parse_arma_orders(ARMA_ORDERS),
        forecast_horizon=FORECAST_HORIZON,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        data_root=DATA_ROOT,
        output_root=OUTPUT_ROOT,
        sweep_name=SWEEP_NAME,
        sweep_name_prefix=SWEEP_NAME_PREFIX,
        timestamp_format=TIMESTAMP_FORMAT,
        trend=TREND,
        clip_negative_predictions=CLIP_NEGATIVE_PREDICTIONS,
        continue_on_error=CONTINUE_ON_ERROR,
    )
    print(f"ARMA outputs saved to: {sweep_dir}")


if __name__ == "__main__":
    main()

