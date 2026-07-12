from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from methods.arma.pipeline import create_arma_split_targets
from methods.tools.precipitation_utils import DEFAULT_PRECIPITATION_COLUMN


class ARMATargetAlignmentTests(unittest.TestCase):
    def test_create_arma_split_targets_aligns_all_lead_days(self) -> None:
        df = pd.DataFrame(
            {
                "Data": pd.date_range("2025-01-01", periods=8, freq="D"),
                DEFAULT_PRECIPITATION_COLUMN: np.arange(8, dtype=float),
            }
        )

        targets = create_arma_split_targets(
            df,
            window_size=3,
            forecast_horizon=2,
            offset=10,
        )

        np.testing.assert_array_equal(targets.window_indices, [10, 11, 12, 13])
        np.testing.assert_array_equal(targets.origin_indices, [12, 13, 14, 15])
        np.testing.assert_allclose(
            targets.y_by_lead_day,
            [
                [3.0, 4.0],
                [4.0, 5.0],
                [5.0, 6.0],
                [6.0, 7.0],
            ],
        )
        np.testing.assert_allclose(targets.y, [4.0, 5.0, 6.0, 7.0])
        self.assertEqual(str(targets.target_dates_by_lead_day[0, 0])[:10], "2025-01-04")
        self.assertEqual(str(targets.target_dates_by_lead_day[0, 1])[:10], "2025-01-05")


if __name__ == "__main__":
    unittest.main()

