"""Tests for precipitation utility helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from methods.tools.precipitation_utils import (
    horizon_precipitation,
    precipitation_bin_edges,
    precipitation_targets,
)


class PrecipitationUtilsTests(unittest.TestCase):
    def test_horizon_precipitation_supports_future_horizons(self) -> None:
        df = pd.DataFrame({"PRECIPITACAO_TOTAL": [0.0, 1.0, 2.0, 3.0, 4.0]})

        horizon_one = horizon_precipitation(df, window_size=2, horizon=1)
        horizon_two = horizon_precipitation(df, window_size=2, horizon=2)

        np.testing.assert_allclose(horizon_one[:3], [2.0, 3.0, 4.0])
        self.assertTrue(np.isnan(horizon_one[3]))
        np.testing.assert_allclose(horizon_two[:2], [3.0, 4.0])
        self.assertTrue(np.all(np.isnan(horizon_two[2:])))

    def test_precipitation_targets_returns_finite_targets_and_indices(self) -> None:
        df = pd.DataFrame({"PRECIPITACAO_TOTAL": [0.0, 1.0, np.nan, 3.0, 4.0]})

        indices, targets = precipitation_targets(
            df,
            window_size=2,
            n_windows=4,
            horizon=1,
        )

        np.testing.assert_array_equal(indices, [1, 2])
        np.testing.assert_allclose(targets, [3.0, 4.0])

    def test_precipitation_bin_edges_handles_empty_and_zero_values(self) -> None:
        np.testing.assert_allclose(precipitation_bin_edges(np.array([])), [0.0, 1.0])
        np.testing.assert_allclose(
            precipitation_bin_edges(np.array([0.0, 0.0])),
            [0.0, 1.0],
        )


if __name__ == "__main__":
    unittest.main()
