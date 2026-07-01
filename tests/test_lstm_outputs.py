"""Tests for LSTM experiment output helpers."""

from __future__ import annotations

import unittest

import numpy as np

from data.lstm_outputs import compressed_time_positions


class LstmOutputTests(unittest.TestCase):
    def test_compressed_time_positions_preserves_small_gaps(self) -> None:
        positions, compressed = compressed_time_positions(
            np.array([2, 3, 6, 9]),
            max_gap=4,
        )
        np.testing.assert_allclose(positions, [0.0, 1.0, 4.0, 7.0])
        np.testing.assert_array_equal(compressed, [False, False, False])

    def test_compressed_time_positions_shortens_large_gaps(self) -> None:
        positions, compressed = compressed_time_positions(
            np.array([2, 3, 30, 32, 100]),
            max_gap=5,
        )
        np.testing.assert_allclose(positions, [0.0, 1.0, 6.0, 8.0, 13.0])
        np.testing.assert_array_equal(compressed, [False, True, False, True])


if __name__ == "__main__":
    unittest.main()
