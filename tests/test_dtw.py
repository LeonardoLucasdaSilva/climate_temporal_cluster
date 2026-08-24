"""Tests for multivariate Dynamic Time Warping helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from methods.cluster.dtw import (
    cross_dtw_distances,
    dtw_distance,
    normalize_dissimilarity_metric,
    pairwise_dtw_distances,
)
from methods.cluster.ng import affinity_matrix_from_distances


class DynamicTimeWarpingTests(unittest.TestCase):
    def test_metric_names_are_normalized(self) -> None:
        self.assertEqual(normalize_dissimilarity_metric("EUCLIDEAN"), "euclidean")
        self.assertEqual(normalize_dissimilarity_metric("DTW"), "dtw")
        self.assertEqual(normalize_dissimilarity_metric("DWT"), "dtw")
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported cluster_dissimilarity_metric",
        ):
            normalize_dissimilarity_metric("manhattan")

    def test_dtw_aligns_shifted_univariate_patterns(self) -> None:
        first = np.array([0.0, 1.0, 1.0])
        second = np.array([0.0, 0.0, 1.0])

        self.assertEqual(dtw_distance(first, first), 0.0)
        self.assertEqual(dtw_distance(first, second), 0.0)
        self.assertGreater(np.linalg.norm(first - second), 0.0)

    def test_pairwise_and_cross_distances_are_consistent(self) -> None:
        windows = np.array(
            [
                [[0.0, 0.0], [1.0, 1.0]],
                [[0.0, 0.0], [2.0, 2.0]],
                [[5.0, 5.0], [6.0, 6.0]],
            ]
        )

        pairwise = pairwise_dtw_distances(windows)
        cross = cross_dtw_distances(windows[:1], windows[1:])

        np.testing.assert_allclose(pairwise, pairwise.T)
        np.testing.assert_allclose(np.diag(pairwise), 0.0)
        np.testing.assert_allclose(cross[0], pairwise[0, 1:])

    def test_gaussian_affinity_accepts_precomputed_dtw_distances(self) -> None:
        distances = np.array([[0.0, 2.0], [2.0, 0.0]])

        affinities = affinity_matrix_from_distances(distances, sigma=2.0)

        self.assertEqual(affinities[0, 0], 0.0)
        self.assertEqual(affinities[1, 1], 0.0)
        self.assertAlmostEqual(affinities[0, 1], np.exp(-0.5))


if __name__ == "__main__":
    unittest.main()
