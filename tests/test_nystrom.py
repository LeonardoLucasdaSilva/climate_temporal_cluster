"""Tests for spectral Nyström out-of-sample cluster assignment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from methods.cluster.ng import (  # noqa: E402
    SpectralClusteringModel,
    build_spectral_training_graph,
    cross_affinity_matrix,
)


class NystromExtensionTests(unittest.TestCase):
    def test_cross_affinity_uses_training_references(self) -> None:
        held_out = np.array([[0.0], [2.0]])
        training = np.array([[0.0], [1.0]])

        affinities = cross_affinity_matrix(held_out, training, sigma=1.0)

        np.testing.assert_allclose(
            affinities,
            [
                [1.0, np.exp(-0.5)],
                [np.exp(-2.0), np.exp(-0.5)],
            ],
        )

    def test_nystrom_routes_held_out_points_in_spectral_space(self) -> None:
        training = np.array(
            [[-3.2], [-3.0], [-2.8], [2.8], [3.0], [3.2]],
            dtype=float,
        )
        held_out = np.array([[-3.1], [3.1]], dtype=float)
        model = SpectralClusteringModel(
            sigma=0.5,
            k=2,
            random_state=42,
        ).fit(training)

        labels = model.predict(held_out)

        self.assertEqual(labels[0], model.labels_[1])
        self.assertEqual(labels[1], model.labels_[4])
        self.assertNotEqual(labels[0], labels[1])
        self.assertTrue(np.all(np.isfinite(model.transform(held_out))))

    def test_nystrom_accepts_precomputed_cross_distances(self) -> None:
        training = np.array([[-2.0], [-1.8], [1.8], [2.0]])
        held_out = np.array([[-1.9], [1.9]])
        train_distances = np.abs(training - training.T)
        cross_distances = np.abs(held_out - training.T)
        model = SpectralClusteringModel(sigma=0.5, k=2).fit(
            training,
            distance_matrix=train_distances,
        )

        direct = model.predict(held_out)
        precomputed = model.predict(
            held_out,
            cross_distance_matrix=cross_distances,
        )

        np.testing.assert_array_equal(precomputed, direct)

    def test_one_training_graph_supports_multiple_cluster_counts(self) -> None:
        training = np.array(
            [[-4.0], [-3.8], [-0.1], [0.1], [3.8], [4.0]],
            dtype=float,
        )
        graph = build_spectral_training_graph(training, sigma=0.4, max_k=3)

        two_clusters = SpectralClusteringModel(
            sigma=0.4,
            k=2,
            random_state=42,
        ).fit_from_training_graph(graph)
        three_clusters = SpectralClusteringModel(
            sigma=0.4,
            k=3,
            random_state=42,
        ).fit_from_training_graph(graph)

        self.assertEqual(graph.max_k, 3)
        self.assertEqual(two_clusters.embedding_.shape, (6, 2))
        self.assertEqual(three_clusters.embedding_.shape, (6, 3))
        self.assertEqual(len(np.unique(two_clusters.labels_)), 2)
        self.assertEqual(len(np.unique(three_clusters.labels_)), 3)

if __name__ == "__main__":
    unittest.main()
