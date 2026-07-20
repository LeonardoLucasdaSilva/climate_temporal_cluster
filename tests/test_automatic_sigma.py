from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from methods.cluster.automatic_sigma import (
    generate_sigma_candidates,
    generate_sigma_candidates_from_features,
    run_automatic_sigma_selection,
)
from methods.tools.sigma_choosing import calculate_sigma_values


class AutomaticSigmaTests(unittest.TestCase):
    def test_sigma_selection_respects_window_stride(self) -> None:
        df = pd.DataFrame({"feature": np.arange(5, dtype=float)})

        sigmas = calculate_sigma_values(
            df,
            n_values=1,
            window_size=2,
            normalize=False,
            lower_quantile=0.0,
            upper_quantile=1.0,
            window_stride=2,
        )

        np.testing.assert_allclose(sigmas, [np.sqrt(8.0)])

    def test_generates_candidates_from_prepared_features(self) -> None:
        features = np.array([[0.0], [1.0], [3.0]])

        sigmas = generate_sigma_candidates_from_features(
            features,
            n_values=3,
            lower_quantile=0.0,
            upper_quantile=1.0,
        )

        np.testing.assert_allclose(sigmas, [1.0, 2.0, 3.0])

    @patch("methods.cluster.automatic_sigma.generate_sigma_candidates_from_features")
    @patch("methods.cluster.automatic_sigma.create_pipeline_clustering_features")
    def test_dataframe_adapter_uses_pipeline_preprocessing(
        self,
        preprocessing_mock,
        generate_mock,
    ) -> None:
        prepared_features = np.array([[1.0], [2.0]])
        preprocessing_mock.return_value = (prepared_features, ["feature"])
        generate_mock.return_value = np.array([0.5, 1.0])
        df = pd.DataFrame({"feature": [1.0, 2.0]})

        result = generate_sigma_candidates(
            df,
            window_size=15,
            n_values=2,
            normalize=False,
            columns=["feature"],
            scaler_type="minmax",
            precipitation_scaler_type="standard",
            train_ratio=0.7,
            pca_variance_threshold=0.9,
            lower_quantile=0.05,
            upper_quantile=0.25,
        )

        np.testing.assert_allclose(result, [0.5, 1.0])
        preprocessing_mock.assert_called_once_with(
            df,
            window_size=15,
            normalize=False,
            columns=["feature"],
            scaler_type="minmax",
            precipitation_scaler_type="standard",
            train_ratio=0.7,
            pca_variance_threshold=0.9,
        )
        generate_mock.assert_called_once_with(
            prepared_features,
            n_values=2,
            lower_quantile=0.05,
            upper_quantile=0.25,
        )

    @patch("methods.cluster.automatic_sigma.generate_sigma_candidates")
    @patch("methods.cluster.automatic_sigma.load_station_daily_data")
    def test_runner_loads_station_and_returns_candidates(
        self,
        load_mock,
        generate_mock,
    ) -> None:
        df = pd.DataFrame({"feature": [1.0, 2.0]})
        load_mock.return_value = df
        generate_mock.return_value = np.array([0.25, 0.5])

        with patch("builtins.print") as print_mock:
            result = run_automatic_sigma_selection(
                state="RS",
                station_id="A801",
                window_size=10,
                n_values=2,
                data_root=Path("station-data"),
            )

        np.testing.assert_allclose(result, [0.25, 0.5])
        load_mock.assert_called_once_with(
            state="RS",
            station_id="A801",
            data_root=Path("station-data"),
        )
        self.assertTrue(
            any("Sigma candidates (2)" in str(call) for call in print_mock.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
