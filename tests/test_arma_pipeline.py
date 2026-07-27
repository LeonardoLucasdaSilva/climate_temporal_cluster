from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from methods.arma.pipeline import (
    ARMAConfig,
    _execute_arma_configuration_jobs,
    create_arma_split_targets,
)
from methods.tools.precipitation_utils import DEFAULT_PRECIPITATION_COLUMN


class ARMATargetAlignmentTests(unittest.TestCase):
    def test_parallel_arma_jobs_preserve_order_and_capture_failures(self) -> None:
        configs = [
            ARMAConfig("RS", "A801", 5, 1, 0),
            ARMAConfig("RS", "A801", 5, 2, 1),
        ]
        jobs = [
            ((None, config, None), {"marker": index})
            for index, config in enumerate(configs)
        ]
        observed_worker_counts: list[int] = []

        class ImmediateFuture:
            def __init__(self, result: object = None, error: Exception | None = None) -> None:
                self._result = result
                self._error = error

            def result(self) -> object:
                if self._error is not None:
                    raise self._error
                return self._result

        class ImmediateExecutor:
            def __init__(self, max_workers: int) -> None:
                observed_worker_counts.append(max_workers)

            def __enter__(self) -> ImmediateExecutor:
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def submit(self, function: object, *args: object) -> ImmediateFuture:
                try:
                    return ImmediateFuture(result=function(*args))
                except Exception as exc:
                    return ImmediateFuture(error=exc)

        def fake_process(
            args: tuple[object, ...],
            kwargs: dict[str, object],
            _plot_style: object,
        ) -> dict[str, object]:
            config = args[1]
            if kwargs["marker"] == 1:
                raise RuntimeError("simulated ARMA failure")
            return {"run_name": config.name, "marker": kwargs["marker"]}

        with (
            patch("methods.arma.pipeline.ProcessPoolExecutor", ImmediateExecutor),
            patch(
                "methods.arma.pipeline.as_completed",
                side_effect=lambda futures: reversed(list(futures)),
            ),
            patch(
                "methods.arma.pipeline._run_arma_configuration_process",
                side_effect=fake_process,
            ),
            patch("methods.arma.pipeline.os.cpu_count", return_value=8),
        ):
            outcomes = _execute_arma_configuration_jobs(
                jobs,
                parallel=True,
                continue_on_error=True,
            )

        self.assertEqual(observed_worker_counts, [2])
        self.assertEqual(outcomes[0][0], {"run_name": configs[0].name, "marker": 0})
        self.assertIsNone(outcomes[0][1])
        self.assertIsNone(outcomes[1][0])
        self.assertIsInstance(outcomes[1][1], RuntimeError)

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
