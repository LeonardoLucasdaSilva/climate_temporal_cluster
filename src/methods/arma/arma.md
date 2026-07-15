# ARMA

This package contains a univariate ARMA precipitation baseline for comparison
with the LSTM+cluster experiment.

- `run_arma.py`: user-facing runner with station, forecast horizon, window-size
  alignment, and ARMA order knobs.
- `pipeline.py`: chronological data preparation, ARMA fitting through
  `statsmodels`, rolling lead-day forecasts, and sweep orchestration.

ARMA is fit as `ARIMA(order=(p, 0, q))` on the training precipitation series
only. Validation and test predictions reuse the fitted parameters and condition
each forecast on observations available up to that forecast origin. The
`WINDOW_SIZES` setting does not change the ARMA state model; it aligns forecast
origins and target rows with the LSTM sliding-window convention so the saved
plots and metrics can be compared directly.

Outputs are written under:

```text
outputs/dd_mm_yy/ARMA/arma_sweep_<STATE>_<STATION>_<timestamp>/
```

Each configuration folder includes `metrics_summary.csv`,
`test_predictions.csv`, `summary.txt`, `evaluation_report.txt`,
`arma_model_summary.txt`, `prediction_overview/`,
`prediction_timeseries_splits/`, `residual_diagnostics/`, `model_fit/`, and
`forecast_horizon_diagnostics/` with per-lead-day metrics and plots.
