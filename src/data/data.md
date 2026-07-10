# Data

This package contains code for loading, cleaning, and visualizing climate data.
It is separate from the root `data/` directory, which stores raw INMET files.

- `load_data.py`: functions for locating and loading station daily CSV files.
- `clean_data.py`: reusable dataframe cleaning helpers.
- `visualize_data.py`: intentionally empty starter module for future data
  visualization code.
- `beamer_report.py`: reusable helpers for discovering selected run plots,
  rendering a clickable Beamer presentation from them, and compiling the
  generated `.tex` into `.pdf`.
- `lstm_outputs.py`: writes experiment metrics, predictions, summaries,
  cross-cluster test model selection reports, and diagnostic plots, including
  chronological actual-versus-predicted and residual plots for each cluster.
  It also writes per-cluster test actual-versus-predicted scatter plots with
  legends.
  It also writes forecast-horizon diagnostics that compare the target at the
  configured horizon with the precipitation observed on the final input-window
  day, plus lead-day diagnostics that compare each D+k prediction output with
  the matching D+k observed precipitation.
  Configuration images are grouped into folders such as `model_fit/`,
  `prediction_overview/`, `prediction_timeseries_splits/`,
  `residual_diagnostics/`, `cluster_diagnostics/`, and
  `forecast_horizon_diagnostics/`, plus per-cluster collections such as
  `cluster_prediction_scatter/`. The `prediction_timeseries_splits/` folder
  contains one `lead_day_XX/` subfolder per forecast lead day, and each lead-day
  folder contains the sequential actual-versus-predicted test split plots.

Typical usage:

```python
from data.load_data import load_station_daily_data

df = load_station_daily_data("RS", "A801", data_root)
```
