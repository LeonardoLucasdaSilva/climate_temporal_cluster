# Data

This package contains code for loading, cleaning, and visualizing climate data.
It is separate from the root `data/` directory, which stores raw INMET files.

- `load_data.py`: functions for locating and loading station daily CSV files.
- `clean_data.py`: reusable dataframe cleaning helpers.
- `visualize_data.py`: intentionally empty starter module for future data
  visualization code.
- `lstm_outputs.py`: writes experiment metrics, predictions, summaries,
  cross-cluster test model selection reports, and diagnostic plots, including
  chronological actual-versus-predicted and residual plots for each cluster.
  Configuration images are grouped into folders such as `model_fit/`,
  `prediction_overview/`, `prediction_timeseries_splits/`,
  `residual_diagnostics/`, and `cluster_diagnostics/`.

Typical usage:

```python
from data.load_data import load_station_daily_data

df = load_station_daily_data("RS", "A801", data_root)
```
