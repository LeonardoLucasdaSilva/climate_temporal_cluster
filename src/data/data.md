# Data

This package contains code for loading, cleaning, and visualizing climate data.
It is separate from the root `data/` directory, which stores raw INMET files.

- `load_data.py`: functions for locating and loading station daily CSV files.
- `clean_data.py`: reusable dataframe cleaning helpers.
- `visualize_data.py`: intentionally empty starter module for future data
  visualization code.
- `beamer_report.py`: reusable helpers for discovering selected run plots,
  rendering a clickable Beamer presentation from them, and compiling the
  generated `.tex` into `.pdf` with shared MiKTeX user state under
  `outputs/.miktex`.
- `arma_outputs.py`: writes ARMA baseline metrics, predictions, summaries, and
  prediction/residual/lead-day diagnostic plots under
  `outputs/dd_mm_yy/ARMA/`.
- `lstm_comparative_outputs.py`: writes the optional sweep-level
  `comparative_analysis/` tree. It aligns same-cluster predictions on the
  intersection of real target dates, recalculates per-lead MSE, RMSE, MAE, and
  R2 on that common interval, compares shared-scale scatter plots, and combines
  cluster training histories using training-sample weights up to the last epoch
  shared by every contributing cluster. Before writing, it removes only its own
  known stale artifacts from a reused comparison folder. It also exports the
  full and aligned prediction rows, tidy histories, comparative metrics,
  manifest, and text summary used by the plots. Pivot aliases such as `K` and
  `lr` are normalized here, and invalid dates or conflicting real values fail
  explicitly.
- `lstm_outputs.py`: writes experiment metrics, predictions, summaries,
  cross-cluster test model selection reports, and diagnostic plots, including
  chronological actual-versus-predicted and residual plots for each cluster.
  Configuration and sweep summaries record the held-out cluster-assignment
  method, window stride, cluster dissimilarity metric, and, for KNN assignment,
  the configured neighbor count.
  The oracle transfer diagnostics compare the LSTM assigned by the test-window
  cluster with the post-hoc best LSTM for that same window, exporting routing
  summaries by assigned cluster and by assigned-to-oracle model pair.
  It also writes per-cluster test actual-versus-predicted scatter plots with
  legends.
  Cluster diagnostics include silhouette analysis plots and summary scores for
  the split feature matrices used by the experiment pipeline. DTW runs use
  precomputed pairwise DTW distances for silhouette instead of flattening the
  window tensors. The cluster
  distribution diagnostic also records each cluster's training count and
  optimizer steps per epoch for the configured batch size, with exact values
  exported to `cluster_training_batch_statistics.csv`. In cluster-only mode,
  `06_cluster_distribution.png` still plots grouped training, validation, and
  test counts; the optimizer-workload table and CSV are omitted because no
  LSTM is trained.
  The cluster precipitation distribution diagnostic combines the existing
  forecast-target boxplot with a second boxplot of each test window's mean
  precipitation across its input days, using the same training, validation, and
  test split colors as the cluster distribution diagnostic.
  It also writes CSV/text forecast-horizon diagnostics that compare the target
  at the configured horizon with the precipitation observed on the final
  input-window day, plus plotted lead-day diagnostics that compare each D+k
  prediction output with the matching D+k observed precipitation.
  Configuration images are grouped into folders such as `model_fit/`,
  `prediction_overview/`, `prediction_timeseries_splits/`,
  `residual_diagnostics/`, `cluster_diagnostics/`, and
  `forecast_horizon_diagnostics/`, plus per-cluster collections such as
  `cluster_prediction_timeseries/` and `cluster_prediction_scatter/`. The
  `cluster_precipitation_histograms/` folder contains one combined subplot
  panel in the folder root and per-cluster histogram PNGs under `individual/`.
  The `prediction_timeseries_splits/` folder contains one `lead_day_XX/` subfolder
  per forecast lead day, and each lead-day folder contains the sequential
  actual-versus-predicted test split plots using target dates from the source
  dataset on the x-axis. The `cluster_prediction_timeseries/` plots use the
  final forecast-horizon target date on the x-axis with the same `dd/mm/YYYY`
  formatting.
  Normal LSTM runs mirror these plot families under `train_performance/`, using
  real training targets and same-cluster training predictions. This folder
  contains `cluster_prediction_histograms/`, `cluster_prediction_scatter/`,
  `cluster_prediction_timeseries/`, and
  `prediction_timeseries_splits/lead_day_XX/`, plus the plotted values in
  `train_predictions.csv`.
  The `cluster_diagnostics/` folder also receives `cluster_timeline.png`, an XY
  plot of every window in chronological split order (training, validation, then
  test) against its assigned cluster label.

Typical usage:

```python
from data.load_data import load_station_daily_data

df = load_station_daily_data("RS", "A801", data_root)
```
