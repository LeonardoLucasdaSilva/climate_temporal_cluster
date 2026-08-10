# Data

This package contains code for loading, cleaning, and visualizing climate data.
It is separate from the root `data/` directory, which stores raw INMET files.

- `load_data.py`: functions for locating and loading station daily CSV files.
- `clean_data.py`: reusable dataframe cleaning helpers.
- `visualize_data.py`: intentionally empty starter module for future data
  visualization code.
- `beamer_report.py`: reusable helpers for discovering selected run plots,
  rendering a Madrid-style Beamer presentation from them with a table-of-
  contents overview and an optional ruled selected-parameter table that accepts
  runner-style names such as `CLUSTERING_ALGORITHM`, and compiling the generated
  `.tex` into `.pdf` with shared MiKTeX user state under `outputs/.miktex`.
- `arma_outputs.py`: writes ARMA baseline metrics, predictions, summaries, and
  prediction/residual/lead-day diagnostic plots under
  `outputs/dd_mm_yy/ARMA/`.
  Its sweep summary records whether independent ARMA order configurations ran
  in parallel.
- `hyperparam_tuning_outputs.py`: checkpoints trial results for the LSTM
  random/grid tuner, writes a readable tuning summary, and persists the best
  validation-selected hyperparameters as JSON.
- `lstm_comparative_outputs.py`: writes the optional sweep-level
  `comparative_analysis/` tree. It aligns same-cluster predictions on the
  intersection of real target dates, recalculates per-lead MSE, RMSE, MAE, and
  R2 on that common interval, and renders one Seaborn
  `01_test_timeseries_comparison_lead_day_XX.png` panel per lead day. Each panel
  uses a shared, subdued observed reference behind thin, semi-transparent
  prediction curves from a colorblind palette so overlapping runs remain
  legible. It compares shared-scale scatter plots and combines cluster training
  histories using training-sample weights up to the last epoch shared by every
  contributing cluster. Before writing, it removes only its own known stale
  artifacts from a reused comparison folder. It also exports the
  full and aligned prediction rows, tidy histories, comparative metrics,
  manifest, text summary, and `report_compare.tex` used by the plots. The
  LaTeX report starts with a linked table of contents and gathers Cluster Step
  diagnostics plus the comparative Prediction Time Series, Prediction Scatter
  plot, Training History, and Test Metrics sections. Test Metrics first
  provides one overall RMSE/MAE/R2 row per pivot value, pooled across all
  forecast days, followed by the detailed common-date and lead-day tables.
  The Test Metrics section also writes an overall three-panel plot with pivot
  on x and RMSE, MAE, and R2 on y, while per-D+k plots show those same metrics
  in a single row of three columns. Every report table bolds the best model for
  each metric. Cluster Step omits silhouette plots for `K = 1`
  and, when `K` is fixed across the sweep, shows fixed cluster diagnostics only
  once instead of repeating them for every run; run-specific
  `05_cluster_performance.png` plots are still shown per run.
  Pivot aliases such as `K` and `lr` are normalized here, and invalid dates or
  conflicting real values fail explicitly.
- `meta_analysis_report.py`: discovers raw LSTM and ARMA metric artifacts across
  multiple saved experiment trees, deduplicates overlapping roots by physical
  run path, and creates a standalone LaTeX report for MSE, MAE, and R2 overall
  metrics plus RMSE, MAE, and R2 lead-day metrics. Report tables place globally
  numbered runs in columns, split wide comparisons into sequential blocks, bold
  the best available metric, and repeat the comparison for every available lead
  day. A longtable registry maps each number to its experiment metadata. When
  `start_date` or `end_date` is supplied, it also
  loads saved prediction rows and writes Seaborn
  `meta_analysis_timeseries/01_meta_timeseries_comparison_lead_day_XX.png`
  plots that compare observed precipitation with `Run 1`, `Run 2`, and the
  remaining selected runs on the common target dates in that period. Explicit
  `comparative_metrics.csv` inputs are supported as a separate
  common-date-aligned scope and cannot be mixed with raw metrics; multiple
  aligned sources must share lead days, date intervals, and common sample
  counts. Overall lead-day values respect the configured forecast horizon
  rather than assuming the greatest lead present in the CSV.
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
  When `SILHOUETTE_INFO=True`, cluster diagnostics include silhouette analysis
  plots and summary scores for the split feature matrices used by the
  experiment pipeline; `SILHOUETTE_INFO=False` skips
  `08_silhouette_analysis.png` and `silhouette_scores.csv`. DTW runs use
  precomputed pairwise DTW distances for silhouette instead of flattening the
  window tensors; K-Shape runs use precomputed pairwise SBD distances on those
  temporal windows. The cluster
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
  When `TRAIN_INFO=True`, normal LSTM runs mirror these plot families under
  `train_performance/`, using real training targets and same-cluster training
  predictions. This folder contains `cluster_prediction_histograms/`,
  `cluster_prediction_scatter/`, `cluster_prediction_timeseries/`, and
  `prediction_timeseries_splits/lead_day_XX/`, plus the plotted values in
  `train_predictions.csv`. `TRAIN_INFO=False` skips that folder. LSTM reports
  and sweep manifests include the configured early-stopping warm-up epochs
  alongside patience and the monitored metric.
  For within-cluster shape analysis, `cluster_diagnostics/clusters_timeseries/`
  contains raw precipitation series from every test input window. Each
  `cluster_<id>/` folder holds a 5x4 overview panel (with extra pages after 20
  windows) and individual series under `individual_windows/`. The folder tree
  always includes every configured cluster, including a placeholder panel when
  a cluster has no test windows. `PLOT_CLUSTER_TIMESERIES=False` skips this
  diagnostic entirely. When it is enabled, `CLUSTER_TIMESERIES_PLOT_LIMIT`
  bounds the total number of selected individual series across clusters; `None`
  keeps all test windows.
  The `cluster_diagnostics/` folder also receives `cluster_timeline.png`, an XY
  plot of every window in chronological split order (training, validation, then
  test) against its assigned cluster label.

Typical usage:

```python
from data.load_data import load_station_daily_data

df = load_station_daily_data("RS", "A801", data_root)
```
