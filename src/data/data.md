# Data

This package contains code for loading, cleaning, and visualizing climate data.
It is separate from the root `data/` directory, which stores raw INMET files.
Daily station files live under `data/inmet_daily`; `data/inmet_hourly` is a
separate reserved input root for future hourly loading and preprocessing.

- `load_data.py`: functions for locating and loading station daily CSV files.
- `output_cleanup.py`: removes empty descendant directories from completed
  experiment output trees while preserving the supplied root directory.
- `final_experiment_storage.py`: writes the final PyTorch sweeps' compact
  prediction/history archives, shared targets and cluster assignments, repairs
  overall and per-horizon ranking registries, and reconstructs the ordinary
  artifact tree for selected configurations without retraining.
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
  validation-only random tuner, writes a readable tuning summary, persists the
  best validation-selected hyperparameters as JSON, and maintains the separate
  shared `random_search_validation_rankings` CSV/TeX/PDF report.
- `random_search_outputs.py`: selects the best spectral sigma per K, performs
  top-fraction screening and complete-seed aggregation, and writes the final
  PyTorch mean/std validation-MAE CSV/TeX/PDF rankings, including the 15
  D+1--D+5 by >0/>10/>30 mm rain-threshold rankings plus overall, 0--10 mm,
  and 10--30 mm rain-regime rankings.
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

The optional numeric columns `DATA_SIN` and `DATA_COS` are loaded when they
exist in a station CSV. Like the other numeric weather columns, they are then
available to the clustering and LSTM feature-selection pipelines. Missing
optional default columns do not prevent a station file from loading.

When duplicate observations occur on one date, extrema columns ending in
`_MAXIMA` or `_MAX` use the daily maximum, while columns ending in `_MINIMA` or
`_MIN` use the daily minimum. Raw `DIRECAO_VENTO` is not an automatic model
feature; wind direction is represented only by `DIRECAO_VENTO_SIN` and
`DIRECAO_VENTO_COS`, whose daily components are averaged.

`experiment_rankings.py` maintains a persistent, backend-specific leaderboard
after every completed LSTM configuration. TensorFlow writes under
`outputs/tensorflow/` and PyTorch under `outputs/pytorch/`; each location gets
an overall `experiment_rankings.csv`, a normalized
`experiment_horizon_rankings.csv`, TeX source, and a PDF. This does not require
the optional PDF for every individual configuration. Rows are upserted by configuration output path and
independently rank MAE and MSE in ascending order and R2 in descending order.
Report tables replace the run identifier with the experiment date, abbreviate
loss names, and include the first and optional second LSTM layer sizes plus a
boolean indicating whether precipitation was
standardized in both clustering and LSTM preprocessing. The `U_2` table column
uses `-` when the model has no second recurrent layer. Legacy rows are recorded
as unstandardized, with MAE patience and 1024 starting units. The landscape
report uses compact natural-width columns rather than fixed-width text cells.
The configured-target table contains the run's final forecast day. Separate
D+1 through D+H tables rank only saved lead-day outputs that were trained model
targets. Each day is written independently as
`experiment_rankings_horizon_dNN.{csv,tex,pdf}`, so older diagnostics that
reused the final-day prediction are not misrepresented as independent
forecasts. Each lead-day artifact contains the same fixed-rain and extreme-tail
threshold tables as the main report; quantile cutoffs and all metric ranks are
calculated independently for that lead day.
The same metrics and ranks are
calculated on actual-precipitation subsets above 0, 10, 20, and 30 mm and above
each run's test-target 95th and 99th percentiles. Each PDF table orders its rows
only by that table's own MAE, with older runs first when MAE values tie, rather
than reusing the overall-test ordering. Batch runners can also request
an atomically updated progress log with the completed and total configuration
counts.
Leaderboard refreshes are serialized across worker processes, so parallel
configuration completions cannot discard another run or leave the CSV, TeX,
and PDF representing different ranking snapshots.
If Windows temporarily locks a derived CSV or TeX ranking, its refreshed
`.tmp` file is retained and a warning is logged instead of terminating model
training. The next ranking audit rebuilds the table from authoritative compact
results and retries the atomic replacement; saved model outputs, metrics, and
configuration checkpoints remain unaffected.
`render_cluster_comparison_rankings` additionally writes overall and per-lead
reports whose tables group rows by identical loss, early-stopping metric, U1,
U2, precipitation standardization, and window size. Clustering method and K
are columns that vary down the rows. The best MAE and MSE inside each matched
group are bold green.
Rendered tables omit the patience and R2 columns (including R2 rank), while
the underlying registries retain those values and patience remains part of the
matched-configuration grouping.
Within every table and matched group, only the spectral configuration with the
lowest MAE for each K is displayed. The choice is recalculated from that
table's own overall or precipitation-subset MAE; K-means and K=1 rows remain.
For every trained lead day, the renderer also writes dedicated
`cluster_comparison_rankings_horizon_dNN_gt_0mm`, `_gt_10mm`, and `_gt_30mm`
TeX/PDF reports. Each dedicated report calculates and displays metrics only
for observations in its named actual-precipitation range; the existing overall
and combined per-lead reports are retained.
`render_horizon_threshold_triptych_rankings` writes a compact filtered report
with one landscape page per D+1 through D+5. Each page places the above-0,
above-10, and above-30 mm cluster-comparison tables side by side. Settings
shared by every selected run are stated above the panels and suppressed from
their columns; the window and loss filters are encoded in the filename.
`render_architecture_threshold_triptych_rankings` produces the analogous single
report with $U_1$ and $U_2$ fixed, window size varying down the rows, and the
same three side-by-side rain-threshold panels on each lead-day page.
At sweep startup, and at most once every five minutes during ranking updates,
the module also discovers completed configurations under the local backend
output root. It adds missing runs and backfills their saved lead-day metrics
without deleting registry rows whose artifacts exist only on another computer.

`quantile_best_k_pdf.py` directly creates one PDF table with horizontal sections
for the 10, 20, and 30 mm and 99th-percentile subsets. For each QWMSE(.9) and
QWMSE(.85) loss at w=15, it shows the strongest K=1/higher-K pair and bolds
the better higher-K row while holding all settings except K fixed. The winning
MAE and MSE values use green text. The compact display contains window, K,
loss, patience metric, MAE, and MSE.
