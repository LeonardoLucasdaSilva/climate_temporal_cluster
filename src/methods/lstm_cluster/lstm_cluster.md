# LSTM Cluster

This folder contains the active LSTM-by-cluster precipitation experiment. The
experiment predicts precipitation at a configurable forecast horizon for one
INMET station by clustering weather windows first, then training one LSTM model per cluster. Train,
validation, and test are chronological dataframe splits made before sliding
windows are created, so raw daily observations are not shared across splits.

## Files

- `run_experiment.py`: user-facing experiment runner. Edit station, sweep,
  training, split, and console settings here.
- `pipeline.py`: library functions that load data, build windows, cluster,
  train models, and save one full sweep. It should not contain experiment
  constants or a script entry point.
- `report.py`: LaTeX report writer for one configuration output folder. It
  includes configuration details, metric tables, and generated figures.
- `config_output.yaml`: output folder and plotting settings, including root
  path, optional fixed sweep name, generated name prefix, timestamp format,
  Seaborn theme defaults, and Matplotlib `rcParams`.
- `console.py`: small helpers for optional progress output.
- `__init__.py`: marks this folder as an importable package.

## Run

After installing the project, run:

```powershell
lstm-cluster
```

Or use the root launcher:

```powershell
.\.venv\Scripts\python.exe run_experiments.py
```

## Main Flow

1. Load daily station data.
2. Select numeric climate features.
3. Split the daily dataframe chronologically into train, validation, and test
   blocks.
4. Build sliding windows independently inside each split.
5. Fit the clustering covariate and precipitation normalizers
   (`CLUSTERING_FEATURE_NORMALIZE` and
   `CLUSTERING_PRECIPITATION_NORMALIZE`) plus optional PCA on training
   rows/windows only, then transform validation and test with those training
   transforms. With `PCA_FOR_CLUSTERING_ONLY = True`, use PCA coordinates only
   for cluster fitting and held-out cluster assignment.
6. Cluster training windows with K-means, spectral, or manual rain clustering,
   then assign validation and test windows with the configured cluster
   assignment method. `"centroid"` uses the nearest training-cluster centroid;
   `"knn"` uses the nearest labeled training windows.
7. Build one target column for each lead day from D+1 through the configured
   forecast horizon inside each split. The final D+`FORECAST_HORIZON` column is
   still kept as the scalar target for legacy metrics and plots.
8. Rebuild the LSTM feature matrix from the original window dimensions and fit
   the LSTM normalizers (`LSTM_FEATURE_NORMALIZE` and
   `LSTM_PRECIPITATION_NORMALIZE`) on training rows/windows only. When
   `LSTM_PRECIPITATION_NORMALIZE` is set, the LSTM target matrix is normalized
   before training; with `LSTM_PRECIPITATION_NORMALIZE = None`, precipitation
   features and targets stay in millimeters. Each LSTM still has one output
   unit per lead day, so the loss is optimized across the full
   D+1..D+`FORECAST_HORIZON` target matrix.
9. Keep train and validation predictions tied to each sample's own cluster
   model.
10. Predict each test sample with the LSTM trained for its assigned cluster.
   When `TEST_ALL_MODELS = True`, additionally evaluate every trained cluster
   LSTM as an oracle-only transfer diagnostic; this never replaces the primary
   predictions, metrics, or plots.
11. Save metrics, reports, plots, predictions, and LaTeX tables.

## Configuration

Change experiment variables in `run_experiment.py`:

- station and data: `STATE`, `STATION_ID`
- clustering sweep: `WINDOW_SIZES`, `N_CLUSTERS_LIST`,
  `CLUSTERING_ALGORITHM`, `MANUAL_CLUSTERING_METHOD`, `FORECAST_HORIZON`,
  `MANUAL_ZERO_TOLERANCE`, `SIGMA_MODE`, `N_SIGMA_VALUES`,
  `MANUAL_SIGMA_VALUES`, `USE_ALL_FEATURES`
- held-out cluster assignment: `CLUSTER_ASSIGNMENT_METHOD` supports
  `"centroid"` and `"knn"`; `CLUSTER_ASSIGNMENT_NEIGHBORS` sets K for KNN
- dimensionality reduction: `PCA_VARIANCE_THRESHOLD` enables PCA and
  `PCA_FOR_CLUSTERING_ONLY` limits it to clustering while preserving the
  original flattened-window dimensionality for LSTM inputs
- normalization: `CLUSTERING_FEATURE_NORMALIZE` and
  `CLUSTERING_PRECIPITATION_NORMALIZE` control clustering space, while
  `LSTM_FEATURE_NORMALIZE` and `LSTM_PRECIPITATION_NORMALIZE` control the
  rebuilt LSTM space and target; supported values are `"standard"`,
  `"minmax"`, and `None`
- test evaluation mode: `TEST_ALL_MODELS`
- execution mode: set `RUN_ONLY_CLUSTER = True` to stop after clustering and
  held-out assignment, without LSTM preprocessing, training, prediction, or
  supervised-output folders
- exported table metrics: `QUANTITATIVE_METRICS`
- sweep comparison: `COMPARATIVE_RUN` enables the post-sweep comparative
  analysis and `PIVOT_PARAMETER` selects the varied parameter used on metric
  axes. Aliases include `K` for `n_clusters` and `lr` for `learning_rate`.
- model hyperparameters: `LSTM_UNITS`, `LSTM_UNITS_2`, `DROPOUT_RATE`,
  `LEARNING_RATE`, `WEIGHT_DECAY`; the optimizer is AdamW and
  `WEIGHT_DECAY` controls its decoupled parameter decay. Every numeric model or
  training setting may be a scalar or a list; lists add their values to the
  Cartesian configuration grid and append stable parameter suffixes to the
  configuration folder names.
- LSTM loss: `LSTM_LOSS_FUNCTION`, `LOSS_ALPHA`, `LOSS_QUANTILES`,
  `LOSS_QUANTILE_WEIGHTS`. The default `"weighted_mse_loss"` minimizes the mean
  of `(1 + LOSS_ALPHA * y_real) * |y_real - y_pred|^2`, where `LOSS_ALPHA`
  must be finite and greater than zero. Use `"quantile_weighted_mse"` to calculate
  cluster-specific precipitation thresholds from training-target quantiles in
  the active target scale and weight rarer intensity bins automatically.
  Because the new weight is applied in the active target scale, keep targets
  non-negative (for example, `LSTM_PRECIPITATION_NORMALIZE = None`); standardized
  negative targets can produce negative weights.
- training settings: `EPOCHS`, `BATCH_SIZE`, `EARLY_STOPPING`, `PATIENCE`,
  `EARLY_STOPPING_METRIC`, `VERBOSE_TRAINING`, `SHOW_CONSOLE_INFO`
- data split: `TRAIN_RATIO`, `VAL_RATIO`, `RANDOM_STATE`
- output and plot styling settings: `OUTPUT_CONFIG`, with details in
  `config_output.yaml`

### PCA modes

Configure PCA with `PCA_VARIANCE_THRESHOLD` and
`PCA_FOR_CLUSTERING_ONLY` in `run_experiment.py`.

To run without PCA, set the variance threshold to `None`. Clustering and the
LSTM keep the original flattened-window dimensionality, each using its own
configured normalizers:

```python
PCA_VARIANCE_THRESHOLD = None
PCA_FOR_CLUSTERING_ONLY = False  # Ignored while PCA is disabled
```

To apply PCA only during clustering, provide an explained-variance threshold
between `0` and `1` and enable clustering-only mode. Clustering uses the PCA
coordinates, while the LSTM inputs retain the original flattened-window
dimensionality:

```python
PCA_VARIANCE_THRESHOLD = 0.90
PCA_FOR_CLUSTERING_ONLY = True
```

To apply PCA to both clustering and LSTM inputs, provide an explained-variance
threshold and disable clustering-only mode. Both stages receive the same
PCA-transformed features:

```python
PCA_VARIANCE_THRESHOLD = 0.90
PCA_FOR_CLUSTERING_ONLY = False
```

Change output naming and generated figure styling in `config_output.yaml`.
The `plot_style.seaborn` section controls the Seaborn theme and palette, while
`plot_style.rc_params` accepts Matplotlib rcParam names such as
`figure.facecolor`, `axes.labelsize`, or `savefig.dpi`.

For spectral clustering, set `SIGMA_MODE = "auto"` to generate
`N_SIGMA_VALUES` candidates with the distance-based heuristic. Set
`SIGMA_MODE = "manual"` to run the exact positive values listed in
`MANUAL_SIGMA_VALUES`.

Set `CLUSTERING_ALGORITHM` to `"kmeans"`, `"spectral"`, or `"manual"`.
For manual clustering, `MANUAL_CLUSTERING_METHOD = "legacy"` preserves the
original behavior: label `0` is reserved for known zero-rain targets and known
positive targets are split into ordered lower-to-heavier rain groups.
`MANUAL_ZERO_TOLERANCE` controls the maximum target precipitation treated as
zero by this legacy method.

Set `MANUAL_CLUSTERING_METHOD = "rain_level"` to calculate the raw mean
`PRECIPITACAO_TOTAL` across all `J` input days of every training window. If the
largest training-window mean is `M`, the edges are
`numpy.linspace(0, M, K + 1)`. Cluster `i` receives values in
`[edge_i, edge_{i+1})`; the final cluster also includes `M`. Thus a value equal
to an internal edge enters the next cluster. The calculation always uses raw
millimeters, independently of `CLUSTERING_PRECIPITATION_NORMALIZE`. Every bin
must receive at least one training window; otherwise the run raises a clear
empty-cluster error.

Set `FORECAST_HORIZON = 1` for next-day rain or use a larger positive integer
to train the LSTM on every lead day from D+1 through that horizon. This horizon
changes legacy labels but does not change `rain_level` labels, which depend only
on the input window.

`CLUSTERING_ALGORITHM` and `CLUSTER_ASSIGNMENT_METHOD` have separate roles.
The clustering algorithm fits labels using training windows only. Validation
and test are never refitted: `CLUSTER_ASSIGNMENT_METHOD = "centroid"` assigns
each held-out window to the closest training-cluster mean by Euclidean
distance, while `"knn"` uses majority voting among the configured number of
nearest training windows. If K is larger than the number of training windows,
the pipeline safely uses all available training windows.

Forecast-horizon precipitation alignment is handled by
`methods.tools.precipitation_utils`. The LSTM pipeline uses the configured
horizon separately for the train, validation, and test dataframes, then expands
the valid windows into lead-day matrices. Windows with any missing target from
D+1 through D+`FORECAST_HORIZON` are dropped, so the loss never receives a
partial or NaN target row.

Set `SHOW_CONSOLE_INFO = False` to hide pipeline progress messages and Keras
training output. The root `run_experiments.py` launcher has the same setting and
passes it to this runner through `LSTM_CLUSTER_SHOW_CONSOLE_INFO`.

Set `TEST_ALL_MODELS = False` to skip the additional transfer analysis. Set
`TEST_ALL_MODELS = True` to evaluate every test sample with every trained LSTM
after the ordinary same-cluster prediction. The best model per sample and
metric is chosen only for the extra oracle diagnostic; it never changes the
primary test result.

### Comparative runs

With `COMPARATIVE_RUN = True`, the pipeline retains the ordinary same-cluster
test predictions, their target dates, and the numeric Keras histories until all
configurations finish. It then creates a comparison for `PIVOT_PARAMETER`.
Canonical parameter fields are accepted directly, with aliases such as
`window`/`window_size`, `K`/`n_clusters`, and `lr`/`learning_rate`. For a sweep
with multiple tests, the selected pivot must contain at least two distinct
non-null values; invalid or constant pivots fail before LSTM training begins.

The numeric list-capable settings are `LSTM_UNITS`, `LSTM_UNITS_2`,
`DROPOUT_RATE`, `LEARNING_RATE`, `WEIGHT_DECAY`, `EPOCHS`, `BATCH_SIZE`, and
`PATIENCE`. Use a list to vary learning rate inside one sweep:

```python
LEARNING_RATE = [0.0001, 0.0005, 0.001]
COMPARATIVE_RUN = True
PIVOT_PARAMETER = "learning_rate"
```

For an interpretable hyperparameter comparison, keep all non-pivot grid
dimensions at one value. `COMPARATIVE_RUN = True` is incompatible with
`RUN_ONLY_CLUSTER = True` because cluster-only runs do not train models or
produce histories and predictions.

Different window sizes have different numbers of test windows, so their series
are aligned by the intersection of source `Target Date` values. They are never
joined by `window_index`. A duplicated or invalid date, an empty intersection,
or conflicting real precipitation for the same date and lead day raises a
clear error instead of producing a misleading plot. Comparison metrics are
recalculated on the aligned dates. Per-configuration histories are weighted by
the number of training samples in each cluster before their curves are
compared and stop at the last epoch shared by all contributing clusters. This
avoids changing the contributing cluster set when independently trained models
stop early and avoids treating cluster identifiers as equivalent after K or
clustering parameters change.

Set `EARLY_STOPPING_METRIC` to `"loss"`, `"mse"`, `"mae"`, or `"r2"` to choose
the validation metric used by Keras early stopping. Loss, MSE, and MAE are
minimized; R2 is maximized. The default `"loss"` preserves the previous
behavior.

## Outputs

Each run creates a sweep folder under the current daily output folder, usually
named like:

```text
outputs/dd_mm_yy/lstm_cluster_sweep_<STATE>_<STATION>_<timestamp>/
```

The daily folder is controlled by `group_outputs_by_day` and
`date_folder_format` in `config_output.yaml`.

The sweep folder contains summary CSV/text files and LaTeX tables. Each
configuration subfolder contains run metrics, predictions, reports, and plots.
Per-configuration output writing is handled by `data.lstm_outputs`; the
optional sweep comparison is written by `data.lstm_comparative_outputs`.
The sweep-level `sweep_summary.txt` records the PCA variance threshold and PCA
mode (`disabled`, `clustering only`, or `clustering and LSTM`) for the run.

When enabled, `comparative_analysis/` is created directly under the sweep
folder. It contains:

- `01_test_timeseries_comparison_lead_day_XX.png`: four chronological panels
  with one real curve and one same-cluster prediction curve per test;
- `02_test_scatter_comparison_lead_day_XX.png`: one shared-scale scatter facet
  per test with the ideal identity line and aligned-date RMSE/R2;
- `03_training_history_comparison.png`: cluster-weighted train and validation
  LOSS, MSE, MAE, and R2 histories;
- `04_test_metrics_vs_<pivot>_lead_day_XX.png`: common-date MSE, RMSE, MAE, and
  R2 against the selected pivot;
- `test_predictions_comparison.csv` and `aligned_test_predictions.csv`: full
  and common-date prediction rows;
- `training_history_comparison.csv`, `comparative_metrics.csv`, and
  `comparison_manifest.csv`: numeric sources for the plots and traceability;
- `comparison_summary.txt`: alignment policy, common date interval, and the
  best test for each displayed metric and lead day.

When `TEST_ALL_MODELS = True`, each configuration also includes an
**Análise de transferência entre clusters**. It is explicitly separated from
the honest same-cluster result:

- `test_predictions.csv` and `test_predictions_same_cluster.csv`: primary
  same-cluster predictions used by metrics and ordinary plots.
- `test_predictions_oracle_selection.csv`: post-hoc oracle prediction for
  comparison only.
- `oracle_model_selection_matrix.csv`: rows are assigned test clusters and
  columns are oracle-selected LSTMs.
- `oracle_cluster_routing_summary.csv`: per assigned test cluster, counts how
  often the oracle leaves the assigned LSTM and how much MAE/RMSE would improve.
- `oracle_cluster_pair_summary.csv`: per `assigned cluster -> oracle LSTM`
  pair, quantifies frequency and average error gain.
- `oracle_vs_same_cluster_summary.txt`: concise interpretation of the routing
  diagnostic.
- `prediction_overview_same_cluster/`: primary prediction plot.
- `oracle_model_selection_diagnostics/`: explicitly labelled oracle plots,
  including the transfer matrix, switch-rate plot, same-vs-oracle MAE by
  assigned cluster, and per-window error-improvement distribution.
- `oracle_model/`: the same complete plot tree generated for the ordinary
  result, but using the per-window oracle-selected prediction. This folder is
  created only when `TEST_ALL_MODELS = True`; its cluster groupings remain the
  original assignments so they can be compared directly with the main plots.

The existing detailed files remain available:

- `test_model_comparison.csv`: errors for every test-sample/model-cluster
  pairing.
- `test_model_selection.csv`: the selected model for each test sample and
  metric.
- `test_model_metric_summary.csv`: aggregate metrics for each metric-specific
  sample-level selection strategy.
- `test_model_selection_report.txt`: a readable comparison between the
  original same-cluster strategy and the selected-model strategy, including a
  paired bootstrap interval for squared-error improvement.

The model selection is performed on the test split at sample level, so this
analysis should be read as an oracle-style diagnostic for cross-cluster
transfer and not as an unbiased estimate of future performance.

Because the selection unit is one scalar target, MSE, RMSE, MAE, and MAPE
usually choose the same model for a sample: for a fixed observed precipitation
value, they all rank candidate predictions by closeness to the actual value.
RMSLE can differ because it ranks closeness after applying the logarithmic
transform.

Each configuration folder also gets `experiment_report.tex`. Its
`Configuration` section includes the selected clustering feature scaler,
clustering precipitation scaler, LSTM feature scaler, LSTM precipitation
scaler, and target scale for that run. Predictions are inverse-transformed to
millimeters before metrics and plots. When all-model test selection is enabled,
this report includes an `Análise de transferência entre clusters` section that
labels the result as oracle-only and shows the routing matrix. If a local LaTeX
compiler is available, the pipeline also writes `experiment_report.pdf` using
the shared MiKTeX state folder at `outputs/.miktex`; if PDF compilation fails,
`experiment_report_compile.log` is saved for troubleshooting.

Each configuration also groups generated images by purpose. General
same-cluster prediction plots go under `prediction_overview_same_cluster/`,
while oracle-only transfer plots go under
`oracle_model_selection_diagnostics/`. When enabled, `oracle_model/` mirrors
the complete normal plot structure with oracle-selected predictions. Split
time-series plots go under
`prediction_timeseries_splits/lead_day_XX/` with four sequential test plots per
forecast lead day and date-formatted x-axis labels from the source dataset,
residual/error plots under `residual_diagnostics/`, cluster diagnostics under
`cluster_diagnostics/`, and training curves under
`model_fit/`. The cluster diagnostics include `08_silhouette_analysis.png` and
`silhouette_scores.csv`, computed from the same split feature matrices and
cluster labels used by the current pipeline. The
`06_cluster_distribution.png` diagnostic compares train, validation, and test
sample counts and displays `n_train` plus `ceil(n_train / batch_size)` for each
cluster; exact values are also written to
`cluster_training_batch_statistics.csv`. Existing per-cluster collections
remain in folders such as
`cluster_precipitation_histograms/`,
`cluster_prediction_histograms/`, `cluster_prediction_timeseries/`, and
`cluster_prediction_scatter/`. The `cluster_prediction_timeseries/` plots use
the final forecast-horizon target date on the x-axis, formatted as
`dd/mm/YYYY`.
Normal LSTM runs also create `train_performance/` inside each configuration.
It mirrors `cluster_prediction_histograms/`, `cluster_prediction_timeseries/`,
`cluster_prediction_scatter/`, and
`prediction_timeseries_splits/lead_day_XX/` using training targets and
same-cluster training predictions. The underlying final-horizon and per-lead
values are exported to `train_performance/train_predictions.csv`. This folder
is intentionally absent when `RUN_ONLY_CLUSTER = True`.
The configuration root also contains `cluster_timeline.png`, an XY plot that
shows the assigned cluster of every window in chronological split order:
training, validation, and test.

Each configuration also saves
`input_forecast_horizon_precipitation_by_cluster.csv`, which assigns the
configured horizon target to every input window and its cluster. The legacy
`input_next_day_precipitation_by_cluster.csv` is still written for older
analysis scripts. Both files include `current_window_precipitation_mm`,
`forecast_horizon_precipitation_mm`, and `target_minus_current_mm`.

Forecast-horizon diagnostics are saved under
`forecast_horizon_diagnostics/`. They compare the precipitation on the final
input-window day with the target at `FORECAST_HORIZON`, include a persistence
baseline that uses current precipitation as the forecast, write CSV/text
summaries, and add the same section to `experiment_report.tex`.

The same folder also includes lead-day diagnostics for the test split. The
pipeline compares each model output D+1, D+2, ..., D+`FORECAST_HORIZON` with
the matching real precipitation for each test window, writes
`test_prediction_by_lead_day.csv` and
`test_prediction_metrics_by_lead_day.csv`, and saves both a combined
true-vs-predicted grid and one true-vs-predicted plot per lead day.

The automatic report includes a `Cluster Prediction Time Series` section with
one out-of-sample performance figure per cluster. Each figure contains:

- actual and predicted test precipitation in original chronological order;
- a residual time series (`actual - predicted`);
- cluster-level sample count, RMSE, MAE, and R2;
- original window indices on the x-axis.

Because the test split is sparse along the original timeline, gaps larger than
ten windows are shortened for readability and marked with vertical dotted
lines. The original indices remain on the ticks, so this compression does not
change sample order or metric calculations.

The automatic report also includes a `Cluster Prediction Scatter` section with
one test actual-versus-predicted scatter plot per cluster. Test samples are red
x markers and each plot keeps its legend.

Each configuration also writes `model_fit/01_training_history_cluster_<cluster>.png`.
This image is a 2x2 panel with per-epoch LOSS, MSE, MAE, and R² curves for the
training and validation histories.
