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
5. Fit the selected covariate normalizer (`SCALER_TYPE`), optional
   precipitation normalizer (`PRECIPITATION_SCALER`), and PCA on training
   rows/windows only, then transform validation and test with those training
   transforms. With `PCA_FOR_CLUSTERING_ONLY = True`, retain the pre-PCA
   flattened windows for LSTM inputs and use PCA coordinates only for cluster
   fitting and held-out cluster assignment.
6. Cluster training windows with K-means, spectral, or manual rain clustering,
   calculate training-cluster centroids, then assign validation and test
   windows to the nearest existing centroid.
7. Build one target column for each lead day from D+1 through the configured
   forecast horizon inside each split. The final D+`FORECAST_HORIZON` column is
   still kept as the scalar target for legacy metrics and plots.
8. When `PRECIPITATION_SCALER` is set, normalize the LSTM target matrix before
   training. With `PRECIPITATION_SCALER = None`, precipitation features and
   targets stay in millimeters. Each LSTM still has one output unit per lead
   day, so the loss is optimized across the full D+1..D+`FORECAST_HORIZON`
   target matrix.
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
  `CLUSTERING_ALGORITHM`, `FORECAST_HORIZON`, `MANUAL_ZERO_TOLERANCE`,
  `SIGMA_MODE`, `N_SIGMA_VALUES`, `MANUAL_SIGMA_VALUES`, `USE_ALL_FEATURES`
- dimensionality reduction: `PCA_VARIANCE_THRESHOLD` enables PCA and
  `PCA_FOR_CLUSTERING_ONLY` limits it to clustering while preserving the
  original flattened-window dimensionality for LSTM inputs
- normalization: `NORMALIZE`, `SCALER_TYPE` for covariates and
  `PRECIPITATION_SCALER` for `PRECIPITACAO_TOTAL` plus the LSTM target;
  supported values are `"standard"`, `"minmax"`, and `None`
- test evaluation mode: `TEST_ALL_MODELS`
- exported table metrics: `QUANTITATIVE_METRICS`
- model hyperparameters: `LSTM_UNITS`, `LSTM_UNITS_2`, `DROPOUT_RATE`,
  `LEARNING_RATE`, `WEIGHT_DECAY`; the optimizer is AdamW and
  `WEIGHT_DECAY` controls its decoupled parameter decay
- LSTM loss: `LSTM_LOSS_FUNCTION`, `LOSS_QUANTILES`,
  `LOSS_QUANTILE_WEIGHTS`. Use `"quantile_weighted_mse"` to calculate
  cluster-specific precipitation thresholds from training-target quantiles in
  the active target scale and weight rarer intensity bins automatically.
- training settings: `EPOCHS`, `BATCH_SIZE`, `EARLY_STOPPING`, `PATIENCE`,
  `VERBOSE_TRAINING`, `SHOW_CONSOLE_INFO`
- data split: `TRAIN_RATIO`, `VAL_RATIO`, `RANDOM_STATE`
- output and plot styling settings: `OUTPUT_CONFIG`, with details in
  `config_output.yaml`

### PCA modes

Configure PCA with `PCA_VARIANCE_THRESHOLD` and
`PCA_FOR_CLUSTERING_ONLY` in `run_experiment.py`.

To run without PCA, set the variance threshold to `None`. The clustering
algorithms and LSTM models will both receive the original flattened-window
features:

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
Manual clustering reserves label `0` for known zero-rain targets and splits
known positive targets into ordered lower-to-heavier rain groups. Set
`FORECAST_HORIZON = 1` for next-day rain or use a larger positive integer to
train the LSTM on every lead day from D+1 through that horizon.
`MANUAL_ZERO_TOLERANCE` controls the maximum precipitation treated as zero.

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
Output writing is handled by `data.lstm_outputs`.
The sweep-level `sweep_summary.txt` records the PCA variance threshold and PCA
mode (`disabled`, `clustering only`, or `clustering and LSTM`) for the run.

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
`Configuration` section includes the selected covariate scaler, precipitation
scaler, and target scale for that run. Predictions are inverse-transformed to
millimeters before metrics and plots. When all-model test selection is enabled,
this report includes an `Análise de transferência entre clusters` section that
labels the result as oracle-only and shows the routing matrix. If a local LaTeX
compiler is available, the pipeline also writes `experiment_report.pdf`; if PDF
compilation fails, `experiment_report_compile.log` is saved for troubleshooting.

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
