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
5. Fit the selected normalizer (`standard` or `minmax`) and PCA on training
   rows/windows only, then transform validation and test with those training
   transforms.
6. Cluster training windows with K-means, spectral, or manual rain clustering,
   calculate training-cluster centroids, then assign validation and test
   windows to the nearest existing centroid.
7. Use precipitation at the configured forecast horizon as the target inside
   each split.
8. Train one LSTM per training cluster.
9. Keep train and validation predictions tied to each sample's own cluster
   model.
10. Evaluate test samples with either their own cluster model only or, when
   `TEST_ALL_MODELS = True`, every trained cluster model with per-metric sample
   selection.
11. Save metrics, reports, plots, predictions, and LaTeX tables.

## Configuration

Change experiment variables in `run_experiment.py`:

- station and data: `STATE`, `STATION_ID`
- clustering sweep: `WINDOW_SIZES`, `N_CLUSTERS_LIST`,
  `CLUSTERING_ALGORITHM`, `FORECAST_HORIZON`, `MANUAL_ZERO_TOLERANCE`,
  `SIGMA_MODE`, `N_SIGMA_VALUES`, `MANUAL_SIGMA_VALUES`, `USE_ALL_FEATURES`
- normalization: `NORMALIZE`, `SCALER_TYPE`; supported scaler values are
  `"standard"` and `"minmax"`
- test evaluation mode: `TEST_ALL_MODELS`
- exported table metrics: `QUANTITATIVE_METRICS`
- model hyperparameters: `LSTM_UNITS`, `LSTM_UNITS_2`, `DROPOUT_RATE`,
  `LEARNING_RATE`
- LSTM loss: `LSTM_LOSS_FUNCTION`, `LOSS_QUANTILES`,
  `LOSS_QUANTILE_WEIGHTS`. Use `"quantile_weighted_mse"` to calculate
  cluster-specific precipitation thresholds from training-target quantiles in
  millimeters and weight rarer intensity bins automatically.
- training settings: `EPOCHS`, `BATCH_SIZE`, `EARLY_STOPPING`, `PATIENCE`,
  `VERBOSE_TRAINING`, `SHOW_CONSOLE_INFO`
- data split: `TRAIN_RATIO`, `VAL_RATIO`, `RANDOM_STATE`
- output and plot styling settings: `OUTPUT_CONFIG`, with details in
  `config_output.yaml`

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
`FORECAST_HORIZON = 1` for next-day rain or use a larger positive integer for
a later horizon. `MANUAL_ZERO_TOLERANCE` controls the maximum precipitation
treated as zero.

Forecast-horizon precipitation alignment is handled by
`methods.tools.precipitation_utils`. The LSTM pipeline uses
`precipitation_targets` separately for the train, validation, and test
dataframes so the final windows in one split do not use target rows from the
next split.

Set `SHOW_CONSOLE_INFO = False` to hide pipeline progress messages and Keras
training output. The root `run_experiments.py` launcher has the same setting and
passes it to this runner through `LSTM_CLUSTER_SHOW_CONSOLE_INFO`.

Set `TEST_ALL_MODELS = False` to use the original test behavior where each
test sample is predicted only by the LSTM trained on its own cluster. Set
`TEST_ALL_MODELS = True` to evaluate every test sample with every trained LSTM,
choose the best model per sample and metric, and save the extra model-selection
diagnostics.

## Outputs

Each run creates a sweep folder under `outputs/`, usually named like:

```text
outputs/lstm_cluster_sweep_<STATE>_<STATION>_<timestamp>/
```

The sweep folder contains summary CSV/text files and LaTeX tables. Each
configuration subfolder contains run metrics, predictions, reports, and plots.
Output writing is handled by `data.lstm_outputs`.

When `TEST_ALL_MODELS = True`, each configuration also includes cross-cluster
test model selection artifacts:

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
report should be read as an oracle-style diagnostic for cross-cluster transfer
and not as an unbiased estimate of future performance.

Because the selection unit is one scalar target, MSE, RMSE, MAE, and MAPE
usually choose the same model for a sample: for a fixed observed precipitation
value, they all rank candidate predictions by closeness to the actual value.
RMSLE can differ because it ranks closeness after applying the logarithmic
transform.

Each configuration folder also gets `experiment_report.tex`. When all-model
test selection is enabled, this report includes a compact `Test Model
Selection` section showing changed sample counts and metric improvements. If a
local LaTeX compiler is available, the pipeline also writes
`experiment_report.pdf`; if PDF compilation fails,
`experiment_report_compile.log` is saved for troubleshooting.

Each configuration also groups generated images by purpose. General prediction
plots go under `prediction_overview/`, split time-series plots under
`prediction_timeseries_splits/`, residual/error plots under
`residual_diagnostics/`, cluster diagnostics under `cluster_diagnostics/`, and
training curves under `model_fit/`. Existing per-cluster collections remain in
folders such as `cluster_precipitation_histograms/`,
`cluster_prediction_histograms/`, `cluster_prediction_timeseries/`, and
`cluster_prediction_scatter/`.

Each configuration also saves
`input_forecast_horizon_precipitation_by_cluster.csv`, which assigns the
configured horizon target to every input window and its cluster. The legacy
`input_next_day_precipitation_by_cluster.csv` is still written for older
analysis scripts. Both files include `current_window_precipitation_mm`,
`forecast_horizon_precipitation_mm`, and `target_minus_current_mm`.

Forecast-horizon diagnostics are saved under
`forecast_horizon_diagnostics/`. They compare the precipitation on the final
input-window day with the target at `FORECAST_HORIZON`, include a persistence
baseline that uses current precipitation as the forecast, and add the same
section to `experiment_report.tex`.

The same folder also includes lead-day diagnostics for the test split. The
pipeline compares the model prediction with the real precipitation observed at
D+1, D+2, ..., D+`FORECAST_HORIZON` for each test window, writes
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
