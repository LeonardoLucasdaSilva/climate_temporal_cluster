# LSTM Cluster

This folder contains the active LSTM-by-cluster precipitation experiment. The
experiment predicts next-day precipitation for one INMET station by clustering
weather windows first, then training one LSTM model per cluster.

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
3. Build sliding windows.
4. Reduce window features with PCA.
5. Cluster windows with K-means, spectral, or manual rain clustering.
6. Use precipitation at the configured forecast horizon as the target.
7. Split samples into train, validation, and test sets.
8. Train one LSTM per training cluster.
9. Merge cluster-specific predictions.
10. Save metrics, reports, plots, predictions, and LaTeX tables.

## Configuration

Change experiment variables in `run_experiment.py`:

- station and data: `STATE`, `STATION_ID`
- clustering sweep: `WINDOW_SIZES`, `N_CLUSTERS_LIST`,
  `CLUSTERING_ALGORITHM`, `FORECAST_HORIZON`, `MANUAL_ZERO_TOLERANCE`,
  `SIGMA_MODE`, `N_SIGMA_VALUES`, `MANUAL_SIGMA_VALUES`, `USE_ALL_FEATURES`
- exported table metrics: `QUANTITATIVE_METRICS`
- model hyperparameters: `LSTM_UNITS`, `LSTM_UNITS_2`, `DROPOUT_RATE`,
  `LEARNING_RATE`
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

Set `SHOW_CONSOLE_INFO = False` to hide pipeline progress messages and Keras
training output. The root `run_experiments.py` launcher has the same setting and
passes it to this runner through `LSTM_CLUSTER_SHOW_CONSOLE_INFO`.

## Outputs

Each run creates a sweep folder under `outputs/`, usually named like:

```text
outputs/lstm_cluster_sweep_<STATE>_<STATION>_<timestamp>/
```

The sweep folder contains summary CSV/text files and LaTeX tables. Each
configuration subfolder contains run metrics, predictions, reports, and plots.
Output writing is handled by `data.lstm_outputs`.

Each configuration folder also gets `experiment_report.tex`. If a local LaTeX
compiler is available, the pipeline also writes `experiment_report.pdf`; if PDF
compilation fails, `experiment_report_compile.log` is saved for troubleshooting.

Each configuration also saves `input_next_day_precipitation_by_cluster.csv`,
which assigns the next-day precipitation target to every input window and its
cluster. The matching horizontal histograms are saved as
`08_input_precipitation_distribution_by_cluster.png` and as one figure per
cluster under `input_precipitation_distribution_by_cluster/`.

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
