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
- `config_output.yaml`: output folder settings, including root path, optional
  fixed sweep name, generated name prefix, and timestamp format.
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
5. Cluster windows with K-means or spectral clustering.
6. Use the day after each window as the precipitation target.
7. Split samples into train, validation, and test sets.
8. Train one LSTM per training cluster.
9. Merge cluster-specific predictions.
10. Save metrics, reports, plots, predictions, and LaTeX tables.

## Configuration

Change experiment variables in `run_experiment.py`:

- station and data: `STATE`, `STATION_ID`
- clustering sweep: `WINDOW_SIZES`, `N_CLUSTERS_LIST`,
  `CLUSTERING_ALGORITHM`, `N_SIGMA_VALUES`, `USE_ALL_FEATURES`
- exported table metrics: `QUANTITATIVE_METRICS`
- model hyperparameters: `LSTM_UNITS`, `LSTM_UNITS_2`, `DROPOUT_RATE`,
  `LEARNING_RATE`
- training settings: `EPOCHS`, `BATCH_SIZE`, `EARLY_STOPPING`, `PATIENCE`,
  `VERBOSE_TRAINING`, `SHOW_CONSOLE_INFO`
- data split: `TRAIN_RATIO`, `VAL_RATIO`, `RANDOM_STATE`
- output settings: `OUTPUT_CONFIG`, with details in `config_output.yaml`

Change output naming in `config_output.yaml`.

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
