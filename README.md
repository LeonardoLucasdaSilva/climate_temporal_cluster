# Climate Temporal Cluster

This project studies Brazilian INMET daily climate data with a temporal
clustering workflow and cluster-specific LSTM precipitation models.

The main experiment is configured in `methods.lstm_cluster.run_experiment` and
executed by `methods.lstm_cluster.pipeline`. It first splits daily station data
chronologically into train, validation, and test dataframes, builds sliding
windows inside each split, clusters training windows, trains one LSTM model per
cluster, tests each sample with every trained LSTM before selecting the best
test-time model per metric, and writes metrics, predictions, plots, and
LaTeX-ready summary tables.

## Quick Start

Use Python 3.12. TensorFlow is installed in the project `.venv`, and the VS Code
settings point the play button at that environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Run the main experiment from the project root:

```powershell
lstm-cluster
```

Run the ARMA baseline from the project root:

```powershell
python run_arma.py
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Project Layout

```text
climate_temporal_cluster/
|-- data/                         # Raw INMET data files
|-- experiments/
|   |-- create_beamer_report.py      # Build Beamer decks from saved run plots
|   |-- temporary_experiments/     # Older scripts kept for review
|   `-- experiments.md
|-- outputs/                       # Generated experiment outputs
|-- run_arma.py                    # Root launcher for the ARMA baseline
|-- src/
|   |-- config.py                  # Project paths and output config helpers
|   |-- data/
|   |   |-- load_data.py           # INMET CSV loading
|   |   |-- clean_data.py          # Data cleaning helpers
|   |   |-- arma_outputs.py        # ARMA baseline output writers
|   |   `-- visualize_data.py      # Starter visualization module
|   |-- methods/
|   |   |-- arma/
|   |   |   |-- run_arma.py        # ARMA baseline runner
|   |   |   |-- arma.md            # Folder documentation
|   |   |   `-- pipeline.py        # ARMA fitting and rolling forecasts
|   |   |-- cluster/
|   |   |   |-- manual.py          # Horizon-rain-guided clustering
|   |   |   |-- ng.py              # Spectral clustering implementation
|   |   |   `-- cluster_pipeline.py
|   |   |-- lstm_cluster/
|   |   |   |-- run_experiment.py  # Main LSTM+Cluster runner
|   |   |   |-- config_output.yaml # Output and plot styling settings
|   |   |   |-- console.py         # Console output helpers
|   |   |   |-- lstm_cluster.md    # Folder documentation
|   |   |   |-- report.py          # Per-configuration LaTeX report writer
|   |   |   `-- pipeline.py        # Pipeline functions
|   |   `-- tools/
|   |       |-- sliding_windows.py
|   |       |-- sigma_choosing.py
|   |       `-- dimensionality_reduction_tools.py
|   |-- models/
|   |   `-- lstm.py                # LSTM model implementation
|   `-- evaluation/
|       |-- metrics.py             # Regression metrics and reports
|       `-- evaluation_plot_tools.py
|-- tests/
|-- requirements.txt
|-- pyproject.toml
`-- README.md
```

## LSTM+Cluster Pipeline

The experiment is configured at the top of
`src/methods/lstm_cluster/run_experiment.py`. Output naming and shared figure
styling live in `src/methods/lstm_cluster/config_output.yaml`.

```python
STATE = "RS"
STATION_ID = "A801"
WINDOW_SIZES = [8, 12, 16, 20, 24, 28]
N_CLUSTERS_LIST = [3, 4, 5]
CLUSTERING_ALGORITHM = "spectral"  # "kmeans", "spectral", or "manual"
FORECAST_HORIZON = 1
N_SIGMA_VALUES = 5
TEST_ALL_MODELS = True
SHOW_CONSOLE_INFO = True
```

For each configuration, the experiment runs these stages:

1. Load daily station data.
2. Select numeric climate features.
3. Split the daily dataframe chronologically into train, validation, and test
   blocks.
4. Build sliding windows independently inside each dataframe split.
5. Fit the selected covariate normalizer (`SCALER_TYPE`) and, when enabled, the
   precipitation normalizer (`PRECIPITATION_SCALER`) on training rows/windows
   only, then transform validation and test with the training transforms.
6. Cluster training windows with K-means, spectral, or manual rain clustering,
   calculate training-cluster centroids, then assign validation and test
   windows to the nearest existing centroid.
7. Create one precipitation target column per lead day from D+1 through the
   configured forecast horizon inside each split.
8. When `PRECIPITATION_SCALER` is set, normalize the LSTM target matrix, train
   one LSTM model per training cluster with one output unit per lead day, then
   inverse-transform predictions before metrics and plots. With
   `PRECIPITATION_SCALER = None`, precipitation features and targets stay in
   millimeters.
9. Predict train and validation precipitation with each sample's own cluster
   model.
10. Evaluate test samples with either their own cluster model only or, when
   `TEST_ALL_MODELS = True`, every trained cluster model with per-metric sample
   selection.
11. Save metrics, reports, predictions, plots, and LaTeX tables.

## ARMA Baseline

The ARMA baseline is configured at the top of
`src/methods/arma/run_arma.py` and can be launched with either
`python run_arma.py` or the installed `arma-baseline` command.

```python
STATE = "RS"
STATION_ID = "A801"
WINDOW_SIZES = [5, 10, 15]
FORECAST_HORIZON = 5
ARMA_ORDERS = [(1, 0), (2, 1), (5, 1)]
```

Each ARMA model is fit as `ARIMA(order=(p, 0, q))` with `statsmodels` on the
training precipitation series only. Validation and test forecasts keep those
parameters fixed and condition each prediction on observations available up to
the forecast origin. `WINDOW_SIZES` is used for target alignment with the LSTM
sliding-window convention, so the ARMA plots use the same D+1 through
D+`FORECAST_HORIZON` lead-day interpretation.

ARMA outputs are saved under:

```text
outputs/dd_mm_yy/ARMA/arma_sweep_<STATE>_<STATION>_<timestamp>/
```

Each configuration folder includes `metrics_summary.csv`,
`test_predictions.csv`, `summary.txt`, `evaluation_report.txt`,
`arma_model_summary.txt`, `prediction_overview/`,
`prediction_timeseries_splits/lead_day_XX/`, `residual_diagnostics/`,
`model_fit/`, and `forecast_horizon_diagnostics/` with
`test_prediction_by_lead_day.csv`,
`test_prediction_metrics_by_lead_day.csv`, lead-day error curves, and
true-vs-predicted plots.

## Data Loading

The project loads one INMET station at a time from `data/inmet`.

```python
from config import DATA_ROOT
from data.load_data import load_station_daily_data

df = load_station_daily_data(
    state="RS",
    station_id="A801",
    data_root=DATA_ROOT,
)
```

The loader returns a daily dataframe with `Data` plus numeric weather columns
such as temperature, humidity, pressure, wind, radiation, and
`PRECIPITACAO_TOTAL`.

Implementation:

- `src/data/load_data.py`
- `src/data/clean_data.py`
- `src/config.py`

## Sliding Windows

Sliding windows convert daily rows into temporal samples. Each sample contains
`window_size` consecutive days and all selected feature columns.

```python
from methods.tools.sliding_windows import create_windows

windows, (scaler, pca) = create_windows(
    df,
    window_size=20,
    columns=["PRECIPITACAO_TOTAL", "TEMPERATURA_MAXIMA"],
    normalize=True,
    variance_threshold=0.90,
)
```

Without PCA, `windows` has shape:

```text
(n_windows, window_size, n_features)
```

With PCA, `windows` becomes a 2D matrix:

```text
(n_windows, n_components)
```

The standalone clustering helpers can use `create_cluster_feature_matrix` to
produce both the original window output and a 2D `windows_flat` matrix. The
LSTM+Cluster experiment instead calls `create_window_split_data`, which first
splits the daily dataframe and then builds split-local windows to avoid sharing
raw observations across train, validation, and test.

```python
from methods.cluster.cluster_pipeline import create_cluster_feature_matrix

windows, windows_flat, scaler, pca, feature_columns = create_cluster_feature_matrix(
    df,
    window_size=20,
    normalize=True,
    variance_threshold=0.90,
)
```

## Sigma Selection

Spectral clustering uses a Gaussian affinity matrix and needs a bandwidth
parameter called `sigma`. The experiment can generate candidate sigma values
from pairwise window distances:

```python
from methods.tools.sigma_choosing import calculate_sigma_values

sigmas = calculate_sigma_values(df, n_values=5)
```

The sigma logic lives in `src/methods/tools/sigma_choosing.py`.
It contains:

- `euclidian_distances`
- `take_sigma`
- `sigma_values_from_distance_distribution`
- `calculate_sigma_values`

## Clustering

Cluster dispatch lives in
`src/methods/cluster/cluster_pipeline.py`.

```python
from methods.cluster.cluster_pipeline import cluster_feature_matrix

labels = cluster_feature_matrix(
    windows_flat,
    n_clusters=3,
    algorithm="spectral",
    sigma=1.0,
    random_state=42,
)
```

Supported algorithms:

- `kmeans`: uses `sklearn.cluster.KMeans`
- `spectral`: uses the local implementation in
  `src/methods/cluster/ng.py`

The `methods.cluster.manual` module provides horizon-rain-guided clustering
through the same `cluster_feature_matrix` dispatcher. Cluster `0` represents
known zero-rain horizons, while known positive horizons are split into `k - 1`
ordered groups from lower to heavier rain. Missing horizons are assigned to the
nearest learned window-feature centroid.

```python
from methods.cluster.cluster_pipeline import cluster_feature_matrix
from methods.tools.precipitation_utils import horizon_precipitation

horizon_rain = horizon_precipitation(df, window_size=20, horizon=1)
labels = cluster_feature_matrix(
    windows_flat,
    n_clusters=3,
    algorithm="manual",
    horizon_rain=horizon_rain,
)
```

The `horizon` argument is general: `1` means the day after the window, and
larger values select later forecast days. See
`src/methods/cluster/cluster.md` for assumptions and the estimator interface.

The spectral implementation follows the Ng, Jordan, and Weiss style workflow:

1. Build a Gaussian affinity matrix.
2. Compute the normalized graph matrix.
3. Extract the top eigenvector embedding.
4. Row-normalize the embedding.
5. Run K-means on the embedding.

## Next-Day Target

For each window, the target is the precipitation on the next day after the
window ends:

```text
window: days t ... t + window_size - 1
target: day t + window_size PRECIPITACAO_TOTAL
```

This turns temporal windows into supervised samples for precipitation
prediction.

## Cluster-Specific LSTM Models

After clustering, the experiment trains one LSTM model per cluster.

Implementation:

- `src/models/lstm.py`
- class: `LSTMPrecipitationPredictor`

The experiment reshapes each flattened window into one LSTM timestep:

```python
X_lstm = X.reshape(X.shape[0], 1, X.shape[1])
```

For each cluster, the training loop:

1. Selects train/validation/test samples assigned to that cluster.
2. Builds an LSTM model with two LSTM layers, dropout, and dense layers.
3. Trains with optional early stopping.
4. Writes train and validation predictions back into aggregate arrays.
5. Evaluates test samples with their own cluster model, or with every trained
   cluster model when `TEST_ALL_MODELS = True`.
6. In all-model mode, selects the best model per sample and per metric; the
   main prediction output uses the RMSE/MSE-equivalent squared-error selection.
7. Calculates selected-model cluster-level test metrics from those sample-level
   predictions.

The model predicts one value per lead day from D+1 through the configured
forecast horizon, in millimeters. The final D+`FORECAST_HORIZON` output remains
the scalar target used by legacy summary metrics.

Set `TEST_ALL_MODELS = False` in `run_experiment.py` to keep the original
behavior where each test sample is predicted only by the LSTM trained on its
own cluster. Set it to `True` to enable the all-model diagnostic selection and
write the extra model-selection reports.

## Evaluation Outputs

Each run writes into a timestamped folder under the current daily output folder,
for example:

```text
outputs/dd_mm_yy/lstm_cluster_sweep_RS_A801_YYYYMMDD_HHMMSS/
```

The daily folder uses the `date_folder_format` setting from
`src/methods/lstm_cluster/config_output.yaml`; by default it is `"%d_%m_%y"`.

Sweep-level files:

- `sweep_results.csv`
- `sweep_summary.txt`
- `overleaf_table.txt`
- `overleaf_cluster_metric_tables.txt`

Each configuration folder contains:

- `metrics_summary.csv`
- `cluster_model_metrics.csv`
- `test_model_comparison.csv`, when `TEST_ALL_MODELS = True`
- `test_model_selection.csv`, when `TEST_ALL_MODELS = True`
- `test_model_metric_summary.csv`, when `TEST_ALL_MODELS = True`
- `test_model_selection_report.txt`, when `TEST_ALL_MODELS = True`
- `input_forecast_horizon_precipitation_by_cluster.csv`
- `input_next_day_precipitation_by_cluster.csv`
- `test_predictions.csv`
- `evaluation_report.txt`
- `summary.txt`
- `experiment_report.tex`
- `experiment_report.pdf`, when a local LaTeX compiler is available
- grouped prediction, residual, error, cluster-performance, and histogram plots
  under folders such as `prediction_overview/`,
  `prediction_timeseries_splits/lead_day_XX/`, `residual_diagnostics/`,
  `cluster_diagnostics/`, `model_fit/`, `cluster_precipitation_histograms/`,
  `cluster_prediction_histograms/`, `cluster_prediction_timeseries/`, and
  `cluster_prediction_scatter/`
- split time-series plots in `prediction_timeseries_splits/lead_day_XX/` use
  target dates from the source dataset on the x-axis, formatted as `dd/mm/YYYY`
- cluster silhouette diagnostics under `cluster_diagnostics/`, including
  `08_silhouette_analysis.png` and `silhouette_scores.csv`
- input-window forecast-horizon precipitation distribution plots by cluster under
  `input_precipitation_distribution_by_cluster/`
- current-window versus forecast-horizon target diagnostics and persistence
  baseline metrics under `forecast_horizon_diagnostics/`
- lead-day diagnostics comparing each D+k output with its matching real
  precipitation under `forecast_horizon_diagnostics/`, including
  `test_prediction_by_lead_day.csv`,
  `test_prediction_metrics_by_lead_day.csv`, a lead-day error curve, a
  true-vs-predicted grid for D+1 through `FORECAST_HORIZON`, and individual
  true-vs-predicted plots under
  `forecast_horizon_diagnostics/true_vs_predicted_by_lead_day/`
- per-cluster test performance time series with actual values, predictions,
  residuals, cluster metrics, and target dates formatted as `dd/mm/YYYY`
- per-cluster test actual-versus-predicted scatter plots with red x markers and
  plot legends

The `Configuration` section of `experiment_report.tex` and the compiled PDF
record the run's selected covariate scaler, precipitation scaler, and LSTM
target scale. Predictions are inverse-transformed to millimeters before metrics
and plots.

Metrics include:

- MSE
- RMSE
- MAE
- RMSLE
- R2
- MAPE
- zero-day and rainy-day metrics
- per-cluster metrics

When `TEST_ALL_MODELS = True`, the test model selection report compares the
original same-cluster test prediction strategy against the selected-model
strategy. It includes the model chosen for each test sample and metric,
same-cluster versus selected aggregate metrics, and a paired bootstrap
confidence interval for the mean squared-error improvement. The automatic
`experiment_report.tex` also includes a compact `Test Model Selection` section.
Because the selection is made on the test set at sample level, the improvement
is descriptive and useful for diagnosing cross-cluster transfer rather than an
unbiased generalization estimate.

To create a slide deck from selected plots in one saved configuration folder,
open `experiments/create_beamer_report.py`, edit `RUN_DIR` and
`SELECTED_PLOTS`, then run:

```powershell
python experiments\create_beamer_report.py
```

The runner writes `beamer.tex` and compiles `beamer.pdf` by default. Set
`COMPILE_PDF = False` in the script if only the TeX source is needed.

The command-line form is still available when needed:

```powershell
python experiments\create_beamer_report.py outputs\dd_mm_yy\lstm_cluster_sweep_RS_A801_YYYYMMDD_HHMMSS\RS_A801_w15_k03_kmeans_sigma_na --plots prediction_overview\02_predictions_vs_actual.png cluster_prediction_scatter\*.png residual_diagnostics\*.png
```

Use `--list-plots` on the same run folder to print all selectable plot paths.

Since all-model selection is still made on the final scalar
D+`FORECAST_HORIZON` output, MSE, RMSE, MAE, and MAPE usually choose the same
model: for a fixed observed precipitation value, all of them rank candidate
predictions by closeness to the actual value. RMSLE can differ because it ranks
closeness after a logarithmic transform.

Plot helpers live in:

```text
src/evaluation/evaluation_plot_tools.py
```

Per-configuration report generation lives in:

```text
src/methods/lstm_cluster/report.py
```

## Running a Smaller Experiment

The default sweep can be expensive. To test quickly, edit
`src/methods/lstm_cluster/run_experiment.py`:

```python
WINDOW_SIZES = [8]
N_CLUSTERS_LIST = [3]
NORMALIZE = True
SCALER_TYPE = "standard"  # covariates: "standard" or "minmax"
PRECIPITATION_SCALER = None  # None keeps PRECIPITACAO_TOTAL and targets in mm
LSTM_LOSS_FUNCTION = "quantile_weighted_mse"
LOSS_QUANTILES = [0.5, 0.75, 0.9, 0.95]
LOSS_QUANTILE_WEIGHTS = "auto"
N_SIGMA_VALUES = 1
EPOCHS = 2
VERBOSE_TRAINING = 1
SHOW_CONSOLE_INFO = True
```

Set `LSTM_LOSS_FUNCTION = "mean_squared_error"` to keep the standard MSE
training objective. Set it to `"quantile_weighted_mse"` to weight each
cluster-specific LSTM by precipitation quantile bins calculated from that
cluster's own training rain targets across all lead days, in the active target
scale. With
`LOSS_QUANTILE_WEIGHTS = "auto"`, rarer rain-intensity bins receive larger
weights automatically.

To silence pipeline progress messages and model-training output, set
`SHOW_CONSOLE_INFO = False` in `run_experiment.py`. If using the root launcher,
the same behavior can be controlled by `SHOW_CONSOLE_INFO` in
`run_experiments.py`.

Then run:

```powershell
lstm-cluster
```

## CLI Clustering Pipeline

For a simpler clustering-only workflow, use the console command configured in
`pyproject.toml`:

```powershell
climate-cluster --state RS --station-id A801 --window-size 20 --clusters 3 --sigma 1.0
```

Or call the function directly:

```python
from methods.cluster.cluster_pipeline import run_clustering_pipeline

results = run_clustering_pipeline(
    state="RS",
    station_id="A801",
    window_size=20,
    n_clusters=3,
    sigma=1.0,
)
```

## Development Notes

- Keep `.venv/`, `.matplotlib_cache/`, generated reports, and outputs out of
  git.
- Commit source files, docs, tests, `requirements.txt`, and `pyproject.toml`.
- Run tests before committing:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Main Modules

| Purpose | Module |
| --- | --- |
| Load station data | `data.load_data` |
| Clean/load data internals | `data` |
| Sliding windows | `methods.tools.sliding_windows` |
| Sigma selection | `methods.tools.sigma_choosing` |
| Cluster feature matrix and dispatch | `methods.cluster.cluster_pipeline` |
| Horizon-rain-guided clustering | `methods.cluster.manual` |
| Spectral clustering | `methods.cluster.ng` |
| LSTM model | `models.lstm` |
| Metrics and reports | `evaluation.metrics` |
| Diagnostic plots | `evaluation.evaluation_plot_tools` |
| Main experiment runner | `methods.lstm_cluster.run_experiment` |
| Main experiment pipeline | `methods.lstm_cluster.pipeline` |
