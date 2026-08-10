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

Run budgeted LSTM hyperparameter tuning:

```powershell
lstm-hyperparam-tuning
```

Run the ARMA baseline from the project root:

```powershell
python run_arma.py
```

Create a cross-experiment metric report:

```powershell
python experiments\create_meta_analysis_report.py <experiment_1> <experiment_2> --output outputs\meta_analysis.tex
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
|   |-- create_meta_analysis_report.py # Compare saved experiment metrics
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
|   |   |-- lstm_outputs.py        # Per-configuration LSTM output writers
|   |   |-- lstm_comparative_outputs.py # Sweep comparison writer
|   |   |-- hyperparam_tuning_outputs.py # Tuning checkpoints and best-trial writer
|   |   |-- meta_analysis_report.py # Cross-experiment LaTeX metric tables
|   |   `-- visualize_data.py      # Starter visualization module
|   |-- methods/
|   |   |-- arma/
|   |   |   |-- run_arma.py        # ARMA baseline runner
|   |   |   |-- arma.md            # Folder documentation
|   |   |   `-- pipeline.py        # ARMA fitting and rolling forecasts
|   |   |-- cluster/
|   |   |   |-- dtw.py             # Multivariate Dynamic Time Warping distances
|   |   |   |-- automatic_sigma.py # Standalone automatic sigma selector
|   |   |   |-- eigengap.py         # Standalone eigengap analysis
|   |   |   |-- kshape.py           # K-Shape time-series clustering
|   |   |   |-- manual.py          # Horizon-rain-guided clustering
|   |   |   |-- ng.py              # Spectral clustering implementation
|   |   |   `-- cluster_pipeline.py
|   |   |-- lstm_cluster/
|   |   |   |-- run_experiment.py  # Main LSTM+Cluster runner
|   |   |   |-- hiperparam_tuning.py # Random/grid hyperparameter tuner
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
WINDOW_STRIDE = 1  # Days between consecutive window starts
N_CLUSTERS_LIST = [3, 4, 5]
PCA_VARIANCE_THRESHOLD = 0.90  # None disables PCA
PCA_FOR_CLUSTERING_ONLY = True
CLUSTERING_ALGORITHM = "spectral"  # or a list such as ["kshape", "kmeans"]
CLUSTER_ONLY_PRECIPITATION = False  # Use only the precipitation series to cluster windows
CLUSTER_DISSIMILARITY_METRIC = "euclidean"  # "euclidean" or "dtw"
MANUAL_CLUSTERING_METHOD = "legacy"  # "legacy" or "rain_level"
MANUAL_ZERO_TOLERANCE = 0.0  # Used only by legacy manual clustering
CLUSTER_ASSIGNMENT_METHOD = "centroid"  # "centroid" or "knn"
CLUSTER_ASSIGNMENT_NEIGHBORS = 5  # Used only by "knn"
FORECAST_HORIZON = 1
N_SIGMA_VALUES = 5
LEARNING_RATE: float | list[float] = 0.001
COMPARATIVE_RUN = True
PIVOT_PARAMETER = "window_size"  # aliases include "K", "lr", and "CLUSTERING_ALGORITHM"
TEST_ALL_MODELS = True
RUN_ONLY_CLUSTER = False  # Comparative plots require the complete LSTM pipeline
TRAIN_INFO = True  # Save train_performance/ diagnostics
SILHOUETTE_INFO = True  # Save cluster_diagnostics/08_silhouette_analysis.png
PLOT_CLUSTER_TIMESERIES = True  # Save per-cluster input precipitation series diagnostics
CLUSTER_TIMESERIES_PLOT_LIMIT = 100  # None saves every individual test-window series
PARALEL = True  # Parallel cluster-only configs or cluster LSTMs
REQUIRE_GPU = True  # Full LSTM runs fail unless TensorFlow detects a GPU
CREATE_REPORT = False  # Write .tex and skip PDF compilation for faster runs
WARM_UP = 25  # Epochs before early stopping starts counting
EARLY_STOPPING_METRIC = "loss"  # "loss", "mse", "mae", or "r2"
SHOW_CONSOLE_INFO = True
```

Set `CLUSTER_ONLY_PRECIPITATION = True` to cluster each window from its
`PRECIPITACAO_TOTAL` time series alone. This does not change the input features
used to train the cluster-specific LSTMs.

Set `PLOT_CLUSTER_TIMESERIES = False` to skip generating
`cluster_diagnostics/clusters_timeseries/` entirely. When enabled,
`CLUSTER_TIMESERIES_PLOT_LIMIT` bounds the total number of individual test
window precipitation series written there. The selected windows are distributed
across clusters; set it to `None` to save every available test-window series.

### PCA configuration modes

Set `PCA_VARIANCE_THRESHOLD` and `PCA_FOR_CLUSTERING_ONLY` in
`src/methods/lstm_cluster/run_experiment.py` according to the desired pipeline:

| Mode | `PCA_VARIANCE_THRESHOLD` | `PCA_FOR_CLUSTERING_ONLY` | Features used by the LSTM |
| --- | --- | --- | --- |
| No PCA | `None` | `False` (ignored) | Original temporal windows `(window_size, n_features)` |
| PCA only for clustering | A value between `0` and `1`, such as `0.90` | `True` | Original temporal windows `(window_size, n_features)` |

PCA is always fitted on training windows only. Validation and test windows use
the fitted training transform. PCA over flattened windows cannot feed the LSTM
because it removes the daily time axis; a complete LSTM run therefore requires
`PCA_FOR_CLUSTERING_ONLY = True` whenever PCA is enabled. See
`src/methods/lstm_cluster/lstm_cluster.md` for complete configuration examples.

When `RUN_ONLY_CLUSTER = True`, the run stops after train-only clustering and
held-out validation/test assignment. It writes cluster assignments, silhouette
and distribution diagnostics, and a cluster-only report; LSTM preprocessing,
training, predictions, and their output folders are skipped. The
`06_cluster_distribution.png` diagnostic still contains grouped training,
validation, and test bars for every cluster; only the LSTM workload table and
its batch-statistics CSV are omitted.

When `PARALEL = True`, execution depends on the selected mode. Cluster-only
sweeps dispatch independent configurations simultaneously, capped by CPU cores
and the number of configurations. Complete LSTM runs dispatch the trainable
cluster LSTMs within each configuration, capped by CPU cores and the number of
clusters. Results and sweep summaries retain the original configuration order.
Set `PARALEL = False` for serial execution.

When `REQUIRE_GPU = True`, complete LSTM runs require a TensorFlow-detected GPU
and place model build, training, prediction, and evaluation on `/GPU:0`. If no
GPU is visible to TensorFlow, the run raises an error instead of silently using
CPU. `RUN_ONLY_CLUSTER = True` does not require a GPU because no LSTM is
trained.

### Comparative sweep analysis

Set `COMPARATIVE_RUN = True` to create sweep-level plots after every test has
finished. `PIVOT_PARAMETER` selects the varied parameter shown on metric axes;
it accepts parameter fields and convenient aliases such as `"K"` for
`n_clusters`, `"lr"` for `learning_rate`, and `"CLUSTERING_ALGORITHM"` for
`clustering_algorithm`. The pivot must have at least two distinct values when
the sweep contains multiple tests. The grid varies `WINDOW_SIZES`,
`N_CLUSTERS_LIST`, `CLUSTERING_ALGORITHM`, and spectral sigma candidates. These
numeric training settings also accept either a scalar or a list: `LSTM_UNITS`,
`LSTM_UNITS_2`, `DROPOUT_RATE`, `LEARNING_RATE`, `WEIGHT_DECAY`, `EPOCHS`,
`BATCH_SIZE`, `PATIENCE`, and `WARM_UP`. `WARM_UP = 0` preserves the previous
behavior, where early stopping can count from the first epoch; larger values
delay early-stopping monitoring until that epoch. `EARLY_STOPPING_METRIC`
chooses the validation metric used by early stopping: `"loss"`, `"mse"`, and
`"mae"` are minimized, while `"r2"` is maximized. Set `LSTM_UNITS_2 = None` to
use a single LSTM layer with `LSTM_UNITS` units. Multiple lists form a
Cartesian product. To compare learning rates, configure:

```python
LEARNING_RATE = [0.0001, 0.0005, 0.001]
PIVOT_PARAMETER = "learning_rate"
```

To compare clustering algorithms, configure:

```python
CLUSTERING_ALGORITHM = ["kshape", "kmeans"]
PIVOT_PARAMETER = "CLUSTERING_ALGORITHM"
```

Keep the other sweep dimensions fixed when the goal is to attribute changes to
one hyperparameter. Every varied numeric training setting receives a stable
suffix in each configuration folder name. `COMPARATIVE_RUN` cannot be combined with
`RUN_ONLY_CLUSTER = True`, because predictions and LSTM histories do not exist
in cluster-only mode.

Comparative time series and scatter plots use same-cluster predictions and the
intersection of target dates shared by all tests. They never align different
window sizes by `window_index`. Metrics are recalculated on that common date
interval, and training histories are aggregated across cluster models using
each cluster's training-sample count as its weight. Each aggregate curve stops
at the final epoch shared by every contributing cluster, so early stopping does
not change the contributing cluster set partway through a line.

For each configuration, the experiment runs these stages:

1. Load daily station data.
2. Select numeric climate features.
3. Split the daily dataframe chronologically into train, validation, and test
   blocks.
4. Build sliding windows independently inside each dataframe split, retaining
   one start every `WINDOW_STRIDE` days. The default stride of `1` retains the
   previous daily-window behavior.
5. Fit the selected clustering normalizers
   (`CLUSTERING_FEATURE_NORMALIZE` and `CLUSTERING_PRECIPITATION_NORMALIZE`)
   and optional clustering PCA on training rows/windows only, then transform
   validation and test with the training transforms.
6. Cluster training windows with K-means, K-Shape, spectral, or manual rain
   clustering,
   then assign validation and test windows with `CLUSTER_ASSIGNMENT_METHOD`.
   K-Shape uses the original 3D temporal windows, z-normalized shape-based
   distance, and its learned shape centroids to assign held-out windows.
   `CLUSTER_DISSIMILARITY_METRIC="dtw"` makes spectral clustering build its
   affinity from multivariate Dynamic Time Warping distances and makes KNN use
   the same metric for held-out assignment.
   For manual runs, `MANUAL_CLUSTERING_METHOD="legacy"` preserves the
   horizon-target grouping, while `"rain_level"` bins the raw mean input-window
   precipitation into `K` equal-width training intervals.
   `"centroid"` uses the nearest training-cluster centroid, preserving the
   previous behavior. `"knn"` uses a majority vote among the
   `CLUSTER_ASSIGNMENT_NEIGHBORS` nearest labeled training windows.
   With `PCA_FOR_CLUSTERING_ONLY = True`, clustering uses PCA coordinates while
   each LSTM receives the retained pre-PCA temporal window tensor.
7. Create one precipitation target column per lead day from D+1 through the
   configured forecast horizon inside each split.
8. Rebuild the LSTM feature tensor as `(samples, window_size, n_features)`, fit
   `LSTM_FEATURE_NORMALIZE` and `LSTM_PRECIPITATION_NORMALIZE` on training
   rows/windows only, train one LSTM model per training cluster with one output
   unit per lead day, then inverse-transform predictions before metrics and
   plots. `LSTM_PRECIPITATION_TRANSFORM = True` applies `log1p` only to LSTM
   precipitation features and targets before normalization; clustering remains
   untransformed. Predictions are denormalized and then restored with
   `expm1`, so test metrics and plots stay in millimeters. With both
   `LSTM_PRECIPITATION_TRANSFORM = False` and
   `LSTM_PRECIPITATION_NORMALIZE = None`, precipitation features and targets
   stay in millimeters.
9. Predict train and validation precipitation with each sample's own cluster
   model.
10. Evaluate test samples with either their own cluster model only or, when
   `TEST_ALL_MODELS = True`, every trained cluster model with per-metric sample
   selection. The all-model path is an oracle diagnostic: it compares the LSTM
   selected by the assigned test cluster with the best LSTM for that same
   window after observing the target, then writes routing summaries and plots.
11. Save metrics, reports, predictions, plots, and LaTeX tables. When
    `TRAIN_INFO = True`, each normal LSTM configuration also writes
    `train_performance/`, containing the
    per-cluster training histograms, scatter plots, chronological time series,
    and `prediction_timeseries_splits/lead_day_XX/` plots for every forecast
    horizon. Set `TRAIN_INFO = False` to skip this folder. Cluster-only runs do
    not create it.

### Hyperparameter tuning

`src/methods/lstm_cluster/hiperparam_tuning.py` reuses all fixed settings from
`run_experiment.py` and initially tunes a configurable subset of
`WINDOW_SIZES`, `N_CLUSTERS_LIST`, `LSTM_UNITS`, `LSTM_UNITS_2`,
`DROPOUT_RATE`, `LEARNING_RATE`, and `WEIGHT_DECAY`. Remove a key from
`HYPERPARAMETER_SPACE` to keep that parameter fixed at the first base-runner
value.

The default random search evaluates up to `N_TRIALS` unique combinations with
a reproducible seed. Set `SEARCH_STRATEGY = "grid"` for Cartesian order or
`N_TRIALS = None` for the complete candidate space. The best trial is selected
only from the configured validation metric: R2 is maximized and the supported
error metrics are minimized. The normal pipeline still produces test outputs
for each trial, but those values do not participate in tuning selection.

The generated tuning folder contains `tuning_results.csv`,
`tuning_summary.txt`, `best_hyperparameters.json`, and an ordinary
single-configuration sweep folder for every `trial_XXXX`. Results are
checkpointed after each attempt; with `FAIL_FAST = False`, failed candidates
are recorded without cancelling the remaining search.

### Cluster dissimilarity metric

`CLUSTER_DISSIMILARITY_METRIC` accepts `"euclidean"` and `"dtw"`; the common
misspelling `"dwt"` is accepted as an alias for DTW. Euclidean preserves the
previous flattened-window behavior. DTW compares the scaled 3D windows and
uses one shared warping path across all weather variables. In the current
implementation DTW requires spectral or manual clustering, KNN held-out
assignment, and `PCA_VARIANCE_THRESHOLD = None`. The silhouette diagnostic and
automatic spectral sigma candidates use the selected metric as well. For
K-Shape, the silhouette diagnostic uses precomputed SBD distances on the
original temporal windows.

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
PARALEL = True  # Run independent window-size and ARMA-order configurations in parallel
```

Each ARMA model is fit as `ARIMA(order=(p, 0, q))` with `statsmodels` on the
training precipitation series only. Validation and test forecasts keep those
parameters fixed and condition each prediction on observations available up to
the forecast origin. `WINDOW_SIZES` is used for target alignment with the LSTM
sliding-window convention, so the ARMA plots use the same D+1 through
D+`FORECAST_HORIZON` lead-day interpretation.
With `PARALEL = True`, each independent `WINDOW_SIZE × ARMA(p, q)` configuration
runs in a separate process, capped by the number of available CPU cores. The
sweep summary records whether this parallel execution was active.

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

## Cross-Experiment Meta-Analysis

`experiments/create_meta_analysis_report.py` creates one `.tex` report from
multiple saved LSTM and/or ARMA experiments. Inputs may be sweep folders,
individual configuration folders, broader output folders, or supported metric
CSV files:

- `test_prediction_metrics_by_lead_day.csv`
- `metrics_summary.csv`
- `sweep_results.csv`
- `comparative_metrics.csv`

The default raw mode recursively discovers every physical run, including a
completed configuration that has not yet been aggregated into
`sweep_results.csv`. Overlapping input roots are deduplicated by the resolved
run folder, while identical `run_name` values from different experiment
folders remain separate runs. Pass a `comparative_analysis/` folder or its
`comparative_metrics.csv` directly to make a separate common-date-aligned
report; raw and aligned scopes cannot be mixed silently.
When more than one aligned comparative artifact is supplied, its lead-day
sets, common date intervals, and common test sample counts must be identical.

The report assigns one global number from `1` through the total number of
runs. Each metric table places runs in columns under a grouped `Run` header.
Overall tables use MSE, MAE, and R2; lead-day tables use RMSE, MAE, and R2.
Long comparisons are split into sequential column blocks without restarting
the run numbering, and the global best value for each metric is bold. A
registry maps every number back to its experiment, original run name, test
sample count, and forecast horizon. When lead-day CSVs are available, the
overall table uses the configured forecast horizon and the same layout is
generated separately for every available D+k.

Set `START_DATE` and `END_DATE` in the runner, or pass `--start-date` and
`--end-date`, to also generate Seaborn time-series comparison plots under
`meta_analysis_timeseries/` beside the `.tex` file. Each lead-day plot compares
the shared observed precipitation series with `Run 1`, `Run 2`, ... in the same
global order used by the registry, restricted to the common target dates inside
the selected period.

Edit `EXPERIMENT_PATHS`, `OUTPUT_PATH`, `RUNS_PER_TABLE`, `START_DATE`, and
`END_DATE` at the top of the runner and execute it without arguments, or use
the CLI:

```powershell
python experiments\create_meta_analysis_report.py `
  outputs\dd_mm_yy\lstm_cluster_sweep_RS_A801_<timestamp> `
  outputs\dd_mm_yy\ARMA\arma_sweep_RS_A801_<timestamp> `
  --output outputs\meta_analysis.tex `
  --start-date 2018-10-01 `
  --end-date 2018-12-31 `
  --runs-per-table 10
```

The values are reproduced from saved test artifacts rather than statistically
pooled. Compare rankings only after checking the registry for differing test
sizes, forecast horizons, stations, and time intervals.

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

For exploratory plots without chronological train/validation/test splitting,
edit `src/methods/lstm_cluster/data_science_run.py` and run:

```powershell
.\.venv\Scripts\python.exe src\methods\lstm_cluster\data_science_run.py
```

That runner uses the same station loader as `run_experiment.py`, saves
histograms, time series, and pairwise scatter plots for the selected variables,
and repeats the same plot families on normalized data when `SCALER_TYPE` is
`"standard"` or `"minmax"`.

Implementation:

- `src/data/load_data.py`
- `src/data/clean_data.py`
- `src/config.py`

## Sliding Windows

Sliding windows convert daily rows into temporal samples. Each sample contains
`window_size` consecutive days and all selected feature columns.

In the LSTM+Cluster experiment, `WINDOW_STRIDE` controls the distance in days
between consecutive window starts inside each chronological split. A value of
`1` keeps every possible window, `2` keeps starts `0, 2, 4, ...`, and a value
equal to `window_size` creates non-overlapping windows. Original start indices,
target dates, and forecast-horizon alignment are preserved.

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

The same heuristic is available as a standalone command for a chosen station
and window configuration:

```powershell
cluster-auto-sigma --state RS --station-id A801 --window-size 15 --n-values 5 --scaler-type standard --precipitation-scaler none --train-ratio 0.6
```

This command uses the same feature preprocessing as the LSTM+cluster pipeline:
scalers are fitted only on the chronological training fraction, covariates and
precipitation use their separately configured scalers, windows are flattened,
and optional PCA is fitted on the training windows. Match
`--scaler-type`, `--precipitation-scaler`, `--train-ratio`, and
`--pca-variance-threshold` to the experiment configuration.

For code that already has its clustering features prepared (for example an
eigengap analysis), avoid loading and windowing the data again:

```python
from methods.cluster.automatic_sigma import generate_sigma_candidates_from_features

sigmas = generate_sigma_candidates_from_features(windows_flat, n_values=5)
```

The sigma logic lives in `src/methods/tools/sigma_choosing.py`.
It contains:

- `euclidian_distances`
- `take_sigma`
- `sigma_values_from_distance_distribution`
- `calculate_sigma_values`

## Eigengap Cluster-Count Analysis

The standalone eigengap command evaluates multiple sliding-window sizes using
the same normalized Gaussian affinity graph as spectral clustering:

```powershell
cluster-eigengap --state RS --station-id A801 --window-sizes 5 10 15 --n-sigma-values 5 --additional-sigma-values 0.5 1.0 2.0 --scaler-type standard --precipitation-scaler none --train-ratio 0.6
```

For each window size, it generates sigma candidates with the automatic
pairwise-distance heuristic, evaluates every candidate, and saves one plot per
window/sigma pair with the first 20 eigengaps. It highlights the largest gap
and prints the suggested cluster count, sigma, and gap value for every result.
The eigengap heuristic uses
`gap(k) = lambda_k - lambda_(k+1)` on eigenvalues sorted from largest to
smallest. Configure the sweep with `--n-sigma-values`, `--lower-quantile`, and
`--upper-quantile`. Additional manually selected values can be applied to every
window configuration with `--additional-sigma-values` (or its shorter
`--sigma-values` alias); duplicates are evaluated once. More options and assumptions are documented in
`src/methods/cluster/cluster.md`. A candidate that produces a degenerate
affinity matrix is reported explicitly; its recommendation is marked `N/A` and
the remaining sigma candidates and window sizes continue to run.

Its feature preprocessing matches the LSTM+cluster pipeline: scalers are fitted
only on the chronological training fraction, covariates and precipitation use
their separately configured scalers, and PCA is optionally fitted after window
flattening. Keep the eigengap `SCALER_TYPE`, `PRECIPITATION_SCALER`,
`TRAIN_RATIO`, and `PCA_VARIANCE_THRESHOLD` values synchronized with
`src/methods/lstm_cluster/run_experiment.py` when comparing recommendations
with an experiment.

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
- `kshape`: uses the local K-Shape implementation in
  `src/methods/cluster/kshape.py`
- `spectral`: uses the local implementation in
  `src/methods/cluster/ng.py`

`CLUSTERING_ALGORITHM` may be a single value or a list. A list creates one
configuration per algorithm, so `["kshape", "kmeans"]` can be compared with
`PIVOT_PARAMETER = "CLUSTERING_ALGORITHM"`.

K-Shape compares z-normalized temporal shapes through normalized
cross-correlation, so it is insensitive to per-window offset and amplitude and
can align a shared temporal shift across multiple weather features. In the
LSTM pipeline it operates on `(samples, days, features)` windows, learns only
from the training split, and predicts validation/test labels from the learned
shape centroids. Set `PCA_VARIANCE_THRESHOLD = None` for K-Shape. The generic
dispatcher also accepts 2D input, interpreted as univariate time series:

```python
labels = cluster_feature_matrix(
    windows[:, :, 0],
    n_clusters=3,
    algorithm="kshape",
    random_state=42,
)
```

The `methods.cluster.manual` module provides two manual rules. `legacy` is the
horizon-rain-guided behavior exposed through the same `cluster_feature_matrix`
dispatcher: cluster `0` represents known zero-rain horizons, while known
positive horizons are split into `k - 1` ordered groups. In the LSTM runner,
`rain_level` instead averages raw `PRECIPITACAO_TOTAL` over all `J` input days
of each training window, divides `[0, max(training window mean)]` into `k`
equal-width intervals, and uses those intervals as training labels. Internal
upper bounds are open; the overall maximum is included in the final cluster.
Validation and test still use the configured held-out assignment method.

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
Training uses AdamW with `LEARNING_RATE` and decoupled `WEIGHT_DECAY`
configured in the active LSTM runner.

Implementation:

- `src/models/lstm.py`
- class: `LSTMPrecipitationPredictor`

The experiment reverses the temporary flattened representation and restores
one LSTM timestep per day in the input window:

```python
X_lstm = X.reshape(X.shape[0], window_size, n_features)
```

For each cluster, the training loop:

1. Selects train/validation/test samples assigned to that cluster.
2. Builds an LSTM model with two LSTM layers, dropout, and dense layers.
3. Trains with optional early stopping, monitored by `EARLY_STOPPING_METRIC`
   after the configured `WARM_UP` epochs.
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
- `comparative_analysis/`, when `COMPARATIVE_RUN = True`

Configuration folder names include `sigma_<value>` only for spectral
clustering. `kmeans` and `kshape` output folders omit the sigma component.

The comparative folder contains one time-series and one scatter panel per lead
day, `03_training_history_comparison.png`, one metric panel per lead day, and
machine-readable `test_predictions_comparison.csv`,
`aligned_test_predictions.csv`, `training_history_comparison.csv`,
`comparative_metrics.csv`, `comparison_manifest.csv`, and
`comparison_summary.txt`. It also writes `report_compare.tex`, a sweep-level
LaTeX report with a linked table of contents plus Cluster Step, Prediction Time
Series by lead day, Prediction Scatter plot, Training History, and Test Metrics
sections. Its Cluster Step section skips silhouette plots for `K = 1` and
deduplicates fixed cluster diagnostics when `K` is fixed across the sweep,
while keeping `05_cluster_performance.png` for each run. The time-series
`01_test_timeseries_comparison_lead_day_XX.png` panels use Seaborn line plots
for each lead day. The shared observed series is rendered as a subdued,
semi-transparent reference behind thin, semi-transparent prediction curves
from a colorblind palette, preserving overlapping temporal behavior across
the common-date splits. The scatter panels use shared axes and an identity
line; the history panel compares cluster-weighted training and validation
LOSS, MSE, MAE, and R2; and the metric panels compare RMSE, MAE, and R2 against
`PIVOT_PARAMETER` on the common-date test interval in one row of three columns.
The Test Metrics section starts with an overall RMSE/MAE/R2 table pooled across
all forecast days for each pivot value, followed by a compact lead-day summary
table grouped across all available D+k columns for each compared run. It also
writes an overall three-panel pivot-versus-metric plot and bolds the best model
for each metric in every report table.

`sweep_summary.txt` records the PCA variance threshold and whether PCA was
disabled or applied only to clustering. Applying flattened-window PCA to LSTM
inputs is rejected because it would remove the temporal axis. The
best-configuration section also includes these PCA fields.

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
- `experiment_report.pdf`, only when `CREATE_REPORT = True` and a local LaTeX
  compiler is available
- grouped prediction, residual, error, cluster-performance, and histogram plots
  under folders such as `prediction_overview/`,
  `prediction_timeseries_splits/lead_day_XX/`, `residual_diagnostics/`,
  `cluster_diagnostics/`, `model_fit/`, `cluster_precipitation_histograms/`,
  `cluster_prediction_histograms/`, `cluster_prediction_timeseries/`, and
  `cluster_prediction_scatter/`
- split time-series plots in `prediction_timeseries_splits/lead_day_XX/` use
  target dates from the source dataset on the x-axis, formatted as `dd/mm/YYYY`
- cluster silhouette diagnostics under `cluster_diagnostics/`, including
  `08_silhouette_analysis.png` and `silhouette_scores.csv` when
  `SILHOUETTE_INFO = True`; DTW runs use precomputed DTW distances, and K-Shape
  runs use precomputed SBD distances. Set `SILHOUETTE_INFO = False` to skip
  these silhouette artifacts
- cluster precipitation histograms now use
  `cluster_precipitation_histograms/all_clusters_precipitation_histograms.png`
  for the combined subplot panel and
  `cluster_precipitation_histograms/individual/` for the per-cluster files
- `06_cluster_distribution.png` and
  `cluster_training_batch_statistics.csv` under `cluster_diagnostics/`, showing
  train/validation/test counts plus `n_train` and
  `ceil(n_train / batch_size)` for every cluster. Cluster-only runs retain all
  three split bars in the PNG but omit the LSTM-specific table and CSV
- `07_precipitation_distribution_by_cluster.png` under `cluster_diagnostics/`,
  with side-by-side boxplots for forecast-target precipitation and mean
  precipitation across the input-window days, using the same training,
  validation, and test split colors as `06_cluster_distribution.png`
- `cluster_diagnostics/cluster_timeline.png`, plotting every training,
  validation, and test window in chronological split order against its assigned
  cluster
- `cluster_diagnostics/clusters_timeseries/cluster_<id>/`, containing raw
  precipitation series for every test input window assigned to that cluster.
  Each cluster folder has a 5x4 overview panel (additional panels are written
  after 20 windows) and one PNG per window under `individual_windows/`. Folders
  are created for every configured cluster from `cluster_0` through
  `cluster_{K-1}`; an empty test cluster receives a placeholder overview panel.
  `PLOT_CLUSTER_TIMESERIES` controls whether this diagnostic is created.
  When enabled, `CLUSTER_TIMESERIES_PLOT_LIMIT` limits the total selected
  individual series across these folders, while `None` leaves it unrestricted.
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
record the manual clustering method when applicable, cluster-assignment method,
KNN neighbor count, PCA variance threshold, PCA mode (`disabled`, `clustering
only`, or `clustering and LSTM`), selected covariate scaler, precipitation
scaler, and LSTM target scale. Predictions are inverse-transformed to
millimeters before metrics and plots.


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
open `experiments/create_beamer_report.py`, edit `RUN_DIR`, `SELECTED_PLOTS`,
and optionally `PARAMS`, then run:

```powershell
python experiments\create_beamer_report.py
```

The runner writes `beamer.tex` and compiles `beamer.pdf` by default. Set
`COMPILE_PDF = False` in the script if only the TeX source is needed.
`PARAMS = ["LEARNING_RATE", "EPOCHS", "CLUSTERING_ALGORITHM"]` adds a first
section with a ruled table of selected run parameters. Names can be copied from
`run_experiment.py` constants, `summary.txt`, `experiment_report.tex`, or
`sweep_results.csv`.

The command-line form is still available when needed:

```powershell
python experiments\create_beamer_report.py outputs\dd_mm_yy\lstm_cluster_sweep_RS_A801_YYYYMMDD_HHMMSS\RS_A801_w15_k03_kmeans --params LEARNING_RATE EPOCHS CLUSTERING_ALGORITHM --plots prediction_overview\02_predictions_vs_actual.png cluster_prediction_scatter\*.png residual_diagnostics\*.png
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
CLUSTERING_FEATURE_NORMALIZE = "standard"
CLUSTERING_PRECIPITATION_NORMALIZE = None
LSTM_FEATURE_NORMALIZE = "standard"
LSTM_PRECIPITATION_NORMALIZE = None  # None keeps PRECIPITACAO_TOTAL and targets in mm
LSTM_PRECIPITATION_TRANSFORM = False  # True applies log1p only in the LSTM path
LSTM_LOSS_FUNCTION = "weighted_mse_loss"
LOSS_ALPHA = 1.0
N_SIGMA_VALUES = 1
EPOCHS = 2
EARLY_STOPPING_METRIC = "loss"
VERBOSE_TRAINING = 1
SHOW_CONSOLE_INFO = True
PARALEL = False
REQUIRE_GPU = True
```

Set `LSTM_LOSS_FUNCTION = "mean_squared_error"` to keep the standard MSE
training objective. The default `"weighted_mse_loss"` averages
`(1 + LOSS_ALPHA * y_real) * |y_real - y_pred|^2`; `LOSS_ALPHA` must be finite
and greater than zero. Keep the active target scale non-negative when using
this loss. The default `LSTM_PRECIPITATION_NORMALIZE = None` satisfies that
condition, while standardization can create negative targets and weights.

Set the loss to `"quantile_weighted_mse"` to weight each cluster-specific LSTM
by precipitation quantile bins calculated from that cluster's own training rain
targets across all lead days, in the active target scale. With
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
| Standalone automatic sigma runner | `methods.cluster.automatic_sigma` |
| Cluster feature matrix and dispatch | `methods.cluster.cluster_pipeline` |
| Horizon-rain-guided clustering | `methods.cluster.manual` |
| K-Shape clustering | `methods.cluster.kshape` |
| Spectral clustering | `methods.cluster.ng` |
| LSTM model | `models.lstm` |
| Metrics and reports | `evaluation.metrics` |
| Diagnostic plots | `evaluation.evaluation_plot_tools` |
| Main experiment runner | `methods.lstm_cluster.run_experiment` |
| Main experiment pipeline | `methods.lstm_cluster.pipeline` |
