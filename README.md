# Climate Temporal Cluster

This project studies Brazilian INMET daily climate data with a temporal
clustering workflow and cluster-specific LSTM precipitation models.

The main experiment is the LSTM+Cluster pipeline in
`experiments/lstm_cluster.py`. It builds sliding windows from daily station
data, clusters those windows with K-means or spectral clustering, trains one
LSTM model per cluster, and writes metrics, predictions, plots, and
LaTeX-ready summary tables.

## Quick Start

Use Python 3.12. TensorFlow is installed in the project `.venv`, and the VS Code
settings point the play button at that environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the main experiment from the project root:

```powershell
.\.venv\Scripts\python.exe experiments\lstm_cluster.py
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
|   |-- lstm_cluster.py            # Main LSTM+Cluster experiment
|   |-- temporary_experiments/     # Older scripts kept for review
|   `-- experiments.md
|-- outputs/                       # Generated experiment outputs
|-- src/
|   |-- config.py                  # Project paths
|   |-- data/
|   |   |-- load_data.py           # INMET CSV loading
|   |   |-- clean_data.py          # Data cleaning helpers
|   |   `-- visualize_data.py      # Starter visualization module
|   |-- methods/
|   |   |-- cluster/
|   |   |   |-- ng.py              # Spectral clustering implementation
|   |   |   `-- cluster_pipeline.py
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

The pipeline is configured at the top of `experiments/lstm_cluster.py`.

```python
STATE = "RS"
STATION_ID = "A801"
WINDOW_SIZES = [8, 12, 16, 20, 24, 28]
N_CLUSTERS_LIST = [3, 4, 5]
CLUSTERING_ALGORITHM = "spectral"  # or "kmeans"
N_SIGMA_VALUES = 5
```

For each configuration, the experiment runs these stages:

1. Load daily station data.
2. Select numeric climate features.
3. Build sliding windows.
4. Apply PCA dimensionality reduction using a variance threshold.
5. Cluster windows with K-means or spectral clustering.
6. Create next-day precipitation targets.
7. Split samples into train, validation, and test sets while preserving cluster
   information when possible.
8. Train one LSTM model per training cluster.
9. Predict train, validation, and test precipitation.
10. Save metrics, reports, predictions, plots, and LaTeX tables.

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
- `src/data/load_data.py`

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

The LSTM+Cluster experiment uses
`create_cluster_feature_matrix` to produce both the original window output and a
2D `windows_flat` matrix for clustering and model input.

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
4. Writes predictions back into aggregate train/validation/test arrays.
5. Calculates cluster-level test metrics.

The model predicts one value: next-day precipitation in millimeters.

## Evaluation Outputs

Each run writes into a timestamped folder under `outputs/`, for example:

```text
outputs/lstm_cluster_sweep_RS_A801_YYYYMMDD_HHMMSS/
```

Sweep-level files:

- `sweep_results.csv`
- `sweep_summary.txt`
- `overleaf_table.txt`
- `overleaf_cluster_metric_tables.txt`

Each configuration folder contains:

- `metrics_summary.csv`
- `cluster_model_metrics.csv`
- `test_predictions.csv`
- `evaluation_report.txt`
- `summary.txt`
- prediction, residual, error, cluster-performance, and histogram plots

Metrics include:

- MSE
- RMSE
- MAE
- RMSLE
- R2
- MAPE
- zero-day and rainy-day metrics
- per-cluster metrics

Plot helpers live in:

```text
src/evaluation/evaluation_plot_tools.py
```

## Running a Smaller Experiment

The default sweep can be expensive. To test quickly, edit
`experiments/lstm_cluster.py`:

```python
WINDOW_SIZES = [8]
N_CLUSTERS_LIST = [3]
N_SIGMA_VALUES = 1
EPOCHS = 2
VERBOSE_TRAINING = 1
```

Then run:

```powershell
.\.venv\Scripts\python.exe experiments\lstm_cluster.py
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
| Spectral clustering | `methods.cluster.ng` |
| LSTM model | `models.lstm` |
| Metrics and reports | `evaluation.metrics` |
| Diagnostic plots | `evaluation.evaluation_plot_tools` |
| Main experiment | `experiments/lstm_cluster.py` |
