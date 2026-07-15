# Cluster Methods

This folder contains clustering algorithms used by the project.

- `cluster_pipeline.py`: reusable clustering pipeline helpers. It contains
  `numeric_feature_columns`, `create_cluster_feature_matrix`, and
  `cluster_feature_matrix`, which prepare window features and dispatch
  K-means, spectral, or manual clustering. It also contains
  `run_clustering_pipeline` and `main`, the CLI entrypoint used by the
  `climate-cluster` command.
- `ng.py`: custom spectral clustering implementation. It builds a
  Gaussian affinity matrix, normalizes it, extracts the largest eigenvectors,
  row-normalizes the embedding, and clusters that embedding.
- `eigengap.py`: standalone multi-window eigengap analysis. It builds the same
  normalized Gaussian affinity graph, plots the first 20 leading-eigenvalue
  gaps for each window size, highlights the largest gap, and reports its
  heuristic cluster-count recommendation in the console.
- `manual.py`: horizon-rain-guided clustering. It reserves cluster `0` for
  windows with a known zero-rain horizon, divides known rainy horizons into
  `k - 1` ordered groups from lower to heavier rain, and assigns windows with
  unavailable horizons to the nearest cluster centroid in window-feature
  space.

K-means is currently provided by scikit-learn:

```python
from sklearn.cluster import KMeans
```

The project uses `KMeans(...).fit_predict(...)` directly for K-means runs and
inside the last step of the spectral clustering implementation.

Typical usage:

```python
from methods.cluster.cluster_pipeline import (
    cluster_feature_matrix,
    create_cluster_feature_matrix,
    numeric_feature_columns,
)

numeric_cols = numeric_feature_columns(df)
_, windows_flat, _, _, feature_columns = create_cluster_feature_matrix(
    df,
    window_size=20,
    columns=numeric_cols,
)
labels = cluster_feature_matrix(
    windows_flat,
    n_clusters=3,
    algorithm="spectral",
    sigma=1.0,
)
```

For manual clustering, pass horizon-aligned precipitation through the same
dispatcher:

```python
from methods.tools.precipitation_utils import horizon_precipitation

horizon_rain = horizon_precipitation(df, window_size=20, horizon=1)
labels = cluster_feature_matrix(
    windows_flat,
    n_clusters=3,
    algorithm="manual",
    horizon_rain=horizon_rain,
)
```

## Manual rain clustering

Manual rain clustering uses the precipitation observed at a selected forecast
horizon to define the clusters:

1. Cluster `0` contains every window whose known horizon precipitation is zero.
2. Known positive-rain windows are sorted by precipitation and divided as
   evenly as possible among clusters `1` through `k - 1`.
3. Those positive-rain labels are ordered from lower to heavier rain.
4. A feature-space centroid is calculated for every cluster.
5. Windows whose horizon is unavailable or missing are assigned to the nearest
   centroid using Euclidean distance.

For example, with `k=3`, the resulting labels represent zero rain, lower rain,
and heavier rain. The positive-rain groups contain similar numbers of known
samples; they are not based on fixed millimeter thresholds.

Use `horizon_precipitation` to align dataframe precipitation with every sliding
window. `horizon=1` selects the day immediately after the window, while larger
values support future multi-day forecast horizons. Targets beyond the available
data are returned as `NaN` and inferred by centroid distance.

```python
from methods.cluster.manual import ManualRainClustering
from methods.tools.precipitation_utils import horizon_precipitation

horizon_rain = horizon_precipitation(
    df,
    window_size=20,
    horizon=1,
)

model = ManualRainClustering(n_clusters=3)
labels = model.fit_predict(windows_flat, horizon_rain)

print(model.rain_ranges_)
```

For a one-call interface:

```python
from methods.cluster.manual import manual_clustering

labels = manual_clustering(
    windows_flat,
    horizon_rain,
    k=3,
)
```

Requirements and behavior:

- `k` must be at least `2`.
- At least one known zero-rain horizon is required.
- At least `k - 1` known positive-rain horizons are required.
- Known precipitation values cannot be negative.
- Window features must be finite and use the same feature representation for
  fitting and prediction.
- `zero_tolerance` can classify very small precipitation measurements as zero
  when exact zero comparison is unsuitable.

The learned `rain_ranges_` maps each label to the minimum and maximum known
precipitation used for that cluster. `centroids_` stores the corresponding
feature-space centroids.

## Eigengap analysis

Edit the defaults at the top of `eigengap.py` or pass window sizes directly:

```powershell
cluster-eigengap --state RS --station-id A801 --window-sizes 5 10 15 --sigma 1.0 --scaler-type standard --precipitation-scaler none --train-ratio 0.6
```

Eigengap preprocessing mirrors the LSTM+cluster pipeline: it takes the same
chronological training fraction, fits the selected covariate scaler on those
training rows, handles `PRECIPITACAO_TOTAL` with its own optional scaler, builds
flattened windows, and optionally fits PCA on the training windows. Keep
`SCALER_TYPE`, `PRECIPITATION_SCALER`, `TRAIN_RATIO`, and
`PCA_VARIANCE_THRESHOLD` aligned between `eigengap.py` and
`lstm_cluster/run_experiment.py`. The equivalent CLI options are
`--scaler-type`, `--precipitation-scaler`, `--train-ratio`, and
`--pca-variance-threshold`.

For every window size, the command saves
`eigengaps_window_<WINDOW_SIZE>.png` under the configured output directory.
Gap `k` is defined as `lambda_k - lambda_(k+1)` after sorting the normalized
affinity eigenvalues from largest to smallest. The largest of the first 20
gaps is highlighted and `k` is reported as the heuristic number of clusters.
The result is sensitive to the Gaussian affinity bandwidth, so record and
compare `--sigma` when interpreting different runs. The affinity matrix is
dense and requires memory proportional to the square of the number of windows;
large station histories can therefore require substantial memory. If `sigma`
is so small that the affinity graph has no usable edges, the runner prints a
degeneracy warning, marks the recommendation as `N/A`, saves an annotated plot,
and continues with the remaining window sizes.
