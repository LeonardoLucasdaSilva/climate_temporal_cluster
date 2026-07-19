# Cluster Methods

This folder contains clustering algorithms used by the project.

- `cluster_pipeline.py`: reusable clustering pipeline helpers. It contains
  `numeric_feature_columns`, `create_cluster_feature_matrix`, and
  `create_pipeline_clustering_features`, which reproduces the LSTM pipeline's
  train-only scaling and optional PCA for standalone analyses, plus
  `cluster_feature_matrix`, which dispatches K-means, spectral, or manual
  clustering. It also contains
  `run_clustering_pipeline` and `main`, the CLI entrypoint used by the
  `climate-cluster` command.
- `ng.py`: custom spectral clustering implementation. It builds a
  Gaussian affinity matrix, normalizes it, extracts the largest eigenvectors,
  row-normalizes the embedding, and clusters that embedding.
- `eigengap.py`: standalone multi-window eigengap analysis. It builds the same
  normalized Gaussian affinity graph, plots the first 20 leading-eigenvalue
  gaps for every automatic sigma candidate of each window size, highlights the
  largest gap, and reports each heuristic cluster-count recommendation.
- `automatic_sigma.py`: standalone and reusable adapter for the project's
  pairwise-distance sigma heuristic. It loads a station when run as a command,
  or accepts a dataframe or prepared feature matrix when imported.
- `manual.py`: rule-based clustering with `legacy` horizon-rain groups and
  `rain_level` equal-width bins of mean input-window precipitation. Both learn
  feature-space centroids from the resulting training labels.

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

`ManualRainClustering(method="legacy")` uses the precipitation observed at a
selected forecast horizon to define the clusters:

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

`ManualRainClustering(method="rain_level")` instead receives one raw mean
precipitation value per input window. For `K` clusters it builds `K + 1` edges
with `numpy.linspace(0, max_training_mean, K + 1)`. Internal intervals are
`[a, b)`, so a value exactly equal to `b` enters the next cluster. The overall
training maximum is included in cluster `K - 1`. Unlike `legacy`, this method
does not use forecast-horizon precipitation to construct training labels.

The active LSTM pipeline calculates those means directly from raw
`PRECIPITACAO_TOTAL` across all `J` days of each training window. Validation and
test remain assigned afterward with the configured centroid or KNN router.

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
- `legacy` requires at least one known zero-rain horizon and at least `k - 1`
  known positive-rain horizons.
- `rain_level` requires finite non-negative window means, a positive training
  maximum, and at least one training window in every equal-width interval.
- Known precipitation values cannot be negative.
- Window features must be finite and use the same feature representation for
  fitting and prediction.
- In `legacy`, `zero_tolerance` can classify very small precipitation
  measurements as zero when exact zero comparison is unsuitable.

The learned `rain_ranges_` maps each label to the minimum and maximum rain value
used for that cluster. `rain_level` also exposes the `K + 1` interval edges in
`thresholds_`. `centroids_` stores the corresponding feature-space centroids.

## Eigengap analysis

Edit the defaults at the top of `eigengap.py` or pass window sizes directly:

```powershell
cluster-eigengap --state RS --station-id A801 --window-sizes 5 10 15 --n-sigma-values 5 --additional-sigma-values 0.5 1.0 2.0 --scaler-type standard --precipitation-scaler none --train-ratio 0.6
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

For every window size, the command generates sigma candidates from the
preprocessed window-distance distribution using `automatic_sigma.py`. It then
evaluates every candidate and saves
`eigengaps_window_<WINDOW_SIZE>_sigma_<SIGMA_INDEX>.png` under the configured
output directory. Use `--lower-quantile` and `--upper-quantile` to configure
the distance range used by the sigma heuristic. Pass manually selected values
with `--additional-sigma-values`; each is appended to every window size's
automatic candidates. Duplicate values are evaluated only once. The shorter
alias `--sigma-values` is also accepted.
Gap `k` is defined as `lambda_k - lambda_(k+1)` after sorting the normalized
affinity eigenvalues from largest to smallest. The largest of the first 20
gaps is highlighted and `k` is reported as the heuristic number of clusters
for that window/sigma pair. The console summary includes all window sizes,
sigma candidates, suggested cluster counts, and largest gaps. The affinity
matrix is dense and requires memory proportional to the square of the number
of windows;
large station histories can therefore require substantial memory. If `sigma`
is so small that the affinity graph has no usable edges, the runner prints a
degeneracy warning, marks the recommendation as `N/A`, saves an annotated plot,
and continues with the remaining sigma candidates and window sizes.

## Automatic sigma candidates

Generate sigma candidates for one station with the installed command:

```powershell
cluster-auto-sigma --state RS --station-id A801 --window-size 15 --n-values 5 --scaler-type standard --precipitation-scaler none --train-ratio 0.6
```

The default heuristic calculates all pairwise Euclidean distances between
flattened windows, sorts one distance per sample pair, and returns evenly
spaced sigma values between positions near the 1st and 20th percentiles. Use
`--lower-quantile` and `--upper-quantile` to change those bounds, or
`--no-normalize` to disable normalization. Feature preparation matches the
LSTM+cluster pipeline: only the chronological training fraction is used,
covariates and precipitation have separately configured scalers, and optional
PCA is fitted after window flattening. Keep `--scaler-type`,
`--precipitation-scaler`, `--train-ratio`, and `--pca-variance-threshold`
aligned with `lstm_cluster/run_experiment.py`.

Other analyses can reuse the same implementation without loading data again:

```python
from methods.cluster.automatic_sigma import (
    generate_sigma_candidates_from_features,
)

sigmas = generate_sigma_candidates_from_features(windows_flat, n_values=5)
```
