# Cluster Methods

This folder contains clustering algorithms used by the project.

- `cluster_pipeline.py`: reusable clustering pipeline helpers. It contains
  `numeric_feature_columns`, `create_cluster_feature_matrix`, and
  `create_pipeline_clustering_features`, which reproduces the LSTM pipeline's
  train-only scaling and optional PCA for standalone analyses, plus
  `cluster_feature_matrix`, which dispatches K-means, K-Shape, spectral, or
  manual clustering. It also contains
  `run_clustering_pipeline` and `main`, the CLI entrypoint used by the
  `climate-cluster` command.
- `dtw.py`: exact dependent multivariate Dynamic Time Warping, pairwise and
  cross-distance matrices, and normalization of the supported cluster
  dissimilarity metric names.
- `kshape.py`: K-Shape estimator and functional interface. It implements
  z-normalization, normalized cross-correlation shape distance, temporal
  alignment, shape extraction, multiple initializations, and held-out
  prediction from learned shape centroids.
- `ng.py`: custom spectral clustering implementation. It builds a
  Gaussian affinity matrix, normalizes it, extracts the largest eigenvectors,
  row-normalizes the embedding, and clusters that embedding.
- `eigengap.py`: standalone multi-window eigengap analysis. It builds the same
  normalized Gaussian affinity graph, plots the first 20 leading-eigenvalue
  gaps at the numerically optimized sigma for each window size, ignores the
  `k=1` gap, and reports each heuristic cluster-count recommendation.
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

K-Shape is implemented locally and has no dependency beyond NumPy and SciPy.
Use the canonical algorithm alias `kshape`. A 2D array is treated as a batch of
univariate series; a 3D array uses `(samples, timestamps, features)`, with one
temporal shift shared by all feature channels. Constant feature channels are
handled as zero-valued normalized shapes.

The active LSTM+cluster pipeline can instead build spectral affinity from DTW
distances between scaled 3D weather windows. DTW uses a shared temporal path
for all features. Because scikit-learn K-means and arithmetic centroids do not
support an arbitrary DTW metric, this mode requires spectral or manual
training labels plus KNN held-out assignment. PCA must remain disabled so the
window time axis is preserved. `dwt` is accepted as an alias and normalized to
the canonical name `dtw`.

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

K-Shape usage with the original temporal windows:

```python
from methods.cluster.kshape import KShape

model = KShape(n_clusters=3, random_state=42)
labels = model.fit_predict(windows)
held_out_labels = model.predict(held_out_windows)
```

In the LSTM pipeline, K-Shape always fits on training window tensors and uses
its learned shape centroids for validation and test. PCA must be disabled
because it removes the temporal axis required for shape alignment. K-Shape has
its own shape-based distance, so it cannot be combined with
`CLUSTER_DISSIMILARITY_METRIC="dtw"`.

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
cluster-eigengap --state RS --station-id A801 --window-sizes 5 10 15 --sigma-bounds 0.000001 100 --sigma-scout-points 15 --max-sigma-evaluations 100 --eigen-max-iterations 300 --scaler-type standard --precipitation-scaler none --train-ratio 0.6
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

For every window size, bounded scalar optimization searches for the sigma that
maximizes the usable eigengap. It searches `(0, 100]` by default (represented
numerically as `[1e-6, 100]`) and saves
`eigengaps_window_<WINDOW_SIZE>_optimal.png`. Use `--sigma-bounds` to change the
interval and `--max-sigma-evaluations` to control the fixed trial budget. The
defaults use 15 evenly spaced scouts and a total budget of 100 evaluations;
bounded scalar optimization uses the remainder to refine the best neighboring
region under the assumption that the objective is continuous there. Change the
initial coverage with `--sigma-scout-points`.
Gap `k` is defined as `lambda_k - lambda_(k+1)` after sorting the normalized
affinity eigenvalues from largest to smallest. Gap `k=1` is ignored; the
largest of the remaining first 20 gaps is highlighted and `k` is reported as
the heuristic number of clusters. The console summary includes one optimal
sigma, suggested cluster count, and largest usable gap per window. The affinity
matrix is dense and requires memory proportional to the square of the number
of windows;
large station histories can therefore require substantial memory. ARPACK is
limited to 300 iterations per sigma by default (`--eigen-max-iterations`), and
non-convergent or degenerate trials are skipped while the search continues.

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
