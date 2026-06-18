# Cluster Methods

This folder contains clustering algorithms used by the project.

- `cluster_pipeline.py`: reusable clustering pipeline helpers. It contains
  `numeric_feature_columns`, `create_cluster_feature_matrix`, and
  `cluster_feature_matrix`, which prepare window features and dispatch either
  K-means or spectral clustering.
- `ng.py`: custom spectral clustering implementation. It builds a
  Gaussian affinity matrix, normalizes it, extracts the largest eigenvectors,
  row-normalizes the embedding, and clusters that embedding.

K-means is currently provided by scikit-learn:

```python
from sklearn.cluster import KMeans
```

The project uses `KMeans(...).fit_predict(...)` directly for K-means runs and
inside the last step of the spectral clustering implementation.

Typical usage:

```python
from climate_cluster.methods.cluster.cluster_pipeline import (
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
