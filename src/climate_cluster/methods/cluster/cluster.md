# Cluster Methods

This folder contains clustering algorithms used by the project.

- `spectral_cluster.py`: custom spectral clustering implementation. It builds a
  Gaussian affinity matrix, normalizes it, extracts the largest eigenvectors,
  row-normalizes the embedding, and clusters that embedding.

K-means is currently provided by scikit-learn:

```python
from sklearn.cluster import KMeans
```

The project uses `KMeans(...).fit_predict(...)` directly for K-means runs and
inside the last step of the spectral clustering implementation.
