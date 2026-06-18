# Tools

This folder contains preprocessing utilities shared by clustering, modeling,
and experiments.

- `dimensionality_reduction_tools.py`: helpers for selecting numeric features,
  flattening windows, and fitting PCA by an explained-variance threshold.
- `sigma_choosing.py`: helpers for spectral-clustering sigma selection.
  Use `euclidian_distances` to build a pairwise distance matrix and
  `take_sigma` or `calculate_sigma_values` to generate candidate sigma values
  from that distribution.
- `sliding_window_tools.py`: sliding-window construction for daily climate station
  data, with optional normalization and PCA.

Typical usage:

```python
from climate_cluster.methods.tools.sliding_window_tools import create_windows
from climate_cluster.methods.tools.sigma_choosing import (
    calculate_sigma_values,
    euclidian_distances,
    take_sigma,
)

windows, (scaler, pca) = create_windows(df, window_size=20, normalize=True)
distances = euclidian_distances(windows.reshape(windows.shape[0], -1))
sigmas = take_sigma(distances)
sigmas_from_df = calculate_sigma_values(df, n_values=20)
```
