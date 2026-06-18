# Methods

This package holds reusable method implementations that are not tied to a
single experiment.

- `cluster/`: clustering algorithms, including the custom spectral clustering
  implementation.
- `tools/`: preprocessing tools used before clustering or modeling, such as
  dimensionality reduction and sliding-window feature creation.

Use these modules from experiments or pipelines, for example:

```python
from climate_cluster.methods.cluster.ng import spectral_clustering
from climate_cluster.methods.tools.sliding_windows import create_windows
```
