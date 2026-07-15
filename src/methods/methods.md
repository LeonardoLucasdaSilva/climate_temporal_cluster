# Methods

This package holds reusable method implementations that are not tied to a
single experiment.

- `cluster/`: clustering algorithms, including the custom spectral clustering
  implementation and horizon-rain-guided manual clustering.
- `arma/`: univariate ARMA precipitation baseline used for comparison with
  LSTM+cluster runs. It keeps chronological split and lead-day target alignment
  compatible with the main experiment and writes outputs under
  `outputs/dd_mm_yy/ARMA/`.
- `lstm_cluster/`: the active LSTM-by-cluster precipitation experiment. It
  contains the user-facing runner, sweep pipeline, output config, plot/report
  writers, and folder-level documentation.
- `tools/`: preprocessing tools used before clustering or modeling, such as
  dimensionality reduction and sliding-window feature creation.

Use these modules from experiments or pipelines, for example:

```python
from methods.cluster.ng import spectral_clustering
from methods.cluster.manual import manual_clustering
from methods.arma.pipeline import run_arma_experiment
from methods.tools.sliding_windows import create_windows
```
