# Plots

This package contains plotting implementations shared by experiments and
reports.

- `evaluation_plot_tools.py`: diagnostic plots for regression predictions,
  residuals, precipitation magnitude bins, and cluster-level performance.

Typical usage:

```python
from climate_cluster.plots import plot_predictions_vs_actual

fig, axes = plot_predictions_vs_actual(y_true, y_pred)
fig.savefig(output_path)
```
