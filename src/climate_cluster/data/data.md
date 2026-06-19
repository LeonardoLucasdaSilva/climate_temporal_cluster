# Data

This package contains code for loading, cleaning, and visualizing climate data.
It is separate from the root `data/` directory, which stores raw INMET files.

- `load_data.py`: functions for locating and loading station daily CSV files.
- `clean_data.py`: reusable dataframe cleaning helpers.
- `visualize_data.py`: intentionally empty starter module for future data
  visualization code.

Typical usage:

```python
from climate_cluster.data.load_data import load_station_daily_data

df = load_station_daily_data("RS", "A801", data_root)
```
