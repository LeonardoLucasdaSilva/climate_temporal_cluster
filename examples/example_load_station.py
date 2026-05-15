"""Example: Loading and using a single station's data."""

from climate_cluster.config import DATA_ROOT
from climate_cluster.config_data import load_single_station

# Example 1: Load all default columns for a station
print("Example 1: Load station SP/A701 with default columns")
print("-" * 60)

try:
    df = load_single_station(
        state="SP",
        station_id="A701",
        data_root=DATA_ROOT,
    )
    print(f"Loaded {len(df)} days of data")
    print("\nFirst 5 rows:")
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
except FileNotFoundError as e:
    print(f"Error: {e}")

print("\n")

# Example 2: Load custom columns only
print("Example 2: Load station TO/A055 with custom columns")
print("-" * 60)

try:
    custom_cols = [
        "DATA",
        "TEMPERATURA_MAXIMA",
        "TEMPERATURA_MIN",
        "PRECIPITACAO_TOTAL",
    ]
    df = load_single_station(
        state="TO",
        station_id="A055",
        data_root=DATA_ROOT,
        cols=custom_cols,
    )
    print(f"Loaded {len(df)} days of data")
    print("\nFirst 5 rows:")
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
except FileNotFoundError as e:
    print(f"Error: {e}")

print("\n")

# Example 3: Get basic statistics
print("Example 3: Summary statistics for station SP/A701")
print("-" * 60)

try:
    df = load_single_station(
        state="SP",
        station_id="A701",
        data_root=DATA_ROOT,
    )
    print(df.describe())
except FileNotFoundError as e:
    print(f"Error: {e}")

