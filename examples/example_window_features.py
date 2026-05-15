"""Example: Creating windowed (tuple) features from station data."""

from climate_cluster.config import DATA_ROOT
from climate_cluster.config_data import load_single_station
from climate_cluster.features.window_features import (
    create_normalized_windows,
    create_windows,
    windows_to_dataframe,
)

print("=" * 70)
print("EXAMPLE 1: Basic windowing (4 consecutive days)")
print("=" * 70)

# Load station data
df = load_single_station(
    state="SP",
    station_id="A701",
    data_root=DATA_ROOT,
)

print(f"\nLoaded {len(df)} days of data")
print(f"\nFirst 10 days of raw data:")
print(df.head(10).to_string())

# Create windows of 4 consecutive days
# Data will be normalized by column (zero mean, unit variance)
columns_to_use = [
    "TEMPERATURA_MAXIMA",
    "TEMPERATURA_MIN",
    "PRECIPITACAO_TOTAL",
    "UMIDADE_MAX",
    "VELOCIDADE_VENTO",
]

windows, scaler = create_normalized_windows(
    df,
    window_size=4,
    columns=columns_to_use,
)

print(f"\n✓ Created {len(windows)} windows from {len(df)} days")
print(f"✓ Each window has shape (4 days, {len(columns_to_use)} features)")
print(f"✓ Windows are normalized (zero mean, unit variance per column)")

print(f"\nWindow shape: {windows.shape}")
print(f"  - Dimension 0: {windows.shape[0]} windows")
print(f"  - Dimension 1: {windows.shape[1]} days per window")
print(f"  - Dimension 2: {windows.shape[2]} features per day")

# Show first window (4 consecutive days, 5 features each)
print(f"\nFirst window (4 days × {len(columns_to_use)} features):")
print(f"Shape: {windows[0].shape}")
print(f"\nValue at windows[0]:")
print(windows[0])

print("\n" + "=" * 70)
print("EXAMPLE 2: Denormalized windows as DataFrame")
print("=" * 70)

# Convert normalized windows back to denormalized DataFrame
df_windows = windows_to_dataframe(
    windows,
    columns=columns_to_use,
    scaler=scaler,
    window_size=4,
)

print(f"\nWindows as denormalized DataFrame:")
print(f"Shape: {df_windows.shape}")
print(f"Columns: {list(df_windows.columns)}")
print(f"\nFirst 5 rows:")
print(df_windows.head())

print("\n" + "=" * 70)
print("EXAMPLE 3: Unnormalized windows (raw values)")
print("=" * 70)

# Without normalization
windows_raw, _ = create_windows(
    df,
    window_size=4,
    columns=columns_to_use,
    normalize=False,
)

print(f"\n✓ Created {len(windows_raw)} raw (unnormalized) windows")
print(f"\nFirst window (raw values):")
print(windows_raw[0])

print("\n" + "=" * 70)
print("EXAMPLE 4: Different window sizes")
print("=" * 70)

for ws in [3, 5, 7]:
    windows_ws, _ = create_normalized_windows(df, window_size=ws, columns=columns_to_use)
    print(f"Window size {ws}: {len(windows_ws)} windows created")

print("\n" + "=" * 70)
print("EXAMPLE 5: Ready for clustering!")
print("=" * 70)

print("\nNow you can pass these windows to your clustering algorithm:")
print(f"- Shape: {windows.shape}")
print(f"- Rows are individual samples (each is 4 consecutive days)")
print(f"- Can be reshaped to (n_samples, 4*5) = (n_samples, 20) for algorithms")

# Flatten for ML algorithms that expect 2D input
windows_flat = windows.reshape(windows.shape[0], -1)
print(f"\nFlattened for ML: {windows_flat.shape}")
print(f"  - {windows_flat.shape[0]} samples")
print(f"  - {windows_flat.shape[1]} features (4 days × 5 columns)")

