"""Complete pipeline example: Load data → Create windows → Ready for clustering."""

from climate_cluster.config import DATA_ROOT
from climate_cluster.config_data import load_single_station
from climate_cluster.features.window_features import create_normalized_windows

print("=" * 80)
print("COMPLETE PIPELINE EXAMPLE")
print("=" * 80)

# STEP 1: Load single station data
print("\n[STEP 1] Loading single station data...")
df = load_single_station(
    state="SP",
    station_id="A701",
    data_root=DATA_ROOT,
)
print(f"✓ Loaded {len(df)} days of data")
print(f"  Available columns: {list(df.columns)}")

# STEP 2: Create sliding windows
print("\n[STEP 2] Creating sliding windows (4 consecutive days)...")
columns = [
    "TEMPERATURA_MAXIMA",
    "TEMPERATURA_MIN",
    "PRECIPITACAO_TOTAL",
    "UMIDADE_MAX",
    "VELOCIDADE_VENTO",
]

windows, scaler = create_normalized_windows(
    df,
    window_size=4,
    columns=columns,
)

print(f"✓ Created {len(windows)} windows")
print(f"  Shape: {windows.shape}")
print(f"    - {windows.shape[0]} samples (4-day windows)")
print(f"    - {windows.shape[1]} days per window")
print(f"    - {windows.shape[2]} features per day")

# STEP 3: Prepare for clustering
print("\n[STEP 3] Preparing data for clustering algorithms...")

# Option A: Use as 3D array
print("\n  Option A: Keep 3D structure (n_samples, 4, 5)")
print(f"  Shape: {windows.shape}")
print(f"  First window:\n{windows[0]}")

# Option B: Flatten to 2D
windows_flat = windows.reshape(windows.shape[0], -1)
print(f"\n  Option B: Flatten to 2D (n_samples, 20)")
print(f"  Shape: {windows_flat.shape}")
print(f"  First sample (first 5 features):\n{windows_flat[0, :5]}")

# STEP 4: Ready for your clustering algorithm
print("\n[STEP 4] Data ready for clustering!")
print(f"✓ Pass {windows_flat.shape[0]} samples with {windows_flat.shape[1]} features")
print(f"  to your clustering algorithm")

print("\n" + "=" * 80)
print("EXAMPLE: Pseudo-clustering")
print("=" * 80)

# Fake clustering just to show the flow
import numpy as np

n_clusters = 3
labels = np.random.randint(0, n_clusters, len(windows_flat))

print(f"\n✓ Got cluster assignments: {len(labels)} labels")
print(f"  Cluster 0: {np.sum(labels == 0)} samples")
print(f"  Cluster 1: {np.sum(labels == 1)} samples")
print(f"  Cluster 2: {np.sum(labels == 2)} samples")

# STEP 5: Analyze results
print("\n[STEP 5] Analyzing results...")

# Which windows belong to each cluster
print(f"\nFirst 5 windows belong to clusters: {labels[:5]}")

# Get the raw values for a cluster
cluster_0_indices = np.where(labels == 0)[0]
cluster_0_windows = windows[cluster_0_indices]

print(f"\nCluster 0 analysis:")
print(f"  - Number of 4-day windows: {len(cluster_0_windows)}")
print(f"  - Mean temperature (max): {cluster_0_windows[:, :, 0].mean():.2f}")
print(f"  - Mean temperature (min): {cluster_0_windows[:, :, 1].mean():.2f}")
print(f"  - Mean precipitation: {cluster_0_windows[:, :, 2].mean():.2f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Pipeline steps completed:
1. ✓ Load single station (load_single_station)
2. ✓ Create windows with normalization (create_normalized_windows)
3. ✓ Flatten or keep 3D format
4. ✓ Feed to clustering algorithm
5. ✓ Analyze results by cluster

Next: Replace the pseudo-clustering with your actual algorithm!
      Edit src/climate_cluster/clustering/ng.py
""")

