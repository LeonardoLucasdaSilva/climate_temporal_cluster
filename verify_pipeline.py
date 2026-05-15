"""Verification script - test all components."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("=" * 80)
print("VERIFICATION: Pipeline Components")
print("=" * 80)

try:
    print("\n[1/5] Testing imports...")
    from climate_cluster.config import DATA_ROOT
    from climate_cluster.config_data import load_single_station
    from climate_cluster.features.window_features import (
        create_normalized_windows,
        create_windows,
        windows_to_dataframe,
    )
    print("✓ All imports successful")

    print("\n[2/5] Checking DATA_ROOT path...")
    print(f"✓ DATA_ROOT = {DATA_ROOT}")
    print(f"✓ Exists: {DATA_ROOT.exists()}")

    print("\n[3/5] Loading sample station...")
    df = load_single_station(state="SP", station_id="A701", data_root=DATA_ROOT)
    print(f"✓ Loaded {len(df)} days")
    print(f"✓ Columns: {list(df.columns)[:5]}...")

    print("\n[4/5] Creating windows...")
    windows, scaler = create_normalized_windows(
        df,
        window_size=4,
        columns=["TEMPERATURA_MAXIMA", "TEMPERATURA_MIN", "PRECIPITACAO_TOTAL"],
    )
    print(f"✓ Created {len(windows)} windows")
    print(f"✓ Shape: {windows.shape}")
    print(f"✓ Scaler: {type(scaler).__name__}")

    print("\n[5/5] Converting back to DataFrame...")
    df_windows = windows_to_dataframe(
        windows,
        columns=["TEMPERATURA_MAXIMA", "TEMPERATURA_MIN", "PRECIPITACAO_TOTAL"],
        scaler=scaler,
    )
    print(f"✓ Denormalized shape: {df_windows.shape}")
    print(f"✓ Columns: {list(df_windows.columns)[:3]}...")

    print("\n" + "=" * 80)
    print("ALL COMPONENTS VERIFIED SUCCESSFULLY")
    print("=" * 80)
    print("""
Your pipeline is ready to use:

1. Load data:  load_single_station(state, station_id, data_root)
2. Windows:    create_normalized_windows(df, window_size, columns)
3. Flatten:    windows.reshape(windows.shape[0], -1)
4. Cluster:    your_algorithm(windows_flat)
5. Analyze:    windows_to_dataframe(windows, columns, scaler)

See examples/ for complete usage patterns.
See WINDOW_FEATURES.md for detailed documentation.
    """)

except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

