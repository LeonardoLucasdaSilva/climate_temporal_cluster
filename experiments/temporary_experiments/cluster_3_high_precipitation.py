"""Extract all high precipitation windows from Cluster 3 with precipitation and humidity values.

This script:
1. Loads RS A801 station data
2. Creates 20-day windows
3. Clusters the windows
4. Filters for Cluster 3
5. Identifies high precipitation (≥30mm) windows
6. Extracts and displays precipitation and humidity minimum values
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from climate_cluster.config import DATA_ROOT, OUTPUTS_DIR
from climate_cluster.config_data import load_single_station
from climate_cluster.methods.tools.sliding_windows import create_windows


def main():
    """Extract high precipitation windows from Cluster 3 with humidity values only."""
    # Configuration
    state = "RS"
    station_id = "A801"
    window_size = 20
    n_clusters = 10
    cluster_target = 3
    precipitation_threshold = 30.0  # High precipitation threshold

    print("=" * 100)
    print(f"Cluster {cluster_target} - High Precipitation Windows Humidity Analysis (≥{precipitation_threshold}mm)")
    print("=" * 100)

    # Step 1: Load the station data
    print(f"\n[1] Loading {state}/{station_id} station data...")
    df = load_single_station(state=state, station_id=station_id, data_root=DATA_ROOT)
    print(f"    ✓ Loaded {len(df)} days of data")

    # Step 2: Create windows
    print(f"\n[2] Creating windows (size={window_size})...")
    windows, (scaler, _pca) = create_windows(
        df,
        window_size=window_size,
        normalize=True,
    )
    print(f"    ✓ Created {len(windows)} windows")

    # Get feature names and find indices
    numeric_cols = [col for col in df.columns if col != "Data" and pd.api.types.is_numeric_dtype(df[col])]
    feature_names = numeric_cols
    print(f"    ✓ Features: {feature_names}")

    # Get indices of humidity features only
    humidity_min_idx = feature_names.index('UMIDADE_MIN')
    humidity_max_idx = feature_names.index('UMIDADE_MAX')

    # Step 3: Flatten windows for clustering
    windows_flat = windows.reshape(windows.shape[0], -1)
    print(f"    ✓ Flattened to shape {windows_flat.shape}")

    # Step 4: Cluster the windows
    print(f"\n[3] Clustering windows (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(windows_flat)
    print(f"    ✓ Clustering complete")

    # Step 5: Filter for Cluster 3
    print(f"\n[4] Filtering for Cluster {cluster_target}...")
    cluster_mask = labels == cluster_target
    cluster_indices = np.where(cluster_mask)[0]
    print(f"    ✓ Found {len(cluster_indices)} windows in Cluster {cluster_target}")

    # Step 6: Extract high precipitation windows with humidity values only
    print(f"\n[5] Extracting high precipitation windows...")

    high_precip_windows = []

    for window_idx in cluster_indices:
        # Get the window (20 days of data)
        window_data = windows[window_idx]  # shape: (20, n_features)

        # Get precipitation for the day AFTER the window
        day_after_idx = window_idx + window_size
        if day_after_idx < len(df):
            precip_day_after = df.iloc[day_after_idx]['PRECIPITACAO_TOTAL']

            # Check if it's high precipitation
            if precip_day_after >= precipitation_threshold:
                # Calculate humidity values over the window (HUMIDITY ONLY)
                mean_humidity_min = np.mean(window_data[:, humidity_min_idx])
                mean_humidity_max = np.mean(window_data[:, humidity_max_idx])

                # Get the end date of the window
                end_date = df.iloc[window_idx + window_size - 1]['Data']
                start_date = df.iloc[window_idx]['Data']

                # Get min/max humidity values within the window
                min_humidity_min = np.min(window_data[:, humidity_min_idx])
                max_humidity_min = np.max(window_data[:, humidity_min_idx])
                min_humidity_max = np.min(window_data[:, humidity_max_idx])
                max_humidity_max = np.max(window_data[:, humidity_max_idx])

                high_precip_windows.append({
                    'window_idx': window_idx,
                    'start_date': start_date,
                    'end_date': end_date,
                    'mean_umidade_min': mean_humidity_min,
                    'mean_umidade_max': mean_humidity_max,
                    'min_umidade_min': min_humidity_min,
                    'max_umidade_min': max_humidity_min,
                    'min_umidade_max': min_humidity_max,
                    'max_umidade_max': max_humidity_max,
                    'precip_day_after': precip_day_after,
                })

    # Convert to DataFrame and sort by precipitation day after
    df_high_precip = pd.DataFrame(high_precip_windows)
    df_high_precip = df_high_precip.sort_values('precip_day_after', ascending=False)

    print(f"    ✓ Found {len(df_high_precip)} high precipitation windows in Cluster {cluster_target}")

    # Step 7: Display results
    print(f"\n[6] High Precipitation Windows in Cluster {cluster_target}:")
    print(f"    (All windows where precipitation day after ≥ {precipitation_threshold}mm)")
    print()

    print(f"    {'Rank':<6} {'End Date':<12} {'Start Date':<12} {'Precip Next Day':<18} {'Mean UMIDADE_MIN':<20} {'Mean UMIDADE_MAX':<20}")
    print(f"    {'-' * 105}")

    for rank, (_, row) in enumerate(df_high_precip.iterrows(), 1):
        print(f"    {rank:<6} {str(row['end_date'].date()):<12} {str(row['start_date'].date()):<12} {row['precip_day_after']:>17.4f} {row['mean_umidade_min']:>19.4f} {row['mean_umidade_max']:>19.4f}")

    # Step 8: Extended view with min/max
    print(f"\n[7] Detailed View - Including Min/Max Humidity Values:")
    print()

    print(f"    {'Rank':<6} {'End Date':<12} {'Precip After':<15} {'Mean UMIN':<15} {'Min UMIN':<15} {'Max UMIN':<15} {'Mean UMAX':<15} {'Min UMAX':<15} {'Max UMAX':<15}")
    print(f"    {'-' * 130}")

    for rank, (_, row) in enumerate(df_high_precip.iterrows(), 1):
        print(f"    {rank:<6} {str(row['end_date'].date()):<12} {row['precip_day_after']:>14.4f} {row['mean_umidade_min']:>14.4f} {row['min_umidade_min']:>14.4f} {row['max_umidade_min']:>14.4f} {row['mean_umidade_max']:>14.4f} {row['min_umidade_max']:>14.4f} {row['max_umidade_max']:>14.4f}")

    # Step 9: Save to CSV
    print(f"\n[8] Saving results...")
    output_dir = OUTPUTS_DIR / "cluster_3_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_high_precip.to_csv(output_dir / "cluster_3_high_precipitation_windows.csv", index=False)
    print(f"    ✓ Results saved: {output_dir / 'cluster_3_high_precipitation_windows.csv'}")

    # Step 10: Create summary statistics
    print(f"\n[9] Summary Statistics:")
    print()
    print(f"    Total high precipitation windows in Cluster {cluster_target}: {len(df_high_precip)}")
    print(f"    Percentage of cluster: {100.0 * len(df_high_precip) / len(cluster_indices):.2f}%")
    print()
    print(f"    Precipitation (day after) - Reference:")
    print(f"      Mean: {df_high_precip['precip_day_after'].mean():.4f} mm")
    print(f"      Min:  {df_high_precip['precip_day_after'].min():.4f} mm")
    print(f"      Max:  {df_high_precip['precip_day_after'].max():.4f} mm")
    print(f"      Std:  {df_high_precip['precip_day_after'].std():.4f} mm")
    print()
    print(f"    Mean UMIDADE_MIN in window (normalized):")
    print(f"      Mean: {df_high_precip['mean_umidade_min'].mean():.4f}")
    print(f"      Min:  {df_high_precip['mean_umidade_min'].min():.4f}")
    print(f"      Max:  {df_high_precip['mean_umidade_min'].max():.4f}")
    print(f"      Std:  {df_high_precip['mean_umidade_min'].std():.4f}")
    print()
    print(f"    Mean UMIDADE_MAX in window (normalized):")
    print(f"      Mean: {df_high_precip['mean_umidade_max'].mean():.4f}")
    print(f"      Min:  {df_high_precip['mean_umidade_max'].min():.4f}")
    print(f"      Max:  {df_high_precip['mean_umidade_max'].max():.4f}")
    print(f"      Std:  {df_high_precip['mean_umidade_max'].std():.4f}")
    print()
    print(f"    Min UMIDADE_MIN in window (normalized):")
    print(f"      Mean: {df_high_precip['min_umidade_min'].mean():.4f}")
    print(f"      Min:  {df_high_precip['min_umidade_min'].min():.4f}")
    print(f"      Max:  {df_high_precip['min_umidade_min'].max():.4f}")
    print()
    print(f"    Max UMIDADE_MIN in window (normalized):")
    print(f"      Mean: {df_high_precip['max_umidade_min'].mean():.4f}")
    print(f"      Min:  {df_high_precip['max_umidade_min'].min():.4f}")
    print(f"      Max:  {df_high_precip['max_umidade_min'].max():.4f}")
    print()
    print(f"    Min UMIDADE_MAX in window (normalized):")
    print(f"      Mean: {df_high_precip['min_umidade_max'].mean():.4f}")
    print(f"      Min:  {df_high_precip['min_umidade_max'].min():.4f}")
    print(f"      Max:  {df_high_precip['min_umidade_max'].max():.4f}")
    print()
    print(f"    Max UMIDADE_MAX in window (normalized):")
    print(f"      Mean: {df_high_precip['max_umidade_max'].mean():.4f}")
    print(f"      Min:  {df_high_precip['max_umidade_max'].min():.4f}")
    print(f"      Max:  {df_high_precip['max_umidade_max'].max():.4f}")

    # Step 11: Create detailed report
    print(f"\n[10] Creating detailed report...")

    with open(output_dir / "cluster_3_high_precipitation_report.txt", "w") as f:
        f.write("=" * 100 + "\n")
        f.write(f"Cluster {cluster_target} - High Precipitation Windows Humidity Analysis\n")
        f.write("=" * 100 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Station: {state}/{station_id}\n")
        f.write(f"  Window Size: {window_size} days\n")
        f.write(f"  Number of Clusters: {n_clusters}\n")
        f.write(f"  Cluster Target: {cluster_target}\n")
        f.write(f"  High Precipitation Threshold: {precipitation_threshold} mm\n")
        f.write(f"  Total windows in Cluster {cluster_target}: {len(cluster_indices)}\n")
        f.write(f"  Date range: {df['Data'].min().date()} to {df['Data'].max().date()}\n\n")

        f.write("=" * 100 + "\n")
        f.write(f"High Precipitation Windows ({len(df_high_precip)} found)\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"{'Rank':<6} {'Start Date':<15} {'End Date':<15} {'Precip After':<20} {'Mean UMIDADE_MIN':<20} {'Mean UMIDADE_MAX':<20}\n")
        f.write(f"{'-' * 100}\n")

        for rank, (_, row) in enumerate(df_high_precip.iterrows(), 1):
            f.write(f"{rank:<6} {str(row['start_date'].date()):<15} {str(row['end_date'].date()):<15} {row['precip_day_after']:>19.4f} {row['mean_umidade_min']:>19.4f} {row['mean_umidade_max']:>19.4f}\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("Detailed View - With Min/Max Humidity Values\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"{'Rank':<6} {'End Date':<15} {'Precip After':<20} {'Mean UMIN':<20} {'Min UMIN':<20} {'Max UMIN':<20} {'Mean UMAX':<20} {'Min UMAX':<20} {'Max UMAX':<20}\n")
        f.write(f"{'-' * 150}\n")

        for rank, (_, row) in enumerate(df_high_precip.iterrows(), 1):
            f.write(f"{rank:<6} {str(row['end_date'].date()):<15} {row['precip_day_after']:>19.4f} {row['mean_umidade_min']:>19.4f} {row['min_umidade_min']:>19.4f} {row['max_umidade_min']:>19.4f} {row['mean_umidade_max']:>19.4f} {row['min_umidade_max']:>19.4f} {row['max_umidade_max']:>19.4f}\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("Summary Statistics\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Total high precipitation windows: {len(df_high_precip)}\n")
        f.write(f"Percentage of cluster: {100.0 * len(df_high_precip) / len(cluster_indices):.2f}%\n\n")

        f.write("Precipitation (day after) - Reference:\n")
        f.write(f"  Mean: {df_high_precip['precip_day_after'].mean():.4f} mm\n")
        f.write(f"  Min:  {df_high_precip['precip_day_after'].min():.4f} mm\n")
        f.write(f"  Max:  {df_high_precip['precip_day_after'].max():.4f} mm\n")
        f.write(f"  Std:  {df_high_precip['precip_day_after'].std():.4f} mm\n\n")

        f.write("Mean UMIDADE_MIN in window (normalized):\n")
        f.write(f"  Mean: {df_high_precip['mean_umidade_min'].mean():.4f}\n")
        f.write(f"  Min:  {df_high_precip['mean_umidade_min'].min():.4f}\n")
        f.write(f"  Max:  {df_high_precip['mean_umidade_min'].max():.4f}\n")
        f.write(f"  Std:  {df_high_precip['mean_umidade_min'].std():.4f}\n\n")

        f.write("Mean UMIDADE_MAX in window (normalized):\n")
        f.write(f"  Mean: {df_high_precip['mean_umidade_max'].mean():.4f}\n")
        f.write(f"  Min:  {df_high_precip['mean_umidade_max'].min():.4f}\n")
        f.write(f"  Max:  {df_high_precip['mean_umidade_max'].max():.4f}\n")
        f.write(f"  Std:  {df_high_precip['mean_umidade_max'].std():.4f}\n\n")

        f.write("Min UMIDADE_MIN in window (normalized):\n")
        f.write(f"  Mean: {df_high_precip['min_umidade_min'].mean():.4f}\n")
        f.write(f"  Min:  {df_high_precip['min_umidade_min'].min():.4f}\n")
        f.write(f"  Max:  {df_high_precip['min_umidade_min'].max():.4f}\n\n")

        f.write("Max UMIDADE_MIN in window (normalized):\n")
        f.write(f"  Mean: {df_high_precip['max_umidade_min'].mean():.4f}\n")
        f.write(f"  Min:  {df_high_precip['max_umidade_min'].min():.4f}\n")
        f.write(f"  Max:  {df_high_precip['max_umidade_min'].max():.4f}\n\n")

        f.write("Min UMIDADE_MAX in window (normalized):\n")
        f.write(f"  Mean: {df_high_precip['min_umidade_max'].mean():.4f}\n")
        f.write(f"  Min:  {df_high_precip['min_umidade_max'].min():.4f}\n")
        f.write(f"  Max:  {df_high_precip['min_umidade_max'].max():.4f}\n\n")

        f.write("Max UMIDADE_MAX in window (normalized):\n")
        f.write(f"  Mean: {df_high_precip['max_umidade_max'].mean():.4f}\n")
        f.write(f"  Min:  {df_high_precip['max_umidade_max'].min():.4f}\n")
        f.write(f"  Max:  {df_high_precip['max_umidade_max'].max():.4f}\n")

    print(f"    ✓ Report saved: {output_dir / 'cluster_3_high_precipitation_report.txt'}")

    print("\n" + "=" * 100)
    print("✓ Cluster 3 high precipitation analysis complete!")
    print(f"✓ Results saved to: {output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()

