"""Analyze clusters for windows preceding highest precipitation days at RS A801.

This script:
1. Loads RS A801 station data
2. Identifies the 10 days with highest precipitation
3. For each day, finds the window ending the day before
4. Reports the cluster assignments for those windows
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from climate_cluster.config import DATA_ROOT, OUTPUTS_DIR
from climate_cluster.config_data import load_single_station
from climate_cluster.features.window_features import create_windows
from climate_cluster.clustering.ng import fit_predict


def create_cluster_precipitation_graphs(df, windows, labels, n_clusters, output_dir):
    """Create and save distribution graphs for precipitation by cluster.

    For each cluster, collects all windows assigned to that cluster,
    finds the precipitation value for the day after each window ends,
    and creates a histogram of those precipitation values (excluding zeros).

    Args:
        df: Original dataframe with daily data
        windows: Windows array (n_windows, window_size, n_features)
        labels: Cluster assignments for each window
        n_clusters: Number of clusters
        output_dir: Directory to save graphs
    """
    window_size = windows.shape[1]

    # Create a dictionary to store precipitation values for each cluster
    cluster_precipitations = {i: [] for i in range(n_clusters)}

    # For each window, get the precipitation value for the day after
    for window_idx, cluster_id in enumerate(labels):
        # The window starts at position window_idx and ends at position window_idx + window_size - 1
        # The day after the window ends is at position window_idx + window_size
        day_after_pos = window_idx + window_size

        # Check if this position is valid (within dataframe bounds)
        if day_after_pos < len(df):
            precip = df.iloc[day_after_pos]['PRECIPITACAO_TOTAL']
            cluster_precipitations[cluster_id].append(precip)

    # Create graphs for each cluster
    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 4 * n_clusters))

    # Handle case where there's only one cluster
    if n_clusters == 1:
        axes = [axes]

    # Set style
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", n_clusters)

    for cluster_id in range(n_clusters):
        ax = axes[cluster_id]
        precips = cluster_precipitations[cluster_id]

        if len(precips) > 0:
            # Count zero and non-zero values
            zero_count = sum(1 for p in precips if p == 0)
            nonzero_precips = [p for p in precips if p > 0]

            # Create histogram with only non-zero values
            if len(nonzero_precips) > 0:
                ax.hist(nonzero_precips, bins=50, color=colors[cluster_id], alpha=0.7, edgecolor='black')

            # Add statistics
            if len(nonzero_precips) > 0:
                mean_precip = np.mean(nonzero_precips)
                median_precip = np.median(nonzero_precips)
                std_precip = np.std(nonzero_precips)

                ax.axvline(mean_precip, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_precip:.2f}')
                ax.axvline(median_precip, color='green', linestyle='--', linewidth=2, label=f'Median: {median_precip:.2f}')
            else:
                mean_precip = 0
                median_precip = 0
                std_precip = 0

            ax.set_xlabel('Precipitation (mm)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'Cluster {cluster_id} - Precipitation Distribution (Day After Window, Non-Zero Values Only)\n'
                        f'Total Windows: {len(precips)}, Zero Values: {zero_count}, Non-Zero: {len(nonzero_precips)}',
                        fontsize=12, fontweight='bold')
            if len(nonzero_precips) > 0:
                ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data for this cluster',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Cluster {cluster_id} - No Windows', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / "cluster_precipitation_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Graphs saved to: {output_path}")
    plt.close()

    # Also save individual graphs for each cluster
    for cluster_id in range(n_clusters):
        precips = cluster_precipitations[cluster_id]

        if len(precips) > 0:
            # Count zero and non-zero values
            zero_count = sum(1 for p in precips if p == 0)
            nonzero_precips = [p for p in precips if p > 0]

            fig, ax = plt.subplots(figsize=(10, 6))

            # Create histogram with only non-zero values
            if len(nonzero_precips) > 0:
                ax.hist(nonzero_precips, bins=50, color=colors[cluster_id], alpha=0.7, edgecolor='black')

                mean_precip = np.mean(nonzero_precips)
                median_precip = np.median(nonzero_precips)
                std_precip = np.std(nonzero_precips)
                min_precip = np.min(nonzero_precips)
                max_precip = np.max(nonzero_precips)

                ax.axvline(mean_precip, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_precip:.2f}')
                ax.axvline(median_precip, color='green', linestyle='--', linewidth=2, label=f'Median: {median_precip:.2f}')
                ax.legend(fontsize=11)
            else:
                mean_precip = 0
                median_precip = 0
                std_precip = 0
                min_precip = 0
                max_precip = 0

            ax.set_xlabel('Precipitation (mm)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Cluster {cluster_id} - Precipitation Distribution\n'
                        f'(Day Following Window End, Non-Zero Values Only)',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add statistics box
            stats_text = f'Total Windows: {len(precips)}\nZero Values: {zero_count}\nNon-Zero: {len(nonzero_precips)}\n\nNon-Zero Stats:\nMean: {mean_precip:.2f}\nMedian: {median_precip:.2f}\nStd: {std_precip:.2f}\nMin: {min_precip:.2f}\nMax: {max_precip:.2f}'
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=10, family='monospace')

            plt.tight_layout()

            individual_path = output_dir / f"cluster_{cluster_id}_precipitation.png"
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            print(f"    ✓ Individual graph saved: {individual_path}")
            plt.close()

    # Save statistics to CSV
    stats_list = []
    for cluster_id in range(n_clusters):
        precips = cluster_precipitations[cluster_id]
        if len(precips) > 0:
            zero_count = sum(1 for p in precips if p == 0)
            nonzero_precips = [p for p in precips if p > 0]

            stats_list.append({
                'cluster': cluster_id,
                'total_windows': len(precips),
                'zero_values': zero_count,
                'nonzero_values': len(nonzero_precips),
                'zero_percentage': 100.0 * zero_count / len(precips) if len(precips) > 0 else 0,
                'mean_precipitation': np.mean(nonzero_precips) if len(nonzero_precips) > 0 else 0,
                'median_precipitation': np.median(nonzero_precips) if len(nonzero_precips) > 0 else 0,
                'std_precipitation': np.std(nonzero_precips) if len(nonzero_precips) > 0 else 0,
                'min_precipitation': np.min(nonzero_precips) if len(nonzero_precips) > 0 else 0,
                'max_precipitation': np.max(nonzero_precips) if len(nonzero_precips) > 0 else 0,
                'q25_precipitation': np.percentile(nonzero_precips, 25) if len(nonzero_precips) > 0 else 0,
                'q75_precipitation': np.percentile(nonzero_precips, 75) if len(nonzero_precips) > 0 else 0,
            })

    stats_df = pd.DataFrame(stats_list)
    stats_path = output_dir / "cluster_precipitation_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"    ✓ Statistics saved to: {stats_path}")

    return cluster_precipitations, stats_df


def find_window_ending_before_date(df, target_date, window_size):
    """Find the window index that ends on the day before target_date.

    Args:
        df: DataFrame with 'Data' column containing dates
        target_date: The precipitation peak date (Timestamp or similar)
        window_size: Number of days in each window

    Returns:
        Tuple of (window_idx, window_end_date) or (None, None) if not found
    """
    # Ensure target_date is a Timestamp
    target_date = pd.Timestamp(target_date)

    # The day before the target
    day_before = target_date - pd.Timedelta(days=1)

    # Find if this day exists in the dataframe
    # Convert Data column to dates for comparison
    df_dates = df['Data'].dt.normalize()
    mask = df_dates == day_before

    if not mask.any():
        return None, None

    end_pos = df.index[mask][0]

    # The window starting position (it ends at end_pos)
    window_start_pos = end_pos - window_size + 1

    if window_start_pos < 0:
        return None, None

    # The window index in the windows array
    window_idx = window_start_pos

    return window_idx, day_before


def main():
    """Run the analysis."""
    # Configuration
    state = "RS"
    station_id = "A801"
    window_size = 20
    n_clusters = 10
    sigma = 1
    sample_rate = 1  # Use 50% of windows to speed up clustering

    print("=" * 80)
    print("RS A801 Precipitation-Precipitation Cluster Analysis")
    print("=" * 80)

    # Step 1: Load the station data
    print(f"\n[1] Loading {state}/{station_id} station data...")
    df = load_single_station(state=state, station_id=station_id, data_root=DATA_ROOT)
    print(f"    ✓ Loaded {len(df)} days of data")
    print(f"    Date range: {df['Data'].min()} to {df['Data'].max()}")

    # Step 2: Get the 10 days with highest precipitation
    print(f"\n[2] Finding 10 days with highest precipitation...")
    top_precip_days = df.nlargest(10, "PRECIPITACAO_TOTAL")
    print(f"    ✓ Found 10 days:")
    for idx, (date_idx, row) in enumerate(top_precip_days.iterrows(), 1):
        print(f"       {idx:2d}. {row['Data'].date()} - {row['PRECIPITACAO_TOTAL']:.1f} mm")

    # Step 3: Create windows and cluster
    print(f"\n[3] Creating windows (size={window_size}) and clustering...")
    windows, scaler = create_windows(
        df,
        window_size=window_size,
        normalize=True,
    )
    print(f"    ✓ Created {len(windows)} windows")

    # Flatten and cluster
    windows_flat = windows.reshape(windows.shape[0], -1)
    print(f"    ✓ Flattened to shape {windows_flat.shape}")

    print(f"    Running spectral clustering (k={n_clusters}, sigma={sigma})...")
    # Use KMeans for speed (spectral clustering is too slow for 9494 samples)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(windows_flat)
    print(f"    ✓ Clustering complete")

    for i in range(n_clusters):
        count = sum(l == i for l in labels)
        pct = 100.0 * count / len(labels)
        print(f"       Cluster {i}: {count:4d} windows ({pct:5.1f}%)")

    # Step 4: Create visualization graphs
    print(f"\n[4] Creating precipitation distribution graphs by cluster...")
    output_dir = OUTPUTS_DIR / "rs_a801_precip_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_precipitations, stats_df = create_cluster_precipitation_graphs(
        df, windows, labels, n_clusters, output_dir
    )

    print(f"    ✓ Graphs and statistics created")

    # Step 5: Find clusters for windows ending before each high-precip day
    print(f"\n[5] Finding clusters for windows ending before high-precipitation days...")
    print(f"\n{'Date':<12} {'Precip (mm)':<15} {'Days Before':<15} {'Window Idx':<15} {'Cluster':<10}")
    print("-" * 70)

    results = []

    for idx, (date_idx, row) in enumerate(top_precip_days.iterrows(), 1):
        date = row['Data']
        precip = row['PRECIPITACAO_TOTAL']

        # Find window ending the day before this date
        window_idx, window_end_date = find_window_ending_before_date(
            df, date, window_size
        )

        if window_idx is None:
            print(f"{date.date():<12} {precip:>13.1f}  {'N/A':<15} {'N/A':<15} {'N/A':<10}")
            results.append({
                'date': date.date(),
                'precipitation_mm': precip,
                'days_before': None,
                'window_idx': None,
                'cluster': None,
                'window_end_date': None,
            })
        else:
            cluster = labels[window_idx]
            days_before = (date - window_end_date).days

            print(f"{date.date():<12} {precip:>13.1f}  {days_before:>14d}  {window_idx:>14d}  {cluster:>9d}")

            results.append({
                'date': date.date(),
                'precipitation_mm': precip,
                'days_before': days_before,
                'window_idx': window_idx,
                'cluster': cluster,
                'window_end_date': window_end_date.date(),
            })

    # Step 6: Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    valid_results = results_df[results_df['cluster'].notna()]

    if len(valid_results) > 0:
        print(f"\nCluster distribution for windows before high-precipitation days:")
        cluster_counts = valid_results['cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            pct = 100.0 * count / len(valid_results)
            print(f"  Cluster {int(cluster_id)}: {count} windows ({pct:.1f}%)")

        print(f"\nWindows analyzed: {len(valid_results)}")
        print(f"Windows with valid data: {len(valid_results)}")

    # Save results

    results_df.to_csv(output_dir / "precipitation_cluster_results.csv", index=False)
    print(f"\n✓ Results saved to: {output_dir / 'precipitation_cluster_results.csv'}")

    # Also save a detailed report
    with open(output_dir / "analysis_report.txt", "w") as f:
        f.write("RS A801 Precipitation-Cluster Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Station: {state}/{station_id}\n")
        f.write(f"  Window Size: {window_size} days\n")
        f.write(f"  Number of Clusters: {n_clusters}\n")
        f.write(f"  Sigma (bandwidth): {sigma}\n")
        f.write(f"  Total windows: {len(windows)}\n")
        f.write(f"  Date range: {df['Data'].min().date()} to {df['Data'].max().date()}\n\n")

        f.write("Results:\n")
        f.write(f"{'Date':<12} {'Precip (mm)':<15} {'Window End':<15} {'Window Idx':<15} {'Cluster':<10}\n")
        f.write("-" * 70 + "\n")

        for _, row in results_df.iterrows():
            if pd.notna(row['cluster']):
                f.write(f"{str(row['date']):<12} {row['precipitation_mm']:>13.1f}  {str(row['window_end_date']):<15} {int(row['window_idx']):>14d}  {int(row['cluster']):>9d}\n")
            else:
                f.write(f"{str(row['date']):<12} {row['precipitation_mm']:>13.1f}  {'N/A':<15} {'N/A':<15} {'N/A':<10}\n")

    print(f"✓ Report saved to: {output_dir / 'analysis_report.txt'}")

    print("\n" + "=" * 80)
    print("✓ Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

