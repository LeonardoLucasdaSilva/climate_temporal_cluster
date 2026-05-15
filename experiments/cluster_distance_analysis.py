"""Analyze distances between high precipitation and other windows in clusters.

This script:
1. Loads RS A801 station data and creates windows
2. Clusters the windows
3. For each cluster, identifies windows that precede high precipitation days
4. Compares distances and feature distributions between high/low precipitation groups
5. Creates visualizations showing which features are most discriminative
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from climate_cluster.config import DATA_ROOT, OUTPUTS_DIR
from climate_cluster.config_data import load_single_station
from climate_cluster.features.window_features import create_windows


def get_precipitation_threshold_fixed():
    """Get fixed precipitation threshold of 30mm."""
    return 30.0


def identify_high_precipitation_windows(df, windows, window_size, threshold):
    """Identify which windows precede high precipitation days.

    Returns:
        Boolean array indicating which windows precede high precipitation days
    """
    high_precip_mask = np.zeros(len(windows), dtype=bool)

    for window_idx in range(len(windows)):
        day_after_pos = window_idx + window_size
        if day_after_pos < len(df):
            precip = df.iloc[day_after_pos]['PRECIPITACAO_TOTAL']
            if precip >= threshold:
                high_precip_mask[window_idx] = True

    return high_precip_mask


def analyze_cluster_distances(windows_flat, labels, cluster_id, high_precip_mask, feature_names):
    """Analyze distances within a cluster between high and low precipitation windows.

    Returns:
        Dictionary with statistics
    """
    cluster_mask = labels == cluster_id
    cluster_indices = np.where(cluster_mask)[0]
    cluster_windows = windows_flat[cluster_indices]
    cluster_high_precip = high_precip_mask[cluster_indices]

    stats = {
        'cluster_id': cluster_id,
        'total_windows': len(cluster_indices),
        'high_precip_windows': np.sum(cluster_high_precip),
        'low_precip_windows': len(cluster_indices) - np.sum(cluster_high_precip),
    }

    if np.sum(cluster_high_precip) > 0 and np.sum(~cluster_high_precip) > 0:
        high_precip_windows = cluster_windows[cluster_high_precip]
        low_precip_windows = cluster_windows[~cluster_high_precip]

        # Calculate mean centroids
        high_centroid = np.mean(high_precip_windows, axis=0)
        low_centroid = np.mean(low_precip_windows, axis=0)

        # Distance between centroids
        centroid_distance = np.linalg.norm(high_centroid - low_centroid)
        stats['centroid_distance'] = centroid_distance

        # Average distances from centroid
        high_to_centroid = np.mean([np.linalg.norm(w - high_centroid) for w in high_precip_windows])
        low_to_centroid = np.mean([np.linalg.norm(w - low_centroid) for w in low_precip_windows])
        stats['avg_high_to_centroid'] = high_to_centroid
        stats['avg_low_to_centroid'] = low_to_centroid

        # Calculate feature-wise differences (which features differ the most)
        feature_differences = np.abs(high_centroid - low_centroid)
        stats['feature_differences'] = feature_differences
        stats['feature_names'] = feature_names
        stats['high_centroid'] = high_centroid
        stats['low_centroid'] = low_centroid

        # Feature-wise statistics
        feature_stats = []
        for feat_idx, feat_name in enumerate(feature_names):
            high_values = high_precip_windows[:, feat_idx]
            low_values = low_precip_windows[:, feat_idx]

            feature_stats.append({
                'feature': feat_name,
                'high_mean': np.mean(high_values),
                'high_std': np.std(high_values),
                'low_mean': np.mean(low_values),
                'low_std': np.std(low_values),
                'difference': np.abs(np.mean(high_values) - np.mean(low_values)),
            })

        stats['feature_stats'] = pd.DataFrame(feature_stats).sort_values('difference', ascending=False)

    return stats


def analyze_high_precip_only(windows_flat, labels, cluster_id, high_precip_mask, feature_names):
    """Analyze feature variation ONLY within high precipitation windows in a cluster.
    
    For each feature, calculates the standard deviation and distance from centroid
    for high precipitation windows only.

    Returns:
        Dictionary with high-precipitation-only statistics
    """
    cluster_mask = labels == cluster_id
    cluster_indices = np.where(cluster_mask)[0]
    cluster_windows = windows_flat[cluster_indices]
    cluster_high_precip = high_precip_mask[cluster_indices]

    stats = {
        'cluster_id': cluster_id,
        'high_precip_windows': np.sum(cluster_high_precip),
    }

    if np.sum(cluster_high_precip) > 1:  # Need at least 2 windows to calculate distance
        high_precip_windows = cluster_windows[cluster_high_precip]
        
        # Calculate centroid of high precipitation windows only
        high_centroid = np.mean(high_precip_windows, axis=0)
        
        # Calculate feature-wise statistics within high precip windows
        feature_stats = []
        for feat_idx, feat_name in enumerate(feature_names):
            high_values = high_precip_windows[:, feat_idx]
            
            # Distance from centroid for this feature
            distances_from_centroid = np.abs(high_values - high_centroid[feat_idx])
            avg_distance_from_centroid = np.mean(distances_from_centroid)
            
            feature_stats.append({
                'feature': feat_name,
                'mean': np.mean(high_values),
                'std': np.std(high_values),
                'min': np.min(high_values),
                'max': np.max(high_values),
                'avg_distance_from_centroid': avg_distance_from_centroid,
            })
        
        stats['feature_stats'] = pd.DataFrame(feature_stats).sort_values('avg_distance_from_centroid', ascending=False)

    return stats


def create_distance_visualizations(cluster_stats_list, output_dir, window_size, feature_names):
    """Create comprehensive visualizations for distance analysis."""

    # 1. Feature Differences by Cluster - Bar Chart
    fig, axes = plt.subplots(len([s for s in cluster_stats_list if 'feature_stats' in s]), 1,
                             figsize=(12, 4 * len([s for s in cluster_stats_list if 'feature_stats' in s])))

    if len([s for s in cluster_stats_list if 'feature_stats' in s]) == 1:
        axes = [axes]

    ax_idx = 0
    for stats in cluster_stats_list:
        if 'feature_stats' not in stats:
            continue

        ax = axes[ax_idx]
        feature_stats = stats['feature_stats']

        # Get top 10 features with largest differences
        top_features = feature_stats.head(10)

        colors = ['red' if stats['high_centroid'][feature_names.index(f)] > stats['low_centroid'][feature_names.index(f)]
                 else 'blue' for f in top_features['feature']]

        ax.barh(range(len(top_features)), top_features['difference'], color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('|Mean Difference| (normalized units)', fontsize=11)
        ax.set_title(f"Cluster {stats['cluster_id']}: Feature Differences (Red=High Precip Higher, Blue=Low Precip Higher)\n"
                    f"High Precip: {stats['high_precip_windows']} windows | Low Precip: {stats['low_precip_windows']} windows",
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        ax_idx += 1

    plt.tight_layout()
    fig.savefig(output_dir / "01_cluster_feature_differences.png", dpi=300, bbox_inches='tight')
    print(f"    ✓ Feature differences chart saved")
    plt.close()

    # 2. Centroid Distance Visualization - Radar/Spider chart would be good but let's do heatmap
    fig, axes = plt.subplots(1, len([s for s in cluster_stats_list if 'feature_stats' in s]),
                             figsize=(5 * len([s for s in cluster_stats_list if 'feature_stats' in s]), 6))

    if len([s for s in cluster_stats_list if 'feature_stats' in s]) == 1:
        axes = [axes]

    ax_idx = 0
    for stats in cluster_stats_list:
        if 'feature_stats' not in stats:
            continue

        ax = axes[ax_idx]

        # Create a matrix of centroids for visualization
        feature_list = stats['feature_names']
        n_features = len(feature_list)

        # Get indices of features
        data_matrix = np.array([
            stats['high_centroid'],
            stats['low_centroid']
        ])

        im = ax.imshow(data_matrix, cmap='RdBu_r', aspect='auto')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['High Precip\nMean', 'Low Precip\nMean'])
        ax.set_xticks(range(0, n_features, max(1, n_features // 10)))
        ax.set_xticklabels([feature_list[i] if i % max(1, n_features // 10) == 0 else ''
                           for i in range(n_features)], rotation=45, ha='right')

        ax.set_title(f"Cluster {stats['cluster_id']}: Feature Value Heatmap\n"
                    f"Centroid Distance: {stats.get('centroid_distance', 0):.3f}",
                    fontsize=11, fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Normalized Value')

        ax_idx += 1

    plt.tight_layout()
    fig.savefig(output_dir / "02_cluster_centroid_heatmaps.png", dpi=300, bbox_inches='tight')
    print(f"    ✓ Centroid heatmaps saved")
    plt.close()

    # 3. Distribution comparison for top discriminative features
    fig, axes = plt.subplots(len([s for s in cluster_stats_list if 'feature_stats' in s]), 5,
                             figsize=(16, 4 * len([s for s in cluster_stats_list if 'feature_stats' in s])))

    if len([s for s in cluster_stats_list if 'feature_stats' in s]) == 1:
        axes = axes.reshape(1, -1)
    elif len([s for s in cluster_stats_list if 'feature_stats' in s]) > 1:
        pass  # Already 2D
    else:
        axes = [axes]

    # This would require access to the full windows array, so we'll skip detailed distributions
    # and instead create a summary table

    # 4. Summary Statistics Table
    summary_data = []
    for stats in cluster_stats_list:
        if 'centroid_distance' in stats:
            summary_data.append({
                'Cluster': stats['cluster_id'],
                'Total Windows': stats['total_windows'],
                'High Precip': stats['high_precip_windows'],
                'Low Precip': stats['low_precip_windows'],
                'High %': 100.0 * stats['high_precip_windows'] / stats['total_windows'],
                'Centroid Distance': stats['centroid_distance'],
                'Avg High to Centroid': stats['avg_high_to_centroid'],
                'Avg Low to Centroid': stats['avg_low_to_centroid'],
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        fig, ax = plt.subplots(figsize=(14, len(summary_data) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')

        table_data = []
        for _, row in summary_df.iterrows():
            table_data.append([
                f"{int(row['Cluster'])}",
                f"{int(row['Total Windows'])}",
                f"{int(row['High Precip'])}",
                f"{int(row['Low Precip'])}",
                f"{row['High %']:.1f}%",
                f"{row['Centroid Distance']:.4f}",
                f"{row['Avg High to Centroid']:.4f}",
                f"{row['Avg Low to Centroid']:.4f}",
            ])

        table = ax.table(cellText=table_data,
                        colLabels=['Cluster', 'Total', 'High\nPrecip', 'Low\nPrecip', 'High %',
                                   'Centroid\nDistance', 'Avg High\nto Centroid', 'Avg Low\nto Centroid'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.08, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Color header
        for i in range(8):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(8):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        plt.title('Distance Analysis Summary by Cluster', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(output_dir / "03_distance_summary_table.png", dpi=300, bbox_inches='tight')
        print(f"    ✓ Summary table saved")
        plt.close()

        # Also save as CSV
        summary_df.to_csv(output_dir / "distance_analysis_summary.csv", index=False)
        print(f"    ✓ Summary CSV saved")


def main():
    """Run the distance analysis."""
    # Configuration
    state = "RS"
    station_id = "A801"
    window_size = 20
    n_clusters = 10
    sigma = 1
    precipitation_threshold = 30.0  # Fixed threshold: 30mm for HIGH precipitation

    print("=" * 80)
    print("Cluster Distance Analysis - High (>30mm) vs Low (<30mm) Precipitation Windows")
    print("=" * 80)

    # Step 1: Load the station data
    print(f"\n[1] Loading {state}/{station_id} station data...")
    df = load_single_station(state=state, station_id=station_id, data_root=DATA_ROOT)
    print(f"    ✓ Loaded {len(df)} days of data")
    print(f"    Date range: {df['Data'].min()} to {df['Data'].max()}")

    # Step 2: Report precipitation threshold
    print(f"\n[2] Precipitation threshold...")
    threshold = precipitation_threshold
    high_precip_count = len(df[df['PRECIPITACAO_TOTAL'] >= threshold])
    print(f"    ✓ Threshold: {threshold:.2f} mm (FIXED)")
    print(f"    ✓ Days with HIGH precipitation (≥{threshold}mm): {high_precip_count} ({100.0 * high_precip_count / len(df):.1f}%)")

    # Step 3: Create windows and get feature names
    print(f"\n[3] Creating windows (size={window_size})...")
    windows, scaler = create_windows(
        df,
        window_size=window_size,
        normalize=True,
    )
    print(f"    ✓ Created {len(windows)} windows")

    # Get feature names (assuming we have numeric columns)
    numeric_cols = [col for col in df.columns if col != "Data" and pd.api.types.is_numeric_dtype(df[col])]
    feature_names = numeric_cols
    print(f"    ✓ Features: {feature_names}")

    # Step 4: Flatten windows for clustering
    windows_flat = windows.reshape(windows.shape[0], -1)
    print(f"    ✓ Flattened to shape {windows_flat.shape}")

    # Step 5: Cluster the windows
    print(f"\n[4] Clustering windows (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(windows_flat)
    print(f"    ✓ Clustering complete")

    for i in range(n_clusters):
        count = sum(l == i for l in labels)
        pct = 100.0 * count / len(labels)
        print(f"       Cluster {i}: {count:4d} windows ({pct:5.1f}%)")

    # Step 6: Identify high precipitation windows
    print(f"\n[5] Identifying high precipitation windows...")
    high_precip_mask = identify_high_precipitation_windows(df, windows, window_size, threshold)
    print(f"    ✓ High precipitation windows: {np.sum(high_precip_mask)} ({100.0 * np.sum(high_precip_mask) / len(windows):.1f}%)")

    # Step 7: Analyze distances within each cluster
    print(f"\n[6] Analyzing distances within clusters...")
    cluster_stats_list = []

    for cluster_id in range(n_clusters):
        stats = analyze_cluster_distances(windows_flat, labels, cluster_id, high_precip_mask, feature_names)
        cluster_stats_list.append(stats)

        if 'feature_stats' in stats:
            print(f"\n    Cluster {cluster_id}:")
            print(f"      Total windows: {stats['total_windows']}")
            print(f"      High precipitation: {stats['high_precip_windows']} ({100.0 * stats['high_precip_windows'] / stats['total_windows']:.1f}%)")
            print(f"      Low precipitation: {stats['low_precip_windows']} ({100.0 * stats['low_precip_windows'] / stats['total_windows']:.1f}%)")
            print(f"      Centroid distance: {stats['centroid_distance']:.4f}")
            print(f"\n      FEATURE RANKING (all features by distance):")
            print(f"      {'Rank':<6} {'Feature':<30} {'Distance':<15}")
            print(f"      {'-' * 50}")
            for rank, (_, row) in enumerate(stats['feature_stats'].iterrows(), 1):
                print(f"      {rank:<6} {row['feature']:<30} {row['difference']:<15.4f}")

    # Step 7b: Print per-cluster feature ranking
    print(f"\n[7] Per-cluster feature ranking (all features)...")

    # Step 7c: Analyze high precipitation windows only
    print(f"\n[7b] Analyzing ONLY high precipitation windows per cluster...")
    high_precip_stats_list = []
    
    for cluster_id in range(n_clusters):
        stats = analyze_high_precip_only(windows_flat, labels, cluster_id, high_precip_mask, feature_names)
        high_precip_stats_list.append(stats)
        
        if 'feature_stats' in stats and stats['high_precip_windows'] > 0:
            print(f"\n    Cluster {cluster_id} (HIGH PRECIPITATION ONLY):")
            print(f"      High precipitation windows: {stats['high_precip_windows']}")
            print(f"\n      FEATURE RANKING (distance from centroid within high precip windows):")
            print(f"      {'Rank':<6} {'Feature':<30} {'Avg Dist from Centroid':<25} {'Mean':<10} {'Std':<10}")
            print(f"      {'-' * 85}")
            for rank, (_, row) in enumerate(stats['feature_stats'].iterrows(), 1):
                print(f"      {rank:<6} {row['feature']:<30} {row['avg_distance_from_centroid']:<25.4f} {row['mean']:<10.4f} {row['std']:<10.4f}")

    # Step 8: Create visualizations
    print(f"\n[8] Creating visualizations...")
    output_dir = OUTPUTS_DIR / "rs_a801_distance_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    create_distance_visualizations(cluster_stats_list, output_dir, window_size, feature_names)

    # Step 9: Save detailed report as markdown
    print(f"\n[9] Saving detailed report...")
    with open(output_dir / "distance_analysis_report.md", "w") as f:
        f.write("# Cluster Distance Analysis - High (>30mm) vs Low (<30mm) Precipitation Windows\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- **Station:** {state}/{station_id}\n")
        f.write(f"- **Window Size:** {window_size} days\n")
        f.write(f"- **Number of Clusters:** {n_clusters}\n")
        f.write(f"- **Precipitation Threshold:** {threshold:.2f} mm (FIXED)\n")
        f.write(f"- **High Precipitation Windows:** {np.sum(high_precip_mask)} ({100.0 * np.sum(high_precip_mask) / len(windows):.1f}%)\n")
        f.write(f"- **Date Range:** {df['Data'].min().date()} to {df['Data'].max().date()}\n")
        f.write(f"- **Total Features:** {len(feature_names)}\n")
        f.write(f"- **Features:** {', '.join(feature_names)}\n\n")

        f.write("## Detailed Cluster Analysis\n\n")

        for stats in cluster_stats_list:
            f.write(f"### Cluster {stats['cluster_id']}\n\n")
            f.write(f"- **Total Windows:** {stats['total_windows']}\n")
            f.write(f"- **High Precipitation:** {stats['high_precip_windows']} ({100.0 * stats['high_precip_windows'] / stats['total_windows']:.1f}%)\n")
            f.write(f"- **Low Precipitation:** {stats['low_precip_windows']} ({100.0 * stats['low_precip_windows'] / stats['total_windows']:.1f}%)\n\n")

            if 'centroid_distance' in stats:
                f.write(f"**Distance Metrics:**\n")
                f.write(f"- Centroid Distance: {stats['centroid_distance']:.4f}\n")
                f.write(f"- Avg Distance from High-Precip Centroid: {stats['avg_high_to_centroid']:.4f}\n")
                f.write(f"- Avg Distance from Low-Precip Centroid: {stats['avg_low_to_centroid']:.4f}\n\n")

                f.write(f"**All Features Ranked by Distance:**\n\n")
                f.write(f"| Rank | Feature | Distance | High Mean | Low Mean |\n")
                f.write(f"|------|---------|----------|-----------|----------|\n")
                for rank, (_, row) in enumerate(stats['feature_stats'].iterrows(), 1):
                    f.write(f"| {rank} | {row['feature']} | {row['difference']:.4f} | {row['high_mean']:.4f} | {row['low_mean']:.4f} |\n")
                f.write("\n")

        # Add high precipitation only analysis
        f.write("## High Precipitation Windows Only - Feature Variation Analysis\n\n")

        for stats in high_precip_stats_list:
            f.write(f"### Cluster {stats['cluster_id']}\n\n")
            f.write(f"- **High Precipitation Windows:** {stats['high_precip_windows']}\n\n")

            if 'feature_stats' in stats and stats['high_precip_windows'] > 0:
                f.write(f"**All Features Ranked by Distance from Centroid (within high precip windows):**\n\n")
                f.write(f"| Rank | Feature | Avg Dist from Centroid | Mean | Std | Min | Max |\n")
                f.write(f"|------|---------|--------|------|-----|-----|-----|\n")
                for rank, (_, row) in enumerate(stats['feature_stats'].iterrows(), 1):
                    f.write(f"| {rank} | {row['feature']} | {row['avg_distance_from_centroid']:.4f} | {row['mean']:.4f} | {row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |\n")
                f.write("\n")

    print(f"    ✓ Report saved to: {output_dir / 'distance_analysis_report.md'}")

    print("\n" + "=" * 80)
    print("✓ Distance analysis complete!")
    print(f"✓ Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

