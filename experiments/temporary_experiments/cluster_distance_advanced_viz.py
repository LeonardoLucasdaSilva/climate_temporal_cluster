"""Advanced visualization: Feature distributions for high vs low precipitation windows.

Creates detailed distribution plots comparing high and low precipitation groups
within each cluster, showing which specific features differ the most.
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
from climate_cluster.methods.tools.sliding_windows import create_windows


def create_feature_distribution_plots(windows, labels, df, window_size, high_precip_mask,
                                      feature_names, n_clusters, output_dir):
    """Create detailed distribution plots for top features in each cluster."""

    # Flatten windows to get feature-by-feature statistics
    n_windows = windows.shape[0]
    n_features = len(feature_names)

    # For each cluster, create a plot showing distributions of top features
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) < 2:
            continue

        cluster_high_precip = high_precip_mask[cluster_indices]

        if np.sum(cluster_high_precip) < 2 or np.sum(~cluster_high_precip) < 2:
            continue

        # Get windows for this cluster
        cluster_windows = windows[cluster_indices]  # shape: (n_cluster_windows, window_size, n_features)

        # Flatten: reshape to (n_cluster_windows, window_size * n_features)
        cluster_windows_flat = cluster_windows.reshape(cluster_windows.shape[0], -1)

        # Calculate which features have biggest differences
        feature_differences = []
        for feat_idx in range(n_features):
            # Get all values for this feature across all days in windows
            feat_values = cluster_windows_flat[:, feat_idx::n_features]
            feat_values_flat = feat_values.flatten()

            high_vals = feat_values_flat[np.repeat(cluster_high_precip, window_size)]
            low_vals = feat_values_flat[np.repeat(~cluster_high_precip, window_size)]

            if len(high_vals) > 0 and len(low_vals) > 0:
                diff = np.abs(np.mean(high_vals) - np.mean(low_vals))
                feature_differences.append((feat_idx, feature_names[feat_idx], diff, high_vals, low_vals))

        if not feature_differences:
            continue

        # Sort by difference and get top 6
        feature_differences.sort(key=lambda x: x[2], reverse=True)
        top_features = feature_differences[:6]

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for plot_idx, (feat_idx, feat_name, diff, high_vals, low_vals) in enumerate(top_features):
            ax = axes[plot_idx]

            # Create violin plot for better distribution visualization
            data_to_plot = pd.DataFrame({
                'Value': np.concatenate([high_vals, low_vals]),
                'Group': np.concatenate([
                    ['High Precip'] * len(high_vals),
                    ['Low Precip'] * len(low_vals)
                ])
            })

            sns.violinplot(data=data_to_plot, x='Group', y='Value', ax=ax,
                          palette=['#FF6B6B', '#4ECDC4'])

            # Add box plot on top
            sns.boxplot(data=data_to_plot, x='Group', y='Value', ax=ax,
                       width=0.3, palette=['#FF6B6B', '#4ECDC4'],
                       showcaps=False, whiskerprops=dict(visible=False),
                       medianprops=dict(color='black', linewidth=2),
                       boxprops=dict(alpha=0.3))

            ax.set_title(f'{feat_name}\n(Difference: {diff:.4f})', fontsize=11, fontweight='bold')
            ax.set_xlabel('Window Group', fontsize=10)
            ax.set_ylabel('Normalized Value', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')

            # Add count annotations
            high_count = len(high_vals)
            low_count = len(low_vals)
            ax.text(0, ax.get_ylim()[1] * 0.95, f'n={high_count}', ha='center', fontsize=9)
            ax.text(1, ax.get_ylim()[1] * 0.95, f'n={low_count}', ha='center', fontsize=9)

        # Hide unused subplots
        for idx in range(len(top_features), 6):
            axes[idx].axis('off')

        plt.suptitle(f'Cluster {cluster_id}: Feature Distributions\n'
                    f'High Precip: {np.sum(cluster_high_precip)} | Low Precip: {np.sum(~cluster_high_precip)} windows',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        output_path = output_dir / f"cluster_{cluster_id:02d}_distributions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Cluster {cluster_id} distributions saved")


def create_scatter_distance_plot(windows_flat, labels, high_precip_mask, n_clusters, output_dir):
    """Create scatter plots showing distances from centroids."""

    from sklearn.decomposition import PCA

    # Use PCA to project to 2D for visualization
    pca = PCA(n_components=2)
    windows_2d = pca.fit_transform(windows_flat)

    fig, axes = plt.subplots(2, 5, figsize=(18, 10))
    axes = axes.flatten()

    for cluster_id in range(n_clusters):
        ax = axes[cluster_id]
        cluster_mask = labels == cluster_id
        cluster_high = high_precip_mask & cluster_mask
        cluster_low = (~high_precip_mask) & cluster_mask

        # Plot low precipitation windows
        if np.sum(cluster_low) > 0:
            ax.scatter(windows_2d[cluster_low, 0], windows_2d[cluster_low, 1],
                      c='#4ECDC4', alpha=0.5, s=30, label='Low Precip', edgecolors='none')

        # Plot high precipitation windows
        if np.sum(cluster_high) > 0:
            ax.scatter(windows_2d[cluster_high, 0], windows_2d[cluster_high, 1],
                      c='#FF6B6B', alpha=0.7, s=60, label='High Precip',
                      marker='*', edgecolors='darkred', linewidth=0.5)

        # Calculate and plot centroids
        if np.sum(cluster_low) > 0:
            low_centroid = np.mean(windows_2d[cluster_low], axis=0)
            ax.scatter(low_centroid[0], low_centroid[1], c='darkblue', s=200,
                      marker='X', edgecolors='black', linewidth=1.5, zorder=5)

        if np.sum(cluster_high) > 0:
            high_centroid = np.mean(windows_2d[cluster_high], axis=0)
            ax.scatter(high_centroid[0], high_centroid[1], c='darkred', s=200,
                      marker='X', edgecolors='black', linewidth=1.5, zorder=5)

            # Draw line between centroids if both exist
            if np.sum(cluster_low) > 0:
                ax.plot([low_centroid[0], high_centroid[0]],
                       [low_centroid[1], high_centroid[1]],
                       'k--', alpha=0.5, linewidth=2)

        high_count = np.sum(cluster_high)
        low_count = np.sum(cluster_low)
        high_pct = 100.0 * high_count / (high_count + low_count) if (high_count + low_count) > 0 else 0

        ax.set_title(f'Cluster {cluster_id}\nHigh: {high_count} ({high_pct:.1f}%), Low: {low_count}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
        ax.grid(True, alpha=0.3)

        if cluster_id == 0:
            ax.legend(loc='best', fontsize=9)

    plt.suptitle(f'Window Distribution by Cluster (PCA Projection)\n'
                f'X markers = Centroids (Blue=Low Precip, Red=High Precip)',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_scatter_pca.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ PCA scatter plot saved")
    print(f"    Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")


def main():
    """Run advanced visualizations."""
    # Configuration
    state = "RS"
    station_id = "A801"
    window_size = 20
    n_clusters = 10
    precipitation_percentile = 75

    print("=" * 80)
    print("Advanced Cluster Distance Visualizations")
    print("=" * 80)

    # Load data
    print(f"\n[1] Loading data...")
    df = load_single_station(state=state, station_id=station_id, data_root=DATA_ROOT)

    # Determine threshold
    nonzero_precip = df[df['PRECIPITACAO_TOTAL'] > 0]['PRECIPITACAO_TOTAL']
    threshold = np.percentile(nonzero_precip, precipitation_percentile)

    # Create windows
    print(f"[2] Creating windows...")
    windows, (scaler, _pca) = create_windows(df, window_size=window_size, normalize=True)
    windows_flat = windows.reshape(windows.shape[0], -1)

    # Get feature names
    numeric_cols = [col for col in df.columns if col != "Data" and pd.api.types.is_numeric_dtype(df[col])]
    feature_names = numeric_cols

    # Cluster
    print(f"[3] Clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(windows_flat)

    # Identify high precipitation windows
    print(f"[4] Identifying high precipitation windows...")
    high_precip_mask = np.zeros(len(windows), dtype=bool)
    for window_idx in range(len(windows)):
        day_after_pos = window_idx + window_size
        if day_after_pos < len(df):
            precip = df.iloc[day_after_pos]['PRECIPITACAO_TOTAL']
            if precip >= threshold:
                high_precip_mask[window_idx] = True

    # Create visualizations
    print(f"\n[5] Creating advanced visualizations...")
    output_dir = OUTPUTS_DIR / "rs_a801_distance_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n    Creating feature distribution plots...")
    create_feature_distribution_plots(windows, labels, df, window_size,
                                     high_precip_mask, feature_names, n_clusters, output_dir)

    print(f"\n    Creating PCA scatter plots...")
    create_scatter_distance_plot(windows_flat, labels, high_precip_mask, n_clusters, output_dir)

    print("\n" + "=" * 80)
    print("✓ Advanced visualizations complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

