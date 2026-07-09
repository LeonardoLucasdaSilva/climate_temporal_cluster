"""Extract and analyze top windows from Cluster 3 by precipitation and humidity.

This script:
1. Loads RS A801 station data
2. Creates 20-day windows
3. Clusters the windows
4. Filters for Cluster 3
5. Calculates composite score (PRECIPITACAO_TOTAL + UMIDADE_MIN)
6. Displays top 30 windows ranked by this score
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import DATA_ROOT, OUTPUTS_DIR
from data.load_data import load_station_daily_data
from methods.tools.sliding_windows import create_windows


def main():
    """Extract and analyze top windows from Cluster 3."""
    # Configuration
    state = "RS"
    station_id = "A801"
    window_size = 20
    n_clusters = 10
    cluster_target = 3
    top_n = 30

    print("=" * 80)
    print(f"Cluster {cluster_target} - Top {top_n} Windows by Precipitation & Humidity")
    print("=" * 80)

    # Step 1: Load the station data
    print(f"\n[1] Loading {state}/{station_id} station data...")
    df = load_station_daily_data(state=state, station_id=station_id, data_root=DATA_ROOT)
    print(f"    ✓ Loaded {len(df)} days of data")
    print(f"    Date range: {df['Data'].min()} to {df['Data'].max()}")

    # Step 2: Create windows
    print(f"\n[2] Creating windows (size={window_size})...")
    windows, (scaler, _pca) = create_windows(
        df,
        window_size=window_size,
        normalize=True,
    )
    print(f"    ✓ Created {len(windows)} windows")

    # Get feature names
    numeric_cols = [col for col in df.columns if col != "Data" and pd.api.types.is_numeric_dtype(df[col])]
    feature_names = numeric_cols
    print(f"    ✓ Features: {feature_names}")

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

    # Step 6: Calculate composite score for each window
    print(f"\n[5] Calculating composite score (PRECIPITACAO_TOTAL + UMIDADE_MIN)...")

    # Get indices of relevant features
    precip_idx = feature_names.index('PRECIPITACAO_TOTAL')
    humidity_idx = feature_names.index('UMIDADE_MIN')

    window_scores = []

    for window_idx in cluster_indices:
        # Get the window (20 days of data)
        window_data = windows[window_idx]  # shape: (20, n_features)

        # Calculate mean precipitation and humidity over the window
        mean_precip = np.mean(window_data[:, precip_idx])
        mean_humidity = np.mean(window_data[:, humidity_idx])

        # Composite score (sum of normalized values)
        score = mean_precip + mean_humidity

        # Get the end date of the window
        end_date = df.iloc[window_idx + window_size - 1]['Data']

        # Get the precipitation for the day AFTER the window
        day_after_idx = window_idx + window_size
        if day_after_idx < len(df):
            precip_day_after = df.iloc[day_after_idx]['PRECIPITACAO_TOTAL']
        else:
            precip_day_after = np.nan

        window_scores.append({
            'window_idx': window_idx,
            'end_date': end_date,
            'mean_precip_in_window': mean_precip,
            'mean_humidity_in_window': mean_humidity,
            'composite_score': score,
            'precip_day_after': precip_day_after,
        })

    # Convert to DataFrame and sort by composite score
    df_scores = pd.DataFrame(window_scores)
    df_scores = df_scores.sort_values('composite_score', ascending=False)

    print(f"    ✓ Scores calculated and sorted")

    # Step 7: Display top N windows
    print(f"\n[6] Top {top_n} Windows in Cluster {cluster_target}:")
    print(f"    (Ranked by PRECIPITACAO_TOTAL + UMIDADE_MIN)")
    print()

    print(f"    {'Rank':<6} {'End Date':<12} {'Precip in Window':<20} {'Humidity in Window':<20} {'Composite Score':<20} {'Precip Day After':<20}")
    print(f"    {'-' * 105}")

    for rank, (_, row) in enumerate(df_scores.head(top_n).iterrows(), 1):
        print(f"    {rank:<6} {str(row['end_date'].date()):<12} {row['mean_precip_in_window']:>19.4f} {row['mean_humidity_in_window']:>19.4f} {row['composite_score']:>19.4f} {row['precip_day_after']:>19.4f}")

    # Step 8: Save to CSV
    print(f"\n[7] Saving results...")
    output_dir = OUTPUTS_DIR / "cluster_3_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    df_scores.to_csv(output_dir / "cluster_3_all_windows.csv", index=False)
    print(f"    ✓ All windows saved: {output_dir / 'cluster_3_all_windows.csv'}")

    # Save top 30
    df_scores.head(top_n).to_csv(output_dir / f"cluster_3_top_{top_n}_windows.csv", index=False)
    print(f"    ✓ Top {top_n} saved: {output_dir / f'cluster_3_top_{top_n}_windows.csv'}")

    # Step 9: Create visualization
    print(f"\n[8] Creating visualizations...")

    # Plot 1: Scatter plot of top 30
    fig, ax = plt.subplots(figsize=(12, 8))

    top_30 = df_scores.head(top_n)
    scatter = ax.scatter(top_30['mean_precip_in_window'],
                        top_30['mean_humidity_in_window'],
                        c=top_30['composite_score'],
                        s=100,
                        cmap='viridis',
                        alpha=0.6,
                        edgecolors='black',
                        linewidth=1)

    # Add rank labels
    for rank, (_, row) in enumerate(top_30.iterrows(), 1):
        ax.annotate(str(rank),
                   (row['mean_precip_in_window'], row['mean_humidity_in_window']),
                   fontsize=8, ha='center', va='center', fontweight='bold')

    ax.set_xlabel('Mean PRECIPITACAO_TOTAL (normalized)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean UMIDADE_MIN (normalized)', fontsize=12, fontweight='bold')
    ax.set_title(f'Cluster {cluster_target}: Top {top_n} Windows\n(by PRECIPITACAO_TOTAL + UMIDADE_MIN)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Composite Score', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "cluster_3_top_30_scatter.png", dpi=300, bbox_inches='tight')
    print(f"    ✓ Scatter plot saved: {output_dir / 'cluster_3_top_30_scatter.png'}")
    plt.close()

    # Plot 2: Bar chart of top 15 by rank
    fig, ax = plt.subplots(figsize=(12, 8))

    top_15 = df_scores.head(15)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_15)))

    bars = ax.bar(range(len(top_15)),
                  top_15['composite_score'].values,
                  color=colors,
                  edgecolor='black',
                  linewidth=1)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_15['composite_score'].values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Composite Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Cluster {cluster_target}: Top 15 Windows by Composite Score',
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(top_15)))
    ax.set_xticklabels([f"#{i+1}" for i in range(len(top_15))], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "cluster_3_top_15_bars.png", dpi=300, bbox_inches='tight')
    print(f"    ✓ Bar chart saved: {output_dir / 'cluster_3_top_15_bars.png'}")
    plt.close()

    # Step 10: Create summary report
    print(f"\n[9] Creating summary report...")

    with open(output_dir / "cluster_3_analysis_report.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Cluster {cluster_target} - Top {top_n} Windows Analysis\n")
        f.write("=" * 80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Station: {state}/{station_id}\n")
        f.write(f"  Window Size: {window_size} days\n")
        f.write(f"  Number of Clusters: {n_clusters}\n")
        f.write(f"  Cluster Target: {cluster_target}\n")
        f.write(f"  Total windows in Cluster {cluster_target}: {len(cluster_indices)}\n")
        f.write(f"  Date range: {df['Data'].min().date()} to {df['Data'].max().date()}\n\n")

        f.write("Scoring Method:\n")
        f.write(f"  Composite Score = Mean(PRECIPITACAO_TOTAL) + Mean(UMIDADE_MIN) over window\n")
        f.write(f"  Both values are normalized (z-score normalized)\n\n")

        f.write("=" * 80 + "\n")
        f.write(f"Top {top_n} Windows Ranked by Composite Score\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Rank':<6} {'End Date':<15} {'Precip Window':<20} {'Humidity Window':<20} {'Composite':<20} {'Precip After':<20}\n")
        f.write(f"{'-' * 105}\n")

        for rank, (_, row) in enumerate(df_scores.head(top_n).iterrows(), 1):
            f.write(f"{rank:<6} {str(row['end_date'].date()):<15} {row['mean_precip_in_window']:>19.4f} {row['mean_humidity_in_window']:>19.4f} {row['composite_score']:>19.4f} {row['precip_day_after']:>19.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Statistics\n")
        f.write("=" * 80 + "\n\n")

        f.write("Composite Score Statistics (all windows in Cluster 3):\n")
        f.write(f"  Mean: {df_scores['composite_score'].mean():.4f}\n")
        f.write(f"  Std:  {df_scores['composite_score'].std():.4f}\n")
        f.write(f"  Min:  {df_scores['composite_score'].min():.4f}\n")
        f.write(f"  Max:  {df_scores['composite_score'].max():.4f}\n\n")

        f.write("Top 30 Composite Score Statistics:\n")
        f.write(f"  Mean: {df_scores.head(top_n)['composite_score'].mean():.4f}\n")
        f.write(f"  Std:  {df_scores.head(top_n)['composite_score'].std():.4f}\n")
        f.write(f"  Min:  {df_scores.head(top_n)['composite_score'].min():.4f}\n")
        f.write(f"  Max:  {df_scores.head(top_n)['composite_score'].max():.4f}\n\n")

        f.write("Precipitation Day After (for top 30):\n")
        top_30_precip = df_scores.head(top_n)['precip_day_after'].dropna()
        if len(top_30_precip) > 0:
            f.write(f"  Mean: {top_30_precip.mean():.4f} mm\n")
            f.write(f"  Std:  {top_30_precip.std():.4f} mm\n")
            f.write(f"  Min:  {top_30_precip.min():.4f} mm\n")
            f.write(f"  Max:  {top_30_precip.max():.4f} mm\n")

    print(f"    ✓ Report saved: {output_dir / 'cluster_3_analysis_report.txt'}")

    print("\n" + "=" * 80)
    print("✓ Cluster 3 analysis complete!")
    print(f"✓ Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

