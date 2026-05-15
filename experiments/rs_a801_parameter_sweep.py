"""Parameter sweep experiment for RS A801 precipitation clustering analysis.

This script sweeps over multiple parameters:
- Window sizes: 5 to 14 days
- Number of clusters: 5 to 10
- Sigma values: Common values from literature [0.1, 0.5, 1.0, 2.0, 5.0]

For each combination, it:
1. Creates windows and clusters the data
2. Analyzes cluster patterns before high precipitation events
3. Records results to a comprehensive results file
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from climate_cluster.config import DATA_ROOT, OUTPUTS_DIR
from climate_cluster.config_data import load_single_station
from climate_cluster.features.window_features import create_windows


def find_window_ending_before_date(df, target_date, window_size):
    """Find the window index that ends on the day before target_date."""
    target_date = pd.Timestamp(target_date)
    day_before = target_date - pd.Timedelta(days=1)

    df_dates = df['Data'].dt.normalize()
    mask = df_dates == day_before

    if not mask.any():
        return None, None

    end_pos = df.index[mask][0]
    window_start_pos = end_pos - window_size + 1

    if window_start_pos < 0:
        return None, None

    return window_start_pos, day_before


def analyze_parameter_combination(df, windows, labels, window_size, n_clusters, sigma, top_precip_days):
    """Analyze a single parameter combination and return statistics."""

    results = []

    for date_idx, row in top_precip_days.iterrows():
        date = row['Data']
        precip = row['PRECIPITACAO_TOTAL']

        window_idx, window_end_date = find_window_ending_before_date(df, date, window_size)

        if window_idx is None:
            cluster = None
        else:
            cluster = labels[window_idx]

        results.append({
            'date': date.date(),
            'precipitation_mm': precip,
            'cluster': cluster,
        })

    results_df = pd.DataFrame(results)
    valid_results = results_df[results_df['cluster'].notna()]

    if len(valid_results) == 0:
        cluster_distribution = {}
        cluster_2_percentage = None
    else:
        cluster_counts = valid_results['cluster'].value_counts().to_dict()
        cluster_distribution = {int(k): int(v) for k, v in cluster_counts.items()}

        # Calculate percentage of most common cluster (for pattern analysis)
        most_common_count = max(cluster_counts.values()) if cluster_counts else 0
        cluster_2_percentage = (most_common_count / len(valid_results) * 100) if len(valid_results) > 0 else 0

    return {
        'window_size': window_size,
        'n_clusters': n_clusters,
        'sigma': sigma,
        'total_windows': len(windows),
        'valid_results': len(valid_results),
        'total_events_analyzed': len(results),
        'cluster_distribution': cluster_distribution,
        'max_cluster_percentage': cluster_2_percentage,
    }


def main():
    """Run parameter sweep experiment."""
    # Configuration for sweep
    state = "RS"
    station_id = "A801"

    window_sizes = list(range(5, 31))  # 5 to 30
    n_clusters_range = list(range(5, 16))  # 5 to 15
    sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]  # Common values from literature

    print("=" * 100)
    print("RS A801 PARAMETER SWEEP EXPERIMENT")
    print("=" * 100)
    print(f"\nConfiguration:")
    print(f"  Station: {state}/{station_id}")
    print(f"  Window sizes: {window_sizes[0]} to {window_sizes[-1]}")
    print(f"  Number of clusters: {n_clusters_range[0]} to {n_clusters_range[-1]}")
    print(f"  Sigma values: {sigmas}")
    print(f"  Total combinations: {len(window_sizes) * len(n_clusters_range) * len(sigmas)}")

    # Load data once
    print(f"\n[1] Loading {state}/{station_id} station data...")
    df = load_single_station(state=state, station_id=station_id, data_root=DATA_ROOT)
    print(f"    ✓ Loaded {len(df)} days of data")

    # Get top 10 precipitation days (constant across all combinations)
    print(f"\n[2] Finding 10 days with highest precipitation...")
    top_precip_days = df.nlargest(10, "PRECIPITACAO_TOTAL")
    print(f"    ✓ Found 10 events")

    # Store all results
    all_results = []

    # Parameter sweep
    print(f"\n[3] Running parameter sweep ({len(window_sizes) * len(n_clusters_range) * len(sigmas)} combinations)...")

    total_combinations = len(window_sizes) * len(n_clusters_range) * len(sigmas)
    current = 0

    for window_size in window_sizes:
        # Create windows once per window size
        if len(df) < window_size:
            print(f"    ⚠️  Window size {window_size} > dataset size, skipping")
            continue

        windows, scaler = create_windows(df, window_size=window_size, normalize=True)

        windows_flat = windows.reshape(windows.shape[0], -1)

        for n_clusters in n_clusters_range:
            for sigma in sigmas:
                current += 1

                # Progress indicator
                pct = 100.0 * current / total_combinations
                print(f"    [{current:3d}/{total_combinations}] {pct:5.1f}% - "
                      f"window={window_size}, k={n_clusters}, σ={sigma}...", end='')

                # Cluster
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(windows_flat)

                # Analyze
                result = analyze_parameter_combination(
                    df, windows, labels, window_size, n_clusters, sigma, top_precip_days
                )
                all_results.append(result)

                print(" ✓")

    # Create results dataframe
    print(f"\n[4] Processing results...")
    results_df = pd.DataFrame(all_results)

    # Expand cluster distribution to separate columns
    for cluster_id in range(max(n_clusters_range)):
        results_df[f'cluster_{cluster_id}_count'] = results_df['cluster_distribution'].apply(
            lambda x: x.get(cluster_id, 0) if isinstance(x, dict) else 0
        )

    # Drop the dict column
    results_df = results_df.drop('cluster_distribution', axis=1)

    # Rename for clarity
    results_df = results_df.rename(columns={
        'window_size': 'Window_Size',
        'n_clusters': 'N_Clusters',
        'sigma': 'Sigma',
        'total_windows': 'Total_Windows',
        'valid_results': 'Valid_Events',
        'total_events_analyzed': 'Total_Events',
        'max_cluster_percentage': 'Max_Cluster_Percentage',
    })

    # Save results
    output_dir = OUTPUTS_DIR / "rs_a801_parameter_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "parameter_sweep_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"    ✓ Results saved to: {results_path}")

    # Create a summary report
    summary_path = output_dir / "parameter_sweep_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("RS A801 PARAMETER SWEEP EXPERIMENT - SUMMARY\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Station: {state}/{station_id}\n")
        f.write(f"Data Period: {df['Data'].min().date()} to {df['Data'].max().date()}\n")
        f.write(f"Total Days: {len(df)}\n\n")

        f.write("PARAMETER RANGES:\n")
        f.write(f"  Window Sizes: {min(window_sizes)} to {max(window_sizes)} days\n")
        f.write(f"  Number of Clusters: {min(n_clusters_range)} to {max(n_clusters_range)}\n")
        f.write(f"  Sigma Values: {sigmas}\n")
        f.write(f"  Total Combinations: {len(results_df)}\n\n")

        f.write("RESULTS SUMMARY:\n")
        f.write(f"  Total events analyzed per combination: 10 (top precipitation days)\n")
        f.write(f"  Valid results per combination: {results_df['Valid_Events'].describe().to_string()}\n\n")

        f.write("TOP 5 PARAMETER COMBINATIONS BY MAX CLUSTER PERCENTAGE:\n")
        f.write("(This indicates how well parameters capture patterns before high precipitation)\n\n")

        top_5 = results_df.nlargest(5, 'Max_Cluster_Percentage')
        for idx, (_, row) in enumerate(top_5.iterrows(), 1):
            f.write(f"{idx}. Window={int(row['Window_Size']):2d}, "
                   f"Clusters={int(row['N_Clusters']):2d}, "
                   f"Sigma={row['Sigma']:4.1f}, "
                   f"Max Cluster %={row['Max_Cluster_Percentage']:6.2f}%\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("DETAILED RESULTS BY WINDOW SIZE:\n")
        f.write("=" * 100 + "\n\n")

        for ws in sorted(results_df['Window_Size'].unique()):
            ws_results = results_df[results_df['Window_Size'] == ws]
            f.write(f"\nWindow Size: {int(ws)} days ({len(ws_results)} combinations)\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'K':<3} {'Sigma':<8} {'Windows':<10} {'Valid':<8} {'Max%':<8}\n")
            f.write("-" * 100 + "\n")

            for _, row in ws_results.iterrows():
                f.write(f"{int(row['N_Clusters']):<3} {row['Sigma']:<8.1f} "
                       f"{int(row['Total_Windows']):<10} {int(row['Valid_Events']):<8} "
                       f"{row['Max_Cluster_Percentage']:>7.2f}%\n")

    print(f"    ✓ Summary saved to: {summary_path}")

    # Print summary to console
    print("\n" + "=" * 100)
    print("EXPERIMENT COMPLETE")
    print("=" * 100)
    print(f"\nTotal parameter combinations tested: {len(results_df)}")
    print(f"Average max cluster percentage: {results_df['Max_Cluster_Percentage'].mean():.2f}%")
    print(f"Best max cluster percentage: {results_df['Max_Cluster_Percentage'].max():.2f}%")
    print(f"Worst max cluster percentage: {results_df['Max_Cluster_Percentage'].min():.2f}%")

    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - parameter_sweep_results.csv (full results)")
    print(f"  - parameter_sweep_summary.txt (analysis summary)")

    # Show top combinations
    print("\nTOP 5 BEST PARAMETER COMBINATIONS:")
    top_5 = results_df.nlargest(5, 'Max_Cluster_Percentage')
    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"  {idx}. Window={int(row['Window_Size']):2d}, "
              f"Clusters={int(row['N_Clusters']):2d}, "
              f"σ={row['Sigma']:4.1f} "
              f"→ Max Cluster Percentage: {row['Max_Cluster_Percentage']:.2f}%")


if __name__ == "__main__":
    main()

