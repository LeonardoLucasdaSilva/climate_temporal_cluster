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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_ROOT, OUTPUTS_DIR
from data.load_data import load_station_daily_data
from methods.cluster.cluster_pipeline import (
    PCA_VARIANCE_THRESHOLD,
    cluster_feature_matrix,
    create_cluster_feature_matrix,
)
from methods.tools.sigma_choosing import calculate_sigma_values


CLUSTERING_ALGORITHM = "spectral"  # Options: "kmeans", "spectral"

# Raw data subset. Use None to analyze the full station history.
# Examples:
#   MAX_RAW_DAYS = 2000, RAW_DATA_SELECTION = "recent"
#   RAW_DATA_START = "2018-01-01", RAW_DATA_END = "2022-12-31"
MAX_RAW_DAYS = 1000
RAW_DATA_SELECTION = "recent"  # Options: "recent", "earliest"
RAW_DATA_START = None  # Optional YYYY-MM-DD string
RAW_DATA_END = None  # Optional YYYY-MM-DD string


def select_raw_data(
    df,
    max_days=None,
    selection="recent",
    start_date=None,
    end_date=None,
):
    """Select the raw daily period used for sigma, windows, and event analysis."""
    selected = df.sort_values("Data").reset_index(drop=True).copy()

    if start_date is not None:
        selected = selected[selected["Data"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        selected = selected[selected["Data"] <= pd.Timestamp(end_date)]

    if max_days is not None:
        if max_days <= 0:
            raise ValueError(f"MAX_RAW_DAYS must be positive or None, got {max_days}")
        if selection == "recent":
            selected = selected.tail(max_days)
        elif selection == "earliest":
            selected = selected.head(max_days)
        else:
            raise ValueError(
                f"RAW_DATA_SELECTION must be 'recent' or 'earliest', got {selection!r}"
            )

    return selected.reset_index(drop=True)


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

    # Calculate cluster sizes as percentages of total windows
    cluster_sizes = {}
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_windows = len(labels)
    for label, count in zip(unique_labels, counts):
        cluster_sizes[int(label)] = (count / total_windows) * 100

    return {
        'window_size': window_size,
        'n_clusters': n_clusters,
        'sigma': sigma,
        'total_windows': len(windows),
        'valid_results': len(valid_results),
        'total_events_analyzed': len(results),
        'cluster_distribution': cluster_distribution,
        'max_cluster_percentage': cluster_2_percentage,
        'cluster_sizes': cluster_sizes,
    }


def format_sigma(sigma):
    """Format sigma values for logs and reports."""
    return "N/A" if sigma is None or pd.isna(sigma) else f"{float(sigma):.4g}"


def main():
    """Run parameter sweep experiment."""
    # Configuration for sweep
    state = "RS"
    station_id = "A801"

    window_sizes = list(range(5, 31))  # 5 to 30
    n_clusters_range = list(range(5, 16))  # 5 to 15

    print("=" * 100)
    print("RS A801 PARAMETER SWEEP EXPERIMENT")
    print("=" * 100)

    # Load data once
    print(f"\n[1] Loading {state}/{station_id} station data...")
    df_full = load_station_daily_data(state=state, station_id=station_id, data_root=DATA_ROOT)
    df = select_raw_data(
        df_full,
        max_days=MAX_RAW_DAYS,
        selection=RAW_DATA_SELECTION,
        start_date=RAW_DATA_START,
        end_date=RAW_DATA_END,
    )
    print(f"    ✓ Loaded {len(df_full)} days of data")
    if len(df) != len(df_full):
        print(
            f"    ✓ Analyzing {len(df)} selected days "
            f"({df['Data'].min().date()} to {df['Data'].max().date()})"
        )
    else:
        print(f"    ✓ Analyzing full period ({df['Data'].min().date()} to {df['Data'].max().date()})")

    max_window_size = max(window_sizes)
    max_windows = max(0, len(df) - max_window_size + 1)
    if max_windows == 0:
        raise ValueError(
            f"Selected raw data has {len(df)} days, but the largest window size is "
            f"{max_window_size}. Increase MAX_RAW_DAYS or lower window_sizes."
        )
    print(
        f"    Largest clustering matrix estimate: {max_windows:,} x {max_windows:,} "
        f"for window_size={max_window_size}"
    )

    # Calculate sigma values based on data distribution
    if CLUSTERING_ALGORITHM.lower() == "spectral":
        print(f"\n[2] Calculating optimal sigma values...")
        print(f"    Creating distance matrix and extracting sigma values...")
        sigmas = calculate_sigma_values(df, n_values=20)
        print(f"    ✓ Generated {len(sigmas)} sigma values")
        print(f"    σ range: [{sigmas.min():.6f}, {sigmas.max():.6f}]")
    else:
        print(f"\n[2] Skipping sigma calculation ({CLUSTERING_ALGORITHM.upper()} does not use sigma)...")
        sigmas = [None]

    print(f"\nConfiguration:")
    print(f"  Station: {state}/{station_id}")
    print(f"  Raw days analyzed: {len(df)} of {len(df_full)}")
    print(f"  Raw data selection: {RAW_DATA_SELECTION}")
    print(f"  Date filter: {RAW_DATA_START or 'start'} to {RAW_DATA_END or 'end'}")
    print(f"  Window sizes: {window_sizes[0]} to {window_sizes[-1]}")
    print(f"  Number of clusters: {n_clusters_range[0]} to {n_clusters_range[-1]}")
    print(f"  Clustering method: {CLUSTERING_ALGORITHM.upper()}")
    print(f"  PCA variance threshold: {PCA_VARIANCE_THRESHOLD:.2f}")
    if CLUSTERING_ALGORITHM.lower() == "spectral":
        print(f"  Sigma values: {len(sigmas)} values from distance-based distribution")
    else:
        print(f"  Sigma values: not used")
    print(f"  Total combinations: {len(window_sizes) * len(n_clusters_range) * len(sigmas)}")

    # Get top 10 precipitation days (constant across all combinations)
    print(f"\n[3] Finding 10 days with highest precipitation...")
    top_precip_days = df.nlargest(10, "PRECIPITACAO_TOTAL")
    print(f"    ✓ Found 10 events")

    # Store all results
    all_results = []

    # Parameter sweep
    print(f"\n[4] Running parameter sweep ({len(window_sizes) * len(n_clusters_range) * len(sigmas)} combinations)...")

    total_combinations = len(window_sizes) * len(n_clusters_range) * len(sigmas)
    current = 0

    for window_size in window_sizes:
        # Create windows once per window size
        if len(df) < window_size:
            print(f"    ⚠️  Window size {window_size} > dataset size, skipping")
            continue

        windows, windows_flat, scaler, pca, feature_columns = create_cluster_feature_matrix(
            df,
            window_size=window_size,
            normalize=True,
            variance_threshold=PCA_VARIANCE_THRESHOLD,
        )

        for n_clusters in n_clusters_range:
            for sigma in sigmas:
                current += 1

                # Progress indicator
                pct = 100.0 * current / total_combinations
                print(f"    [{current:3d}/{total_combinations}] {pct:5.1f}% - "
                      f"window={window_size}, k={n_clusters}, σ={format_sigma(sigma)}...", end='')

                labels = cluster_feature_matrix(
                    windows_flat,
                    n_clusters=n_clusters,
                    algorithm=CLUSTERING_ALGORITHM,
                    sigma=sigma,
                    random_state=42,
                )

                # Analyze
                result = analyze_parameter_combination(
                    df, windows, labels, window_size, n_clusters, sigma, top_precip_days
                )
                all_results.append(result)

                print(" ✓")

    # Create results dataframe
    print(f"\n[5] Processing results...")
    results_df = pd.DataFrame(all_results)

    # Expand cluster distribution to separate columns
    for cluster_id in range(max(n_clusters_range)):
        results_df[f'cluster_{cluster_id}_count'] = results_df['cluster_distribution'].apply(
            lambda x: x.get(cluster_id, 0) if isinstance(x, dict) else 0
        )

    # Expand cluster sizes (percentages) to separate columns
    for cluster_id in range(max(n_clusters_range)):
        results_df[f'cluster_{cluster_id}_size_pct'] = results_df['cluster_sizes'].apply(
            lambda x: x.get(cluster_id, 0.0) if isinstance(x, dict) else 0.0
        )

    # Drop the dict columns
    results_df = results_df.drop(['cluster_distribution', 'cluster_sizes'], axis=1)

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
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("RS A801 PARAMETER SWEEP EXPERIMENT - SUMMARY\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Station: {state}/{station_id}\n")
        f.write(f"Full Data Period: {df_full['Data'].min().date()} to {df_full['Data'].max().date()}\n")
        f.write(f"Full Data Days: {len(df_full)}\n")
        f.write(f"Data Period: {df['Data'].min().date()} to {df['Data'].max().date()}\n")
        f.write(f"Analyzed Days: {len(df)}\n")
        f.write(f"MAX_RAW_DAYS: {MAX_RAW_DAYS}\n")
        f.write(f"RAW_DATA_SELECTION: {RAW_DATA_SELECTION}\n")
        f.write(f"RAW_DATA_START: {RAW_DATA_START}\n")
        f.write(f"RAW_DATA_END: {RAW_DATA_END}\n\n")

        f.write("PARAMETER RANGES:\n")
        f.write(f"  Window Sizes: {min(window_sizes)} to {max(window_sizes)} days\n")
        f.write(f"  Number of Clusters: {min(n_clusters_range)} to {max(n_clusters_range)}\n")
        f.write(f"  Clustering Method: {CLUSTERING_ALGORITHM.upper()}\n")
        f.write(f"  PCA Variance Threshold: {PCA_VARIANCE_THRESHOLD:.2f}\n")
        if CLUSTERING_ALGORITHM.lower() == "spectral":
            f.write(f"  Sigma Values: {sigmas}\n")
        else:
            f.write("  Sigma Values: not used\n")
        f.write(f"  Total Combinations: {len(results_df)}\n\n")

        f.write("RESULTS SUMMARY:\n")
        f.write(f"  Total events analyzed per combination: 10 (top precipitation days)\n")
        f.write(f"  Valid results per combination: {results_df['Valid_Events'].describe().to_string()}\n\n")

        f.write("TOP 25 PARAMETER COMBINATIONS BY MAX CLUSTER PERCENTAGE:\n")
        f.write("(This indicates how well parameters capture patterns before high precipitation)\n\n")
        top_25 = results_df.nlargest(25, 'Max_Cluster_Percentage')
        for idx, (_, row) in enumerate(top_25.iterrows(), 1):
            # Find which cluster has the maximum event count (dominant cluster for precipitation)
            max_cluster_id = None
            max_event_count = 0
            max_cluster_pct = 0
            for cluster_id in range(int(row['N_Clusters'])):
                count_col = f'cluster_{cluster_id}_count'
                if count_col in row.index and row[count_col] > max_event_count:
                    max_event_count = row[count_col]
                    max_cluster_id = cluster_id
                    size_col = f'cluster_{cluster_id}_size_pct'
                    max_cluster_pct = row[size_col] if size_col in row.index else 0

            # Get count of events in the dominant cluster
            count_col = f'cluster_{max_cluster_id}_count'
            event_count = row[count_col] if count_col in row.index else 0

            f.write(f"{idx}. Window={int(row['Window_Size']):2d}, "
                   f"Clusters={int(row['N_Clusters']):2d}, "
                   f"Sigma={format_sigma(row['Sigma'])}, "
                   f"Max Cluster %={row['Max_Cluster_Percentage']:6.2f}%\n")
            f.write(f"    └─ Cluster {max_cluster_id}: {event_count}/{int(row['Valid_Events'])} events "
                   f"({event_count/row['Valid_Events']*100:.1f}% of valid events)\n")
            f.write(f"    └─ Overall cluster size: {max_cluster_pct:.1f}% of all windows\n\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("DETAILED RESULTS BY WINDOW SIZE:\n")
        f.write("=" * 100 + "\n\n")

        for ws in sorted(results_df['Window_Size'].unique()):
            ws_results = results_df[results_df['Window_Size'] == ws]
            f.write(f"\nWindow Size: {int(ws)} days ({len(ws_results)} combinations)\n")
            f.write("-" * 100 + "\n")

            # Get max n_clusters for this window size to know how many cluster columns to show
            max_k = int(ws_results['N_Clusters'].max())

            # Header
            header = f"{'K':<3} {'Sigma':<8} {'Windows':<10} {'Valid':<8} {'Max%':<8}"
            for cluster_id in range(max_k):
                header += f" C{cluster_id}(%)"
            f.write(header + "\n")
            f.write("-" * 100 + "\n")

            for _, row in ws_results.iterrows():
                line = f"{int(row['N_Clusters']):<3} {format_sigma(row['Sigma']):<8} " \
                       f"{int(row['Total_Windows']):<10} {int(row['Valid_Events']):<8} " \
                       f"{row['Max_Cluster_Percentage']:>7.2f}%"

                # Add cluster size percentages
                for cluster_id in range(max_k):
                    cluster_size_col = f'cluster_{cluster_id}_size_pct'
                    if cluster_size_col in row.index:
                        line += f" {row[cluster_size_col]:>6.1f}%"
                    else:
                        line += f" {'0.0':>6}%"

                f.write(line + "\n")

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
    top_25 = results_df.nlargest(25, 'Max_Cluster_Percentage')
    for idx, (_, row) in enumerate(top_25.iterrows(), 1):
        if idx > 25:
            break

        # Find which cluster has the maximum event count (dominant cluster for precipitation)
        max_cluster_id = None
        max_event_count = 0
        max_cluster_pct = 0
        for cluster_id in range(int(row['N_Clusters'])):
            count_col = f'cluster_{cluster_id}_count'
            if count_col in row.index and row[count_col] > max_event_count:
                max_event_count = row[count_col]
                max_cluster_id = cluster_id
                size_col = f'cluster_{cluster_id}_size_pct'
                max_cluster_pct = row[size_col] if size_col in row.index else 0

        # Get count of events in the dominant cluster
        count_col = f'cluster_{max_cluster_id}_count'
        event_count = row[count_col] if count_col in row.index else 0

        print(f"  {idx}. Window={int(row['Window_Size']):2d}, "
              f"Clusters={int(row['N_Clusters']):2d}, "
              f"σ={format_sigma(row['Sigma'])} "
              f"→ Max Cluster Percentage: {row['Max_Cluster_Percentage']:.2f}%")
        print(f"      ├─ Cluster {max_cluster_id}: {event_count}/{int(row['Valid_Events'])} precipitation events")
        print(f"      └─ Cluster size: {max_cluster_pct:.1f}% of all windows\n")


if __name__ == "__main__":
    main()

