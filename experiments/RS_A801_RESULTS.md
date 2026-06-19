# Experiments - RS A801 Precipitation Clustering Analysis

## Overview

This experiment analyzes weather patterns preceding high-precipitation events at the RS A801 weather station in Rio Grande do Sul, Brazil.

## Methodology

1. **Data Loading**: Loaded all available daily data from RS A801 station (9,497 days from 2000-2025)
2. **High Precipitation Identification**: Identified the 10 days with the highest total precipitation
3. **Window Creation**: Created 4-day sliding windows from the daily data (9,494 total windows)
4. **Clustering**: Applied KMeans clustering (k=3) on normalized window features
5. **Analysis**: For each high-precipitation day, retrieved the cluster assignment of the window ending the day before

## Key Findings

### Top 10 High Precipitation Days and Preceding Window Clusters

| Rank | Date       | Precipitation | Window Ends | Cluster | Window Index |
|------|-----------|----------------|-------------|---------|-------------|
| 1    | 2024-05-23 | 126.2 mm      | 2024-05-22  | **2**   | 8905        |
| 2    | 2013-11-11 | 124.0 mm      | 2013-11-10  | **0**   | 5059        |
| 3    | 2023-06-16 | 122.6 mm      | 2023-06-15  | **2**   | 8563        |
| 4    | 2020-06-30 | 112.0 mm      | 2020-06-29  | **2**   | 7482        |
| 5    | 2025-06-18 | 109.2 mm      | 2025-06-17  | **2**   | 9296        |
| 6    | 2024-04-30 | 109.0 mm      | 2024-04-29  | **2**   | 8882        |
| 7    | 2007-06-10 | 106.4 mm      | 2007-06-09  | **2**   | 2713        |
| 8    | 2014-07-04 | 103.6 mm      | 2014-07-03  | **2**   | 5294        |
| 9    | 2024-05-02 | 94.4 mm       | 2024-05-01  | **2**   | 8884        |
| 10   | 2025-08-22 | 92.4 mm       | 2025-08-21  | **2**   | 9361        |

### Cluster Distribution for Pre-Precipitation Windows

- **Cluster 0**: 1 window (10.0%)
- **Cluster 2**: 9 windows (90.0%)
- **Cluster 1**: 0 windows (0.0%)

### Overall Dataset Clustering

- **Cluster 0**: 4,387 windows (46.2%)
- **Cluster 1**: 1,063 windows (11.2%)
- **Cluster 2**: 4,044 windows (42.6%)

## Interpretation

The analysis reveals a **strong association between Cluster 2 and high precipitation events**:

- **90% of high-precipitation events** (9 out of 10) are preceded by windows assigned to Cluster 2
- Only 1 event (2013-11-11) breaks this pattern, preceded by a Cluster 0 window
- This suggests that **Cluster 2 represents meteorological conditions that precede heavy rainfall**

### Meteorological Implications

Cluster 2 characteristics may include:
- Specific temperature, humidity, and pressure patterns
- Wind speed and direction combinations
- Atmospheric conditions conducive to precipitation

## Technical Details

- **Station**: RS/A801 (Rio Grande do Sul)
- **Data Period**: 2000-01-01 to 2025-12-31 (9,497 days)
- **Window Size**: 4 consecutive days
- **Features**: All 11 INMET climate variables (normalized)
- **Clustering Algorithm**: KMeans (k=3)
- **Total Windows**: 9,494

## Output Files

- `precipitation_cluster_results.csv`: Tabular results of the analysis
- `analysis_report.txt`: Detailed text report
- This summary document

## Conclusions

The strong correlation between Cluster 2 and high-precipitation events suggests that:

1. **Weather patterns are predictable**: The 4-day window preceding high precipitation shows consistent clustering behavior
2. **Cluster 2 is a precipitation indicator**: This cluster appears to represent favorable atmospheric conditions for heavy rainfall
3. **Statistical significance**: With 90% of the top 10 precipitation events showing this pattern, it's unlikely to be random

## Next Steps

Potential improvements to this analysis:

1. Investigate the specific climate features that define Cluster 2
2. Test predictive capability: use this pattern to forecast precipitation
3. Apply spectral clustering (slow but potentially more accurate) for smaller datasets
4. Analyze other stations to see if this pattern is universal
5. Extend window analysis backwards 2, 3, 5, and 10 days to see if pattern extends

## Files

- Main script: `rs_a801_precipitation_clusters.py`
- Results: `outputs/rs_a801_precip_analysis/`

