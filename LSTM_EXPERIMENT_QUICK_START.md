# LSTM Clustering Experiment - Quick Start Guide

## What This Does

This experiment combines **spectral clustering** and **LSTM deep learning** to predict daily precipitation at the RS A801 weather station. 

- **Clustering**: Groups 15-day weather patterns into 5 meteorological regimes
- **LSTM Model**: Learns to predict next-day precipitation from these patterns
- **Evaluation**: Comprehensive metrics and visualizations

## Installation (One-time setup)

```bash
# Install dependencies
pip install -r requirements.txt
pip install tensorflow keras scikit-learn pandas matplotlib seaborn
```

## Running the Experiment

```bash
cd experiments
python lstm_cluster.py
```

**Expected time**: 5-10 minutes on CPU, 1-2 minutes on GPU

## Output Files

All results go to: `outputs/lstm_cluster_RS_A801_w15_k5/`

### Visualizations
- `01_training_history.png` - How model learns over time
- `02_predictions_vs_actual.png` - Forecast vs observations
- `03_residuals_analysis.png` - Prediction errors
- `04_error_by_magnitude.png` - Error patterns by rain intensity
- `05_cluster_performance.png` - Performance per weather pattern
- `06_cluster_distribution.png` - Sample counts per cluster

### Reports & Data
- `evaluation_report.txt` - Detailed text report with all metrics
- `metrics.csv` - Train/Val/Test metrics in table format
- `test_predictions.csv` - All predictions with residuals and cluster assignments
- `experiment_config.txt` - Configuration used for this run

## Key Metrics to Look For

| Metric | What It Means | Good Value |
|--------|---------------|-----------|
| **RMSE** | Average prediction error (mm) | < 10% of mean rainfall |
| **MAE** | Typical absolute error (mm) | Lower is better |
| **R²** | How much variance explained (0-1) | > 0.6 is good |
| **Rainy Days MAE** | Error on rainy days only (mm) | Critical for forecasting |

## Configuration

Edit these parameters in `experiments/lstm_cluster.py`:

```python
# Data
WINDOW_SIZE = 15        # Change to 5-30 to adjust pattern window
N_CLUSTERS = 5          # Change to 3-10 to adjust cluster count

# Model
LSTM_UNITS = 64         # First LSTM layer size
LSTM_UNITS_2 = 32       # Second LSTM layer size
EPOCHS = 50             # Training iterations
BATCH_SIZE = 32         # Samples per batch

# Training
EARLY_STOPPING = True   # Stop if no improvement
PATIENCE = 10           # Epochs to wait before stopping
```

## Common Issues

**"ModuleNotFoundError: No module named 'tensorflow'"**
```bash
pip install tensorflow keras
```

**"No data found"**
- Ensure data exists: `data/inmet/RS/A801/`
- Check file permissions

**Poor results?**
- Try different `WINDOW_SIZE` (5-30 days)
- Adjust `N_CLUSTERS` (3-10)
- Increase `EPOCHS` (50-200)
- Lower `LEARNING_RATE` (0.001 → 0.0001)

## Understanding the Output

### Example evaluation_report.txt snippet:
```
TEST SET METRICS - PRIMARY EVALUATION
RMSE:    5.1234
MAE:     2.3456
R2:      0.7123

ZERO vs RAINY DAYS ANALYSIS
Zero days: 156 (62.4%)
Rainy days (≥1mm): 94
  Rainy days MAE: 4.1234 mm
  Rainy days RMSE: 6.2345 mm

PERFORMANCE BY CLUSTER
  Cluster 0 (47 samples):
    MAE:  2.5432 mm
    RMSE: 3.8765 mm
    R²:   0.7234
```

### What to check:
1. ✓ Is RMSE reasonable? (typically < 5-10 mm for precipitation)
2. ✓ Is R² > 0.6? (explains most of the variance)
3. ✓ Do rainy days have higher MAE? (harder to predict heavy rain)
4. ✓ Are clusters balanced in performance? (no single cluster dominates)

## Next Steps

### For Better Results
1. **Multi-step forecasting**: Predict 2-7 days ahead instead of just next day
2. **Cluster-specific models**: Train separate LSTM for each weather pattern
3. **Weighted loss**: Penalize errors on rainy days more heavily
4. **Feature engineering**: Add lag features (yesterday's rain, etc.)

### For Production Use
1. **Automatic retraining**: Update model with new data monthly/quarterly
2. **Uncertainty quantification**: Add confidence intervals
3. **Real-time pipeline**: Stream new data and generate daily forecasts
4. **Model monitoring**: Track performance over time

## Reference

For detailed information, see: `LSTM_CLUSTERING_EXPERIMENT.md`

---

**Quick Links**
- Main script: `experiments/lstm_cluster.py`
- LSTM code: `src/climate_cluster/pipeline/lstm.py`
- Evaluation code: `src/climate_cluster/evaluation/metrics.py`
- Documentation: `LSTM_CLUSTERING_EXPERIMENT.md`

