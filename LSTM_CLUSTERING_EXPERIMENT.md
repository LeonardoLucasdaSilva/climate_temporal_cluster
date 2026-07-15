# LSTM Clustering Experiment for Precipitation Prediction

## Overview

This document details a complete pipeline to predict next-day precipitation for station RS A801 using LSTM models guided by prior clustering of temporal patterns. We first apply PCA to windowed features built from all available numeric variables, then perform spectral clustering in the reduced space. With clusters defined, we train a separate LSTM model per cluster to better specialize on distinct weather regimes. The final evaluation aggregates cluster-wise predictions into overall metrics.

Key changes and clarifications implemented:
- PCA is applied before clustering (variance_threshold=0.95), making dimensionality reduction explicit and configurable.
- All numeric features are used by default, but the code allows easy replacement with a subset.
- Train/Validation/Test final proportions are 60% / 10% / 30% with stratification by cluster.
- One LSTM model is trained per cluster; predictions are aggregated across clusters for global metrics and plots.
- Comprehensive evaluation includes regression metrics, per-cluster performance,
  input precipitation distribution plots, and per-configuration LaTeX reports.

**Main Goal**: Leverage temporal weather patterns (via PCA + clustering) and cluster-specific LSTM models to improve precipitation forecasting accuracy.

---

## Pipeline Architecture

The complete pipeline consists of 10 sequential steps:

### Step 1: Data Loading & Preparation
- **File**: `src/data/load_data.py`
- **Input**: RS A801 station daily data from INMET
- **Output**: DataFrame with all available climate features
- **Key Statistics**:
  - Date range: Full historical data available
  - Features: All numeric columns (temperature, precipitation, humidity, pressure, etc.)
  - Data quality: Cleaned daily aggregated data

### Step 2: Feature Extraction via Sliding Windows
- **File**: `src/methods/tools/sliding_windows.py`
- **Method**: Create sliding windows of consecutive days
- **Parameters**:
  - Window size: 15 days (configurable)
  - Normalization: StandardScaler on each feature
  - Output shape: (n_windows, window_size, n_features)
- **Purpose**: Capture temporal dynamics of climate variables

### Step 3: Spectral Clustering
- **File**: `src/methods/cluster/ng.py`
- **Algorithm**: Normalized spectral clustering using Gaussian kernel
- **Parameters**:
  - Number of clusters: 5 (configurable)
  - Bandwidth (sigma): 1.0 (configurable)
  - Kernel: Gaussian/RBF
- **Process**:
  1. Flatten windows: (n_windows, window_size × n_features)
  2. Build affinity matrix using Gaussian kernel
  3. Compute normalized Laplacian
  4. Extract k largest eigenvectors
  5. Apply K-means on eigenvector space
- **Output**: Cluster assignments for each weather pattern window

### Step 4: Target Variable Preparation
- **Method**: Extract precipitation for the day after each window ends
- **Target**: PRECIPITACAO_TOTAL (daily precipitation in mm)
- **Window-Target Alignment**:
  - Window i: days [t, t+1, ..., t+window_size-1]
  - Target: precipitation on day t+window_size
- **Statistics**: Min, max, mean, std, and zero-day percentage

### Step 5: Data Stratification & Splitting
- **Strategy**: Stratified random split maintaining cluster distribution
- **Ratios**:
  - Training: 60%
  - Validation: 10%
  - Test: 30%
- **Implementation**: Two-stage stratified split
  1. Split test set from full data, keeping clusters balanced when possible
  2. Split validation set from training+validation, keeping clusters balanced when possible

### Step 6: LSTM Model Architecture
- **File**: `src/models/lstm.py`
- **Class**: `LSTMPrecipitationPredictor`
- **Architecture**:

```
Input (n_samples, 1, n_features)
    ↓
LSTM(64 units, activation='relu', return_sequences=True)
    ↓
Dropout(0.2)
    ↓
LSTM(32 units, activation='relu', return_sequences=False)
    ↓
Dropout(0.2)
    ↓
Dense(16, activation='relu')
    ↓
Dropout(0.2)
    ↓
Dense(8, activation='relu')
    ↓
Dense(1, activation='linear')  ← Precipitation prediction
```

- **Configuration**:
  - LSTM units: 64 (1st layer), 32 (2nd layer)
  - Dropout rate: 0.2 (regularization)
  - Optimizer: Adam (learning_rate=0.001)
  - Loss function: Mean Squared Error (MSE)
  - Metrics: MAE, MSE

- **Hyperparameters**:
  - Epochs: 50
  - Batch size: 32
  - Early stopping: Enabled (patience=10)

### Step 7: Model Training
- **Validation Data**: Used for early stopping
- **Early Stopping**: Restores best weights if val_loss doesn't improve for 10 epochs
- **Outputs**: Training history (loss, MAE over epochs)

### Step 8: Model Evaluation
- **Prediction Generation**: Apply trained model to train, val, and test sets
- **Evaluation Metrics** (see next section)

### Step 9: Comprehensive Visualization
- **Diagnostic plots** generated and saved:
  1. Training history (loss and MAE curves)
  2. Predictions vs actual values (time series + scatter)
  3. Residual analysis (temporal + distribution)
  4. Error by precipitation magnitude (10 bins)
  5. Performance metrics by cluster (MAE, RMSE)
  6. Cluster distribution in test set
  7. Test precipitation distribution by cluster
  8. Input-window next-day precipitation distribution by cluster
  9. Per-cluster actual, predicted, and residual histograms

### Step 10: Results & Reporting
- **Output Files**:
  - `evaluation_report.txt`: Detailed quantitative analysis
  - `metrics_summary.csv`: Train/Val/Test metrics table
  - `cluster_model_metrics.csv`: Per-cluster test metrics
  - `test_predictions.csv`: Predictions with residuals and cluster assignments
  - `input_next_day_precipitation_by_cluster.csv`: Next-day precipitation target
    assigned to every clustered input window
  - `summary.txt`: Full experiment configuration and compact metrics
  - `experiment_report.tex`: LaTeX report for the configuration
  - `experiment_report.pdf`: PDF report when a local LaTeX compiler is available
  - PNG plots with resolution controlled by `config_output.yaml`

---

## Evaluation Metrics

### Regression Metrics
Calculated for each data set (train, val, test):

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **MSE** | $\dfrac{1}{n}\sum(y_{true} - y_{pred})²$ | Mean squared error - penalizes large errors |
| **RMSE** | $\sqrt{mse}$ | Root mean squared error - same units as target |
| **MAE** | $\dfrac{1}{n}\sum\|y_{true} - y_{pred}$\| | Mean absolute error - robust to outliers |
| **RMSLE** | $\dfrac{\sum(log(y_{true}+1) - log(y_{pred}+1))²}{\sqrt{n}}$ | Root mean squared log error - for positive values |
| **R²** | 1 - $\dfrac{SS_{res}}{SS_{tot}}$ | Coefficient of determination (0-1 scale) |
| **MAPE** | $ \dfrac{1}{n} \sum \|\dfrac{y_{true} - y_p{pred}}{y_{true}}\| × 100$ |  Mean absolute percentage error |

### Precipitation-Specific Metrics
Account for high frequency of zero-precipitation days:

- **Zero Days Analysis**:
  - Count and percentage of dry days (0mm)
  - Separate MAE and RMSE for zero-precipitation days
  - Separate MAE, RMSE, and R² for rainy days (≥1mm)

### Cluster-Wise Performance
- **Per-Cluster Metrics**: MAE, RMSE, R² for each cluster
- **Purpose**: Identify clusters easier/harder to predict
- **Interpretation**: May reveal cluster-specific patterns

### Error Analysis by Magnitude
- **Bin Creation**: Partition test data by precipitation magnitude
- **Metrics by Bin**: MAE and RMSE for each bin
- **Purpose**: Understand if model performs differently for light vs heavy rain

---

## Key Implementation Details

### Feature Engineering
```python
# All available numeric features used
numeric_cols = [col for col in df.columns 
                if col != "Data" and pd.api.types.is_numeric_dtype(df[col])]

# Normalize each feature independently
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Create sliding windows
windows.shape = (n_windows, window_size, n_features)
```

### Clustering Decision
- **Why Spectral Clustering?**
  - Captures non-linear patterns in weather data
  - Works well with Gaussian kernels (smooth transitions)
  - Eigenvector approach captures global structure
  
- **Parameter Selection**:
  - **Sigma**: Controls bandwidth of Gaussian kernel
    - Smaller σ → sharper clusters, more sensitive
    - Larger σ → smoother affinity, broader clusters
  - **k (clusters)**: Balance between granularity and sample size
    - Too few: oversimplification
    - Too many: data sparsity per cluster

### LSTM Architecture Rationale
- **Two LSTM Layers**: 
  - Captures hierarchical temporal patterns
  - First layer: local temporal dependencies
  - Second layer: higher-level patterns
  
- **Dropout (0.2)**:
  - Prevents overfitting on small weather pattern clusters
  - Encourages robust feature learning
  
- **Linear Output Layer**:
  - Appropriate for regression (unbounded precipitation)
  - No activation constraint on predictions

### Data Stratification
- **Why Stratification by Cluster?**
  - Ensures train/val/test see all weather pattern types
  - Prevents bias toward common clusters
  - Better evaluation of generalization

---

## Configuration Parameters

### Easily Configurable in `src/methods/lstm_cluster/run_experiment.py`

```python
# Station
STATE = "RS"
STATION_ID = "A801"

# Clustering
WINDOW_SIZE = 15          # Days per window [5-30 recommended]
N_CLUSTERS = 5            # Number of clusters [3-10 recommended]
SIGMA = 1.0               # Gaussian bandwidth [0.5-5.0 range]
USE_ALL_FEATURES = True   # Use all numeric features

# LSTM
LSTM_UNITS = 64           # First LSTM layer
LSTM_UNITS_2 = 32         # Second LSTM layer
DROPOUT_RATE = 0.2        # Regularization
LEARNING_RATE = 0.001     # Adam optimizer

# Training
EPOCHS = 50               # Max epochs
BATCH_SIZE = 32           # Samples per batch
EARLY_STOPPING = True     # Stop if no improvement
PATIENCE = 10             # Epochs to wait

# Split
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
# Test ratio is computed as 1 - TRAIN_RATIO - VAL_RATIO
```

Output folder naming and shared plotting defaults are configured in
`src/methods/lstm_cluster/config_output.yaml`.

### Customization Ideas
1. **Change clustering approach**: Modify `spectral_clustering()` call
2. **Different precipitation threshold**: Adjust in `calculate_zero_precipitation_metrics()`
3. **Cluster-specific models**: Train separate LSTM per cluster
4. **Temporal sequence**: Change input reshape from (1, features) to (timesteps, features)
5. **Features subset**: Specify column list instead of using all

---

## Output Directory Structure

```
outputs/dd_mm_yy/lstm_cluster_sweep_RS_A801_<timestamp>/
|-- sweep_results.csv
|-- sweep_summary.txt
|-- overleaf_table.txt
|-- overleaf_cluster_metric_tables.txt
`-- RS_A801_w08_k03_spectral_sigma_0p01/
    |-- 01_training_history_cluster_<cluster>.png
    |-- 02_predictions_vs_actual.png
    |-- 02_predictions_timeseries_split_<n>_of_04.png
    |-- 03_residuals_analysis.png
    |-- 04_error_by_magnitude.png
    |-- 05_cluster_performance.png
    |-- 06_cluster_distribution.png
    |-- 07_precipitation_distribution_by_cluster.png
    |-- 08_input_precipitation_distribution_by_cluster.png
    |-- cluster_precipitation_histograms/
    |-- cluster_prediction_histograms/
    |-- input_precipitation_distribution_by_cluster/
    |-- evaluation_report.txt
    |-- experiment_report.tex
    |-- experiment_report.pdf
    |-- input_next_day_precipitation_by_cluster.csv
    |-- metrics_summary.csv
    |-- cluster_model_metrics.csv
    |-- summary.txt
    `-- test_predictions.csv
```

---

## How to Run

### Prerequisites
1. Ensure data exists: `data/inmet/RS/A801/`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install tensorflow keras scikit-learn pandas matplotlib seaborn
   ```

### Execute the Experiment
```bash
lstm-cluster
```

### Expected Runtime
- ~5-10 minutes on CPU (depends on data size and epochs)
- ~1-2 minutes on GPU (with TensorFlow-GPU)

### Output
- Console output with step-by-step progress
- All results saved to a timestamped sweep folder under the current
  `outputs/dd_mm_yy/` daily folder, with one configuration subfolder per
  window, cluster count, algorithm, and sigma.

---

## Interpreting Results

### Good Model Indicators
- ✓ Training loss decreases smoothly
- ✓ Validation loss tracks training loss (no overfitting)
- ✓ RMSE < 10% of target mean
- ✓ R² > 0.6 for test set
- ✓ Residuals centered near zero with no patterns
- ✓ Predictions show correlation with actuals

### Potential Issues
- ✗ Large gap between train and val loss → overfitting
- ✗ Validation loss increasing → poor generalization
- ✗ Residuals show trends → model missing patterns
- ✗ High error for specific clusters → cluster-specific model needed
- ✗ Poor performance on rainy days → class imbalance issue

### Improvements to Try
1. **Better performance on rainy days**:
   - Use weighted loss (higher weight for non-zero days)
   - Include precipitation event indicators
   - Use quantile regression instead of MSE

2. **Reduce overfitting**:
   - Increase dropout rate
   - Reduce model complexity (fewer units)
   - Add L1/L2 regularization

3. **Better clustering**:
   - Adjust sigma parameter
   - Try different number of clusters
   - Experiment with PCA dimensionality reduction

4. **Temporal patterns**:
   - Use proper sequence format: (samples, timesteps, features)
   - Add lag features (previous day's precipitation)
   - Consider seasonal patterns

---

## Code Structure

### Core Modules Created/Used

#### `src/models/lstm.py`
- `LSTMPrecipitationPredictor`: Complete LSTM wrapper
- `prepare_sequences()`: Data reshaping utility

#### `src/evaluation/metrics.py` (NEW)
- `calculate_regression_metrics()`: Standard metrics
- `calculate_zero_precipitation_metrics()`: Specialized for dry days
- `plot_predictions_vs_actual()`: Visualization
- `plot_residuals()`: Residual analysis
- `plot_error_by_magnitude()`: Binned error analysis
- `plot_cluster_performance()`: Per-cluster metrics
- `create_evaluation_report()`: Text report generation

#### `src/methods/lstm_cluster/pipeline.py`
- `run_experiment()`: Orchestrates the configured sweep
- `run_configuration()`: Runs one window/cluster/sigma configuration
- `split_by_cluster()`: Stratified data splitting
- `setup_styling()`: Visualization setup

#### `src/methods/lstm_cluster/report.py`
- `generate_config_report()`: Writes `experiment_report.tex` and optionally
  compiles `experiment_report.pdf`

#### Existing Modules (Reused)
- `config`: Paths and configuration
- `data.load_data`: Station data loading
- `methods.tools.sliding_windows`: Window creation
- `methods.cluster.ng`: Spectral clustering

---

## Scientific Rationale

### Why This Approach?

1. **Weather as Clustered Patterns**
   - Weather systems evolve in distinct patterns
   - Clustering captures these meteorological regimes
   - Each cluster may have predictive characteristics

2. **LSTM for Temporal Prediction**
   - Weather has temporal dependencies (recent conditions matter)
   - LSTM captures long-range dependencies
   - Better than simple regression for sequence data

3. **Combined Approach Benefits**
   - Clustering provides interpretable weather regimes
   - LSTM provides powerful temporal modeling
   - Allows cluster-specific analysis and optimization

4. **Data Stratification**
   - Ensures all weather patterns represented in train/test
   - Better evaluation of generalization capability
   - Fair comparison across different climate regimes

---

## Future Extensions

1. **Ensemble Methods**
   - Train separate models per cluster
   - Ensemble predictions weighted by cluster confidence

2. **Multi-Step Forecasting**
   - Predict 2-7 days ahead instead of next-day only
   - Use recursive or direct forecasting

3. **Feature Importance**
   - Analyze which climate variables matter most
   - Integrate with attention mechanisms

4. **Transfer Learning**
   - Pre-train on multiple stations
   - Fine-tune for specific stations

5. **Operational Integration**
   - Real-time predictions with streaming data
   - Uncertainty quantification
   - Model retraining pipeline

---

## References

- **LSTM Networks**: Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"
- **Spectral Clustering**: Ng, Jordan & Weiss (2002) - "On Spectral Clustering"
- **Time Series Forecasting**: Goodfellow, Bengio & Courville (2016) - "Deep Learning"
- **Weather Prediction**: Reichstein et al. (2019) - "Deep learning and process understanding for data-driven Earth system science"

---

## Contact & Support

For questions about this experiment, refer to:
- Main code: `src/methods/lstm_cluster/pipeline.py`
- LSTM implementation: `src/models/lstm.py`
- Evaluation functions: `src/evaluation/metrics.py`
- Report generation: `src/methods/lstm_cluster/report.py`
- Clustering: `src/methods/cluster/ng.py`
- Features: `src/methods/tools/sliding_windows.py`

---

**Last Updated**: 2024
**Experiment Status**: Ready for production use
**Tested On**: RS A801 station (Rio Grande do Sul, Brazil)

