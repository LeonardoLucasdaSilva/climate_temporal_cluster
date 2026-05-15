# Climate_cluster

Project for clustering INMET climate stations in Brazil.
This project aims to provide a time clustering approach to detect extreme weather events.

## Configuration

### Quick Start (After Cloning from GitHub)

1. **Clone the repository:**
   ```powershell
   git clone <your-repository-url>
   cd climate_cluster
   ```

2. **Create and activate virtual environment:**
   ```powershell
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure Python Interpreter in Your IDE:**

   **PyCharm:**
   - File → Settings → Project → Python Interpreter
   - Click ⚙️ → Add → Existing Environment
   - Navigate to `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (macOS/Linux)
   - Click OK

   **VS Code:**
   - Open Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
   - Search "Python: Select Interpreter"
   - Choose `.venv` from the list
   - If not listed, select "Enter interpreter path" and navigate to `.venv/Scripts/python.exe`

   **Jupyter Notebook / JupyterLab:**
   - Activate the environment, then install:
     ```powershell
     pip install ipykernel
     python -m ipykernel install --user --name climate_cluster --display-name "Climate Cluster"
     ```
   - In notebook, select Kernel → Change kernel → Climate Cluster

   **PyDev (Eclipse):**
   - Window → Preferences → PyDev → Interpreters → Python Interpreters
   - Click "New"
   - Set Interpreter Name: `climate_cluster`
   - Set Executable: `.venv/Scripts/python.exe`

5. **Verify installation:**
   ```powershell
   python examples/example_load_station.py
   ```
   If you see output without import errors, you're ready to go.

### Project Structure
```
climate_cluster/
├── src/                    # Main module source code
│   └── climate_cluster/
│       ├── config.py       # Configuration
│       ├── config_data.py  # Data loading
│       ├── clustering/     # Clustering algorithms
│       ├── features/       # Feature engineering
│       ├── data/           # Data utilities
│       └── pipeline/       # End-to-end pipeline
├── data/                   # INMET station data (by state)
├── examples/               # Example scripts
├── tests/                  # Unit tests
├── .venv/                  # Virtual environment (auto-created)
├── requirements.txt        # Python dependencies
└── README.md
```

---

##  Key Functions

| Function | Purpose | File |
|----------|---------|------|
| `load_single_station()` | Load INMET data | config_data.py |
| `create_normalized_windows()` | Create windowed samples | features/window_features.py |
| `create_windows()` | Low-level window creation | features/window_features.py |
| `windows_to_dataframe()` | Denormalize windows | features/window_features.py |
| `fit_predict()` | Spectral clustering | clustering/custom_algorithm.py |
| `spectral_clustering()` | Full algorithm | clustering/custom_algorithm.py |
| `run_clustering_pipeline()` | End-to-end pipeline | pipeline/run.py |

## Loading single-station data

The project loads a single INMET station's data from a local file and groups by **days**.

### Basic usage

```python
from pathlib import Path
from climate_cluster.config import DATA_ROOT
from climate_cluster.config_data import load_single_station

# Load station SP/A701 with default columns
df = load_single_station(
    state="SP",
    station_id="A701",
    data_root=DATA_ROOT,
)

print(df.head())
```

### Custom columns

```python
cols = [
    "DATA",
    "TEMPERATURA_MAXIMA",
    "TEMPERATURA_MIN",
    "PRECIPITACAO_TOTAL",
]

df = load_single_station(
    state="SP",
    station_id="A701",
    data_root=DATA_ROOT,
    cols=cols,
)
```

### Available columns

All INMET columns in your data:
- `TEMPERATURA`, `TEMPERATURA_MAXIMA`, `TEMPERATURA_MIN`
- `UMIDADE_MAX`, `UMIDADE_MIN`, `UMIDADE`
- `PRESSAO`, `PRESSAO_MAX`, `PRESSAO_MIN`
- `VELOCIDADE_VENTO`, `DIRECAO_VENTO`, `RAJADA_VENTO`
- `PRECIPITACAO_TOTAL`
- `RADIACAO`

### Run the example

```powershell
python examples/example_load_station.py
```

## Creating Window Features (Sliding Windows)

Convert daily data into sliding windows of consecutive days, normalized by column:

```python
from climate_cluster.features.window_features import create_normalized_windows

# Create windows of 4 consecutive days
windows, scaler = create_normalized_windows(
    df,
    window_size=4,
    columns=['TEMPERATURA_MAXIMA', 'TEMPERATURA_MIN', 'PRECIPITACAO_TOTAL']
)

print(windows.shape)
# Output: (n_samples, 4, 3)
# - n_samples: number of 4-day windows
# - 4: days per window
# - 3: selected columns

# Flatten for clustering algorithms
windows_flat = windows.reshape(windows.shape[0], -1)
# Shape: (n_samples, 12)  → 4 days × 3 features
```

**Features:**
- ✓ Create windows of any size (3, 4, 5, 7... days)
- ✓ All columns included in each window
- ✓ Automatic normalization (zero mean, unit variance per column)
- ✓ Denormalization support for interpretation

### Run the windowing example

```powershell
python examples/example_window_features.py
```

See [WINDOW_FEATURES.md](WINDOW_FEATURES.md) for detailed documentation.

## Spectral Clustering

This project implements the NG, Jordan-Weiss spectral clustering algorithm.

```python
from climate_cluster.clustering.ng import fit_predict

# Cluster the windowed data
labels = fit_predict(
    windows_flat,  # (n_samples, n_features)
    sigma=1.0,  # Affinity bandwidth
    k=3,  # Number of clusters
    random_state=42  # Reproducibility
)

print(f"Cluster assignments: {len(labels)}")
```

**Algorithm:**
- Gaussian affinity matrix with bandwidth σ
- Normalized Laplacian eigenvector method
- K-means clustering on eigenvectors

**Parameters:**
- `sigma`: Controls neighborhood size (0.5-5.0 typical)
- `k`: Number of clusters to find
- `random_state`: Random seed for reproducibility

### Complete Pipeline Example

```python
from climate_cluster.config import DATA_ROOT
from climate_cluster.config_data import load_single_station
from climate_cluster.features.window_features import create_normalized_windows
from climate_cluster.clustering.ng import fit_predict

# 1. Load data
df = load_single_station(state='SP', station_id='A701', data_root=DATA_ROOT)

# 2. Create windows
windows, scaler = create_normalized_windows(df, window_size=4)

# 3. Cluster
windows_flat = windows.reshape(windows.shape[0], -1)
labels = fit_predict(windows_flat, sigma=1.0, k=3)

# 4. Analyze
for i in range(3):
    print(f"Cluster {i}: {sum(l == i for l in labels)} samples")
```

Or in a simple one-liner:

```python
from climate_cluster.pipeline.run import run_clustering_pipeline

# One-liner for the entire pipeline
results = run_clustering_pipeline(
    state='SP',
    station_id='A701',
    window_size=4,
    n_clusters=3,
    sigma=1.0
)

labels = results['labels']
windows = results['windows']
scaler = results['scaler']
```

### Run via CLI

```powershell

# Default (SP/A701, 3 clusters, sigma=1.0, window=4)
python -m climate_cluster.pipeline.run

# Custom parameters
python -m climate_cluster.pipeline.run \
    --state BA \
    --station-id A401 \
    --clusters 5 \
    --sigma 2.0 \
    --window-size 7

# Specific columns
python -m climate_cluster.pipeline.run \
    --columns TEMPERATURA_MAXIMA PRECIPITACAO_TOTAL UMIDADE_MAX
```

### To Run Tests
1. Command: `python -m unittest discover -s tests -p "test_*.py"`
2. Or: `python verify_pipeline.py`
3. Or: `python verify_spectral.py`

## Useful snippets

### Parameter sweep
```python
results = {}
for sigma in [0.5, 1.0, 2.0, 3.0]:
    for k in [2, 3, 4, 5]:
        labels = fit_predict(windows_flat, sigma=sigma, k=k)
        counts = [sum(l == i for l in labels) for i in range(k)]
        results[(sigma, k)] = counts
        print(f"σ={sigma}, k={k}: {counts}")
```

### Denormalization (Back to Original Scale)
```python
from climate_cluster.features.window_features import windows_to_dataframe

df_original = windows_to_dataframe(
    windows=windows,
    columns=['TEMPERATURA_MAXIMA', 'TEMPERATURA_MIN', 'PRECIPITACAO_TOTAL'],
    scaler=scaler
)

print(df_original.head())
# Columns: TEMPERATURA_MAXIMA_day0, TEMPERATURA_MAXIMA_day1, ...
```

### Processing Multiple Stations
```python
stations = [('SP', 'A701'), ('BA', 'A401'), ('TO', 'A055')]

for state, station_id in stations:
    print(f"\n{'='*60}\nProcessing {state}/{station_id}\n{'='*60}")
    results = run_clustering_pipeline(
        state=state,
        station_id=station_id,
        window_size=4,
        n_clusters=3,
        sigma=1.0
    )
    print(f"Labels shape: {len(results['labels'])}")
```

### Reproducibility
```python
# Always set random_state for reproducible results
labels = fit_predict(
    windows_flat,
    sigma=1.0,
    k=3,
    random_state=42  # ← Important!
)
```

### Memory Optimization
```python
# For large datasets, process chunks
chunk_size = 2000
for i in range(0, len(windows_flat), chunk_size):
    chunk = windows_flat[i:i+chunk_size]
    labels_chunk = fit_predict(chunk, sigma=1.0, k=3)
    # Process chunk...
```
