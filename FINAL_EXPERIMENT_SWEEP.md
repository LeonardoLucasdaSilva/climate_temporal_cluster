# Final PyTorch Experiment Sweep

This document specifies the compact, resumable final experiment implemented by
`experiments/run_final_experiment_sweeps.py`. The active run is repetition 1,
uses batch size 8, MAE-only early stopping, unstandardized precipitation, three
LSTM architectures, and trains at most two independent cluster-specific LSTMs
at the same time.

## Objective and dataset

The sweep evaluates whether clustering historical climate windows before
training cluster-specific LSTMs improves five-day precipitation forecasts.

| Setting | Value |
|---|---|
| State | Rio Grande do Sul (`RS`) |
| INMET station | `A801` |
| Current source date range | 2000-09-22 through 2026-06-30 |
| Split strategy | Chronological, before sliding-window construction |
| Training ratio | 60% |
| Validation ratio | 10% |
| Test ratio | 30% |
| Forecast horizon | 5 days (`D+1` through `D+5`) |
| Window stride | 1 day |
| Random seed | 42 |
| Backend | PyTorch with required CUDA GPU |
| Deterministic execution | Enabled |

The current numeric feature set is:

1. `PRECIPITACAO_TOTAL`
2. `PRESSAO_MIN`
3. `PRESSAO_MAX`
4. `RADIACAO`
5. `TEMPERATURA_MAXIMA`
6. `TEMPERATURA_MIN`
7. `UMIDADE_MAX`
8. `UMIDADE_MIN`
9. `RAJADA_VENTO`
10. `VELOCIDADE_VENTO`
11. `DIRECAO_VENTO_SIN`
12. `DIRECAO_VENTO_COS`

## Swept dimensions

| Dimension | Values | Count |
|---|---|---:|
| Input window | 15, 30, 45 days | 3 |
| Cluster count | 1, 3, 5, 7, 9 | 5 |
| Clustering algorithm | K-means, spectral | 2 where applicable |
| Spectral sigma | 5 automatically selected values per preprocessing/window pair | 5 |
| LSTM architecture | 64→output; 128→output; 64→32→output | 3 |
| Loss thresholds | Q0.90; Q0.80; mixed Q0.75/Q0.90/Q0.95/Q0.99 | 3 |
| Early-stopping metric | MAE only | 1 |
| Precipitation standardization | Disabled | 1 |
| Repetition | 1, 2, 3, 4, 5 | 5 planned |

The loss-batch identifiers are:

- `q090`: one threshold at the cluster training-target 90th percentile.
- `q080`: one threshold at the cluster training-target 80th percentile.
- `mixed_q075_q090_q095_q099`: thresholds at the 75th, 90th, 95th,
  and 99th percentiles.

Every LSTM uses quantile-weighted MSE. Thresholds are calculated separately
from each cluster's training targets. Automatic bin weights are inversely
proportional to the observed bin frequencies and normalized so the most common
non-empty bin has weight 1.

### K=1 baseline

`K=1` is the unclustered LSTM baseline. It bypasses clustering completely and
is created once per otherwise-distinct LSTM configuration. It is not repeated
for both clustering algorithms and does not sweep spectral sigma. Its internal
configuration name retains the grid's `kmeans` label for compatibility, while
ranking tables display `-` as its clustering method.

### Spectral sigma selection

The five spectral sigma candidates are derived from the exact training
features for each window size and precipitation-standardization setting. They
are therefore data-dependent rather than fixed constants. For a shared
preprocessing/window/sigma combination, the runner reuses the spectral affinity
matrix and leading eigensystem across `K=3,5,7,9`; only the selected embedding,
embedding K-means, and downstream LSTMs vary with K.

## Configuration count

For each window size:

- `K=1`: `1 baseline × 3 LSTM architectures = 3` configurations.
- `K=3,5,7,9` with K-means: `4 K values × 1 method × 3 architectures = 12`.
- `K=3,5,7,9` with spectral clustering:
  `4 K values × 5 sigmas × 3 architectures = 60`.

Therefore:

```text
75 configurations per window
× 3 windows
= 225 configurations per loss/early-stop/standardization batch

225 configurations
× 3 loss-threshold configurations
× 1 early-stopping metric (MAE)
× 1 precipitation-standardization setting (disabled)
= 675 configurations per repetition

675 configurations
× 5 repetitions
= 3,375 planned configurations in the full study
```

The active repetition-1 run contains 675 configurations. The remaining
repetitions are separate output directories and are not started by a command
restricted to `--repetitions 1`.

## Fixed clustering and preprocessing settings

| Setting | Value |
|---|---|
| Clustering features | All 12 numeric features listed above |
| Covariate normalization | Standard scaling |
| Precipitation normalization | Disabled |
| PCA variance threshold | Disabled (`None`) |
| PCA use | Clustering only |
| Cluster only precipitation | No |
| Held-out K-means assignment | Nearest training centroid |
| Held-out spectral assignment | Spectral model prediction/Nyström extension |
| Minimum spectral training-cluster fraction | 5% |

Spectral configurations whose smallest training cluster contains less than 5%
of the training windows are marked `invalid_clustering` and skipped without
training LSTMs. K-means configurations are not filtered by this spectral
validity rule.

## Fixed LSTM training settings

| Setting | Value |
|---|---|
| Architecture variants | U1=64/U2=None; U1=128/U2=None; U1=64/U2=32 |
| Recurrent layers | One layer for the first two variants; two for the third |
| Dense head | 16 units, then 8 units, then 5 forecast outputs |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Learning rate | 0.001 (`1e-3`) |
| Weight decay | 0.0001 |
| Maximum epochs | 300 |
| Batch size | 8 |
| Early stopping | Enabled |
| Patience | 50 epochs |
| Warm-up before early stopping | 10 epochs |
| Loss | Quantile-weighted MSE |
| Quantile weights | Automatic, cluster-specific |
| Precipitation log transform | Disabled |
| Test every model on every cluster | Disabled |

`LOSS_ALPHA=0.01` is retained in metadata for compatibility, but alpha is not
used by the active quantile-weighted MSE loss.

## Parallel and deterministic execution

Each cluster has its own independent LSTM. For `K>1`, the runner trains at most
two of these LSTMs concurrently in separate worker processes. `K=1` remains
sequential. Configurations themselves remain sequential so checkpoint,
manifest, and ranking writes cannot collide.

The two-worker setting was validated against a completed `K=3`, window-15,
64-unit configuration:

| Measurement | Sequential | Two workers |
|---|---:|---:|
| Runtime | 367.17 seconds | 210.51 seconds |
| Speedup | 1.00× | 1.74× |
| Compared prediction/history arrays | 29 | 29 |
| Maximum absolute numerical difference | — | 0.0 |

All 29 arrays were bit-for-bit identical. During the live two-worker run, GPU
utilization reached approximately 85–94%, while observed memory remained well
below the RTX 4050 Laptop GPU's 6 GB capacity.

The runner also enforces deterministic PyTorch algorithms, disables cuDNN
benchmark selection, uses deterministic cuDNN behavior, and uses random seed
42. Parallelism changes scheduling and total runtime, not model inputs,
optimizer steps, batch ordering, or saved numerical results.

## Execution order and identifiers

The outer-to-inner iteration order is:

1. Repetition.
2. Loss-threshold configuration.
3. Early-stopping metric (fixed to MAE).
4. Precipitation-standardization flag (fixed to false).
5. Window size.
6. Cluster count.
7. Clustering algorithm.
8. Spectral sigma, when applicable.
9. LSTM architecture, ordered as 64, 128, then 64→32.

A batch identifier has this form:

```text
<loss>_patience_mae_precip_std_false
```

An example full configuration identifier is:

```text
q090_patience_mae_precip_std_false/
RS_A801_w30_k05_spectral_sigma_32p42130602307423_u1_64_u2_none
```

## Resume and validity behavior

The sweep is resumable at configuration boundaries:

- A configuration with a valid `metadata.json` and `result.npz` is skipped.
- A manifest entry marked `invalid_clustering` is also skipped.
- An interrupted, incomplete configuration is trained again from the start.
- Completed configurations are synchronized into rankings when a run starts
  and periodically during the sweep.
- Ranking repair is additive: locally absent results are not used to remove
  rankings copied from another computer.

The manifest and `progress.txt` count both trained configurations and spectral
configurations deliberately skipped as invalid.

The root-level `SWEEP_PROGRESS.md` is a human-readable dashboard. It is
refreshed when the runner starts and after every configuration checkpoint. It
separates successfully trained configurations from invalid/skipped clusterings,
and shows the processed total, percentage, remaining count, latest
configuration, and update time. Failure to refresh this convenience file (for
example, because another Windows program has locked it) is reported as a
warning and does not interrupt the sweep.

## Output layout

The active repetition is stored under:

```text
outputs/pytorch/final_experiment_sweep_1/
|-- experiment_design.json
|-- configuration_manifest.csv
|-- progress.txt
|-- compact_results/
|   `-- <batch>/<configuration>/
|       |-- metadata.json
|       `-- result.npz
|-- _shared/
|   |-- source_data.csv.gz
|   |-- targets/
|   `-- cluster_assignments/
|-- experiment_rankings.csv
|-- experiment_rankings.tex
|-- experiment_horizon_rankings.csv
|-- experiment_rankings_horizon_d01...d05.{csv,tex,pdf}
|-- cluster_comparison_rankings_horizon_d01...d05.{tex,pdf}
|-- cluster_comparison_rankings_horizon_d01...d05_gt_{0,10,30}mm.{tex,pdf}
|-- cluster_comparison_rankings_w30_qwmse_0p9_rain_thresholds.{tex,pdf}
`-- cluster_comparison_rankings_u1_64_u2_none_qwmse_0p9_rain_thresholds.{tex,pdf}
```

Compact results store predictions and per-cluster training histories. Shared
source data, observed targets, and cluster assignments are written once and
referenced from each configuration's metadata. Full plots, CSV diagnostics,
LaTeX, and PDF reports can be reconstructed later without retraining.
Ranking tables include both `U_1` and `U_2`; one-layer architectures display
`-` in the `U_2` column.
Comparison tables group identical loss, patience, `U_1`, `U_2`, precipitation
standardization, and window size. Clustering method and `K` are columns whose
values vary by row within each matched group.
Patience and R2 are not displayed as table columns, although patience remains
part of the internal matched-configuration key.
For each `K`, every table keeps only the spectral sigma with the lowest MAE
calculated for that table's own precipitation subset. K-means and the `K=1`
baseline remain for comparison.
Each horizon also has dedicated above-0 mm, above-10 mm, and above-30 mm
comparison PDFs whose metrics are calculated only from the named precipitation
range. The existing overall and combined horizon reports are preserved.

The previous batch-size-4 run is kept separately at
`outputs/pytorch/final_experiment_sweep_1_batch4_archive_20260907/`; it is not
mixed with the active batch-size-8 results.

The earlier learning-rate-`1e-3` run is kept at
`outputs/pytorch/final_experiment_sweep_1_lr1e3_archive_20260907/`. That run
used the old single-layer 64/128/256 architecture grid, so its results are not
mixed with the active 64/128/64-to-32 grid.

The interrupted learning-rate-`1e-2` run with the current architecture grid is
preserved at
`outputs/pytorch/final_experiment_sweep_1_lr1e2_current_arch_archive_20260907/`.
It contains 17 processed checkpoints and is not mixed with the active `1e-3`
results.

## Commands

Run or resume only repetition 1:

```powershell
.\.venv\Scripts\python.exe -u experiments\run_final_experiment_sweeps.py --repetitions 1
```

Run or resume all five repetitions:

```powershell
.\.venv\Scripts\python.exe -u experiments\run_final_experiment_sweeps.py
```

Validate the grid without training:

```powershell
.\.venv\Scripts\python.exe experiments\run_final_experiment_sweeps.py --dry-run
```

Repair and render repetition-1 rankings without training:

```powershell
.\.venv\Scripts\python.exe experiments\run_final_experiment_sweeps.py --render-only --repetitions 1
```

Reconstruct one selected compact configuration:

```powershell
.\.venv\Scripts\python.exe experiments\reconstruct_final_experiment_artifacts.py outputs\pytorch\final_experiment_sweep_1\compact_results\<batch>\<configuration>
```

Monitor the current checkpoint:

```powershell
Get-Content SWEEP_PROGRESS.md
```

The current two-worker runtime log is
`outputs/pytorch/final_experiment_lr1e3_current_arch_u2.stdout.log`; errors are
written to the matching `.stderr.log` file.

## Benchmark evidence

The reusable benchmark scripts and their generated evidence are:

- `experiments/benchmark_final_sweep_batch_sizes.py`
- `outputs/pytorch/batch_size_benchmark.csv`
- `outputs/pytorch/batch_size_benchmark_8.csv`
- `experiments/benchmark_final_sweep_parallel_training.py`
- `outputs/pytorch/parallel_training_benchmark.json`
- `experiments/benchmark_second_lstm_layer.py`
- `outputs/pytorch/second_lstm_layer_benchmark.json`

These benchmark outputs are diagnostic artifacts and are not part of the 675
configuration count.

The second-layer pilot used learning rate `1e-3`, held all other settings
fixed, and compared `LSTM_UNITS_2=None` with `LSTM_UNITS_2=32` for matched K=1,
K=3, and K=5 window-15 configurations. The active sweep uses the 64-to-32
architecture as its third architecture variant in place of the former
256-unit single-layer variant.
