# Two-stage PyTorch random search

This folder contains the validation-only hyperparameter-search framework in
`run_random_search.py`. Edit the candidate lists and search settings at the top
of that file, then run it from the repository root:

```powershell
.\.venv\Scripts\python.exe -u experiments\random_search\run_random_search.py
```

The framework never predicts or evaluates the test split. Training can use the
validation split for early stopping, spectral-sigma selection, screening, and
multi-seed ranking. The untouched test split is reserved for a later, separate
evaluation of a finally chosen configuration.

## Search design

1. `N_RANDOM_CONFIGURATIONS` unique hyperparameter configurations are sampled
   without replacement from the Cartesian product in `SEARCH_SPACE`, using
   `SCREENING_SEED` for reproducibility. Loss is sampled as a valid paired
   setting: quantile-weighted MSE with each editable quantile set, MAE, or
   ordinary MSE. MAE and MSE use one fixed placeholder quantile tuple because
   quantiles do not affect those losses; this avoids redundant effective
   configurations.
2. Each sampled configuration is expanded over every `K = 1, 2, ..., 10` and
   both K-means and spectral clustering. Following the repository-wide baseline
   rule, `K=1` is trained once as the unclustered baseline rather than being
   duplicated under both algorithm labels. K-means and spectral clustering are
   both evaluated for every `K=2,...,10`.
3. Spectral clustering uses `N_AUTOMATIC_SIGMAS` values selected heuristically
   from training-only, preprocessing-matched pairwise distances. The sigma grid
   is calculated once per sampled window size and cached in
   `automatic_sigma_values.json`.
4. For each random configuration and each clustered K, only the spectral sigma
   with the lowest validation MAE remains as a configuration directory.
   `stage1_all_sigma_metrics.csv` retains the small metric audit trail for all
   attempted sigmas. Before pruning, the framework also evaluates every K,
   algorithm, and sigma for each horizon and rain regime. Each random
   hyperparameter configuration receives a `best_by_horizon_rain_regime.csv`
   file identifying its best K/algorithm/sigma for every combination. The
   consolidated file
   `stage1_configuration_best_by_horizon_rain_regime.csv` makes these winners
   easy to compare across sampled hyperparameter configurations. A threshold-
   only subset is also written as
   `stage1_configuration_best_by_horizon_threshold.csv`.
5. The K=1 baseline, every K-means result, and each winning spectral result are
   ranked together by validation MAE. The number retained is
   `ceil(number_of_valid_results * TOP_FRACTION)`, with at least one result.
   With the default `TOP_FRACTION = 0.10`, configuration directories for the
   bottom 90% are deleted; their ranking rows remain for reproducibility.
6. Each retained exact configuration, including its chosen K, algorithm, and
   spectral sigma, is rerun once for every value in `CONFIRMATION_SEEDS`. These
   seeds must be different from the screening seed.
7. A candidate enters the final rankings only after every confirmation seed
   completes. The original sweep ranking remains and uses mean validation MAE.
   The framework also produces 30 independent horizon/rain-regime rankings:
   D+1 through D+5 crossed with overall MAE, `>0` mm, `0--10` mm, `>10` mm,
   `10--30` mm, and `>30` mm. Each detailed ranking is ordered by mean
   validation MAE for that exact horizon/regime and reports the sample standard
   deviation across confirmation seeds. The original 15 cumulative-threshold
   rankings (`>0`, `>10`, and `>30` mm) are still exported as a convenient
   subset.
8. Every confirmation seed saves four-panel train/validation curves for each
   cluster model under its `model_fit/` directory. No test curve or test metric
   is produced.

With five heuristic sigmas, one random hyperparameter configuration trains 55
screening configurations: one K=1 baseline, nine K-means configurations, and
45 spectral K/sigma configurations. After sigma reduction it contributes at
most 19 rows to global screening. The current 100 random configurations
therefore plan 5,500 stage-1 training runs, retain about 190 rows at 10%, and
plan 950 confirmation runs across five new seeds. The exact planned counts are
saved in `framework_design.json` before training begins.

The current editable parameter lists define 46,656 unique effective
hyperparameter configurations before clustering expansion. One hundred are
sampled. The selected `lstm_loss_function` and `loss_quantiles` are included
in the sampled-configuration registry, stage-one winner tables, confirmation
metrics, and final ranking CSVs.

The top-10% screening boundary continues to use the sweep's validation MAE so
that there is one coherent finalist set. The detailed rankings independently
rerank that same fully confirmed finalist set. Regime membership is based on
actual validation precipitation at the specific forecast horizon. `Overall`
uses every finite observation; `0--10` includes both endpoints; `10--30` means
strictly greater than 10 and less than or equal to 30. The cumulative
thresholds remain strict `>` comparisons. The test split is never used.

## Output layout

Each invocation creates or resumes:

```text
outputs/pytorch/random_search/<search_id>/
|-- framework_design.json
|-- automatic_sigma_values.json
|-- sampled_hyperparameter_configurations.csv
|-- stage1_all_sigma_metrics.csv
|-- stage1_all_horizon_rain_regime_metrics.csv
|-- stage1_configuration_best_by_horizon_rain_regime.csv
|-- stage1_configuration_best_by_horizon_threshold.csv
|-- stage1_screening_rankings.csv
|-- stage1_retained_top_10_percent.csv
|-- stage1_screening/
|   `-- random_XXXX/
|       |-- all_sigma_results.csv
|       |-- retained_clustering_results.csv
|       |-- all_horizon_rain_regime_metrics.csv
|       |-- best_by_horizon_rain_regime.csv
|       |-- best_by_horizon_threshold.csv
|       `-- <retained configuration folders>
|-- confirmation_seed_metrics.csv
|-- confirmation_horizon_threshold_metrics.csv
|-- confirmation_horizon_rain_regime_metrics.csv
|-- stage2_confirmation/
|   `-- <candidate_id>/
|       |-- selected_screening_configuration.csv
|       `-- seed_<seed>/
|           `-- <configuration>/
|               |-- validation_mae_by_horizon_threshold.csv
|               |-- validation_mae_by_horizon_rain_regime.csv
|               `-- model_fit/*.png
|-- final_validation_rankings.csv
|-- final_validation_rankings.tex
|-- final_validation_rankings.pdf
|-- final_validation_rankings_by_horizon_threshold.csv
|-- final_validation_rankings_by_horizon_threshold.tex
|-- final_validation_rankings_by_horizon_threshold.pdf
|-- final_rankings_by_horizon_threshold/
|   |-- validation_mae_d01_gt0mm.csv
|   |-- ...
|   `-- validation_mae_d05_gt30mm.csv
|-- final_validation_rankings_by_horizon_rain_regime.csv
|-- final_validation_rankings_by_horizon_rain_regime.tex
|-- final_validation_rankings_by_horizon_rain_regime.pdf
|-- final_rankings_by_horizon_rain_regime/
|   |-- validation_mae_d01_overall.csv
|   |-- validation_mae_d01_0_to_10mm.csv
|   |-- ...
|   `-- validation_mae_d05_gt_30mm.csv
|-- failures.csv
`-- framework_complete.json
```

Set `SEARCH_NAME` to a stable value to resume an interrupted search. Completed
stage-1 configurations and completed candidate/seed confirmation runs are
reused. If the saved design does not match the current parameter space, seed
lists, clustering grid, or retention fraction, the runner stops and asks for a
new search name rather than mixing incompatible results.

All final PDFs are explicitly labeled as multi-seed validation rankings. The
threshold PDF places `>0`, `>10`, and `>30` mm side by side for each horizon.
The complete rain-regime PDF uses two three-table pages per horizon to include
overall and interval MAEs without making the tables unreadable. The best rank,
mean MAE, and standard deviation in every ranking are bold green. Failed or
incomplete candidates remain in `failures.csv` but cannot enter any final
ranking. Set `FAIL_FAST = True` if the first failure should stop the run.
