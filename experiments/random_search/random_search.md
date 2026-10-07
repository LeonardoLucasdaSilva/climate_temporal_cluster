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
   `SCREENING_SEED` for reproducibility.
2. Each sampled configuration is expanded over `K = 1, 3, 5, 7, 9` and both
   K-means and spectral clustering. Following the repository-wide baseline
   rule, `K=1` is trained once as the unclustered baseline rather than being
   duplicated under both algorithm labels.
3. Spectral clustering uses `N_AUTOMATIC_SIGMAS` values selected heuristically
   from training-only, preprocessing-matched pairwise distances. The sigma grid
   is calculated once per sampled window size and cached in
   `automatic_sigma_values.json`.
4. For each random configuration and each clustered K, only the spectral sigma
   with the lowest validation MAE remains as a configuration directory.
   `stage1_all_sigma_metrics.csv` retains the small metric audit trail for all
   attempted sigmas.
5. The K=1 baseline, every K-means result, and each winning spectral result are
   ranked together by validation MAE. The number retained is
   `ceil(number_of_valid_results * TOP_FRACTION)`, with at least one result.
   With the default `TOP_FRACTION = 0.10`, configuration directories for the
   bottom 90% are deleted; their ranking rows remain for reproducibility.
6. Each retained exact configuration, including its chosen K, algorithm, and
   spectral sigma, is rerun once for every value in `CONFIRMATION_SEEDS`. These
   seeds must be different from the screening seed.
7. A candidate enters the final ranking only after every confirmation seed
   completes. The final order uses mean validation MAE. The report also shows
   the sample standard deviation of validation MAE across seeds as the
   stability measure.
8. Every confirmation seed saves four-panel train/validation curves for each
   cluster model under its `model_fit/` directory. No test curve or test metric
   is produced.

With five heuristic sigmas, one random hyperparameter configuration trains 25
screening configurations: one K=1 baseline, four K-means configurations, and
twenty spectral K/sigma configurations. After sigma reduction it contributes
at most nine rows to global screening. The default 20 random configurations
therefore plan 500 stage-1 training runs, retain about 18 rows at 10%, and plan
90 confirmation runs across five new seeds. The exact planned counts are saved
in `framework_design.json` before training begins.

## Output layout

Each invocation creates or resumes:

```text
outputs/pytorch/random_search/<search_id>/
|-- framework_design.json
|-- automatic_sigma_values.json
|-- sampled_hyperparameter_configurations.csv
|-- stage1_all_sigma_metrics.csv
|-- stage1_screening_rankings.csv
|-- stage1_retained_top_10_percent.csv
|-- stage1_screening/
|   `-- random_XXXX/
|       |-- all_sigma_results.csv
|       |-- retained_clustering_results.csv
|       `-- <retained configuration folders>
|-- confirmation_seed_metrics.csv
|-- stage2_confirmation/
|   `-- <candidate_id>/
|       |-- selected_screening_configuration.csv
|       `-- seed_<seed>/
|           `-- <configuration>/model_fit/*.png
|-- final_validation_rankings.csv
|-- final_validation_rankings.tex
|-- final_validation_rankings.pdf
|-- failures.csv
`-- framework_complete.json
```

Set `SEARCH_NAME` to a stable value to resume an interrupted search. Completed
stage-1 configurations and completed candidate/seed confirmation runs are
reused. If the saved design does not match the current parameter space, seed
lists, clustering grid, or retention fraction, the runner stops and asks for a
new search name rather than mixing incompatible results.

The final PDF is explicitly labeled as a multi-seed validation ranking. Its
best rank, mean MAE, and standard deviation are bold green. Failed or
incomplete candidates remain in `failures.csv` but cannot enter the final
ranking. Set `FAIL_FAST = True` if the first failure should stop the run.
