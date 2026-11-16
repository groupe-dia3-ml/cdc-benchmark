# CDC Diabetes Health Indicators benchmark

Phase 1 of the ESILV A4 DIA3 machine learning project. The completed benchmark
compares lifestyle/demographic inputs with an enriched health-indicator model.
The target is current survey status, not future diabetes onset.

## Results

The full 253,680-row export was evaluated with a fixed stratified 80/20 split,
five development folds and 28 predeclared configurations. All preprocessing and
feature selection stay inside training folds. The primary selection metric is
average precision (AP), not trapezoidal PR-AUC.

| Frozen comparator | Inputs | Development AP | Test AP | Test ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Majority class | 21 | 0.1393 | 0.1393 | 0.5000 |
| Lifestyle logistic, C=1 | 9 | 0.2542 | 0.2571 | 0.7113 |
| Enriched logistic, C=1 | 21 | 0.4076 | 0.3942 | 0.8192 |
| Lifestyle gradient boosting | 9 | 0.2747 | 0.2763 | 0.7244 |
| Enriched gradient boosting | 21 | **0.4359** | **0.4247** | **0.8274** |
| RFE logistic | 10 | 0.4048 | 0.3928 | 0.8186 |

The development-selected primary model is histogram gradient boosting with
15 leaves, 150 iterations, learning rate 0.08 and L2 regularization 1. Its AP
95% stratified test-bootstrap interval is [0.4141, 0.4350] (200 replicates,
conditional on the fitted model and observed prevalence).

At its development-OOF F1 threshold of 0.2316, test recall is 61.11%, precision
38.01%, F1 0.4687 and false-positive rate 16.14%. At threshold 0.5, recall is
16.16% and precision 55.73%. The threshold is an experimental choice, not a
validated clinical operating point. The majority classifier achieves 86.07%
accuracy with zero recall, illustrating why accuracy is insufficient here.

Read the [executed notebook](notebooks/01_cdc_diabetes_benchmark.ipynb) for EDA,
full model comparisons, calibration, coefficients, feature selection and errors
by age and recorded sex. Machine-readable evidence is in [results](results).

## Reproduce

Tested with Python 3.14.10. Use Python 3.14 and the pinned dependencies:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m src.data_loader
MPLBACKEND=Agg .venv/bin/python -m src.experiment
.venv/bin/python scripts/execute_notebook.py
```

The complete experiment took about 70 seconds on the development machine;
runtime depends on hardware. It uses at most two numerical-library threads and
two random-forest workers. It downloads the exact UCI CSV if absent.

The notebook normally verifies source fingerprints and renders the saved run;
missing results trigger fitting. To retrain and execute the notebook together:

```sh
RERUN_EXPERIMENTS=1 MPLBACKEND=Agg .venv/bin/python scripts/execute_notebook.py
```

`results/frozen_protocol.json` records the chosen configurations and thresholds
before test evaluation. `run_manifest.json` records source hashes, versions and
seed. CSV outputs provide all fold scores, timings, final metrics, subgroup
counts, selected features and coefficients. `environment.txt` captures the full
package environment used for the submitted run. Re-running overwrites generated
results; do not modify the selection protocol after inspecting test outcomes.

## Verification

```sh
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m ruff format --check src tests scripts
.venv/bin/python -m mypy src scripts
.venv/bin/python -m pytest -q
git diff --check
```

Tests cover schema rejection, checksum enforcement, reproducible disjoint splits,
metadata exclusion, repeated-profile retention, training-only scaling, threshold
selection, AP semantics and degenerate evaluation groups.

## Data provenance and interpretation

- [UCI dataset 891 and variable definitions](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- [Exact binary export](https://archive.ics.uci.edu/static/public/891/data.csv)
- SHA-256: `9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964`
- 21 predictors, `ID` excluded, target `Diabetes_binary`: 218,334 zeros and 35,346 ones.
- The selected UCI dictionary describes 1 as prediabetes or diabetes. The landing
  page also discusses a three-class variant; alternative files are not substituted.
- No missing values. Age is a category band, not exact age. Ordinal predictors
  remain numeric, with that modeling assumption explicitly discussed.
- Repeated profiles are retained. There are 18,580 repeated development predictor
  profiles and 6,836 test rows whose predictor hash also appears in development;
  identical answers alone do not identify duplicate participants.

The study is observational and does not establish causal effects, future disease
risk, clinical utility or transfer to another population. Some health indicators
may reflect an existing diagnosis. No survey weights are available in this export.
Subgroup comparisons are descriptive. Brier score measures overall probability
error, not calibration alone. CV scores used for selection are not nested,
unbiased performance estimates.

CDC Diabetes Health Indicators [Dataset]. (2017). UCI Machine Learning Repository.
[DOI: 10.24432/C53919](https://doi.org/10.24432/C53919). The citation year is not a
claim about the survey year. Preserve UCI acknowledgements and consult the
[original dataset collection](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
for the source terms. Raw downloads are not committed.
