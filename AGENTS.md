# CDC benchmark conventions

This repository implements the real-data phase of the DIA3 diabetes-status study.
Deliverables are in English. Report measured results and observational associations.

## Run and verify

Use Python 3.14 and install `requirements-dev.txt` in a virtual environment.
Run `python -m src.experiment` from the repository root for the full experiment.
Run `python scripts/execute_notebook.py` to save executed notebook outputs.
Set `RERUN_EXPERIMENTS=1` to retrain from the notebook. Otherwise it verifies source
hashes and displays the saved run. Missing results trigger a complete run.

Before delivery: `python -m ruff check src tests scripts`,
`python -m ruff format --check src tests scripts`, `python -m mypy src scripts`,
`python -m pytest -q`, notebook execution, and `git diff --check`.
Use `MPLBACKEND=Agg` for headless experiment execution.

## Invariants and pitfalls

- Use only the exact UCI binary CSV and pinned SHA-256. Fail on schema/content drift.
- Keep raw data out of Git; retain source and data-term acknowledgements.
- ID and target are never inputs. Age is a band, not exact age.
- Ordinal variables are numeric by design; explain the linearity assumption.
- Split before EDA; fit scalers and selectors only inside training folds.
- Keep the 20% test labels out of model and threshold selection.
- Write frozen configurations before test evaluation; never retune from test scores.
- AP means average precision, not trapezoidal PR-AUC. Brier is not calibration alone.
- Repeated responses are not necessarily duplicate participants. Retain them.
- Bootstrap intervals condition on the fixed model and observed prevalence.
- Recompute outputs after source changes; source fingerprints in notebooks must match.
- Do not add unmeasured performance claims or clinical/causal conclusions.

Contribute on a feature/fix branch using conventional commits and PRs to main.
