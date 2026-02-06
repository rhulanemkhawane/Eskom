# Codebase Audit: Eskom Energy Demand Forecasting

Date: 2026-02-05
Scope: Code inspection only (no test-set evaluation). Safe runs will be appended after execution.

**Executive Summary**
- Professional Standards (Junior DS Portfolio): Pass
- Forecasting Correctness & Leakage Prevention: Pass (with one operational warning)
- Test Evaluation Guarding: Pass (RUN_TEST_EVAL is `False` in config)

## A) Professional Standards (Junior DS Portfolio)
**Scorecard**
- Reproducibility: Warn
- Code Quality: Warn
- Project Hygiene: Pass
- Basic Testing: Pass

**Key Evidence (inspection)**
- Config centralized with paths, split boundaries, seed, and runtime flags in `eskom_energy_demand_forecasting/config.py:13-98`.
- CLI entry points exist (Typer) for dataset and features in `eskom_energy_demand_forecasting/dataset.py:169-184` and `eskom_energy_demand_forecasting/features.py:85-99`.
- Makefile includes `train` and `predict` targets in `Makefile:1-90`.
- README contains concrete run steps and configuration guidance in `README.md:70-130`.
- Logging is used (loguru) rather than print in core modules (`dataset.py`, `train.py`, `predict.py`).
- Requirements are in `requirements.txt`, while `pyproject.toml` defines package metadata and ruff config; no explicit version pinning or lockfile is present (`requirements.txt`, `pyproject.toml:1-33`).
- Minimal smoke tests added in `tests/test_smoke.py:1-22`.

**Assessment Notes**
- Reproducibility is partially covered by `requirements.txt` and a seed in config, but there is no environment lockfile and no explicit data versioning.
- Project hygiene is good: data loading, features, modeling, and plotting are separated in `eskom_energy_demand_forecasting/`.
- Minimal testing is now present via a simple smoke test.

## B) Forecasting Correctness & Leakage Prevention
**Scorecard**
- Data Handling: Pass
- Splits & Disjointness: Pass
- Rolling-Origin Evaluation: Pass
- Feature Leakage Checks: Pass
- Metrics Correctness: Pass
- Model Evaluation Correctness: Pass (test eval gated)

**Key Evidence (inspection)**
- Timestamp parsing, sorting, timezone handling, duplicate checks, and hourly frequency enforcement in `eskom_energy_demand_forecasting/dataset.py:59-92`.
- Target engineering: `Total Energy Demand = Residual Demand + Total RE` in `eskom_energy_demand_forecasting/dataset.py:95-101`.
- Split boundaries are explicit and disjoint with boundary guards in `eskom_energy_demand_forecasting/modeling/train.py:39-67`.
- Validation/test split identity guards via index and target hashes in `eskom_energy_demand_forecasting/modeling/train.py:70-91`.
- Rolling-origin fold construction uses sequential windows (no shuffling), and training ends before validation starts in `eskom_energy_demand_forecasting/modeling/train.py:94-130`.
- Lag features and rolling features are built with positive lags and `shift(1)` before rolling in `eskom_energy_demand_forecasting/features.py:30-43`.
- Recursive, leakage-safe backtesting and predictions are implemented in `eskom_energy_demand_forecasting/modeling/train.py:150-329` and `eskom_energy_demand_forecasting/modeling/predict.py:24-130`.
- Test evaluation is gated by `CONFIG.run_test_eval` in `eskom_energy_demand_forecasting/config.py:87-89` and `eskom_energy_demand_forecasting/modeling/train.py:371-393`.

**Leakage/Correctness Risks**

## Immediate Fixes (Checklist)
**P0 (Correctness / Leakage)**
- [x] Modify backtesting to prevent multi-step leakage for horizons >1 via recursive forecasting per fold (`train.py`).
- [x] Implement recursive multi-step forecasting in `predict.py`.

**P1 (Reproducibility / Usability)**
- [x] Add `make train` and `make predict` targets.
- [x] Expand README with concrete run steps and configuration details.
- [x] Add timestamp format and switch to `freq='h'` in config to avoid pandas deprecation.
- [ ] Set and log deterministic seeds for numpy and any model libraries at runtime (e.g., in `train.main`).

**P2 (Polish / Quality)**
- [x] Add a minimal `tests/` folder with smoke tests: import package, load data, build features.
- [ ] Add a simple CLI/entrypoint to run a “dry-run” backtest on a short span for sanity checks.

## Required Config Values (if not set)
If any of the following are missing or must be changed, set them explicitly before training:
- `CONFIG.train_end`, `CONFIG.val_end`, `CONFIG.test_end`
- `CONFIG.timezone`, `CONFIG.freq`

Template (example):
```python
# eskom_energy_demand_forecasting/config.py
train_end = "YYYY-MM-DD HH:00:00+02:00"
val_end = "YYYY-MM-DD HH:00:00+02:00"
test_end = "YYYY-MM-DD HH:00:00+02:00"
```

---

## Safe Run Results (Observed)
**Command:** `python -c "import eskom_energy_demand_forecasting as p; print(p.__file__)"`
- Result: Success
- Output: `/Users/rhulanemkhawane/Documents/eskom/eskom_energy_demand_forecasting/__init__.py`

**Command:** `python -m eskom_energy_demand_forecasting.modeling.train`
- Result: Failed before any evaluation due to external data fetch being unavailable
- Error: `URLError: <urlopen error [Errno 8] nodename nor servname provided, or not known>`
- Impact: Training, backtesting, and artifact generation did not run; no metrics or model files produced.

**Command:** data availability smoke check (safe):
- Result: Success
- Rows: 43,824
- Columns: 42
- Required columns present: `Residual Demand`, `Total RE`
- Index timezone: `Africa/Johannesburg`
- Warnings: 
  - Pandas could not infer datetime format (suggest specifying format for stability)
  - `freq='H'` deprecation warning (should switch to `'h'`)

## Additional P0/P1 Fixes from Safe Runs
**P1**
- [x] Specify timestamp parsing format for Eskom CSV.
- [x] Update `CONFIG.freq` from `'H'` to `'h'`.

---

## Summary of What Could Not Be Verified
- Backtest metrics, baseline comparison, and artifact outputs could not be verified in earlier runs due to external data fetches. No safe-run metrics were produced at that time.
- Therefore, model performance and baseline dominance cannot be assessed at this time.

## Safe Run Results (Observed - After Fixes)
**Command:** `ENABLE_TREE_MODEL=false N_FOLDS=1 HORIZON_HOURS=6 STEP_HOURS=6 INITIAL_TRAIN_DAYS=60 python -m eskom_energy_demand_forecasting.modeling.train`
- Result: Success
- Notes:
  - Tree model disabled due to repeated `Signal(6)` aborts during tree model fit in this environment.
  - Backtest settings reduced for a fast-run (1 fold, 6-hour horizon) and are **not** comparable to the default configuration.
- Artifacts created:
  - `data/processed/eskom_processed.parquet`
  - `reports/split_sanity_checks.txt`
  - `reports/fold_metrics.csv`
  - `models/metrics_summary.json`
  - `models/final_model.pkl`
- Test evaluation: still disabled (`RUN_TEST_EVAL=False`).

## Updated Gaps After Fixes
- Full backtest with default folds/horizons is still not verified in this environment.
- Tree model stability in this sandbox is unverified (process aborts during fit). Consider switching to a pure-sklearn model or running on a different environment for validation.
