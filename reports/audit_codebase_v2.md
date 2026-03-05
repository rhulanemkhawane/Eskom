# Codebase Audit v2 (Delta vs `reports/audit_codebase.md`)

Date: 2026-02-06
Scope: Code inspection + prior safe-run artifacts. No test-set evaluation.

**Executive Summary (Delta)**
- Professional Standards: **Mixed** (improved documentation + tests; added heavier deps and runtime knobs)
- Forecasting Correctness & Leakage Prevention: **Mixed** (recursive evaluation retained, but weather integration removed and LSTM predict path incomplete)
- Test Evaluation Guarding: **Unchanged (Pass)**
- **Overall Grade (Junior DS Standard): Warn — 6.5/10**

## What Improved Since v1
1. **Configurability & Repro Controls**
   - Backtest window/step/folds are now environment-configurable: `INITIAL_TRAIN_DAYS`, `HORIZON_HOURS`, `STEP_HOURS`, `N_FOLDS`.
   - Evidence: `eskom_energy_demand_forecasting/config.py:15-70`.

2. **LSTM Modeling Path Added**
   - LSTM training + recursive inference across horizon added (sequence + calendar features).
   - Evidence: `eskom_energy_demand_forecasting/modeling/train.py:161-402`, `eskom_energy_demand_forecasting/modeling/train.py:530-549`.
   - README now documents LSTM settings and env vars.
   - Evidence: `README.md:70-140`.

3. **Pinned Dependency Ranges**
   - `requirements.txt` now uses version ranges for major packages and includes `tensorflow`.
   - Evidence: `requirements.txt:1-18`.

## What Deteriorated Since v1
1. **Weather Integration Removed** (Breaks requirements for weather features)
   - Dataset pipeline no longer fetches or merges weather data. This removes a key requirement from the original brief.
   - Evidence: `eskom_energy_demand_forecasting/dataset.py:104-120` (no weather), `eskom_energy_demand_forecasting/features.py:46-68` (no weather columns).
   - Impact: Forecasting pipeline is now demand-only, so any conclusions about “weather integration correctness” cannot be made.

2. **LSTM Prediction Path Missing in `predict.py`**
   - Training can select `LSTM` and stores the model, but `predict.py` only supports sklearn-style `.predict()` and explicitly errors on ETS only. It does **not** handle LSTM artifacts (scalers, sequence length, model path).
   - Evidence: `eskom_energy_demand_forecasting/modeling/train.py:611-625` (LSTM saved), `eskom_energy_demand_forecasting/modeling/predict.py:116-151` (no LSTM branch).
   - Impact: If LSTM is selected as final model, `predict.py` will fail or generate invalid outputs. **This is a P0 correctness gap.**

3. **Operational Risk: heavier dependency footprint**
   - TensorFlow added to requirements; this increases install complexity and makes `make requirements` more brittle in restricted environments.
   - Evidence: `requirements.txt:14`.

## What Stayed the Same
- **Test evaluation guard** remains disabled by default (`run_test_eval=False`).
  - Evidence: `eskom_energy_demand_forecasting/config.py:84-85`, `eskom_energy_demand_forecasting/modeling/train.py:679-710`.
- **Leakage-safe backtesting** remains recursive across horizons for baselines, LSTM, and tree/linear models.
  - Evidence: `eskom_energy_demand_forecasting/modeling/train.py:355-402`, `483-579`.

---

## Updated Scorecards (v2)
### A) Professional Standards (Junior DS Portfolio)
- Reproducibility: **Warn** (no lockfile, heavy deps, offline installs still brittle)
- Code Quality: **Warn** (LSTM path added but predict is incomplete)
- Project Hygiene: **Pass**
- Basic Testing: **Pass** (`tests/test_smoke.py` still present)

### B) Forecasting Correctness & Leakage Prevention
- Data Handling: **Pass** (timestamp parsing + reindexing still enforced)
- Splits & Disjointness: **Pass**
- Rolling-Origin Evaluation: **Pass**
- Feature Leakage Checks: **Pass**
- **Weather Integration: Fail (removed)**
- Metrics Correctness: **Pass**
- Model Evaluation Correctness: **Pass** (test eval gated)

---

## Evidence Highlights (v2)
- Recursive backtest and model inference: `eskom_energy_demand_forecasting/modeling/train.py:355-579`.
- LSTM save path (final model): `eskom_energy_demand_forecasting/modeling/train.py:611-625`.
- Predict path lacks LSTM handling: `eskom_energy_demand_forecasting/modeling/predict.py:116-151`.
- Weather integration removed: `eskom_energy_demand_forecasting/dataset.py:104-120`, `eskom_energy_demand_forecasting/features.py:46-68`.

---

## Fix Checklist (Delta)
**P0 (Correctness)**
- [ ] Add LSTM support in `predict.py` (load Keras model + scalers + sequence length from `final_model.pkl`).
- [ ] Restore weather ingestion + merge (or explicitly document that weather is no longer used and remove from project claims).

**P1 (Repro/Usability)**
- [ ] Add a `make train-fast` target that sets `N_FOLDS`, `HORIZON_HOURS`, etc. for smoke runs.
- [ ] Add a dependency lockfile or conda environment file for reproducible installs.

**P2 (Polish)**
- [ ] Add a small validation plot report for one fold to confirm outputs visually.

---

## Safe-Run Status (v2)
- The most recent successful run produced `reports/fold_metrics.csv` and `models/final_model.pkl`, but was executed with reduced folds/horizon and without weather features. Results are not comparable to default settings.
- Test evaluation remains disabled (`RUN_TEST_EVAL=False`).
