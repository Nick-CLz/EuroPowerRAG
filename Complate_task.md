# Completed Tasks

## T-Series — Bootstrap and Plumbing
- [x] T1. Populate `data/processed/prices_history.parquet`
- [x] T2. Add `anthropic` SDK to the stack
  - [x] Update requirements.txt
  - [x] Update .env.example
  - [x] Update src/utils/budget.py (added Claude pricing + cached_tokens)
- [x] T3. Wire prompt caching on every Claude call
  - Singleton client, retry with backoff, cache_control on system block
- [x] T4. Create `src/evaluation/eval_forecast.py`
  - Dynamic model registry, CLI args, per-model debug logging
- [x] T5. Create `src/evaluation/eval_backtest.py`
  - Sharpe + Sortino + MaxDD + WinRate, pass/fail gate, exit codes
- [x] T6. Implement `src/backtest/runner.py`
  - Walk-forward with tqdm, O(log n) next-day lookup, confidence tracking
- [x] T7. Initialize `data/eval/eval_log.jsonl`

## Phase 2 — Forecast
- [x] P2.2 ARIMA forecaster (`src/forecast/arima.py`)
  - Residual-based uncertainty, warnings suppression, shared loader
- [x] P2.3 XGBoost forecaster (`src/forecast/xgb.py`)
  - rolling_7_std feature, real sentiment join from parquet, residual uncertainty
- [x] P2.4 XGBoost + sentiment features
  - Lagged merge from `sentiment_daily.parquet`, graceful zero-fill
- [x] P2.5 Forecast API (`src/forecast/api.py`)
  - XGB → ARIMA → rolling_mean cascade with per-level logging

## Phase 4 — Decision Agent
- [x] P4.2 Implement decision agent (`src/agent/decide.py`)
  - Fixed broken imports, proper sentiment/risk helpers, structured Claude prompt
- [x] P4.3 Self-critique loop
  - Checks confidence, rationale length, and input-category references
- [x] P4.4 CLI tool (`python -m src.agent.decide --country DE_LU --date 2026-04-29`)

## Phase 5 — Backtest + Paper Trade
- [x] P5.3 6-month walk-forward run (`src/backtest/run_6mo.py`)
  - Per-country summary, automatic postmortem on failure, CLI --days
- [x] P5.4 Paper trading scheduler (updated `scheduler.py`)
- [x] P5.5 Streamlit dashboard page (added to `app.py`)

## Polish
- [x] Add CI on GitHub Actions (`.github/workflows/ci.yml`)
- [x] Write blog post / case study (`docs/blog_post.md`)
- [x] Shared price loader (`src/forecast/loader.py`) — eliminates duplication

## Quality Upgrade Pass (Opus 4.7)
- [x] `anthropic_client.py` — singleton client, retry with exponential backoff, proper error classes
- [x] `arima.py` — shared loader, warnings.catch_warnings, residual std uncertainty
- [x] `xgb.py` — shared loader, real sentiment join, rolling_7_std, subsample regularization
- [x] `api.py` — explicit 3-model cascade with per-level logging
- [x] `decide.py` — fixed broken imports, proper sentiment/risk helpers, structured Claude prompt, content-aware self-critique
- [x] `runner.py` — O(log n) next-day lookup via np.searchsorted, tqdm, confidence column
- [x] `eval_forecast.py` — dynamic model registry, debug logging on failures, CLI args
- [x] `eval_backtest.py` — Sortino, total/avg PnL, pass/fail formatting, exit codes
- [x] `run_6mo.py` — progress reporting, automatic postmortem, CLI --days

## Verification
- [x] All imports pass cleanly
- [x] 71/71 existing tests pass
- [x] Forecast eval runs end-to-end (6 models × 6 countries)
- [x] Decision agent works with heuristic fallback when Claude not configured
- [x] Self-critique correctly overrides low-confidence signals to HOLD
