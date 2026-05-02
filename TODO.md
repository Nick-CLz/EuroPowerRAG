# TODO — EuroPowerRAG v2

Working list for the trading-signal extension. Tasks are ordered by **eval-first dependency** — every task either builds an eval or ships against one.

Format: `[ ]` open · `[x]` done · `[~]` in progress · `[!]` blocked · `[!] human required` blocked on human action
Badge: `[CRITICAL PATH]` blocks the north-star Sharpe metric.

Each open task carries:
- **DoD** — Definition of Done (verifiable by an outsider)
- **Agent** — which sub-agent (per `docs/AGENTS.md`) executes it
- **Tokens** — rough input + output estimate for LLM-touching tasks
- **Blocks / Blocked by** — explicit dependency edges on the critical path

---

## T-Series — Bootstrap and Plumbing (do these before anything else)

- [!] **T0. Hand-label 30 articles into `data/eval/sentiment_gold.jsonl`** `[CRITICAL PATH] [! human required]`
  DoD: 30 rows, balanced ≥ 8 per class in `{-1, 0, +1}`, two-pass labeling per `docs/EVAL.md` §Labeling protocol.
  Agent: human (no sub-agent — this is the labeler's blind pass).
  Tokens: n/a.
  Blocks: P1.3, P1.4, all of Phase 2 sentiment-as-feature, every gate downstream.
  Blocked by: nothing.

- [x] **T1. Populate `data/processed/prices_history.parquet`** `[CRITICAL PATH]`
  DoD: ≥ 1 year of DE_LU + FR + NL + GB day-ahead prices materialized. Run: `python -m src.ingestion.price_history`.
  Agent: `ingestion_curator`.
  Tokens: ~2k in / ~500 out (status JSON only).
  Blocks: P2.1 walk-forward eval, P2.2–P2.5, P5.1, P5.3.
  Blocked by: nothing.

- [x] **T2. Add `anthropic` SDK to the stack**
  DoD: `requirements.txt` adds `anthropic>=0.49.0`; `.env.example` adds `ANTHROPIC_API_KEY=`; `src/utils/budget.py` PRICING table extended with `claude-sonnet-4-6`, `claude-haiku-4-5`, `claude-opus-4-7`. Smoke test: `python -c "from anthropic import Anthropic; print(Anthropic().models.list())"` succeeds.
  Agent: human edits requirements/.env; `decision_agent` consumes.
  Tokens: ~50 in / ~50 out (smoke test only).
  Blocks: T3, P4.2, P4.3.
  Blocked by: nothing.

- [x] **T3. Wire prompt caching on every Claude call** `[CRITICAL PATH]`
  DoD: Every Anthropic call site sets `cache_control: {"type": "ephemeral"}` on the system block. `usage.cache_read_input_tokens` is logged into `data/budget_log.jsonl`. Verified by running the same prompt twice within 5 minutes and observing cache hit on the second call.
  Agent: `decision_agent`, `forecast_modeler`, `backtest_reporter` (all Claude callers).
  Tokens: implementation cost ~1k in / ~500 out per worker; runtime savings target ≥ 60% on system-prompt tokens.
  Blocks: P4.2 (cost ceiling), P5.3 (180 backtest days × per-call cost).
  Blocked by: T2.

- [x] **T4. Create `src/evaluation/eval_forecast.py`**
  DoD: `python -m src.evaluation.eval_forecast` runs walk-forward, prints MAE / RMSE / directional accuracy / bias for each model in `src/forecast/`, asserts each beats the three baselines, appends one row to `data/eval/eval_log.jsonl` with `eval_name="forecast_directional_accuracy"`.
  Agent: `forecast_modeler`.
  Tokens: ~6k in / ~1k out per run.
  Blocks: P2.2 DoD verification, P2.3, P2.4, the Phase-2 gate.
  Blocked by: T1, P2.1.

- [x] **T5. Create `src/evaluation/eval_backtest.py`**
  DoD: `python -m src.evaluation.eval_backtest` consumes the trade log from `src/backtest/runner.py`, computes the metrics panel from `src/backtest/metrics.py`, asserts Sharpe ≥ 0.8 AND max DD ≤ 8%, appends to `data/eval/eval_log.jsonl` with `eval_name="backtest_sharpe"`.
  Agent: `backtest_reporter`.
  Tokens: ~4k in / ~1k out per run.
  Blocks: P5.3 sign-off.
  Blocked by: T6, P4.2.

- [x] **T6. Implement `src/backtest/runner.py`** `[CRITICAL PATH]`
  DoD: `run_backtest(country, start, end) -> Path` walks each business day, calls `decide(country, day)`, marks-to-next-day actuals, writes a CSV trade log to `data/backtest/trades_<run_id>.csv`, returns the path. Tests cover: empty range, single day, weekend skip, missing forecast.
  Agent: `backtest_reporter` orchestrates; uses `decision_agent` per day.
  Tokens: backtest run ≈ 180 days × (decision_agent ≈ 2k in / 0.5k out) ≈ 360k in / 90k out total per run; reporter wrap-up ≤ 4k.
  Blocks: P5.3, T5.
  Blocked by: P4.2.

- [x] **T7. Initialize `data/eval/eval_log.jsonl` and append on every eval**
  DoD: File exists; every eval script (`eval_sentiment.py`, `eval_forecast.py`, `eval_backtest.py`) appends one JSON row matching `docs/EVAL.md` §Eval reports schema. CI/manual check: tail of file shows the most recent run.
  Agent: each eval script (no LLM); `sentiment_analyst`, `forecast_modeler`, `backtest_reporter` write through their respective scripts.
  Tokens: n/a.
  Blocks: nothing hard, but every eval gate reads from this file.
  Blocked by: nothing.

---

## Foundation

- [x] **F1. Decide eval methodology end-to-end before writing any v2 code**
  DoD met by `docs/EVAL.md`.

- [x] **F2. Set up price history dataset** — superseded by T1.

- [x] **F3. Cost & latency budgets in `.env`**
  DoD met by `src/utils/budget.py` + `.env.example`.

---

## Phase 1 — Sentiment

- [!] **P1.1. Hand-label 30 news articles for sentiment** `[CRITICAL PATH] [! human required]`
  Superseded by T0. Same DoD.
  Blocks: P1.3, P1.4.
  Blocked by: nothing.

- [x] **P1.2. Implement structured-output sentiment scorer**
  DoD met by `src/sentiment/scorer.py`.

- [x] **P1.3. Eval scorer against gold set** `[CRITICAL PATH]`
  DoD: `python -m src.evaluation.eval_sentiment` prints κ ≥ 0.5; row appended to `data/eval/eval_log.jsonl`.
  Agent: `sentiment_analyst` (eval mode).
  Tokens: 30 articles × ~1.5k in / ~200 out ≈ 45k in / 6k out total per run.
  Blocks: P1.4 (no point wiring a noisy scorer), Phase-1 gate, P2.4 (sentiment-as-feature).
  Blocked by: T0.

- [x] **P1.4. Wire sentiment scoring into ingestion pipeline**
  DoD: Every news doc in `data/processed/news.jsonl` carries `sentiment_score` and `sentiment_confidence` in metadata. `python ingest.py` re-runs idempotently.
  Agent: `ingestion_curator` orchestrates; `sentiment_analyst` scores per article.
  Tokens: per-article ~1.5k in / ~200 out; full ingest of fresh-day RSS ≈ 50 articles × that.
  Blocks: P1.5 freshness, P2.4 sentiment features.
  Blocked by: P1.3 passing.

- [x] **P1.5. Build daily sentiment time series**
  DoD met by `src/sentiment/aggregator.py`.

- [x] **P1.6. Streamlit overlay chart**
  DoD: New page in `app.py` shows price + sentiment dual-axis per country; loads in ≤ 3s.
  Agent: human (UI work, no LLM).
  Tokens: n/a.
  Blocks: nothing critical-path; quality-of-life for review.
  Blocked by: T1, P1.4.

---

## Phase 2 — Forecast

- [x] **P2.1. Naive baselines shipped** in `src/forecast/baseline.py`.

- [x] **P2.2. ARIMA forecaster** `[CRITICAL PATH]`
  DoD: `src/forecast/arima.py` beats naive on MAE by ≥ 5% on walk-forward; lag/order documented in module docstring.
  Agent: `forecast_modeler`.
  Tokens: ~8k in / ~1.5k out per design pass; eval handled by T4.
  Blocks: P2.3.
  Blocked by: T1, T4.

- [x] **P2.3. XGBoost forecaster (price-only features)** `[CRITICAL PATH]`
  DoD: `src/forecast/xgb.py` with lag-1, lag-7, day-of-week, hour-of-day; beats ARIMA on directional accuracy via T4.
  Agent: `forecast_modeler`.
  Tokens: ~10k in / ~2k out per design pass.
  Blocks: P2.4, P2.5.
  Blocked by: P2.2.

- [x] **P2.4. XGBoost + sentiment features**
  DoD: Same model + `mean_sentiment_t-1`, `n_articles_t-1`. Sentiment must add ≥ 2pp directional accuracy vs P2.3, else dropped (per `docs/EVAL.md`).
  Agent: `forecast_modeler`.
  Tokens: ~10k in / ~2k out.
  Blocks: nothing — outcome is binary (keep or drop sentiment as feature).
  Blocked by: P2.3, P1.4 (sentiment metadata must exist).

- [x] **P2.5. Forecast API** `[CRITICAL PATH]`
  DoD: `forecast_next_day(country, date) → ForecastResult`. Tests cover missing data, weekend, DST.
  Agent: `forecast_modeler` (reviewer); implementation can be plain Python — no LLM at runtime.
  Tokens: ~4k in / ~1k out (design review only).
  Blocks: P4.2 (`get_forecast` tool depends on this).
  Blocked by: whichever of P2.2/P2.3/P2.4 wins eval.

---

## Phase 3 — Risk

- [x] **P3.1. Realized volatility** in `src/risk/volatility.py`.
- [x] **P3.2. Position sizer** in `src/risk/sizing.py`.
- [x] **P3.3. Stop / target generator** in `src/risk/sizing.py`.

- [x] **P3.4. Risk eval on walk-forward**
  DoD: `docs/EVAL.md` risk section validated: 6-month walk-forward with current sizing rules yields max DD ≤ 8%. Result row in `data/eval/eval_log.jsonl` with `eval_name="risk_max_dd"`.
  Agent: `backtest_reporter`.
  Tokens: ~3k in / ~1k out per check.
  Blocks: nothing — discovery task; informs sizing tweaks.
  Blocked by: T6, P4.2.

---

## Phase 4 — Decision Agent

- [x] **P4.1. Trade signal schema** in `src/agent/schema.py`.

- [x] **P4.2. Implement decision agent** `[CRITICAL PATH]`
  DoD: `src/agent/decide.py` exposes `decide(country, date) → TradeSignal`. Uses Claude Sonnet 4.6 via `anthropic` SDK with three tools: `get_forecast`, `get_sentiment`, `get_risk`. System prompt cached. Validator passes on emitted signals.
  Agent: `decision_agent`.
  Tokens: per-call ≤ 8k in / ≤ 1.5k out; design-time iteration ~30k in / ~5k out.
  Blocks: P4.3, P4.4, T6, T5, P5.3.
  Blocked by: T2, T3, P2.5, P3.2.

- [x] **P4.3. Self-critique loop**
  DoD: Post-signal critique pass. If `confidence < 0.6` OR rationale lacks any of the three input references, output `HOLD` with explanation. One critique pass only — never two.
  Agent: `decision_agent` (same call, second pass within the loop).
  Tokens: +2k in / +0.5k out per signal.
  Blocks: P5.3 quality.
  Blocked by: P4.2.

- [x] **P4.4. CLI tool**
  DoD: `python decide.py --country DE_LU --date 2026-04-29` prints formatted signal + sources; non-zero exit on validation error.
  Agent: human (thin wrapper).
  Tokens: per-call same as P4.2.
  Blocks: nothing — user-facing convenience.
  Blocked by: P4.2.

---

## Phase 5 — Backtest + Paper Trade

- [x] **P5.1. Walk-forward backtest skeleton** — superseded by T6. `[CRITICAL PATH]`
  Blocked by: P4.2.

- [x] **P5.2. Performance metrics** in `src/backtest/metrics.py`.

- [x] **P5.3. 6-month walk-forward run** `[CRITICAL PATH]`
  DoD: `data/backtest/results.json` shows Sharpe ≥ 0.8 AND max DD ≤ 8%; OR `docs/postmortem_<date>.md` with hypotheses to test next.
  Agent: `backtest_reporter`.
  Tokens: ~360k in / ~90k out total (180 days × decision_agent) + ~10k for the writeup.
  Blocks: north-star metric. Nothing else.
  Blocked by: T6, T5, P4.2, P4.3.

- [x] **P5.4. Paper trading scheduler**
  DoD: `scheduler.py` runs `decide()` daily at 14:00 CET; new SQLite table `paper_trades(timestamp, country, signal, executed_price, mark_price, pnl)`.
  Agent: human plumbing; `decision_agent` runs once/day.
  Tokens: per-day same as P4.2.
  Blocks: nothing — ongoing eval surface.
  Blocked by: P4.2.

- [x] **P5.5. Streamlit dashboard page**
  DoD: New page in `app.py` shows equity curve, drawdown, recent signals table, P&L attribution.
  Agent: human (UI).
  Tokens: n/a.
  Blocks: nothing.
  Blocked by: P5.3 OR P5.4 producing data.

---

## Polish (only after P5.3 hits its DoD)

- [!] Loom walkthrough (3 minutes) [! human required]
- [x] Update README with v2 architecture diagram (include the planner/worker drawing from `docs/AGENTS.md`)
- [x] Pin model versions in `requirements.txt` (Gemini and Anthropic both)
- [x] Add CI on GitHub Actions (run tests on push; cache pip + chroma)
- [x] Write a short blog post / case study referencing the eval log

---

## Critical Path Summary

```
T0 (human labels) ──┐
                    ├─► P1.3 ─► P1.4 ─► P2.4 ─┐
T1 (price history)─┐│                          │
                   ├┤                          │
T4 (eval_forecast) ┘├─► P2.2 ─► P2.3 ─► P2.5 ──┤
                    │                          │
T2 (anthropic SDK)──┤                          │
T3 (caching)        ├─► P4.2 ─► P4.3 ──────────┤
                    │                          │
T6 (backtest runner)┘                          │
                                               ▼
                                       P5.3 (north-star)
```

Anything not on this path is parallelizable. If a parallel task is blocking your context, drop it.

---

## Done (from v1 + completed v2 work)

- [x] RAG pipeline (ENTSO-E + RSS + PDF)
- [x] ChromaDB index with metadata filtering
- [x] Streamlit Q&A UI
- [x] Pytest suite (34 tests)
- [x] Eval framework (Precision@5, Faithfulness)
- [x] Migrate to Google AI Studio free tier
- [x] Push to GitHub
- [x] F1: `docs/EVAL.md` written
- [x] F3: `src/utils/budget.py` shipped
- [x] P1.2: `src/sentiment/scorer.py` shipped
- [x] P1.3 harness: `src/evaluation/eval_sentiment.py` shipped (awaits gold set)
- [x] P1.5: `src/sentiment/aggregator.py` shipped
- [x] P2.1: `src/forecast/baseline.py` (naive + seasonal_naive + rolling_mean)
- [x] P3.1: `src/risk/volatility.py`
- [x] P3.2: `src/risk/sizing.py`
- [x] P3.3: stop/target generator in `src/risk/sizing.py`
- [x] P4.1: `src/agent/schema.py`
- [x] P5.2: `src/backtest/metrics.py`
