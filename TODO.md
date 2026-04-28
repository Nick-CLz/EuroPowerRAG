# TODO — EuroPowerRAG v2

Working list for the trading-signal extension. Tasks are ordered by **eval-first dependency** — every task either builds an eval or ships against one.

Format: `[ ]` open · `[x]` done · `[~]` in progress · `[!]` blocked

Each task has an explicit **DoD** (Definition of Done). If you can't write a DoD that an outsider could verify, the task is too vague — break it down.

---

## Foundation (do these first, no exceptions)

- [ ] **F1. Decide eval methodology end-to-end before writing any v2 code**
  DoD: `docs/EVAL.md` exists. Lists the 3 metrics that determine v2 success (sentiment κ, forecast directional accuracy, backtest Sharpe). Specifies test/holdout split.

- [ ] **F2. Set up price history dataset**
  DoD: `data/processed/prices_history.parquet` contains ≥ 1 year of DE_LU + FR + NL + GB day-ahead prices. Either via ENTSO-E (preferred) or Ember CSV fallback.

- [ ] **F3. Add cost & latency budgets to `.env`**
  DoD: `.env.example` documents `MAX_DAILY_LLM_USD=5`, `MAX_QUERY_LATENCY_S=10`. `src/utils/budget.py` enforces them at call sites.

---

## Phase 1 — Sentiment

- [ ] **P1.1. Hand-label 30 news articles for sentiment**
  DoD: `data/eval/sentiment_gold.jsonl` with `{title, summary, label ∈ {-1, 0, +1}}`. You labeled them yourself, blind to LLM output.

- [ ] **P1.2. Implement structured-output sentiment scorer**
  DoD: `src/sentiment/scorer.py` exposes `score_article(title, summary) → SentimentScore` with Pydantic schema. Uses Gemini structured output (response_schema). Single call, < 2 s p95.

- [ ] **P1.3. Eval scorer against gold set**
  DoD: `python -m src.evaluation.eval_sentiment` prints Cohen's κ. Pass threshold: κ ≥ 0.5. If below, iterate prompt before continuing.

- [ ] **P1.4. Wire sentiment scoring into ingestion pipeline**
  DoD: Every news doc in `data/processed/news.jsonl` has `sentiment_score` and `sentiment_confidence` in metadata. Run `python ingest.py` to verify.

- [ ] **P1.5. Build daily sentiment time series**
  DoD: `data/processed/sentiment_daily.parquet` with columns `[date, country, mean_score, n_articles, std_score]`.

- [ ] **P1.6. Streamlit overlay chart**
  DoD: New page in `app.py` showing price (when available) + sentiment as dual-axis chart per country.

---

## Phase 2 — Forecast

- [ ] **P2.1. Build naive baseline forecaster**
  DoD: `src/forecast/baseline.py` predicts next-day price as today's close. Walk-forward eval reports MAE / directional accuracy on last 90 days.

- [ ] **P2.2. ARIMA forecaster**
  DoD: `src/forecast/arima.py`. Beats naive on MAE by ≥ 5%. Document the lag / order selected and why.

- [ ] **P2.3. XGBoost forecaster (price-only features)**
  DoD: `src/forecast/xgb.py` with lag-1, lag-7, day-of-week, hour-of-day. Beats ARIMA on directional accuracy.

- [ ] **P2.4. XGBoost + sentiment features**
  DoD: Same model + `mean_sentiment_t-1`, `n_articles_t-1`. Compare to P2.3. Sentiment must add ≥ 2 percentage points of directional accuracy or it gets dropped from the model.

- [ ] **P2.5. Forecast API**
  DoD: `forecast_next_day(country, date) → (price, std)` works end-to-end. Tests cover happy path + edge cases (missing data, weekend, DST).

---

## Phase 3 — Risk

- [ ] **P3.1. Realized volatility calculator**
  DoD: `src/risk/volatility.py` returns rolling 20-day σ per country. Tests verify against manually-computed numbers on 5 fixture rows.

- [ ] **P3.2. Position sizer**
  DoD: `src/risk/sizing.py` implements fractional Kelly with a hard cap. Inputs: signal strength, vol, account size. Output: position size in MWh.

- [ ] **P3.3. Stop / target generator**
  DoD: Returns `{stop_price, target_price, max_loss_eur}` given entry, direction, vol.

- [ ] **P3.4. Risk eval**
  DoD: `docs/EVAL.md` includes risk section. On a 6-month walk-forward, max drawdown ≤ 8% with these rules.

---

## Phase 4 — Decision Agent

- [ ] **P4.1. Define structured signal schema**
  DoD: `src/agent/schema.py` with `TradeSignal` Pydantic model: `direction`, `size`, `confidence`, `rationale`, `forecast_price`, `sentiment_score`, `volatility`, `sources`.

- [ ] **P4.2. Implement decision agent**
  DoD: `src/agent/decide.py` exposes `decide(country, date) → TradeSignal`. Uses LangChain agent with `get_forecast`, `get_sentiment`, `get_risk` tools.

- [ ] **P4.3. Self-critique loop**
  DoD: After producing a signal, agent runs a critique pass. If `confidence < 0.6` OR `rationale` doesn't reference all three inputs, output `HOLD` with explanation.

- [ ] **P4.4. CLI tool**
  DoD: `python decide.py --country DE_LU --date 2026-04-29` prints a formatted signal + sources.

---

## Phase 5 — Backtest + Paper Trade

- [ ] **P5.1. Walk-forward backtest skeleton**
  DoD: `src/backtest/runner.py` takes a date range, runs `decide()` daily, marks to next-day actual. Outputs trade log CSV.

- [ ] **P5.2. Performance metrics**
  DoD: `src/backtest/metrics.py` computes Sharpe, Sortino, max drawdown, win rate. Tests on a known fixture P&L stream.

- [ ] **P5.3. 6-month walk-forward run**
  DoD: `data/backtest/results.json` with ≥ 0.8 Sharpe and ≤ 8% drawdown — or a written analysis of why not, with hypotheses to test next.

- [ ] **P5.4. Paper trading scheduler**
  DoD: `scheduler.py` runs `decide()` daily at 14:00 CET, logs to SQLite. New table: `paper_trades(timestamp, country, signal, executed_price, mark_price, pnl)`.

- [ ] **P5.5. Streamlit dashboard page**
  DoD: New page in `app.py` showing equity curve, drawdown, recent signals table, P&L attribution.

---

## Polish (only after P5.3 hits its DoD)

- [ ] Loom walkthrough (3 minutes)
- [ ] Update README with v2 architecture diagram
- [ ] Pin model versions in `requirements.txt`
- [ ] Add CI on GitHub Actions (run tests on push)
- [ ] Write a short blog post / case study

---

## Done (from v1)

- [x] RAG pipeline (ENTSO-E + RSS + PDF)
- [x] ChromaDB index with metadata filtering
- [x] Streamlit Q&A UI
- [x] Pytest suite (34 tests)
- [x] Eval framework (Precision@5, Faithfulness)
- [x] Migrate to Google AI Studio free tier
- [x] Push to GitHub
