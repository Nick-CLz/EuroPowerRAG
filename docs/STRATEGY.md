# EuroPowerRAG v2 — Strategy

**Mission:** Turn the v1 retrieval system into a decision-support stack that quantifies sentiment, forecasts price, manages risk, and produces tradeable signals on European power markets.

This is **not a bot that places real trades.** It is a research-grade signal engine that an analyst would consume. Real money decisions stay with humans.

---

## Why this is the right next bet

v1 proves we can ingest, retrieve, and ground answers. v2 turns that retrieval into a **measurable trading signal**. That's the leap from "RAG demo" to "quantitative research tool" — exactly the language Cobblestone's JD asks for: *"task-specific agents… integrated programmatically into analytical workflows."*

The work also stress-tests three things every commodity trading shop cares about:

1. **Signal quality** — does sentiment + forecast actually predict next-day moves?
2. **Risk discipline** — can the system size positions without blowing up?
3. **Reproducibility** — can we backtest, eval, and iterate without snowflakes?

---

## North-Star Metric

**Out-of-sample Sharpe ratio on a walk-forward backtest of DE_LU day-ahead price direction, ≥ 0.8 over 6 months of paper trading, with max drawdown ≤ 8%.**

If we can't hit that, no amount of polish makes the system useful. The eval is the product.

---

## Architecture (target state)

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER (extends v1)                       │
│   ENTSO-E prices │ ENTSO-E generation │ News (RSS) │ PDFs        │
└─────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  ┌──────────┐       ┌─────────────┐     ┌────────────┐
  │ Sentiment│       │  Forecast   │     │  Risk      │
  │  Scorer  │       │   Module    │     │  Module    │
  │  (LLM)   │       │ (ARIMA→XGB) │     │ (vol, VaR) │
  └──────────┘       └─────────────┘     └────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                   ┌─────────────────┐
                   │  Decision Agent │
                   │   (LangChain)   │
                   └─────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Backtest Engine │
                   │  + Paper Trade  │
                   └─────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Streamlit Dash  │
                   │ (Signals + P&L) │
                   └─────────────────┘
```

---

## Phased Plan

Each phase ships **end-to-end value** before moving to the next. No phase is "scaffolding for the next phase" — every phase produces a measurable artifact.

### Phase 1 — Sentiment Layer (3–4 days)

**Goal:** Daily sentiment time series per country, joinable with price data.

- Add `sentiment_score ∈ [-1, +1]` to every news doc at ingest time, via Gemini structured-output prompt
- Aggregate to daily country-level series (`data/sentiment/daily.parquet`)
- Eval: human-label 30 articles, measure agreement with LLM scores (target Cohen's κ ≥ 0.5)
- **Ship:** A chart in the Streamlit UI showing sentiment vs. price overlay per country

### Phase 2 — Price Forecast (3–5 days)

**Goal:** Next-day day-ahead price forecast, with directional accuracy beating naive baseline.

- Pull ≥ 1 year of ENTSO-E day-ahead prices for DE_LU, FR, NL, GB
- Build hierarchy of models: naive (yesterday's close) → ARIMA → XGBoost (with lag, calendar, weather optional, sentiment features)
- Walk-forward eval on last 90 days: MAE, RMSE, directional accuracy
- **Ship:** `forecast.py` with one-line API: `forecast_next_day(country, date) → (price, confidence)`

### Phase 3 — Risk Module (2–3 days)

**Goal:** Position-size and stop-loss recommendations grounded in volatility.

- Realized + GARCH-style volatility per country
- Kelly-fraction sizing capped at conservative fraction (e.g. 0.25× full Kelly)
- Stop-loss / take-profit at k × σ
- **Ship:** `risk.py` returning `{position_size, stop, target, max_loss}`

### Phase 4 — Decision Agent (3–4 days)

**Goal:** Single agent call that reads forecast + sentiment + risk and returns a structured trade signal.

- LangChain agent with three tools: `get_forecast`, `get_sentiment`, `get_risk`
- Structured Pydantic output: `{direction: BUY/SELL/HOLD, size: float, confidence: float, rationale: str, sources: list[str]}`
- Hard rule: every signal cites the forecast number, sentiment score, and risk constraint that drove it
- **Ship:** CLI tool `python decide.py --country DE_LU --date today`

### Phase 5 — Backtest + Paper Trade (4–6 days)

**Goal:** Walk-forward backtest producing the north-star Sharpe number. Paper-trade live for ongoing eval.

- Walk-forward backtest on the last 6 months
- Metrics: Sharpe, Sortino, max drawdown, win rate, hit rate, P&L distribution
- Paper trading loop: APScheduler runs decide.py daily, logs signals to SQLite, marks-to-market against next-day actuals
- **Ship:** Streamlit page showing equity curve, drawdown chart, recent signals table

---

## Tech Choices (and why)

| Component | Choice | Why |
|-----------|--------|-----|
| Sentiment model | Gemini 2.5 Flash structured output | Already wired up; cheap; structured JSON via `response_schema` |
| Forecast | ARIMA → XGBoost | Start dumb. Ship working baseline. Beat it on merit. |
| Risk | Numpy + simple GARCH | Don't over-engineer. Most quant signals fail in spec, not implementation. |
| Agent | LangChain LCEL | Already in the stack. Tool use is well-supported. |
| Backtest | Pure pandas, no `vectorbt`/`backtrader` | Tiny scope. Custom code is 80 lines and you understand every line. |
| Storage | Parquet for time series, SQLite for signals | Right tool for each. |

---

## Out of Scope (resist scope creep)

- Real money execution
- Order book / intraday data
- Cross-border / interconnector arb
- Reinforcement learning
- Web dashboard beyond Streamlit
- Multi-asset portfolios

If you finish Phase 5 with time to spare, write better evals before adding scope. Karpathy's "the eval is the product" — better evals compound.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Sentiment doesn't predict price | Build the eval in Phase 1 BEFORE Phase 2. If κ < 0.4 or correlation < 0.1, scrap sentiment-as-feature and use it only as a context layer. |
| Free-tier API limits | Cache aggressively. Embeddings already cached. Move sentiment scoring to nightly batch, not real-time. |
| Backtest overfit | Walk-forward only. No look-ahead. Hold out last 30 days. Run sensitivity analysis on hyperparameters. |
| ENTSO-E token never arrives | Phase 2 needs price history. Fallback: scrape Ember's daily price CSV (works without token). Document the workaround. |
| Scope creep | Each phase has a single shippable artifact. If a phase isn't shippable end-to-end in its budget, cut features, don't slip. |
