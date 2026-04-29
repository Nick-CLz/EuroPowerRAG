# Evaluation Methodology — EuroPowerRAG v2

**Status:** F1 — Foundation. Read this before writing any v2 code.

The premise: every phase ships against a measurable target. No phase is "done" until its eval passes its threshold on a held-out set. Evals are first-class artifacts in this repo — they live in code, run on demand, and gate every release.

---

## The Three Evals (one per layer)

| Eval | What it measures | Threshold to ship | Runs in |
|------|------------------|-------------------|---------|
| **Sentiment κ** | Agreement between LLM and human labels on energy news | Cohen's κ ≥ 0.5 | `src/evaluation/eval_sentiment.py` |
| **Forecast directional accuracy** | % of next-day moves where the model gets the sign right | ≥ 55% on walk-forward | `src/evaluation/eval_forecast.py` |
| **Backtest Sharpe** | Risk-adjusted return on simulated trades | ≥ 0.8 over 6 months walk-forward, max DD ≤ 8% | `src/evaluation/eval_backtest.py` |

If any of these fall short, the corresponding phase is **not done**. No exceptions.

---

## Sentiment Eval

### Dataset

`data/eval/sentiment_gold.jsonl` — 30 hand-labeled news articles drawn from the existing v1 ingestion.

Each row:
```json
{
  "id": "guardian_2026_04_22_eu_taxes",
  "title": "EU plans to cut electricity taxes...",
  "summary": "The European Commission announced...",
  "country": "EU",
  "date": "2026-04-22",
  "label": -1
}
```

`label` is one of `{-1, 0, +1}` meaning bearish / neutral / bullish for **next-day day-ahead price** in the relevant country.

### Labeling protocol

- 30 articles minimum, balanced across `{-1, 0, +1}` (≥ 8 per class)
- Labeler does NOT see model output before labeling
- Two passes: label, then a 24h cooldown, then re-label. Disagreements with self get marked `unsure` and excluded.
- Edge cases (articles with no clear directional signal) get `0`, not excluded — the model needs to learn neutrality

### Metric

**Cohen's κ** (linear-weighted) between LLM discretized output and gold labels.

Discretization: `score < -0.33 → -1`, `score > 0.33 → +1`, else `0`.

### Thresholds

- κ < 0.3 → prompt is broken, redesign
- 0.3 ≤ κ < 0.5 → iterate prompt, do not ship
- κ ≥ 0.5 → ship Phase 1
- κ ≥ 0.7 → publish-quality

### Re-run cadence

- After every prompt change
- Weekly during active development
- Before every release tag

---

## Forecast Eval

### Dataset

`data/processed/prices_history.parquet` — ≥ 1 year of daily-aggregated day-ahead prices for DE_LU, FR, NL, GB.

### Splits

**Walk-forward only.** No random splits, no IID assumptions, no leakage.

```
Train window:   rolling 180 days
Validation:     next 30 days
Step forward:   30 days
```

For 365 days of history, this yields ~6 forecast windows per country. Aggregate metrics across all windows.

### Metrics

For each forecast horizon (next-day):

| Metric | Definition | Notes |
|--------|------------|-------|
| MAE | mean(\|y_true − y_pred\|) | EUR/MWh |
| RMSE | sqrt(mean((y_true − y_pred)²)) | Penalizes large misses |
| Directional accuracy | mean(sign(y_true_t − y_true_{t-1}) == sign(y_pred_t − y_true_{t-1})) | The one that matters for trading |
| Bias | mean(y_pred − y_true) | Detects systematic over/underestimation |

### Baselines (must beat these)

1. **Naive:** `y_pred(t+1) = y_true(t)` — yesterday's close
2. **Seasonal naive:** `y_pred(t+1) = y_true(t-7)` — last week's same-day price
3. **Rolling mean:** `y_pred(t+1) = mean(y_true[t-7:t])`

A new forecasting model **must beat all three baselines on directional accuracy** to be worth shipping. It's fine to lose on MAE if directional accuracy is meaningfully better — directional accuracy is what trading P&L depends on.

### Sentiment-as-feature acceptance test

Before sentiment can be added to the production forecaster, it must add ≥ **2 percentage points of directional accuracy** vs. the price-only model on the held-out set. If it doesn't, sentiment stays as a context layer for the analyst, not a feature.

This is the entire reason for Phase 1 → Phase 2 ordering. Build sentiment, eval it standalone, then prove it adds lift to the forecast.

---

## Backtest Eval

### Setup

Walk-forward backtest over the most recent 6 months of price history.

- For each day in the test period: agent calls `decide(country, date)` using only data available up to that date (no look-ahead)
- Position is opened at next-day open price, closed at next-day close
- Costs assumed: 0.1% slippage + flat €0 commission (commodity-grade assumption — refine when paper trading)

### Metrics

| Metric | Threshold | Why |
|--------|-----------|-----|
| Sharpe ratio (annualized) | ≥ 0.8 | Industry-recognized risk-adjusted return |
| Sortino ratio | ≥ 1.0 | Penalizes downside vol only |
| Max drawdown | ≤ 8% | Risk discipline check |
| Win rate | ≥ 50% | Sanity check (signal isn't pure noise) |
| P&L distribution | Right-skewed | Cuts losers fast, lets winners run |

The Sharpe target is **deliberately modest** — 0.8 out-of-sample on a single-asset directional strategy is real, hard, and useful. Anyone claiming Sharpe > 2 from a research toy on free data is overfitting.

### What "fail" looks like

If after 5 honest iterations the system can't hit Sharpe ≥ 0.8 with DD ≤ 8%, the correct response is **document the failure** in `docs/POSTMORTEM.md` and ship what we learned. That document is more interview-valuable than a perfect-but-fake number.

---

## Anti-patterns to avoid

| Pattern | Why it's wrong |
|---------|----------------|
| Test set used during model selection | Information leakage. Use validation, not test, until you ship. |
| Random train/test split on time series | Leaks future into past. Always walk-forward. |
| Optimizing on a single metric | Hide failures elsewhere. Always report the full panel. |
| Rerunning eval until it passes | This *is* overfitting on the eval set. Pre-register changes. |
| "It worked on the last week" | Cherry-picking. Show all walk-forward windows. |
| Ignoring the baseline | If you can't beat naive, stop. Naive is the floor. |

---

## Eval reports

After every eval run, append a row to `data/eval/eval_log.jsonl`:

```json
{
  "timestamp": "2026-04-29T12:00:00Z",
  "eval_name": "sentiment_kappa",
  "version": "git-sha-or-tag",
  "metric": 0.62,
  "threshold": 0.5,
  "passed": true,
  "notes": "After v3 prompt rewrite — added severity calibration"
}
```

This log is the audit trail. It also makes it impossible to lie to yourself about progress over time.

---

## Eval ownership

Every eval has one owner — the person responsible for keeping it green. Right now: you. When you bring on a collaborator, document who owns what in this file.
