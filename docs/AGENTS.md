# Multi-Agent Playbook — EuroPowerRAG v2

**Audience:** the Claude Code CLI orchestrator and any human reviewer.
**Scope:** how the five sub-agents that build, evaluate, and run the trading-signal stack are wired together.
**Status:** authoritative. If code disagrees with this doc, fix the code.

This file replaces the v1 "10 Rules" prose. Those rules are now compressed into the operating contract below; the discipline they encode (eval-first, structured output, ≤3 tools, self-critique, budgets, human-in-the-loop) is enforced by the agent definitions themselves rather than by appeal to principle.

The architecture is a **planner–worker** topology: one Opus orchestrator routes work to four specialised workers (Sonnet/Haiku). Workers never call each other — every handoff goes through the orchestrator, which means context is shaped per-hop and failures are caught at one place.

```
                    ┌─────────────────────────────┐
                    │   orchestrator (opus-4-7)   │
                    │   plans · routes · gates    │
                    └──────────────┬──────────────┘
        ┌────────────┬─────────────┼─────────────┬────────────┐
        ▼            ▼             ▼             ▼            ▼
   ingestion_   sentiment_    forecast_     decision_    backtest_
   curator      analyst       modeler       agent        reporter
   (sonnet)     (haiku)       (sonnet)      (sonnet)     (sonnet)
```

---

## 1. Agent Roster

### 1.1 `ingestion_curator`

| Field | Value |
|---|---|
| **Model** | `claude-sonnet-4-6` |
| **Role** | Orchestrate fetching, deduping, and persisting raw market data (ENTSO-E prices, news RSS, PDFs) to the parquet/jsonl cache. |
| **Trigger** | Manually (T1 bootstrap) or by `scheduler.py` daily at 06:00 CET before any downstream agent runs. |
| **Inputs** | `.env` credentials, the existing `src/ingestion/{entsoe_client,rss_scraper,pdf_loader,price_history}.py` modules, last-success timestamp from `data/processed/ingest_state.json`. |
| **Outputs** | Updated `data/processed/prices_history.parquet`, `news.jsonl`, `entsoe.jsonl`; a single-line JSON status report `{rows_added, sources_failed[], duration_s}`. |
| **Token budget** | ≤ 12k input / ≤ 2k output. Mostly tool I/O; the model only summarises results. |
| **Tools allowed** | `bash` (run `python -m src.ingestion.*`), `read_file`, `write_file` (for the status report only). No external API calls — those go through the existing Python clients. |
| **Handoff** | Returns status JSON to the orchestrator. On `sources_failed` non-empty, orchestrator decides whether to proceed with stale data or halt. |

### 1.2 `sentiment_analyst`

| Field | Value |
|---|---|
| **Model** | `claude-haiku-4-5` |
| **Role** | Score one news article (or batch) for next-day price impact and emit a `SentimentScore`. Also runs the κ-eval against the gold set on demand. |
| **Trigger** | Called per-article during ingest (P1.4 wiring), and called once when the orchestrator runs the sentiment gate (P1.3). |
| **Inputs** | `{title: str, summary: str}` — nothing else. The system prompt is constant (cache-eligible). For eval mode: path to `data/eval/sentiment_gold.jsonl`. |
| **Outputs** | `SentimentScore` (Pydantic) per article. For eval: appends one row to `data/eval/eval_log.jsonl` and returns `{kappa, n, passed}`. |
| **Token budget** | ≤ 1.5k input / ≤ 200 output per article. System prompt cached (`cache_control: {"type": "ephemeral"}`). |
| **Tools allowed** | None. Pure JSON-out call via `response_format` / tool-use. |
| **Handoff** | Score is persisted to `news.jsonl` metadata. Eval result returns to orchestrator and gates Phase 1. |

### 1.3 `forecast_modeler`

| Field | Value |
|---|---|
| **Model** | `claude-sonnet-4-6` |
| **Role** | Choose, fit, and evaluate the next-day price forecaster. Compares naive/seasonal_naive/rolling_mean against ARIMA and XGBoost variants on a walk-forward split. Returns the winning model name plus a `ForecastResult`-shaped prediction for the request date. |
| **Trigger** | Once per Phase-2 build cycle (P2.2 → P2.4 → P2.5), and once per backtest day inside the runner. |
| **Inputs** | `data/processed/prices_history.parquet` slice (≤ 2 years), `data/processed/sentiment_daily.parquet`, the target date, the eval thresholds from `docs/EVAL.md`. The model never sees the raw frame — the orchestrator passes a serialized summary (column names + last-90-day stats) and the file path. |
| **Outputs** | `ForecastResult` Pydantic, plus a markdown row appended to `data/eval/eval_log.jsonl` (`forecast_directional_accuracy`). |
| **Token budget** | ≤ 14k input / ≤ 2k output. Most tokens go to the eval-table summary, not raw data. |
| **Tools allowed** | `bash` (run `python -m src.forecast.*` and `python -m src.evaluation.eval_forecast`), `read_file` (parquet metadata only via a small helper, never the full frame). |
| **Handoff** | If walk-forward directional accuracy ≥ 55% and beats all three baselines, the orchestrator unlocks Phase 4. Otherwise the agent returns a written diagnosis and Phase 4 stays blocked. |

### 1.4 `decision_agent`

| Field | Value |
|---|---|
| **Model** | `claude-sonnet-4-6` (using Anthropic's tool-use; preferred over Gemini here for better tool calling and structured rationale). |
| **Role** | Compose a `TradeSignal` from forecast + sentiment + risk. Run a self-critique pass before emitting. |
| **Trigger** | (a) CLI: `python decide.py --country DE_LU --date YYYY-MM-DD`. (b) Backtest runner once per day in the test window. |
| **Inputs** | `(country, target_date)` only. The agent uses three tools to fetch the rest. |
| **Outputs** | `TradeSignal` Pydantic. Validated by `rationale_must_be_substantive`. |
| **Token budget** | ≤ 8k input / ≤ 1.5k output. Tool responses must be small (Pydantic-serialized, not raw frames). |
| **Tools allowed** | Exactly three: `get_forecast(country, date) → ForecastResult`, `get_sentiment(country, date) → SentimentScore`, `get_risk(country, signal_strength, vol_window) → RiskParameters`. No `bash`, no file I/O — every external effect is through one of these three. |
| **Handoff** | Signal is logged to SQLite (paper trade) or returned to backtest runner. Confidence < 0.6 or rationale missing any of the three inputs forces `HOLD`. |

### 1.5 `backtest_reporter`

| Field | Value |
|---|---|
| **Model** | `claude-sonnet-4-6` |
| **Role** | Run the 6-month walk-forward backtest, compute `BacktestSummary`, write `data/backtest/results.json`, and produce a one-page markdown writeup explaining wins, losses, and hypotheses for the next iteration. |
| **Trigger** | After Phase 4 ships and on demand for each strategy iteration. Also called by the scheduler weekly to refresh the dashboard. |
| **Inputs** | Date range, `data/processed/prices_history.parquet`, the trade log produced by `src/backtest/runner.py`. The agent receives the *already-computed* metrics as a small dict (≤ 1k tokens) — it does not re-run pandas. |
| **Outputs** | `BacktestSummary` (Pydantic), `data/backtest/results.json`, `docs/postmortem_<date>.md` if `passed_thresholds=False`. |
| **Token budget** | ≤ 10k input / ≤ 3k output (writeup is the longest output in the system). |
| **Tools allowed** | `bash` (run `python -m src.backtest.runner` and `python -m src.evaluation.eval_backtest`), `write_file` (results.json + postmortem only). |
| **Handoff** | Summary returns to orchestrator; orchestrator publishes to the Streamlit dashboard. |

---

## 2. Orchestrator

**Model:** `claude-opus-4-7` — reserved for planning and routing only. Opus never edits code, never scores articles, never runs models. Its job is to decide *which worker runs next, with what context*.

**Run loop (one full pipeline pass):**

```
1. ingestion_curator         → status JSON
2. if status.sources_failed → halt or warn
3. sentiment_analyst (eval)  → kappa
4. GATE: kappa ≥ 0.5 ? else stop, write postmortem
5. forecast_modeler (eval)   → directional_accuracy
6. GATE: dir_acc ≥ 55% AND beats baselines ? else stop
7. decision_agent (one day or full backtest range)
8. backtest_reporter         → BacktestSummary
9. GATE: sharpe ≥ 0.8 AND max_dd ≤ 0.08 ? else postmortem path
10. publish to Streamlit
```

**Context engineering rules the orchestrator MUST follow:**

- Pass each worker only the *minimum slice* it needs. Never forward another worker's full output unless the next worker's contract explicitly requires it.
- Strip large arrays before forwarding. The orchestrator turns "the parquet has 8760 rows" into a 6-line summary, not 8760 rows of context.
- Cache the system prompt of every worker the first time it runs in a session (`cache_control: {"type": "ephemeral"}`). The 5-minute TTL means a full pipeline pass benefits; isolated reruns do not.
- Never let two workers share a context window. Each handoff is a fresh call with shaped inputs.

**Failure protocol (fallback ladder):**

1. **Single tool error** → retry once with a clarifying note appended to the input.
2. **Worker raises** → orchestrator catches, logs to `data/eval/agent_errors.jsonl`, falls through to the next agent in the chain only if the failure is non-blocking (e.g., one ingest source down, others succeeded). Otherwise halt.
3. **Eval gate failure** → DO NOT proceed. Open a Polish-track diagnostic task ("why did κ slip?") and stop the pipeline. Eval failure is the loudest possible signal — reacting by retrying is overfitting.
4. **Budget exceeded** (`BudgetExceeded` from `src/utils/budget.py`) → halt every worker, surface the daily-cost summary, and require human resume.
5. **Self-critique loop divergence** in `decision_agent` → after one critique pass, force `HOLD`. No second critique. The point of self-critique is to catch a hallucinated rationale, not to converge to a confident answer.

**Eval-gates the orchestrator enforces (hard, not advisory):**

| Gate | Threshold | Source |
|---|---|---|
| Sentiment κ | ≥ 0.5 | `src/evaluation/eval_sentiment.py` |
| Forecast directional accuracy | ≥ 55% AND beats naive/seasonal/rolling | `src/evaluation/eval_forecast.py` |
| Backtest Sharpe + DD | Sharpe ≥ 0.8 AND max DD ≤ 8% | `src/evaluation/eval_backtest.py` |

If the gate fails, the orchestrator's job is to **stop**, not to debug. Debugging is a human action.

---

## 3. Token Optimization Rules

These are the eight concrete rules the orchestrator and workers follow. They map directly to Anthropic's prompt-caching and context-engineering guidance.

1. **Cache every system prompt.** Each worker's system prompt is stable across calls within a session. Set `cache_control: {"type": "ephemeral"}` on the system block. Cache hits are reported in `usage.cache_read_input_tokens` — log it. The TTL is 5 minutes; a single pipeline pass (~3 minutes end-to-end on the M-series target machine) sits well inside that window.
2. **Pass file paths, not file contents.** When the next agent only needs to *open* a parquet, send `path + schema + last_n_rows` (≤ 200 tokens), never the frame.
3. **Pre-summarise long evidence.** If an article body is > 1.2k tokens, the orchestrator summarises it down before forwarding. The sentiment_analyst sees title + summary, never the full body.
4. **Structured tool returns are mandatory.** A tool that returns more than ~500 tokens of free text is a design smell — make it return a Pydantic JSON object whose fields the next prompt directly references.
5. **Early-stop on eval failure.** If a gate fails, don't keep generating context. The pipeline halts, costing ~0 marginal tokens.
6. **Bound retries to 1 per call.** Every tool call retries at most once with the validation error appended; after that, fail loudly. Bounded retry beats blind retry every time.
7. **One worker, one model.** Don't ask Sonnet to do Haiku's job (article scoring) — the cost difference compounds across thousands of articles. Don't ask Opus to do Sonnet's job — you'll burn the daily budget on routing.
8. **Track against the budget.** Every Anthropic call is wrapped in `budget.track(label=..., model=...)` from `src/utils/budget.py`. `MAX_DAILY_LLM_USD=5` is the hard ceiling. The orchestrator queries `budget.daily_cost_usd` before every worker invocation and refuses to proceed if remaining headroom is < estimated cost of the next call.

---

## 4. Inter-Agent Contract (Pydantic)

All cross-agent values flow as instances of these classes (live in `src/agent/schema.py`). The orchestrator validates on every handoff; a `ValidationError` is treated as a worker failure.

| Class | Producer | Consumer | Purpose |
|---|---|---|---|
| `SentimentScore` | `sentiment_analyst` | `decision_agent` (via `get_sentiment` tool), forecast features | One LLM scoring of one article: `{score, confidence, reasoning}`. |
| `ForecastResult` | `forecast_modeler` | `decision_agent` (via `get_forecast` tool) | `{country, target_date, point_forecast_eur_mwh, std_eur_mwh, model_name, features_used[]}`. |
| `RiskParameters` | risk module (no LLM) wrapped by `decision_agent`'s `get_risk` tool | `decision_agent` | `{position_size_mwh, stop_price, target_price, max_loss_eur, realized_vol_annualized, kelly_fraction}`. |
| `TradeSignal` | `decision_agent` | backtest runner, paper-trade scheduler, Streamlit | `{country, target_date, direction, size_mwh, confidence, rationale, forecast_*, sentiment_score, realized_vol_annualized, sources[], timestamp}`. Validator: `rationale_must_be_substantive` (length ≥ 50). |
| `BacktestSummary` | `backtest_reporter` | orchestrator, dashboard | `{start_date, end_date, n_trades, sharpe_annualized, sortino_annualized, max_drawdown_pct, win_rate, total_pnl_eur, passed_thresholds}`. |

These classes already exist in `src/agent/schema.py`. **Do not duplicate them**. If a worker needs a new field, edit the schema, bump the prompt-version comment, and re-run all evals.

---

## 5. Failure Modes & Guardrails

| # | Failure mode | Guardrail | Where it lives |
|---|---|---|---|
| 1 | **API rate limit** (Anthropic or Gemini) | Exponential backoff capped at 3 retries; if all 3 fail, raise `BudgetExceeded`-style halt. Concurrency cap of 4 in-flight calls. | `src/utils/budget.py` (extend with rate-limit handling), worker call sites. |
| 2 | **Eval gate failure** | Orchestrator halts the pipeline on first failed gate; writes `docs/postmortem_<date>.md`. No automatic retry. | Orchestrator gate logic. |
| 3 | **Missing data** (parquet empty / news.jsonl stale) | `ingestion_curator` returns `sources_failed`; orchestrator refuses to run downstream agents that need that source. T1 must run first. | `ingestion_curator` + orchestrator. |
| 4 | **Self-critique infinite loop** in `decision_agent` | One critique pass only. Disagreement → force `HOLD` with rationale "self-critique flagged inconsistency". No second critique pass. | `src/agent/decide.py`. |
| 5 | **Budget overrun** | `budget.track` raises `BudgetExceeded` before the call begins if `daily_cost_usd ≥ daily_limit_usd`. Orchestrator pre-flights this check. | `src/utils/budget.py`. |

Bonus guardrail: **no autonomous order placement, ever.** This is non-negotiable per `docs/STRATEGY.md`. Signals are research output, not execution.

---

## 6. Pre-Flight Checklist (per agent)

### 6.1 ingestion_curator

```
TASK:           Refresh raw price + news data into the project cache.
WHY:            Every downstream agent needs current data; staleness silently corrupts evals.
SUCCESS METRIC: prices_history.parquet rows ≥ 365 days × 4 countries; news.jsonl ≥ 50 fresh items in last 24h.
DATASET:        ENTSO-E day-ahead prices, RSS feeds in src/ingestion/rss_scraper.py, PDFs.
BASELINE:       Manual `python -m src.ingestion.price_history` + `python ingest.py`.
COST BUDGET:    $0 (no LLM calls in this agent).
LATENCY BUDGET: ≤ 90s p95.
CONTEXT INPUTS: env credentials, last-success timestamp, list of sources to refresh.
OUTPUT SCHEMA:  {rows_added: int, sources_failed: list[str], duration_s: float}
EVAL CADENCE:   Daily (scheduler) + before each pipeline pass.
ROLLBACK PLAN:  Last-good parquet snapshot in data/processed/.bak/.
```

### 6.2 sentiment_analyst

```
TASK:           Score one article for next-day price impact OR run κ-eval over the gold set.
WHY:            Sentiment is the differentiating signal. If κ < 0.5 the entire forecast→agent chain is built on noise.
SUCCESS METRIC: Cohen's κ ≥ 0.5 (linear-weighted) on data/eval/sentiment_gold.jsonl.
DATASET:        30 hand-labeled articles, balanced ≥ 8 per class. Owner: human (T0).
BASELINE:       Class-prior majority vote (κ = 0 by definition).
COST BUDGET:    ≤ $0.001 per article via Haiku; ≤ $0.05 per full eval pass.
LATENCY BUDGET: ≤ 2s p95 per article.
CONTEXT INPUTS: System prompt (cached), title, summary.
OUTPUT SCHEMA:  SentimentScore (src/agent/schema.py).
EVAL CADENCE:   On every prompt-version bump; weekly during dev; before every release.
ROLLBACK PLAN:  Pin previous prompt version in src/sentiment/scorer.py and re-run eval.
```

### 6.3 forecast_modeler

```
TASK:           Produce next-day price forecast and prove it beats baselines.
WHY:            Directional accuracy is the trade-P&L leading indicator.
SUCCESS METRIC: Walk-forward directional accuracy ≥ 55% AND beats naive/seasonal/rolling on the same metric.
DATASET:        data/processed/prices_history.parquet, ≥ 1y; data/processed/sentiment_daily.parquet (P2.4 only).
BASELINE:       Three baselines already in src/forecast/baseline.py.
COST BUDGET:    ≤ $0.10 per eval pass (most work is local sklearn/xgboost, not LLM).
LATENCY BUDGET: Eval ≤ 60s; per-day forecast call ≤ 5s.
CONTEXT INPUTS: Parquet path, schema, last-90-day summary, baseline metrics.
OUTPUT SCHEMA:  ForecastResult.
EVAL CADENCE:   On every model change; before unlocking Phase 4.
ROLLBACK PLAN:  Pinned best-model artifact + previous eval row in data/eval/eval_log.jsonl.
```

### 6.4 decision_agent

```
TASK:           Compose forecast + sentiment + risk into a TradeSignal with self-critique.
WHY:            This is the agent the user actually consumes; everything upstream exists to feed it.
SUCCESS METRIC: 100% of emitted signals pass `rationale_must_be_substantive`; 0 hallucinated source IDs.
DATASET:        Live tools — no static eval set. Tested via backtest (north-star).
BASELINE:       Rule-based: BUY if forecast_change > 1σ AND sentiment > 0.33 AND vol low; else HOLD.
COST BUDGET:    ≤ $0.02 per signal via Sonnet with caching.
LATENCY BUDGET: ≤ 8s p95 (inside MAX_QUERY_LATENCY_S=10).
CONTEXT INPUTS: country, target_date, three tool definitions, cached system prompt.
OUTPUT SCHEMA:  TradeSignal.
EVAL CADENCE:   Every backtest run (north-star).
ROLLBACK PLAN:  Previous prompt version pinned in src/agent/decide.py; rule-based fallback always available.
```

### 6.5 backtest_reporter

```
TASK:           Run 6-month walk-forward backtest and write the writeup.
WHY:            North-star metric. Without this number, no claim about the system is defensible.
SUCCESS METRIC: Sharpe ≥ 0.8 AND max DD ≤ 8% on data/backtest/results.json.
DATASET:        Last 6 months of prices_history.parquet + the live decision_agent.
BASELINE:       Buy-and-hold DE_LU (compute Sharpe of that — it's the floor).
COST BUDGET:    ≤ $0.50 per full backtest run (~180 decision_agent calls × $0.002).
LATENCY BUDGET: ≤ 15 minutes for a full 6-month run.
CONTEXT INPUTS: Date range, computed metrics dict, top-10 winning + top-10 losing trades.
OUTPUT SCHEMA:  BacktestSummary + optional postmortem markdown.
EVAL CADENCE:   On every Phase 4 prompt change; weekly during paper-trade phase.
ROLLBACK PLAN:  Previous results.json kept as data/backtest/results_<date>.json.
```

---

## 7. Operating Notes

- **Calling sub-agents from Claude Code CLI:** the orchestrator invokes each worker through the `Agent` tool with `model` set explicitly to the value in §1. Never let the CLI auto-pick the model.
- **Prompts are versioned.** Every worker has a `PROMPT_VERSION` constant in its source file (see `src/sentiment/scorer.py` for the pattern). Bumping the version and re-running the relevant eval is one commit.
- **The Streamlit dashboard is the human-review surface.** Every signal, every backtest summary, every eval row is visible there. If it's not visible to a human, it doesn't count as shipped.
- **No autonomous trading. Ever.** The system surfaces signals; humans place orders.
