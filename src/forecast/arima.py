"""ARIMA forecaster — Phase 2.2.

Status: skeleton. Implement when P2.1 (baseline) eval is published in
data/eval/eval_log.jsonl. Do not implement before then — AGENTS.md rule #3.

When you implement:
  - Use statsmodels.tsa.arima.model.ARIMA
  - Auto-select (p, d, q) on rolling 180-day train window
  - Walk-forward eval as defined in docs/EVAL.md
  - Must beat naive on MAE by ≥ 5% to ship
"""

from datetime import date

from src.agent.schema import ForecastResult


def arima_forecast(country: str, target_date: date) -> ForecastResult:
    raise NotImplementedError(
        "P2.2 not yet implemented. See docs/EVAL.md for acceptance criteria."
    )
