"""XGBoost forecaster — Phase 2.3 / 2.4.

Status: skeleton. Implement after ARIMA proves insufficient.

Two variants land here:
  - xgb_price_only      — lag features + calendar
  - xgb_with_sentiment  — adds sentiment_t-1, n_articles_t-1

Sentiment-as-feature acceptance test (docs/EVAL.md):
  Sentiment must add ≥ 2 percentage points of directional accuracy vs.
  the price-only model. If it doesn't, sentiment stays as analyst context.
"""

from datetime import date

from src.agent.schema import ForecastResult


def xgb_forecast(country: str, target_date: date, with_sentiment: bool = False) -> ForecastResult:
    raise NotImplementedError(
        "P2.3/P2.4 not yet implemented. See docs/EVAL.md for acceptance criteria."
    )
