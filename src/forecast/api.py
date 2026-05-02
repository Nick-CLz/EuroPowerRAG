"""Forecast API — Phase 2.5.

Provides a single entry-point for the rest of the system to obtain
a next-day price forecast.  The cascade order is:

    XGBoost → ARIMA → rolling-mean baseline

Each level is tried in turn; the first one that succeeds wins.
"""

from __future__ import annotations

import logging
from datetime import date

from src.agent.schema import ForecastResult

log = logging.getLogger(__name__)

__all__ = ["forecast_next_day"]

# Import order matches cascade priority
from src.forecast.xgb import xgb_forecast          # noqa: E402
from src.forecast.arima import arima_forecast       # noqa: E402
from src.forecast.baseline import rolling_mean_forecast  # noqa: E402


def forecast_next_day(country: str, target_date: date) -> ForecastResult:
    """Return the best available next-day forecast for *country*.

    Cascade:
        1. XGBoost (price-only features)
        2. ARIMA(1,1,1)
        3. Rolling-mean baseline (always succeeds if data exists)

    Args:
        country:     Zone code (e.g. ``DE_LU``).
        target_date: The date to forecast.

    Returns:
        A ``ForecastResult`` from whichever model succeeded first.

    Raises:
        Exception: only if *all* models fail (including the baseline).
    """
    cascade: list[tuple[str, callable]] = [
        ("xgb", lambda: xgb_forecast(country, target_date, with_sentiment=False)),
        ("arima", lambda: arima_forecast(country, target_date)),
        ("rolling_mean", lambda: rolling_mean_forecast(country, target_date)),
    ]

    last_error: Exception | None = None
    for name, fn in cascade:
        try:
            result = fn()
            log.debug("forecast_next_day: using %s for %s %s", name, country, target_date)
            return result
        except Exception as exc:
            log.info("forecast_next_day: %s failed for %s %s — %s", name, country, target_date, exc)
            last_error = exc

    # Should be unreachable unless even the baseline can't find data
    raise RuntimeError(
        f"All forecast models failed for {country} on {target_date}"
    ) from last_error
