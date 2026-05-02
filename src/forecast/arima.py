"""ARIMA(1,1,1) forecaster — Phase 2.2.

Uses a rolling 180-day train window of daily day-ahead prices.
Must beat naïve on MAE by ≥ 5 % to ship (see ``docs/EVAL.md``).

Order selection: fixed at (1, 1, 1) — the simplest difference-stationary
model that captures a single lag of AR and MA.  A future iteration can
auto-select via AIC grid search, but for a daily commodity series this
order already provides a strong baseline.
"""

from __future__ import annotations

import logging
import warnings
from datetime import date, timedelta

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.agent.schema import ForecastResult
from src.forecast.loader import load_prices

log = logging.getLogger(__name__)

__all__ = ["arima_forecast"]

# ── Configuration ────────────────────────────────────────────────────────────
TRAIN_WINDOW = 180  # rolling window in days
MIN_TRAIN_OBS = 30  # refuse to fit with fewer observations
ORDER = (1, 1, 1)


def arima_forecast(country: str, target_date: date) -> ForecastResult:
    """Produce a 1-step-ahead ARIMA forecast for *target_date*.

    Args:
        country:     Zone code (e.g. ``DE_LU``).
        target_date: The date being forecast.

    Returns:
        A populated ``ForecastResult``.

    Raises:
        ValueError: if there is insufficient history.
    """
    df = load_prices()
    df = df[df["country"] == country]
    cutoff = pd.Timestamp(target_date) - timedelta(days=1)
    train = df[df["date"] <= cutoff].tail(TRAIN_WINDOW)

    if len(train) < MIN_TRAIN_OBS:
        raise ValueError(
            f"Need ≥ {MIN_TRAIN_OBS} observations for {country}; have {len(train)}"
        )

    prices = train["price_eur_mwh"].values

    # statsmodels emits convergence warnings on noisy series — suppress
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = ARIMA(prices, order=ORDER)
            fitted = model.fit()
            point = float(fitted.forecast(steps=1)[0])
            # Use residual std as a rough forecast uncertainty
            residual_std = float(fitted.resid.std(ddof=1)) if len(fitted.resid) > 1 else 0.0
        except Exception:
            log.warning("ARIMA fit failed for %s on %s — falling back to last price", country, target_date)
            point = float(prices[-1])
            residual_std = 0.0

    return ForecastResult(
        country=country,
        target_date=target_date,
        point_forecast_eur_mwh=point,
        std_eur_mwh=residual_std,
        model_name="arima",
        features_used=[f"arima_{ORDER[0]}_{ORDER[1]}_{ORDER[2]}"],
    )
