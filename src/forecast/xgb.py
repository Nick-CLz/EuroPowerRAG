"""XGBoost forecaster — Phase 2.3 / 2.4.

Two variants:
  - ``xgb_forecast(…, with_sentiment=False)``  — price-only lag features + calendar
  - ``xgb_forecast(…, with_sentiment=True)``   — adds ``sentiment_t-1``, ``n_articles_t-1``

Acceptance criteria (``docs/EVAL.md``):
  - Price-only must beat ARIMA on directional accuracy.
  - Sentiment variant must add ≥ 2 pp of directional accuracy vs price-only,
    otherwise sentiment stays as analyst context only.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.agent.schema import ForecastResult
from src.forecast.loader import load_prices

log = logging.getLogger(__name__)

__all__ = ["xgb_forecast"]

# ── Configuration ────────────────────────────────────────────────────────────
TRAIN_WINDOW = 180
MIN_TRAIN_OBS = 30
SENTIMENT_PATH = Path("data/processed/sentiment_daily.parquet")

_PRICE_FEATURES = ["lag_1", "lag_7", "rolling_7_mean", "rolling_7_std", "day_of_week", "month"]
_SENTIMENT_FEATURES = ["sentiment_t_1", "n_articles_t_1"]


# ── Feature engineering ──────────────────────────────────────────────────────

def _build_features(
    df: pd.DataFrame,
    with_sentiment: bool = False,
) -> pd.DataFrame:
    """Add lag / calendar / sentiment columns. Returns only valid rows."""
    out = df.copy()
    out["lag_1"] = out["price_eur_mwh"].shift(1)
    out["lag_7"] = out["price_eur_mwh"].shift(7)
    out["rolling_7_mean"] = out["lag_1"].rolling(7).mean()
    out["rolling_7_std"] = out["lag_1"].rolling(7).std()
    out["day_of_week"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month

    if with_sentiment:
        out = _merge_sentiment(out)

    return out.dropna(subset=_PRICE_FEATURES)


def _merge_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join the daily sentiment aggregates (lagged by 1 day)."""
    if not SENTIMENT_PATH.exists():
        log.debug("No sentiment data at %s — filling zeros", SENTIMENT_PATH)
        df["sentiment_t_1"] = 0.0
        df["n_articles_t_1"] = 0
        return df

    sent = pd.read_parquet(SENTIMENT_PATH)
    sent["date"] = pd.to_datetime(sent["date"])
    sent = sent.rename(columns={
        "mean_weighted_score": "sentiment_t_1",
        "n_articles": "n_articles_t_1",
    })
    # Shift sentiment by one day — we only know yesterday's news
    sent["date"] = sent["date"] + pd.Timedelta(days=1)

    df = df.merge(
        sent[["date", "country", "sentiment_t_1", "n_articles_t_1"]],
        on=["date", "country"],
        how="left",
    )
    df["sentiment_t_1"] = df["sentiment_t_1"].fillna(0.0)
    df["n_articles_t_1"] = df["n_articles_t_1"].fillna(0)
    return df


# ── Public API ───────────────────────────────────────────────────────────────

def xgb_forecast(
    country: str,
    target_date: date,
    with_sentiment: bool = False,
) -> ForecastResult:
    """Produce a 1-step-ahead XGBoost forecast for *target_date*.

    Args:
        country:        Zone code (e.g. ``DE_LU``).
        target_date:    The date being forecast.
        with_sentiment: Whether to include sentiment lag features.

    Returns:
        A populated ``ForecastResult``.

    Raises:
        ValueError: if there is insufficient history.
    """
    df = load_prices()
    df = df[df["country"] == country].copy()

    cutoff = pd.Timestamp(target_date) - timedelta(days=1)

    # Append a row for target_date so lag features can be computed for it
    placeholder = pd.DataFrame([{
        "date": pd.Timestamp(target_date),
        "country": country,
        "price_eur_mwh": np.nan,
    }])
    df = pd.concat([df[df["date"] <= cutoff], placeholder], ignore_index=True)

    df = _build_features(df, with_sentiment=with_sentiment)

    train = df[df["date"] <= cutoff].tail(TRAIN_WINDOW)
    if len(train) < MIN_TRAIN_OBS:
        raise ValueError(
            f"Need ≥ {MIN_TRAIN_OBS} observations for {country}; have {len(train)}"
        )

    features = _PRICE_FEATURES + (_SENTIMENT_FEATURES if with_sentiment else [])
    X_train = train[features].values
    y_train = train["price_eur_mwh"].values

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # Predict on the latest row (target_date)
    X_pred = df.tail(1)[features].values
    point = float(model.predict(X_pred)[0])

    # Use residual std on training data as rough uncertainty
    y_hat_train = model.predict(X_train)
    residual_std = float(np.std(y_train - y_hat_train, ddof=1))

    model_name = "xgb_sentiment" if with_sentiment else "xgb"
    log.debug("%s forecast for %s %s: %.2f EUR/MWh", model_name, country, target_date, point)

    return ForecastResult(
        country=country,
        target_date=target_date,
        point_forecast_eur_mwh=point,
        std_eur_mwh=residual_std,
        model_name=model_name,
        features_used=features,
    )
