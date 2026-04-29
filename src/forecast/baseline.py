"""Baseline forecasters — the floor every smarter model must beat.

P2.1: Ship the dumbest version first (AGENTS.md rule #3).

Three baselines, all working with no training:
  - naive:          y_hat(t+1) = y(t)
  - seasonal_naive: y_hat(t+1) = y(t-7)
  - rolling_mean:   y_hat(t+1) = mean(y[t-6:t+1])
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.agent.schema import ForecastResult

PRICES_PATH = Path("data/processed/prices_history.parquet")


def _load_prices() -> pd.DataFrame:
    if not PRICES_PATH.exists():
        raise FileNotFoundError(
            f"{PRICES_PATH} not found — run `python -m src.ingestion.price_history` first."
        )
    df = pd.read_parquet(PRICES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["country", "date"])


def naive_forecast(country: str, target_date: date) -> ForecastResult:
    df = _load_prices()
    df = df[df["country"] == country]
    cutoff = pd.Timestamp(target_date) - timedelta(days=1)
    last = df[df["date"] <= cutoff].tail(1)
    if last.empty:
        raise ValueError(f"No prices for {country} on or before {cutoff.date()}")

    last_price = float(last["price_eur_mwh"].iloc[0])
    return ForecastResult(
        country=country,
        target_date=target_date,
        point_forecast_eur_mwh=last_price,
        std_eur_mwh=0.0,  # Naive has no variance estimate
        model_name="naive",
        features_used=["last_price"],
    )


def seasonal_naive_forecast(country: str, target_date: date) -> ForecastResult:
    df = _load_prices()
    df = df[df["country"] == country]
    target_minus_7 = pd.Timestamp(target_date) - timedelta(days=7)
    match = df[df["date"] == target_minus_7]
    if match.empty:
        raise ValueError(f"No price for {country} on {target_minus_7.date()}")

    return ForecastResult(
        country=country,
        target_date=target_date,
        point_forecast_eur_mwh=float(match["price_eur_mwh"].iloc[0]),
        std_eur_mwh=0.0,
        model_name="seasonal_naive",
        features_used=["lag_7"],
    )


def rolling_mean_forecast(country: str, target_date: date, window: int = 7) -> ForecastResult:
    df = _load_prices()
    df = df[df["country"] == country]
    cutoff = pd.Timestamp(target_date) - timedelta(days=1)
    recent = df[df["date"] <= cutoff].tail(window)
    if len(recent) < window:
        raise ValueError(f"Need {window} prior days; have {len(recent)}")

    return ForecastResult(
        country=country,
        target_date=target_date,
        point_forecast_eur_mwh=float(recent["price_eur_mwh"].mean()),
        std_eur_mwh=float(recent["price_eur_mwh"].std()),
        model_name="rolling_mean",
        features_used=[f"rolling_{window}"],
    )
