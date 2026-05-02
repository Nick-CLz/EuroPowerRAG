"""Tests for forecast models (arima, xgb, api, loader)."""

from datetime import date
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.agent.schema import ForecastResult


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_price_df(n_days: int = 200, country: str = "DE_LU") -> pd.DataFrame:
    """Build a synthetic price DataFrame matching the parquet schema."""
    rng = np.random.default_rng(99)
    dates = pd.date_range(end="2026-04-20", periods=n_days)
    base = 60.0
    prices = base + rng.normal(0, 5, n_days).cumsum()
    return pd.DataFrame({
        "date": dates,
        "country": country,
        "price_eur_mwh": prices,
    })


@pytest.fixture(autouse=True)
def _patch_loader(monkeypatch):
    """Patch load_prices globally so tests never touch disk."""
    df = _make_price_df()
    monkeypatch.setattr("src.forecast.loader._cache", df)


# ── loader ───────────────────────────────────────────────────────────────────

def test_load_prices_returns_sorted_df():
    from src.forecast.loader import load_prices
    df = load_prices()
    assert "date" in df.columns
    assert "country" in df.columns
    assert "price_eur_mwh" in df.columns


# ── ARIMA ────────────────────────────────────────────────────────────────────

def test_arima_returns_forecast_result():
    from src.forecast.arima import arima_forecast
    result = arima_forecast("DE_LU", date(2026, 4, 20))
    assert isinstance(result, ForecastResult)
    assert result.model_name == "arima"
    assert result.country == "DE_LU"
    assert result.point_forecast_eur_mwh != 0


def test_arima_raises_on_insufficient_data(monkeypatch):
    from src.forecast.arima import arima_forecast
    short_df = _make_price_df(n_days=10)
    monkeypatch.setattr("src.forecast.loader._cache", short_df)
    with pytest.raises(ValueError, match="Need ≥"):
        arima_forecast("DE_LU", date(2026, 4, 20))


# ── XGBoost ──────────────────────────────────────────────────────────────────

def test_xgb_returns_forecast_result():
    from src.forecast.xgb import xgb_forecast
    result = xgb_forecast("DE_LU", date(2026, 4, 20), with_sentiment=False)
    assert isinstance(result, ForecastResult)
    assert result.model_name == "xgb"


def test_xgb_sentiment_variant():
    from src.forecast.xgb import xgb_forecast
    result = xgb_forecast("DE_LU", date(2026, 4, 20), with_sentiment=True)
    assert result.model_name == "xgb_sentiment"
    assert "sentiment_t_1" in result.features_used


def test_xgb_raises_on_insufficient_data(monkeypatch):
    from src.forecast.xgb import xgb_forecast
    short_df = _make_price_df(n_days=10)
    monkeypatch.setattr("src.forecast.loader._cache", short_df)
    with pytest.raises(ValueError, match="Need ≥"):
        xgb_forecast("DE_LU", date(2026, 4, 20))


def test_xgb_produces_positive_std():
    from src.forecast.xgb import xgb_forecast
    result = xgb_forecast("DE_LU", date(2026, 4, 20))
    assert result.std_eur_mwh > 0


# ── Forecast API ─────────────────────────────────────────────────────────────

def test_forecast_api_returns_result():
    from src.forecast.api import forecast_next_day
    result = forecast_next_day("DE_LU", date(2026, 4, 20))
    assert isinstance(result, ForecastResult)


def test_forecast_api_cascade_falls_through(monkeypatch):
    """If XGB fails, should fall through to ARIMA or baseline."""
    from src.forecast import api

    def _broken_xgb(*a, **kw):
        raise RuntimeError("XGB broken")

    monkeypatch.setattr("src.forecast.api.xgb_forecast", _broken_xgb)
    result = api.forecast_next_day("DE_LU", date(2026, 4, 20))
    assert isinstance(result, ForecastResult)
    # Should have used arima or rolling_mean
    assert result.model_name in ("arima", "rolling_mean")
