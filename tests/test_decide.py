"""Tests for the decision agent (decide.py)."""

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.agent.schema import ForecastResult, TradeDirection, TradeSignal


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_price_df(n_days: int = 200, country: str = "DE_LU") -> pd.DataFrame:
    rng = np.random.default_rng(99)
    dates = pd.date_range(end="2026-04-20", periods=n_days)
    base = 60.0
    prices = base + rng.normal(0, 5, n_days).cumsum()
    return pd.DataFrame({"date": dates, "country": country, "price_eur_mwh": prices})


@pytest.fixture(autouse=True)
def _patch_loader(monkeypatch):
    df = _make_price_df()
    monkeypatch.setattr("src.forecast.loader._cache", df)


# ── decide ───────────────────────────────────────────────────────────────────

def test_decide_returns_trade_signal():
    from src.agent.decide import decide
    signal = decide("DE_LU", date(2026, 4, 20))
    assert isinstance(signal, TradeSignal)
    assert signal.country == "DE_LU"


def test_decide_falls_back_to_heuristic_without_claude():
    """When ANTHROPIC_API_KEY is unset, decide() should still return a signal."""
    from src.agent.decide import decide
    signal = decide("DE_LU", date(2026, 4, 20))
    assert isinstance(signal, TradeSignal)
    # Should mention "Heuristic" or "critique" in rationale
    assert "euristic" in signal.rationale or "critique" in signal.rationale.lower()


# ── _get_sentiment ───────────────────────────────────────────────────────────

def test_get_sentiment_returns_zero_when_no_file():
    from src.agent.decide import _get_sentiment
    score = _get_sentiment("DE_LU", date(2026, 4, 20))
    assert score == 0.0


# ── _get_volatility ─────────────────────────────────────────────────────────

def test_get_volatility_returns_float():
    from src.agent.decide import _get_volatility
    vol = _get_volatility("DE_LU")
    assert isinstance(vol, float)
    assert vol > 0


# ── _heuristic_decide ────────────────────────────────────────────────────────

def test_heuristic_buy_on_cheap_price():
    from src.agent.decide import _heuristic_decide
    forecast = ForecastResult(
        country="DE_LU", target_date=date(2026, 4, 20),
        point_forecast_eur_mwh=30.0, std_eur_mwh=5.0,
        model_name="test", features_used=[]
    )
    direction, size, conf, rationale = _heuristic_decide(forecast, sentiment=0.0, vol=0.3)
    assert direction == TradeDirection.BUY
    assert size > 0


def test_heuristic_sell_on_expensive_price():
    from src.agent.decide import _heuristic_decide
    forecast = ForecastResult(
        country="DE_LU", target_date=date(2026, 4, 20),
        point_forecast_eur_mwh=100.0, std_eur_mwh=5.0,
        model_name="test", features_used=[]
    )
    direction, size, conf, rationale = _heuristic_decide(forecast, sentiment=0.0, vol=0.3)
    assert direction == TradeDirection.SELL


def test_heuristic_hold_on_mid_price():
    from src.agent.decide import _heuristic_decide
    forecast = ForecastResult(
        country="DE_LU", target_date=date(2026, 4, 20),
        point_forecast_eur_mwh=65.0, std_eur_mwh=5.0,
        model_name="test", features_used=[]
    )
    direction, size, conf, rationale = _heuristic_decide(forecast, sentiment=0.0, vol=0.3)
    assert direction == TradeDirection.HOLD
    assert size == 0.0


# ── _self_critique ───────────────────────────────────────────────────────────

def test_self_critique_overrides_low_confidence():
    from src.agent.decide import _self_critique
    direction, size, conf, rationale = _self_critique(
        TradeDirection.BUY, 10.0, 0.3,
        "forecast=50 EUR/MWh, sentiment=+0.1, vol=30%. Cheap price."
    )
    assert direction == TradeDirection.HOLD
    assert size == 0.0


def test_self_critique_passes_good_signal():
    from src.agent.decide import _self_critique
    direction, size, conf, rationale = _self_critique(
        TradeDirection.BUY, 10.0, 0.8,
        "forecast=45 EUR/MWh is below threshold, sentiment=+0.2 supportive, vol=25% moderate."
    )
    assert direction == TradeDirection.BUY
    assert size == 10.0
