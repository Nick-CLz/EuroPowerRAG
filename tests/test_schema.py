"""Tests for Pydantic schemas in src/agent/schema.py."""

from datetime import date

import pytest
from pydantic import ValidationError

from src.agent.schema import (
    BacktestSummary,
    ForecastResult,
    RiskParameters,
    SentimentScore,
    TradeDirection,
    TradeSignal,
)


# ── SentimentScore ───────────────────────────────────────────────────────────

def test_sentiment_score_valid():
    s = SentimentScore(score=0.5, confidence=0.8, reasoning="Capacity outage cited.")
    assert s.score == 0.5


def test_sentiment_score_out_of_bounds():
    with pytest.raises(ValidationError):
        SentimentScore(score=2.0, confidence=0.5, reasoning="x")


def test_sentiment_score_confidence_bounds():
    with pytest.raises(ValidationError):
        SentimentScore(score=0.0, confidence=1.5, reasoning="x")


# ── ForecastResult ───────────────────────────────────────────────────────────

def test_forecast_result_valid():
    f = ForecastResult(
        country="DE_LU",
        target_date=date(2026, 5, 1),
        point_forecast_eur_mwh=85.0,
        std_eur_mwh=10.0,
        model_name="naive",
    )
    assert f.country == "DE_LU"
    assert f.features_used == []


def test_forecast_negative_std_rejected():
    with pytest.raises(ValidationError):
        ForecastResult(
            country="DE_LU",
            target_date=date(2026, 5, 1),
            point_forecast_eur_mwh=85.0,
            std_eur_mwh=-1.0,
            model_name="naive",
        )


# ── RiskParameters ───────────────────────────────────────────────────────────

def test_risk_parameters_valid():
    r = RiskParameters(
        position_size_mwh=10.0,
        stop_price_eur_mwh=80.0,
        target_price_eur_mwh=100.0,
        max_loss_eur=200.0,
        realized_vol_annualized=0.3,
        kelly_fraction=0.1,
    )
    assert r.kelly_fraction == 0.1


def test_kelly_fraction_capped():
    with pytest.raises(ValidationError):
        RiskParameters(
            position_size_mwh=10.0,
            stop_price_eur_mwh=80.0,
            target_price_eur_mwh=100.0,
            max_loss_eur=200.0,
            realized_vol_annualized=0.3,
            kelly_fraction=1.5,  # > 1.0
        )


# ── TradeSignal ──────────────────────────────────────────────────────────────

def _signal_kwargs(**overrides) -> dict:
    base = dict(
        country="DE_LU",
        target_date=date(2026, 5, 1),
        direction=TradeDirection.BUY,
        size_mwh=5.0,
        confidence=0.7,
        rationale=(
            "Forecast 95 EUR/MWh vs spot 85, sentiment 0.4 bullish on outage news, "
            "vol 0.3 supports position."
        ),
        forecast_price_eur_mwh=95.0,
        forecast_std_eur_mwh=8.0,
        sentiment_score=0.4,
        realized_vol_annualized=0.3,
    )
    base.update(overrides)
    return base


def test_trade_signal_valid():
    sig = TradeSignal(**_signal_kwargs())
    assert sig.direction == TradeDirection.BUY


def test_trade_signal_short_rationale_rejected():
    with pytest.raises(ValidationError):
        TradeSignal(**_signal_kwargs(rationale="too short"))


def test_trade_signal_invalid_sentiment():
    with pytest.raises(ValidationError):
        TradeSignal(**_signal_kwargs(sentiment_score=1.5))


# ── BacktestSummary ──────────────────────────────────────────────────────────

def test_backtest_summary_valid():
    s = BacktestSummary(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 7, 1),
        n_trades=30,
        sharpe_annualized=1.1,
        sortino_annualized=1.5,
        max_drawdown_pct=0.05,
        win_rate=0.6,
        total_pnl_eur=5000.0,
        passed_thresholds=True,
    )
    assert s.passed_thresholds


def test_drawdown_bounded():
    with pytest.raises(ValidationError):
        BacktestSummary(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 7, 1),
            n_trades=1,
            sharpe_annualized=0.0,
            sortino_annualized=0.0,
            max_drawdown_pct=1.5,  # > 1.0 invalid
            win_rate=0.5,
            total_pnl_eur=0.0,
            passed_thresholds=False,
        )
