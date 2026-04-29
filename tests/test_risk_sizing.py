"""Tests for position sizing and Kelly fraction."""

import pytest

from src.agent.schema import TradeDirection
from src.risk.sizing import KELLY_CAP, kelly_fraction, size_position


def test_kelly_fraction_caps_at_quarter():
    # Strong edge, low vol — should cap at KELLY_CAP
    assert kelly_fraction(edge=0.5, vol_annualized=0.2) == KELLY_CAP


def test_kelly_fraction_zero_for_zero_edge():
    assert kelly_fraction(edge=0.0, vol_annualized=0.3) == 0.0


def test_kelly_fraction_zero_for_zero_vol():
    assert kelly_fraction(edge=0.05, vol_annualized=0.0) == 0.0


def test_kelly_fraction_negative_edge_returns_zero():
    assert kelly_fraction(edge=-0.05, vol_annualized=0.3) == 0.0


def test_size_position_hold_returns_zero_size():
    r = size_position(
        direction=TradeDirection.HOLD,
        confidence=0.9,
        expected_return_pct=0.05,
        realized_vol_annualized=0.3,
        account_size_eur=10000.0,
        current_price_eur_mwh=80.0,
    )
    assert r.position_size_mwh == 0.0
    assert r.kelly_fraction == 0.0


def test_size_position_buy_produces_positive_size():
    r = size_position(
        direction=TradeDirection.BUY,
        confidence=0.7,
        expected_return_pct=0.05,
        realized_vol_annualized=0.3,
        account_size_eur=10000.0,
        current_price_eur_mwh=80.0,
    )
    assert r.position_size_mwh > 0
    assert r.target_price_eur_mwh > 80.0
    assert r.stop_price_eur_mwh < 80.0
    assert r.max_loss_eur >= 0


def test_size_position_sell_inverts_stop_target():
    r = size_position(
        direction=TradeDirection.SELL,
        confidence=0.7,
        expected_return_pct=0.05,
        realized_vol_annualized=0.3,
        account_size_eur=10000.0,
        current_price_eur_mwh=80.0,
    )
    assert r.target_price_eur_mwh < 80.0
    assert r.stop_price_eur_mwh > 80.0


def test_size_position_zero_confidence_returns_zero():
    r = size_position(
        direction=TradeDirection.BUY,
        confidence=0.0,
        expected_return_pct=0.05,
        realized_vol_annualized=0.3,
        account_size_eur=10000.0,
        current_price_eur_mwh=80.0,
    )
    assert r.position_size_mwh == 0.0
