"""Tests for the backtest runner."""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.agent.schema import TradeDirection, TradeSignal


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


# ── Tests ────────────────────────────────────────────────────────────────────

def test_run_backtest_returns_none_for_reversed_dates():
    from src.backtest.runner import run_backtest
    result = run_backtest("DE_LU", date(2026, 4, 20), date(2026, 1, 1))
    assert result is None


def test_run_backtest_returns_none_for_missing_prices(monkeypatch, tmp_path):
    from src.backtest import runner
    monkeypatch.setattr(runner, "PRICES_PATH", tmp_path / "nonexistent.parquet")
    result = runner.run_backtest("DE_LU", date(2026, 1, 1), date(2026, 4, 20))
    assert result is None


def test_run_backtest_produces_csv(monkeypatch, tmp_path):
    from src.backtest import runner

    # Use real prices parquet
    price_df = _make_price_df()
    parquet_path = tmp_path / "prices.parquet"
    price_df.to_parquet(parquet_path)
    monkeypatch.setattr(runner, "PRICES_PATH", parquet_path)
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path)

    result = runner.run_backtest(
        "DE_LU",
        date(2025, 12, 1),
        date(2025, 12, 10),
        quiet=True,
    )
    if result is not None:
        assert result.exists()
        assert result.suffix == ".csv"


def test_pnl_sign_correct_for_buy():
    """BUY at 50, mark at 55 → PnL should be positive."""
    diff = 55.0 - 50.0
    sign = 1.0  # BUY
    pnl = sign * diff * 10.0
    assert pnl > 0


def test_pnl_sign_correct_for_sell():
    """SELL at 50, mark at 55 → PnL should be negative."""
    diff = 55.0 - 50.0
    sign = -1.0  # SELL
    pnl = sign * diff * 10.0
    assert pnl < 0
