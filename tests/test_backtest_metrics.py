"""Tests for backtest performance metrics."""

import math

import pytest

from src.backtest.metrics import max_drawdown, sharpe_ratio, sortino_ratio, win_rate


def test_sharpe_zero_for_constant_returns():
    assert sharpe_ratio([0.01, 0.01, 0.01, 0.01]) == 0.0


def test_sharpe_positive_for_positive_drift():
    s = sharpe_ratio([0.005, 0.01, 0.008, 0.012, 0.006])
    assert s > 0


def test_sharpe_handles_empty():
    assert sharpe_ratio([]) == 0.0


def test_sortino_zero_when_no_downside():
    assert sortino_ratio([0.01, 0.02, 0.005]) == 0.0


def test_sortino_handles_empty():
    assert sortino_ratio([]) == 0.0


def test_max_drawdown_no_loss_is_zero():
    assert max_drawdown([100, 110, 120, 130]) == 0.0


def test_max_drawdown_basic():
    # Peak 120, trough 90 → 25% drawdown
    dd = max_drawdown([100, 120, 90, 110])
    assert math.isclose(dd, 0.25, abs_tol=1e-6)


def test_max_drawdown_handles_empty():
    assert max_drawdown([]) == 0.0


def test_win_rate_basic():
    assert win_rate([1.0, -0.5, 2.0, -1.0]) == 0.5


def test_win_rate_all_winners():
    assert win_rate([1.0, 2.0, 3.0]) == 1.0


def test_win_rate_handles_empty():
    assert win_rate([]) == 0.0
