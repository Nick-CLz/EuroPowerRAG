"""Backtest performance metrics.

P5.2 — Sharpe, Sortino, max drawdown, win rate. Pure functions.
Implement now (no model dependencies); useful for unit tests.
"""

import math
from typing import Sequence

import numpy as np

TRADING_DAYS = 252


def sharpe_ratio(returns: Sequence[float], risk_free: float = 0.0) -> float:
    """Annualized Sharpe assuming daily returns."""
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0 or arr.std(ddof=1) == 0:
        return 0.0
    excess = arr - (risk_free / TRADING_DAYS)
    return float(excess.mean() / arr.std(ddof=1) * math.sqrt(TRADING_DAYS))


def sortino_ratio(returns: Sequence[float], risk_free: float = 0.0) -> float:
    """Annualized Sortino — downside-only volatility."""
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    excess = arr - (risk_free / TRADING_DAYS)
    downside = arr[arr < 0]
    if downside.size == 0 or downside.std(ddof=1) == 0:
        return 0.0
    return float(excess.mean() / downside.std(ddof=1) * math.sqrt(TRADING_DAYS))


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Largest peak-to-trough drawdown as a positive fraction (e.g. 0.12 = 12%)."""
    arr = np.asarray(equity_curve, dtype=float)
    if arr.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(arr)
    drawdowns = (running_max - arr) / running_max
    return float(drawdowns.max())


def win_rate(pnls: Sequence[float]) -> float:
    arr = np.asarray(pnls, dtype=float)
    if arr.size == 0:
        return 0.0
    return float((arr > 0).mean())
