"""Realized volatility from price history.

P3.1 — single source of truth for "how big a move should we expect."
"""

from pathlib import Path

import numpy as np
import pandas as pd

PRICES_PATH = Path("data/processed/prices_history.parquet")


def realized_volatility(
    country: str,
    window_days: int = 20,
    annualize: bool = True,
) -> float:
    """Rolling standard deviation of daily log returns.

    Returns:
        Annualized realized vol (multiply by sqrt(252)) if annualize=True,
        else daily vol.
    """
    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"{PRICES_PATH} not found.")

    df = pd.read_parquet(PRICES_PATH)
    df = df[df["country"] == country].sort_values("date")
    if len(df) < window_days + 1:
        raise ValueError(f"Need {window_days + 1} prices; have {len(df)} for {country}")

    prices = df["price_eur_mwh"].astype(float).values
    # log returns; use absolute prices guarded against zero
    safe = np.where(prices <= 0, 1e-3, prices)
    log_returns = np.diff(np.log(safe))
    recent = log_returns[-window_days:]
    daily_vol = float(np.std(recent, ddof=1))

    return daily_vol * np.sqrt(252) if annualize else daily_vol
