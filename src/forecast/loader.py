"""Shared data loading utilities for forecast models.

Centralises the ``_load_prices`` helper that was duplicated across
``arima.py``, ``xgb.py``, and ``baseline.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["load_prices", "invalidate_cache"]

PRICES_PATH = Path("data/processed/prices_history.parquet")


_cache: pd.DataFrame | None = None


def load_prices() -> pd.DataFrame:
    """Load and cache the price-history parquet.

    Returns a DataFrame sorted by ``(country, date)`` with a
    ``datetime64`` ``date`` column.

    Call ``invalidate_cache()`` after re-ingestion to pick up new data.

    Raises:
        FileNotFoundError: if the parquet hasn't been generated yet.
    """
    global _cache
    if _cache is not None:
        return _cache
    if not PRICES_PATH.exists():
        raise FileNotFoundError(
            f"{PRICES_PATH} not found — run `python -m src.ingestion.price_history` first."
        )
    df = pd.read_parquet(PRICES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    _cache = df.sort_values(["country", "date"]).reset_index(drop=True)
    return _cache


def invalidate_cache() -> None:
    """Clear the cached price DataFrame.  Call after ingestion."""
    global _cache
    _cache = None
