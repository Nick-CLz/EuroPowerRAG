import pytest
from datetime import date
from unittest.mock import patch

def _make_price_df(n_days=10, country="DE_LU"):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(99)
    dates = pd.date_range(end="2026-04-20", periods=n_days)
    base = 60.0
    prices = base + rng.normal(0, 5, n_days).cumsum()
    return pd.DataFrame({
        "date": dates,
        "country": country,
        "price_eur_mwh": prices,
    })

def test_debug(monkeypatch):
    from src.forecast.arima import arima_forecast
    import src.forecast.loader as loader
    short_df = _make_price_df(n_days=10)
    monkeypatch.setattr(loader, "_cache", short_df)
    print("cache len:", len(loader._cache))
    df = loader.load_prices()
    print("load_prices len:", len(df))
    try:
        arima_forecast("DE_LU", date(2026, 4, 20))
    except Exception as e:
        print("Raised:", e)

if __name__ == "__main__":
    test_debug(pytest.MonkeyPatch())
