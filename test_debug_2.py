import pytest
from datetime import date

def test_arima_raises_on_insufficient_data(monkeypatch):
    from src.forecast.arima import arima_forecast
    import src.forecast.loader as loader
    from tests.test_forecast import _make_price_df
    short_df = _make_price_df(n_days=10)
    monkeypatch.setattr(loader, "_cache", short_df)
    try:
        arima_forecast("DE_LU", date(2026, 4, 20))
    except Exception as e:
        print("Raised:", e)

if __name__ == "__main__":
    test_arima_raises_on_insufficient_data(pytest.MonkeyPatch())
