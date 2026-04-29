"""Long-horizon price history ingestion for forecasting.

F2: needs ≥ 1 year of daily-aggregated day-ahead prices for forecasting.

Strategy:
  1. Try ENTSO-E if a token is set (preferred — clean structured data)
  2. Fallback: Ember Climate's monthly European wholesale price CSV (free, no token)

Output: data/processed/prices_history.parquet
"""

import io
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

OUTPUT_PATH = Path("data/processed/prices_history.parquet")

ZONES = {
    "DE_LU": "Germany",
    "FR": "France",
    "NL": "Netherlands",
    "GB": "Great Britain",
    "ES": "Spain",
    "BE": "Belgium",
}


def fetch_entsoe_history(days: int = 365) -> pd.DataFrame:
    """ENTSO-E day-ahead prices, aggregated daily."""
    from entsoe import EntsoePandasClient

    token = os.getenv("ENTSOE_API_TOKEN")
    if not token or token == "your_token_here":
        raise EnvironmentError("No ENTSOE_API_TOKEN — falling back to Ember")

    client = EntsoePandasClient(api_key=token)
    end = pd.Timestamp.now(tz="UTC").floor("D")
    start = end - pd.Timedelta(days=days)

    rows = []
    for zone, name in ZONES.items():
        try:
            prices = client.query_day_ahead_prices(zone, start=start, end=end)
            daily = prices.resample("D").mean()
            for d, p in daily.items():
                if pd.notna(p):
                    rows.append(
                        {"date": d.strftime("%Y-%m-%d"), "country": zone, "price_eur_mwh": float(p)}
                    )
            print(f"  [ENTSO-E] {zone}: {len(daily)} days")
        except Exception as e:
            print(f"  [ENTSO-E] {zone} failed: {e}")

    return pd.DataFrame(rows)


def fetch_ember_history() -> pd.DataFrame:
    """Free fallback — Ember Climate's European wholesale electricity prices CSV.

    Reference: https://ember-climate.org/data-catalogue/european-wholesale-electricity-price-data/
    """
    url = "https://ember-energy.org/app/uploads/2024/05/european_wholesale_electricity_price_data_monthly.csv"

    print(f"  [Ember] Downloading {url}")
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "EuroPowerRAG/1.0"})
        resp.raise_for_status()
    except Exception as e:
        print(f"  [Ember] Download failed: {e}")
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(resp.text))

    # Ember columns: Country, Date, Price (EUR/MWhe)
    rename_map = {
        "Country": "country_name",
        "Date": "date",
        "Price (EUR/MWhe)": "price_eur_mwh",
    }
    df = df.rename(columns=rename_map)
    df = df[["country_name", "date", "price_eur_mwh"]].copy()

    # Map Ember country names to our zone codes
    name_to_zone = {v: k for k, v in ZONES.items()}
    df["country"] = df["country_name"].map(name_to_zone)
    df = df.dropna(subset=["country", "price_eur_mwh"])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    return df[["date", "country", "price_eur_mwh"]]


def run() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[price_history] Fetching long-horizon prices...")
    try:
        df = fetch_entsoe_history(days=365)
        if df.empty:
            raise RuntimeError("ENTSO-E returned no rows")
        source = "ENTSO-E"
    except Exception as e:
        print(f"  ENTSO-E unavailable: {e}")
        df = fetch_ember_history()
        source = "Ember"

    if df.empty:
        print("  No price history retrieved from either source.")
        return 0

    df = df.sort_values(["country", "date"]).drop_duplicates(["country", "date"])
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"  Saved {len(df)} rows ({source}) → {OUTPUT_PATH}")
    print(f"  Coverage: {df['country'].nunique()} countries, "
          f"{df['date'].min()} → {df['date'].max()}")
    return len(df)


if __name__ == "__main__":
    run()
