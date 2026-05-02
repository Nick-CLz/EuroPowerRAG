"""Long-horizon price history ingestion for forecasting.

F2: needs ≥ 1 year of daily-aggregated day-ahead prices for forecasting.

Source cascade (first success wins):
  1. ENTSO-E        — preferred; requires free API token (ENTSOE_API_TOKEN)
  2. Energy-Charts  — free, no token; covers DE_LU, FR, NL via public API
  3. Synthetic      — deterministic RNG fallback; labelled in parquet for transparency

Output: data/processed/prices_history.parquet
Schema: date (str YYYY-MM-DD), country (str), price_eur_mwh (float), source (str)
"""

import os
import time
from datetime import date, timedelta
from pathlib import Path

import logging
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/processed/prices_history.parquet")

# Zones with real data available on each source
ZONES_ENTSOE = {"DE_LU": "Germany", "FR": "France", "NL": "Netherlands",
                "GB": "Great Britain", "ES": "Spain", "BE": "Belgium"}
ZONES_ENERGY_CHARTS = {"DE_LU": "DE-LU", "FR": "FR", "NL": "NL"}
# GB has no free public API without token; filled with synthetic below
ZONES_SYNTHETIC = {"GB": "Great Britain"}


# ── 1. ENTSO-E (token required) ──────────────────────────────────────────────

def fetch_entsoe_history(days: int = 400) -> pd.DataFrame:
    """ENTSO-E day-ahead prices via entsoe-py. Requires ENTSOE_API_TOKEN."""
    from entsoe import EntsoePandasClient

    token = os.getenv("ENTSOE_API_TOKEN", "")
    if not token or token.startswith("your_"):
        raise EnvironmentError("ENTSOE_API_TOKEN not set")

    client = EntsoePandasClient(api_key=token)
    end = pd.Timestamp.now(tz="UTC").floor("D")
    start = end - pd.Timedelta(days=days)

    rows = []
    for zone in ZONES_ENTSOE:
        try:
            prices = client.query_day_ahead_prices(zone, start=start, end=end)
            daily = prices.resample("D").mean()
            for d, p in daily.items():
                if pd.notna(p):
                    rows.append({"date": d.strftime("%Y-%m-%d"), "country": zone,
                                 "price_eur_mwh": float(p), "source": "entsoe"})
            log.info("[ENTSO-E] %s: %d days", zone, len(daily))
        except Exception as exc:
            log.warning("[ENTSO-E] %s failed: %s", zone, exc)

    return pd.DataFrame(rows)


# ── 2. Energy-Charts (free, no token) ────────────────────────────────────────

_EC_BASE = "https://api.energy-charts.info/price"
_EC_HEADERS = {"User-Agent": "EuroPowerRAG/1.0 (research)"}


def _fetch_ec_chunk(bzn: str, start: date, end: date) -> list[dict]:
    """Fetch one month of hourly prices from Energy-Charts and return daily rows."""
    def _iso(d: date) -> str:
        return d.strftime("%Y-%m-%dT00:00+01:00").replace("+", "%2B")

    url = f"{_EC_BASE}?bzn={bzn}&start={_iso(start)}&end={_iso(end)}"
    resp = requests.get(url, timeout=20, headers=_EC_HEADERS)
    if resp.status_code != 200 or not resp.text.strip():
        return []
    data = resp.json()
    unix_times = data.get("unix_seconds", [])
    prices = data.get("price", [])
    if not unix_times or not prices:
        return []

    df = pd.DataFrame({"ts": pd.to_datetime(unix_times, unit="s", utc=True),
                       "price": prices})
    df["date"] = df["ts"].dt.tz_convert("Europe/Berlin").dt.date
    daily = df.groupby("date")["price"].mean().reset_index()
    return daily.to_dict("records")


def fetch_energy_charts_history(days: int = 400) -> pd.DataFrame:
    """Energy-Charts public API — free, no token. Covers DE_LU, FR, NL."""
    end = date.today()
    start = end - timedelta(days=days)

    rows = []
    for zone, bzn in ZONES_ENERGY_CHARTS.items():
        log.info("[Energy-Charts] Fetching %s (%s)…", zone, bzn)
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=30), end)
            try:
                chunk = _fetch_ec_chunk(bzn, cursor, chunk_end)
                for r in chunk:
                    rows.append({"date": str(r["date"]), "country": zone,
                                 "price_eur_mwh": float(r["price"]), "source": "energy_charts"})
            except Exception as exc:
                log.warning("[Energy-Charts] %s %s–%s: %s", zone, cursor, chunk_end, exc)
            cursor = chunk_end + timedelta(days=1)
            time.sleep(0.3)   # be polite to free API

        zone_rows = [r for r in rows if r["country"] == zone]
        log.info("[Energy-Charts] %s: %d daily rows", zone, len(zone_rows))

    return pd.DataFrame(rows)


# ── 3. Synthetic fallback ─────────────────────────────────────────────────────

# Realistic base prices and volatility per zone (EUR/MWh, annualised %)
_SYNTHETIC_PARAMS = {
    "DE_LU": (70.0, 0.35), "FR": (72.0, 0.32), "NL": (71.0, 0.34),
    "GB":    (85.0, 0.40), "ES": (65.0, 0.30), "BE": (71.5, 0.33),
}


def generate_synthetic_history(zones: list[str], days: int = 400) -> pd.DataFrame:
    """Deterministic synthetic prices using geometric Brownian motion.

    Clearly labelled source='synthetic' in the output — never silently substituted
    without the caller knowing.
    """
    rng = np.random.default_rng(42)
    end = date.today()
    dates = [end - timedelta(days=i) for i in range(days - 1, -1, -1)]
    rows = []
    for zone in zones:
        mu, sigma = _SYNTHETIC_PARAMS.get(zone, (70.0, 0.35))
        dt = 1 / 252
        log_returns = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), days)
        prices = mu * np.exp(np.cumsum(log_returns))
        # Add seasonal pattern: slightly higher in winter
        seasonal = 10 * np.cos(2 * np.pi * np.arange(days) / 365)
        prices = np.clip(prices + seasonal, 5.0, 300.0)
        for d, p in zip(dates, prices):
            rows.append({"date": d.strftime("%Y-%m-%d"), "country": zone,
                         "price_eur_mwh": round(float(p), 2), "source": "synthetic"})
    log.info("[Synthetic] Generated %d rows for %s", len(rows), zones)
    return pd.DataFrame(rows)


# ── Main runner ───────────────────────────────────────────────────────────────

def run(days: int = 400) -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    covered: set[str] = set()

    # 1. ENTSO-E (best quality, needs token)
    try:
        df_entsoe = fetch_entsoe_history(days=days)
        if not df_entsoe.empty:
            frames.append(df_entsoe)
            covered |= set(df_entsoe["country"].unique())
            log.info("[Runner] ENTSO-E provided: %s", sorted(covered))
    except Exception as exc:
        log.info("[Runner] ENTSO-E skipped: %s", exc)

    # 2. Energy-Charts (free API) for zones not yet covered
    ec_needed = {z: bzn for z, bzn in ZONES_ENERGY_CHARTS.items() if z not in covered}
    if ec_needed:
        try:
            df_ec = fetch_energy_charts_history(days=days)
            # Keep only zones not already covered by ENTSO-E
            df_ec = df_ec[df_ec["country"].isin(ec_needed)]
            if not df_ec.empty:
                frames.append(df_ec)
                covered |= set(df_ec["country"].unique())
                log.info("[Runner] Energy-Charts provided: %s", sorted(df_ec["country"].unique()))
        except Exception as exc:
            log.warning("[Runner] Energy-Charts failed: %s", exc)

    # 3. Synthetic for anything still missing (including GB)
    missing = [z for z in ZONES_ENTSOE if z not in covered]
    if missing:
        df_syn = generate_synthetic_history(missing, days=days)
        frames.append(df_syn)
        covered |= set(missing)
        log.info("[Runner] Synthetic filled: %s", missing)

    if not frames:
        log.error("[Runner] No data from any source.")
        return 0

    df = (pd.concat(frames, ignore_index=True)
            .sort_values(["country", "date"])
            .drop_duplicates(["country", "date"])
            .reset_index(drop=True))

    df.to_parquet(OUTPUT_PATH, index=False)

    # Summary
    log.info("=" * 60)
    log.info("Saved %d rows → %s", len(df), OUTPUT_PATH)
    for country, grp in df.groupby("country"):
        sources = grp["source"].unique().tolist()
        log.info("  %-6s  %d days  %s → %s  [%s]",
                 country, len(grp), grp["date"].min(), grp["date"].max(),
                 ", ".join(sources))
    return len(df)


if __name__ == "__main__":
    run()
