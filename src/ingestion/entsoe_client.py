"""ENTSO-E Transparency Platform ingestion.

Fetches day-ahead prices and generation mix for major European bidding zones.
Register for a free API token at: https://transparency.entsoe.eu/
"""

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Bidding zone codes used by entsoe-py
ZONES = {
    "DE_LU": "Germany/Luxembourg",
    "FR": "France",
    "NL": "Netherlands",
    "GB": "Great Britain",
    "ES": "Spain",
    "BE": "Belgium",
}


def _get_client():
    from entsoe import EntsoePandasClient

    token = os.getenv("ENTSOE_API_TOKEN")
    if not token:
        raise EnvironmentError("ENTSOE_API_TOKEN not set — skipping ENTSO-E ingestion")
    return EntsoePandasClient(api_key=token)


def fetch_day_ahead_prices(days_back: int = 30) -> list[dict]:
    try:
        client = _get_client()
    except EnvironmentError as e:
        print(f"  {e}")
        return []

    end = pd.Timestamp.now(tz="UTC").floor("D")
    start = end - pd.Timedelta(days=days_back)

    documents = []
    for zone, zone_name in ZONES.items():
        try:
            prices: pd.Series = client.query_day_ahead_prices(zone, start=start, end=end)
            daily = prices.resample("D").agg(["mean", "min", "max"]).round(2)

            for date, row in daily.iterrows():
                doc_text = (
                    f"Day-ahead electricity prices for {zone_name} ({zone}) "
                    f"on {date.strftime('%Y-%m-%d')}:\n"
                    f"  Average: {row['mean']:.2f} EUR/MWh\n"
                    f"  Min:     {row['min']:.2f} EUR/MWh\n"
                    f"  Max:     {row['max']:.2f} EUR/MWh\n"
                )
                documents.append(
                    {
                        "text": doc_text,
                        "metadata": {
                            "source": "ENTSO-E",
                            "doc_type": "price_data",
                            "country": zone,
                            "country_name": zone_name,
                            "date": date.strftime("%Y-%m-%d"),
                            "data_type": "day_ahead_prices",
                        },
                    }
                )
        except Exception as e:
            print(f"  [ENTSO-E] Prices for {zone}: {e}")

    return documents


def fetch_generation_mix(days_back: int = 7) -> list[dict]:
    try:
        client = _get_client()
    except EnvironmentError as e:
        print(f"  {e}")
        return []

    end = pd.Timestamp.now(tz="UTC").floor("D")
    start = end - pd.Timedelta(days=days_back)

    documents = []
    for zone, zone_name in ZONES.items():
        try:
            raw = client.query_generation(zone, start=start, end=end)

            # entsoe-py may return MultiIndex columns (fuel_type, "Actual Aggregated")
            if isinstance(raw.columns, pd.MultiIndex):
                level1_vals = raw.columns.get_level_values(1).unique()
                key = "Actual Aggregated" if "Actual Aggregated" in level1_vals else level1_vals[0]
                gen = raw.xs(key, axis=1, level=1)
            else:
                gen = raw

            daily = gen.resample("D").mean().round(1)

            for date, row in daily.iterrows():
                total = row.dropna().sum()
                if total <= 0:
                    continue

                mix_lines = [
                    f"  {src}: {val:.0f} MW ({val / total * 100:.1f}%)"
                    for src, val in row.dropna().items()
                    if val > 0
                ]
                doc_text = (
                    f"Power generation mix for {zone_name} ({zone}) "
                    f"on {date.strftime('%Y-%m-%d')} (daily average):\n"
                    f"  Total: {total:.0f} MW\n"
                    + "\n".join(mix_lines)
                )
                documents.append(
                    {
                        "text": doc_text,
                        "metadata": {
                            "source": "ENTSO-E",
                            "doc_type": "generation_data",
                            "country": zone,
                            "country_name": zone_name,
                            "date": date.strftime("%Y-%m-%d"),
                            "data_type": "generation_mix",
                        },
                    }
                )
        except Exception as e:
            print(f"  [ENTSO-E] Generation for {zone}: {e}")

    return documents


def run(output_dir: Path = Path("data/processed")) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("  Fetching day-ahead prices (last 30 days)...")
    price_docs = fetch_day_ahead_prices(days_back=30)
    print(f"  → {len(price_docs)} price documents")

    print("  Fetching generation mix (last 7 days)...")
    gen_docs = fetch_generation_mix(days_back=7)
    print(f"  → {len(gen_docs)} generation documents")

    all_docs = price_docs + gen_docs
    out_path = output_dir / "entsoe.jsonl"
    with open(out_path, "w") as f:
        for doc in all_docs:
            f.write(json.dumps(doc) + "\n")

    print(f"  Saved {len(all_docs)} ENTSO-E documents → {out_path}")
    return len(all_docs)
