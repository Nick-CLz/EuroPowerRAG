"""Walk-forward backtest runner (T6 / P5.1).

Iterates over business days in ``[start, end]``, calls the decision agent
for each day, marks-to-market against next-day actuals, and writes a CSV
trade log to ``data/backtest/trades_<run_id>.csv``.

Tests should cover: empty range, single day, weekend skip, missing forecast.
"""

from __future__ import annotations

import csv
import logging
import uuid
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.agent.decide import decide
from src.agent.schema import TradeDirection

log = logging.getLogger(__name__)

__all__ = ["run_backtest"]

OUTPUT_DIR = Path("data/backtest")
PRICES_PATH = Path("data/processed/prices_history.parquet")

FIELDNAMES = [
    "date", "country", "signal", "confidence",
    "executed_price", "mark_price", "size_mwh", "pnl", "return",
]


def run_backtest(
    country: str,
    start: date,
    end: date,
    *,
    quiet: bool = False,
) -> Optional[Path]:
    """Run a walk-forward backtest for *country* over ``[start, end]``.

    Args:
        country: Zone code (e.g. ``DE_LU``).
        start:   First business day (inclusive).
        end:     Last business day (inclusive).
        quiet:   Suppress the progress bar.

    Returns:
        Path to the CSV trade log, or ``None`` if the run couldn't start.
    """
    if start > end:
        log.error("start (%s) is after end (%s)", start, end)
        return None

    if not PRICES_PATH.exists():
        log.error("%s not found — run price_history ingestion first", PRICES_PATH)
        return None

    # ── Load & index prices ──────────────────────────────────────────────
    df = pd.read_parquet(PRICES_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    country_df = df[df["country"] == country].sort_values("date").set_index("date")

    if country_df.empty:
        log.error("No price data for country=%s", country)
        return None

    # Pre-compute sorted date array for efficient next-day lookup
    all_dates = np.array(sorted(country_df.index.unique()))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    out_path = OUTPUT_DIR / f"trades_{run_id}.csv"

    bdays = pd.bdate_range(start=start, end=end).date
    if len(bdays) == 0:
        log.warning("No business days in [%s, %s]", start, end)
        return None

    # ── Walk-forward loop ────────────────────────────────────────────────
    trades: list[dict] = []
    skipped = 0

    for current_date in tqdm(bdays, desc=f"Backtest {country}", disable=quiet):
        # Decision
        try:
            signal = decide(country, current_date)
        except NotImplementedError:
            skipped += 1
            continue
        except Exception as exc:
            log.debug("Agent error on %s: %s", current_date, exc)
            skipped += 1
            continue

        if signal.direction == TradeDirection.HOLD or signal.size_mwh == 0:
            continue

        # Mark-to-market
        if current_date not in country_df.index:
            continue

        executed_price = float(country_df.loc[current_date, "price_eur_mwh"])

        # Efficient next-day lookup via sorted array
        idx = np.searchsorted(all_dates, current_date, side="right")
        if idx >= len(all_dates):
            continue
        next_day = all_dates[idx]
        if next_day not in country_df.index:
            continue
        mark_price = float(country_df.loc[next_day, "price_eur_mwh"])

        # PnL
        diff = mark_price - executed_price
        sign = 1.0 if signal.direction == TradeDirection.BUY else -1.0
        pnl = sign * diff * signal.size_mwh
        ret = sign * diff / executed_price if executed_price != 0 else 0.0

        trades.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "country": country,
            "signal": signal.direction.value,
            "confidence": round(signal.confidence, 3),
            "executed_price": round(executed_price, 2),
            "mark_price": round(mark_price, 2),
            "size_mwh": round(signal.size_mwh, 3),
            "pnl": round(pnl, 2),
            "return": round(ret, 6),
        })

    # ── Write output ─────────────────────────────────────────────────────
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(trades)

    log.info(
        "Backtest %s [%s → %s]: %d trades, %d skipped → %s",
        country, start, end, len(trades), skipped, out_path,
    )
    return out_path
