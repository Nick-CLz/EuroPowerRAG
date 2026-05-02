"""Evaluation harness for forecasting models (T4).

Computes MAE, RMSE, directional accuracy, and bias for every model
registered in ``src/forecast/``.  Walk-forward: the last *test_days*
of each country's series are held out; forecasts are generated using
only data available up to the day before.

Results are appended to ``data/eval/eval_log.jsonl``.

Usage::

    python -m src.evaluation.eval_forecast          # default 30 test days
    python -m src.evaluation.eval_forecast --days 60
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.forecast.baseline import (
    naive_forecast,
    rolling_mean_forecast,
    seasonal_naive_forecast,
)

log = logging.getLogger(__name__)

__all__ = ["run_eval"]

PRICES_PATH = Path("data/processed/prices_history.parquet")
LOG_PATH = Path("data/eval/eval_log.jsonl")


# ── Model registry ───────────────────────────────────────────────────────────

def _build_model_registry() -> dict[str, Callable]:
    """Dynamically collect every available forecast function."""
    models: dict[str, Callable] = {
        "naive": naive_forecast,
        "seasonal_naive": seasonal_naive_forecast,
        "rolling_mean": rolling_mean_forecast,
    }

    try:
        from src.forecast.arima import arima_forecast
        models["arima"] = arima_forecast
    except ImportError:
        pass

    try:
        from src.forecast.xgb import xgb_forecast
        models["xgb"] = lambda c, d: xgb_forecast(c, d, with_sentiment=False)
        models["xgb_sentiment"] = lambda c, d: xgb_forecast(c, d, with_sentiment=True)
    except ImportError:
        pass

    return models


# ── Metric accumulators ──────────────────────────────────────────────────────

def _empty_accum() -> dict:
    return {"errors": [], "abs_errors": [], "sq_errors": [], "dir_correct": [], "n": 0}


def _compute_metrics(acc: dict) -> dict:
    if acc["n"] == 0:
        return {}
    return {
        "mae": float(np.mean(acc["abs_errors"])),
        "rmse": float(np.sqrt(np.mean(acc["sq_errors"]))),
        "bias": float(np.mean(acc["errors"])),
        "directional_accuracy": float(np.mean(acc["dir_correct"])),
        "n_obs": acc["n"],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run_eval(test_days: int = 30) -> dict[str, dict]:
    """Run the walk-forward forecast evaluation.

    Returns:
        A dict mapping model name → metrics dict.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not PRICES_PATH.exists():
        log.error("No price history at %s — run price_history ingestion first.", PRICES_PATH)
        return {}

    df = pd.read_parquet(PRICES_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["country", "date"])

    models = _build_model_registry()
    accums = {name: _empty_accum() for name in models}

    countries = df["country"].unique()

    for country in countries:
        cdf = df[df["country"] == country]
        dates = cdf["date"].unique()

        # Need enough history for lag-7 + test window
        if len(dates) < test_days + 8:
            log.info("Skipping %s — only %d dates available", country, len(dates))
            continue

        test_dates = dates[-test_days:]

        for i, target_date in enumerate(test_dates):
            actual = float(cdf[cdf["date"] == target_date]["price_eur_mwh"].iloc[0])
            prev_date = dates[-(test_days - i) - 1]
            prev_actual = float(cdf[cdf["date"] == prev_date]["price_eur_mwh"].iloc[0])
            true_dir = np.sign(actual - prev_actual)

            for model_name, fn in models.items():
                try:
                    result = fn(country, target_date)
                    pred = result.point_forecast_eur_mwh
                except Exception as exc:
                    log.debug("%s failed on %s %s: %s", model_name, country, target_date, exc)
                    continue

                err = pred - actual
                accums[model_name]["errors"].append(err)
                accums[model_name]["abs_errors"].append(abs(err))
                accums[model_name]["sq_errors"].append(err ** 2)

                pred_dir = np.sign(pred - prev_actual)
                correct = 1 if (true_dir == pred_dir) else 0
                accums[model_name]["dir_correct"].append(correct)
                accums[model_name]["n"] += 1

    # ── Summarise ────────────────────────────────────────────────────────
    all_metrics: dict[str, dict] = {}
    for name, acc in accums.items():
        m = _compute_metrics(acc)
        if not m:
            continue
        all_metrics[name] = m
        print(
            f"  {name:20s}  MAE={m['mae']:7.2f}  RMSE={m['rmse']:7.2f}  "
            f"Bias={m['bias']:+7.2f}  DirAcc={m['directional_accuracy']:.1%}  n={m['n_obs']}"
        )

    # ── Persist ──────────────────────────────────────────────────────────
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_name": "forecast_directional_accuracy",
        "test_days": test_days,
        "metrics": all_metrics,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"\n  → Appended to {LOG_PATH}")
    return all_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30, help="Number of test days")
    args = parser.parse_args()
    run_eval(test_days=args.days)
