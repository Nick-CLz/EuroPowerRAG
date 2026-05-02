"""Evaluation harness for backtest results (T5).

Consumes a trade-log CSV produced by ``src.backtest.runner``, computes
Sharpe, max drawdown, and win rate, asserts against target thresholds,
and appends a result row to ``data/eval/eval_log.jsonl``.

Usage::

    python -m src.evaluation.eval_backtest                         # default path
    python -m src.evaluation.eval_backtest data/backtest/trades_abc123.csv
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.metrics import max_drawdown, sharpe_ratio, sortino_ratio, win_rate

log = logging.getLogger(__name__)

__all__ = ["run_eval"]

LOG_PATH = Path("data/eval/eval_log.jsonl")

# ── Thresholds from TODO.md ──────────────────────────────────────────────────
SHARPE_TARGET = 0.8
MAX_DD_TARGET = 0.08  # 8 %


def run_eval(csv_path: Path | str) -> bool:
    """Evaluate a backtest trade log.

    Args:
        csv_path: Path to the CSV produced by ``run_backtest()``.

    Returns:
        ``True`` if *both* thresholds are met, ``False`` otherwise.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        log.error("Trade log not found: %s", csv_path)
        return False

    df = pd.read_csv(csv_path)

    required = {"pnl", "return"}
    missing = required - set(df.columns)
    if missing:
        log.error("CSV is missing columns: %s", missing)
        return False

    if df.empty:
        log.warning("Trade log is empty — nothing to evaluate")
        _append_log(csv_path, {}, passed=False)
        return False

    returns = df["return"].values.astype(float)
    pnls = df["pnl"].values.astype(float)

    # Equity curve (cumulative returns)
    equity = np.cumprod(1.0 + returns)

    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    mdd = max_drawdown(equity)
    wr = win_rate(pnls)
    total_pnl = float(pnls.sum())
    avg_pnl = float(pnls.mean())

    metrics = {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(mdd, 4),
        "win_rate": round(wr, 4),
        "total_pnl_eur": round(total_pnl, 2),
        "avg_pnl_eur": round(avg_pnl, 2),
        "n_trades": len(df),
    }

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\n  Backtest Evaluation — {csv_path.name}")
    print(f"  {'─' * 40}")
    print(f"  Sharpe Ratio : {sharpe:+.2f}  (target ≥ {SHARPE_TARGET})")
    print(f"  Sortino Ratio: {sortino:+.2f}")
    print(f"  Max Drawdown : {mdd:.1%}   (target ≤ {MAX_DD_TARGET:.0%})")
    print(f"  Win Rate     : {wr:.1%}")
    print(f"  Total PnL    : €{total_pnl:,.2f}")
    print(f"  Avg PnL/trade: €{avg_pnl:,.2f}")
    print(f"  Trades       : {len(df)}")

    # ── Gate ─────────────────────────────────────────────────────────────
    passed = True
    if sharpe < SHARPE_TARGET:
        print(f"  ✗ FAIL  Sharpe {sharpe:.2f} < {SHARPE_TARGET}")
        passed = False
    if mdd > MAX_DD_TARGET:
        print(f"  ✗ FAIL  Max DD {mdd:.1%} > {MAX_DD_TARGET:.0%}")
        passed = False
    if passed:
        print("  ✓ PASS  All thresholds met")

    _append_log(csv_path, metrics, passed_sharpe=(sharpe >= SHARPE_TARGET), passed_mdd=(mdd <= MAX_DD_TARGET))
    return passed


def _append_log(csv_path: Path, metrics: dict, *, passed_sharpe: bool, passed_mdd: bool) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    record_sharpe = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_name": "backtest_sharpe",
        "trade_log": str(csv_path),
        "passed": passed_sharpe,
        "metrics": metrics,
    }
    
    record_risk = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_name": "risk_max_dd",
        "trade_log": str(csv_path),
        "passed": passed_mdd,
        "metrics": {"max_drawdown": metrics.get("max_drawdown", 1.0)},
    }
    
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record_sharpe) + "\n")
        f.write(json.dumps(record_risk) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    if csv_arg:
        path = Path(csv_arg)
    else:
        # Try to find the latest trades_*.csv in data/backtest
        bt_dir = Path("data/backtest")
        if bt_dir.exists():
            logs = list(bt_dir.glob("trades_*.csv"))
            if logs:
                # Sort by mtime to find most recent
                path = max(logs, key=lambda p: p.stat().st_mtime)
                print(f"  [auto] Using latest trade log: {path}")
            else:
                path = Path("data/backtest/trades.csv")
        else:
            path = Path("data/backtest/trades.csv")
            
    ok = run_eval(path)
    sys.exit(0 if ok else 1)
