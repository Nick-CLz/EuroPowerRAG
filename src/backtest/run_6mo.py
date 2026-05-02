"""6-month walk-forward run (P5.3).

Runs a walk-forward backtest for each major zone, evaluates the trade logs,
and writes a summary postmortem to ``docs/`` if any threshold is missed.

Usage::

    python -m src.backtest.run_6mo
    python -m src.backtest.run_6mo --days 90    # shorter window for dev
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.backtest.runner import run_backtest
from src.evaluation.eval_backtest import run_eval

log = logging.getLogger(__name__)

COUNTRIES = ["DE_LU", "FR", "NL", "GB"]
POSTMORTEM_DIR = Path("docs")


def main(days: int = 180) -> None:
    end = date.today()
    start = end - timedelta(days=days)

    print(f"\n{'═' * 60}")
    print(f"  P5.3  Walk-forward backtest  [{start} → {end}]")
    print(f"{'═' * 60}\n")

    results: dict[str, dict] = {}
    all_passed = True

    for country in COUNTRIES:
        print(f"\n── {country} ──")
        trade_log = run_backtest(country, start, end, quiet=False)

        if trade_log is None:
            log.warning("Backtest produced no output for %s", country)
            results[country] = {"status": "no_output"}
            all_passed = False
            continue

        passed = run_eval(trade_log)
        results[country] = {"trade_log": str(trade_log), "passed": passed}
        if not passed:
            all_passed = False

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  SUMMARY")
    print(f"{'═' * 60}")
    for c, r in results.items():
        status = "✓ PASS" if r.get("passed") else "✗ FAIL"
        print(f"  {c:8s}  {status}")

    if all_passed:
        print("\n  🎯 North-star metric achieved: all zones pass thresholds.\n")
    else:
        # Write postmortem
        POSTMORTEM_DIR.mkdir(parents=True, exist_ok=True)
        today_str = date.today().isoformat()
        pm_path = POSTMORTEM_DIR / f"postmortem_{today_str}.md"
        _write_postmortem(pm_path, results, start, end)
        print(f"\n  📝 Postmortem written to {pm_path}\n")


def _write_postmortem(path: Path, results: dict, start: date, end: date) -> None:
    lines = [
        f"# Postmortem — Walk-forward {start} → {end}\n",
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n",
        "\n## Results\n",
    ]
    for country, r in results.items():
        status = "PASS" if r.get("passed") else "FAIL"
        lines.append(f"- **{country}**: {status}")
        if r.get("trade_log"):
            lines.append(f"  Trade log: `{r['trade_log']}`")
        lines.append("")

    lines += [
        "\n## Hypotheses to test next\n",
        "- [ ] Tune XGBoost hyperparameters (grid search over max_depth, n_estimators)",
        "- [ ] Expand ARIMA order search to (p,d,q) ∈ {0..3}³",
        "- [ ] Wire real sentiment data (requires T0 human labelling)",
        "- [ ] Investigate per-country volatility regimes",
        "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180, help="Backtest window in days")
    args = parser.parse_args()
    main(days=args.days)
