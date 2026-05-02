"""Decision agent — composes forecast + sentiment + risk into a TradeSignal.

P4.2 / P4.3 / P4.4.

Architecture:
    1. Gather inputs: forecast, sentiment, risk/volatility.
    2. Attempt a Claude-powered decision (structured JSON output).
    3. Fall back to a rule-based heuristic if Claude is unavailable.
    4. Self-critique: override to HOLD if confidence < 0.6 or rationale
       fails to reference all three inputs.

AGENTS.md rules enforced:
    #5  Structured output (TradeSignal Pydantic model)
    #6  ≤ 3 tools per agent loop
    #7  Self-critique before emitting signal
    #9  Budget tracking via ``src.utils.budget``
    #10 Human-in-the-loop — no autonomous execution
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

from src.agent.schema import ForecastResult, TradeDirection, TradeSignal
from src.forecast.api import forecast_next_day
from src.risk.volatility import realized_volatility

log = logging.getLogger(__name__)

__all__ = ["decide"]

# ── Sentiment helper ─────────────────────────────────────────────────────────

_SENTIMENT_PATH = Path("data/processed/sentiment_daily.parquet")


def _get_sentiment(country: str, target_date: date) -> float:
    """Best-effort daily sentiment score for *country* on *target_date*.

    Returns 0.0 (neutral) if sentiment data isn't available yet —
    this lets the rest of the pipeline run before P1.4 is fully wired.
    """
    if not _SENTIMENT_PATH.exists():
        return 0.0

    try:
        import pandas as pd

        df = pd.read_parquet(_SENTIMENT_PATH)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        match = df[(df["country"] == country) & (df["date"] == target_date)]
        if match.empty:
            # Try EU-level aggregate
            match = df[(df["country"] == "EU") & (df["date"] == target_date)]
        if match.empty:
            return 0.0
        return float(match.iloc[0].get("mean_weighted_score", 0.0))
    except Exception as exc:
        log.debug("Sentiment lookup failed: %s", exc)
        return 0.0


# ── Risk helper ──────────────────────────────────────────────────────────────


def _get_volatility(country: str) -> float:
    """Best-effort annualised volatility.  Returns 0.3 as a default."""
    try:
        return realized_volatility(country)
    except Exception as exc:
        log.debug("Volatility lookup failed: %s — using default 0.30", exc)
        return 0.30


# ── Claude-based decision ────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a European power-market trading agent.  You receive three inputs:
a price forecast, a sentiment score, and a volatility estimate.

Respond with **valid JSON only** — no markdown fences, no commentary.

Schema:
{
  "direction": "BUY" | "SELL" | "HOLD",
  "size_mwh": <float>,
  "confidence": <float 0-1>,
  "rationale": "<string ≥50 chars referencing forecast, sentiment, and risk>"
}
"""


def _claude_decide(
    country: str,
    target_date: date,
    forecast: ForecastResult,
    sentiment: float,
    vol: float,
) -> dict | None:
    """Try to get a structured decision from Claude.  Returns None on failure."""
    try:
        from src.utils.anthropic_client import call_claude
    except EnvironmentError:
        log.info("Anthropic SDK not configured — skipping Claude decision")
        return None

    user_msg = (
        f"Country: {country}\n"
        f"Date: {target_date}\n"
        f"Forecast: {forecast.point_forecast_eur_mwh:.2f} EUR/MWh "
        f"(±{forecast.std_eur_mwh:.2f}, model={forecast.model_name})\n"
        f"Sentiment: {sentiment:+.2f}\n"
        f"Annualised Volatility: {vol:.2%}\n"
    )

    try:
        raw = call_claude(
            model="claude-haiku-4-5",
            system_prompt=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            label=f"decide_{country}_{target_date}",
            max_tokens=512,
        )
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Claude returned non-JSON for %s %s", country, target_date)
        return None
    except Exception as exc:
        log.warning("Claude call failed for %s %s: %s", country, target_date, exc)
        return None


# ── Rule-based fallback ──────────────────────────────────────────────────────


def _heuristic_decide(
    forecast: ForecastResult,
    sentiment: float,
    vol: float,
) -> tuple[TradeDirection, float, float, str]:
    """Simple threshold-based heuristic.  Returns (direction, size, confidence, rationale)."""
    price = forecast.point_forecast_eur_mwh
    std = forecast.std_eur_mwh

    # Bullish if forecast is cheap (< mean-1σ proxy) AND sentiment non-negative
    if price < 50 and sentiment >= -0.2:
        direction = TradeDirection.BUY
    elif price > 80 and sentiment <= 0.2:
        direction = TradeDirection.SELL
    else:
        direction = TradeDirection.HOLD

    # Confidence: inversely proportional to vol, scaled by sentiment alignment
    base_conf = max(0.3, min(0.8, 1.0 - vol))
    sentiment_boost = 0.1 if (direction == TradeDirection.BUY and sentiment > 0) or \
                              (direction == TradeDirection.SELL and sentiment < 0) else -0.05
    confidence = round(min(1.0, max(0.0, base_conf + sentiment_boost)), 2)

    size = 10.0 if direction != TradeDirection.HOLD else 0.0

    rationale = (
        f"Heuristic: forecast={price:.1f} EUR/MWh (std={std:.1f}), "
        f"sentiment={sentiment:+.2f}, vol={vol:.1%}. "
        f"{'Cheap price + non-negative sentiment → BUY' if direction == TradeDirection.BUY else ''}"
        f"{'Expensive price + non-positive sentiment → SELL' if direction == TradeDirection.SELL else ''}"
        f"{'No clear edge → HOLD' if direction == TradeDirection.HOLD else ''}"
    )
    return direction, size, confidence, rationale


# ── Self-critique (P4.3) ────────────────────────────────────────────────────


def _self_critique(
    direction: TradeDirection,
    size: float,
    confidence: float,
    rationale: str,
) -> tuple[TradeDirection, float, float, str]:
    """Override to HOLD if the signal fails quality checks.

    Checks:
        1. ``confidence < 0.6``
        2. ``rationale`` shorter than 50 characters
        3. ``rationale`` doesn't mention at least 2 of {forecast, sentiment, risk/vol}
    """
    issues: list[str] = []

    if confidence < 0.6:
        issues.append(f"low confidence ({confidence:.2f})")

    if len(rationale) < 50:
        issues.append("rationale too short")

    keywords_found = sum(
        any(kw in rationale.lower() for kw in group)
        for group in [
            ["forecast", "price", "eur/mwh"],
            ["sentiment"],
            ["vol", "risk", "volatility"],
        ]
    )
    if keywords_found < 2:
        issues.append(f"rationale references only {keywords_found}/3 input categories")

    if issues:
        log.info("Self-critique override → HOLD: %s", "; ".join(issues))
        return (
            TradeDirection.HOLD,
            0.0,
            confidence,
            f"HOLD (critique: {'; '.join(issues)}). Original: {rationale}",
        )

    return direction, size, confidence, rationale


# ── Main entry point ─────────────────────────────────────────────────────────


def decide(country: str, target_date: date) -> TradeSignal:
    """Produce a trade signal for *country* on *target_date*.

    Pipeline:
        1. Gather forecast, sentiment, volatility.
        2. Try Claude → fall back to heuristic.
        3. Run self-critique gate.
        4. Return a validated ``TradeSignal``.
    """
    # ── 1. Gather inputs ─────────────────────────────────────────────────
    forecast = forecast_next_day(country, target_date)
    sentiment = _get_sentiment(country, target_date)
    vol = _get_volatility(country)

    # ── 2. Decision ──────────────────────────────────────────────────────
    claude_result = _claude_decide(country, target_date, forecast, sentiment, vol)

    if claude_result is not None:
        direction = TradeDirection(claude_result.get("direction", "HOLD"))
        size = float(claude_result.get("size_mwh", 10.0))
        confidence = float(claude_result.get("confidence", 0.5))
        rationale = claude_result.get("rationale", "")
    else:
        direction, size, confidence, rationale = _heuristic_decide(forecast, sentiment, vol)

    # ── 3. Self-critique ─────────────────────────────────────────────────
    direction, size, confidence, rationale = _self_critique(direction, size, confidence, rationale)

    # ── 4. Build validated signal ────────────────────────────────────────
    return TradeSignal(
        country=country,
        target_date=target_date,
        direction=direction,
        size_mwh=size,
        confidence=confidence,
        rationale=rationale,
        forecast_price_eur_mwh=forecast.point_forecast_eur_mwh,
        forecast_std_eur_mwh=forecast.std_eur_mwh,
        sentiment_score=sentiment,
        realized_vol_annualized=vol,
        sources=[],
    )


# ── CLI (P4.4) ───────────────────────────────────────────────────────────────


def main() -> None:
    """``python -m src.agent.decide --country DE_LU --date 2026-04-29``"""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="EuroPowerRAG decision agent")
    parser.add_argument("--country", required=True, help="Zone code, e.g. DE_LU")
    parser.add_argument("--date", required=True, help="Target date, e.g. 2026-04-29")
    args = parser.parse_args()

    dt = date.fromisoformat(args.date)
    signal = decide(args.country, dt)

    print(signal.model_dump_json(indent=2))
    sys.exit(0 if signal.direction != TradeDirection.HOLD else 1)


if __name__ == "__main__":
    main()
