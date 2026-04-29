"""Sentiment scoring for energy news articles.

P1.2 — uses Gemini structured output with a Pydantic response_schema.

Per AGENTS.md:
  - rule #4: temperature=0 for deterministic scoring
  - rule #5: structured output via response_schema
  - rule #8: prompt is versioned in this file

Cost: free tier on gemini-2.5-flash.
Latency target: < 2s p95 per article.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.agent.schema import SentimentScore
from src.utils.budget import budget

load_dotenv()

# v1 sentiment prompt — bump version on every change, log in eval_log.jsonl
PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are an energy market analyst scoring news for next-day price impact.

For the given article, return:
  score:      a number in [-1, +1]
              -1.0  =  strongly bearish for next-day day-ahead electricity price
               0.0  =  neutral / no clear directional signal
              +1.0  =  strongly bullish for next-day day-ahead electricity price
  confidence: a number in [0, 1] — your self-rated confidence
  reasoning:  one sentence explaining the score, citing the specific market mechanism

Rules:
- Bullish drivers: supply disruption, generator outages, cold weather, gas price spikes,
  CO2 price rises, regulatory tightening, geopolitical risk to imports.
- Bearish drivers: new generation coming online, demand drop, mild weather, gas falls,
  oversupply, regulatory loosening, demand-side response programs.
- Pure financial / corporate news with no production or demand impact: 0.0
- News older than the trade window (>3 days stale) should still be scored, but with
  lower confidence.
- Do not speculate beyond the article content.
"""


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


def score_article(title: str, summary: str) -> SentimentScore:
    """Score one article. Deterministic (temperature=0)."""
    client = _get_client()
    user_message = f"Title: {title}\n\nSummary: {summary}"

    with budget.track(label="sentiment_score", model="gemini-2.5-flash") as record:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=SentimentScore,
                temperature=0.0,
            ),
        )
        # Best-effort token bookkeeping
        usage = getattr(response, "usage_metadata", None)
        if usage:
            record.input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            record.output_tokens = getattr(usage, "candidates_token_count", 0) or 0

    return SentimentScore.model_validate_json(response.text)


def discretize_score(score: float, threshold: float = 0.33) -> int:
    """Convert continuous score in [-1, +1] to bucket {-1, 0, +1} for κ eval."""
    if score < -threshold:
        return -1
    if score > threshold:
        return 1
    return 0


if __name__ == "__main__":
    # Smoke test
    result = score_article(
        title="French nuclear output drops 20% on unplanned outages",
        summary="EDF announced today that several reactors at the Cattenom and Bugey "
        "sites are offline for unplanned maintenance, removing roughly 6 GW of capacity "
        "from the French grid through next week.",
    )
    print(f"Score: {result.score}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
