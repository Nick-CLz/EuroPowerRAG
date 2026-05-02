"""Corpus Judge — Claude Opus 4.7 generates gold-standard evaluation data.

Unblocks T0: generates data/eval/sentiment_gold.jsonl (30 labeled articles)
using Opus 4.7's European power market domain knowledge as a proxy for
human labeling during development.

Two-model split:
  - Google AI Studio Gemini Flash  — runtime sentiment scoring, RAG chain (existing)
  - Claude Opus 4.7 (Anthropic)    — corpus judgement / gold label generation only

Note: LLM-generated labels are a DEVELOPMENT PROXY. Human audit is recommended
before relying on eval κ in production. See Human Review Tracker printed at end.

Usage:
    python -m src.evaluation.corpus_judge              # generate 30 articles
    python -m src.evaluation.corpus_judge --dry-run    # preview without API calls
    python -m src.evaluation.corpus_judge --n 30       # explicit count
"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from src.utils.budget import budget

load_dotenv()

GOLD_PATH = Path("data/eval/sentiment_gold.jsonl")
JUDGE_VERSION = "v1"

SYSTEM_PROMPT = """You are an expert European power market analyst with deep knowledge of:
- Day-ahead electricity price drivers for DE_LU (Germany/Luxembourg), FR (France),
  NL (Netherlands), GB (Great Britain), and pan-EU energy markets
- French nuclear fleet status and EDF reactor availability (Cattenom, Bugey, Flamanville)
- German wind/solar variability, gas storage levels, coal-to-gas switching
- CO2 ETS price pass-through to power prices
- Interconnector flows, import/export balances, and congestion
- Seasonal demand patterns, weather effects, heating/cooling degree days
- UK gas peakers, Dutch TTF gas hub, Nordic hydro balance

Your role: generate realistic synthetic energy news articles for model evaluation.

Requirements per article:
- Title: ~10 words, headline-style, specific and factual in tone
- Summary: 50–80 words, one paragraph, cite specific numbers (GW, EUR/MWh, %) where natural
- The sentiment for NEXT-DAY day-ahead electricity prices must be unambiguous
- Ground every scenario in a real market mechanism; use plausible company names
  (EDF, RWE, E.ON, Engie, National Grid, TenneT, Statoil/Equinor, Gazprom)
"""

ARTICLE_TOOL = {
    "name": "generate_labeled_article",
    "description": "Generate one labeled energy news article for evaluation purposes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Unique article ID following pattern cj_{sentiment}_{country}_{nn}",
            },
            "title": {
                "type": "string",
                "description": "Headline, ~10 words, headline-style",
            },
            "summary": {
                "type": "string",
                "description": "Article body, 50–80 words",
            },
            "country": {
                "type": "string",
                "enum": ["DE_LU", "FR", "NL", "GB", "EU"],
                "description": "Primary market affected by this news",
            },
            "date": {
                "type": "string",
                "description": "ISO date YYYY-MM-DD — should look realistic and recent",
            },
            "label": {
                "type": "integer",
                "enum": [-1, 0, 1],
                "description": "-1 = bearish, 0 = neutral, +1 = bullish for next-day day-ahead price",
            },
            "label_reasoning": {
                "type": "string",
                "description": "One sentence: the specific market mechanism that drives the label",
            },
        },
        "required": ["id", "title", "summary", "country", "date", "label", "label_reasoning"],
    },
}

LABEL_SCENARIOS = {
    -1: {
        "name": "bearish",
        "directive": (
            "Generate a BEARISH article (label = -1). "
            "The article must describe a scenario that will push next-day day-ahead electricity prices DOWN. "
            "Valid bearish drivers: new generation coming online, mild/warm weather reducing heating demand, "
            "gas price drop, nuclear reactor restart, high wind/solar forecast, demand-side response program, "
            "unexpected demand softness, grid oversupply, regulatory loosening on supply."
        ),
    },
    0: {
        "name": "neutral",
        "directive": (
            "Generate a NEUTRAL article (label = 0). "
            "The article must describe a scenario with NO clear directional signal for next-day prices. "
            "Valid neutral scenarios: administrative appointment, corporate restructuring with no output change, "
            "long-term policy still under consultation, M&A/financing news, ESG rating changes, "
            "multi-year infrastructure projects with no near-term grid impact."
        ),
    },
    1: {
        "name": "bullish",
        "directive": (
            "Generate a BULLISH article (label = +1). "
            "The article must describe a scenario that will push next-day day-ahead electricity prices UP. "
            "Valid bullish drivers: unplanned generation outage, cold snap boosting heating demand, "
            "gas price surge, nuclear reactor trip, low wind/solar forecast, import constraint, "
            "CO2 price jump, supply disruption, LNG terminal issue."
        ),
    },
}

COUNTRIES = ["DE_LU", "FR", "NL", "GB", "EU"]


def _article_date(offset_days: int) -> str:
    return (date(2026, 4, 30) - timedelta(days=offset_days)).isoformat()


def _generate_article(
    client: anthropic.Anthropic,
    label: int,
    country: str,
    seq: int,
) -> dict:
    scenario = LABEL_SCENARIOS[label]

    user_message = (
        f"{scenario['directive']}\n\n"
        f"Country: {country}\n"
        f"Article ID: cj_{scenario['name']}_{country.lower()}_{seq:02d}\n"
        f"Date: {_article_date(seq * 3 + (label + 1) * 10)}\n\n"
        f"The label field MUST be {label}. Make the directional signal clear."
    )

    with budget.track(label=f"corpus_judge_{scenario['name']}", model="claude-opus-4-7") as record:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=600,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
            tools=[ARTICLE_TOOL],
            tool_choice={"type": "tool", "name": "generate_labeled_article"},
        )
        record.input_tokens = response.usage.input_tokens
        record.output_tokens = response.usage.output_tokens

    tool_block = next(b for b in response.content if b.type == "tool_use")
    return tool_block.input


def _check_api_key() -> None:
    """Fail early with a clear message if ANTHROPIC_API_KEY is not set."""
    import os
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key.startswith("your_"):
        raise SystemExit(
            "\n❌  ANTHROPIC_API_KEY is not set.\n"
            "   Add it to .env:  ANTHROPIC_API_KEY=sk-ant-...\n"
            "   Get a key at:    https://console.anthropic.com/settings/keys\n"
        )


def generate_sentiment_gold(
    n: int = 30,
    output_path: Path = GOLD_PATH,
    dry_run: bool = False,
) -> list[dict]:
    """Generate n labeled articles balanced across {-1, 0, +1} and COUNTRIES.

    Returns the list of gold rows (also written to output_path unless dry_run).
    """
    if not dry_run:
        _check_api_key()

    client = anthropic.Anthropic()

    per_class = n // 3
    remainder = n % 3
    distribution = {-1: per_class, 0: per_class, 1: per_class + remainder}
    total = sum(distribution.values())

    print(
        f"\nCorpus Judge {JUDGE_VERSION} — {total} articles via Claude Opus 4.7"
        f"  bearish={distribution[-1]}  neutral={distribution[0]}  bullish={distribution[1]}"
    )
    if dry_run:
        print("DRY RUN — no API calls, no file writes\n")
    print(f"Output: {output_path}\n{'='*62}")

    articles: list[dict] = []
    call_num = 0

    for label in (-1, 0, 1):
        count = distribution[label]
        name = LABEL_SCENARIOS[label]["name"].upper()
        for i in range(count):
            country = COUNTRIES[i % len(COUNTRIES)]
            call_num += 1
            print(f"  [{call_num:02d}/{total}] {name:8s} | {country:5s} | ", end="", flush=True)

            if dry_run:
                print("skipped")
                articles.append(
                    {
                        "id": f"cj_{name.lower()}_{country.lower()}_{i+1:02d}",
                        "title": f"[DRY RUN] {name} for {country}",
                        "summary": "Dry run — no content generated.",
                        "country": country,
                        "date": _article_date(i * 3),
                        "label": label,
                    }
                )
                continue

            try:
                result = _generate_article(client, label, country, i + 1)
                row = {k: result[k] for k in ("id", "title", "summary", "country", "date", "label")}
                articles.append(row)
                reasoning = result.get("label_reasoning", "")
                print(f"✓  {reasoning[:58]}")
            except Exception as exc:
                print(f"✗  {exc}")

    if not dry_run and articles:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            for row in articles:
                fh.write(json.dumps(row) + "\n")
        print(f"\n✅  Wrote {len(articles)}/{total} articles → {output_path}")
        cache_hits = getattr(budget, "_last_cache_read", None)
        print(f"    Budget: ${budget.daily_cost_usd:.4f} used today  |  {budget.calls_today} calls")

    _print_human_review_tracker()
    return articles


def _print_human_review_tracker() -> None:
    items = [
        (
            "0. Set ANTHROPIC_API_KEY in .env  [BLOCKS EVERYTHING]",
            "   Get key: https://console.anthropic.com/settings/keys\n"
            "   Then add to .env:  ANTHROPIC_API_KEY=sk-ant-...\n"
            "   Then re-run:  python -m src.evaluation.corpus_judge",
            "Before corpus_judge.py can generate any articles",
        ),
        (
            "1. Audit data/eval/sentiment_gold.jsonl",
            "   LLM-generated labels = development proxy only.\n"
            "   Read all 30 rows, correct any mislabeled articles,\n"
            "   then re-run:  python -m src.evaluation.eval_sentiment",
            "Before shipping Phase 1 κ result as production-grade",
        ),
        (
            "2. Record Loom walkthrough (3 min)",
            "   README §Demo still shows <!-- Add Loom link here -->.\n"
            "   Record and paste URL into README.md.",
            "After P5.3 hits Sharpe ≥ 0.8 DoD",
        ),
    ]
    w = 60
    print(f"\n{'╔' + '═' * w + '╗'}")
    print(f"║{'  ⚠  HUMAN REVIEW REQUIRED — 3 items':^{w}}║")
    print(f"{'╠' + '═' * w + '╣'}")
    for task, detail, when in items:
        print(f"║  {task:<{w - 2}}║")
        for line in detail.splitlines():
            print(f"║{line:<{w + 1}}║")
        print(f"║  When: {when:<{w - 7}}║")
        print(f"║{' ' * w}║")
    print(f"{'╚' + '═' * w + '╝'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate gold-standard sentiment eval data using Claude Opus 4.7"
    )
    parser.add_argument("--n", type=int, default=30, help="Articles to generate (default: 30)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without API calls")
    args = parser.parse_args()

    generate_sentiment_gold(n=args.n, dry_run=args.dry_run)
