"""Daily sentiment aggregation.

P1.5 — turns per-article scores into a daily time series joinable with prices.

Reads:  data/processed/news.jsonl  (with sentiment_score in metadata)
Writes: data/processed/sentiment_daily.parquet
"""

import json
from pathlib import Path

import pandas as pd

NEWS_PATH = Path("data/processed/news.jsonl")
OUTPUT_PATH = Path("data/processed/sentiment_daily.parquet")


def load_scored_news() -> pd.DataFrame:
    if not NEWS_PATH.exists():
        raise FileNotFoundError(f"{NEWS_PATH} not found — run ingestion first")

    rows = []
    with open(NEWS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            meta = doc.get("metadata", {})
            score = meta.get("sentiment_score")
            if score is None:
                continue  # Article wasn't scored — skip
            rows.append(
                {
                    "date": meta.get("date"),
                    "country": meta.get("country", "EU"),
                    "score": float(score),
                    "confidence": float(meta.get("sentiment_confidence", 0.5)),
                }
            )

    return pd.DataFrame(rows)


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, country) — confidence-weighted mean + dispersion."""
    if df.empty:
        return df

    df = df.copy()
    df["weighted"] = df["score"] * df["confidence"]

    grouped = df.groupby(["date", "country"]).agg(
        n_articles=("score", "count"),
        mean_score=("score", "mean"),
        weighted_score=("weighted", "sum"),
        weight_sum=("confidence", "sum"),
        std_score=("score", "std"),
    ).reset_index()

    # Confidence-weighted mean (handles zero-confidence articles cleanly)
    grouped["mean_weighted_score"] = grouped["weighted_score"] / grouped["weight_sum"].replace(0, 1)
    grouped["std_score"] = grouped["std_score"].fillna(0.0)
    return grouped[
        ["date", "country", "n_articles", "mean_score", "mean_weighted_score", "std_score"]
    ]


def run() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_scored_news()
    if df.empty:
        print("  No scored news articles — run ingestion + scoring first.")
        return 0

    daily = aggregate_daily(df)
    daily.to_parquet(OUTPUT_PATH, index=False)

    print(f"  [sentiment] {len(daily)} (date, country) rows → {OUTPUT_PATH}")
    print(f"  Coverage: {daily['country'].nunique()} countries, "
          f"{daily['date'].min()} → {daily['date'].max()}")
    return len(daily)


if __name__ == "__main__":
    run()
