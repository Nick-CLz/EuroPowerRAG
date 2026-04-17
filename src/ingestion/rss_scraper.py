"""RSS feed scraper for European energy market news."""

import json
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

RSS_FEEDS = [
    {
        "url": "https://www.eia.gov/todayinenergy/rss.xml",
        "source": "EIA Today in Energy",
        "country": "GLOBAL",
    },
    {
        "url": "https://ember-climate.org/feed/",
        "source": "Ember Climate",
        "country": "EU",
    },
    {
        "url": "https://feeds.bbci.co.uk/news/business/rss.xml",
        "source": "BBC Business",
        "country": "GB",
    },
    {
        "url": "https://feeds.theguardian.com/theguardian/environment/energy/rss",
        "source": "Guardian Energy",
        "country": "EU",
    },
    {
        "url": "https://www.power-technology.com/feed/",
        "source": "Power Technology",
        "country": "EU",
    },
    {
        "url": "https://www.rechargenews.com/rss",
        "source": "Recharge News",
        "country": "EU",
    },
]

ENERGY_KEYWORDS = {
    "electricity", "power", "energy", "grid", "nuclear", "renewable", "wind",
    "solar", "gas", "coal", "baseload", "capacity", "megawatt", "gigawatt",
    "eur/mwh", "generation", "transmission", "entso-e", "epex", "eex",
    "european", "germany", "france", "netherlands", "britain", "spain",
}


def _is_energy_relevant(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in ENERGY_KEYWORDS)


def _parse_date(entry) -> str:
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        val = getattr(entry, attr, None)
        if val:
            try:
                return datetime(*val[:6], tzinfo=timezone.utc).strftime("%Y-%m-%d")
            except Exception:
                pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def scrape_feeds(max_per_feed: int = 25) -> list[dict]:
    documents = []

    for feed_cfg in RSS_FEEDS:
        url = feed_cfg["url"]
        source = feed_cfg["source"]
        country = feed_cfg["country"]

        try:
            feed = feedparser.parse(url)
            count = 0

            for entry in feed.entries[:max_per_feed]:
                title = getattr(entry, "title", "")
                summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
                link = getattr(entry, "link", "")

                combined = f"{title} {summary}"
                if not _is_energy_relevant(combined):
                    continue

                # Strip basic HTML tags from summary
                import re
                summary_clean = re.sub(r"<[^>]+>", " ", summary).strip()
                summary_clean = re.sub(r"\s+", " ", summary_clean)

                doc_text = (
                    f"Energy News: {title}\n\n"
                    f"{summary_clean}\n\n"
                    f"Source: {source} | Link: {link}"
                )

                documents.append(
                    {
                        "text": doc_text,
                        "metadata": {
                            "source": source,
                            "doc_type": "news",
                            "country": country,
                            "country_name": country,
                            "date": _parse_date(entry),
                            "title": title,
                            "url": link,
                        },
                    }
                )
                count += 1

            print(f"  [{source}] {count} energy articles")

        except Exception as e:
            print(f"  [RSS] Error fetching {source}: {e}")

    return documents


def run(output_dir: Path = Path("data/processed")) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("  Scraping RSS feeds...")
    docs = scrape_feeds()
    print(f"  → {len(docs)} news documents (energy-filtered)")

    out_path = output_dir / "news.jsonl"
    with open(out_path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    print(f"  Saved {len(docs)} news documents → {out_path}")
    return len(docs)
