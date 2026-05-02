"""Main ingestion entry point.

Runs all three ingestion sources in sequence, then chunks and indexes
the output into ChromaDB.

Usage:
    python ingest.py            # full run
    python ingest.py --no-index # ingest only, skip embedding
"""

import argparse
import sys
import time
from pathlib import Path

from src.ingestion import entsoe_client, pdf_loader, rss_scraper
from src.sentiment import aggregator
from src.pipeline import chunker


def main(skip_index: bool = False) -> int:
    start = time.time()
    total_raw = 0

    print("\n=== EuroPowerRAG Ingestion ===\n")

    print("[1/4] ENTSO-E Transparency Platform")
    try:
        total_raw += entsoe_client.run()
    except Exception as e:
        print(f"  ENTSO-E failed: {e}")

    print("\n[2/4] RSS News Feeds")
    try:
        total_raw += rss_scraper.run()
    except Exception as e:
        print(f"  RSS failed: {e}")

    print("\n[3/4] PDF Reports")
    try:
        total_raw += pdf_loader.run()
    except Exception as e:
        print(f"  PDF failed: {e}")

    print(f"\n  Total raw documents: {total_raw}")

    if skip_index:
        print("\n[4/5] Skipping indexing (--no-index)")
    else:
        print("\n[4/5] Chunking & Indexing into ChromaDB")
        try:
            n_chunks = chunker.run()
            print(f"  Total chunks indexed: {n_chunks}")
        except Exception as e:
            print(f"  Indexing failed: {e}")
            return 1

    print("\n[5/5] Sentiment Aggregation")
    try:
        aggregator.run()
    except Exception as e:
        print(f"  Sentiment aggregation failed: {e}")

    elapsed = time.time() - start
    print(f"\n=== Done in {elapsed:.1f}s ===\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest European power market data")
    parser.add_argument("--no-index", action="store_true", help="Skip ChromaDB indexing")
    args = parser.parse_args()
    sys.exit(main(skip_index=args.no_index))
