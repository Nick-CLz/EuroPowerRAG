"""Scheduled ingestion using APScheduler.

Runs a full ingest + re-index every day at 07:00 UTC (after ENTSO-E
publishes new day-ahead prices for D+1, typically by 13:00 CET).

Usage:
    python scheduler.py          # start scheduler (blocking)
    python scheduler.py --once   # run ingestion once and exit
"""

import argparse
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def run_ingestion():
    from ingest import main
    log.info("Scheduled ingestion starting...")
    rc = main()
    if rc == 0:
        log.info("Scheduled ingestion completed successfully")
    else:
        log.error("Scheduled ingestion finished with errors (rc=%d)", rc)


def start_scheduler():
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        run_ingestion,
        trigger=CronTrigger(hour=7, minute=0),
        id="daily_ingest",
        name="Daily ENTSO-E + News Ingestion",
        replace_existing=True,
    )

    log.info("Scheduler started — ingestion runs daily at 07:00 UTC")
    log.info("Press Ctrl+C to stop")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("Scheduler stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EuroPowerRAG ingestion scheduler")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    if args.once:
        run_ingestion()
        sys.exit(0)
    else:
        start_scheduler()
