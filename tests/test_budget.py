"""Tests for budget tracker."""

import json
from pathlib import Path

import pytest

from src.utils.budget import BudgetExceeded, BudgetTracker, CallRecord


@pytest.fixture
def tmp_tracker(tmp_path):
    return BudgetTracker(
        daily_limit_usd=1.0,
        max_query_latency_s=10.0,
        log_path=tmp_path / "budget_log.jsonl",
    )


def test_track_records_one_entry(tmp_tracker):
    with tmp_tracker.track(label="test_call", model="gemini-2.5-flash") as r:
        assert isinstance(r, CallRecord)
    assert tmp_tracker.calls_today == 1


def test_track_writes_jsonl(tmp_tracker):
    with tmp_tracker.track(label="x", model="gemini-2.5-flash"):
        pass
    contents = tmp_tracker.log_path.read_text().strip().splitlines()
    assert len(contents) == 1
    record = json.loads(contents[0])
    assert record["label"] == "x"
    assert record["model"] == "gemini-2.5-flash"
    assert record["duration_s"] >= 0


def test_estimate_cost_zero_for_free_tier(tmp_tracker):
    assert tmp_tracker.estimate_cost("gemini-2.5-flash", 1000, 1000) == 0.0


def test_estimate_cost_unknown_model_zero(tmp_tracker):
    assert tmp_tracker.estimate_cost("unknown-model", 999_999, 999_999) == 0.0


def test_budget_exceeded_blocks_new_calls(tmp_path):
    """If a record above the limit is in the log, the next track call must raise."""
    log_path = tmp_path / "budget_log.jsonl"
    # Pre-seed log with a record exceeding the limit
    from datetime import datetime, timezone

    today_iso = datetime.now(timezone.utc).isoformat()
    with open(log_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": today_iso,
                    "label": "huge",
                    "duration_s": 0.1,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 100.0,  # over limit
                    "model": "gemini-2.5-flash",
                }
            )
            + "\n"
        )

    tracker = BudgetTracker(daily_limit_usd=1.0, log_path=log_path)
    with pytest.raises(BudgetExceeded):
        with tracker.track(label="should_block"):
            pass


def test_calls_yesterday_dont_count_today(tmp_path):
    log_path = tmp_path / "budget_log.jsonl"
    with open(log_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": "1970-01-01T00:00:00+00:00",
                    "label": "ancient",
                    "duration_s": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "model": "gemini-2.5-flash",
                }
            )
            + "\n"
        )

    tracker = BudgetTracker(daily_limit_usd=1.0, log_path=log_path)
    assert tracker.calls_today == 0
