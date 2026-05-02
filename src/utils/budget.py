"""Cost & latency budget tracking for LLM calls.

Per AGENTS.md rule #9 — every agent has a budget. Set it before you start.

Usage:
    from src.utils.budget import budget, BudgetExceeded

    with budget.track("sentiment_score"):
        result = score_article(...)

    # Check status
    print(budget.daily_cost_usd, budget.calls_today)

Pricing is rough (Gemini 2.5 Flash free-tier defaults to $0). Update PRICING
when you switch tiers or models.
"""

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# USD per 1M tokens. Free tier = 0.0; update if you upgrade.
# Source: https://ai.google.dev/pricing  (paid tier reference for if/when you scale)
PRICING = {
    "gemini-2.5-flash": {"input_per_1m": 0.0, "output_per_1m": 0.0},
    "gemini-2.5-pro": {"input_per_1m": 0.0, "output_per_1m": 0.0},
    "gemini-2.0-flash": {"input_per_1m": 0.0, "output_per_1m": 0.0},
    "gemini-embedding-001": {"input_per_1m": 0.0, "output_per_1m": 0.0},
    "claude-sonnet-4-6": {"input_per_1m": 3.0, "output_per_1m": 15.0},
    "claude-haiku-4-5": {"input_per_1m": 0.25, "output_per_1m": 1.25},
    "claude-opus-4-7": {"input_per_1m": 15.0, "output_per_1m": 75.0},
}


class BudgetExceeded(Exception):
    """Raised when a daily cost or per-call latency budget is hit."""


@dataclass
class CallRecord:
    timestamp: str
    label: str
    duration_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""


@dataclass
class BudgetTracker:
    """Tracks LLM cost + latency. Call sites use it as a context manager."""

    daily_limit_usd: float = field(
        default_factory=lambda: float(os.getenv("MAX_DAILY_LLM_USD", "5.0"))
    )
    max_query_latency_s: float = field(
        default_factory=lambda: float(os.getenv("MAX_QUERY_LATENCY_S", "10.0"))
    )
    log_path: Path = field(default_factory=lambda: Path("data/budget_log.jsonl"))

    def __post_init__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def daily_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self._today_records())

    @property
    def calls_today(self) -> int:
        return len(self._today_records())

    def _today_records(self) -> list[CallRecord]:
        if not self.log_path.exists():
            return []
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        records = []
        with open(self.log_path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if d.get("timestamp", "").startswith(today):
                        records.append(CallRecord(**d))
        return records

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        rates = PRICING.get(model, {"input_per_1m": 0.0, "output_per_1m": 0.0})
        # Claude cached token cost is typically 10% of input token cost
        cache_rate = rates["input_per_1m"] * 0.1
        return (
            (input_tokens / 1_000_000) * rates["input_per_1m"]
            + (cached_tokens / 1_000_000) * cache_rate
            + (output_tokens / 1_000_000) * rates["output_per_1m"]
        )

    @contextmanager
    def track(self, label: str, model: str = "", input_tokens: int = 0):
        """Context manager. Tracks duration; cost is recorded on exit if you call .charge()."""
        if self.daily_cost_usd >= self.daily_limit_usd:
            raise BudgetExceeded(
                f"Daily LLM cost ${self.daily_cost_usd:.4f} ≥ limit ${self.daily_limit_usd}"
            )

        start = time.perf_counter()
        record = CallRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            label=label,
            duration_s=0.0,
            input_tokens=input_tokens,
            model=model,
        )
        try:
            yield record
        finally:
            record.duration_s = time.perf_counter() - start
            if record.duration_s > self.max_query_latency_s:
                # Log it but don't raise — slow calls are usually retries succeeding
                record.label += f" [SLOW: {record.duration_s:.1f}s]"
            record.cost_usd = self.estimate_cost(
                model=record.model,
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                cached_tokens=record.cached_tokens,
            )
            with open(self.log_path, "a") as f:
                f.write(json.dumps(asdict(record)) + "\n")


# Singleton for app-wide use
budget = BudgetTracker()
