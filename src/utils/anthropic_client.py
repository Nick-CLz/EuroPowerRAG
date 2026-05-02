"""Anthropic client wrapper with prompt caching (T3).

Every Claude call in the project routes through this module to guarantee:
  - System-prompt caching via ``cache_control: {"type": "ephemeral"}``
  - Budget tracking via ``src.utils.budget``
  - Automatic retry on transient 429 / 529 errors
  - A single shared client instance (connection pooling)

Usage::

    from src.utils.anthropic_client import call_claude

    answer = call_claude(
        model="claude-haiku-4-5",
        system_prompt="You are a helpful assistant.",
        messages=[{"role": "user", "content": "Hello"}],
        label="my_feature",
    )
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from anthropic import Anthropic, APIError, RateLimitError
from dotenv import load_dotenv

from src.utils.budget import budget

load_dotenv()

log = logging.getLogger(__name__)

__all__ = ["call_claude"]

# ── Singleton client ─────────────────────────────────────────────────────────
_client: Anthropic | None = None


def _get_client() -> Anthropic:
    """Return a shared Anthropic client (creates once, reuses thereafter)."""
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key.startswith("your_"):
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Get one at https://console.anthropic.com/ and add it to .env"
            )
        _client = Anthropic(api_key=api_key)
    return _client


# ── Retry config ─────────────────────────────────────────────────────────────
_MAX_RETRIES = 3
_RETRY_BACKOFF_S = (1.0, 3.0, 8.0)


def call_claude(
    model: str,
    system_prompt: str,
    messages: list[dict[str, Any]],
    *,
    label: str = "claude_call",
    max_tokens: int = 1_024,
    temperature: float = 0.0,
) -> str:
    """Execute a Claude call with prompt caching and budget tracking.

    Args:
        model:         Anthropic model identifier (e.g. ``claude-haiku-4-5``).
        system_prompt: System-level instructions. Automatically wrapped with
                       ``cache_control`` for prompt caching.
        messages:      Standard Anthropic ``messages`` list.
        label:         Human-readable label for the budget log.
        max_tokens:    Maximum tokens in the response.
        temperature:   Sampling temperature (0 = deterministic).

    Returns:
        The text content of the first response block.

    Raises:
        EnvironmentError:  if ``ANTHROPIC_API_KEY`` is missing.
        BudgetExceeded:    if the daily spend ceiling has been reached.
        anthropic.APIError: after exhausting retries on transient errors.
    """
    client = _get_client()

    system_block = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    last_error: Exception | None = None

    with budget.track(label=label, model=model) as record:
        for attempt in range(_MAX_RETRIES):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_block,
                    messages=messages,
                )

                # Record token usage
                record.input_tokens = response.usage.input_tokens
                record.output_tokens = response.usage.output_tokens
                record.cached_tokens = getattr(
                    response.usage, "cache_read_input_tokens", 0
                )

                log.debug(
                    "Claude %s  in=%d  out=%d  cached=%d  attempt=%d",
                    model,
                    record.input_tokens,
                    record.output_tokens,
                    record.cached_tokens,
                    attempt + 1,
                )
                return response.content[0].text

            except RateLimitError as exc:
                last_error = exc
                wait = _RETRY_BACKOFF_S[min(attempt, len(_RETRY_BACKOFF_S) - 1)]
                log.warning("Rate-limited (attempt %d/%d), retrying in %.1fs", attempt + 1, _MAX_RETRIES, wait)
                time.sleep(wait)

            except APIError as exc:
                # 529 = overloaded — retry; other API errors are fatal
                if exc.status_code == 529:
                    last_error = exc
                    wait = _RETRY_BACKOFF_S[min(attempt, len(_RETRY_BACKOFF_S) - 1)]
                    log.warning("API overloaded (attempt %d/%d), retrying in %.1fs", attempt + 1, _MAX_RETRIES, wait)
                    time.sleep(wait)
                else:
                    raise

        # All retries exhausted
        raise last_error  # type: ignore[misc]
