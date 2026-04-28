"""Tests for the RSS scraper."""

import pytest

from src.ingestion.rss_scraper import _is_energy_relevant, _parse_date


def test_energy_relevant_with_keyword():
    assert _is_energy_relevant("German electricity prices hit record high")


def test_energy_relevant_with_country():
    assert _is_energy_relevant("European grid stability concerns grow")


def test_energy_relevant_case_insensitive():
    assert _is_energy_relevant("NUCLEAR power output declines in France")


def test_not_energy_relevant():
    assert not _is_energy_relevant("Football match results from the weekend")


def test_not_energy_relevant_generic_business():
    assert not _is_energy_relevant("Stock market closes higher on positive earnings")


def test_parse_date_fallback():
    """Returns today's date string when no date is present on the entry."""
    from datetime import datetime

    class FakeEntry:
        published_parsed = None
        updated_parsed = None
        created_parsed = None

    date_str = _parse_date(FakeEntry())
    # Should be a valid YYYY-MM-DD string
    datetime.strptime(date_str, "%Y-%m-%d")


def test_parse_date_from_published():
    import time

    class FakeEntry:
        published_parsed = time.strptime("2024-03-15", "%Y-%m-%d")
        updated_parsed = None
        created_parsed = None

    assert _parse_date(FakeEntry()) == "2024-03-15"
