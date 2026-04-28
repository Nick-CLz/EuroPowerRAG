"""Tests for the RAG chain (metadata filtering and context formatting)."""

import pytest
from langchain_core.documents import Document

from src.pipeline.rag_chain import _build_where_clause, _format_context


# ── Metadata filter builder ───────────────────────────────────────────────────

def test_no_filters_returns_none():
    assert _build_where_clause(None, None, None, None) is None


def test_single_country_filter():
    result = _build_where_clause(["DE_LU"], None, None, None)
    assert result == {"country": {"$in": ["DE_LU"]}}


def test_single_doc_type_filter():
    result = _build_where_clause(None, ["price_data"], None, None)
    assert result == {"doc_type": {"$in": ["price_data"]}}


def test_date_range_filter():
    result = _build_where_clause(None, None, "2024-01-01", "2024-01-31")
    assert result == {
        "$and": [
            {"date": {"$gte": "2024-01-01"}},
            {"date": {"$lte": "2024-01-31"}},
        ]
    }


def test_combined_filters_use_and():
    result = _build_where_clause(["FR"], ["generation_data"], "2024-01-01", None)
    assert result is not None
    assert "$and" in result
    conditions = result["$and"]
    assert {"country": {"$in": ["FR"]}} in conditions
    assert {"doc_type": {"$in": ["generation_data"]}} in conditions
    assert {"date": {"$gte": "2024-01-01"}} in conditions


def test_multiple_countries():
    result = _build_where_clause(["DE_LU", "FR", "NL"], None, None, None)
    assert result == {"country": {"$in": ["DE_LU", "FR", "NL"]}}


# ── Context formatter ─────────────────────────────────────────────────────────

def make_doc(text, source="ENTSO-E", doc_type="price_data", country="DE_LU", date="2024-01-01"):
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "doc_type": doc_type,
            "country": country,
            "country_name": "Germany/Luxembourg",
            "date": date,
        },
    )


def test_format_context_single_doc():
    docs = [make_doc("Average price: 85.50 EUR/MWh")]
    context = _format_context(docs)
    assert "[Doc 1]" in context
    assert "ENTSO-E" in context
    assert "85.50 EUR/MWh" in context


def test_format_context_multiple_docs():
    docs = [make_doc("Price data"), make_doc("Generation data", doc_type="generation_data")]
    context = _format_context(docs)
    assert "[Doc 1]" in context
    assert "[Doc 2]" in context
    assert "---" in context  # separator between docs


def test_format_context_empty():
    assert _format_context([]) == ""


def test_format_context_includes_date():
    docs = [make_doc("Some text", date="2024-06-15")]
    context = _format_context(docs)
    assert "2024-06-15" in context
