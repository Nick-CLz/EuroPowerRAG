"""Tests for the document chunker."""

import pytest
from langchain_core.documents import Document

from src.pipeline.chunker import _get_splitter, build_chunks


def test_splitter_returns_correct_type():
    splitter = _get_splitter("price_data")
    assert splitter is not None


def test_price_data_chunk_size():
    """Price data uses smaller chunks to preserve atomic facts."""
    splitter = _get_splitter("price_data")
    assert splitter._chunk_size == 300


def test_news_chunk_size():
    """News uses larger chunks to keep sentence context together."""
    splitter = _get_splitter("news")
    assert splitter._chunk_size == 600


def test_report_chunk_size():
    """Reports use the largest chunks."""
    splitter = _get_splitter("report")
    assert splitter._chunk_size == 800


def test_unknown_doc_type_uses_default():
    splitter = _get_splitter("unknown_type")
    assert splitter._chunk_size == 500


def test_build_chunks_basic():
    raw_docs = [
        {
            "text": "Day-ahead electricity prices for Germany on 2024-01-01: Average: 85.50 EUR/MWh",
            "metadata": {
                "source": "ENTSO-E",
                "doc_type": "price_data",
                "country": "DE_LU",
                "country_name": "Germany/Luxembourg",
                "date": "2024-01-01",
            },
        }
    ]
    chunks = build_chunks(raw_docs)
    assert len(chunks) >= 1
    assert isinstance(chunks[0], Document)


def test_build_chunks_preserves_metadata():
    raw_docs = [
        {
            "text": "Some energy news article about European electricity markets.",
            "metadata": {
                "source": "BBC",
                "doc_type": "news",
                "country": "GB",
                "country_name": "Great Britain",
                "date": "2024-01-15",
            },
        }
    ]
    chunks = build_chunks(raw_docs)
    assert chunks[0].metadata["source"] == "BBC"
    assert chunks[0].metadata["country"] == "GB"
    assert chunks[0].metadata["date"] == "2024-01-15"


def test_build_chunks_assigns_chunk_ids():
    raw_docs = [
        {
            "text": "Test document for chunk ID assignment.",
            "metadata": {"doc_type": "news", "source": "Test"},
        }
    ]
    chunks = build_chunks(raw_docs)
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_id"].startswith("doc0_chunk")


def test_build_chunks_empty_input():
    assert build_chunks([]) == []


def test_build_chunks_multiple_docs():
    raw_docs = [
        {"text": "First document.", "metadata": {"doc_type": "news", "source": "A"}},
        {"text": "Second document.", "metadata": {"doc_type": "price_data", "source": "B"}},
    ]
    chunks = build_chunks(raw_docs)
    # Chunk IDs from different source docs should not collide
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
