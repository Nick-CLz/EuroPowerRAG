"""Tests for the evaluation metrics."""

import pytest
from langchain_core.documents import Document

from src.evaluation.evaluator import retrieval_precision_at_k


def make_doc(text):
    return Document(page_content=text, metadata={})


def test_precision_at_k_all_relevant():
    docs = [
        make_doc("Germany electricity day-ahead price 85 EUR/MWh"),
        make_doc("German baseload price average this week"),
        make_doc("DE_LU day-ahead market clearing price"),
    ]
    score = retrieval_precision_at_k(docs, keywords=["germany", "price"], k=3)
    assert score == 1.0


def test_precision_at_k_none_relevant():
    docs = [
        make_doc("Football results from the Premier League"),
        make_doc("Weather forecast for London"),
    ]
    score = retrieval_precision_at_k(docs, keywords=["nuclear", "france"], k=2)
    assert score == 0.0


def test_precision_at_k_partial():
    docs = [
        make_doc("France nuclear output reduced"),   # relevant
        make_doc("Spain solar generation record"),   # not relevant to france/nuclear
        make_doc("French nuclear plant offline"),    # relevant
        make_doc("UK wind farm capacity"),           # not relevant
    ]
    score = retrieval_precision_at_k(docs, keywords=["france", "nuclear", "french"], k=4)
    assert score == 0.5


def test_precision_at_k_respects_k():
    """Only the first k docs should be considered."""
    docs = [
        make_doc("Irrelevant document one"),
        make_doc("Irrelevant document two"),
        make_doc("Germany price data relevant"),  # 3rd — outside k=2
    ]
    score = retrieval_precision_at_k(docs, keywords=["germany", "price"], k=2)
    assert score == 0.0


def test_precision_at_k_empty_docs():
    assert retrieval_precision_at_k([], keywords=["germany"], k=5) == 0.0


def test_precision_at_k_empty_keywords():
    docs = [make_doc("Some document")]
    assert retrieval_precision_at_k(docs, keywords=[], k=5) == 0.0


def test_precision_checks_metadata_too():
    """Keywords should also match against document metadata."""
    doc = Document(
        page_content="Average: 85.50 EUR/MWh",
        metadata={"country": "DE_LU", "source": "ENTSO-E"},
    )
    score = retrieval_precision_at_k([doc], keywords=["de_lu"], k=1)
    assert score == 1.0
