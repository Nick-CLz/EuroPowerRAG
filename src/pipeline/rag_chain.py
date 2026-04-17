"""Core RAG chain.

Retrieval: LangChain + ChromaDB with metadata filtering.
Generation: Anthropic claude-sonnet-4-6 via direct SDK call with prompt caching.

Prompt caching is applied to the system instructions, which are identical across
all queries. This cuts cost and latency on repeated questions.
"""

import os
from dataclasses import dataclass, field

import anthropic
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.pipeline.chunker import CHROMA_DIR, COLLECTION_NAME

load_dotenv()

SYSTEM_PROMPT = """You are an expert European power market analyst supporting commodity traders and analysts.

Your job is to answer questions using ONLY the market data and news provided in the context below.
Always cite the source and date for every claim you make.

Rules:
- Ground every statement in the provided context. Do not speculate beyond it.
- Use specific numbers (prices, volumes, percentages) wherever available.
- If the context is insufficient to answer, say so clearly.
- Format your answer in clear paragraphs. Start with the direct answer, then supporting detail.
- At the end, list the sources you used under a "Sources" heading."""

_anthropic_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _anthropic_client


def _get_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def _build_where_clause(
    country_filter: list[str] | None,
    doc_type_filter: list[str] | None,
    date_from: str | None,
    date_to: str | None,
) -> dict | None:
    conditions = []
    if country_filter:
        conditions.append({"country": {"$in": country_filter}})
    if doc_type_filter:
        conditions.append({"doc_type": {"$in": doc_type_filter}})
    if date_from:
        conditions.append({"date": {"$gte": date_from}})
    if date_to:
        conditions.append({"date": {"$lte": date_to}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        source_line = (
            f"[Doc {i}] Source: {m.get('source', 'Unknown')} | "
            f"Type: {m.get('doc_type', '?')} | "
            f"Country: {m.get('country_name', m.get('country', '?'))} | "
            f"Date: {m.get('date', 'Unknown')}"
        )
        parts.append(f"{source_line}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _generate(question: str, context_docs: list[Document]) -> str:
    context_text = _format_context(context_docs)
    user_message = f"Context documents:\n\n{context_text}\n\n---\n\nQuestion: {question}"

    client = _get_client()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        # Cache the system prompt — it's constant across all queries
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


@dataclass
class RAGResult:
    answer: str
    sources: list[Document]
    query: str
    n_docs_retrieved: int = field(default=0)

    def __post_init__(self):
        self.n_docs_retrieved = len(self.sources)


def query(
    question: str,
    country_filter: list[str] | None = None,
    doc_type_filter: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    n_results: int = 5,
) -> RAGResult:
    """Retrieve relevant documents and generate an answer with citations."""
    vectorstore = _get_vectorstore()

    where = _build_where_clause(country_filter, doc_type_filter, date_from, date_to)
    search_kwargs: dict = {"k": n_results}
    if where:
        search_kwargs["filter"] = where

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(question)

    if not docs:
        return RAGResult(
            answer="No relevant documents found in the index. Try broadening your filters or running ingestion first.",
            sources=[],
            query=question,
        )

    answer = _generate(question, docs)
    return RAGResult(answer=answer, sources=docs, query=question)
