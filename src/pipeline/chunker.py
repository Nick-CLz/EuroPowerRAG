"""Load processed documents from JSONL files, chunk them, and index into ChromaDB."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

load_dotenv()

PROCESSED_DIR = Path("data/processed")
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
COLLECTION_NAME = "europowerrag"

# Chunk sizes tuned per document type:
#   Structured ENTSO-E records are already compact — small chunks preserve atomic facts.
#   News articles benefit from slightly larger chunks to keep sentence context together.
CHUNK_CONFIG = {
    "price_data": {"chunk_size": 300, "chunk_overlap": 30},
    "generation_data": {"chunk_size": 300, "chunk_overlap": 30},
    "news": {"chunk_size": 600, "chunk_overlap": 80},
    "report": {"chunk_size": 800, "chunk_overlap": 100},
}
DEFAULT_CHUNK = {"chunk_size": 500, "chunk_overlap": 50}


def _get_splitter(doc_type: str) -> RecursiveCharacterTextSplitter:
    cfg = CHUNK_CONFIG.get(doc_type, DEFAULT_CHUNK)
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        length_function=len,
    )


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def load_jsonl_files(processed_dir: Path = PROCESSED_DIR) -> list[dict]:
    raw = []
    for jsonl_path in sorted(processed_dir.glob("*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))
    return raw


def build_chunks(raw_docs: list[dict]) -> list[Document]:
    chunks: list[Document] = []
    for i, raw in enumerate(raw_docs):
        text = raw.get("text", "")
        meta = raw.get("metadata", {})
        doc_type = meta.get("doc_type", "unknown")
        splitter = _get_splitter(doc_type)

        sub_docs = splitter.create_documents([text], metadatas=[meta])
        for j, sub in enumerate(sub_docs):
            # Stable chunk ID enables upserts without duplicating on re-index
            sub.metadata["chunk_id"] = f"doc{i}_chunk{j}"
        chunks.extend(sub_docs)

    return chunks


def get_vectorstore(embeddings=None) -> Chroma:
    if embeddings is None:
        embeddings = _get_embeddings()
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def index_documents(chunks: list[Document], batch_size: int = 100) -> Chroma:
    embeddings = _get_embeddings()
    vectorstore = get_vectorstore(embeddings)

    print(f"  Indexing {len(chunks)} chunks into ChromaDB (batch_size={batch_size})...")
    for i in tqdm(range(0, len(chunks), batch_size), desc="  Embedding"):
        batch = chunks[i : i + batch_size]
        ids = [doc.metadata["chunk_id"] for doc in batch]
        vectorstore.add_documents(batch, ids=ids)

    return vectorstore


def run(processed_dir: Path = PROCESSED_DIR) -> int:
    print("  Loading processed documents from JSONL files...")
    raw_docs = load_jsonl_files(processed_dir)
    print(f"  → {len(raw_docs)} raw documents loaded")

    if not raw_docs:
        print("  No documents found — run ingest.py first")
        return 0

    print("  Chunking documents...")
    chunks = build_chunks(raw_docs)
    print(f"  → {len(chunks)} chunks created")

    index_documents(chunks)
    print(f"  ChromaDB index saved to {CHROMA_DIR}")
    return len(chunks)
