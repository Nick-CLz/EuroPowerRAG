"""PDF report loader.

Loads energy market reports (PDFs) from:
  1. Local files placed in data/raw/pdfs/
  2. URLs configured in PDF_URLS below

To add reports: either drop .pdf files into data/raw/pdfs/ or add URLs to PDF_URLS.
"""

import json
import tempfile
from pathlib import Path

import requests

LOCAL_PDF_DIR = Path("data/raw/pdfs")

# Public energy market PDFs — add your own here
PDF_URLS: list[dict] = [
    # Example format:
    # {
    #     "url": "https://example.com/report.pdf",
    #     "source": "ACER",
    #     "country": "EU",
    #     "doc_type": "report",
    # }
]

HEADERS = {"User-Agent": "EuroPowerRAG/1.0 (research tool)"}


def _load_pdf_file(pdf_path: Path, source: str, country: str = "EU") -> list[dict]:
    from langchain_community.document_loaders import PyPDFLoader

    try:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        docs = []
        for page in pages:
            text = page.page_content.strip()
            if len(text) < 100:
                continue
            docs.append(
                {
                    "text": text,
                    "metadata": {
                        "source": source,
                        "doc_type": "report",
                        "country": country,
                        "country_name": country,
                        "date": "",
                        "page": page.metadata.get("page", 0),
                        "filename": pdf_path.name,
                    },
                }
            )
        return docs
    except Exception as e:
        print(f"  [PDF] Error loading {pdf_path.name}: {e}")
        return []


def load_local_pdfs() -> list[dict]:
    LOCAL_PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = list(LOCAL_PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("  No PDFs in data/raw/pdfs/ — drop PDF reports there to include them")
        return []

    docs = []
    for pdf_path in pdf_files:
        source = pdf_path.stem.replace("_", " ").replace("-", " ").title()
        file_docs = _load_pdf_file(pdf_path, source=source)
        print(f"  [{pdf_path.name}] {len(file_docs)} pages loaded")
        docs.extend(file_docs)
    return docs


def download_and_load_pdfs() -> list[dict]:
    if not PDF_URLS:
        return []

    docs = []
    for cfg in PDF_URLS:
        url = cfg["url"]
        source = cfg.get("source", "Unknown")
        country = cfg.get("country", "EU")

        try:
            resp = requests.get(url, timeout=60, headers=HEADERS)
            resp.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = Path(tmp.name)

            file_docs = _load_pdf_file(tmp_path, source=source, country=country)
            for doc in file_docs:
                doc["metadata"]["url"] = url

            print(f"  [{source}] {len(file_docs)} pages downloaded and loaded")
            docs.extend(file_docs)
            tmp_path.unlink(missing_ok=True)

        except Exception as e:
            print(f"  [PDF] Error downloading {source} ({url}): {e}")

    return docs


def run(output_dir: Path = Path("data/processed")) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("  Loading local PDFs...")
    local_docs = load_local_pdfs()

    print("  Downloading configured PDFs...")
    remote_docs = download_and_load_pdfs()

    all_docs = local_docs + remote_docs
    if not all_docs:
        print("  → No PDF documents loaded")
        return 0

    out_path = output_dir / "pdfs.jsonl"
    with open(out_path, "w") as f:
        for doc in all_docs:
            f.write(json.dumps(doc) + "\n")

    print(f"  Saved {len(all_docs)} PDF documents → {out_path}")
    return len(all_docs)
