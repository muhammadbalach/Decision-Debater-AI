"""CLI document ingestion: docs/ -> chunk -> embed (BGE) -> persist to ChromaDB.

Usage:
    python ingest.py

Place PDFs / .txt / .md files under docs/. Sub-folders become the `domain` metadata
tag (e.g. docs/career/foo.pdf -> domain="career"), enabling optional domain filtering
at retrieval time. Files directly under docs/ are tagged domain="general".
"""
import config

import glob
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from retrieval import get_vectorstore


def _domain_for(path: str, docs_dir: str) -> str:
    rel = os.path.relpath(path, docs_dir)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) > 1 else "general"


def load_documents(docs_dir: str):
    documents = []
    for pattern in ("**/*.pdf", "**/*.txt", "**/*.md"):
        for path in glob.glob(os.path.join(docs_dir, pattern), recursive=True):
            try:
                if path.lower().endswith(".pdf"):
                    loaded = PyPDFLoader(path).load()
                else:
                    loaded = TextLoader(path, encoding="utf-8").load()
            except Exception as exc:
                print(f"  ! skipped {path}: {exc}")
                continue
            domain = _domain_for(path, docs_dir)
            for doc in loaded:
                doc.metadata["source"] = os.path.basename(path)
                doc.metadata["domain"] = domain
            documents.extend(loaded)
            print(f"  + {path} (domain={domain}, {len(loaded)} page(s))")
    return documents


def main():
    documents = load_documents(config.DOCS_DIR)
    if not documents:
        print(
            f"No documents found in {config.DOCS_DIR}/. "
            "Add PDFs or .txt/.md files (optionally in domain sub-folders) and re-run."
        )
        return

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    total = len(chunks)
    print(f"\nLoaded {len(documents)} page(s) -> {total} chunk(s).")

    # Ingestion is a standalone job, so use more CPU threads for embedding speed.
    # (The running app keeps OMP single-threaded for stability; this process is separate.)
    try:
        import torch

        torch.set_num_threads(max(1, os.cpu_count() or 2))
    except Exception:
        pass

    vectorstore = get_vectorstore()

    # Clear any existing chunks so re-running ingestion rebuilds cleanly (no duplicates).
    try:
        existing = vectorstore.get(include=[])
        ids = existing.get("ids") if existing else None
        if ids:
            vectorstore.delete(ids=ids)
            print(f"Cleared {len(ids)} existing chunk(s) for a clean rebuild.")
    except Exception as exc:
        print(f"(could not clear existing store: {exc})")

    batch = config.INGEST_BATCH_SIZE
    print(f"Embedding {total} chunk(s) in batches of {batch} "
          "(first run also downloads/loads the BGE model)...")
    for i in range(0, total, batch):
        vectorstore.add_documents(chunks[i : i + batch])
        done = min(i + batch, total)
        print(f"  embedded {done}/{total} ({done * 100 // total}%)", flush=True)

    print(
        f"Done. Ingested {total} chunk(s) into {config.CHROMA_DIR} "
        f"(collection '{config.CHROMA_COLLECTION}')."
    )


if __name__ == "__main__":
    main()
