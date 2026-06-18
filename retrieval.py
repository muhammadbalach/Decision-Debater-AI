
"""Retrieval layer: RAG (Chroma + BGE, optional reranker) and Tavily web search.

Every public function is fault-tolerant: a missing store, missing key, or upstream
error returns a human-readable note rather than raising, so the pipeline never breaks.
"""
import config

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embeddings():
    """BGE embeddings with the recommended query instruction + normalization."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True, "batch_size": config.INGEST_BATCH_SIZE},
        model_kwargs={"device": "cpu"},
    )


@lru_cache(maxsize=1)
def get_vectorstore():
    return Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_DIR,
    )


def get_retriever(domain: str | None = None):
    """Return a retriever, optionally domain-filtered and/or reranked."""
    vs = get_vectorstore()
    k = config.RETRIEVER_FETCH_K if config.USE_RERANKER else config.RETRIEVER_K
    search_kwargs = {"k": k}
    if domain:
        search_kwargs["filter"] = {"domain": domain}
    base = vs.as_retriever(search_kwargs=search_kwargs)

    if not config.USE_RERANKER:
        return base

    # Optional cross-encoder reranking for higher precision.
    # (These live under langchain_classic in LangChain v1.)
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    encoder = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=encoder, top_n=config.RERANK_TOP_N)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)


def rag_context(query: str, domain: str | None = None) -> str:
    """Retrieve grounding passages from the document store."""
    try:
        docs = get_retriever(domain).invoke(query)
    except Exception as exc:  # store missing/corrupt, reranker download failed, etc.
        return f"(RAG unavailable: {exc})"

    if not docs:
        return "(no indexed documents - run `python ingest.py` after adding files to docs/)"

    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        parts.append(f"[{src}] {d.page_content.strip()}")
    return "\n\n".join(parts)


def web_search(query: str) -> str:
    """Real-time grounding via Tavily, cleaned for LLM consumption."""
    if not config.TAVILY_API_KEY:
        return "(web search skipped: TAVILY_API_KEY not set)"
    try:
        from langchain_tavily import TavilySearch

        tool = TavilySearch(max_results=config.WEB_MAX_RESULTS, include_answer=True)
        raw = tool.invoke({"query": query})
    except Exception as exc:
        return f"(web search failed: {exc})"
    return clean_web_results(raw)


def clean_web_results(raw) -> str:
    """Reduce Tavily output to a synthesized answer + deduped, truncated bullets."""
    if isinstance(raw, str):
        return raw.strip() or "(no web results)"
    if not isinstance(raw, dict):
        return str(raw)

    lines = []
    answer = (raw.get("answer") or "").strip()
    if answer:
        lines.append(f"Summary: {answer}")

    seen = set()
    for r in raw.get("results", []):
        content = (r.get("content") or "").strip()
        if not content:
            continue
        key = content[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        title = (r.get("title") or "").strip()
        snippet = content[: config.WEB_SNIPPET_CHARS]
        lines.append(f"- {title}: {snippet}" if title else f"- {snippet}")

    return "\n".join(lines) if lines else "(no web results)"
