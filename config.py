"""Central configuration for Decision Debater AI.

All tunables live here so the rest of the codebase stays declarative.
Environment variables are loaded from a local .env (see .env.example).
"""
import os

# Tell transformers/sentence-transformers not to import TensorFlow or Keras. Keras 3 without
# tf-keras breaks the TF import path; we only use the torch backend. Must be set
# before any model loads, so it lives at the top of this (first-imported) module.
os.environ["USE_TF"] = "0"
os.environ["USE_KERAS"] = "0"

# torch (MKL/libiomp) and chromadb's onnxruntime each bundle an OpenMP runtime;
# loading both in one process intermittently segfaults on Windows. Allow the
# duplicate runtime and keep OpenMP single-threaded to avoid the conflict. These
# must also be set before any native lib loads.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Avoid HF tokenizers spawning threads under our RunnableParallel advocates.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv

# override=True makes .env authoritative, so a stale GROQ_API_KEY/TAVILY_API_KEY left
# in the OS or shell environment can never shadow the value in .env.
load_dotenv(override=True)

# --- API keys (read from environment / .env) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- LLMs (Groq) ---
REASONING_MODEL = "llama-3.3-70b-versatile"  # advocates + critic-judge
FAST_MODEL = "llama-3.1-8b-instant"          # framing / router
LLM_TEMPERATURE = 0.2
# Cap output length to keep per-run token usage (and free-tier quota burn) modest.
# Generous enough for the bullet arguments and the structured verdict.
MAX_OUTPUT_TOKENS = 1024

# --- Embeddings / RAG ---
# BGE gives stronger retrieval than MiniLM. Swap to "intfloat/e5-base-v2" if preferred
# (note: e5 expects "query:"/"passage:" prefixes instead of the BGE instruction).
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"

CHROMA_DIR = "./chroma_db"
CHROMA_COLLECTION = "decision_docs"
DOCS_DIR = "./docs"

# Token-based chunking (via tiktoken encoder).
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80

RETRIEVER_K = 4          # final chunks fed to the agents
RETRIEVER_FETCH_K = 12   # candidates fetched before reranking
RERANK_TOP_N = 4
INGEST_BATCH_SIZE = 64   # chunks embedded+persisted per batch during ingestion

# --- Web search (Tavily) ---
WEB_MAX_RESULTS = 5
WEB_SNIPPET_CHARS = 400

# --- Scoring rubric (weighted combination, see pipeline.py) ---
WEIGHT_EVIDENCE = 0.4    # support from indexed documents (RAG)
WEIGHT_MARKET = 0.3      # validation from real-world / web data
WEIGHT_REASONING = 0.3   # internal logical strength
# The verdict always names a winner (the higher weighted total); no "Balanced" tie.

# --- Gradio server ---
# Bind to loopback so the printed URL is a clickable http://localhost:PORT
# (0.0.0.0 exposes the app on the network and isn't browseable as a URL).
SERVER_NAME = os.getenv("SERVER_NAME", "127.0.0.1")
SERVER_PORT = int(os.getenv("PORT", "7860"))
