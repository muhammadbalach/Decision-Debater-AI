---
title: Decision Debater AI
emoji: ⚖️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
python_version: 3.11
pinned: false
---

# ⚖️ Decision Debater AI


A multi-agent **decision intelligence** system. Give it any **A vs B** decision and it
frames the two options, grounds them with Retrieval-Augmented Generation (RAG) + live
web search, runs two advocate agents **in parallel**, then a merged **Critic-Judge**
scores each side on a weighted rubric and renders a structured verdict — behind a
Gradio UI.

## Architecture

Three sequential stages, ~4 LLM calls per query:

```
Query
  │
  ▼
1. Framing / Router  (fast model)  → option_a, option_b, route ∈ {RAG, WEB, BOTH}
  │
  ▼
2. Context retrieval               → BGE/Chroma RAG  ⊕  cleaned Tavily web results
  │
  ▼
3. Advocate A  ║  Advocate B       → run concurrently (RunnableParallel)
  │
  ▼
4. Critic-Judge (structured)       → critique + per-option scores → weighted verdict
```

| Concern        | Choice |
|----------------|--------|
| LLMs           | **Groq** — `llama-3.3-70b-versatile` (reasoning), `llama-3.1-8b-instant` (framing) |
| Embeddings     | **`BAAI/bge-base-en-v1.5`** (local, normalized, with BGE query instruction) |
| Vector store   | **ChromaDB** (persisted in `./chroma_db`) |
| Web search     | **Tavily** (`include_answer` + cleaned, deduped snippets) |
| Reranking      | Optional **`BAAI/bge-reranker-base`** cross-encoder (set `USE_RERANKER=true`) |

### Scoring rubric

The Critic-Judge scores **each option 0–10** on three dimensions; the final decision is
computed deterministically from the weighted total (weights in `config.py`):

```
total = 0.4 * evidence  +  0.3 * market  +  0.3 * reasoning
```

`decision` = the option with the higher weighted total. The verdict **always names a
winner** (an exact total tie is broken by the reasoning, then evidence, then market
sub-score).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in GROQ_API_KEY and TAVILY_API_KEY
```

- Get a Groq key: https://console.groq.com/keys
- Get a Tavily key: https://app.tavily.com

> **First run** downloads the BGE embedding model (~440 MB; +reranker if enabled).

## Ingest documents (RAG)

Optional but recommended. Drop PDFs / `.txt` / `.md` into `docs/`. Sub-folders become a
`domain` metadata tag (e.g. `docs/career/`, `docs/startup/`):

```bash
python ingest.py
```

The app works with an **empty** store too — RAG context simply reports that no documents
are indexed and the debate proceeds on web search / reasoning.

## Run

```bash
python app.py
# open http://localhost:7860
```

## Fault tolerance

The system never hard-crashes on infrastructure gaps:
- empty/missing Chroma → "(no indexed documents)" note,
- missing Tavily key or search error → skip-and-continue note,
- malformed model output → defensive parse then a safe default verdict.

## Project layout

```
config.py      # all tunables + env loading
retrieval.py   # BGE/Chroma retriever (+ optional reranker) and Tavily search
ingest.py      # CLI: docs/ -> chunk -> embed -> persist
agents.py      # framing, advocate, critic-judge chains (Pydantic structured outputs)
pipeline.py    # run_debate() orchestration + weighted scoring
app.py         # Gradio UI
docs/          # your knowledge base (domain sub-folders)
```

## Possible enhancements

- Stream agent outputs token-by-token into the UI.
- Per-domain retrieval filtering surfaced as a UI control.
- LLM-based reranking as an alternative to the cross-encoder.
