"""Orchestration: frame -> retrieve -> debate (parallel) -> critic-judge -> score.

Three sequential stages, ~4 LLM calls:
    1) framing (fast model)
    2) two advocates, run concurrently
    3) merged critic-judge (structured)
The final decision is computed deterministically from the rubric weights, not by the LLM.
"""
import config

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from langchain_core.runnables import RunnableLambda, RunnableParallel

from agents import advocate_chain, critic_judge, frame_query
from retrieval import rag_context, web_search


def _weighted_total(scores) -> float:
    return round(
        config.WEIGHT_EVIDENCE * scores.evidence_score
        + config.WEIGHT_MARKET * scores.market_score
        + config.WEIGHT_REASONING * scores.reasoning_score,
        2,
    )


def _build_context(query: str, route: str) -> str:
    # RAG (CPU-bound embedding) and web search (network-bound Tavily) are
    # independent, so for the BOTH route run them concurrently and pay only the
    # slower of the two instead of their sum.
    want_rag = route in ("RAG", "BOTH")
    want_web = route in ("WEB", "BOTH")

    with ThreadPoolExecutor(max_workers=2) as pool:
        rag_future = pool.submit(rag_context, query) if want_rag else None
        web_future = pool.submit(web_search, query) if want_web else None

        parts = []
        if rag_future is not None:
            parts.append("== Document Knowledge (RAG) ==\n" + rag_future.result())
        if web_future is not None:
            parts.append("== Web Search ==\n" + web_future.result())

    return "\n\n".join(parts) if parts else "(no context retrieved)"


def _assemble_result(option_a, option_b, route, context, arg_a, arg_b) -> dict:
    """Merged critic-judge + deterministic scoring → final result dict.

    Shared by both run_debate() and run_debate_stream() so the produced output is
    identical regardless of whether the advocates were streamed.
    """
    verdict = critic_judge(option_a, option_b, arg_a, arg_b, context)
    total_a = _weighted_total(verdict.option_a_scores)
    total_b = _weighted_total(verdict.option_b_scores)

    # Always declare a winner — no "Balanced". On an exact total tie, break it
    # deterministically by reasoning, then evidence, then market sub-score.
    if total_a != total_b:
        decision = option_a if total_a > total_b else option_b
    else:
        sa, sb = verdict.option_a_scores, verdict.option_b_scores
        tiebreak = (
            (sa.reasoning_score - sb.reasoning_score)
            or (sa.evidence_score - sb.evidence_score)
            or (sa.market_score - sb.market_score)
        )
        decision = option_a if tiebreak >= 0 else option_b

    return {
        "option_a": option_a,
        "option_b": option_b,
        "route": route,
        "context": context,
        "arg_a": arg_a,
        "arg_b": arg_b,
        "critique": verdict.critique,
        "scores_a": verdict.option_a_scores.model_dump(),
        "scores_b": verdict.option_b_scores.model_dump(),
        "total_a": total_a,
        "total_b": total_b,
        "decision": decision,
        "reasoning_summary": verdict.reasoning_summary,
    }


def run_debate(query: str) -> dict:
    # 1) Framing / routing
    framing = frame_query(query)
    option_a, option_b, route = framing.option_a, framing.option_b, framing.route

    # 2) Grounding context
    context = _build_context(query, route)

    # 3) Advocates — run concurrently (RunnableParallel uses a thread pool)
    advocate = advocate_chain()
    base = {"option_a": option_a, "option_b": option_b, "context": context}
    parallel = RunnableParallel(
        arg_a=RunnableLambda(lambda _: {**base, "side": option_a}) | advocate,
        arg_b=RunnableLambda(lambda _: {**base, "side": option_b}) | advocate,
    )
    debate = parallel.invoke({})

    # 4) Merged critic-judge + deterministic scoring
    return _assemble_result(
        option_a, option_b, route, context, debate["arg_a"], debate["arg_b"]
    )


def run_debate_stream(query: str):
    """Streaming variant of run_debate(): emits advocate tokens as they arrive.

    Yields event tuples for the UI to consume:
        ("framed", {option_a, option_b, route})
        ("advocate_token", acc_a, acc_b)   — throttled partial advocate text
        ("advocates_done", arg_a, arg_b)   — complete advocate text
        ("result", <result dict>)          — same shape/content as run_debate()

    The final result is identical to run_debate(); only delivery is incremental.
    """
    # 1) Framing / routing
    framing = frame_query(query)
    option_a, option_b, route = framing.option_a, framing.option_b, framing.route
    yield ("framed", {"option_a": option_a, "option_b": option_b, "route": route})

    # 2) Grounding context
    context = _build_context(query, route)

    # 3) Advocates — stream both sides concurrently, interleaving tokens via a queue
    advocate = advocate_chain()
    base = {"option_a": option_a, "option_b": option_b, "context": context}
    q: queue.Queue = queue.Queue()
    acc = {"a": "", "b": ""}

    def worker(side_key: str, side_value: str):
        try:
            for chunk in advocate.stream({**base, "side": side_value}):
                q.put((side_key, chunk))
        except Exception as exc:  # mid-stream failure degrades gracefully
            q.put((side_key, f"\n\n_(advocate error: {exc})_"))
        finally:
            q.put((side_key, None))  # sentinel: this side finished

    for side_key, side_value in (("a", option_a), ("b", option_b)):
        threading.Thread(target=worker, args=(side_key, side_value), daemon=True).start()

    # Drain the queue, accumulating per side. Throttle UI emits to ~20/sec so the
    # browser isn't flooded with a Markdown re-render on every single token.
    done = 0
    last_emit = 0.0
    while done < 2:
        side_key, chunk = q.get()
        if chunk is None:
            done += 1
            continue
        acc[side_key] += chunk
        now = time.monotonic()
        if now - last_emit >= 0.05:
            last_emit = now
            yield ("advocate_token", acc["a"], acc["b"])

    arg_a, arg_b = acc["a"], acc["b"]
    yield ("advocates_done", arg_a, arg_b)

    # 4) Merged critic-judge + deterministic scoring
    yield ("result", _assemble_result(option_a, option_b, route, context, arg_a, arg_b))
