"""Orchestration: frame -> retrieve -> debate (parallel) -> critic-judge -> score.

Three sequential stages, ~4 LLM calls:
    1) framing (fast model)
    2) two advocates, run concurrently
    3) merged critic-judge (structured)
The final decision is computed deterministically from the rubric weights, not by the LLM.
"""
import config

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
    parts = []
    if route in ("RAG", "BOTH"):
        parts.append("== Document Knowledge (RAG) ==\n" + rag_context(query))
    if route in ("WEB", "BOTH"):
        parts.append("== Web Search ==\n" + web_search(query))
    return "\n\n".join(parts) if parts else "(no context retrieved)"


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
    arg_a, arg_b = debate["arg_a"], debate["arg_b"]

    # 4) Merged critic-judge + deterministic scoring
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
