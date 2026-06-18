"""Agent definitions: framing/router, advocate, and merged critic-judge.

Structured outputs use Pydantic + `with_structured_output`; each structured call has a
defensive fallback so malformed model output degrades to a safe default instead of
crashing the pipeline.
"""
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

import config


# --------------------------------------------------------------------------- models
class Framing(BaseModel):
    option_a: str = Field(description="First competing option, as a short label")
    option_b: str = Field(description="Second competing option, as a short label")
    route: Literal["RAG", "WEB", "BOTH"] = Field(
        description="Retrieval route: RAG (documents), WEB (real-time), or BOTH"
    )


class OptionScores(BaseModel):
    evidence_score: float = Field(ge=0, le=10, description="Support from indexed documents")
    market_score: float = Field(ge=0, le=10, description="Validation from real-world/web data")
    reasoning_score: float = Field(ge=0, le=10, description="Internal logical strength")


class Verdict(BaseModel):
    critique: str = Field(description="Critique of both arguments: flaws, bias, missing risks")
    option_a_scores: OptionScores
    option_b_scores: OptionScores
    reasoning_summary: str = Field(description="Concise explanation of the trade-offs")


# ----------------------------------------------------------------------------- llms
def _llm(model: str) -> ChatGroq:
    # GROQ_API_KEY is read from the environment (loaded by config via .env).
    return ChatGroq(
        model=model,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.MAX_OUTPUT_TOKENS,
    )


# --------------------------------------------------------------------------- prompts
FRAMING_PROMPT = ChatPromptTemplate.from_template(
    "You analyze a user's decision question and identify the two competing options.\n\n"
    "Question: {query}\n\n"
    "Extract the two opposing options as short labels (option_a, option_b). "
    "If the question explicitly names two options, use them; otherwise infer the two "
    "most reasonable opposing choices.\n"
    "Then choose a retrieval route:\n"
    "- RAG: needs factual/historical knowledge from indexed documents\n"
    "- WEB: needs current/real-time statistics or market data\n"
    "- BOTH: needs both."
)

ADVOCATE_PROMPT = ChatPromptTemplate.from_template(
    "You are an expert advocate in a structured debate about: {option_a} vs {option_b}.\n"
    "Argue persuasively and specifically IN FAVOR of: '{side}'.\n\n"
    "Ground your argument in the context below and reference it where relevant. "
    "If the context is empty or irrelevant, rely on well-established reasoning.\n\n"
    "Context:\n{context}\n\n"
    "Provide 4-6 structured arguments as concise, evidence-driven bullet points."
)

CRITIC_JUDGE_PROMPT = ChatPromptTemplate.from_template(
    "You are a rigorous Critic-Judge in a structured decision debate about "
    "{option_a} vs {option_b}.\n\n"
    "Argument FOR {option_a}:\n{arg_a}\n\n"
    "Argument FOR {option_b}:\n{arg_b}\n\n"
    "Grounding context that was available to both sides:\n{context}\n\n"
    "Tasks:\n"
    "1. critique: critically analyze BOTH arguments — logical flaws, bias, missing risks.\n"
    "2. Score EACH option from 0-10 on three dimensions:\n"
    "   - evidence_score: support from the grounding documents/context\n"
    "   - market_score: validation by real-world / market data\n"
    "   - reasoning_score: internal logical strength and completeness\n"
    "3. reasoning_summary: concisely explain the key trade-offs.\n"
    "Be objective and avoid favoring either side without justification."
)


# ----------------------------------------------------------------------------- chains
def frame_query(query: str) -> Framing:
    llm = _llm(config.FAST_MODEL).with_structured_output(Framing)
    try:
        return (FRAMING_PROMPT | llm).invoke({"query": query})
    except Exception:
        return Framing(option_a="Option A", option_b="Option B", route="BOTH")


def advocate_chain():
    """Reusable advocate chain; the assigned side is supplied per-invocation."""
    return ADVOCATE_PROMPT | _llm(config.REASONING_MODEL) | StrOutputParser()


def critic_judge(option_a, option_b, arg_a, arg_b, context) -> Verdict:
    llm = _llm(config.REASONING_MODEL).with_structured_output(Verdict)
    payload = {
        "option_a": option_a,
        "option_b": option_b,
        "arg_a": arg_a,
        "arg_b": arg_b,
        "context": context,
    }
    try:
        return (CRITIC_JUDGE_PROMPT | llm).invoke(payload)
    except Exception:
        neutral = OptionScores(evidence_score=5, market_score=5, reasoning_score=5)
        return Verdict(
            critique="(Critic-Judge output could not be parsed; returning a neutral verdict.)",
            option_a_scores=neutral,
            option_b_scores=neutral.model_copy(),
            reasoning_summary="Defaulted to a balanced verdict due to a parsing error.",
        )
