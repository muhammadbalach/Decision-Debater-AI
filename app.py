"""Decision Debater AI — Premium SaaS UI.

Architecture:
    styles.css    — all visual tokens, dark-mode design system
    components.py — reusable HTML string builders
    app.py        — Gradio Blocks layout + event logic  (this file)

Pipeline (pipeline.py / agents.py / retrieval.py) is untouched.
"""

# config MUST be imported first: it sets native-library env guards
# (USE_TF / KMP_DUPLICATE_LIB_OK / OMP_NUM_THREADS) that have to be in place BEFORE
# gradio/torch/chromadb load. Otherwise two OpenMP runtimes initialize and the
# pipeline segfaults mid-run (UI hangs on the loading state).
import config

import html as html_lib
import re
import time
from pathlib import Path

import gradio as gr

from components import (
    advocate_card_empty,
    critic_empty,
    divider_html,
    footer_html,
    hero_html,
    loading_advocate_md,
    loading_critic_md,
    loading_verdict_html,
    navbar_html,
    pipeline_done_html,
    pipeline_idle_html,
    pipeline_loading_html,
    stats_html,
    verdict_empty_html,
    verdict_html,
)
from pipeline import run_debate_stream


# ─────────────────── Load CSS from file ───────────────────────────────────────

_CSS_PATH = Path(__file__).parent / "styles.css"
CSS = _CSS_PATH.read_text(encoding="utf-8")

# Gradio 6 scopes everything passed via `css=` and corrupts the `:root { … }`
# block (rewriting its selector to an unrelated internal class), which leaves
# every CSS custom property undefined — so var(--card) etc. fall back to
# transparent and card backgrounds/borders vanish. Extract the :root block and
# inject it through gr.Blocks(head=…), which is injected raw into <head> and is
# NOT scoped, so the variables resolve globally for the whole design.
_root_match = re.search(r":root\s*\{[^}]*\}", CSS)
ROOT_VARS_HEAD = f"<style>{_root_match.group(0)}</style>" if _root_match else ""


# ─────────────────── Helpers ──────────────────────────────────────────────────

def _count_web_sources(context: str) -> int:
    """Estimate number of web sources from context string."""
    return context.count("- ") if "== Web Search ==" in context else 0


def _count_doc_chunks(context: str) -> int:
    return context.count("[") if "== Document Knowledge" in context else 0


def _format_score_md(label: str, option: str, body: str, scores: dict,
                     side: str) -> str:
    """Build markdown content for an advocate card including score bars."""
    pct_ev  = min(scores["evidence_score"]  * 10, 100)
    pct_mkt = min(scores["market_score"]    * 10, 100)
    pct_rsn = min(scores["reasoning_score"] * 10, 100)

    # Score bars are rendered via inline HTML inside Markdown
    bars = f"""
<div class="score-bars">
  <div class="score-row">
    <span class="s-label">Evidence</span>
    <span class="s-track"><span class="s-fill" style="width:{pct_ev:.0f}%"></span></span>
    <span class="s-val">{scores['evidence_score']:.1f}</span>
  </div>
  <div class="score-row">
    <span class="s-label">Market</span>
    <span class="s-track"><span class="s-fill" style="width:{pct_mkt:.0f}%"></span></span>
    <span class="s-val">{scores['market_score']:.1f}</span>
  </div>
  <div class="score-row">
    <span class="s-label">Reasoning</span>
    <span class="s-track"><span class="s-fill" style="width:{pct_rsn:.0f}%"></span></span>
    <span class="s-val">{scores['reasoning_score']:.1f}</span>
  </div>
</div>"""

    return f"### {label}\n\n{body}\n{bars}"


def _streaming_md(side: str, option: str | None, body: str) -> str:
    """Advocate card markdown while tokens are still streaming in.

    No score bars yet — the scores come from the critic, which runs only after
    both advocates have finished.
    """
    icon = "🟢" if side == "a" else "🔵"
    if option:
        label = f"{icon} {option}"
    else:
        label = "🟢 Option A" if side == "a" else "🔵 Option B"
    body = body.strip()
    if not body:
        return f"### {label}\n\n_Building argument…_"
    return f"### {label}\n\n{body}"


# ─────────────────── Error card ───────────────────────────────────────────────

def _error_card(exc: Exception) -> str:
    """Render a clear, user-facing error card — friendly for rate limits."""
    msg = str(exc)
    low = msg.lower()
    if "rate_limit" in low or "tokens per day" in low or "error code: 429" in low:
        m = re.search(r"try again in ([\dhms.]+)", msg)
        retry = f" Please try again in {m.group(1).rstrip('.')}." if m else ""
        title = "Groq rate limit reached"
        note = (
            "The Groq free-tier token limit for this model has been reached." + retry
            + " This is an API quota limit, not an app error — wait for the quota to "
            "reset (it resets daily) or upgrade your Groq plan at "
            "console.groq.com/settings/billing."
        )
    else:
        title = "Pipeline error"
        note = "The debate could not be completed: " + html_lib.escape(msg)
    return (
        '<div class="dd-verdict-card"><div class="ruling empty">'
        f'<span class="eyebrow">{title}</span><p class="note">{note}</p>'
        "</div></div>"
    )


# ─────────────────── Main analyze function ────────────────────────────────────

def analyze(query: str):
    """Generator: loading → framing → live advocate tokens → final results."""

    # ── Validate input ──────────────────────────────────────────────
    if not query or not query.strip():
        yield (
            advocate_card_empty("a"),
            advocate_card_empty("b"),
            critic_empty(),
            verdict_empty_html(),
            pipeline_idle_html(),
            stats_html(),
            gr.update(value="⚡ Analyze Decision", interactive=True),
        )
        return

    # ── FIRST YIELD: instant loading state ──────────────────────────
    yield (
        loading_advocate_md("a"),
        loading_advocate_md("b"),
        loading_critic_md(),
        loading_verdict_html(),
        pipeline_loading_html(),
        stats_html(),
        gr.update(value="⏳ Analysing…", interactive=False),
    )

    # ── Run pipeline (streaming the advocate arguments live) ─────────
    t0 = time.perf_counter()
    option_a = option_b = None
    r = None

    def _streaming_frame(body_a: str, body_b: str):
        """7-tuple shown while advocates stream — critic/verdict stay loading."""
        return (
            _streaming_md("a", option_a, body_a),
            _streaming_md("b", option_b, body_b),
            loading_critic_md(),
            loading_verdict_html(),
            pipeline_loading_html(),
            stats_html(),
            gr.update(value="⏳ Analysing…", interactive=False),
        )

    try:
        for event in run_debate_stream(query):
            kind = event[0]
            if kind == "framed":
                meta = event[1]
                option_a, option_b = meta["option_a"], meta["option_b"]
                yield _streaming_frame("", "")
            elif kind in ("advocate_token", "advocates_done"):
                yield _streaming_frame(event[1], event[2])
            elif kind == "result":
                r = event[1]
                break
    except Exception as exc:
        yield (
            advocate_card_empty("a"),
            advocate_card_empty("b"),
            critic_empty(),
            _error_card(exc),
            pipeline_idle_html(),
            stats_html(),
            gr.update(value="⚡ Analyze Decision", interactive=True),
        )
        return

    if r is None:  # stream ended without a result (shouldn't happen) — bail safely
        return

    elapsed  = time.perf_counter() - t0
    context  = r.get("context", "")
    web_srcs = _count_web_sources(context)
    doc_chks = _count_doc_chunks(context)

    # Confidence from score gap
    ta, tb   = r["total_a"], r["total_b"]
    gap      = abs(ta - tb)
    conf_pct = int(50 + min(gap * 5, 50))

    # ── Build advocate markdown ──────────────────────────────────────
    md_a = _format_score_md(
        label   = f"🟢 {r['option_a']}",
        option  = r["option_a"],
        body    = r["arg_a"],
        scores  = r["scores_a"],
        side    = "a",
    )
    md_b = _format_score_md(
        label   = f"🔵 {r['option_b']}",
        option  = r["option_b"],
        body    = r["arg_b"],
        scores  = r["scores_b"],
        side    = "b",
    )

    # ── SECOND YIELD: real results ───────────────────────────────────
    yield (
        md_a,
        md_b,
        r["critique"],
        verdict_html(r),
        pipeline_done_html(),
        stats_html(
            doc_count     = doc_chks,
            web_sources   = web_srcs,
            response_time = f"{elapsed:.1f}s",
            confidence    = f"{conf_pct}%",
        ),
        gr.update(value="⚡ Analyze Decision", interactive=True),
    )


# ─────────────────── Gradio Blocks layout ─────────────────────────────────────

_THEME = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#0B1120",
    body_background_fill_dark="#0B1120",
    block_background_fill="#1F2937",
    block_background_fill_dark="#1F2937",
    input_background_fill="#1F2937",
    input_background_fill_dark="#1F2937",
    border_color_primary="rgba(255,255,255,0.08)",
    border_color_primary_dark="rgba(255,255,255,0.08)",
    color_accent="#4F46E5",
    color_accent_soft="rgba(79,70,229,0.15)",
    button_primary_background_fill="linear-gradient(135deg,#4F46E5,#6366F1)",
    button_primary_background_fill_hover="linear-gradient(135deg,#5B52F0,#818CF8)",
    button_primary_text_color="#FFFFFF",
)

with gr.Blocks(
    title="Decision Debater AI — Evidence-Based AI Decision Intelligence",
    head=ROOT_VARS_HEAD,   # global CSS variables (see ROOT_VARS_HEAD above)
) as demo:

    # ── Navigation ──────────────────────────────────────────────────
    gr.HTML(navbar_html())

    # ── Hero ────────────────────────────────────────────────────────
    gr.HTML(hero_html())

    # ── Input Row ───────────────────────────────────────────────────
    with gr.Row(elem_id="dd-input-row"):
        user_input = gr.Textbox(
            label="",
            placeholder='Should I join a startup or take a corporate job?',
            elem_id="dd-question",
            lines=2,
            max_lines=5,
            scale=5,
            show_label=False,
        )
        analyze_btn = gr.Button(
            "⚡ Analyze Decision",
            variant="primary",
            elem_id="dd-analyze-btn",
            scale=1,
            min_width=160,
        )

    # ── Stats row (updates after analysis) ──────────────────────────
    stats_out = gr.HTML(
        stats_html(),
        elem_id="dd-stats-wrapper",
    )

    # ── Section divider ─────────────────────────────────────────────
    gr.HTML(divider_html("Debate Floor"))

    # ── Main debate layout: pipeline sidebar + two columns ──────────
    with gr.Row(elem_id="dd-debate-section"):
        # LEFT: Pipeline status panel
        with gr.Column(scale=0, min_width=220, elem_id="dd-pipeline-panel"):
            pipeline_out = gr.HTML(pipeline_idle_html())

        # RIGHT: Advocate debate columns
        with gr.Column(scale=1):
            with gr.Row(elem_id="dd-debate-cols"):
                # Side A
                with gr.Column(elem_classes=["dd-advocate-card", "side-a"]):
                    gr.HTML(
                        '<div class="card-header">'
                        '<span class="advocate-label">🟢 Option A Advocate</span>'
                        '<span class="confidence-badge">Advocate</span>'
                        '</div>'
                    )
                    with gr.Column(elem_classes=["card-body"]):
                        camp_a = gr.Markdown(
                            advocate_card_empty("a"),
                            elem_classes=["prose"],
                        )

                # Side B
                with gr.Column(elem_classes=["dd-advocate-card", "side-b"]):
                    gr.HTML(
                        '<div class="card-header">'
                        '<span class="advocate-label">🔵 Option B Advocate</span>'
                        '<span class="confidence-badge">Advocate</span>'
                        '</div>'
                    )
                    with gr.Column(elem_classes=["card-body"]):
                        camp_b = gr.Markdown(
                            advocate_card_empty("b"),
                            elem_classes=["prose"],
                        )

    # ── Critic analysis ─────────────────────────────────────────────
    gr.HTML(divider_html("Critic Analysis"))

    with gr.Row(elem_id="dd-critic-section"):
        with gr.Column():
            gr.HTML(
                '<div class="dd-critic-card">'
                '<div class="card-header">'
                '<span class="critic-label">🔍 Critic Analysis · Cross-Examination</span>'
                '</div>'
                '<div class="card-body">'
            )
            critic_out = gr.Markdown(
                critic_empty(),
                elem_classes=["prose"],
            )
            gr.HTML("</div></div>")

    # ── Final verdict ───────────────────────────────────────────────
    gr.HTML(divider_html("Final Recommendation"))

    with gr.Row(elem_id="dd-verdict-section"):
        with gr.Column():
            verdict_out = gr.HTML(verdict_empty_html())

    # ── Footer ──────────────────────────────────────────────────────
    gr.HTML(footer_html())

    # ── Event wiring ────────────────────────────────────────────────
    # NOTE: this list must match the 7 values each analyze() yield returns, in order
    # (the 7th updates the Analyze button's label / enabled state).
    outputs = [camp_a, camp_b, critic_out, verdict_out, pipeline_out, stats_out, analyze_btn]

    analyze_btn.click(
        fn=analyze,
        inputs=user_input,
        outputs=outputs,
        show_progress="minimal",
    )
    user_input.submit(
        fn=analyze,
        inputs=user_input,
        outputs=outputs,
        show_progress="minimal",
    )


# ─────────────────── Entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    # Warm the embedding model + vector store in the background while Gradio boots,
    # so the first user query doesn't eat the cold-start model-load penalty.
    import threading

    from retrieval import warm_up

    threading.Thread(target=warm_up, daemon=True).start()

    _launch_kwargs = dict(
        server_name=config.SERVER_NAME,
        theme=_THEME,
        css=CSS,
    )
    try:
        demo.launch(server_port=config.SERVER_PORT, **_launch_kwargs)
    except OSError:
        # Port busy — find next free port automatically
        demo.launch(server_port=None, **_launch_kwargs)
