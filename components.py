"""HTML component builders for Decision Debater AI premium UI.

Each function returns an HTML string used in gr.HTML() blocks.
Kept separate from app.py for clarity and reuse.
"""
import html as html_lib


# ─────────────────── NAVBAR ───────────────────────────────────────────────────

def navbar_html() -> str:
    return """
<div id="dd-navbar">
  <div class="nav-brand">
    <span class="brand-icon">🧠</span>
    Decision Debater AI
  </div>
  <div class="nav-tagline">Evidence-Based AI Decision Intelligence</div>
  <div class="nav-status">
    <span class="status-dot">RAG Ready</span>
    <span class="status-dot accent">Web Connected</span>
    <span class="status-dot primary">Groq Online</span>
  </div>
</div>
"""


# ─────────────────── HERO ─────────────────────────────────────────────────────

def hero_html() -> str:
    return """
<div id="dd-hero">
  <div class="hero-chips">
    <span class="chip active">⚡ Groq</span>
    <span class="chip active">📚 RAG</span>
    <span class="chip active">🌐 Tavily</span>
    <span class="chip active">🤖 Multi-Agent</span>
    <span class="chip active">📡 Live Data</span>
  </div>
  <h1>Make Better Decisions<br><span class="highlight">with AI Debate</span></h1>
  <p class="hero-sub">
    Compare two choices using AI Agents + RAG + Live Web Data.<br>
    Two advocates argue, a critic cross-examines, a judge scores.
  </p>
</div>
"""


# ─────────────────── STATS ────────────────────────────────────────────────────

def stats_html(doc_count: int = 0, web_sources: int = 0,
               response_time: str = "—", confidence: str = "—") -> str:
    return f"""
<div id="dd-stats">
  <div class="dd-stat-card">
    <span class="stat-icon">📄</span>
    <span class="stat-label">Documents Indexed</span>
    <span class="stat-value">{doc_count}</span>
  </div>
  <div class="dd-stat-card">
    <span class="stat-icon">🌐</span>
    <span class="stat-label">Web Sources Used</span>
    <span class="stat-value">{web_sources}</span>
  </div>
  <div class="dd-stat-card">
    <span class="stat-icon">⚡</span>
    <span class="stat-label">Response Time</span>
    <span class="stat-value">{response_time}</span>
  </div>
  <div class="dd-stat-card">
    <span class="stat-icon">🎯</span>
    <span class="stat-label">Confidence</span>
    <span class="stat-value">{confidence}</span>
  </div>
</div>
"""


# ─────────────────── PIPELINE STATUS ─────────────────────────────────────────

_PIPELINE_STEPS = [
    ("🔀", "Router"),
    ("📚", "RAG Retrieval"),
    ("🌐", "Web Search"),
    ("⚔️",  "Debate"),
    ("⚖️",  "Judge"),
]


def pipeline_html(active_step: int = -1, done_steps: set | None = None,
                  running_step: int = -1) -> str:
    """Render the pipeline sidebar.

    active_step:  currently spinning step (legacy alias, same as running_step).
    done_steps:   set of completed step indices.
    running_step: step shown with spinner ring (overrides active_step).
    """
    done = done_steps or set()
    spin = running_step if running_step >= 0 else active_step
    steps_html = ""
    for i, (icon, label) in enumerate(_PIPELINE_STEPS):
        if i in done:
            cls = "done"
            disp = "✅"
        elif i == spin:
            cls = "running"
            disp = ""          # spinner rendered via CSS border
        else:
            cls = ""
            disp = icon
        steps_html += f"""
  <div class="pipeline-step {cls}">
    <div class="step-icon">{disp}</div>
    <span class="step-label">{label}</span>
  </div>"""

    return f"""
<div class="dd-pipeline-card">
  <h3>Pipeline Status</h3>
  {steps_html}
</div>
"""


def pipeline_idle_html() -> str:
    return pipeline_html()


def pipeline_loading_html() -> str:
    """Show Router step spinning — used immediately on button click."""
    return pipeline_html(running_step=0)


def pipeline_done_html() -> str:
    return pipeline_html(done_steps={0, 1, 2, 3, 4})


# ─────────────────── SECTION DIVIDERS ─────────────────────────────────────────

def divider_html(label: str) -> str:
    return f'<div class="dd-section-divider">{label}</div>'


# ─────────────────── ADVOCATE CARD (empty state) ──────────────────────────────

def _score_bar(label: str, value: float | None, side: str) -> str:
    if value is None:
        return ""
    pct = min(max(value * 10, 0), 100)
    return f"""
    <div class="score-row">
      <span class="s-label">{label}</span>
      <span class="s-track"><span class="s-fill" style="width:{pct:.0f}%"></span></span>
      <span class="s-val">{value:.1f}</span>
    </div>"""


def advocate_card_empty(side: str) -> str:
    """Placeholder markdown returned before analysis."""
    if side == "a":
        return "### 🟢 Option A Advocate\n\n*The case for the first option will appear here once you submit a decision.*"
    return "### 🔵 Option B Advocate\n\n*The case for the second option will appear here once you submit a decision.*"


def loading_advocate_md(side: str) -> str:
    """Skeleton markdown shown immediately when analysis starts."""
    icon  = "🟢" if side == "a" else "🔵"
    label = "Option A" if side == "a" else "Option B"
    # HTML skeleton embedded in Markdown (Gradio renders it)
    return f"""### {icon} {label} — Analysing

<div class='dd-loading-card side-{side}'>
  <div class='lc-header'><div class='skel skel-title lc-badge'></div></div>
  <div class='lc-body'>
    <div class='skel skel-line'></div>
    <div class='skel skel-line sm'></div>
    <div class='skel skel-line'></div>
    <div class='skel skel-line xs'></div>
    <div class='skel skel-line sm'></div>
  </div>
</div>"""


def critic_empty() -> str:
    return "*The critic's analysis of both arguments will appear here after debate.*"


def loading_critic_md() -> str:
    """Skeleton critic card shown during analysis."""
    return """<div class='dd-loading-card' style='border-left:3px solid #F59E0B'>
  <div class='lc-header'><div class='skel skel-title'></div></div>
  <div class='lc-body'>
    <div class='skel skel-line'></div>
    <div class='skel skel-line sm'></div>
    <div class='skel skel-line'></div>
    <div class='skel skel-line xs'></div>
  </div>
</div>"""


def loading_verdict_html() -> str:
    """Large glowing loading card shown while judge deliberates."""
    return """
<div class="dd-verdict-card">
  <div class="dd-loading-verdict">
    <div class="lv-ring"></div>
    <span class="lv-label">AI Tribunal Deliberating
      <span class="dd-thinking"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>
    </span>
    <p class="lv-sub">Advocates are building their cases,<br>the critic is cross-examining, the judge is scoring.</p>
    <div class="lv-steps">
      <span class="lv-step active">Frame</span>
      <span class="lv-sep">›</span>
      <span class="lv-step">Retrieve</span>
      <span class="lv-sep">›</span>
      <span class="lv-step">Debate</span>
      <span class="lv-sep">›</span>
      <span class="lv-step">Critique</span>
      <span class="lv-sep">›</span>
      <span class="lv-step">Verdict</span>
    </div>
  </div>
</div>
"""


# ─────────────────── VERDICT HTML ─────────────────────────────────────────────

def verdict_empty_html() -> str:
    return """
<div class="dd-verdict-card">
  <div class="ruling empty">
    <span class="eyebrow">Final Recommendation</span>
    <p class="note">
      Pose a decision above — two AI advocates debate, a critic cross-examines,
      and a scored verdict names the winner with evidence-backed reasoning.
    </p>
  </div>
</div>
"""


def verdict_html(r: dict) -> str:
    """Build the full glowing verdict card from pipeline results."""
    a  = html_lib.escape(str(r["option_a"]))
    b  = html_lib.escape(str(r["option_b"]))
    sa = r["scores_a"]
    sb = r["scores_b"]
    ta = r["total_a"]
    tb = r["total_b"]
    side     = "a" if r["decision"] == r["option_a"] else "b"
    decision = html_lib.escape(str(r["decision"]))
    reasoning= html_lib.escape(str(r["reasoning_summary"]))

    def row(label, va, vb):
        return f"<tr><td>{label}</td><td>{va:g}</td><td>{vb:g}</td></tr>"

    ta_pct = min(ta * 10, 100)
    tb_pct = min(tb * 10, 100)

    # Confidence derived from score gap (0–10 → 50–100 %)
    gap = abs(ta - tb)
    confidence = int(50 + min(gap * 5, 50))

    return f"""
<div class="dd-verdict-card">
  <div class="ruling">
    <span class="eyebrow">Final Recommendation</span>
    <div class="headline">
      <span class="gavel">⚖</span>
      <span class="winner {side}">{decision}</span>
      <span class="winner-sub">wins the debate</span>
    </div>

    <table class="scorecard">
      <thead>
        <tr>
          <th>Dimension</th>
          <th class="col-a">{a}</th>
          <th class="col-b">{b}</th>
        </tr>
      </thead>
      <tbody>
        {row("Evidence", sa["evidence_score"], sb["evidence_score"])}
        {row("Market",   sa["market_score"],   sb["market_score"])}
        {row("Reasoning",sa["reasoning_score"],sb["reasoning_score"])}
      </tbody>
    </table>

    <div class="bars">
      <div class="bar">
        <span class="bar-lab">{a}</span>
        <span class="track"><span class="fill a" style="width:{ta_pct:.0f}%"></span></span>
        <span class="bar-val">{ta:.1f}</span>
      </div>
      <div class="bar">
        <span class="bar-lab">{b}</span>
        <span class="track"><span class="fill b" style="width:{tb_pct:.0f}%"></span></span>
        <span class="bar-val">{tb:.1f}</span>
      </div>
    </div>

    <p class="why"><span class="lab">Reasoning Summary</span>{reasoning}</p>
  </div>
</div>
"""


# ─────────────────── FOOTER ───────────────────────────────────────────────────

def footer_html() -> str:
    return """
<div id="dd-footer">
  <span class="footer-brand">🧠 Decision Debater AI</span>
  <div class="footer-stack">
    <span>Groq · Llama 3.3 70B</span>
    <span>·</span>
    <span>BGE + ChromaDB</span>
    <span>·</span>
    <span>Tavily Web Search</span>
    <span>·</span>
    <span>LangChain</span>
  </div>
</div>
"""
