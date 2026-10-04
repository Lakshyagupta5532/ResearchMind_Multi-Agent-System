import re
import time

import streamlit as st

from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain

st.set_page_config(page_title="Multi-Agent Research", page_icon="⚡", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.stApp {
    background: radial-gradient(900px 380px at 50% -8%, rgba(255,106,31,.16), transparent 65%), #07070A;
}
.block-container { max-width: 1180px; padding-top: 3rem; padding-bottom: 4rem; }

.hero h1 { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 3.1rem; letter-spacing: -0.02em;
    line-height: 1.05; color: #F2F2F6; margin: 0 0 .7rem 0; }
.hero p { color: #7C7C8A; font-size: 1.05rem; max-width: 560px; margin: 0 0 2.6rem 0; }

.lbl { font-family: 'DM Sans', sans-serif; font-weight: 500; font-size: .85rem; letter-spacing: .12em;
    text-transform: uppercase; color: #FF6A1F; margin-bottom: .5rem; }
.try { font-family: 'DM Mono', monospace; font-size: .75rem; letter-spacing: .18em; color: #6A6A78;
    text-transform: uppercase; margin: 1.4rem 0 .5rem 0; }
.panel-title { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.7rem; color: #F2F2F6; margin: 0 0 1rem 0; }

div[data-baseweb="input"] > div { background: #1A1D27 !important; border: 1px solid #FF6A1F !important;
    border-radius: 14px !important; min-height: 54px; }
div[data-baseweb="input"] input { color: #fff; font-size: 1rem; }

.stButton > button { border-radius: 10px; border: 1px solid rgba(255,255,255,.08); background: #12131A;
    color: #C9C9D3; min-height: 0; padding: 6px 16px; transition: all .15s ease; }
.stButton > button:hover { border-color: #FF6A1F; color: #fff; }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, #FF8A1F, #FF5A1F); border: none; min-height: 56px; border-radius: 14px;
    box-shadow: 0 0 44px -10px rgba(255,106,31,.65); }
.stButton > button[kind="primary"] p, .stButton > button[data-testid="stBaseButton-primary"] p {
    color: #1B0B00; font-weight: 600; font-size: 1.05rem; }

.step { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px;
    background: #0E0E12; border: 1px solid rgba(255,255,255,.07); border-radius: 18px;
    padding: 20px 22px; margin-bottom: 14px; transition: all .25s ease; }
.step-l { display: flex; gap: 16px; }
.num { font-family: 'DM Mono', monospace; color: #FF6A1F; font-size: .9rem; padding-top: 3px; }
.name { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.1rem; color: #F2F2F6; }
.desc { color: #7C7C8A; font-size: .92rem; margin-top: 4px; }
.stat { font-family: 'DM Mono', monospace; font-size: .75rem; letter-spacing: .16em; color: #5E5E6B; white-space: nowrap; }
.step.running { border-color: #FF6A1F; box-shadow: 0 0 34px -12px #FF6A1F; animation: glow 1.5s ease-in-out infinite; }
.step.running .stat { color: #FF6A1F; }
.step.done .stat { color: #3DDC97; }
.step.error { border-color: #EF4444; } .step.error .stat { color: #EF4444; }
@keyframes glow { 50% { box-shadow: 0 0 6px -12px #FF6A1F; } }
@media (prefers-reduced-motion: reduce) { .step.running { animation: none; } }

.result-head { display: flex; align-items: center; gap: 18px; margin: 2.4rem 0 .6rem 0; }
.score { min-width: 100px; text-align: center; padding: 12px 16px; border-radius: 16px; background: #0E0E12; border: 1px solid var(--s); }
.score-num { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: var(--s); }
.score-of { color: #7C7C8A; }
.score-label { color: #7C7C8A; font-size: .78rem; }
.result-topic { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: #F2F2F6; }
.result-meta { color: #7C7C8A; font-size: .9rem; }
.stTabs [aria-selected="true"] { color: #FF6A1F; }
.stTabs [data-baseweb="tab-highlight"] { background: #FF6A1F; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

STEPS = [
    ("search", "Search Agent", "Gathers recent web information"),
    ("reader", "Reader Agent", "Scrapes & extracts deep content"),
    ("writer", "Writer Chain", "Drafts the full research report"),
    ("critic", "Critic Chain", "Reviews & scores the report"),
]
EXAMPLES = ["LLM agents 2026", "CRISPR gene editing", "Fusion energy progress"]


def to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def render_tracker(status: dict, secs: dict) -> str:
    html = ""
    for i, (key, name, desc) in enumerate(STEPS, 1):
        s = status[key]
        label = {"idle": "WAITING", "running": "RUNNING", "done": f"DONE {secs.get(key, 0):.1f}S", "error": "FAILED"}[s]
        html += (f'<div class="step {s}"><div class="step-l"><span class="num">{i:02d}</span>'
                 f'<div><div class="name">{name}</div><div class="desc">{desc}</div></div></div>'
                 f'<div class="stat">{label}</div></div>')
    return html


def set_topic(t: str):
    st.session_state["topic"] = t


def step_search(topic, state):
    res = build_search_agent().invoke({"messages": [("user",
        f"Find recent, reliable and detailed information about: {topic}. "
        "Return at least 5 relevant results with their URLs.")]})
    state["search_results"] = to_text(res["messages"][-1].content)


def step_reader(topic, state):
    res = build_reader_agent().invoke({"messages": [("user",
        f"Topic: {topic}\n\nHere are the search results:\n\n{state['search_results']}\n\n"
        "From these results, pick the 3 most relevant and different URLs. Use the scrape_url "
        "tool on each of them, one call per URL. Then return the key information from each "
        "page, grouped by source, with the source URL written next to it. If a page fails "
        "to scrape, skip it and continue with the others. Do NOT ask me for a URL.")]})
    state["scraped_content"] = to_text(res["messages"][-1].content)


def step_writer(topic, state):
    research = f"SEARCH RESULTS:\n{state['search_results']}\n\nSCRAPED CONTENT:\n{state['scraped_content']}"
    state["report"] = writer_chain.invoke({"topic": topic, "research": research})


def step_critic(topic, state):
    state["feedback"] = critic_chain.invoke({"report": state["report"]})


PIPELINE = [("search", step_search), ("reader", step_reader), ("writer", step_writer), ("critic", step_critic)]

def call_with_retry(fn, topic, state, tries=4, wait=15):
    for attempt in range(tries):
        try:
            return fn(topic, state)
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if is_rate_limit and attempt < tries - 1:
                time.sleep(wait)
                continue
            raise

def run_pipeline(topic: str, slot):
    status = {k: "idle" for k, *_ in STEPS}
    secs, state = {}, {"topic": topic}
    for key, fn in PIPELINE:
        status[key] = "running"
        slot.markdown(render_tracker(status, secs), unsafe_allow_html=True)
        start = time.time()
        try:
            call_with_retry(fn, topic, state)
        except Exception as e:
            status[key] = "error"
            slot.markdown(render_tracker(status, secs), unsafe_allow_html=True)
            st.error(f"{key.capitalize()} step failed: {e}")
            return None
        secs[key] = time.time() - start
        status[key] = "done"
        slot.markdown(render_tracker(status, secs), unsafe_allow_html=True)
    state["total_secs"] = sum(secs.values())
    return state


st.markdown('<div class="hero"><h1>ResearchMind </h1>'
            '<h5 class="hero">(Multi-Agent Research Pipeline)</h5>'
            "<p>Four AI agents search the web, read the best source, write a full report and review it.</p></div>",
            unsafe_allow_html=True)

left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown('<div class="lbl">Research Topic</div>', unsafe_allow_html=True)
    topic = st.text_input("Topic", key="topic", placeholder="What do you want to research?",
                          label_visibility="collapsed")
    run = st.button("⚡ Run Research Pipeline", type="primary", use_container_width=True)
    st.markdown('<div class="try">Try →</div>', unsafe_allow_html=True)
    for ex in EXAMPLES:
        st.button(ex, on_click=set_topic, args=(ex,), key=f"ex_{ex}")

with right:
    st.markdown('<div class="panel-title">Pipeline</div>', unsafe_allow_html=True)
    slot = st.empty()

result = st.session_state.get("result")
idle = {k: "idle" for k, *_ in STEPS}
done = {k: "done" for k, *_ in STEPS}
slot.markdown(render_tracker(done if result else idle, {}), unsafe_allow_html=True)

if run:
    if not topic.strip():
        st.warning("Enter a topic first.")
    else:
        st.session_state["result"] = None
        result = run_pipeline(topic.strip(), slot)
        st.session_state["result"] = result

if result:
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", result["feedback"])
    score = float(m.group(1)) if m else None
    s_color = "#7C7C8A" if score is None else "#3DDC97" if score >= 8 else "#FF9F1F" if score >= 6 else "#FB7185"
    s_text = "–" if score is None else f"{score:g}"
    st.markdown(
        f'<div class="result-head"><div class="score" style="--s:{s_color}">'
        f'<span class="score-num">{s_text}</span><span class="score-of">/10</span>'
        f'<div class="score-label">Critic score</div></div>'
        f'<div><div class="result-topic">{result["topic"]}</div>'
        f'<div class="result-meta">Generated in {result["total_secs"]:.0f}s</div></div></div>',
        unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["Report", "Critic feedback", "Search results", "Scraped content"])
    with t1:
        st.markdown(result["report"])
        st.download_button("Download report (.md)", data=result["report"],
                           file_name="research_report.md", mime="text/markdown")
    with t2:
        st.markdown(result["feedback"])
    with t3:
        st.markdown(result["search_results"])
    with t4:
        st.markdown(result["scraped_content"])