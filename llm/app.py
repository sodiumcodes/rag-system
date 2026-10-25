"""

Streamlit Web UI for RAG System.

Connects directly to the existing backend pipeline in llm/src.

"""



import sys

import time

from pathlib import Path

from dotenv import load_dotenv

import streamlit as st



# ---------------------------------------------------------------------------

# Path & Environment Setup

# ---------------------------------------------------------------------------

app_dir = Path(__file__).resolve().parent

project_root = app_dir.parent if app_dir.name == "llm" else app_dir

if str(project_root) not in sys.path:

    sys.path.insert(0, str(project_root))



# Load environment variables

env_file = app_dir / ".env"

if env_file.exists():

    load_dotenv(dotenv_path=env_file)

load_dotenv()



# Backend imports (strictly using existing modules)

from llm.src.generation.llm import generate_answer

from llm.src.tools.search_tool import get_retriever

from llm.src.config import COLLECTION_NAME, EMBEDDING_MODEL



# ---------------------------------------------------------------------------

# Page Configuration

# ---------------------------------------------------------------------------

st.set_page_config(

    page_title="Explainable AI | RAG Intelligence",

    page_icon="⚡",

    layout="wide",

    initial_sidebar_state="expanded",

)



# ---------------------------------------------------------------------------

# High-End Dark Aesthetic Custom CSS

# ---------------------------------------------------------------------------

PREMIUM_DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #080d16;
    --panel: rgba(17, 25, 40, 0.82);
    --panel-strong: #111a2a;
    --line: rgba(148, 163, 184, 0.16);
    --text: #edf2fa;
    --muted: #9aa9bf;
    --accent: #8b9cff;
    --accent-2: #55d6be;
}

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text) !important;
}
.stApp {
    background:
      radial-gradient(ellipse at 50% -15%, rgba(83, 103, 180, .19), transparent 48%),
      radial-gradient(ellipse at 100% 40%, rgba(43, 155, 143, .07), transparent 36%),
      var(--bg) !important;
    background-attachment: fixed !important;
}
.block-container { max-width: 1480px; padding-top: 1.7rem; padding-bottom: 2.5rem; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(139, 156, 255, .28); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(139, 156, 255, .5); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1220 0%, #080d16 100%) !important;
    border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] > div { padding-top: 1.25rem; }
[data-testid="stSidebar"] * { color: #cbd5e1; }
[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important; font-size: .78rem; font-weight: 800;
    letter-spacing: .09em; text-transform: uppercase; margin: 1.25rem 0 .7rem;
}
.sidebar-brand {
    display: flex; align-items: center; gap: 12px; padding: 15px 16px;
    background: linear-gradient(135deg, rgba(139,156,255,.14), rgba(85,214,190,.06));
    border: 1px solid rgba(139,156,255,.22); border-radius: 15px; margin-bottom: 22px;
}
.sidebar-brand-icon {
    width: 40px; height: 40px; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #7c8cff, #55d6be); border-radius: 12px;
    font-size: 1.2rem; box-shadow: 0 7px 20px rgba(93,120,220,.24);
}
.sidebar-brand-text { font-size: .98rem; font-weight: 800; color: #fff; line-height: 1.25; }
.sidebar-brand-sub { font-size: .67rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .1em; margin-top: 3px; }

.stat-tile {
    background: rgba(17, 26, 42, .76); border: 1px solid var(--line);
    border-radius: 12px; padding: 13px 15px; margin-bottom: 10px;
    transition: border-color .2s ease, transform .2s ease, background .2s ease;
}
.stat-tile:hover { border-color: rgba(139,156,255,.35); background: rgba(24,35,55,.9); transform: translateY(-1px); }
.stat-tile-label { font-size: .66rem; font-weight: 700; text-transform: uppercase; letter-spacing: .09em; color: #8291a9; margin-bottom: 5px; }
.stat-tile-val { font-size: .91rem; font-weight: 700; color: #f1f5f9; overflow-wrap: anywhere; }
.live-pill {
    display: inline-flex; align-items: center; gap: 7px; font-size: .7rem; font-weight: 700;
    color: #6ee7c8; background: rgba(16,185,129,.1); border: 1px solid rgba(52,211,153,.25);
    padding: 6px 11px; border-radius: 999px; white-space: nowrap;
}
.pulsing-dot { width: 7px; height: 7px; background: #34d399; border-radius: 50%; box-shadow: 0 0 10px rgba(52,211,153,.7); display: inline-block; }

.app-header-box {
    display: flex; align-items: center; justify-content: space-between; gap: 24px;
    padding: 26px 30px; margin-bottom: 24px; border-radius: 20px;
    background: linear-gradient(125deg, rgba(23,33,53,.94), rgba(13,19,32,.88));
    border: 1px solid rgba(139,156,255,.2); box-shadow: 0 18px 50px rgba(0,0,0,.2);
    backdrop-filter: blur(16px);
}
.header-title-text {
    font-size: clamp(1.6rem, 3vw, 2.25rem); font-weight: 800; letter-spacing: -.045em;
    line-height: 1.15; background: linear-gradient(95deg, #fff 10%, #cbd5e1 58%, #a5b4fc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
}
.header-subtitle-text { font-size: .91rem; color: var(--muted); margin: 9px 0 0; line-height: 1.65; }

.welcome-hero { text-align: center; padding: 34px 20px 22px; margin-bottom: 12px; }
.welcome-badge {
    display: inline-flex; align-items: center; gap: 8px; background: rgba(139,156,255,.1);
    border: 1px solid rgba(139,156,255,.26); color: #b8c1ff; font-size: .76rem;
    font-weight: 700; padding: 7px 14px; border-radius: 999px; margin-bottom: 18px;
}
.welcome-title { font-size: clamp(1.7rem, 3.2vw, 2.35rem); font-weight: 800; letter-spacing: -.04em; color: #f4f7fc; margin: 0 0 12px; }
.welcome-desc { font-size: .94rem; color: #a2afc3; max-width: 700px; margin: 0 auto 20px; line-height: 1.8; }

.prompt-card {
    background: linear-gradient(150deg, rgba(20,29,46,.92), rgba(13,20,33,.92));
    border: 1px solid rgba(148,163,184,.14); border-radius: 16px; padding: 19px 19px 17px;
    min-height: 158px; transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease;
}
.prompt-card:hover { border-color: rgba(139,156,255,.42); transform: translateY(-3px); box-shadow: 0 14px 30px rgba(0,0,0,.22); }
.prompt-card-icon { font-size: 1.5rem; margin-bottom: 12px; }
.prompt-card-title { font-size: .9rem; font-weight: 800; color: #f1f5f9; margin-bottom: 7px; }
.prompt-card-body { font-size: .78rem; color: #9eacc1; line-height: 1.65; }

[data-testid="stChatMessage"] {
    background: rgba(17,26,42,.74) !important; border: 1px solid rgba(148,163,184,.14) !important;
    border-radius: 16px !important; padding: 18px 22px !important; margin-bottom: 14px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,.12) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(83,103,180,.13), rgba(17,26,42,.86)) !important;
    border-color: rgba(139,156,255,.25) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) { border-left: 3px solid #8797ff !important; }
.meta-chip {
    display: inline-flex; align-items: center; gap: 6px; font-size: .7rem; font-weight: 600;
    color: #a5b4c8; background: rgba(255,255,255,.035); border: 1px solid rgba(148,163,184,.14);
    padding: 5px 9px; border-radius: 7px;
}
.mode-badge { display: inline-flex; align-items: center; gap: 7px; padding: 7px 11px; border-radius: 9px; font-size: .76rem; font-weight: 700; margin-bottom: 10px; }
.mode-answer { background: rgba(96,165,250,.1); color: #a5caff; border: 1px solid rgba(96,165,250,.25); }
.mode-fast { background: rgba(52,211,153,.1); color: #79e8c3; border: 1px solid rgba(52,211,153,.24); }

[data-testid="stChatInput"] { border: 1px solid rgba(139,156,255,.22) !important; border-radius: 16px !important; box-shadow: 0 12px 32px rgba(0,0,0,.22) !important; background: rgba(15,23,42,.96) !important; }
[data-testid="stChatInput"] > div { background: transparent !important; border-radius: 16px !important; }
[data-testid="stChatInput"] textarea {
    background: transparent !important; color: #f1f5f9 !important; border: none !important;
    border-radius: 14px !important; font-size: .94rem !important; caret-color: #a5b4fc !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #718096 !important; opacity: 1 !important; }
[data-testid="stChatInput"]:focus-within { border-color: rgba(139,156,255,.65) !important; box-shadow: 0 0 0 3px rgba(139,156,255,.1), 0 12px 32px rgba(0,0,0,.2) !important; }
[data-testid="stChatInput"] button { color: #dbe4ff !important; background: rgba(139,156,255,.14) !important; border-radius: 10px !important; }
[data-testid="stChatInput"] button:hover { background: rgba(139,156,255,.28) !important; }

.stButton > button {
    background: rgba(27,38,57,.88) !important; color: #dbe4f2 !important;
    border: 1px solid rgba(148,163,184,.17) !important; border-radius: 11px !important;
    font-weight: 700 !important; min-height: 42px; transition: all .2s ease !important;
}
.stButton > button:hover { background: rgba(49,63,91,.92) !important; border-color: rgba(139,156,255,.42) !important; color: #fff !important; transform: translateY(-1px); }
.stRadio label { color: #d5deeb !important; font-size: .84rem !important; }
hr { border: none !important; height: 1px !important; background: var(--line) !important; margin: 20px 0 !important; }

@media (max-width: 800px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .app-header-box { padding: 21px; align-items: flex-start; flex-direction: column; }
    .app-header-box > div:last-child { align-items: flex-start !important; text-align: left !important; }
    .welcome-hero { padding: 25px 8px 15px; }
    [data-testid="stChatMessage"] { padding: 15px !important; }
}
</style>
"""

st.markdown(PREMIUM_DARK_CSS, unsafe_allow_html=True)



# ---------------------------------------------------------------------------

# Backend Resource Initialization (Cached)

# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)

def load_backend_retriever():

    """Retrieve database metadata without reloading ChromaDB each turn."""

    try:

        return get_retriever()

    except Exception:

        return None



retriever_instance = load_backend_retriever()



# ---------------------------------------------------------------------------

# Session State Setup

# ---------------------------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []



# ---------------------------------------------------------------------------

# Sidebar: Brand, Mode Controls, Knowledge Stats & Specs

# ---------------------------------------------------------------------------

with st.sidebar:

    # Top Brand Emblem

    st.markdown(

        """

        <div class="sidebar-brand">

            <div class="sidebar-brand-icon">⚡</div>

            <div>

                <div class="sidebar-brand-text">XAI Intelligence</div>

                <div class="sidebar-brand-sub">RAG System v1.0</div>

            </div>

        </div>

        """,

        unsafe_allow_html=True,

    )



    st.markdown("### ⚙️ Retrieval Strategy")



    mode_options = {

        "⚡ Hybrid Fusion (BM25 + Dense)": "fast_answer",

        "🎯 Dense Vector Cosine (Answer)": "answer",

    }



    selected_mode_label = st.radio(

        "Mode Selection",

        options=list(mode_options.keys()),

        index=0,

        label_visibility="collapsed",

        help=(

            "Hybrid Fusion: Combines dense vector semantics with BM25 keyword matching via Reciprocal Rank Fusion (Fast & Precise).\n\n"

            "Dense Vector Cosine: Standard dense embedding similarity search."

        ),

    )

    current_mode = mode_options[selected_mode_label]



    if current_mode == "fast_answer":

        st.markdown(

            """

            <div class="mode-badge mode-fast">

                ⚡ Hybrid Search Active (RRF k=60)

            </div>

            <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.45; margin-bottom: 12px;">

                Fuses BM25 keyword rankings with dense vector embeddings for maximum precision and recall.

            </div>

            """,

            unsafe_allow_html=True,

        )

    else:

        st.markdown(

            """

            <div class="mode-badge mode-answer">

                🎯 Dense Vector Search (top_k=5)

            </div>

            <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.45; margin-bottom: 12px;">

                Calculates cosine distance over OpenRouter embeddings stored in ChromaDB.

            </div>

            """,

            unsafe_allow_html=True,

        )



    st.markdown("---")

    st.markdown("### 📚 Knowledge Base")



    chunk_count = retriever_instance.collection.count() if retriever_instance else 174

    st.markdown(

        f"""

        <div class="stat-tile">

            <div class="stat-tile-label">Vector Collection</div>

            <div class="stat-tile-val">{COLLECTION_NAME}</div>

        </div>

        <div class="stat-tile">

            <div class="stat-tile-label">Indexed Knowledge Chunks</div>

            <div class="stat-tile-val" style="color: #38bdf8;">{chunk_count} Chunks</div>

        </div>

        <div class="stat-tile">

            <div class="stat-tile-label">Document Source</div>

            <div class="stat-tile-val" style="font-size: 0.85rem; font-weight: 600;">

                Explainable-AI-for-Practitioners.pdf

            </div>

        </div>

        <div class="stat-tile">

            <div class="stat-tile-label">Document Metadata</div>

            <div class="stat-tile-val" style="font-size: 0.82rem; color: #cbd5e1;">

                74 Pages • Michael Munn & David Pitman

            </div>

        </div>

        """,

        unsafe_allow_html=True,

    )



    st.markdown("---")

    st.markdown("### 🧩 Architecture Specs")

    st.markdown(

        f"""

        <div style="font-size: 0.8rem; line-height: 1.7; color: #94a3b8;">

            • <b>LLM:</b> <code style="color: #a5b4fc;">openai/gpt-oss-120b</code><br>

            • <b>Engine:</b> <span style="color: #f1f5f9;">Groq Hosted Inference</span><br>

            • <b>Embeddings:</b> <code style="color: #38bdf8;">{EMBEDDING_MODEL or 'liquid/lfm-2.5-350m'}</code><br>

            • <b>Vector Store:</b> <span style="color: #f1f5f9;">ChromaDB (Persistent)</span><br>

            • <b>Keyword Search:</b> <span style="color: #f1f5f9;">BM25 Okapi Algorithm</span>

        </div>

        """,

        unsafe_allow_html=True,

    )



    st.markdown("---")

    if st.button("🗑️ Clear Conversation", use_container_width=True):

        st.session_state.messages = []

        st.rerun()



# ---------------------------------------------------------------------------

# Main Header Banner

# ---------------------------------------------------------------------------

st.markdown(

    """

    <div class="app-header-box">

        <div>

            <h1 class="header-title-text">Explainable AI Assistant</h1>

            <p class="header-subtitle-text">

                Context-grounded reasoning & question answering over <i>Explainable AI for Practitioners</i>

            </p>

        </div>

        <div style="text-align: right; display: flex; flex-direction: column; align-items: flex-end; gap: 8px;">

            <div class="live-pill">

                <span class="pulsing-dot"></span> ChromaDB Connected

            </div>

            <span style="font-size: 0.75rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">

                Ground-Truth Verifier

            </span>

        </div>

    </div>

    """,

    unsafe_allow_html=True,

)



# ---------------------------------------------------------------------------

# Main Chat Area

# ---------------------------------------------------------------------------

# Welcome Hero & Clickable Prompt Cards (rendered when no messages exist)

if len(st.session_state.messages) == 0:

    st.markdown(

        """

        <div class="welcome-hero">

            <div class="welcome-badge">

                <span>✨</span> Document-Grounded Knowledge Assistant

            </div>

            <h2 class="welcome-title">What would you like to explore?</h2>

            <p class="welcome-desc">

                Ask any question regarding Explainable AI concepts, black-box vs interpretable models, feature attribution methods (SHAP, LIME), or production deployment challenges.

            </p>

        </div>

        """,

        unsafe_allow_html=True,

    )



    # 4 Structured Clickable Cards

    prompt_cards = [

        {

            "icon": "🔍",

            "title": "Concept Overview",

            "desc": "What is Explainable AI and why is it essential for machine learning systems?",

            "prompt": "What is Explainable AI and why is it important?",

        },

        {

            "icon": "⚖️",

            "title": "Interpretable vs Black-Box",

            "desc": "Compare inherently interpretable models against complex black-box architectures.",

            "prompt": "What are the main differences between interpretable models and black-box models?",

        },

        {

            "icon": "📊",

            "title": "Feature Attribution",

            "desc": "Explain how feature attribution techniques like SHAP and LIME assign importance.",

            "prompt": "How does feature attribution work in machine learning?",

        },

        {

            "icon": "🚀",

            "title": "Production Deployment",

            "desc": "Key challenges and operational considerations when putting XAI into production.",

            "prompt": "What are common challenges when deploying explainability in production?",

        },

    ]



    card_cols = st.columns(4)

    for i, card in enumerate(prompt_cards):

        with card_cols[i]:

            card_html = f"""

            <div class="prompt-card">

                <div class="prompt-card-icon">{card['icon']}</div>

                <div class="prompt-card-title">{card['title']}</div>

                <div class="prompt-card-body">{card['desc']}</div>

            </div>

            """

            st.markdown(card_html, unsafe_allow_html=True)

            if st.button("Ask Question →", key=f"btn_prompt_{i}", use_container_width=True):

                st.session_state["active_input"] = card["prompt"]

                st.rerun()



    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)



# Display Existing Chat Messages

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        if msg["role"] == "assistant":

            mode_tag = "⚡ Hybrid RRF" if msg.get("mode") == "fast_answer" else "🎯 Dense Vector"

            latency = msg.get("latency", 0.0)

            st.markdown(

                f"""

                <div style="display: flex; gap: 8px; margin-bottom: 10px;">

                    <span class="meta-chip">Mode: <b style="color: #38bdf8;">{mode_tag}</b></span>

                    <span class="meta-chip">Latency: <b style="color: #34d399;">{latency:.2f}s</b></span>

                    <span class="meta-chip">Source: <b style="color: #cbd5e1;">XAI for Practitioners</b></span>

                </div>

                """,

                unsafe_allow_html=True,

            )

        st.markdown(msg["content"])



# ---------------------------------------------------------------------------

# Chat Input & Answer Generation Flow

# ---------------------------------------------------------------------------

default_prompt = st.session_state.pop("active_input", None)

user_prompt = st.chat_input("Ask a question about Explainable AI...", key="chat_input_box")



active_prompt = default_prompt if default_prompt else user_prompt



if active_prompt:

    # 1. Append and render user message

    st.session_state.messages.append({

        "role": "user",

        "content": active_prompt,

    })

    with st.chat_message("user"):

        st.markdown(active_prompt)



    # 2. Generate and render assistant response

    with st.chat_message("assistant"):

        spinner_msg = (

            "Performing Hybrid Search (BM25 + Dense RRF) & synthesizing grounded response..."

            if current_mode == "fast_answer"

            else "Retrieving semantically closest vector chunks & generating answer..."

        )

        with st.spinner(spinner_msg):

            start_time = time.time()

            try:

                response_text = generate_answer(active_prompt, mode=current_mode)

                elapsed = time.time() - start_time

            except Exception as exc:

                response_text = f"❌ **An error occurred during generation:** \`{str(exc)}\`"

                elapsed = time.time() - start_time



        mode_tag = "⚡ Hybrid RRF" if current_mode == "fast_answer" else "🎯 Dense Vector"

        st.markdown(

            f"""

            <div style="display: flex; gap: 8px; margin-bottom: 10px;">

                <span class="meta-chip">Mode: <b style="color: #38bdf8;">{mode_tag}</b></span>

                <span class="meta-chip">Latency: <b style="color: #34d399;">{elapsed:.2f}s</b></span>

                <span class="meta-chip">Source: <b style="color: #cbd5e1;">XAI for Practitioners</b></span>

            </div>

            """,

            unsafe_allow_html=True,

        )

        st.markdown(response_text)



        st.session_state.messages.append({

            "role": "assistant",

            "content": response_text,

            "mode": current_mode,

            "latency": elapsed,

        })
