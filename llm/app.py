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

# Backend imports (pure connection to existing modules)
from llm.src.generation.llm import generate_answer
from llm.src.tools.search_tool import answer, fast_answer, get_retriever
from llm.src.config import COLLECTION_NAME, EMBEDDING_MODEL

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Assistant | Explainable AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Modern, Professional AI/RAG-Themed Styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Font and Base Styling */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* App Header */
.app-header {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(56, 189, 248, 0.08) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    backdrop-filter: blur(10px);
}

.header-title {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #6366f1 0%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header-subtitle {
    font-size: 0.92rem;
    color: #94a3b8;
    margin-top: 4px;
    margin-bottom: 0;
}

/* Metric Cards */
.metric-card {
    background: rgba(30, 41, 59, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 14px 18px;
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.metric-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
}

.metric-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f1f5f9;
}

/* Chunk Display Card */
.chunk-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
}

.chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.chunk-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(99, 102, 241, 0.2);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.35);
}

.chunk-score {
    font-size: 0.8rem;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
}

.chunk-body {
    font-size: 0.88rem;
    line-height: 1.6;
    color: #cbd5e1;
    white-space: pre-wrap;
}

/* Mode Pill Badges */
.mode-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 8px;
}

.mode-answer {
    background: rgba(59, 130, 246, 0.15);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.35);
}

.mode-fast {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

/* Example Prompt Pill */
div[data-testid="stButton"] > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s ease;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached Resource Initialization
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_backend_retriever():
    """Cache the retriever client so ChromaDB & BM25 load once."""
    try:
        return get_retriever()
    except Exception as exc:
        return None

retriever_instance = load_backend_retriever()

# ---------------------------------------------------------------------------
# Session State Setup
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar: Configuration, Pipeline Info & System Status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Retrieval Settings")

    mode_options = {
        "🎯 Dense Similarity Search (Answer)": "answer",
        "⚡ Hybrid Search: Dense + BM25 (Fast Answer)": "fast_answer",
    }

    selected_mode_label = st.radio(
        "Retrieval Mode",
        options=list(mode_options.keys()),
        index=0,
        help=(
            "Answer: Dense cosine similarity search over vector embeddings.\n\n"
            "Fast Answer: Hybrid search fusing dense semantic search with BM25 keyword matching via RRF."
        ),
    )
    current_mode = mode_options[selected_mode_label]

    # Visual Mode Description
    if current_mode == "answer":
        st.markdown(
            """
            <div class="mode-badge mode-answer">
                🎯 Mode: Dense Similarity Search (top_k=5)
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.4;">
                Uses OpenRouter vector embeddings to retrieve semantically nearest chunks from ChromaDB.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="mode-badge mode-fast">
                ⚡ Mode: Hybrid Search (RRF Fusion)
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.4;">
                Combines dense semantic vectors + BM25 keyword ranks using Reciprocal Rank Fusion (rrf_k=60).
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📊 Knowledge Base")

    # Document details from existing database
    chunk_count = retriever_instance.collection.count() if retriever_instance else 0
    st.markdown(
        f"""
        <div class="metric-card" style="margin-bottom: 10px;">
            <div class="metric-label">Collection Name</div>
            <div class="metric-value">{COLLECTION_NAME}</div>
        </div>
        <div class="metric-card" style="margin-bottom: 10px;">
            <div class="metric-label">Total Indexed Chunks</div>
            <div class="metric-value">{chunk_count} Chunks</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Source Document</div>
            <div class="metric-value" style="font-size: 0.95rem;">Explainable-AI-for-Practitioners.pdf</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 🤖 Pipeline Architecture")
    st.markdown(
        f"""
        - **LLM**: `openai/gpt-oss-120b` (Groq)
        - **Embedding**: `{EMBEDDING_MODEL or 'liquid/lfm-2.5-embedding-350m:free'}`
        - **Vector Store**: `ChromaDB` (Persistent)
        - **Fusion**: Reciprocal Rank Fusion (k=60)
        """
    )

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div>
            <h1 class="header-title">Explainable AI Assistant</h1>
            <p class="header-subtitle">
                Context-grounded question answering powered by LangChain, Groq LLM & ChromaDB Vector Store
            </p>
        </div>
        <div style="text-align: right;">
            <span class="chunk-badge">RAG Active</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs: Chat Q&A and Retrieval Inspector
# ---------------------------------------------------------------------------
tab_qa, tab_retrieval = st.tabs(["💬 Ask & Generate", "🔍 Retrieval Inspector"])

# ===========================================================================
# TAB 1: Chat & Answer Generation
# ===========================================================================
with tab_qa:
    # Example Questions Section (shown if no chat history yet)
    if len(st.session_state.messages) == 0:
        st.markdown("##### 💡 Suggested Questions from the Knowledge Base:")
        example_prompts = [
            "What is Explainable AI and why is it important?",
            "What are the main differences between interpretable models and black-box models?",
            "How does feature attribution work in machine learning?",
            "What are common challenges when deploying explainability in production?",
        ]
        cols = st.columns(len(example_prompts))
        for idx, prompt_text in enumerate(example_prompts):
            with cols[idx]:
                if st.button(prompt_text, key=f"example_{idx}", use_container_width=True):
                    st.session_state["active_input"] = prompt_text
                    st.rerun()

    # Display Existing Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                mode_label = "⚡ Hybrid (Fast Answer)" if msg.get("mode") == "fast_answer" else "🎯 Dense (Answer)"
                st.caption(f"Mode: **{mode_label}** • Generated in **{msg.get('latency', 0.0):.2f}s**")
            st.markdown(msg["content"])

    # Chat Input Box
    default_prompt = st.session_state.pop("active_input", None)
    user_prompt = st.chat_input("Ask a question about Explainable AI...", key="chat_input_box")

    active_prompt = default_prompt if default_prompt else user_prompt

    if active_prompt:
        # Display user message
        st.session_state.messages.append({
            "role": "user",
            "content": active_prompt,
        })
        with st.chat_message("user"):
            st.markdown(active_prompt)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            status_text = (
                "Retrieving semantically relevant chunks via Dense Search & generating answer..."
                if current_mode == "answer"
                else "Performing Hybrid Search (Dense + BM25 Fusion) & generating answer..."
            )
            with st.spinner(status_text):
                start_time = time.time()
                try:
                    response_text = generate_answer(active_prompt, mode=current_mode)
                    elapsed = time.time() - start_time
                except Exception as exc:
                    response_text = f"❌ An error occurred during answer generation: `{str(exc)}`"
                    elapsed = time.time() - start_time

            mode_label = "⚡ Hybrid (Fast Answer)" if current_mode == "fast_answer" else "🎯 Dense (Answer)"
            st.caption(f"Mode: **{mode_label}** • Generated in **{elapsed:.2f}s**")
            st.markdown(response_text)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "mode": current_mode,
                "latency": elapsed,
            })

# ===========================================================================
# TAB 2: Retrieval Inspector (Direct inspection of existing search tools)
# ===========================================================================
with tab_retrieval:
    st.markdown("#### 🔎 Inspect Raw Chunk Retrieval")
    st.caption("Directly test the backend search tools (`answer` and `fast_answer`) to inspect retrieved document chunks and relevance scores.")

    col_q, col_btn = st.columns([4, 1])
    with col_q:
        inspect_query = st.text_input(
            "Search Query",
            placeholder="e.g. feature attribution methods, SHAP, LIME",
            label_visibility="collapsed",
        )
    with col_btn:
        run_inspect = st.button("Retrieve Chunks", type="primary", use_container_width=True)

    if run_inspect and inspect_query.strip():
        with st.spinner("Executing retrieval..."):
            start_search = time.time()
            if current_mode == "answer":
                chunks = answer.invoke({"query": inspect_query.strip()})
            else:
                chunks = fast_answer.invoke({"query": inspect_query.strip()})
            search_duration = time.time() - start_search

        if not chunks:
            st.warning("No chunks were retrieved for the given query.")
        else:
            st.success(f"Retrieved **{len(chunks)}** chunks in **{search_duration:.2f}s** using **{current_mode}** tool.")

            for rank, chunk in enumerate(chunks, start=1):
                chunk_id = chunk.get("id", "N/A")
                meta = chunk.get("metadata", {})
                page = meta.get("page", meta.get("page_label", "N/A"))
                source = Path(str(meta.get("source", "Unknown"))).name
                score_str = ""

                if "distance" in chunk and chunk["distance"] is not None:
                    score_str = f"Distance: {chunk['distance']:.4f}"
                elif "rrf_score" in chunk and chunk["rrf_score"] is not None:
                    score_str = f"RRF Score: {chunk['rrf_score']:.4f}"

                st.markdown(
                    f"""
                    <div class="chunk-card">
                        <div class="chunk-header">
                            <div>
                                <span class="chunk-badge">Rank #{rank}</span>
                                <span style="font-size: 0.8rem; color: #94a3b8; margin-left: 8px;">
                                    Page {page} • {source}
                                </span>
                            </div>
                            <div class="chunk-score">{score_str}</div>
                        </div>
                        <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 6px; font-family: 'JetBrains Mono', monospace;">
                            ID: {chunk_id}
                        </div>
                        <div class="chunk-body">{chunk.get('text', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander(f"View Metadata & Full Details (Rank #{rank})"):
                    st.json({
                        "id": chunk_id,
                        "metadata": meta,
                        "distance": chunk.get("distance"),
                        "rrf_score": chunk.get("rrf_score"),
                    })
    elif run_inspect:
        st.info("Please enter a search query.")
