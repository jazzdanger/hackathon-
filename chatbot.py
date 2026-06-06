import os
import sqlite3
import uuid
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

# Community tools for RAG
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader
from docx import Document

# ==========================================
# 1. ENVIRONMENT & GEMINI SETUP
# ==========================================
load_dotenv()

google_gemini_api_key = os.getenv("API_Key")
if not google_gemini_api_key:
    raise ValueError("API_Key not found in .env file")

client = OpenAI(
    api_key=google_gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ==========================================
# 2. SQLITE SHORT/LONG TERM MEMORY
# ==========================================
db_conn = sqlite3.connect("hackathon_memory.db", check_same_thread=False)
db_cursor = db_conn.cursor()

db_cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

db_cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New Chat',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
db_conn.commit()


def create_new_session() -> str:
    session_id = str(uuid.uuid4())[:8]
    db_cursor.execute(
        "INSERT INTO chat_sessions (session_id, title) VALUES (?, ?)",
        (session_id, "New Chat")
    )
    db_conn.commit()
    return session_id


def get_all_sessions() -> list:
    db_cursor.execute(
        "SELECT session_id, title FROM chat_sessions ORDER BY updated_at DESC"
    )
    return db_cursor.fetchall()


def update_session_title(session_id: str, first_message: str):
    title = first_message.strip()[:40]
    if len(first_message.strip()) > 40:
        title += "…"
    db_cursor.execute(
        "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
        (title, session_id)
    )
    db_conn.commit()


def touch_session(session_id: str):
    db_cursor.execute(
        "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
        (session_id,)
    )
    db_conn.commit()


def delete_session(session_id: str):
    db_cursor.execute("DELETE FROM chat_logs WHERE session_id = ?", (session_id,))
    db_cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
    db_conn.commit()


def save_to_memory(session_id: str, role: str, content: str):
    db_cursor.execute(
        "INSERT INTO chat_logs (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    db_conn.commit()


def load_memory_context(session_id: str, limit: int = 6) -> str:
    db_cursor.execute(
        "SELECT role, content FROM chat_logs WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    )
    rows = db_cursor.fetchall()[::-1]
    context = ""
    for role, content in rows:
        context += f"{role.upper()}: {content}\n"
    return context


def load_ui_history(session_id: str) -> list:
    db_cursor.execute(
        "SELECT role, content FROM chat_logs WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    )
    rows = db_cursor.fetchall()
    history = []
    for role, content in rows:
        history.append({"role": role, "content": content})
    return history


def get_message_count(session_id: str) -> int:
    db_cursor.execute(
        "SELECT COUNT(*) FROM chat_logs WHERE session_id = ? AND role = 'user'",
        (session_id,)
    )
    return db_cursor.fetchone()[0]


# ==========================================
# 3. RAG CORE: PARSING & FAISS VECTOR DB
# ==========================================
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = None


def extract_text_from_file(file_path: str) -> str:
    text = ""
    ext = file_path.lower()
    if ext.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    elif ext.endswith(".docx"):
        doc = Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
    elif ext.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    return text


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def process_and_index_document(file) -> str:
    global vector_store
    if file is None:
        return "No file uploaded."
    try:
        raw_text = extract_text_from_file(file.name)
        if not raw_text.strip():
            return "❌ Document is empty or unreadable."
        text_chunks = chunk_text(raw_text)
        if vector_store is None:
            vector_store = FAISS.from_texts(text_chunks, embeddings)
        else:
            vector_store.add_texts(text_chunks)
        return f"✅ Indexed {len(text_chunks)} segments into RAG DB."
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ==========================================
# 4. TOOLS: WEB SEARCH & KNOWLEDGE RETRIEVAL
# ==========================================

def _search_via_duckduckgo_html(query: str) -> str:
    """Scrape DuckDuckGo HTML endpoint — no API key needed, reliable fallback."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    titles   = soup.find_all("a",  class_="result__a",       limit=4)
    snippets = soup.find_all("a",  class_="result__snippet",  limit=4)
    urls     = soup.find_all("a",  class_="result__url",      limit=4)

    if not titles:
        return ""

    lines = []
    for i in range(min(len(titles), len(snippets))):
        title   = titles[i].get_text(strip=True)
        snippet = snippets[i].get_text(strip=True)
        href    = urls[i].get_text(strip=True) if i < len(urls) else ""
        lines.append(f"- {title}: {snippet}" + (f" (Source: {href})" if href else ""))

    return "\n".join(lines)


def _search_via_ddgs_library(query: str) -> str:
    """Try the duckduckgo-search library (may fail if rate-limited)."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return ""
        return "\n".join(
            f"- {r['title']}: {r['body']} (Source: {r['href']})"
            for r in results
        )
    except Exception as e:
        print(f"[DDGS library failed]: {e}")
        return ""


def execute_web_search(query: str) -> str:
    """Try DDGS library first; fall back to HTML scraping."""
    print(f"\n[WEB SEARCH] Query: {query}")

    # Attempt 1: duckduckgo-search library
    result = _search_via_ddgs_library(query)
    if result:
        print(f"[WEB SEARCH] DDGS library succeeded.\n{result[:300]}...")
        return result

    # Attempt 2: HTML scrape fallback
    print("[WEB SEARCH] DDGS library returned nothing, trying HTML scrape...")
    try:
        result = _search_via_duckduckgo_html(query)
        if result:
            print(f"[WEB SEARCH] HTML scrape succeeded.\n{result[:300]}...")
            return result
    except Exception as e:
        print(f"[WEB SEARCH] HTML scrape also failed: {e}")

    print("[WEB SEARCH] Both methods failed — returning empty.")
    return "Web search returned no results. Please try rephrasing your query."


def execute_rag_search(query: str) -> str:
    global vector_store
    if vector_store is None:
        return "No internal enterprise documents have been loaded yet."
    docs = vector_store.similarity_search(query, k=3)
    return "\n---\n".join([doc.page_content for doc in docs])


# ==========================================
# 5. INTELLIGENT AGENT ROUTER LOGIC
# ==========================================
def route_intent(user_query: str, history_context: str) -> str:
    router_prompt = f"""
You are the routing engine of an enterprise AI agent. Analyze the user's query and recent context.
Determine which tool is required. Respond with EXACTLY one of these words: 'GENERAL', 'WEB', or 'RAG'.

CRITERIA:
- Choose 'RAG' if the user asks about company policies, internal documents, HR handbooks,
  specific uploaded summaries, or references an uploaded file.
- Choose 'WEB' if the user asks about real-time events, current news, stock prices, sports scores,
  weather, or anything that requires up-to-date information from the internet.
- Choose 'GENERAL' for coding, general logic, mathematical proofs, definitions, or basic creative writing.

CONVERSATION HISTORY:
{history_context}

USER QUERY: {user_query}
CLASSIFICATION:"""

    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": router_prompt}],
            temperature=0.0
        )
        decision = response.choices[0].message.content.strip().upper()
        # Strip any surrounding punctuation just in case
        for word in ["GENERAL", "WEB", "RAG"]:
            if word in decision:
                return word
        return "GENERAL"
    except Exception as e:
        print(f"[ROUTER ERROR]: {e}")
        return "GENERAL"


# ==========================================
# 6. MASTER ORCHESTRATION ENGINE
# ==========================================
def chat_engine(user_message, chat_history, session_id):
    if not user_message.strip():
        yield "", chat_history, "🤖 Mode 1: General LLM Knowledge", build_session_html(session_id)
        return

    if chat_history is None:
        chat_history = []

    # Auto-title session from first message
    if get_message_count(session_id) == 0:
        update_session_title(session_id, user_message)

    history_context = load_memory_context(session_id)
    mode = route_intent(user_message, history_context)
    print(f"[ROUTER] Mode selected: {mode}")

    tool_context = ""
    mode_label = "🤖 Mode 1: General LLM Knowledge"

    if mode == "RAG":
        tool_context = execute_rag_search(user_message)
        mode_label = "📂 Mode 3: Enterprise RAG DB"
    elif mode == "WEB":
        tool_context = execute_web_search(user_message)
        mode_label = "🌐 Mode 2: Web Search Engine"
        print(f"[WEB CONTEXT LENGTH]: {len(tool_context)} chars")

    # Build system prompt — forceful about using retrieved context
    if mode == "WEB" and tool_context and "no results" not in tool_context.lower():
        system_instruction = f"""You are an intelligent enterprise AI assistant.

CRITICAL: The web has already been searched. The results are in the RETRIEVED WEB CONTEXT below.
You MUST answer using these results directly. Do NOT say you cannot access real-time data —
you already have it. Summarize and cite the sources provided.

CONVERSATION HISTORY:
{history_context}

RETRIEVED WEB CONTEXT (answer from this):
{tool_context}
"""
    elif mode == "RAG" and tool_context:
        system_instruction = f"""You are an intelligent enterprise AI assistant.

Answer the user's question using ONLY the internal document context below.
If the context doesn't contain the answer, say so clearly.

CONVERSATION HISTORY:
{history_context}

INTERNAL DOCUMENT CONTEXT:
{tool_context}
"""
    else:
        system_instruction = f"""You are an intelligent enterprise AI assistant.
Answer the user's question comprehensively using your general knowledge.

CONVERSATION HISTORY:
{history_context}
"""

    api_messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message}
    ]

    save_to_memory(session_id, "user", user_message)
    touch_session(session_id)

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": f"*{mode_label}*\n\n"})
    yield "", chat_history, mode_label, build_session_html(session_id)

    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=api_messages,
            stream=True
        )

        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                chat_history[-1]["content"] += delta
                yield "", chat_history, mode_label, build_session_html(session_id)

        save_to_memory(session_id, "assistant", chat_history[-1]["content"])

    except Exception as e:
        error_msg = f"Execution error: {str(e)}"
        print(f"[CHAT ENGINE ERROR]: {error_msg}")
        chat_history[-1]["content"] += error_msg
        yield "", chat_history, "⚠️ System Failure", build_session_html(session_id)


# ==========================================
# 7. SESSION SIDEBAR HELPERS
# ==========================================

def get_session_choices() -> list:
    """Return [(title, session_id), ...] for the Radio component."""
    sessions = get_all_sessions()
    return [(title, sid) for sid, title in sessions]


def new_chat_session(current_session_id: str):
    new_id = create_new_session()
    choices = get_session_choices()
    return [], "🤖 Mode 1: General LLM Knowledge", new_id, gr.update(choices=choices, value=new_id)


def switch_session(selected_sid: str):
    if not selected_sid:
        return gr.update(), gr.update(), gr.update(), gr.update()
    history = load_ui_history(selected_sid)
    choices = get_session_choices()
    return history, "🤖 Mode 1: General LLM Knowledge", selected_sid, gr.update(choices=choices, value=selected_sid)


def delete_current_session(current_session_id: str):
    delete_session(current_session_id)
    sessions = get_all_sessions()
    if not sessions:
        new_id = create_new_session()
        active_id = new_id
        history = []
    else:
        active_id = sessions[0][0]
        history = load_ui_history(active_id)
    choices = get_session_choices()
    return history, "🤖 Mode 1: General LLM Knowledge", active_id, gr.update(choices=choices, value=active_id)


def clear_current_session(session_id: str):
    db_cursor.execute("DELETE FROM chat_logs WHERE session_id = ?", (session_id,))
    db_cursor.execute(
        "UPDATE chat_sessions SET title = 'New Chat', updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
        (session_id,)
    )
    db_conn.commit()
    choices = get_session_choices()
    return [], "🤖 Mode 1: General LLM Knowledge", gr.update(choices=choices, value=session_id)


def init_app():
    sessions = get_all_sessions()
    if not sessions:
        sid = create_new_session()
    else:
        sid = sessions[0][0]
    history = load_ui_history(sid)
    choices = get_session_choices()
    return history, "🤖 Mode 1: General LLM Knowledge", sid, gr.update(choices=choices, value=sid)


def refresh_sidebar_after_chat(user_message, chat_history, session_id):
    """Wrapper: run chat_engine and also refresh the Radio choices after (for title update)."""
    gen = chat_engine(user_message, chat_history, session_id)
    for msg_box, history, mode, _ in gen:
        choices = get_session_choices()
        yield msg_box, history, mode, gr.update(choices=choices, value=session_id)


# ==========================================
# 8. GRADIO FRONTEND INTERFACE
# ==========================================
css = """
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 8px !important;
    margin: 0 !important;
}
footer { display: none !important; }

/* Sidebar panel */
#sidebar {
    background: #111827 !important;
    border-radius: 10px !important;
    padding: 10px 8px !important;
    min-height: 88vh !important;
}

/* New chat button */
#new-chat-btn button {
    width: 100% !important;
    background: #1d4ed8 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    padding: 8px !important;
    margin-bottom: 8px !important;
}
#new-chat-btn button:hover { background: #2563eb !important; }

/* Delete button */
#delete-chat-btn button {
    width: 100% !important;
    font-size: 12px !important;
    margin-top: 6px !important;
}

/* Radio list — style each item like a chat row */
#session-radio label {
    display: block !important;
    padding: 9px 12px !important;
    margin: 2px 0 !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    border-left: 3px solid transparent !important;
    color: #cdd !important;
    font-size: 13px !important;
    transition: background 0.15s !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
#session-radio label:hover {
    background: #1e293b !important;
}
#session-radio input[type="radio"]:checked + span,
#session-radio .selected {
    background: #1e3a5f !important;
    border-left: 3px solid #3b82f6 !important;
}
/* Hide the radio circle dots */
#session-radio input[type="radio"] { display: none !important; }
#session-radio .wrap { gap: 0 !important; }

/* Chatbot height */
#chatbot {
    height: calc(100vh - 220px) !important;
    min-height: 400px !important;
}

/* Status strip */
#status-strip textarea {
    min-height: 26px !important;
    max-height: 26px !important;
    font-size: 11px !important;
    padding: 3px 8px !important;
    border: none !important;
    background: transparent !important;
    color: #6b7280 !important;
    resize: none !important;
}

/* Mode label */
#mode-label .output-class {
    font-size: 12px !important;
    padding: 3px 8px !important;
}
"""

with gr.Blocks(title="Multi-Source Enterprise Agent") as demo:

    session_store = gr.State("")

    with gr.Row():

        # ── SIDEBAR ──
        with gr.Column(scale=1, elem_id="sidebar", min_width=220):
            gr.Markdown("### 🚀 AI Agent")

            new_chat_btn = gr.Button("＋  New Chat", elem_id="new-chat-btn", variant="primary")

            session_radio = gr.Radio(
                choices=[],
                value=None,
                label="Chats",
                elem_id="session-radio",
                interactive=True,
            )

            delete_chat_btn = gr.Button(
                "🗑 Delete This Chat",
                elem_id="delete-chat-btn",
                variant="secondary",
                size="sm"
            )

        # ── MAIN AREA ──
        with gr.Column(scale=4):

            with gr.Row():
                gr.Markdown("### 💬 Communication Stream")
                active_mode_view = gr.Label(
                    value="🤖 Mode 1: General LLM Knowledge",
                    label="Routing",
                    elem_id="mode-label",
                    scale=1
                )
                clear_btn = gr.Button("✖ Clear Chat", variant="secondary", scale=0, min_width=100)

            chatbot_interface = gr.Chatbot(show_label=False, elem_id="chatbot")

            with gr.Row(equal_height=True):
                upload_btn = gr.UploadButton(
                    "📎",
                    file_types=[".pdf", ".docx", ".txt"],
                    variant="secondary",
                    min_width=46,
                    scale=0
                )
                user_input_field = gr.Textbox(
                    show_label=False,
                    placeholder="Message the agent or upload a document...",
                    container=False,
                    scale=10,
                    autofocus=True
                )

            indexing_status = gr.Textbox(
                show_label=False,
                interactive=False,
                container=False,
                elem_id="status-strip"
            )

    # ── Event wiring ──

    # Page load
    demo.load(
        init_app,
        inputs=[],
        outputs=[chatbot_interface, active_mode_view, session_store, session_radio]
    )

    # New chat
    new_chat_btn.click(
        new_chat_session,
        inputs=[session_store],
        outputs=[chatbot_interface, active_mode_view, session_store, session_radio]
    )

    # Switch session — fires when user clicks a radio item
    session_radio.change(
        switch_session,
        inputs=[session_radio],
        outputs=[chatbot_interface, active_mode_view, session_store, session_radio]
    )

    # Delete current session
    delete_chat_btn.click(
        delete_current_session,
        inputs=[session_store],
        outputs=[chatbot_interface, active_mode_view, session_store, session_radio]
    )

    # Clear current session messages
    clear_btn.click(
        clear_current_session,
        inputs=[session_store],
        outputs=[chatbot_interface, active_mode_view, session_radio]
    )

    # File upload
    upload_btn.upload(
        process_and_index_document,
        inputs=[upload_btn],
        outputs=[indexing_status]
    )

    # Send message
    user_input_field.submit(
        refresh_sidebar_after_chat,
        inputs=[user_input_field, chatbot_interface, session_store],
        outputs=[user_input_field, chatbot_interface, active_mode_view, session_radio]
    )

if __name__ == "__main__":
    demo.queue().launch(share=False, theme=gr.themes.Soft(), css=css)