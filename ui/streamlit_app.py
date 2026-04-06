import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st

from app.config import APP_NAME, PROVIDER, DATA_DIR
from app.query_engine import ask_question


st.set_page_config(
    page_title=APP_NAME,
    page_icon="⚖️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .hero {
            padding: 1.4rem 1.4rem 1.1rem 1.4rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #10233d 0%, #1f3b63 100%);
            color: #f8fafc;
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2rem;
            color: #f8fafc;
        }

        .hero p {
            margin: 0.2rem 0;
            color: #e2e8f0;
        }

        .info-card {
            padding: 1rem 1.1rem;
            border: 1px solid #dbe4f0;
            border-radius: 16px;
            background: #f8fbff;
            color: #10233d;
            margin-bottom: 0.8rem;
        }

        .info-card strong,
        .info-card div,
        .info-card p,
        .info-card li {
            color: #10233d;
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
            color: #10233d;
        }

        .source-card {
            padding: 0.9rem 1rem;
            border: 1px solid #dbe4f0;
            border-radius: 14px;
            background: #f8fbff;
            color: #10233d;
            margin-bottom: 0.8rem;
        }

        .source-card strong,
        .source-card div,
        .source-card p,
        .source-card li {
            color: #10233d;
        }

        .pill {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: #e8f0ff;
            color: #1d4ed8;
            font-size: 0.8rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }

        .small-muted {
            color: #475467;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


def list_data_files():
    files = []
    for file_path in sorted(DATA_DIR.rglob("*.txt")):
        relative = file_path.relative_to(DATA_DIR)
        parts = relative.parts
        top_folder = parts[0] if len(parts) > 0 else ""
        subfolder = "/".join(parts[1:-1]) if len(parts) > 2 else ""
        files.append(
            {
                "name": file_path.name,
                "path": file_path,
                "relative_path": str(relative),
                "folder": top_folder,
                "subfolder": subfolder,
            }
        )
    return files


def read_text_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def render_intro() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Wotukilandia Legal Research Workspace</h1>
            <p><strong>About Wotukilandia:</strong> Wotukilandia is a fictional interstellar jurisdiction with its own constitution, statutes, regulations, courts, and evidentiary rules.</p>
            <p><strong>About us:</strong> We are a legal team operating in Wotukilandia and using this workspace to analyze law, precedent, and evidence in active matters.</p>
            <p><strong>Current active matter:</strong> Orin Tal v. Synaptech Dynamics</p>
            <p>This site allows you to ask questions the way a lawyer working the case would: about governing law, past cases, legal risks, and supporting evidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_case_overview() -> None:
    st.markdown(
        """
        <div class="info-card">
            <div class="section-title">Active matter</div>
            <div><strong>Case:</strong> Orin Tal v. Synaptech Dynamics</div>
            <div class="small-muted" style="margin-top:0.4rem;">
                Alleged unauthorized deep cognitive access during a diagnostic session.
            </div>
            <div style="margin-top:0.6rem;">
                <span class="pill">Case file</span>
                <span class="pill">Evidence bundle</span>
                <span class="pill">Past precedent</span>
                <span class="pill">Telepathic privacy</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_example_questions() -> None:
    st.markdown(
        """
        <div class="info-card">
            <div class="section-title">Example questions</div>
            <ul>
                <li>What evidence suggests unauthorized deep access?</li>
                <li>Does the evidence show a layered consent violation?</li>
                <li>What arguments can Synaptech make?</li>
                <li>What precedent is most relevant to the Orin Tal matter?</li>
                <li>Can degraded memory echoes be used as evidence?</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary(answer: str) -> None:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="section-title">Summary</div>
            <div>{answer}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: list) -> None:
    st.markdown("### Evidence and supporting sources")

    if not sources:
        st.info("No sources returned.")
        return

    for i, source in enumerate(sources, start=1):
        meta = source.get("metadata", {})
        score = source.get("score", None)

        document_title = meta.get("document_title", "Unknown source")
        source_file = meta.get("source_file", "")
        document_type = meta.get("document_type", "")
        section = meta.get("section", "")
        article = meta.get("article", "")
        heading = meta.get("heading", "")
        subfolder = meta.get("subfolder", "")

        st.markdown(
            f"""
            <div class="source-card">
                <div class="section-title">Source {i}: {document_title}</div>
                <div class="small-muted" style="margin-bottom:0.45rem;">
                    <strong>Type:</strong> {document_type or 'N/A'} |
                    <strong>File:</strong> {source_file or 'N/A'}
                </div>
                <div class="small-muted" style="margin-bottom:0.45rem;">
                    <strong>Section:</strong> {section or 'N/A'} |
                    <strong>Article/Subsection:</strong> {article or 'N/A'} |
                    <strong>Heading:</strong> {heading or 'N/A'}
                </div>
                <div class="small-muted">
                    <strong>Case folder:</strong> {subfolder or 'N/A'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if score is not None:
            st.caption(f"Retrieval score: {round(score, 4)}")

        with st.expander(f"Open chunk preview for Source {i}", expanded=False):
            st.code(source["text"][:1500], language="text")


def render_document_explorer() -> None:
    st.subheader("Document Explorer")
    st.write("Browse the full Wotukilandia corpus, including laws, cases, memos, case files, and evidence.")

    all_files = list_data_files()
    if not all_files:
        st.warning("No data files found.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        folder_options = ["all"] + sorted({f["folder"] for f in all_files})
        selected_folder = st.selectbox("Filter by category", folder_options)

    filtered_files = all_files
    if selected_folder != "all":
        filtered_files = [f for f in filtered_files if f["folder"] == selected_folder]

    with col2:
        labels = [
            f"{f['relative_path']}"
            for f in filtered_files
        ]
        selected_label = st.selectbox("Open a document", labels)

    selected_file = next(f for f in filtered_files if f["relative_path"] == selected_label)
    content = read_text_file(selected_file["path"])

    meta_col, read_col = st.columns([1, 2], gap="large")

    with meta_col:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="section-title">Document details</div>
                <div><strong>File:</strong> {selected_file["name"]}</div>
                <div><strong>Category:</strong> {selected_file["folder"]}</div>
                <div><strong>Subfolder:</strong> {selected_file["subfolder"] or 'N/A'}</div>
                <div><strong>Path:</strong> {selected_file["relative_path"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with read_col:
        st.markdown("### Full text")
        st.text_area(
            "Document contents",
            value=content,
            height=650,
            label_visibility="collapsed",
        )


with st.sidebar:
    st.header("Workspace")
    st.write(f"**Mode:** {PROVIDER}")
    st.write("**Jurisdiction:** Wotukilandia")
    st.write("**Current active case:** Orin Tal v. Synaptech Dynamics")

    st.divider()

    st.header("What this site does")
    st.write(
        "This tool searches the Wotukilandian legal record and current matter materials "
        "so a lawyer can quickly find governing law, precedent, and evidence."
    )

    st.divider()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()


tab_research, tab_explorer = st.tabs(["Research Assistant", "Document Explorer"])

with tab_research:
    render_intro()

    top_left, top_right = st.columns([1.6, 1.1], gap="large")
    with top_left:
        render_case_overview()
    with top_right:
        render_example_questions()

    st.markdown("---")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a legal question about the active Wotukilandia matter...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Reviewing the Wotukilandian record..."):
                result = ask_question(question, PROVIDER)

            answer = result["answer"]
            st.markdown("## Response")

            summary_col, stats_col = st.columns([1.5, 0.8], gap="large")

            with summary_col:
                render_summary(answer)

            with stats_col:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="section-title">Retrieved sources</div>
                        <div style="font-size:2rem; font-weight:700; color:#10233d;">{len(result["sources"])}</div>
                        <div class="small-muted">Supporting documents</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            render_sources(result["sources"])

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.last_result = result

with tab_explorer:
    render_document_explorer()
