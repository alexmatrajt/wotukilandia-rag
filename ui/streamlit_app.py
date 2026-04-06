import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st

from app.config import APP_NAME, PROVIDER
from app.query_engine import ask_question


st.set_page_config(
    page_title=APP_NAME,
    page_icon="⚖️",
    layout="wide",
)

st.title("Wotukilandia Legal Research Assistant")
st.caption(
    "Ask questions about the laws, regulations, cases, memos, case files, and evidence of Wotukilandia."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


with st.sidebar:
    st.header("About")
    st.write(
        "This is a RAG app over your Wotukilandia legal corpus. "
        "It retrieves relevant chunks and answers only from those sources."
    )

    st.header("Current mode")
    st.write(f"**Provider:** {PROVIDER}")

    st.header("Tips")
    st.write("- Ask specific legal questions.")
    st.write("- Try questions about statutes, cases, or the Orin Tal matter.")
    st.write("- Check the source panel after every answer.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask a legal question about Wotukilandia...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Researching the Wotukilandian record..."):
            result = ask_question(question, PROVIDER)

        answer = result["answer"]
        st.markdown(answer)

        with st.expander("Sources used", expanded=True):
            if result["sources"]:
                for i, source in enumerate(result["sources"], start=1):
                    meta = source["metadata"]

                    st.markdown(f"### Source {i}")
                    st.write(f"**Document title:** {meta.get('document_title', '')}")
                    st.write(f"**Source file:** {meta.get('source_file', '')}")
                    st.write(f"**Document type:** {meta.get('document_type', '')}")
                    st.write(f"**Section:** {meta.get('section', '')}")
                    st.write(f"**Article/Subsection:** {meta.get('article', '')}")
                    st.write(f"**Heading:** {meta.get('heading', '')}")

                    score = source.get("score", None)
                    if score is not None:
                        st.write(f"**Retrieval score:** {round(score, 4)}")

                    st.markdown("**Chunk preview:**")
                    st.code(source["text"][:1200], language="text")
                    st.divider()
            else:
                st.write("No sources returned.")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_result = result
