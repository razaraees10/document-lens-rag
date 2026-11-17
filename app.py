"""Launch: streamlit run app.py"""
from __future__ import annotations

import hashlib
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai

from rag import ANSWER_MODEL, answer_with_context, chunk_pages, embed_texts, extract_pdf_pages, find_relevant_chunks

load_dotenv()
st.set_page_config(page_title="Document Lens · Free Gemini RAG", page_icon="📄", layout="centered")
st.title("📄 Document Lens")
st.caption("PDF Q&A with Gemini embeddings + cosine similarity + grounded answers")

with st.sidebar:
    st.header("1. Gemini API (Free tier)")
    api_key = st.text_input("Google AI Studio API key", type="password", value=os.getenv("GEMINI_API_KEY", ""), help="Create a key in a Free-tier AI Studio project. Keep it private.")
    model = st.text_input("Answer model", value=ANSWER_MODEL, help="Use a model available on your AI Studio project's Free tier.")
    st.caption("Google AI Pro does not include Gemini API billing. A separate free API tier is available with limits.")
    st.header("2. Your PDF")
    upload = st.file_uploader("Upload a text-based PDF", type=["pdf"])
    process = st.button("Process PDF", type="primary", use_container_width=True)
    st.caption("For a demo only. Free-tier requests may be used to improve Google products. Don't upload private documents.")

st.session_state.setdefault("document", None)
st.session_state.setdefault("messages", [])

if process:
    if upload is None:
        st.warning("Choose a PDF first (sample_document.pdf is included in the ZIP).")
    elif not api_key.strip():
        st.warning("Enter your free Google AI Studio API key.")
    else:
        data = upload.getvalue()
        if len(data) > 10 * 1024 * 1024:
            st.error("PDF must be under 10 MB for this demo.")
        else:
            try:
                with st.spinner("Extracting pages, chunking, and generating Gemini embeddings..."):
                    pages = extract_pdf_pages(data)
                    chunks = chunk_pages(pages)
                    if len(chunks) > 80:
                        raise ValueError(f"This demo allows at most 80 chunks; the PDF generated {len(chunks)}.")
                    client = genai.Client(api_key=api_key.strip())
                    vectors = embed_texts(client, [c["text"] for c in chunks])
                    st.session_state.document = {
                        "hash": hashlib.sha256(data).hexdigest(),
                        "name": upload.name,
                        "pages": len(pages),
                        "chunks": chunks,
                        "vectors": vectors,
                    }
                    st.session_state.messages = []
                st.success(f"Indexed {upload.name}: {len(pages)} pages, {len(chunks)} chunks.")
            except Exception as exc:
                st.error(f"Unable to index PDF: {exc}")
                st.info("Check your API key, Free-tier model access and project quotas in Google AI Studio.")

document = st.session_state.document
file_changed = bool(document and upload and hashlib.sha256(upload.getvalue()).hexdigest() != document["hash"])
if document:
    st.info(f"Indexed **{document['name']}** · {document['pages']} text pages · {len(document['chunks'])} chunks")
    if file_changed:
        st.warning("You selected a different PDF. Click **Process PDF** before asking about it.")
else:
    st.info("Upload and process sample_document.pdf to test your chatbot.")

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["text"])
        if item.get("sources"):
            with st.expander("Retrieved PDF passages"):
                for passage in item["sources"]:
                    st.markdown(f"**Page {passage['page']}** · cosine similarity {passage['similarity']:.3f}")
                    st.write(passage["text"])

question = st.chat_input("Ask about the PDF...", disabled=document is None or file_changed)
if question and document:
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not api_key.strip():
            st.warning("Enter your API key first.")
        else:
            try:
                with st.spinner("Retrieving evidence and asking Gemini..."):
                    client = genai.Client(api_key=api_key.strip())
                    sources = find_relevant_chunks(client, question, document["chunks"], document["vectors"])
                    answer = answer_with_context(client, question, sources, model=model)
                st.markdown(answer)
                with st.expander("Retrieved PDF passages"):
                    for passage in sources:
                        st.markdown(f"**Page {passage['page']}** · cosine similarity {passage['similarity']:.3f}")
                        st.write(passage["text"])
                st.session_state.messages.append({"role": "assistant", "text": answer, "sources": sources})
            except Exception as exc:
                st.error(f"Gemini API request failed: {exc}")
                st.caption("Check your project's free-tier model availability, API quota and key.")
