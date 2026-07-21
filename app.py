"""
Streamlit RAG Pipeline App
Pipeline: PyPDFLoader -> RecursiveCharacterTextSplitter -> HuggingFace Embeddings
          -> FAISS vector store -> retriever -> PromptTemplate -> Gemini LLM -> answer

This mirrors the notebook pipeline the user built, wrapped in a Streamlit UI.
"""

import os
import tempfile

import streamlit as st
import faiss

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG Pipeline", page_icon="📄", layout="wide")
st.title("📄 RAG Pipeline — Chat with your PDF")
st.caption(
    "PyPDFLoader → RecursiveCharacterTextSplitter → MiniLM Embeddings → FAISS → Gemini"
)

PROMPT_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""

EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vector_store(pdf_path: str, chunk_size: int, chunk_overlap: int, _embeddings):
    """Load PDF, chunk it, embed it, and build a FAISS vector store."""
    loader = PyPDFLoader(pdf_path)
    pdf_data = loader.load()

    chunker = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunked_data = chunker.split_documents(pdf_data)

    index = faiss.IndexFlatL2(EMBED_DIM)
    vector_store = FAISS(
        embedding_function=_embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(chunked_data)
    return vector_store, len(chunked_data), len(pdf_data)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_store, k: int, model_name: str, api_key: str):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "Google API Key (Gemini)",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        help="Get a key from https://aistudio.google.com/app/apikey",
    )

    model_name = st.selectbox(
        "Gemini model",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite","gemini-3.1-flash-lite"],
        index=0,
    )

    st.divider()
    st.subheader("Chunking")
    chunk_size = st.slider("Chunk size", 100, 2000, 500, step=50)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 100, step=25)

    st.divider()
    st.subheader("Retrieval")
    top_k = st.slider("Top-k chunks retrieved", 1, 20, 10)

    st.divider()
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    build_clicked = st.button("Build / Rebuild index", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "index_info" not in st.session_state:
    st.session_state.index_info = None

# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------
if build_clicked:
    if uploaded_file is None:
        st.sidebar.error("Please upload a PDF first.")
    elif not api_key:
        st.sidebar.error("Please provide a Google API key.")
    else:
        with st.spinner("Loading PDF, chunking, and embedding... this may take a moment"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            embeddings = load_embeddings()
            vector_store, n_chunks, n_pages = build_vector_store(
                tmp_path, chunk_size, chunk_overlap, embeddings
            )
            os.unlink(tmp_path)

            rag_chain, retriever = build_rag_chain(
                vector_store, top_k, model_name, api_key
            )

            st.session_state.vector_store = vector_store
            st.session_state.rag_chain = rag_chain
            st.session_state.retriever = retriever
            st.session_state.index_info = {
                "filename": uploaded_file.name,
                "pages": n_pages,
                "chunks": n_chunks,
            }
            st.session_state.messages = []
        st.sidebar.success("Index built successfully!")

if st.session_state.index_info:
    info = st.session_state.index_info
    st.sidebar.info(
        f"**{info['filename']}**\n\n{info['pages']} pages → {info['chunks']} chunks indexed"
    )

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for i, doc in enumerate(msg["sources"], start=1):
                    page = doc.metadata.get("page_label", doc.metadata.get("page", "?"))
                    st.markdown(f"**Chunk {i} (page {page})**")
                    st.text(doc.page_content)

question = st.chat_input("Ask a question about the PDF...")

if question:
    if st.session_state.rag_chain is None:
        st.error("Please upload a PDF and click 'Build / Rebuild index' first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                sources = st.session_state.retriever.invoke(question)
                answer = st.session_state.rag_chain.invoke(question)
                st.markdown(answer)
                with st.expander("Sources"):
                    for i, doc in enumerate(sources, start=1):
                        page = doc.metadata.get("page_label", doc.metadata.get("page", "?"))
                        st.markdown(f"**Chunk {i} (page {page})**")
                        st.text(doc.page_content)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

if not st.session_state.messages and st.session_state.rag_chain is None:
    st.info(
        "👈 Upload a PDF in the sidebar, add your Gemini API key, and click "
        "**Build / Rebuild index** to get started."
    )
