import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import faiss

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="RAG Chat", page_icon="📄", layout="wide")
st.title("📄 RAG Chat — PDF Question Answering")
st.caption("Upload one or more PDFs, then ask questions across them. Powered by Gemini + FAISS.")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # dimensionality of the MiniLM-L6-v2 embeddings

PROMPT_TEMPLATE = """Answer the question based only on the following context.
If the answer is not contained in the context, say you don't know.

Context:
{context}

Question: {question}
"""

# ----------------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------------
defaults = {
    "vector_store": None,
    "processed_file_names": set(),  # names of PDFs already indexed
    "chat_history": [],  # list of (question, answer, sources) tuples
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def load_and_chunk_pdf(pdf_path: str, display_name: str):
    """Load a single PDF and split it into chunks, tagged with its filename."""
    loader = PyPDFLoader(pdf_path)
    pdf_data = loader.load()

    # Tag each page with the original filename (tempfile path isn't useful to show)
    for doc in pdf_data:
        doc.metadata["source"] = display_name

    chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return chunker.split_documents(pdf_data)


def new_empty_vector_store():
    """Create an empty FAISS vector store ready to receive documents."""
    embeddings = get_embeddings()
    index = faiss.IndexFlatL2(EMBED_DIM)
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )


def add_pdf_to_store(vector_store, pdf_path: str, display_name: str):
    """Chunk a PDF and add it to an existing (or freshly created) vector store."""
    chunks = load_and_chunk_pdf(pdf_path, display_name)
    if vector_store is None:
        vector_store = new_empty_vector_store()
    vector_store.add_documents(chunks)
    return vector_store, len(chunks)


def get_rag_chain(vector_store, api_key: str, model_name: str, k: int):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
    )
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever


# ----------------------------------------------------------------------------
# Sidebar — configuration
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "Google API Key",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        help="Get a key from https://aistudio.google.com/apikey. "
        "Not stored anywhere except this session.",
    )

    model_name = st.selectbox(
        "Gemini model",
        options=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash","gemini-3.1-flash-lite"],
        index=0,
    )

    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=5)

    st.divider()
    st.header("📎 Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        # Only process files we haven't indexed yet (by name).
        new_files = [
            f for f in uploaded_files if f.name not in st.session_state.processed_file_names
        ]
        if new_files:
            with st.spinner(f"Indexing {len(new_files)} new PDF(s)..."):
                for f in new_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getvalue())
                        tmp_path = tmp.name
                    try:
                        st.session_state.vector_store, n_chunks = add_pdf_to_store(
                            st.session_state.vector_store, tmp_path, f.name
                        )
                        st.session_state.processed_file_names.add(f.name)
                        st.success(f"Indexed {n_chunks} chunks from {f.name}")
                    finally:
                        os.unlink(tmp_path)

    if st.session_state.processed_file_names:
        st.info(
            "Active documents:\n"
            + "\n".join(f"- {name}" for name in sorted(st.session_state.processed_file_names))
        )

    if st.session_state.processed_file_names and st.button("Remove all documents"):
        st.session_state.vector_store = None
        st.session_state.processed_file_names = set()
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

# ----------------------------------------------------------------------------
# Main — chat interface
# ----------------------------------------------------------------------------
if st.session_state.vector_store is None:
    st.info("👈 Upload one or more PDFs in the sidebar to get started.")
else:
    for question, answer, sources in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
            if sources:
                with st.expander("Sources used"):
                    for i, src in enumerate(sources, 1):
                        doc_name = src.metadata.get("source", "?")
                        page = src.metadata.get("page", "?")
                        st.markdown(f"**Chunk {i} — {doc_name} (page {page})**")
                        st.text(src.page_content[:500])

    question = st.chat_input("Ask a question across your documents...")

    if question:
        if not api_key:
            st.error("Please enter your Google API key in the sidebar first.")
        else:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        rag_chain, retriever = get_rag_chain(
                            st.session_state.vector_store, api_key, model_name, top_k
                        )
                        answer = rag_chain.invoke(question)
                        sources = retriever.invoke(question)
                        st.write(answer)
                        if sources:
                            with st.expander("Sources used"):
                                for i, src in enumerate(sources, 1):
                                    doc_name = src.metadata.get("source", "?")
                                    page = src.metadata.get("page", "?")
                                    st.markdown(f"**Chunk {i} — {doc_name} (page {page})**")
                                    st.text(src.page_content[:500])
                        st.session_state.chat_history.append((question, answer, sources))
                    except Exception as e:
                        st.error(f"Something went wrong: {e}")
