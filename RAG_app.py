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
st.caption("Upload a PDF, then ask questions about it. Powered by Gemini + FAISS.")

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
    "processed_file_name": None,
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


def build_vector_store(pdf_path: str):
    """Load a PDF, chunk it, embed the chunks, and build a FAISS index."""
    loader = PyPDFLoader(pdf_path)
    pdf_data = loader.load()

    chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_data = chunker.split_documents(pdf_data)

    embeddings = get_embeddings()
    index = faiss.IndexFlatL2(EMBED_DIM)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(chunked_data)
    return vector_store, len(chunked_data)


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
    st.header("📎 Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        # Only re-process if a new file was uploaded
        if st.session_state.processed_file_name != uploaded_file.name:
            with st.spinner("Reading, chunking, and embedding PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                try:
                    vector_store, n_chunks = build_vector_store(tmp_path)
                    st.session_state.vector_store = vector_store
                    st.session_state.processed_file_name = uploaded_file.name
                    st.session_state.chat_history = []
                    st.success(f"Indexed {n_chunks} chunks from {uploaded_file.name}")
                finally:
                    os.unlink(tmp_path)

    if st.session_state.processed_file_name:
        st.info(f"Active document: **{st.session_state.processed_file_name}**")

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

# ----------------------------------------------------------------------------
# Main — chat interface
# ----------------------------------------------------------------------------
if st.session_state.vector_store is None:
    st.info("👈 Upload a PDF in the sidebar to get started.")
else:
    for question, answer, sources in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
            if sources:
                with st.expander("Sources used"):
                    for i, src in enumerate(sources, 1):
                        page = src.metadata.get("page", "?")
                        st.markdown(f"**Chunk {i} (page {page})**")
                        st.text(src.page_content[:500])

    question = st.chat_input("Ask a question about the document...")

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
                                    page = src.metadata.get("page", "?")
                                    st.markdown(f"**Chunk {i} (page {page})**")
                                    st.text(src.page_content[:500])
                        st.session_state.chat_history.append((question, answer, sources))
                    except Exception as e:
                        st.error(f"Something went wrong: {e}")
