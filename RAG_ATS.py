import os
import json
import hashlib
import tempfile

import streamlit as st
import pandas as pd
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
st.set_page_config(page_title="ATS Resume Screener", page_icon="🧑‍💼", layout="wide")
st.title("🧑‍💼 ATS Resume Screener (RAG-powered)")
st.caption(
    "Upload a job description and a pool of resumes. Get every candidate scored "
    "against the JD, and chat over the resume pool. Powered by Gemini + FAISS."
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

SCORING_PROMPT = """You are an experienced technical recruiter screening a candidate \
for a role. Read the job description and the candidate's resume, then evaluate the fit.

Job Description:
{jd}

Candidate Resume ({candidate_name}):
{resume}

Respond with ONLY a valid JSON object (no markdown fences, no commentary) with exactly \
these keys:
{{
  "score": <integer 0-100, overall fit for this role>,
  "matched_skills": [<short list of skills/requirements the candidate clearly meets>],
  "missing_skills": [<short list of skills/requirements the JD asks for but resume lacks>],
  "strengths": "<1-2 sentence summary of the candidate's strongest points for this role>",
  "concerns": "<1-2 sentence summary of gaps or concerns>"
}}
"""

CHAT_PROMPT_TEMPLATE = """You are helping a recruiter answer questions about a pool of \
candidate resumes. Answer based only on the following resume excerpts. Always mention \
which candidate(s) your answer is about. If the answer isn't in the context, say so.

Context:
{context}

Question: {question}
"""

# ----------------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------------
defaults = {
    "vector_store": None,
    "candidates": {},  # name -> full resume text
    "processed_file_names": set(),
    "scores": {},  # name -> dict(score, matched_skills, missing_skills, strengths, concerns)
    "scored_jd_hash": None,
    "chat_history": [],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def format_docs(docs):
    return "\n\n".join(f"[{d.metadata.get('candidate', '?')}] {d.page_content}" for d in docs)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def new_empty_vector_store():
    embeddings = get_embeddings()
    index = faiss.IndexFlatL2(EMBED_DIM)
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )


def extract_full_text(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return "\n".join(p.page_content for p in pages)


def chunk_and_tag(pdf_path: str, candidate_name: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    for p in pages:
        p.metadata["candidate"] = candidate_name
        p.metadata["source"] = candidate_name
    chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return chunker.split_documents(pages)


def get_llm(api_key: str, model_name: str, temperature: float = 0.0):
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature)


def score_candidate(resume_text: str, jd_text: str, candidate_name: str, api_key: str, model_name: str):
    llm = get_llm(api_key, model_name, temperature=0.0)
    prompt = SCORING_PROMPT.format(jd=jd_text, candidate_name=candidate_name, resume=resume_text)
    raw = llm.invoke(prompt).content.strip()
    # Strip accidental markdown code fences
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.lower().startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "score": None,
            "matched_skills": [],
            "missing_skills": [],
            "strengths": "Could not parse model output.",
            "concerns": raw[:300],
        }
    return data


def get_chat_chain(vector_store, api_key: str, model_name: str, k: int):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    prompt = PromptTemplate.from_template(CHAT_PROMPT_TEMPLATE)
    llm = get_llm(api_key, model_name, temperature=0.2)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def jd_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# Sidebar — configuration
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "Google API Key",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        help="Get a key from https://aistudio.google.com/apikey.",
    )

    model_name = st.selectbox(
        "Gemini model",
        options=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash","gemini-3.1-flash-lite"],
        index=0,
    )

    top_k = st.slider("Chunks to retrieve for chat (k)", min_value=1, max_value=15, value=6)

    st.divider()
    st.header("📋 Job Description")

    jd_pdf = st.file_uploader("Upload JD as PDF (optional)", type=["pdf"], key="jd_pdf")
    if jd_pdf is not None and st.session_state.get("_jd_pdf_name") != jd_pdf.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(jd_pdf.getvalue())
            tmp_path = tmp.name
        try:
            st.session_state["jd_text"] = extract_full_text(tmp_path)
            st.session_state["_jd_pdf_name"] = jd_pdf.name
        finally:
            os.unlink(tmp_path)

    jd_text = st.text_area(
        "Job description text",
        value=st.session_state.get("jd_text", ""),
        height=200,
        placeholder="Paste the job description here, or upload a PDF above...",
        key="jd_text",
    )

    st.divider()
    st.header("📎 Resumes")
    uploaded_resumes = st.file_uploader(
        "Upload candidate resumes (PDF)", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_resumes:
        new_files = [
            f for f in uploaded_resumes if f.name not in st.session_state.processed_file_names
        ]
        if new_files:
            with st.spinner(f"Processing {len(new_files)} new resume(s)..."):
                for f in new_files:
                    candidate_name = os.path.splitext(f.name)[0]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getvalue())
                        tmp_path = tmp.name
                    try:
                        st.session_state.candidates[candidate_name] = extract_full_text(tmp_path)
                        chunks = chunk_and_tag(tmp_path, candidate_name)
                        if st.session_state.vector_store is None:
                            st.session_state.vector_store = new_empty_vector_store()
                        st.session_state.vector_store.add_documents(chunks)
                        st.session_state.processed_file_names.add(f.name)
                    finally:
                        os.unlink(tmp_path)
            st.success(f"Processed {len(new_files)} resume(s).")

    if st.session_state.candidates:
        st.info(
            "Candidates loaded:\n"
            + "\n".join(f"- {name}" for name in sorted(st.session_state.candidates))
        )

    if st.session_state.candidates and st.button("Remove all candidates"):
        st.session_state.vector_store = None
        st.session_state.candidates = {}
        st.session_state.processed_file_names = set()
        st.session_state.scores = {}
        st.session_state.scored_jd_hash = None
        st.session_state.chat_history = []
        st.rerun()

# ----------------------------------------------------------------------------
# Main — tabs
# ----------------------------------------------------------------------------
tab_rank, tab_chat = st.tabs(["📊 Candidate Ranking", "💬 Chat with Resume Pool"])

# --- Ranking tab ---
with tab_rank:
    if not st.session_state.candidates:
        st.info("👈 Upload resumes in the sidebar to get started.")
    elif not jd_text.strip():
        st.info("👈 Add a job description in the sidebar to score candidates.")
    else:
        current_hash = jd_hash(jd_text)
        jd_changed = current_hash != st.session_state.scored_jd_hash
        unscored = [n for n in st.session_state.candidates if n not in st.session_state.scores]

        col1, col2 = st.columns([1, 3])
        with col1:
            score_clicked = st.button("🔍 Score / Re-score candidates", type="primary")
        with col2:
            if jd_changed and st.session_state.scores:
                st.warning("Job description changed since last scoring — re-score to update.")
            elif unscored:
                st.caption(f"{len(unscored)} candidate(s) not yet scored.")

        if score_clicked:
            if not api_key:
                st.error("Please enter your Google API key in the sidebar first.")
            else:
                progress = st.progress(0.0, text="Scoring candidates...")
                names = list(st.session_state.candidates.keys())
                for i, name in enumerate(names):
                    try:
                        st.session_state.scores[name] = score_candidate(
                            st.session_state.candidates[name], jd_text, name, api_key, model_name
                        )
                    except Exception as e:
                        st.session_state.scores[name] = {
                            "score": None,
                            "matched_skills": [],
                            "missing_skills": [],
                            "strengths": "",
                            "concerns": f"Error: {e}",
                        }
                    progress.progress((i + 1) / len(names), text=f"Scored {name}")
                st.session_state.scored_jd_hash = current_hash
                progress.empty()
                st.rerun()

        if st.session_state.scores:
            rows = []
            for name, data in st.session_state.scores.items():
                rows.append(
                    {
                        "Candidate": name,
                        "Score": data.get("score"),
                        "Matched Skills": ", ".join(data.get("matched_skills", [])),
                        "Missing Skills": ", ".join(data.get("missing_skills", [])),
                    }
                )
            df = pd.DataFrame(rows).sort_values("Score", ascending=False, na_position="last")
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.subheader("Candidate details")
            for name in df["Candidate"]:
                data = st.session_state.scores[name]
                score_display = data.get("score")
                score_display = f"{score_display}/100" if score_display is not None else "N/A"
                with st.expander(f"{name} — {score_display}"):
                    st.markdown(f"**Strengths:** {data.get('strengths', '')}")
                    st.markdown(f"**Concerns:** {data.get('concerns', '')}")
                    st.markdown(f"**Matched skills:** {', '.join(data.get('matched_skills', [])) or '—'}")
                    st.markdown(f"**Missing skills:** {', '.join(data.get('missing_skills', [])) or '—'}")

# --- Chat tab ---
with tab_chat:
    if st.session_state.vector_store is None:
        st.info("👈 Upload resumes in the sidebar to chat over the candidate pool.")
    else:
        for question, answer, sources in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)
                if sources:
                    with st.expander("Sources used"):
                        for i, src in enumerate(sources, 1):
                            candidate = src.metadata.get("candidate", "?")
                            page = src.metadata.get("page", "?")
                            st.markdown(f"**Chunk {i} — {candidate} (page {page})**")
                            st.text(src.page_content[:500])

        question = st.chat_input("Ask about the candidate pool, e.g. 'Who has AWS experience?'")

        if question:
            if not api_key:
                st.error("Please enter your Google API key in the sidebar first.")
            else:
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            chain, retriever = get_chat_chain(
                                st.session_state.vector_store, api_key, model_name, top_k
                            )
                            answer = chain.invoke(question)
                            sources = retriever.invoke(question)
                            st.write(answer)
                            if sources:
                                with st.expander("Sources used"):
                                    for i, src in enumerate(sources, 1):
                                        candidate = src.metadata.get("candidate", "?")
                                        page = src.metadata.get("page", "?")
                                        st.markdown(f"**Chunk {i} — {candidate} (page {page})**")
                                        st.text(src.page_content[:500])
                            st.session_state.chat_history.append((question, answer, sources))
                        except Exception as e:
                            st.error(f"Something went wrong: {e}")
