import os
import ssl
import warnings

import httpx
import streamlit as st
import urllib3
import requests

# Correct import path for InsecureRequestWarning
from urllib3.exceptions import InsecureRequestWarning

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from review_criteria import REVIEW_CRITERIA, build_criteria_documents

# Disable SSL warnings
urllib3.disable_warnings(InsecureRequestWarning)

# Patch requests.Session to ignore SSL globally
# NOTE: this is scoped to work around a corporate HTTPS-intercepting proxy.
# See review_criteria.py -> Security -> "Disabled SSL Verification" for the
# tradeoff this introduces; prefer trusting a corporate CA bundle instead
# of verify=False once one is available.
_old_request = requests.Session.request

def _request_no_ssl(self, *args, **kwargs):
    kwargs['verify'] = False  # disable SSL verification
    return _old_request(self, *args, **kwargs)

requests.Session.request = _request_no_ssl

# Configure client behavior for the proxy environment
client = httpx.Client(verify=False)

# Fetch config safely from secrets (with fallbacks if missing)
API_KEY = st.secrets.get("GENAI_API_KEY")
BASE_URL = st.secrets.get("BASE_URL", "https://genailab.tcs.in")
LLM_MODEL = st.secrets.get("LLM_MODEL", "azure_ai/genailab-maas-DeepSeek-V3-0324")
EMBEDDING_MODEL = st.secrets.get("EMBEDDING_MODEL", "azure/genailab-maas-text-embedding-3-large")

# Safety Check: Warn the user if they forgot to set up their secrets file
if not API_KEY:
    st.error("Missing API Key! Please configure `GENAI_API_KEY` in your `.streamlit/secrets.toml` file.")
    st.stop()

# Initialize LLM using the secrets
llm = ChatOpenAI(
    base_url=BASE_URL,
    model=LLM_MODEL,
    api_key=API_KEY,
    http_client=client,
)

# Initialize Embeddings using the secrets
embeddings = OpenAIEmbeddings(
    base_url=BASE_URL,
    model=EMBEDDING_MODEL,
    api_key=API_KEY,
    http_client=client,
)


@st.cache_resource(show_spinner=False)
def get_criteria_vectorstore(_embeddings):
    """
    Build (once per session) a vectorstore of the labeled review-criteria
    taxonomy from review_criteria.py. This is independent of any uploaded
    file, so it is cached rather than rebuilt on every review.
    """
    criteria_docs = build_criteria_documents(REVIEW_CRITERIA)
    return Chroma.from_documents(
        documents=criteria_docs,
        embedding=_embeddings,
        collection_name="review_criteria",
    )


def get_relevant_criteria(criteria_vectorstore, code_chunks, k_per_chunk=2, max_total=8):
    """
    For each code chunk, retrieve the most similar known issue patterns from
    the criteria taxonomy, then dedupe by subcategory so the same pattern
    isn't repeated in the prompt just because it matched several chunks.
    Returns a list of metadata dicts, capped at max_total.
    """
    seen_subcategories = set()
    matched = []

    for chunk in code_chunks:
        results = criteria_vectorstore.similarity_search(chunk.page_content, k=k_per_chunk)
        for doc in results:
            subcat = doc.metadata.get("subcategory")
            if subcat in seen_subcategories:
                continue
            seen_subcategories.add(subcat)
            matched.append(doc.metadata | {"matched_snippet": chunk.page_content[:200]})
            if len(matched) >= max_total:
                return matched

    return matched


def format_criteria_for_prompt(matched_criteria):
    """
    Render matched criteria metadata into a compact block for the prompt,
    grouped implicitly by the order they were matched (most relevant first).
    """
    if not matched_criteria:
        return "No specific known issue patterns matched this code."

    lines = []
    for m in matched_criteria:
        lines.append(
            f"- [{m['category']} / {m['severity']}] {m['subcategory']}: {m['tags']}\n"
            f"  Suggested fix pattern:\n{m['good_pattern']}"
        )
    return "\n".join(lines)


# Streamlit UI
st.set_page_config(page_title="AI assistant Code Reviewer")
st.title("📄 AI assistant Code Reviewer")

criteria_vectorstore = get_criteria_vectorstore(embeddings)

uploaded_file = st.file_uploader("Upload Python file", type=["py"])

if uploaded_file:
    # Read the uploaded .py file as text
    try:
        raw_bytes = uploaded_file.read()
        try:
            code_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            code_text = raw_bytes.decode("latin-1")
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    if not code_text.strip():
        st.warning("Uploaded file appears to be empty.")
        st.stop()

    with st.expander("Show code preview"):
        st.code(code_text, language="python")

    # Splitter respects Python syntax boundaries so functions/classes are
    # less likely to be sliced mid-definition.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\ndef ", "\nclass ", "\n\n", "\n", " ", ""],
    )

    docs = [
        Document(page_content=chunk, metadata={"source": uploaded_file.name})
        for chunk in splitter.split_text(code_text)
    ]

    code_vectorstore = None
    try:
        with st.spinner("Creating in-memory vector index..."):
            code_vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
            )
    except Exception as e:
        st.error(
            "Failed to build the vector index (embedding call failed). "
            f"Details: {e}"
        )
        st.stop()

    code_retriever = code_vectorstore.as_retriever(search_kwargs={"k": 8})

    # System message carries the reviewer persona and hard rules.
    # Human message carries the actual data: known criteria + the code itself,
    # clearly delimited with tags so uploaded code can't be mistaken for
    # instructions.
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a senior Python code reviewer with expertise in security, "
            "performance, and Pythonic style. You review only what is shown to "
            "you inside <code> tags — never assume behavior not visible in the "
            "code. Every issue you report must reference a specific line or "
            "snippet from the code under review. If you are unsure whether "
            "something is a real issue, mark it as Low severity rather than "
            "omitting it or overstating it. Use the patterns inside <criteria> "
            "as reference precedent: if the code matches one, name which "
            "known pattern it matches; if it doesn't match any known pattern "
            "but is still a real issue, flag it as a new issue type."
        ),
        (
            "human",
            "Known issue patterns to check against:\n"
            "<criteria>\n{criteria}\n</criteria>\n\n"
            "Code fragments retrieved as most relevant/problematic:\n"
            "<context>\n{context}\n</context>\n\n"
            "Full code under review (may be partial if file size limit was exceeded):\n"
            "<code>\n{code}\n</code>\n\n"
            "Return a concise markdown report with these sections:\n"
            "1. Summary\n"
            "2. Key Issues — as a table: Severity | Line/Snippet | Issue | Matched Known Pattern (or 'New')\n"
            "3. Recommendations (actionable changes, with short code snippets)\n"
            "4. Nice-to-Haves\n"
            "5. Annotated Notes (quote small relevant lines only)"
        ),
    ])

    if st.button("Generate Review"):
        try:
            with st.spinner("Analyzing code..."):
                # Retrieve the most representative/problematic code chunks
                retrieved_docs = code_retriever.invoke(
                    "syntax errors bugs logic flaws security vulnerabilities credentials exposed"
                )
                context = "\n\n--- Chunk ---\n".join(doc.page_content for doc in retrieved_docs)

                # Retrieve matching known issue patterns per retrieved chunk
                matched_criteria = get_relevant_criteria(criteria_vectorstore, retrieved_docs)
                criteria_text = format_criteria_for_prompt(matched_criteria)

                # Small/medium files: send full code for a thorough review.
                # Large files: fall back to the retrieved context only.
                if len(code_text) <= 12000:
                    code_for_review = code_text
                else:
                    code_for_review = "File too large for full payload. See context snippets above."

                messages = prompt.invoke({
                    "code": code_for_review,
                    "context": context,
                    "criteria": criteria_text,
                })
                response = llm.invoke(messages)

            st.subheader("Review")
            st.write(response.content)

            with st.expander("Matched known issue patterns used in this review"):
                if matched_criteria:
                    for m in matched_criteria:
                        st.markdown(f"**[{m['severity']}] {m['category']} — {m['subcategory']}**")
                else:
                    st.write("No known patterns matched; review relied on general judgment only.")

        except Exception as e:
            st.error(f"Review generation failed: {e}")
