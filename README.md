# AI Code Review Assistant

A Streamlit app that reviews uploaded Python files using a RAG (Retrieval-Augmented
Generation) pipeline grounded in a curated taxonomy of labeled bug, security,
performance, and style examples — rather than relying on a single freeform prompt.

## How it works

1. **Upload** — a `.py` file is uploaded and read as text.
2. **Chunking** — the code is split into overlapping chunks using syntax-aware
   separators (`\ndef `, `\nclass `, ...) so functions/classes are less likely
   to be sliced mid-definition.
3. **Two vector stores**:
   - `code_vectorstore` — embeddings of the uploaded file's chunks (rebuilt per file).
   - `criteria_vectorstore` — embeddings of a fixed taxonomy of ~30 labeled
     issue examples (`review_criteria.py`), cached once per session.
4. **Retrieval** — the most relevant/problematic code chunks are retrieved, then
   each is matched against the criteria taxonomy to find known issue patterns
   it resembles (deduplicated so the same pattern isn't repeated).
5. **Review generation** — a system/human prompt sends the matched criteria,
   retrieved context, and code (tagged with `<criteria>`, `<context>`, `<code>`)
   to the LLM, which returns a structured markdown report referencing which
   known pattern (if any) each issue matches.

## Project structure

```
.
├── app.py                          # Streamlit app entry point
├── review_criteria.py              # Labeled taxonomy: bugs, security, performance, style
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── secrets.toml.example        # Template — copy to secrets.toml and fill in real values
└── .gitignore                      # Excludes real secrets.toml, caches, venvs
```

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy the secrets template and fill in real values:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   Then edit `.streamlit/secrets.toml` with your actual `GENAI_API_KEY`.
   **This file is gitignored — never commit it.**

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Extending the review criteria

Add new labeled examples to `review_criteria.py` by appending a `CriteriaExample(...)`
entry to the relevant list (`BUGS`, `SECURITY`, `PERFORMANCE`, or `STYLE`). No changes
to `app.py` are needed — new entries are picked up automatically the next time the
criteria vectorstore is built.

## Notes on the SSL/TLS handling in app.py

`app.py` disables SSL verification globally to work around a corporate
HTTPS-intercepting proxy. This is a known tradeoff (see `review_criteria.py` →
Security → "Disabled SSL Verification") and should be replaced with a scoped
corporate CA bundle (`verify="/path/to/ca-bundle.pem"`) when one is available,
rather than left as `verify=False` in a production deployment.
