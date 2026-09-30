# AI Homework Lab

This university lab builds a retrieval-augmented generation (RAG) system and,
later, an AI agent from first principles. Important mechanisms stay visible in
plain Python instead of being hidden by frameworks such as LangChain or
LlamaIndex.

## Current scope: Slice 3C uploaded-document question answering

The project currently implements:

- loading local plain-text knowledge documents;
- splitting documents into overlapping word chunks;
- preserving source and word-position metadata; and
- converting chunk text into 384-dimensional embeddings with Ollama;
- L2-normalizing and saving the embedding matrix; and
- saving chunk metadata in exactly the same row order;
- embedding and normalizing a user question; and
- ranking all chunks with an explicit NumPy dot product;
- constructing labelled context from the raw Top-3 chunks; and
- generating a grounded, source-cited answer with local Qwen;
- applying a calibrated pre-generation relevance threshold; and
- evaluating answerable and unanswerable questions.

Reranking, vector databases, and hybrid search are intentionally not part of
Phase 1. Phase 2B adds a visible, bounded native tool-calling loop around the
four Phase 2A tools. It has no shell, file-writing, or external-network tool.
Phase 2C adds evaluation and final lecturer-facing documentation without
changing the Agent architecture.
Slice 1 adds a minimal local FastAPI wrapper that reuses the existing Agent.
Slice 2 adds a temporary in-browser React chat history that calls only that API.
Slice 3A accepts local TXT, text-based PDF, and DOCX uploads, extracts their
text, and stores safe metadata.
Slice 3B can turn one uploaded document into a separate local chunk and vector
index. Slice 3C can query exactly one indexed upload through its own API and
CLI path. Uploaded-document questions are intentionally not connected to the
Agent, the static `/chat` endpoint, or the React UI.

## Project layout

```text
knowledge/          Small source documents used by the study assistant
storage/            Generated index data in later phases
src/                Explicit RAG building blocks
tests/              Tests added as the project grows
config.py           Shared paths and chunk settings
inspect_chunks.py   Phase 1A inspection entry point
build_index.py      Builds and saves the Phase 1B embedding index
inspect_embeddings.py  Inspects saved vectors without performing retrieval
inspect_retrieval.py   Prints every score and the complete Top-K chunks
ask.py                 Runs the visible thresholded RAG pipeline
evaluate_rag.py        Runs the seven-question Phase 1E evaluation
inspect_tools.py       Manually exercises every Phase 2A tool
run_agent.py           Runs the visible five-step Phase 2B Agent loop
evaluate_agent.py      Evaluates six Agent scenarios and four safety checks
FINAL_SUMMARY.md       Concise lecturer-friendly project summary
run_api.py             Starts the local FastAPI development server
src/api/               Request/response schemas and HTTP endpoints
frontend/              Minimal React and Vite chat interface
uploads/               Ignored local document storage, organized by document ID
src/documents/          Upload storage and TXT/PDF/DOCX text extraction
inspect_document.py    Inspects extraction from one local document
inspect_uploaded_index.py  Inspects one uploaded document's chunks and vectors
ask_document.py        Runs visible retrieval and grounded QA for one upload
```

## Run Phase 1A

From the project root:

```powershell
.\.venv\Scripts\python.exe inspect_chunks.py
```

If the virtual environment is already activated, `python inspect_chunks.py`
is equivalent.

Build and inspect the persistent embedding index with:

```powershell
.\.venv\Scripts\python.exe build_index.py
.\.venv\Scripts\python.exe inspect_embeddings.py
```

Inspect retrieval with the default question or a supplied question:

```powershell
.\.venv\Scripts\python.exe inspect_retrieval.py
.\.venv\Scripts\python.exe inspect_retrieval.py "What is an AI agent?"
```

Run the complete grounded RAG path with:

```powershell
.\.venv\Scripts\python.exe ask.py "What is an AI agent and how does it use tools?"
```

Run the complete evaluation with:

```powershell
.\.venv\Scripts\python.exe evaluate_rag.py
```

Inspect the standalone Phase 2A tools with:

```powershell
.\.venv\Scripts\python.exe inspect_tools.py
```

Run the controlled Agent with a user goal:

```powershell
.\.venv\Scripts\python.exe run_agent.py "Calculate (125 * 8) / 4."
```

Run the final Agent evaluation with:

```powershell
.\.venv\Scripts\python.exe evaluate_agent.py
```

## Run the local API

Start the backend from the project root:

```powershell
.\.venv\Scripts\python.exe run_api.py
```

In a second PowerShell window, check health and send chat requests:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/chat" `
  -ContentType "application/json" `
  -Body (@{ message = "Say hello in one short sentence." } | ConvertTo-Json)

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/chat" `
  -ContentType "application/json" `
  -Body (@{ message = "What is RAG? Use the study knowledge base." } | ConvertTo-Json)

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/chat" `
  -ContentType "application/json" `
  -Body (@{ message = "Calculate 10 * 25." } | ConvertTo-Json)
```

`POST /chat` accepts one non-empty `message` and returns the final `answer`,
the number of model `steps`, and the ordered `tools_used`. The API does not
store chat history between requests.

## Run the chat UI

Start the backend in the first terminal:

```powershell
python run_api.py
```

Start the frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The browser sends messages only to FastAPI at
`http://127.0.0.1:8000/chat`; it never calls Ollama directly. Message history
exists only in React state and is cleared when the page is refreshed.

## Upload and inspect documents

With the backend running, upload one supported document from PowerShell:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/documents/upload" `
  -Form @{ file = Get-Item ".\notes.pdf" }
```

Retrieve safe metadata using the returned document ID:

```powershell
Invoke-RestMethod -Method Get `
  -Uri "http://127.0.0.1:8000/documents/<document_id>"
```

Inspect extraction without uploading:

```powershell
python inspect_document.py ".\notes.docx"
```

Uploads are limited to 10 MB. TXT files must be UTF-8. PDF extraction supports
normal text-based PDFs only; scanned image PDFs require OCR, which is not part
of this slice. The API returns at most a 500-character preview and never returns
the complete extracted document text.

Index an uploaded document using the ID returned by the upload endpoint:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/documents/<document_id>/index"
```

Inspect the generated document-specific index:

```powershell
python inspect_uploaded_index.py "<document_id>"
```

Ask a question against exactly that uploaded document:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/documents/<document_id>/ask" `
  -ContentType "application/json" `
  -Body (@{ question = "What is the main topic?" } | ConvertTo-Json)

python ask_document.py "<document_id>" "What is the main topic?"
```

The endpoint loads only the index belonging to the supplied document ID. It
embeds the question with `all-minilm:latest`, ranks that document's normalized
vectors with a NumPy dot product, and sends labelled Top-3 context to Qwen only
when the highest score reaches `MIN_SIMILARITY`. The `0.30` cutoff is an
educational heuristic calibrated on the static homework corpus, not a universal
confidence score; different uploaded-document collections may require new
calibration.

Each index is stored separately under
`uploads/<document_id>/index/` as `chunks.json` and `embeddings.npy`. Repeating
the indexing request rebuilds a temporary complete index and replaces the one
canonical `index/` directory, so chunks are never appended or duplicated. The
static homework index under `storage/` is not modified.

The script prints each document's path, title, and word count. It then prints
every chunk, its ID, its 0-based end-exclusive word range, its overlap with the
previous chunk, and its full text.

## Chunk settings

- Chunk size: 180 words
- Overlap: 30 words
- Step between chunk starts: 150 words

For example, the first two full chunks use word ranges `[0, 180)` and
`[150, 330)`. The words at positions 150 through 179 occur in both chunks.

## Dependencies

Document loading and chunking use only the Python standard library. Phase 1B
uses NumPy for the vector matrix and the official Ollama Python client to call
the local `all-minilm:latest` embedding model. Slice 3A uses `pypdf` for
text-based PDFs, `python-docx` for DOCX files, and `python-multipart` for
FastAPI file uploads. No OCR library is included.

## Known Limitations

- Uploaded-document indexes can be queried one at a time, but are not connected to the Agent, static `/chat`, or React UI.
- Each uploaded document has its own index; combined document retrieval is not implemented.
- The knowledge base contains only five small documents.
- Chunking uses word counts rather than semantic or token-aware boundaries.
- `all-minilm` can miss nuance or rank generally related text above the best passage.
- Retrieval uses raw cosine similarity without a learned relevance model.
- There is no reranker to reconsider the initial vector ranking.
- The similarity threshold is a heuristic calibrated on only seven questions.
- There is no vector database; every stored vector is searched directly with NumPy.
- There is no hybrid keyword and vector search.
- Answer generation uses a small local 4B model with limited reasoning capacity.
- Generated claims are not automatically checked against source text for factual accuracy.
