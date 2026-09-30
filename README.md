# AI Homework Lab

This university lab builds a retrieval-augmented generation (RAG) system and,
later, an AI agent from first principles. Important mechanisms stay visible in
plain Python instead of being hidden by frameworks such as LangChain or
LlamaIndex.

## Current scope: complete local web application

The project currently implements:

- loading local plain-text knowledge documents;
- splitting documents into overlapping word chunks;
- preserving source and word-position metadata; and
- converting chunk text into 384-dimensional embeddings with Ollama;
- L2-normalizing embeddings and storing them with chunk records in local Chroma;
- embedding and normalizing a user question; and
- querying a cosine-distance Chroma collection for Top-K chunks;
- constructing labelled context from the raw Top-3 chunks; and
- generating a grounded, source-cited answer with local Qwen by default or
  optional Gemini;
- applying a calibrated pre-generation relevance threshold; and
- evaluating answerable and unanswerable questions.

Reranking and hybrid search are intentionally not part of this project. The
first educational implementation exposed cosine similarity with NumPy; the
final exam-facing implementation uses Chroma as its local vector database.
Phase 2B adds a visible, bounded native tool-calling loop around the
four Phase 2A tools. It has no shell, file-writing, or external-network tool.
Phase 2C adds evaluation and final lecturer-facing documentation without
changing the Agent architecture.
Slice 1 adds a minimal local FastAPI wrapper that reuses the existing Agent.
Slice 2 adds a temporary in-browser React chat history that calls only that API.
Slice 3A accepts local TXT, Markdown, text-based PDF, and DOCX uploads, extracts their
text, and stores safe metadata.
Slice 3B can turn one uploaded document into a separate local chunk and vector
index. Slice 3C can query exactly one indexed upload through its own API and
CLI path. Slice 3D exposes upload, indexing, and questions in the React UI. The
final interface supports one or more active uploaded documents: without active
uploads, messages use `/chat`; with uploads, messages use the isolated
multi-document RAG endpoint.

## Project layout

```text
knowledge/          Small source documents used by the study assistant
storage/chroma/     Ignored local Chroma database for the static knowledge base
src/                Explicit RAG building blocks
tests/              Tests added as the project grows
config.py           Shared paths and chunk settings
inspect_chunks.py   Phase 1A inspection entry point
build_index.py      Rebuilds the static local Chroma index
inspect_embeddings.py  Inspects embeddings stored in the Chroma collection
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
src/documents/          Upload storage and TXT/MD/PDF/DOCX text extraction
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

Open `http://localhost:5173`. The page has one conversation and one composer.
Without active uploaded documents, Send calls `/chat` and uses the Agent,
approved tools, and the static sample study knowledge. When one or more TXT,
Markdown, PDF, or DOCX files are ready, Send calls `/documents/ask` with their
document IDs and searches only those isolated uploaded indexes. Detaching one
file removes its ID from later requests without deleting its server-side index;
detaching all files routes later messages back to `/chat`.

The browser calls only FastAPI and never calls Ollama or Gemini directly. Chat
messages, active uploaded documents, and document answers exist only in React
state and are cleared when the page is refreshed.

## Final web architecture

Route selection and general chat:

```text
React Send
├── no active uploads → POST /chat → Agent → tools / static sample RAG / Qwen
└── active uploads    → POST /documents/ask → uploaded-document RAG
```

Uploaded-document preparation:

```text
React multi-file selection
→ for each file: POST /documents/upload
→ safe storage and text extraction
→ for each document ID: POST /documents/{document_id}/index
→ chunking + embeddings + one isolated Chroma index per document
```

Uploaded-document questions:

```text
React question
→ POST /documents/ask with the active document IDs
→ one query embedding
→ up to Top-K candidates from each selected document index
→ merge by cosine similarity and keep one global Top-K
→ labelled context
→ configured RAG generation provider (Ollama/Qwen or Gemini)
→ grounded answer plus retrieval metadata
```

## Generation providers

The RAG answer generator supports two providers:

1. `ollama` is the default. It runs `qwen3:4b-instruct` locally.
2. `gemini` is optional cloud inference through the official `google-genai`
   SDK and requires Gemini API authentication.

These settings are read when Python starts:

```text
LLM_PROVIDER=ollama
OLLAMA_GENERATION_MODEL=qwen3:4b-instruct
GEMINI_MODEL=gemini-3.5-flash
GEMINI_API_KEY=<environment variable only>
```

Both providers receive the same grounded system and user prompts from
`src/prompt_builder.py`. No Gemini key is stored in source code or sent to
React. A missing key while `LLM_PROVIDER=gemini`, or an unknown provider name,
produces a controlled configuration error and never silently falls back.

The embedding model, vector database, and generation model have separate jobs:

- `all-minilm:latest` converts chunks and questions into embedding vectors.
- local Chroma stores those vectors and performs Top-K retrieval.
- Qwen or Gemini reads the retrieved text and generates the cited answer.

Switching between Qwen and Gemini does **not** require rebuilding Chroma because
the stored `all-minilm` embeddings are unchanged. Changing the embedding model
would require re-embedding and re-indexing every document.

The Agent is intentionally separate: `/chat` continues to use its native
Ollama/Qwen tool-calling loop. `LLM_PROVIDER` selects only standalone and
uploaded-document RAG answer generation; it does not make Gemini control the
Agent.

### Test Gemini from PowerShell

Set current-session variables before starting or restarting the backend:

```powershell
$env:GEMINI_API_KEY="paste-your-key-here"
$env:LLM_PROVIDER="gemini"
$env:GEMINI_MODEL="gemini-3.5-flash"
python run_api.py
```

Run the React frontend in its usual second terminal, open
`http://localhost:5173`, and upload a small TXT document containing a distinctive
fact, for example: `The Aurora Lab review meeting is every Thursday at 3 PM.`
After indexing completes, ask in the browser:

```text
When is the Aurora Lab review meeting?
```

The request must be in uploaded Knowledge Base mode so it uses
`/documents/ask`. General `/chat` remains the Ollama Agent.

To return to local generation, stop and restart the backend after running:

```powershell
$env:LLM_PROVIDER="ollama"
$env:OLLAMA_GENERATION_MODEL="qwen3:4b-instruct"
Remove-Item Env:GEMINI_API_KEY -ErrorAction SilentlyContinue
python run_api.py
```

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

Uploads are limited to 10 MB. TXT and Markdown files must be UTF-8. PDF extraction supports
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

Ask across a selected set of indexed uploads:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/documents/ask" `
  -ContentType "application/json" `
  -Body (@{
    document_ids = @("<first_document_id>", "<second_document_id>")
    question = "What do these documents say about security?"
  } | ConvertTo-Json)
```

The backend embeds the question once, queries each selected document's separate
Chroma collection, merges those candidates by cosine similarity, and sends only
the final global `TOP_K` chunks to the grounded generator. The static sample
collection in `storage/chroma/` is never included in uploaded-document retrieval.

The endpoint loads only the Chroma index belonging to the supplied document ID.
It embeds the question with `all-minilm:latest`, asks Chroma for that document's
cosine Top-3, and sends labelled context to the configured RAG generator only
when the highest score reaches `MIN_SIMILARITY`. The `0.30` cutoff is an
educational heuristic calibrated on the static homework corpus, not a universal
confidence score; different uploaded-document collections may require new
calibration.

Uploaded-document answerability has two visible checks:

1. `retrieval_gate_passed` reports whether the highest cosine similarity meets
   the heuristic threshold. If it fails, `generation_called` is false.
2. If the gate passes, the configured generator receives only the retrieved
   context. It can still
   return the exact insufficient-information response when the requested fact
   is absent; in that case `generation_called` is true and `answerable` is false.

The legacy `sufficient_context` response field remains as a compatibility alias
for `retrieval_gate_passed`; it does not claim that the configured generation
provider produced an answer.

Each uploaded index is stored separately under
`uploads/<document_id>/chroma/`. Repeating the indexing request deletes and
recreates that document's one deterministic Chroma collection, so chunks are
never appended or duplicated. The static collection under `storage/chroma/`
is not modified. Uploaded retrieval also applies a `document_id` metadata
filter as a second isolation check.

The `inspect_chunks.py` script prints each static knowledge document's path,
title, and word count. It then prints every chunk, its ID, its 0-based
end-exclusive word range, its overlap with the previous chunk, and its full text.

## Chunk settings

- Chunk size: 180 words
- Overlap: 30 words
- Step between chunk starts: 150 words

For example, the first two full chunks use word ranges `[0, 180)` and
`[150, 330)`. The words at positions 150 through 179 occur in both chunks.

## Dependencies

Document loading and chunking use only the Python standard library. The
official Ollama Python client calls local `all-minilm:latest`; NumPy performs
L2 normalization before the vectors are supplied explicitly to local Chroma.
Chroma stores and queries those vectors and does not select an embedding model.
The official `google-genai` SDK is used only when Gemini is selected for RAG
answer generation.
Slice 3A uses `pypdf` for
text-based PDFs, `python-docx` for DOCX files, and `python-multipart` for
FastAPI file uploads. No OCR library is included.

## Where the RAG mechanisms live

- `config.py` selects `all-minilm:latest`, `TOP_K`, the 180/30 word chunk
  settings, and both local Chroma collection names.
- `src/document_loader.py` loads the static text files;
  `src/documents/extractor.py` extracts uploaded TXT, Markdown, PDF, and DOCX text.
- `src/chunker.py` creates overlapping chunks and deterministic chunk IDs.
- `src/embedder.py` contains the exact `ollama.embed(...)` call and L2
  normalization.
- `src/vector_store.py` writes explicit vectors and metadata with Chroma
  `upsert(...)`, supplies explicit query vectors to Chroma `query(...)`, and
  converts cosine distance to `similarity = 1 - distance`.
- `build_index.py` rebuilds the static `static_knowledge` collection;
  `src/documents/indexer.py` rebuilds one isolated uploaded-document collection.
- `src/prompt_builder.py` places retrieved chunks into labelled context, and
  `src/generator.py` selects the RAG provider and contains the exact Ollama and
  Gemini inference calls.
- `inspect_retrieval.py`, `ask.py`, and `ask_document.py` make Top-K records,
  scores, the relevance decision, context, answers, and chunk-ID citations
  visible for verification.

`MIN_SIMILARITY = 0.30` still means cosine similarity. Chroma is configured
with cosine space and returns cosine distance, so the application converts it
before applying the same educational heuristic. A failed gate skips generation;
after a passed gate, the grounded model can still refuse an unsupported answer.

## Known Limitations

- Uploaded-document questions remain separate from the Agent and static `/chat` knowledge path.
- The browser loses its active uploaded-document selection when refreshed.
- Each uploaded document has its own index; cross-document retrieval merges candidates at request time rather than creating a combined persistent index.
- The knowledge base contains only five small documents.
- Chunking uses word counts rather than semantic or token-aware boundaries.
- `all-minilm` can miss nuance or rank generally related text above the best passage.
- Retrieval uses raw cosine similarity without a learned relevance model.
- There is no reranker to reconsider the initial vector ranking.
- The similarity threshold is a heuristic calibrated on only seven questions and cannot perfectly separate every answerable and unanswerable query.
- Chroma is local and single-process for this homework; there is no hosted or distributed vector service.
- There is no hybrid keyword and vector search.
- The default local 4B generation model has limited reasoning capacity; Gemini
  requires an external cloud request and valid API credentials.
- Generated claims are not automatically checked against source text for factual accuracy.
