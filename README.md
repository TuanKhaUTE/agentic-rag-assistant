# AI Homework Lab — Document-Grounded RAG Assistant

This midterm template is a closed-domain Retrieval-Augmented Generation (RAG) chat assistant. A user uploads one or more study documents, selects the active documents, and asks questions whose answers must be grounded in those documents.

There is no Agent loop, tool calling, static runtime knowledge corpus, general-purpose chat fallback, LangChain, or LlamaIndex. The important mechanisms are ordinary Python functions that can be explained line by line.

## What the application does

```text
Teacher/user uploads 1-N files
  → validate and store safely
  → extract TXT / MD / text-based PDF / DOCX text
  → split text into overlapping word chunks
  → embed every chunk with Ollama all-minilm
  → store explicit vectors in one local Chroma index per document

User asks a question with active document IDs
  → embed the question once with the same model
  → query each active document index with that vector
  → merge all per-document candidates
  → select one GLOBAL Top-K of 3 chunks
  → apply the heuristic relevance threshold
  → build labelled, instruction-resistant context
  → generate with Qwen or optional Gemini
  → return an answer with document-scoped chunk citations
```

With no active documents, the React UI stops before making a chat request and displays: `Please upload at least one document to create the Knowledge Base.` The API also rejects an empty `document_ids` list during request validation, so no generation model is called.

## Project structure

```text
ai-homework-lab/
├── backend/
│   ├── app.py                 FastAPI routes and controlled HTTP errors
│   ├── config.py              Small set of RAG/model/upload settings
│   ├── schemas.py             Pydantic request and response contracts
│   ├── documents/
│   │   ├── extractor.py       TXT, MD, PDF, and DOCX text extraction
│   │   ├── storage.py         Safe upload IDs, local files, and metadata
│   │   └── indexer.py         Extract → chunk → embed → Chroma orchestration
│   └── rag/
│       ├── chunker.py         Word chunks with overlap and trace metadata
│       ├── embedder.py        Ollama embeddings and L2 normalization
│       ├── vector_store.py    Explicit-vector Chroma storage/query code
│       ├── retriever.py       One query vector, merge, global Top-K
│       ├── prompt_builder.py  Labelled context and grounding rules
│       ├── generator.py       Isolated Qwen/Gemini generation boundary
│       └── service.py         Retrieval gate and final QA orchestration
├── frontend/                  React + Vite chat/upload UI
├── scripts/
│   ├── run_api.py             Local Uvicorn entry point
│   ├── inspect_document.py    Extraction inspection
│   ├── inspect_uploaded_index.py
│   └── evaluate_rag.py        Uploaded-fixture RAG evaluation
├── tests/
│   ├── fixtures/              Test-only documents; not runtime knowledge
│   └── rag_cases.json
├── uploads/                   Ignored local upload/index data
├── requirements.txt
├── README.md
└── MIDTERM_CODE_DEFENSE.md
```

## Requirements

- Windows and Python
- Node.js/npm
- Ollama running locally
- Ollama models:

```powershell
ollama pull all-minilm:latest
ollama pull qwen3:4b-instruct
```

Create the Python environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create the local configuration file from the safe example:

```powershell
Copy-Item .env.example .env
```

The backend loads this root `.env` automatically. Keep `.env` private; it is
ignored by Git.

## Run locally

Backend terminal:

```powershell
.\.venv\Scripts\python.exe scripts\run_api.py
```

Frontend terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

In the UI:

1. Attach one or more `.txt`, `.md`, `.pdf`, or `.docx` files.
2. Wait until extraction, chunking, embedding, and indexing finish.
3. Ask a question about the active documents.
4. Expand **View RAG details** to show similarity, threshold, candidate pool, query-vector dimension, global Top-K, filenames, and chunk IDs.
5. Detach a document to exclude it from later retrieval. Detaching changes the active selection; it does not delete the locally stored upload.

## HTTP API

### Health

```http
GET /health
```

### Upload

```http
POST /documents/upload
Content-Type: multipart/form-data
```

The response contains the server-generated `document_id`, safe metadata, and a short text preview. The complete extracted text and filesystem path are not returned.

### Index

```http
POST /documents/{document_id}/index
```

Indexing is idempotent: the document's Chroma collection is rebuilt rather than duplicated.

### Document-grounded chat

```http
POST /chat
Content-Type: application/json

{
  "question": "Who reviews the incident report?",
  "document_ids": ["document-id-one", "document-id-two"]
}
```

The response includes the answer, selected filenames, query embedding dimension, candidate count, highest similarity, threshold decision, whether generation was called, and the final retrieved chunks. `top_k` is the final total across all active documents, not a per-file prompt limit.

## RAG mechanisms

### Extraction and storage

Uploads receive a random 32-character server-side ID. The original filename is kept only as metadata; it never controls a filesystem location. Files are limited to 10 MB and supported extensions only. Text-based PDFs are supported; scanned image PDFs require OCR, which is intentionally not implemented.

Each upload is stored under:

```text
uploads/<document_id>/
├── original.<extension>
├── metadata.json
└── chroma/
```

### Chunking

The chunker uses 180 words with 30 words of overlap. The next start position advances by `180 - 30 = 150` words. Every chunk records its document-scoped ID, original source filename, chunk index, text, and start/end word positions.

Overlap helps preserve facts that cross a chunk boundary, at the cost of some repeated text and storage.

### Embeddings and Chroma

`all-minilm:latest` converts each chunk into a 384-dimensional semantic vector. Vectors are converted to `float32` and L2-normalized. Chroma receives those explicit vectors with `embedding_function=None`; it does not silently choose or run another embedding model.

Each uploaded document has a separate persistent Chroma directory. The embedding model name and dimension are stored and validated. Changing the embedding model requires re-indexing every document because old document vectors and new query vectors would otherwise occupy incompatible semantic spaces.

### Multi-document retrieval

The question is embedded exactly once. That normalized vector is reused when querying every selected document collection. Each collection contributes up to three candidates, the candidates are merged and sorted by cosine similarity, and only the best three total chunks enter the prompt.

For Chroma cosine space:

```text
cosine similarity = 1 - Chroma cosine distance
```

### Relevance gate and grounding

`MIN_SIMILARITY = 0.30` is checked before generation. When the highest score is lower, the service does not call Qwen or Gemini and returns exactly:

> The knowledge base does not contain enough information to answer this question.

The threshold is an educational heuristic, not a universal boundary. A higher threshold can cause false negatives; a lower threshold can allow irrelevant context.

When retrieval passes, every context block is labelled with both filename and a document-scoped chunk ID. The prompt says that documents are data, not instructions; outside knowledge, invented citations, and instructions inside documents must be ignored.

## Generation provider

Choose the provider in the root `.env` file. The default is local Qwen through
Ollama:

```text
LLM_PROVIDER=ollama
OLLAMA_GENERATION_MODEL=qwen3:4b-instruct
```

Optional Gemini generation uses:

```text
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-3.5-flash
GEMINI_API_KEY=your-api-key
```

Set `LLM_PROVIDER=ollama` or `LLM_PROVIDER=gemini`, then restart the backend.
Provider selection affects only answer generation. Extraction, chunking,
all-minilm embeddings, Chroma indexing, retrieval, global Top-K, thresholding,
context, and citations remain the same. API keys stay in the ignored backend
`.env` file or process environment and are never sent to React.

## Inspection and evaluation

Inspect extraction:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_document.py tests\fixtures\policy.txt
```

Inspect an already indexed upload:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_uploaded_index.py <document_id>
```

Run the RAG fixture evaluation (requires working Ollama embeddings and the configured generation provider):

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag.py
```

The evaluation creates document IDs from `tests/fixtures`, indexes them independently, asks direct, paraphrased, cross-document, and unanswerable questions, validates retrieved sources/citations, and prints the selected provider/model. Generated uploads remain under the ignored `uploads/` directory.

## Safety boundaries

- supported extensions only: TXT, MD, PDF, DOCX
- 10 MB maximum upload size and empty-file rejection
- random server-side document IDs; user filenames are metadata only
- validated IDs and no arbitrary input paths
- no OCR, shell, file-write chat capability, internet retrieval, or Agent tools
- uploaded text is treated as untrusted data in the prompt
- CORS permits only the local Vite development origins
- unexpected backend errors are logged server-side and returned as controlled messages

## Known limitations

- Word-based chunking ignores sentence and section boundaries.
- `all-minilm` and raw cosine retrieval can rank broadly related text above the best passage.
- The `0.30` threshold is heuristic and corpus-dependent.
- There is no reranker, hybrid keyword/vector search, OCR, authentication, or automatic factual verification.
- A local 4B Qwen model may produce weaker wording or citation behavior than larger models.
- Gemini requires a valid API key and external network access.
- Chroma indexes persist locally, but chat history and the active-document selection exist only in the current browser session.
- This is an educational modular monolith, not a production multi-user service.
