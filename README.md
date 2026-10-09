# Lumen Institute Student Handbook — Document-Grounded RAG

This repository is the focused midterm version of the AI Homework Lab. It is a closed-domain Retrieval-Augmented Generation (RAG) assistant that answers from exactly one assigned knowledge base:

```text
knowledge/KB-A_Lumen-Institute-Student-Handbook.md
```

The runtime chat endpoint rejects any document ID other than the initialized handbook ID. There is no general-chat fallback, Agent loop, tool calling, LangChain, LlamaIndex, OCR, reranker, hybrid search, authentication, or external web retrieval.

## Midterm configuration

| Component | Current value |
|---|---|
| Frontend | React + Vite |
| Backend | Python + FastAPI |
| Assigned knowledge base | `KB-A_Lumen-Institute-Student-Handbook.md` |
| Embedding model | Ollama `all-minilm:latest` |
| Embedding dimension | 384 |
| Vector database | Persistent local Chroma |
| Distance metric | Cosine |
| Chunk size | 180 words |
| Chunk overlap | 30 words |
| Retrieval | One query embedding, Global Top-K = 3 |
| Relevance gate | `MIN_SIMILARITY = 0.30` (heuristic) |
| Default generation LLM | Ollama `qwen3:4b-instruct` |
| Optional generation provider | Gemini `gemini-3.5-flash` |

Section 11 of the handbook contains amendments. The grounded system prompt explicitly tells the generation model to apply a later amendment when it conflicts with an earlier rule and to preserve exceptions that the amendment says remain unchanged.

## RAG flow

Initialization:

```text
assigned handbook
  → UTF-8 Markdown text extraction
  → 180-word chunks with 30-word overlap
  → explicit all-minilm embeddings
  → float32 L2 normalization
  → one persistent Chroma index
  → assigned_knowledge_base.json records the active document ID and source hash
```

Question answering:

```text
question + assigned handbook document ID
  → embed the question once with all-minilm
  → Chroma cosine query
  → Global Top-3 chunks
  → highest-similarity threshold gate
  → labelled context with full chunk IDs
  → Qwen local or optional Gemini generation
  → grounded answer with citations and retrieval debug metadata
```

The LLM does not search Chroma. Python performs retrieval first and only sends the retrieved chunk text to the configured generation provider.

For Markdown indexing, layout-only markers such as heading hashes and table pipes are removed from the embedding input. The original chunk text remains unchanged in Chroma for evidence and citations. The final chunk has no following overlap window, so its last 30 words are moved to the front of its embedding input to keep end-of-file FAQ facts visible to the compact embedding model. This does not change chunk boundaries, stored text, chunk IDs, or the context sent to the generation model.

For Markdown indexing, layout-only markers such as heading hashes and table pipes are removed from the embedding input. The original chunk text remains unchanged in Chroma for evidence and citations. The final chunk has no following overlap window, so its last 30 words are moved to the front of its embedding input to keep end-of-file FAQ facts visible to the compact embedding model. This does not change chunk boundaries, stored text, chunk IDs, or the context sent to the generation model.

## Project structure

```text
ai-homework-lab/
├── knowledge/
│   └── KB-A_Lumen-Institute-Student-Handbook.md
├── backend/
│   ├── app.py                  FastAPI routes and fixed-KB enforcement
│   ├── config.py               Paths, model names, chunking, Top-K, threshold
│   ├── knowledge_base.py       Idempotent assigned-handbook initialization
│   ├── schemas.py              Pydantic API contracts
│   ├── documents/
│   │   ├── extractor.py        TXT/MD/PDF/DOCX text extraction
│   │   ├── storage.py          Safe local document identity and metadata
│   │   └── indexer.py          Extract → chunk → embed → Chroma
│   └── rag/
│       ├── chunker.py
│       ├── embedder.py
│       ├── vector_store.py
│       ├── retriever.py
│       ├── prompt_builder.py
│       ├── generator.py
│       └── service.py
├── frontend/                   React/Vite handbook chat and debug UI
├── scripts/
│   ├── initialize_kb.py        Initialize/reuse the assigned handbook index
│   ├── evaluate_rag.py         Focused 20-question midterm evaluation
│   └── run_api.py              Uvicorn entry point
├── tests/rag_cases.json        Expected facts, amendment checks, refusals
├── requirements.txt
└── README.md
```

## Install

Requirements: Windows, Python, Node.js/npm, and a working local Ollama installation.

Pull the local models:

```powershell
ollama pull all-minilm:latest
ollama pull qwen3:4b-instruct
```

Create the Python environment and install backend dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

Create local configuration from the safe example:

```powershell
Copy-Item .env.example .env
```

The backend loads the root `.env` automatically. `.env` is ignored by Git and API keys are never returned to React or printed.

## Initialize or re-index the assigned handbook

From the repository root, with Ollama running:

```powershell
.\.venv\Scripts\python.exe scripts\initialize_kb.py
```

The command uses the existing extraction, chunking, embedding, normalization, and Chroma code. It is idempotent: if the same handbook content already has a valid index, it reuses it. If the source file changes or the stored index is unavailable, it creates a fresh assigned index and updates the ignored local state file.

This is indexing, not LLM training. No model weights are changed.

## Run the application

Backend terminal, from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\run_api.py
```

Frontend terminal:

```powershell
cd frontend
npm run dev
```

Open `http://localhost:5173`. The UI automatically loads the initialized assigned handbook. It does not ask the user to choose arbitrary runtime documents.

## API

Health check:

```http
GET /health
```

Assigned Knowledge Base metadata:

```http
GET /knowledge-base
```

The response exposes safe metadata such as the title, filename, document ID, status, word count, chunk count, embedding model, and embedding dimension. It never exposes a filesystem path.

Document-grounded chat:

```http
POST /chat
Content-Type: application/json

{
  "question": "What is the current make-up exam fee after the amendment?",
  "document_ids": ["the-initialized-assigned-document-id"]
}
```

`/chat` verifies that the request contains exactly the assigned handbook ID. The response includes:

- grounded answer;
- full document-scoped citations;
- retrieved source and chunk IDs;
- cosine similarity scores;
- final Global Top-K;
- candidate count;
- query-vector dimension;
- heuristic threshold and gate result;
- `generation_called` and `answerable` flags.

## Debug view

After asking a question, expand **View RAG details** in the React UI. It shows the exact retrieved chunk IDs and sources, their scores, candidate count, Global Top-K, 384-dimensional query-vector size, threshold, retrieval gate, generation flag, and answerability result.

The command-line evaluation prints a text preview for each retrieved Top-K chunk, making retrieval errors distinguishable from generation errors.

## Focused evaluation

Run with Ollama and the configured generation provider available:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag.py
```

The evaluator initializes/reuses only the assigned handbook, then runs 20 cases from `tests/rag_cases.json`:

- 16 direct handbook questions;
- 2 semantic paraphrases;
- 2 questions not answered by the handbook;
- explicit amendment retrieval checks for the make-up exam fee, library fine, illness exception, and Honors installment restriction;
- expected fact checks and valid-citation checks;
- full Top-K debug output for every question.

An answerable case passes only when generation occurs, required facts are present, citations refer to actual retrieved chunks, and any required amendment chunk appears in Top-K. An unanswerable case passes only when the exact insufficient-information response is returned.

## Generation provider

Default local generation in `.env`:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_GENERATION_MODEL=qwen3:4b-instruct
```

Optional Gemini generation:

```dotenv
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-3.5-flash
GEMINI_API_KEY=your-api-key
```

Restart the backend after changing `.env`. Provider selection changes only answer generation. The all-minilm embedding space and Chroma index do not change, so switching Qwen/Gemini does not require re-indexing.

## Grounding and refusal behavior

If Top-1 similarity is below `0.30`, Python does not call the generation provider and returns:

> The knowledge base does not contain enough information to answer this question.

If retrieval passes but the supplied chunks still do not support the answer, the system instruction requires the model to return the same sentence. Every factual paragraph must cite an exact retrieved full chunk ID. The prompt forbids outside knowledge, invented citations, and following instructions found inside document text.

The `0.30` threshold is an educational heuristic, not a confidence percentage or universal boundary. It can still produce false positives or false negatives.

## Known limitations

- The knowledge base is one small assigned handbook.
- Word-based chunking ignores sentence and section boundaries.
- all-minilm and raw cosine retrieval can rank imperfectly.
- Global Top-K is limited to three chunks and there is no reranker.
- The relevance threshold is heuristic.
- Text-based PDFs are supported, but OCR is not implemented.
- A local 4B generation model can be probabilistic or phrase an answer imperfectly.
- There is no persistent chat memory, authentication, SQL database, Agent, or external internet tool.
- The system checks retrieved citations but does not perform independent factual verification.
