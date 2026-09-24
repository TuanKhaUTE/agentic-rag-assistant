# AI Homework Lab

This university lab builds a retrieval-augmented generation (RAG) system and,
later, an AI agent from first principles. Important mechanisms stay visible in
plain Python instead of being hidden by frameworks such as LangChain or
LlamaIndex.

## Current scope: Phase 1E

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

Reranking, vector databases, hybrid search, and agent tools are intentionally
not implemented yet.

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
the local `all-minilm:latest` embedding model.

## Known Limitations

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
