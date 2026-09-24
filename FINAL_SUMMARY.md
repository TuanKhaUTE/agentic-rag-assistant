# AI Homework Lab — Final Summary

## What Was Built

### Part 1: Standalone RAG

```text
documents
→ chunks
→ all-minilm embeddings
→ normalized NumPy vectors
→ cosine similarity
→ Top-K retrieval
→ relevance threshold
→ labelled context
→ grounded Qwen answer with citations
```

The RAG system loads five local study documents, creates overlapping chunks,
and stores one 384-dimensional vector per chunk. A question is embedded with
the same model and compared with every stored vector. Low-relevance questions
are rejected before generation; relevant chunks become labelled prompt context.

### Part 2: Controlled Agent

```text
user goal
→ Qwen decision
→ native structured tool call
→ Python allowlist and argument validation
→ tool execution
→ observation
→ Qwen next decision
→ final answer
```

The loop is visible and limited to five model steps. Python, not the model,
executes functions. Every tool call, validation decision, and observation is
printed.

## RAG Versus Agent

RAG follows a fixed retrieval pipeline: it searches the knowledge base before
generation and supplies the retrieved evidence to the model.

The Agent dynamically decides whether a tool is required. It can choose among
several tools, use multiple tools across several steps, or answer directly. The
standalone RAG retriever becomes one Agent tool through `search_knowledge`.

## Available Tools

- `search_knowledge`: retrieves relevant local knowledge chunks; it does not generate.
- `calculator`: evaluates basic arithmetic through a restricted AST parser.
- `list_files`: lists files inside an approved project directory.
- `read_file`: reads size-limited UTF-8 text files inside the project.

## Study and Software Engineering Uses

For study, the Agent can search course notes, summarize approved files,
calculate values, and combine retrieved information with calculations.

For software engineering work, the same pattern can inspect project
documentation, search technical knowledge, and perform calculations. It could
later be extended with separately approved tools for tests or Git, but those
capabilities are **not implemented in this homework**.

## Safety

- Only four explicitly allowlisted tools can execute.
- Tool arguments are validated against simple schemas.
- Calculator input cannot use `eval`, function calls, names, or attributes.
- File paths must remain inside the project and cannot use parent traversal.
- There is no shell tool.
- There is no external-network tool.
- There is no file-writing tool.
- `MAX_STEPS = 5` prevents an unbounded model/tool loop.
- Tool failures become structured observations instead of uncontrolled crashes.

## Limitations

- The local 4B model can misunderstand a request or construct imperfect arguments.
- Only four tools are available.
- The RAG corpus contains only five small documents.
- There is no persistent memory between runs.
- There is no internet access.
- Vector retrieval can rank a generally related chunk above the best passage.
- Agent behavior is probabilistic, so repeated runs may differ.
- There is no reranker, vector database, or automatic factual verification.

## Recommended Lecturer Demo

1. Ask a standalone RAG question with `ask.py`.
2. Show its query embedding, similarity scores, and Top-K chunks with
   `inspect_retrieval.py`.
3. Show the labelled context and grounded cited answer.
4. Run a calculator Agent task and point out the structured call and observation.
5. Run a knowledge Agent task and show RAG being selected as a tool.
6. Run the multi-tool task so Qwen chooses knowledge search, then calculator.
7. Explain the repeating **Model → Tool Call → Python Validation → Observation → Model** loop.

Suggested commands:

```powershell
.\.venv\Scripts\python.exe inspect_retrieval.py "What is RAG?"
.\.venv\Scripts\python.exe ask.py "What is RAG and why is it useful?"
.\.venv\Scripts\python.exe run_agent.py "Calculate (125 * 8) / 4."
.\.venv\Scripts\python.exe run_agent.py "What is an AI agent? Use the study knowledge base."
.\.venv\Scripts\python.exe evaluate_agent.py
```
