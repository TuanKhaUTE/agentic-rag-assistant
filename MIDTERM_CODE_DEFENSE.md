# Midterm Code Defense Map — AI Homework Lab

> Mục tiêu: tài liệu học để bảo vệ đúng **repository hiện tại**. Toàn bộ số dòng được audit lại từ working tree ngày 2026-10-01. Đây là bản đồ code thực tế, không phải kiến trúc lý tưởng chung chung.

## Audit snapshot và consistency verdict

- Working tree sạch trước khi refresh tài liệu.
- Đã đọc lại 25 core runtime files được yêu cầu: React services/UI, FastAPI schemas/endpoints, document package, shared RAG modules, Agent loop/registry, toàn bộ `src/tools/*`, config, static builder và hai evaluation scripts.
- Các entry point, package facade, static loader, inspection scripts, test data, dependencies và ignore rules cũng được đối chiếu bổ sung.
- Runtime nhất quán: `all-minilm:latest` tạo embedding; Chroma local lưu/query vector; RAG generation mặc định dùng `qwen3:4b-instruct` và có thể chuyển sang `gemini-3.5-flash`; Agent luôn dùng Ollama/Qwen.
- Uploaded Knowledge Base hỗ trợ 1–N file TXT/MD/PDF/DOCX. Mỗi document có Chroma index riêng; query lấy candidate từ mọi document đang active rồi chọn **một GLOBAL Top-K**.
- `knowledge/*.txt` là sample/regression corpus cho static RAG và Agent tool. Nó không phải teacher-uploaded Knowledge Base.
- Không có thay đổi runtime trong lần refresh này; chỉ file này được cập nhật.

## 1. Assistant architecture

### 1.1 General mode

```text
React
→ POST /chat
→ bounded Ollama/Qwen Agent
→ optional allowlisted tool call(s)
   ├── search_knowledge → static sample Chroma
   ├── calculator
   ├── list_files
   └── read_file
→ observation
→ Qwen next decision
→ final answer
```

Khi `activeDocuments.length === 0`, `App.sendMessage` đi nhánh `sendChatMessage` tại `frontend/src/App.jsx:239-250`. FastAPI `/chat` gọi đúng shared Agent tại `src/api/app.py:59-74`. Agent có thể trả lời trực tiếp hoặc gọi tool; nó không tự động dùng uploaded indexes.

### 1.2 Uploaded Knowledge Base mode

```text
1–N files
→ upload từng file
→ extract TXT / MD / PDF / DOCX
→ chunk từng file
→ all-minilm embeddings + L2 normalization
→ mỗi document có một local Chroma index riêng

Question
→ one all-minilm query embedding
→ query every ACTIVE uploaded document
→ collect per-document candidates
→ merge candidates
→ sort globally by cosine similarity
→ FINAL GLOBAL TOP_K
→ relevance gate
→ labelled context
→ grounded prompt
→ configured RAG generator
   ├── Ollama / qwen3:4b-instruct
   └── Gemini / gemini-3.5-flash
→ answer + document-scoped citations
```

Khi có ít nhất một active document, `App.sendMessage` snapshot array và gửi ID list tại `frontend/src/App.jsx:211-228`. Endpoint `POST /documents/ask` nhận list tại `src/api/app.py:172-215`; single-document compatibility vẫn tồn tại qua `POST /documents/{document_id}/ask` tại lines 218-259.

### 1.3 Frontend entry và mode ownership

| File / lines | Trách nhiệm hiện tại | Điểm bảo vệ |
|---|---|---|
| `frontend/index.html:17-18` | Tạo `#root`, nạp `/src/main.jsx`. | React bắt đầu từ DOM mount này. |
| `frontend/src/main.jsx:1-12` | Import `App`/CSS và render trong `StrictMode`. | Không có model logic ở entry. |
| `frontend/src/App.jsx:154-177` | Giữ chat state, `activeDocuments`, `pendingDocuments`, aggregate chunk count và mode label. | State chỉ nằm trong browser; refresh sẽ mất selection/history. |
| `frontend/src/App.jsx:204-266` | Router thật của unified chat: active IDs → uploaded RAG; empty array → `/chat`. | Không có hai trang/chat riêng. |
| `frontend/src/styles.css:95-1106` | Academic Evergreen layout, messages, RAG evidence, active-document chips và composer. | Presentation only; không chứa AI logic. |

### 1.4 Current end-to-end flows

#### Flow A — Teacher provides three files

1. `<input multiple>` nhận 1–N file tại `App.jsx:567-575`.
2. `selectDocuments` chuyển `FileList` thành array, validate từng file tại lines 324-338.
3. `prepareDocuments` tạo pending records rồi loop `for (const pending of queued)` tại lines 281-314.
4. Mỗi vòng gọi `uploadDocument` rồi `indexDocument` tại lines 295-301.
5. `/documents/upload` đọc tối đa `MAX_UPLOAD_BYTES + 1` và gọi `create_document` tại `app.py:77-108`.
6. Storage sinh UUID, lưu `original.ext`, extract và lưu metadata tại `storage.py:48-88`.
7. `/documents/{id}/index` gọi indexer tại `app.py:131-151`.
8. Indexer re-extracts, chunks, embeds, normalizes và rebuilds Chroma riêng tại `indexer.py:30-73`.
9. Mỗi thành công được append vào `activeDocuments` tại `App.jsx:298-301`.

#### Flow B — Question across three active files

1. UI snapshot `activeDocuments` tại `App.jsx:211`.
2. IDs được collect bằng `.map(document => document.document_id)` tại lines 225-227.
3. `askDocuments` gửi `{document_ids, question}` đến `/documents/ask` tại `documentApi.js:72-85`.
4. Pydantic nhận non-empty list qua `MultiDocumentQuestionRequest` tại `schemas.py:45-46`.
5. API gọi `answer_uploaded_documents` tại `app.py:176-180`.
6. Retriever load/validate từng ID trong loop tại `retriever.py:82-110`.
7. Question được embed đúng một lần tại line 114.
8. Mỗi collection được query với cùng vector tại lines 116-132.
9. Candidates được merge bằng `candidates.extend(...)` line 132.
10. Global sort và final slice xảy ra tại lines 137-140.
11. QA dùng score cao nhất của merged set để gate tại `question_answering.py:44-46`.
12. Prompt labels giữ filename + full document-scoped chunk ID tại `prompt_builder.py:49-65`.
13. Generator switch provider tại `generator.py:58-70` và trả answer có citation.

#### Flow C — Detach one file

`detachDocument` filter đúng `document_id` khỏi frontend array tại `App.jsx:340-346`. Server-side file/index không bị xóa. Lần hỏi sau ID đó không xuất hiện trong request array, nên retriever không load hay query collection đó.

#### Flow D — Detach all

`detachAllDocuments` đặt array thành `[]` tại `App.jsx:351-357`. Lần send sau điều kiện ở line 224 false, nên đi `sendChatMessage` lines 239-250 và trở lại General Agent `/chat`.

#### Flow E — Same uploaded Knowledge Base with Qwen versus Gemini

Upload, extraction, chunks, all-minilm vectors, Chroma candidates, global ranking, gate và prompt đều giống nhau. Điểm switch duy nhất là `generate_answer` tại `generator.py:58-70`: `LLM_PROVIDER=ollama` gọi lines 17-34; `LLM_PROVIDER=gemini` gọi lines 37-55. Đổi generator không rebuild Chroma.

#### Flow F — General calculator Agent

1. No active uploads → `/chat`.
2. Qwen sees tool schemas tại `agent_loop.py:81-92`.
3. Model proposes `calculator` call; Python parses call lines 93-115.
4. Registry validates allowlist/schema và executes function tại `tool_registry.py:65-108`.
5. Calculator walks restricted AST tại `calculator_tool.py:39-76`.
6. Tool result becomes JSON observation tại `agent_loop.py:119-134`.
7. Next Qwen turn reads observation and returns final answer.

## 2. Model selection, switching and inference

### 2.1 Configuration

| File / lines | Exact responsibility | Defense answer |
|---|---|---|
| `config.py:18` | `EMBEDDING_MODEL = "all-minilm:latest"`. | Same semantic vector space for chunks and questions. |
| `config.py:20` | `LLM_PROVIDER`, default `ollama`. | Selects only fixed RAG answer generation. |
| `config.py:21-24` | Qwen default and optional Ollama model override. | Default local RAG generator is `qwen3:4b-instruct`. |
| `config.py:25` | Gemini model env/default. | Current intended default is exactly `gemini-3.5-flash`. |
| `config.py:26` | Gemini key from environment, blank by default. | Secret is backend-only and not committed. |
| `config.py:28-29` | `GENERATION_MODEL = DEFAULT_QWEN_MODEL`. | Agent remains Ollama/Qwen even if RAG provider is Gemini. |

### 2.2 Inference code

- Embedding inference: `src/embedder.py:14` calls `ollama.embed(model=EMBEDDING_MODEL, input=texts)`.
- Ollama RAG generation: `src/generator.py:17-34` calls `ollama.chat` with low temperature, 4096 context and 512 output limit.
- Gemini RAG generation: `src/generator.py:37-55` validates key/model then calls `client.models.generate_content(model=GEMINI_MODEL, ...)`.
- Provider switch: `src/generator.py:58-70`; unknown provider causes controlled error and never silently falls back.
- Agent inference/tool selection: `src/agent/agent_loop.py:83-92` uses fixed `GENERATION_MODEL` plus tool schemas.

### 2.3 Responsibility table

| Component | Input → output | Responsibility | Not responsible for |
|---|---|---|---|
| `all-minilm:latest` | text(s) → observed 384-dimensional vectors | Chunk/query embedding | Answers, tools, path safety |
| Chroma | explicit vectors/metadata ↔ distances/records | Persistent local vector index and cosine nearest-neighbor query | Embedding model choice, answer generation |
| `qwen3:4b-instruct` | messages/context/schemas → text or tool call | Default grounded RAG answer and only Agent controller | Extracting/indexing files |
| `gemini-3.5-flash` | grounded RAG prompt → answer text | Optional cloud RAG generation | Agent loop/tool selection |

Không có training/fine-tuning. Project chỉ chạy inference: `extract → chunk → embed → index`; lúc hỏi: `embed query → retrieve → generate`.

## 3. Document loading and extraction

### 3.1 Static sample loader

`src/document_loader.py:12-33` chỉ glob `knowledge/*.txt` theo filename order, đọc UTF-8, bỏ empty file và tạo records gồm `path/source/title/text/word_count`. Static builder dùng loader này; uploaded exam files không đi qua nó.

### 3.2 Uploaded file selection and HTTP upload

| Location | Current behavior |
|---|---|
| `App.jsx:30-31` | Client recognizes TXT/MD/PDF/DOCX and mirrors 10 MB limit. |
| `App.jsx:131-146` | Reject missing, unsupported, empty or oversized selected file for UX. |
| `App.jsx:324-338` | Validates every selected file before preparing the batch. |
| `App.jsx:567-575` | File input has `accept=".txt,.md,.pdf,.docx"` and `multiple`. |
| `documentApi.js:26-38` | Sends one file per multipart `POST /documents/upload`; browser supplies boundary. |
| `app.py:77-108` | Backend authoritative validation, bounded read, safe exception mapping and file close. |

### 3.3 Exact format extraction

| Format | Current code | Explanation |
|---|---|---|
| Allowlist | `extractor.py:10,21-28` | Only `.txt`, `.md`, `.pdf`, `.docx`; extension normalized lowercase. |
| TXT | `extractor.py:31-47` | Shared UTF-8 reader; rejects invalid encoding/blank text. |
| MD | `extractor.py:50-52` | Calls the same UTF-8 plain-text helper; no Markdown parser/library. |
| PDF | `extractor.py:55-66` | `PdfReader`; joins text from pages. Empty text reports likely scanned PDF/OCR need. |
| DOCX | `extractor.py:69-83` | `python-docx`; collects paragraphs and table cells; rejects corrupt/empty files. |
| Dispatcher | `extractor.py:86-96` | Maps allowed extension to exact extractor. |

OCR is intentionally absent. An image-only PDF is different from a text-based PDF because `page.extract_text()` cannot recover pixels as text.

### 3.4 Safe upload storage

- Filename sanitation: `storage.py:25-30` keeps only basename metadata.
- Document ID validation: `storage.py:33-36` allows only 32 lowercase hexadecimal characters before joining paths.
- Size/empty/type checks and UUID directory creation: `storage.py:48-64`.
- Original bytes, extraction and safe metadata: `storage.py:65-88`.
- Original filename is stored in `metadata["filename"]` at lines 68-75; this later becomes chunk `source`.
- Safe internal original path supports all four types at `storage.py:111-120`.
- Metadata status update uses temporary file then replace at `storage.py:123-142`.

Upload/extraction is ingestion, not yet RAG. Chunking, embeddings and Chroma happen only after `/index`.

## 4. Chunking

### FILE: `src/chunker.py`

| Lines | Mechanism | Why it matters |
|---|---|---|
| 1-12 | Validate positive size, nonnegative overlap and overlap smaller than size. | Prevent invalid/infinite windows. |
| 14-19 | Split whitespace words; `step = chunk_size - overlap`. | With 180/30, next start advances 150 words. |
| 21-34 | Build `[start_word,end_word)` window and metadata. | Exact chunk text remains traceable. |
| 25 | ID is `<prefix>#chunk-NNN`. | Static prefix is filename; uploaded prefix is document ID. |
| 26-33 | Preserve `source`, index, ranges and optional `document_id`. | Filename/ID survive vector storage and multi-document merging. |
| 36-38 | Stop after last word, otherwise advance by step. | No duplicate tail/infinite loop. |
| 43-53 | Batch wrapper for static documents. | Uploaded indexer deliberately reuses `chunk_document`. |

Shared configuration is `CHUNK_SIZE_WORDS = 180` and `CHUNK_OVERLAP_WORDS = 30` at `config.py:15-16`. Overlap preserves ideas near boundaries but increases duplicate text/vector count.

## 5. Embeddings

### FILE: `src/embedder.py`

| Lines | Mechanism | Defense point |
|---|---|---|
| 7-14 | Validate ordered non-empty strings; call official Ollama embedding API. | One chunk becomes one vector; order is preserved. |
| 15-25 | Verify output count and equal dimensions. | Prevent chunk/vector misalignment. |
| 27-31 | Convert to 2D `numpy.float32`. | Compact predictable matrix type. |
| 34-48 | Compute L2 norm, reject zero vectors, divide and verify norm≈1. | Supports cosine geometry and catches bad vectors. |

Documents and query must use the same model/dimension because cosine similarity is meaningful only inside the same embedding space. Changing Qwen/Gemini does not affect these embeddings; changing all-minilm requires re-indexing.

## 6. Chroma vector database

### 6.1 Shared vector adapter: `src/vector_store.py`

| Lines | Exact behavior | Lecturer point |
|---|---|---|
| 16-22 | `chromadb.PersistentClient(path=..., telemetry=False)`. | Local persistent DB creation. |
| 32-47 | Validate chunk count, 2D rows, finite values and normalized vectors. | Enforces one vector row per chunk. |
| 50-61 | Delete existing named collection before rebuild. | Re-index is idempotent, not append/duplicate. |
| 63-77 | Store model/dimension/cosine metadata; create with `embedding_function=None`. | Chroma cannot hide or select an embedder. |
| 79-92 | Build IDs, document texts and metadata including source and optional document ID. | Source identity is persisted. |
| 94-102 | Explicit `collection.upsert(... embeddings=embeddings.tolist())`; verify count. | Exact vector database write. |
| 105-126 | Load and validate model, cosine configuration and non-empty collection. | Prevent incompatible/stale index use. |
| 129-130 | Load static collection from `storage/chroma`. | Static path stays separate from uploads. |

### 6.2 Static index

`build_index.py:20-65` performs visible `load_documents → chunk_documents → embed_texts → alignment check → normalize_embeddings → rebuild_chroma_collection`. The collection metadata marks `index_kind=static_knowledge` at line 64.

### 6.3 Per-upload indexes

`indexer.py:26-27` resolves `uploads/<document_id>/chroma`. `index_uploaded_document` at lines 30-73 re-extracts, creates `document_id`/source-aware records, uses the shared chunker/embedder, and rebuilds one collection. `load_uploaded_collection` lines 76-93 validates collection identity and chunk count.

Separate physical indexes make per-document isolation and rebuilding easy to explain. Multi-file retrieval combines results at request time; it does not create another combined persistent index.

## 7. Retrieval, cosine similarity and GLOBAL Top-K

### 7.1 Generic Chroma query

| `src/vector_store.py` lines | Current mechanism |
|---|---|
| 133-143 | Validate, embed and L2-normalize one question vector; verify expected dimension. |
| 146-163 | Convenience query embeds text then delegates to the shared-vector function. |
| 166-190 | Validate precomputed vector for multi-document reuse. |
| 192-199 | Send explicit `query_embeddings`, bounded `n_results`, fields and optional metadata filter to `collection.query`. |
| 201-206 | Require aligned returned IDs/documents/metadata/distances. |
| 208-233 | Convert cosine distance using `similarity = 1.0 - distance`; preserve rank, source, text, ranges and document ID. |
| 262-267 | Static `search` loads only `static_knowledge` then uses the same query path. |

### 7.2 Single-document compatibility

`retrieve_uploaded_document` remains at `retriever.py:17-62`. It validates indexed/model/dimension state, queries the one physical collection with `where={"document_id": document_id}`, then verifies returned identities. The legacy UI no longer uses this path, but CLI and `POST /documents/{id}/ask` remain compatible.

### 7.3 Multi-document retrieval

| `src/documents/retriever.py` lines | Exact behavior |
|---|---|
| 65-80 | Validate non-empty ID list, string IDs, no duplicates and positive integer K. |
| 82-110 | Loop through every selected ID; require indexed state, same embedding model/dimension; load its collection. |
| 114 | Embed the question **once** after all dimensions agree. |
| 116-124 | Query each selected collection with the same query vector and up to K candidates. |
| 125-132 | Recheck document identity then merge with `candidates.extend(document_candidates)`. |
| 137 | Sort all merged candidates descending by `score`. |
| 138-140 | `top_results = candidates[:top_k]`; reset final global ranks. |
| 142-147 | Return metadata, one query vector, candidate count and final results. |

`TOP_K = 3` means at most **three final chunks total**, not three chunks per file in the prompt. Internally the code may retrieve up to three candidates from each separate collection so they can compete in one global ranking.

Top-K always returns the nearest available records even when all are irrelevant. Therefore ranking is not proof of answerability; the relevance gate is separate.

### 7.4 Source identity after merge

1. Indexer copies original filename to `source` at `indexer.py:36-40`.
2. Chroma metadata stores `source` and `document_id` at `vector_store.py:83-92`.
3. Query mapping restores both at `vector_store.py:217-232`.
4. Candidate merge retains entire result dict at `retriever.py:132`.
5. Prompt label becomes `[SOURCE filename | document_id#chunk-NNN]` at `prompt_builder.py:55-65`.

## 8. Prompt construction and grounded generation

### 8.1 Relevance gate: `src/documents/question_answering.py`

| Lines | Meaning |
|---|---|
| 15-19 | Documents that 0.30 is an educational heuristic calibrated on static data. |
| 22-28 | Single-document compatibility wrapper. |
| 31-37 | Multi-document wrapper calls merged retriever. |
| 40-46 | Shared QA gets final results and gates on the highest globally ranked score. |
| 48-60 | Gate pass builds prompt/calls generation; gate fail returns exact refusal without provider call. |
| 62-64 | Even after gate pass, generator may return exact refusal; that makes `answerable=False`. |
| 66-80 | Return retrieval data, threshold, gate, generation flag, prompt/context and answer. |

Ba flag phải nói đúng:

- `retrieval_gate_passed`: highest global score ≥ `MIN_SIMILARITY`.
- `generation_called`: provider thực sự được gọi; false khi gate fail.
- `answerable`: final output khác exact insufficient-information sentence.

### 8.2 Prompt builder: `src/prompt_builder.py`

| Lines | Current behavior |
|---|---|
| 1-3 | One exact insufficient-information response shared by gate/model checks. |
| 6-17 | Static grounding rules: context-only, retrieved text is data, exact citations/refusal. |
| 20-33 | Uploaded rules require full document-scoped chunk IDs and treat all uploaded texts as untrusted data. |
| 36-45 | Validate non-empty question/chunks. |
| 47-65 | Sanitize filename label; preserve source and chunk ID; construct labelled blocks. |
| 67-83 | Join context, list valid uploaded citation IDs and build user message. |
| 86-95 | Static wrapper. |
| 98-108 | Uploaded Knowledge Base wrapper. |

Grounding is an instruction plus labelled evidence, not a mathematical guarantee. Production API does not deterministically reject a malformed citation after generation; `evaluate_rag.py` does validate citations for the static test suite.

### 8.3 Generator boundary

`generator.py` does not retrieve anything. It accepts only prepared system/user messages. Ollama implementation is lines 17-34; Gemini implementation lines 37-55; provider branch lines 58-70. A correct-sounding answer unsupported by final context is still a RAG failure.

## 9. RAG verification and unanswerable behavior

### 9.1 Static evaluation: `evaluate_rag.py`

| Lines | Verification |
|---|---|
| 25-29 | Load non-empty `tests/test_questions.json`. |
| 32-56 | Run static Chroma retrieval and record Top-1/Top-3/expected-source presence. |
| 86-118 | Print answerable/unanswerable score ranges and separation. |
| 121-154 | Explain why 0.30 is a corpus/test heuristic. |
| 157-158 | Extract only valid static citation shape. |
| 161-188 | Apply threshold before provider generation. |
| 190-240 | Validate citations, expected source and unanswerable refusal behavior. |
| 275-302 | Print Chroma/model/provider/pass summary. |

`tests/test_questions.json:2-25` contains four answerable cases; lines 26-43 contain three intentionally unanswerable cases. This is evaluation data, not training data.

### 9.2 Failure vocabulary

- False positive: irrelevant highest score still reaches 0.30, so generation runs on weak evidence.
- False negative: relevant evidence scores below 0.30, so generation is skipped.
- Retrieval failure: relevant chunk is absent from final Top-K or gate fails.
- Generation failure: evidence is correct but model answers incorrectly, omits/changes citation or refuses incorrectly.
- Threshold is not a probability or universal confidence score.

### 9.3 Inspection scripts

| Script / current lines | Visible evidence |
|---|---|
| `inspect_chunks.py:6-56` | Static documents, ranges, overlap, full chunks and totals. |
| `inspect_embeddings.py:7-55` | Chroma count/dimension/dtype, first values, norms and one-vector-per-chunk check. |
| `inspect_retrieval.py:14-74` | Query vector, all Chroma scores, Top-K full chunks and distance/similarity explanation. |
| `inspect_document.py:7-28` | Extraction metadata/preview for any currently supported extension, including MD. |
| `inspect_uploaded_index.py:10-56` | Per-upload Chroma metadata, chunks/vectors and normalization. |

## 10. Agent and tools

### 10.1 Agent loop: `src/agent/agent_loop.py`

| Lines | Mechanism |
|---|---|
| 12 | `MAX_STEPS = 5`. |
| 14-30 | System rules for tool use, multi-tool completion, search grounding and citations. |
| 35-39 | Immutable result with answer, step count and ordered tools. |
| 63-75 | Validate goal and initialize system/user conversation. |
| 81-92 | Bounded loop; Qwen receives messages plus Ollama tool schemas. |
| 93-103 | No tool calls means final answer; direct answers are allowed. |
| 105-115 | Parse proposed calls and pass each to Python dispatcher. |
| 116-134 | Serialize validation/result/error and append `role="tool"` observation. |
| 136-142 | Deterministic safe stop after exhausted steps. |
| 145-160 | CLI string wrapper and API metadata wrapper reuse the same loop. |

### 10.2 Registry and validation: `src/agent/tool_registry.py`

- Allowlisted mapping is lines 6-11.
- Ollama function schema wrapper is lines 18-21.
- JSON type/required/extra/minimum validation is lines 24-62.
- Unknown names/non-object arguments are rejected at lines 65-81.
- Validated callable execution is lines 83-108.

The model proposes a name and arguments; Python validates and executes. Qwen never directly executes Python or shell commands.

### 10.3 Four tools

| Tool | Current code | Responsibility and safety |
|---|---|---|
| `search_knowledge` | `knowledge_tool.py:28-47` | Retrieval-only static sample search; no RAG generator. |
| `calculator` | `calculator_tool.py:39-76` | Restricted AST for numbers, `+ - * /`, parentheses/unary signs; no `eval`/`exec`. |
| `list_files` | `file_tools.py:73-85` | Direct child files only inside resolved project root. |
| `read_file` | `file_tools.py:88-108` | Allowed extension, UTF-8 and 1 MB maximum; no writes. |

Tool schemas are assembled at `src/tools/__init__.py:11-16`. `evaluate_agent.py:19-60` defines six scenarios; lines 63-106 check citations/results; lines 109-152 record real runs; lines 172-248 verify unknown tool rejection, calculator rejection, traversal rejection and MAX_STEPS.

Uploaded Knowledge Base retrieval is deliberately not a new Agent tool. General Agent `search_knowledge` still queries only sample static Chroma through `vector_store.search`.

## 11. Exact exam code locations

| # | Current file : lines | Exact purpose |
|---:|---|---|
| 1 | `frontend/src/App.jsx:159-177` | `activeDocuments`/pending state, aggregate chunks and current mode. |
| 2 | `frontend/src/App.jsx:211-228` | Snapshot active documents, collect ID array and call uploaded Knowledge Base API. |
| 3 | `frontend/src/App.jsx:239-250` | Fallback to general `/chat` when active array is empty. |
| 4 | `frontend/src/App.jsx:281-314` | Upload/index loop for multiple selected files. |
| 5 | `frontend/src/App.jsx:324-338` | Convert multi-file input and validate every file. |
| 6 | `frontend/src/App.jsx:340-346` | Detach one active document by filtering its ID. |
| 7 | `frontend/src/App.jsx:351-357` | Detach all documents and restore empty active set. |
| 8 | `frontend/src/App.jsx:518-545` | Render aggregate Knowledge Base plus per-document detachable chips. |
| 9 | `frontend/src/App.jsx:567-575` | Exact multiple-file input and four accepted extensions. |
| 10 | `frontend/src/App.jsx:88-128` | RAG evidence UI including filenames, chunk IDs, scores and gate flags. |
| 11 | `frontend/src/services/chatApi.js:4-26` | Send/validate normal Agent `/chat` request. |
| 12 | `frontend/src/services/documentApi.js:26-38` | Multipart upload for one selected file. |
| 13 | `frontend/src/services/documentApi.js:41-50` | Trigger per-document indexing. |
| 14 | `frontend/src/services/documentApi.js:53-69` | Preserved single-document ask client. |
| 15 | `frontend/src/services/documentApi.js:72-85` | Send `document_ids` array and question to `POST /documents/ask`. |
| 16 | `src/api/schemas.py:6-13` | General chat request/response contract. |
| 17 | `src/api/schemas.py:16-38` | Safe upload/index metadata including MD file type. |
| 18 | `src/api/schemas.py:41-46` | Non-empty question plus non-empty multi-document ID list schema. |
| 19 | `src/api/schemas.py:49-57` | Safe retrieved chunk response retains source/document/chunk identity. |
| 20 | `src/api/schemas.py:76-90` | Multi-document answer response contract. |
| 21 | `src/api/app.py:59-74` | `/chat` delegates to shared Agent and returns observable metadata. |
| 22 | `src/api/app.py:77-108` | Upload boundary, bounded read and controlled errors. |
| 23 | `src/api/app.py:131-151` | Per-document index endpoint. |
| 24 | `src/api/app.py:154-169` | Map internal retrieval records to safe HTTP chunk metadata. |
| 25 | `src/api/app.py:172-215` | `POST /documents/ask`: receive ID list, run multi-doc QA and return flags/results. |
| 26 | `src/api/app.py:218-259` | Preserved single-document endpoint. |
| 27 | `src/documents/extractor.py:10,21-28` | Four-type allowlist and extension validation. |
| 28 | `src/documents/extractor.py:31-52` | Shared UTF-8 TXT/MD extraction; MD is plain text. |
| 29 | `src/documents/extractor.py:55-66` | Text-based PDF extraction and scanned-PDF refusal. |
| 30 | `src/documents/extractor.py:69-83` | DOCX paragraph/table extraction. |
| 31 | `src/documents/storage.py:25-36` | Safe filename and validated server-generated document directory. |
| 32 | `src/documents/storage.py:48-88` | Store/extract one upload and preserve original filename metadata. |
| 33 | `src/documents/indexer.py:30-59` | Re-extract, chunk, embed and rebuild one isolated upload index. |
| 34 | `src/documents/indexer.py:63-73` | Mark document indexed with model/dimension/count. |
| 35 | `src/chunker.py:21-38` | Exact overlapping word-window loop and metadata. |
| 36 | `config.py:15-16` | Shared 180-word size and 30-word overlap. |
| 37 | `src/embedder.py:7-31` | Exact all-minilm embedding request and float32 alignment checks. |
| 38 | `src/embedder.py:34-48` | L2 normalization and verification. |
| 39 | `src/vector_store.py:16-22` | Open persistent local Chroma client. |
| 40 | `src/vector_store.py:50-77` | Replace/create explicit cosine collection with no hidden embedder. |
| 41 | `src/vector_store.py:79-102` | Preserve source metadata and upsert explicit aligned vectors. |
| 42 | `src/vector_store.py:133-143` | Embed/normalize one query and check dimension. |
| 43 | `src/vector_store.py:166-199` | Query Chroma with one precomputed normalized vector. |
| 44 | `src/vector_store.py:208-233` | Convert distance to similarity and reconstruct full retrieval records. |
| 45 | `src/documents/retriever.py:65-114` | Validate/load all selected IDs and embed question once. |
| 46 | `src/documents/retriever.py:116-132` | Query every active document and merge candidate records. |
| 47 | `src/documents/retriever.py:137-140` | Global similarity sort, final `candidates[:top_k]`, final ranks. |
| 48 | `src/documents/question_answering.py:31-46` | Multi-document QA wrapper and highest-global-score gate. |
| 49 | `src/documents/question_answering.py:48-64` | Skip/call generation and distinguish answerability. |
| 50 | `config.py:31-32` | `TOP_K = 3`; heuristic threshold `0.30`. |
| 51 | `src/prompt_builder.py:20-33` | Uploaded grounding/citation/refusal system instruction. |
| 52 | `src/prompt_builder.py:47-83` | Filename + document-scoped source labels and final grounded user prompt. |
| 53 | `config.py:20-26` | RAG provider plus Qwen/Gemini 3.5 Flash configuration. |
| 54 | `src/generator.py:17-34` | Exact local Qwen RAG inference. |
| 55 | `src/generator.py:37-55` | Exact Gemini 3.5 Flash SDK inference path. |
| 56 | `src/generator.py:58-70` | Single RAG generation-provider switch point. |
| 57 | `src/agent/agent_loop.py:81-115` | Qwen decision, tool-call parsing and dispatch. |
| 58 | `src/agent/tool_registry.py:65-108` | Tool allowlist, argument validation and Python execution. |
| 59 | `src/agent/agent_loop.py:119-134` | Observation serialization and return to Qwen. |
| 60 | `evaluate_rag.py:161-240` | Threshold-before-generation and deterministic static citation/pass checks. |

### Ten highest-priority oral-defense locations

1. `frontend/src/App.jsx:204-250` — unified UI route: active IDs versus `/chat`.
2. `frontend/src/App.jsx:281-338` — multi-file validation/upload/index loop.
3. `src/api/app.py:172-215` — multi-document HTTP boundary.
4. `src/documents/retriever.py:65-147` — query each active index, merge and global Top-K.
5. `src/embedder.py:7-48` — all-minilm inference and normalization.
6. `src/vector_store.py:50-102,166-233` — Chroma upsert/query and distance conversion.
7. `src/documents/question_answering.py:40-80` — gate/generation/answerability semantics.
8. `src/prompt_builder.py:20-83` — uploaded context labels and grounding rules.
9. `src/generator.py:17-70` — exact Qwen/Gemini calls and switch.
10. `src/agent/agent_loop.py:81-142` — bounded Tool Call → Observation loop.

## Appendix A — Project-specific exam question bank

**Q1. Vì sao không dùng LangChain/LlamaIndex?**

A. Mục tiêu lab là nhìn thấy trực tiếp extraction, chunking, embedding, Chroma query, prompt và Agent loop; framework sẽ che các cơ chế cần bảo vệ.

**Q2. React làm gì và FastAPI làm gì?**

A. React giữ temporary UI state và gọi HTTP. FastAPI validates schemas, maps controlled errors và delegates Agent/document services; model keys, prompt, tools và filesystem đều ở backend.

**Q3. Vì sao frontend không gọi Ollama/Gemini trực tiếp?**

A. Backend phải giữ key, grounded prompt, allowlist, path restrictions và consistent behavior. Frontend chỉ gọi `/chat` hoặc document endpoints.

**Q4. Khi nào request đi `/chat`, khi nào đi `/documents/ask`?**

A. `App.jsx:224-250`: có `documentsAtSendTime.length` thì gửi ID array qua `askDocuments`; empty array thì `sendChatMessage` tới `/chat`.

**Q5. Local và cloud inference khác gì?**

A. all-minilm/Qwen chạy local qua Ollama; Gemini chạy cloud, cần key/network và có latency/cost khác. Chỉ RAG generation được switch.

**Q6. Đổi Qwen sang Gemini có rebuild Chroma không?**

A. Không. Switch ở `generator.py:58-70`, sau retrieval/prompt. Đổi embedding model mới cần rebuild.

**Q7. Gemini model hiện tại là gì?**

A. `config.py:25` mặc định chính xác `gemini-3.5-flash`; inference dùng `generator.py:45-54`.

**Q8. Có hai loại Knowledge Base nào?**

A. Static sample `knowledge/*.txt` dùng regression/Agent `search_knowledge`; teacher uploads nằm dưới `uploads/<id>` và được chọn bằng `activeDocuments`. Hai nguồn không tự trộn.

**Q9. Upload có phải RAG hoặc training ngay không?**

A. Không. Upload chỉ validate/store/extract. `/index` mới chunk/embed/store; không có update model weights.

**Q10. TXT, MD, PDF và DOCX được đọc thế nào?**

A. TXT/MD dùng UTF-8 helper (`extractor.py:31-52`), MD chỉ là plain text; PDF dùng pypdf lines 55-66; DOCX dùng python-docx lines 69-83.

**Q11. Scanned PDF thì sao?**

A. Không có OCR. Nếu pypdf không lấy được text, controlled error tại `extractor.py:62-65` nói PDF có thể scan và cần OCR.

**Q12. Chunk boundary tính ra sao?**

A. Size 180, overlap 30, step 150. Full ranges đầu là `[0,180)`, `[150,330)`; tail dùng `min`.

**Q13. Vì sao overlap?**

A. Giữ ý gần boundary ở hai chunk; trade-off là thêm duplicate text/vectors.

**Q14. Chunk giữ metadata nào?**

A. `chunk_id`, source filename, index, text, start/end word; uploaded chunks thêm `document_id` (`chunker.py:24-34`).

**Q15. all-minilm làm gì?**

A. `ollama.embed` biến text thành semantic vectors; observed dimension là 384. Nó không viết answer.

**Q16. Vì sao query và document dùng cùng embedding model?**

A. Chỉ vectors trong cùng space/dimension mới so cosine có nghĩa; metadata/dimension checks enforce điều đó.

**Q17. Vì sao L2 normalize?**

A. Mỗi vector norm≈1, làm cosine geometry ổn định và được `_validate_embeddings` kiểm tra trước Chroma.

**Q18. Chroma lưu gì?**

A. Chunk ID, text, metadata source/index/ranges/document ID và explicit normalized embedding; collection metadata giữ model/dimension/cosine.

**Q19. `embedding_function=None` để làm gì?**

A. Không cho Chroma tự chọn hidden embedder; project tự cung cấp all-minilm vectors.

**Q20. Distance và similarity liên hệ thế nào?**

A. Trong cosine space, code dùng `similarity = 1 - distance` tại `vector_store.py:215-222`; similarity cao hơn được rank trước.

**Q21. Top-K có đảm bảo relevant không?**

A. Không. Nó luôn lấy K nearest available records; threshold và grounded refusal xử lý answerability riêng.

**Q22. Threshold 0.30 có phải probability không?**

A. Không. Nó là heuristic calibrated trên bảy static tests; uploaded note nói rõ có thể cần recalibration.

**Q23. False positive và false negative là gì?**

A. False positive: weak evidence vẫn qua gate. False negative: valid evidence score thấp nên bị từ chối.

**Q24. Retrieval failure khác generation failure thế nào?**

A. Retrieval failure mất relevant chunk/gate; generation failure xảy ra sau khi evidence đúng đã vào prompt nhưng output vẫn sai/thiếu citation.

**Q25. Grounding được làm thế nào?**

A. Context-only system instruction, retrieved text treated as data, exact valid labels, refusal rule và low-temperature generator.

**Q26. Citation có guaranteed ở production không?**

A. Prompt bắt buộc nhưng model probabilistic; static evaluation validates deterministically, production API chưa có post-generation citation validator.

**Q27. Agent khác fixed RAG thế nào?**

A. RAG luôn retrieve/gate/prompt/generate. Agent để Qwen quyết định direct answer hay tool(s), có observation loop và MAX_STEPS.

**Q28. Tool schema là gì?**

A. JSON-compatible name/description/parameters để model đề xuất call. Python registry vẫn validate; schema không tự chạy code.

**Q29. Ai thực sự chạy tool?**

A. `execute_tool` tại `tool_registry.py:65-108`; Qwen chỉ đề xuất structured name/arguments.

**Q30. Observation là gì?**

A. JSON validation/success/result/error được append với role `tool` tại `agent_loop.py:119-134` cho lượt model tiếp theo.

**Q31. Calculator an toàn hơn `eval` thế nào?**

A. Restricted AST chỉ cho numeric constants và allowlisted operators; names, calls, attributes, power/import bị reject.

**Q32. File tool chống traversal thế nào?**

A. `file_tools.py:54-70` rejects absolute/drive/`..`, resolves path và requires nó vẫn nằm dưới project root.

**Q33. `MAX_STEPS` bảo vệ gì?**

A. `agent_loop.py:81` giới hạn năm model turns; exhaustion trả deterministic message lines 136-142.

**Q34. `search_knowledge` có gọi RAG generator không?**

A. Không. `knowledge_tool.py:28-47` chỉ returns static chunks/scores/gate; Agent Qwen dùng observation để viết final answer.

**Q35. Uploaded-document isolation ở đâu?**

A. Mỗi ID có Chroma directory riêng (`indexer.py:26-27`), collection identity checks, metadata filter và post-query ID checks.

**Q36. Vì sao re-index không duplicate?**

A. `vector_store.py:60-61` deletes same named collection before create/upsert.

**Q37. Ba flag gate/generation/answerable khác nhau?**

A. Lần lượt: score đạt threshold; provider thật sự chạy; output cuối không phải exact refusal.

**Q38. Model switch nằm đâu?**

A. RAG settings `config.py:20-26`, branch `generator.py:58-70`; Agent fixed Qwen ở `config.py:28-29` và `agent_loop.py:84`.

**Q39. Vì sao support nhiều documents?**

A. Teacher có thể chia knowledge thành policy, handbook, FAQ. `activeDocuments` cho phép một câu hỏi tìm evidence trên toàn bộ set mà vẫn giữ citation/source identity.

**Q40. Vì sao mỗi uploaded document có Chroma index riêng?**

A. Isolation, idempotent rebuild và detach-by-ID dễ giải thích. Multi retrieval merge ở request time, không cần database/collection kết hợp mới.

**Q41. Query nhiều documents như thế nào?**

A. `retriever.py:82-124` load mỗi selected collection, embed question once, rồi query từng collection bằng cùng normalized vector.

**Q42. Vì sao lấy candidates per document trước global ranking?**

A. Collections tách vật lý nên phải lấy nearest candidates từng nơi, sau đó mới có một pool để so score và chọn evidence tốt nhất toàn set.

**Q43. GLOBAL Top-K nghĩa là gì?**

A. Sau merge, sort toàn bộ candidates và slice một lần tại `retriever.py:137-140`. Đó là final prompt evidence across all active files.

**Q44. Top-K=3 có nghĩa ba chunks mỗi file không?**

A. Không. Có thể lấy tối đa ba candidate mỗi collection nội bộ, nhưng final context chỉ tối đa ba chunks tổng cộng.

**Q45. Source filename/citation sống qua merge thế nào?**

A. Filename becomes `source` ở `indexer.py:39`, được Chroma lưu/restore, dict merge không bỏ field, prompt dùng `[SOURCE filename | full-chunk-id]`.

**Q46. Detach một document thì sao?**

A. `App.jsx:340-346` removes ID khỏi `activeDocuments`; future ID arrays không chứa nó nên backend không query collection đó. Server index vẫn còn.

**Q47. Static `knowledge/*.txt` có thuộc teacher-uploaded KB không?**

A. Không. Nó là sample/regression corpus và nguồn cho Agent `search_knowledge`; `/documents/ask` chỉ nhận/query uploaded IDs.

**Q48. Vì sao vẫn giữ static sample knowledge?**

A. Để giảng dạy/reproduce Phase 1 pipeline, regression evaluation và chứng minh RAG-as-Agent-tool mà không phụ thuộc teacher uploads.

**Q49. Exam dùng một file và năm file khác nhau thế nào?**

A. Frontend/API body vẫn cùng shape ID array. Một ID query một collection; năm IDs loop năm collections; cả hai vẫn output một global Top-K tối đa ba chunks.

**Q50. Single-document compatibility còn không?**

A. Có: `askDocument` (`documentApi.js:53-69`), API `/documents/{id}/ask` (`app.py:218-259`), single retriever (`retriever.py:17-62`) và CLI `ask_document.py` vẫn hoạt động.

**Q51. Production limitations chính?**

A. Small corpus/local 4B, word chunking, raw cosine + heuristic threshold, no reranker/hybrid/OCR/auth/persistent chat/post-generation citation verification; Gemini requires cloud network/key; Agent output remains probabilistic.

**Q52. Chroma thay manual NumPy search ở đâu?**

A. Runtime retrieval gọi `collection.query` tại `vector_store.py:192-199`. NumPy vẫn cần cho float32 conversion, L2 normalization và vector validation, nhưng không còn `embeddings @ query_vector` làm production search.

**Q53. Làm sao nhìn thấy và verify retrieval?**

A. `inspect_retrieval.py:14-74` in query vector, mọi score và Top-K; `evaluate_rag.py:32-56` kiểm tra expected source trong Top-3 và lines 86-154 audit score ranges/threshold.

**Q54. Vì sao câu unanswerable vẫn nhận được Top-K?**

A. Nearest-neighbor query luôn trả records gần nhất nếu collection có dữ liệu, dù score thấp và facts không tồn tại. Vì vậy QA gate ở `question_answering.py:44-60` mới quyết định có gọi generator hay trả exact refusal.

---

**Audit completion:** Bản refresh này phản ánh current multi-file Knowledge Base, GLOBAL Top-K, TXT/MD/PDF/DOCX support và Gemini default `gemini-3.5-flash`. Không phát hiện runtime/config architecture mismatch thực tế; các limitation ở Q51 là deliberate scope, không phải documentation inconsistency.
