# Midterm Code Defense Map — AI Homework Lab

> Mục tiêu: tài liệu học để bảo vệ **đúng code hiện tại**. Số dòng được chốt theo working tree tại ngày 2026-09-30. Đây không phải mô tả kiến trúc lý tưởng chung chung.

## Audit snapshot và verdict

- `git status --short`: sạch trước khi audit.
- 10 commit gần nhất:
  - `9066541 feat: add optional Gemini RAG provider`
  - `c3d9b2d feat: migrate RAG vector storage to Chroma`
  - `565ebd7 refactor: unify chat interface and clarify answerability`
  - `a4720f2 feat: add uploaded document chat interface`
  - `507a0d3 feat: add uploaded document question answering`
  - `509c12d feat: add uploaded document indexing`
  - `0dce346 feat: add document upload and text extraction`
  - `e636ea3 feat: add React chat assistant interface`
  - `a7b81e3 feat: add FastAPI chat backend`
  - `4510451 feat: add agent evaluation and final documentation`
- Đã rà toàn repository, loại trừ `.venv`, `node_modules`, `frontend/dist`, dữ liệu Chroma sinh ra, uploads sinh ra và `.git`.
- Verdict: kiến trúc runtime nhất quán với React → FastAPI → Agent hoặc RAG; embedding `all-minilm:latest`; vector DB Chroma local; Qwen local mặc định; optional RAG generation dùng Gemini 3.5 Flash; Agent chỉ dùng Ollama/Qwen.

### Consistency result after final cleanup

- `config.py:25` và các ví dụ README cùng dùng `gemini-3.5-flash`.
- Tài liệu uploaded RAG dùng wording provider-neutral vì generator có thể là Ollama/Qwen hoặc Gemini.
- Đoạn inspection gọi rõ tên `inspect_chunks.py` và phạm vi static knowledge documents.
- Không còn stale active/config/documentation reference tới model Gemini cũ.

Không tìm thấy dead NumPy RAG path, obsolete `.npy`/JSON index, `Path A`/`Path B`, `gemini-3.8`, hay dependency runtime thừa rõ ràng. `numpy` vẫn cần cho vector; `python-multipart` được FastAPI dùng ngầm cho `UploadFile`; `pypdf`, `python-docx`, `chromadb`, `ollama`, `google-genai`, `fastapi`, `uvicorn` đều có đường chạy thực tế. Dòng README nói về NumPy trước đây (`README.md:26-27`) là lịch sử được đánh dấu rõ, không phải mô tả runtime hiện tại.

## 1. Overall Assistant architecture

```text
Browser / React
├─ General mode → POST /chat → bounded Qwen Agent
│  └─ allowlisted tools → static Chroma knowledge / calculator / safe files
└─ Document mode
   ├─ upload → extract TXT/PDF/DOCX
   ├─ index → chunk → all-minilm → normalize → isolated Chroma
   └─ ask → query embedding → Chroma Top-K → gate → prompt
             → configured RAG generator (Ollama/Qwen OR Gemini) → cited answer
```

Hai đường chat cố ý khác nhau. `/chat` là Agent động và **luôn** Qwen/Ollama. `/documents/{id}/ask` là RAG cố định, có thể đổi **chỉ generation** bằng `LLM_PROVIDER`. Static CLI `ask.py` cũng dùng provider RAG đó. Upload không nhập vào static collection và không trở thành Agent memory.

### FILE: `frontend/index.html`

**Purpose:** HTML shell để Vite mount React.

| Lines / exact snippet | Giải thích đơn giản, I/O và flow | Tại sao / nếu bỏ / câu hỏi dễ gặp |
|---|---|---|
| 13-14: `<div id="root"></div>`; `<script type="module" src="/src/main.jsx">` | Trình duyệt tạo điểm mount rồi tải entry JS. Input là page load; output là module React chạy; gọi tiếp `main.jsx`. | Bỏ `root` thì `createRoot` không có DOM node. Hỏi: “React bắt đầu ở đâu?” |

### FILE: `frontend/src/main.jsx`

**Purpose:** React bootstrap.

| Lines / exact snippet | Giải thích đơn giản, I/O và flow | Tại sao / nếu bỏ / câu hỏi dễ gặp |
|---|---|---|
| 1-5: imports React, `ReactDOM`, `App`, CSS | Nạp React runtime, component gốc và style. Caller là browser module loader; gọi tiếp `App`. | Bỏ `App` không có UI; bỏ CSS chỉ mất presentation. Hỏi: “Entry point frontend là dòng nào?” |
| 8-12: `ReactDOM.createRoot(...).render(<React.StrictMode><App />...)` | Gắn component tree vào `#root`. Input DOM element; output UI; `StrictMode` giúp phát hiện lỗi dev. | Bỏ thì trang trống. Hỏi: “Component đầu tiên là gì?” |

### Entry-point files

| File / lines | Purpose và flow | Nếu bỏ / lecturer question |
|---|---|---|
| `run_api.py:1-10` — `uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)` | Import Uvicorn và chạy FastAPI local. Input: chạy script; output: HTTP server; Uvicorn import `app`. | Không có tiện ích khởi động backend. Hỏi: “Server bind ở đâu, port nào?” |
| `run_agent.py:1-13` — `run_agent(goal)` | Lấy goal từ CLI hoặc default, gọi đúng shared Agent loop. Output được loop tự print. | Mất CLI demo, API vẫn chạy. Hỏi: “CLI có duplicate Agent không?” — Không. |
| `ask.py:23-83` | CLI static RAG nhìn thấy retrieval, gate, prompt/provider và answer. Input argv; gọi `search` → `build_grounded_prompt` → `generate_answer`. | Mất demo static RAG, core vẫn còn. Hỏi: “Gate xảy ra trước model ở đâu?” — lines 41-54. |
| `ask_document.py:17-86` | CLI uploaded RAG; gọi `answer_uploaded_document`, rồi in vector, Top-K, gate, context, provider, answer. | Mất demo document RAG, API vẫn còn. Hỏi: “CLI và API có dùng chung QA không?” — Có, line 26. |
| `build_index.py:17-78` | Orchestrator tạo static Chroma: load → chunk → embed → normalize → rebuild. | Không thể reproducibly build static KB. Hỏi: “Static index được build ở đâu?” |

## 2. Frontend / React

### FILE: `frontend/src/App.jsx`

**Purpose:** Một trang chat, giữ state tạm thời, chọn general/document mode, upload-index-ask và trình bày metadata.

| Lines / exact snippet | Giải thích đơn giản, input/output/caller/next | Tại sao / nếu bỏ / likely question |
|---|---|---|
| 1-9: `useEffect/useRef/useState`; service imports | Hook quản lý state/DOM; mọi network call đi qua service. Caller `main.jsx`; gọi `chatApi`/`documentApi`. | Tách UI khỏi endpoint details. Hỏi: “React có gọi Ollama không?” — Không. |
| 12-19: examples, extensions, `MAX_UPLOAD_BYTES` | UI hints và client-side 10 MB/type validation; server vẫn kiểm tra lại. | Bỏ validation client thì UX kém, security server vẫn giữ. Hỏi: “Client validation có đủ an toàn không?” — Không. |
| 22-29 `GeneralMetadata` | Render `steps`, `tools_used`; input response `/chat`, output JSX. | Lecturer thấy Agent hoạt động. |
| 32-70 `DocumentMetadata` | Render filename, similarity, Top-K, threshold, gate, generation, answerable và chunk IDs. | Là bằng chứng RAG; bỏ không làm backend sai nhưng mất transparency. |
| 73-88 `validateSelectedFile(file)` | Reject thiếu file, extension sai, 0 byte, >10 MB. Output chuỗi lỗi hoặc `""`. | Frontend guard; server là authority. |
| 91-93 `friendlyDocumentError` | Chỉ hiện controlled `DocumentApiError`; lỗi khác dùng fallback. | Tránh phơi chi tiết lạ. |
| 96-113 state và derived mode | `messages/input/loading/error` là chat tạm; `documentInfo/status` điều khiển routing. `activeDocument` chỉ có khi `ready`. | Bỏ status dễ hỏi document trước indexing. Hỏi: “Persistent memory?” — Không, state mất khi refresh. |
| 115-117 `useEffect(...scrollIntoView...)` | Sau message/loading change, scroll xuống cuối. Presentation only. |
| 125-184 `sendMessage` | Trim/reject; snapshot active doc; append user; **branch lines 145-170**: document gọi `askDocument`, general gọi `sendChatMessage`; append assistant metadata; finally unlock. | Đây là router thật của UI. Bỏ branch sẽ trộn Agent và document RAG. Hỏi: “Mode được quyết định ở đâu?” |
| 186-191 `handleKeyDown` | Enter gửi, Shift+Enter xuống dòng. Calls `sendMessage`. |
| 193-221 `prepareDocument` | `uploadDocument(file)` rồi `indexDocument(id)`; chỉ set `ready` sau cả hai. On failure phục hồi document cũ. | Đảm bảo ask chỉ dùng index đã sẵn sàng. Hỏi: “Upload có tự là RAG chưa?” — Chưa, indexing là call thứ hai. |
| 223-237 `selectDocument` | Lấy file input, validate, rồi calls `prepareDocument`. |
| 239-251 `detachDocument` | Xóa selection khỏi React state; không xóa server data. | Chuyển lại general mode, không phải delete. |
| 253-412 render tree | Message history, attach state, `<textarea>`, disabled states, send form. Lines 299-307 chọn metadata theo mode; 383-406 form gọi `sendMessage`. | Bỏ disabled có thể double-submit. Hỏi: “Conversation lưu ở đâu?” — `messages` state line 97. |

### FILE: `frontend/src/services/chatApi.js`

**Purpose:** HTTP client duy nhất cho normal Agent chat.

| Lines / snippet | Giải thích và flow | Why/break/question |
|---|---|---|
| 1: `CHAT_URL = "http://127.0.0.1:8000/chat"` | Backend address local. | Sai URL thì general chat fail. |
| 4-11: `fetch(CHAT_URL, {method:"POST", ... body: JSON.stringify({ message })})` | Input string; gửi JSON `/chat`; caller `App:158`. | Đây là exact normal request. Hỏi: “Frontend gửi message ở đâu?” |
| 13-26 response validation | Reject HTTP lỗi hoặc JSON thiếu `answer/steps/tools_used`; output validated object. | Tránh UI dùng shape sai. |

### FILE: `frontend/src/services/documentApi.js`

**Purpose:** HTTP client cho upload, index và ask một document.

| Lines / snippet | Giải thích và flow | Why/break/question |
|---|---|---|
| 1: `API_BASE_URL` | Base backend local. |
| 4-10 `DocumentApiError` | Error có friendly message + HTTP status. Caller UI helper. |
| 13-23 `controlledError` | Đọc FastAPI `{detail}` nếu có, fallback nếu parse fail. | Không lộ stack trace. |
| 26-38 `uploadDocument` | `FormData.append("file", file)` → `POST /documents/upload`; output metadata extracted. | Không tự set multipart header để browser thêm boundary. |
| 41-50 `indexDocument` | POST encoded document ID `/index`; output index metadata. | Tách ingestion khỏi indexing. |
| 53-69 `askDocument` | POST JSON `{question}` vào encoded `/ask`; output grounded result + retrieval metadata. | Exact uploaded-question request. |

### FILE: `frontend/src/styles.css`

**Purpose:** projector-readable, responsive visual layer; không chứa AI logic.

| Lines | Meaning / impact |
|---|---|
| 1-32 | Global typography, colors, box sizing, keyboard focus accessibility. |
| 34-183 | Page/chat layout và user-vs-assistant bubbles. Nếu bỏ, chức năng còn nhưng khó demo. |
| 185-257 | Agent/document metadata và source-score layout. |
| 264-407 | Errors, attachment controls, composer/loading/disabled affordances. |
| 409-435 | Mobile breakpoint stacks controls vertically. |

## 3. FastAPI backend

### FILE: `src/api/schemas.py`

**Purpose:** Pydantic validates HTTP input/output contracts.

| Lines / snippet | Giải thích, I/O và flow | Why/break/question |
|---|---|---|
| 1-3 imports `Annotated`, `Literal`, `BaseModel`, `StringConstraints` | Tạo typed schemas và non-empty trimmed strings. | Bỏ constraints thì empty chat/question lọt vào business logic. |
| 6-13 `ChatRequest/ChatResponse` | Request `{message}`; response `{answer,steps,tools_used}`. FastAPI calls validation automatically. |
| 16-29 metadata/upload models | Chỉ expose safe metadata; status restricted `extracted|indexed`; upload thêm preview. | Không expose filesystem path/full text. |
| 32-38 `DocumentIndexResponse` | Contract sau index: ID/name/status/chunks/model/dimension. |
| 41-42 `DocumentQuestionRequest` | Reject blank document question. |
| 45-53 `RetrievedDocumentChunkResponse` | Safe Top-K metadata: score/similarity/distance/IDs; deliberately no full text. |
| 56-69 `DocumentQuestionResponse` | Expose gate vs final answerability và whether generation ran. | Giúp phân biệt retrieval failure/generator refusal. |

### FILE: `src/api/app.py`

**Purpose:** HTTP boundary; translate validated requests/exceptions into responses; delegates all AI logic.

| Lines / snippet | Giải thích, input/output/calls | Why/break/question |
|---|---|---|
| 1-30 imports | FastAPI/CORS plus shared Agent, document storage/index/QA and controlled errors. | Chứng minh endpoint không duplicate core logic. |
| 35-46 `FastAPI` + `CORSMiddleware` | Cho localhost:5173/127.0.0.1:5173 gọi GET/POST JSON/multipart. | Browser khác origin sẽ bị chặn; CORS không phải auth. |
| 49-51 `GET /health` | Input none; output `{"status":"ok"}`. |
| 54-69 `POST /chat` | Pydantic → `run_agent_with_metadata(request.message)` → `ChatResponse`; unexpected lỗi thành generic 500, server log giữ type. | Shared Agent behavior; không lộ stack trace. |
| 72-103 `POST /documents/upload` | Validate extension, bounded read `MAX+1`, `create_document`; map 415/400/413/422; always close UploadFile. | Prevent arbitrary/oversize uploads and resource leak. |
| 106-123 `GET /documents/{id}` | Calls `get_document_metadata`; 404 controlled; returns safe schema. |
| 126-146 `POST .../index` | Calls `index_uploaded_document`; maps not-found/extract/index errors. | Endpoint only orchestrates, indexer owns pipeline. |
| 149-202 `POST .../ask` | Calls `answer_uploaded_document`; 404 invalid ID, 409 not indexed, 503 bad generator config; maps Top-K to safe response. | Keeps text/context internal. Hỏi: “generation status returned where?” lines 188-201. |

## 4. Configuration

### FILE: `config.py`

**Purpose:** Single source of paths, model IDs, chunk/retrieval settings and provider env.

| Lines / exact snippet | Giải thích và consumers | Why/break/question |
|---|---|---|
| 5-13 paths/limits | Root, static knowledge/storage/Chroma, uploaded collection/storage, 10 MB and 500-char preview. Used by loaders/storage/API. | Centralizes safe absolute roots. |
| 15 `CHUNK_SIZE_WORDS = 180` | Mỗi full chunk tối đa 180 words. Used static/upload indexers. |
| 16 `CHUNK_OVERLAP_WORDS = 30` | Start step 150; giữ context tại boundary. |
| 18 `EMBEDDING_MODEL = "all-minilm:latest"` | Cùng model cho chunks và query. Nếu đổi phải rebuild mọi index. |
| 20 `LLM_PROVIDER = os.getenv(..."ollama")...` | Read once at Python import/start; chooses RAG generator only. |
| 21-24 Qwen/Ollama settings | Default model `qwen3:4b-instruct`; RAG local model can be env override. |
| 25-26 Gemini settings | Model/key read from env; default model is `gemini-3.5-flash`; key default blank. |
| 28-29 `GENERATION_MODEL = DEFAULT_QWEN_MODEL` | Agent remains fixed Ollama/Qwen regardless `LLM_PROVIDER`. |
| 31 `TOP_K = 3` | Default three retrieved chunks. |
| 32 `MIN_SIMILARITY = 0.30` | Educational pre-generation heuristic. |

## 5. Document upload and text extraction

### FILE: `src/documents/extractor.py`

**Purpose:** Convert supported file formats into usable plain text; no OCR.

| Lines / snippet | Giải thích, I/O/caller/next | Why/break/question |
|---|---|---|
| 1-7 imports `Path`, DOCX, pypdf errors | Format-specific parsers and controlled exceptions. |
| 10 `SUPPORTED_EXTENSIONS` | Allowlist `.txt/.pdf/.docx`. | Không dựa MIME/user path. |
| 13-18 exception classes | Business errors mapped to HTTP 415/422. |
| 21-28 `supported_extension` | Input filename; lowercase suffix; output allowed extension or reject. Called upload/storage and dispatcher. |
| 31-39 `extract_txt_text` | UTF-8 read, reject encoding/empty. Output raw text. |
| 42-53 `extract_pdf_text` | `PdfReader`; join page text; empty means likely scan/OCR needed. | Bỏ page loop không có PDF content. Hỏi: “Scanned PDF?” — rejected, no OCR. |
| 56-70 `extract_docx_text` | `Document(file)`; collect paragraphs and table cells; reject corrupt/empty. |
| 73-82 `extract_text` | Dispatcher by extension → exact extractor. Caller storage/indexer/inspection. | Một public entry keeps behavior consistent. |

### FILE: `src/documents/storage.py`

**Purpose:** Safely persist original upload + JSON metadata under server-generated ID.

| Lines / snippet | Giải thích, I/O/caller/next | Why/break/question |
|---|---|---|
| 1-7 imports | JSON, path sanitation, UUID; config limits; extractor reuse. |
| 10 `DOCUMENT_ID_PATTERN = ...32 hex` | Only UUID-hex IDs accepted. | Prevent user-controlled path components. |
| 13-22 errors | Empty/too-large/not-found become controlled API statuses. |
| 25-30 `_safe_metadata_filename` | Normalize slashes, keep basename only, reject `.`/`..`; original name metadata only. |
| 33-36 `_document_directory` | Validate ID before `UPLOADS_DIR / id`. | First path traversal defense. |
| 39-45 `_discard_incomplete_document` | Cleanup only files in newly created document folder if extraction fails. | Avoid half-created records. |
| 48-88 `create_document` | Input bytes+filename; validate size/name/type; UUID; write `original.ext`; extract; write metadata; output metadata+preview. Called API upload; next index may re-extract. | Upload is ingestion, not RAG/training. |
| 91-101 `get_document_metadata` | Validated ID → parse `metadata.json`; corrupt/missing becomes not-found. |
| 104-120 directory/original getters | Recheck metadata/file type/existence; output safe server path for internal callers only. |
| 123-142 `update_document_metadata` | Merge index fields; write temp then atomic-ish `replace`; cleanup temp. | Prevent partial metadata on normal replace. |

### FILE: `src/documents/__init__.py`

**Purpose:** Lines 1-32 re-export the approved extractor/storage public API. Callers (`app.py`, CLIs) import stable names without reaching private helpers. Removing exports breaks those imports; likely question: “Package facade dùng để làm gì?”

## 6. Chunking

### FILE: `src/chunker.py`

**Purpose:** Same transparent word-window algorithm for static and uploaded documents.

| Lines / snippet | Giải thích, I/O/caller/next | Why/break/question |
|---|---|---|
| 1-12 `chunk_document` validation | Input document dict, size, overlap; reject impossible windows. Called both indexers. |
| 14-19 setup; `step = size - overlap` | Split whitespace words; prefix is filename or uploaded ID; with 180/30, step=150. |
| 21-34 while/build chunk | End-exclusive window; ID `#chunk-001`; store source/index/text/start/end and optional document_id. Output ordered chunk list. | Metadata links retrieval back to source. |
| 36-38 break/advance | Stop exactly at last word; otherwise advance 150. Prevent infinite/duplicate tail. |
| 43-53 `chunk_documents` | Batch wrapper extends chunks in input order for static KB. |

## 7. Embedding

### FILE: `src/embedder.py`

**Purpose:** Explicit Ollama embedding and NumPy L2 normalization shared by indexing/query.

| Lines / exact snippet | Giải thích, I/O/caller/next | Why/break/question |
|---|---|---|
| 1-4 imports | NumPy handles matrices; official Ollama client calls local embedding model; config chooses model. |
| 7-14 `embed_texts`; `ollama.embed(model=EMBEDDING_MODEL, input=texts)` | Input ordered non-empty strings; Ollama returns one vector each. Called indexers and `embed_query`. | Exact inference call. It is embedding inference, not LLM training. |
| 15-25 count/dimension checks | Ensure response rows align and every vector dimension equal (all-minilm is 384 here). |
| 27-31 `np.asarray(...dtype=np.float32)` | Output 2D float32 matrix; smaller/stable storage and Chroma input. |
| 34-48 `normalize_embeddings` | Compute L2 norm per row, reject zeros, divide, verify ≈1. Output normalized float32. | Makes cosine geometry explicit and validates vectors before DB. |

## 8. Chroma indexing

### FILE: `src/vector_store.py` (client/write/load blocks)

**Purpose:** Shared local Chroma adapter; application supplies its own explicit embeddings.

| Lines / exact snippet | Giải thích, I/O/caller/next | Why/break/question |
|---|---|---|
| 1-13 imports/`COSINE_DISTANCE_METRIC` | Chroma persistence + NumPy validation; database space is cosine. |
| 16-22 `chromadb.PersistentClient(path=..., Settings(...False))` | Create directory and persistent local client; telemetry disabled. | Exact DB creation; remove persistence loses local index access. |
| 25-29 `_collection_names` | Normalize Chroma version return objects/names into set. Used rebuild. |
| 32-47 `_validate_embeddings` | Requires chunks, 2D, row alignment, finite values, norm≈1. | Protects `chunks[i] ↔ vector[i]`. |
| 50-61 rebuild/delete | Validate; open client; delete same named collection first. | Idempotent rebuild, avoids duplicates/stale records. |
| 63-77 metadata + `create_collection(...)` | Store embedding model/dimension/cosine; `embedding_function=None` prevents hidden embedding model. | Makes first-principles mechanism visible. |
| 79-92 IDs/documents/metadatas | Convert each chunk into parallel Chroma fields; optional document ID retained. |
| 94-99 `collection.upsert(ids=..., documents=..., metadatas=..., embeddings=...)` | Exact vector DB write. Input aligned records/vectors; output persisted collection. |
| 100-102 count check | Verify exactly one DB record per chunk. |
| 105-126 `load_chroma_collection` | Load without embedding function and validate model, cosine config, non-empty count. | Prevent querying incompatible/stale index. |
| 129-130 `load_static_collection` | Binds generic loader to static path/name. |

### FILE: `build_index.py`

**Purpose:** Build static `knowledge/*.txt` collection reproducibly.

| Lines / snippet | Giải thích và flow | Why/break/question |
|---|---|---|
| 1-14 imports | Pull one config and shared load/chunk/embed/store functions. |
| 20-24 load | `load_documents(KNOWLEDGE_DIR)`; fail empty. |
| 26-34 chunk | `chunk_documents(...180,30)`; fail empty. |
| 36-48 embed/alignment | List texts → `embed_texts`; assert rows equal chunks. |
| 50-56 normalize/inspect | Normalize and print norm range. |
| 58-74 rebuild | `rebuild_chroma_collection(CHROMA_DIR, STATIC_CHROMA_COLLECTION, ...)`; marks `index_kind=static_knowledge`; prints IDs. |

### FILE: `src/documents/indexer.py`

**Purpose:** Build/load exactly one uploaded document's isolated Chroma database.

| Lines / snippet | Giải thích, I/O/caller/next | Why/break/question |
|---|---|---|
| 1-19 imports | Reuses extractor, chunker, embedder, generic Chroma adapter and safe storage. |
| 22-23 `DocumentIndexError` | Controlled index-domain error. |
| 26-27 `uploaded_chroma_directory` | Validated document directory + `/chroma`; physical per-document isolation. |
| 30-44 extract/document/chunk | Metadata + original path → re-extract text → dict with `document_id`/ID prefix → shared chunker; reject no chunks. |
| 46-59 embed/rebuild | `embed_texts` → normalize → generic Chroma rebuild with uploaded collection metadata and document ID. |
| 63-73 metadata update | Status indexed, count/model/dimension/database/collection. Output metadata to API. |
| 76-93 `load_uploaded_collection` | Load metadata + its Chroma; verify collection `document_id` and count. | Detect wrong/corrupt index before retrieval. |

## 9. RAG retrieval / Top-K

### FILE: `src/vector_store.py` (query blocks)

| Lines / exact snippet | Giải thích, I/O/caller/next | Why/break/question |
|---|---|---|
| 133-143 `embed_query` | Validate question; `embed_texts([query])[0]`; check stored dimension; reshape/normalize; output one vector. | Documents/query must share vector space and dimension. |
| 146-171 `query_chroma_collection` setup | Validate `top_k`; read dimension; embed query; construct explicit `query_embeddings`, bounded `n_results`, returned fields, optional `where`; call `collection.query(**query_arguments)`. | Exact Chroma query; Chroma does not embed secretly. |
| 173-178 aligned response check | Extract first query's ids/docs/metadata/distances and require equal lengths. |
| 181-205 result mapping | Rank in Chroma order; reject nonfinite distance; compute `similarity = 1.0 - distance`; preserve `score` compatibility and metadata. | Top-K means highest similarity/lowest distance among this collection. |
| 208-231 `get_collection_records` | Inspection-only `collection.get`; optionally reconstruct float32 vectors. Not used for retrieval. |
| 234-239 `search` | Static convenience wrapper: load static collection, query it, return `(query_vector, results)`. |

### FILE: `src/documents/retriever.py`

**Purpose:** Uploaded-document retrieval with model/dimension/document isolation checks.

| Lines / snippet | Giải thích, I/O/caller/next | Why/break/question |
|---|---|---|
| 1-6 imports | Same configured embedding/Top-K; uses uploaded loader and generic Chroma query. |
| 9-10 `DocumentNotIndexedError` | Allows API to return HTTP 409. |
| 13-27 status/model checks | Input ID+question+K; require `status=indexed` and same `EMBEDDING_MODEL`. | Prevent query against absent/incompatible vectors. |
| 29-35 load/dimension check | Metadata and Chroma collection must agree. |
| 37-41 K validation | Reject bool/non-int/nonpositive; cap at collection size. |
| 42-48 query with `where={"document_id": document_id}` | Query only the loaded document DB plus metadata filter. | Two isolation layers: separate path + filter. |
| 49-58 result identity/return | Require results and verify every `document_id`; return metadata, query vector, Top-K. | Stops cross-document leakage. |

Top-K **luôn trả về các record gần nhất nếu collection có record**, kể cả tất cả đều không liên quan. Vì vậy Top-K là ranking, không phải proof of answerability; relevance gate ở section 10 xử lý bước tiếp theo.

## 10. Relevance gate and answerability

### FILE: `src/documents/question_answering.py`

**Purpose:** Uploaded-document RAG orchestrator; separates retrieval gate, generation execution, and final answerability.

| Lines / exact snippet | Giải thích, I/O/caller/next | Why/break/question |
|---|---|---|
| 1-9 imports | Threshold/K, uploaded retriever, provider-neutral generator, prompt/refusal. |
| 12-16 `UPLOADED_THRESHOLD_NOTE` | Explicitly says 0.30 is static-corpus educational heuristic and uploads may need recalibration. |
| 19-27 retrieve/highest/gate | Input ID/question/K; retrieve; top score; `retrieval_gate_passed = highest_similarity >= MIN_SIMILARITY`. | Gate happens **before** model call. |
| 29-32 defaults | Empty prompt/context and `generation_called=False` make skipped generation observable. |
| 33-41 branch | Passed: build labelled prompt then `generate_answer`, set true. Failed: exact refusal, no model call. | Prevents model guessing on weak retrieval. |
| 43-45 `answerable = answer != INSUFFICIENT_INFORMATION_MESSAGE` | Even after gate, grounded generator can refuse. Gate pass ≠ answer found. |
| 47-61 return dict | Combines retrieval, flags, prompt/context and answer for CLI/API. `sufficient_context` is compatibility alias of gate. |

### Static CLI gate (`ask.py`)

- Lines 41-42: compute `highest_similarity` and threshold boolean.
- Lines 51-54: failed gate prints exact refusal and returns before prompt/generation.
- Lines 56-77: only a passed gate builds context and calls generator.

### Ba khái niệm phải nói đúng

- `retrieval_gate_passed`: score Top-1 đạt heuristic 0.30.
- `generation_called`: code thật sự đã gọi provider; false khi gate fail.
- `answerable`: final answer khác exact refusal; generator vẫn có quyền từ chối sau gate.
- False positive: irrelevant Top-1 vẫn ≥0.30 → model được gọi không cần thiết/có nguy cơ trả sai.
- False negative: relevant Top-1 <0.30 → hệ thống từ chối dù tài liệu có câu trả lời.

## 11. Prompt construction

### FILE: `src/prompt_builder.py`

**Purpose:** Turn raw retrieved chunks into untrusted labelled data and strict grounded messages.

| Lines / exact snippet | Giải thích, I/O/caller/next | Why/break/question |
|---|---|---|
| 1-3 `INSUFFICIENT_INFORMATION_MESSAGE` | Một exact refusal dùng chung gate và generator check. |
| 6-17 `SYSTEM_INSTRUCTION` | Static KB rules: context-only, no outside knowledge, documents are data/not instructions, exact chunk citations each factual paragraph, exact refusal. | Grounding + prompt-injection defense. |
| 20-33 uploaded system instruction | Same rules but requires full document-scoped chunk ID. | Prevent ambiguous `[chunk-001]` across uploads. |
| 36-45 `_build_labelled_prompt` validation | Input question/chunks/system flag; reject empty/missing. |
| 47-65 label loop | Validate `chunk_id/text`; uploaded also source; sanitize CR/LF/`]` in filename; context exact format `[SOURCE label]\ntext`. | Retrieved text remains data; labels trace evidence. |
| 61-62 uploaded label/citations | Display source + full chunk ID; permitted answer citation is only `[document_id#chunk-NNN]`. |
| 67-83 context/user message/return | Join blocks; list valid uploaded citations; add question and citation command; output `(context, system, user)`. Caller passes last two to generator. |
| 86-95 `build_grounded_prompt` | Static wrapper chooses static instruction. |
| 98-108 `build_uploaded_document_prompt` | Uploaded wrapper chooses uploaded instruction/labels. |

“Citations are enforced” ở đây nghĩa là **prompt requires them**; static evaluation (`evaluate_rag.py:157-217`) validates citations after generation. Production API does not run a deterministic post-generation citation validator, nên model vẫn probabilistic.

## 12. LLM generation

### FILE: `src/generator.py`

**Purpose:** Provider-neutral RAG generation boundary. Retrieval is intentionally absent.

| Lines / snippet | Giải thích, I/O/caller/next | Why/break/question |
|---|---|---|
| 1-10 imports | Official Ollama and Google GenAI clients plus provider/model/key config. |
| 13-14 `GenerationConfigurationError` | Controlled config failure; API maps to safe status where applicable. |
| 17-34 `_generate_with_ollama` | Validate model; call `ollama.chat` with system/user, temperature .1, context 4096, max 512; output stripped text. | Conservative grounded QA. |
| 37-55 `_generate_with_gemini` | Require key/model; scoped `genai.Client`; `generate_content` with same system/user intent/temp/max; output stripped text. | Key stays backend. |
| 58-74 `generate_answer` | Validate prompts; branch exact `LLM_PROVIDER`; no fallback; reject unknown/empty output; return answer. | One switch point means retrieval/index unchanged. |

## 13. Ollama/Qwen

### RAG role

- Config: `config.py:21-24` selects `qwen3:4b-instruct` by default for Ollama RAG generation.
- Inference: `src/generator.py:22-33` calls local `ollama.chat` after retrieval/gate/prompt.
- Input: two messages containing rules and retrieved context/question. Output: cited natural-language answer.
- It does **not** search Chroma, extract documents, calculate similarity, or validate upload paths.

### Agent role

- Config: `config.py:28-29` fixes `GENERATION_MODEL = DEFAULT_QWEN_MODEL`.
- Inference/tool decision: `src/agent/agent_loop.py:83-92` calls `ollama.chat(... tools=OLLAMA_TOOL_SCHEMAS)`.
- Qwen may output final text or structured `tool_calls`; Python, not Qwen, validates and executes.
- `LLM_PROVIDER=gemini` does not affect this Agent path.

## 14. Gemini

- Responsibility: optional **RAG answer generation only** after the same all-minilm/Chroma retrieval and same grounded prompt.
- Runs: Google cloud through `google-genai`; unlike default Ollama, it requires network/API key.
- Config read: `config.py:20,25-26`; current default is `gemini-3.5-flash`.
- Client/inference: `src/generator.py:45-54`; `genai.Client(api_key=...)` then `client.models.generate_content(...)`.
- It does not control Agent tools. `/chat` still calls `run_agent_with_metadata` and Qwen.
- Exact switch point: `generate_answer`, lines 63-66. Everything before that call is identical for Qwen/Gemini.

## 15. Static knowledge RAG

### FILE: `src/document_loader.py`

**Purpose:** Load only repository `knowledge/*.txt` into deterministic document records.

| Lines / snippet | Giải thích, I/O/caller/next | Why/break/question |
|---|---|---|
| 4-9 `_extract_title` | First non-empty line stripped of leading `#`, else filename stem. |
| 12-15 `load_documents` guard | Input `Path`; require directory. Called `build_index`/inspection. |
| 17-31 sorted glob/read/append | Sorted filename order; UTF-8; skip empty; output path/source/title/text/word_count. | Stable order and source traceability. |
| 33 return | List goes next to `chunk_documents`. |

### Static build and query

1. `build_index.py:20-65`: knowledge files → shared chunks → all-minilm → normalized matrix → `static_knowledge` Chroma.
2. `src/vector_store.py:129-130`: load static collection.
3. `src/vector_store.py:234-239`: static `search(query,K)`.
4. Standalone: `ask.py:27` → gate → prompt → provider.
5. Agent tool: `knowledge_tool.py:30` calls the same `search`; it returns retrieval only.

## 16. Uploaded-document RAG

### Lifecycle

```text
POST upload → uploads/<uuid>/original.ext + metadata(status=extracted)
POST index  → re-extract → chunks → vectors → uploads/<uuid>/chroma
             → metadata(status=indexed, model, dimension, count)
POST ask    → load only that directory/collection + document_id filter
             → query vector → Top-K → gate → prompt → configured RAG provider
```

### Why separate from static KB

- Static KB is curated homework knowledge; uploads are user-specific, dynamic and potentially untrusted.
- Separate persistence makes deletion/rebuild/isolation understandable and avoids polluting static evaluation.
- Same extractor/chunker/embedder/vector adapter proves one reusable RAG mechanism; only source and storage scope differ.
- Re-indexing deletes/recreates the deterministic per-document collection (`vector_store.py:60-61`), so it is idempotent rather than append/duplicate.

### Core package facade

`src/documents/__init__.py:1-32` exposes only public storage/extraction functions/errors. `src/api/app.py` imports this facade; index/retriever/QA modules remain explicit because they represent later pipeline phases.

## 17. Agent loop

### FILE: `src/agent/agent_loop.py`

**Purpose:** One visible, bounded Model → Tool Call → Python → Observation → Model loop.

| Lines / exact snippet | Giải thích, I/O/caller/next | Why/break/question |
|---|---|---|
| 1-9 imports | JSON for visible observations; dataclass result; Ollama; fixed Agent model; registry schemas/executor. |
| 12 `MAX_STEPS = 5` | Hard bound on model turns, not number of individual calls inside one turn. | Prevent unbounded loop/cost. |
| 14-30 `AGENT_SYSTEM_INSTRUCTION` | Tells Qwen when tools are mandatory, multi-tool completion, grounding/citations and outputs-as-data. | Guides probabilistic selection; Python still enforces allowlist. |
| 32 `MAX_STEPS_MESSAGE` | Deterministic safe termination output. |
| 35-39 `AgentRunResult` | Immutable result: answer, steps, ordered tools. API uses metadata; CLI can keep string API. |
| 42-45 `_message_value` | Supports SDK objects or dicts. Input message/name; output field/default. |
| 48-52 `_tool_call_parts` | Pull structured `function.name` and `arguments` from Qwen tool call. |
| 55-60 `_assistant_message_for_history` | Convert model message to dict before appending; rejects unsupported type. | Tool-call message must be preserved in conversation. |
| 63-75 `_run_agent` setup | Validate goal; create system+user messages; empty tools-used list. Called CLI/API/eval wrappers. |
| 81-92 bounded loop and `chat_callable` | For each step call Qwen with full messages and `OLLAMA_TOOL_SCHEMAS`; low temp/settings. | Exact point model sees tool definitions. Injectable callable supports safety evaluation. |
| 93-103 final-answer branch | Read `response.message.tool_calls`; if none, content is final answer; return result with current step/tools. | Agent need not call a tool for every goal. |
| 105-115 tool-call branch | Append assistant tool-call message; parse each call; record name; `execute_tool(name,args)`. | Model proposes, Python executes only approved function. |
| 116-127 observation serialization | Print validation and JSON result/error. Output is structured observation, not direct final answer. |
| 128-134 `messages.append({"role":"tool",...})` | Feed observation back on next loop iteration. | Without it Qwen cannot use tool result. |
| 136-142 exhaustion | After 5 turns return deterministic stop. |
| 145-151 `run_agent` | Compatibility wrapper returns only answer for CLI/evaluation. |
| 154-160 `run_agent_with_metadata` | Same loop returns dataclass for `/chat`; no duplicate Agent. |

### FILE: `src/agent/__init__.py`

**Purpose:** Lines 1-17 re-export Agent result, MAX_STEPS, registry/executor and two runner APIs. `src/api/app.py:6` imports the metadata runner here. Removing exports breaks public imports, not the loop algorithm.

## 18. Tool registry

### FILE: `src/agent/tool_registry.py`

**Purpose:** Convert simple schemas for Ollama, validate model arguments, and dispatch only allowlisted Python functions.

| Lines / snippet | Giải thích, I/O/caller/next | Why/break/question |
|---|---|---|
| 1-3 imports | Four normal tool functions + schemas. |
| 6-11 `TOOL_REGISTRY` | Explicit name→callable allowlist. Unknown model name cannot resolve arbitrary Python. |
| 13-16 `TOOL_SCHEMAS_BY_NAME` | Fast schema lookup during validation. |
| 18-21 `OLLAMA_TOOL_SCHEMAS` | Wrap each function schema as Ollama `{"type":"function","function":...}`. Passed to Qwen. |
| 24-35 `_matches_json_type` | Strict small JSON-type checker; notably bool is not accepted as integer/number. |
| 38-62 `_validate_arguments` | Check required, reject extras, verify type/minimum. Input tool+dict; output none or exception. | Model output is untrusted. |
| 65-81 `execute_tool` early branches | Unknown tool or non-object args return structured rejected observation. |
| 83-91 schema failure | Catch validation errors; do not call function. |
| 93-101 approved execution failure | `TOOL_REGISTRY[tool_name](**arguments)`; function error becomes `success:false`, validation still allowed. | Distinguishes bad permission/schema from runtime failure. |
| 103-108 success | Return tool/result as JSON-ready observation. |

## 19. Individual tools

### FILE: `src/tools/knowledge_tool.py`

**Purpose:** Expose static Chroma retrieval to Agent, with no generation.

| Lines / snippet | Meaning / flow | Why/break/question |
|---|---|---|
| 1-2 imports | Same `TOP_K`, threshold and static `search`; no generator import. |
| 5-25 `SEARCH_KNOWLEDGE_SCHEMA` | Qwen-visible name/description/query/top_k/minimum/no extra args. |
| 28-47 `search_knowledge` | Calls `search(query,K)`; projects rank/score/ID/source/text; reports highest and heuristic sufficient context. | RAG retrieval becomes one tool; Agent Qwen writes final answer/citation from observation. |

### FILE: `src/tools/calculator_tool.py`

**Purpose:** Safe basic arithmetic without `eval`/`exec`.

| Lines / snippet | Meaning / flow | Why/break/question |
|---|---|---|
| 1-2 imports `ast`, `operator` | Parse syntax tree and map only permitted operations. |
| 5-22 schema | One required string `expression`; no extras. |
| 25-36 operator maps/limits | Only +,-,*,/ and unary signs; max 200 chars/50 AST nodes. |
| 39-58 `_calculate_node` | Recursively allow numeric constants, allowed binary/unary nodes; catch divide-by-zero; reject names/calls/attributes/power/import. |
| 61-76 `calculator` | Validate string/length; `ast.parse(..., mode="eval")`; complexity guard; evaluate only `tree.body`. | `eval` would execute arbitrary code; AST walker cannot. |

### FILE: `src/tools/file_tools.py`

**Purpose:** Read-only, project-confined listing/UTF-8 reading.

| Lines / snippet | Meaning / flow | Why/break/question |
|---|---|---|
| 6-36 two schemas | Qwen sees relative directory/path only; `read_file` requires path. |
| 39-51 root/limits/extensions | Resolved trusted root; max 1,000,000 bytes; text allowlist. |
| 54-70 `_resolve_project_path` | Reject empty, absolute/drive, literal `..`; resolve and require `relative_to(_PROJECT_ROOT)`. | Layered traversal/symlink escape defense. |
| 73-85 `list_files` | Resolve; require directory; return only direct child file paths relative to project. No recursive exposure/write. |
| 88-108 `read_file` | Resolve; require file + allowed extension + size; UTF-8 read or controlled error. No writes. |

### FILE: `src/tools/__init__.py`

**Purpose:** Lines 1-16 assemble exactly four schema objects in `TOOL_SCHEMAS`; lines 18-24 export only four functions plus schemas. Tool registry imports this facade. Removing one schema makes it invisible to Qwen even if function remains; removing export breaks registry import.

## 20. Evaluation / verification

### FILE: `evaluate_rag.py`

**Purpose:** Seven-question retrieval/generation evaluation with score calibration and deterministic citation checks.

| Lines / snippet | Meaning / verification | Lecturer angle |
|---|---|---|
| 22-29 path/loader | Loads `tests/test_questions.json`, requires non-empty list. |
| 32-56 `collect_retrieval_results` | Runs static `search` for every test; records Top-1, Top-3 sources and expected-source presence. |
| 86-118 score ranges | Separates answerable/unanswerable observed Top-1 ranges and reports whether gap is clean. |
| 121-154 threshold calibration | Checks 0.30 lies inside observed gap; explicitly calls it corpus/test heuristic. |
| 157-158 `extract_citations` | Regex accepts only static `.txt#chunk-NNN` format. |
| 161-188 final runs | Applies threshold before provider generation; false cases use exact refusal. |
| 190-217 citation/pass logic | Citations must exist, belong to retrieved IDs, and answerable tests cite expected source. |
| 223-228 unanswerable pass | Requires gate fail, generation not called and exact refusal. |
| 275-302 summary | Reads Chroma records and prints docs/chunks/model/dimension/provider/pass counts. |
| 305-317 main | Runs collection → calibration → final pipeline → tables → summary. |

### FILE: `tests/test_questions.json`

- Lines 2-25: tests A-D are answerable, including semantic paraphrase D; expected source recorded.
- Lines 26-43: tests E-G are intentionally unanswerable (tuition 2027, FIFA 2030, current Melbourne weather) with `expected_source: null`.
- Đây là evaluation data, không phải pytest unit tests và không được đưa vào knowledge base.

### FILE: `evaluate_agent.py`

**Purpose:** Execute six real Agent scenarios and four controlled safety checks.

| Lines / snippet | Meaning / verification |
|---|---|
| 19-60 `SCENARIOS` | Knowledge, arithmetic, listing, reading, multi-tool, direct answer; exact expected tool sequence. |
| 63-74 citation checker | Final Agent citations must be subset of IDs actually observed from search tool. |
| 77-106 scenario pass logic | Requires exact tools, final answer, successful events; case-specific result checks. |
| 109-152 `run_scenario` | Wrap real Ollama call, patch executor only to record events, capture transcript, call shared `run_agent`. |
| 172-248 safety checks | Unknown tool, invalid power operator, traversal, repeated unknown calls to prove MAX_STEPS. |
| 266-297 main | Runs scenarios/checks, prints summary, exits nonzero on failure. |

### Inspection/support scripts (reviewed, not core behavior)

| File | What it verifies |
|---|---|
| `inspect_chunks.py:6-56` | Documents, word counts, chunk IDs/ranges/overlap/full text and totals. |
| `inspect_embeddings.py:7-55` | Chroma count/dimension/dtype, first vectors and norms, one vector per chunk. |
| `inspect_retrieval.py:10-74` | Query vector, **all** Chroma scores, Top-K full chunks and 1-distance explanation. |
| `inspect_tools.py:12-65` | All schemas/tools, unsafe AST rejection and path traversal rejection. |
| `inspect_document.py:7-28` | Extract local TXT/PDF/DOCX and show first 500 chars. |
| `inspect_uploaded_index.py:10-56` | Per-upload Chroma metadata, first three vectors/chunks, row count and norm≈1. |

### Dependency and ignored-artifact audit

- `requirements.txt:2-10`: all nine packages have a legitimate code/runtime role; no clearly unused dependency found.
- `.gitignore:1-13`: ignores venv, secrets `.env`, caches, frontend dependencies/build, upload data and `storage/chroma`; keeps `.gitkeep` markers.
- No active code reads/writes `chunks.json` or `embeddings.npy`; no active manual `embeddings @ query_vector` path.
- NumPy remains necessary in `src/embedder.py`, `src/vector_store.py`, build/inspection CLIs.

## 21. End-to-end request flows

### Flow 1 — Uploaded PDF → extraction → chunking → embedding → Chroma

1. React `prepareDocument` (`App.jsx:193-206`) calls `uploadDocument` then `indexDocument`.
2. `uploadDocument` (`documentApi.js:26-38`) sends multipart to `upload_document` (`app.py:77`).
3. API reads at most `MAX_UPLOAD_BYTES + 1` (`app.py:82`) and calls `create_document` (`app.py:83`).
4. `create_document` (`storage.py:48-88`) creates UUID folder, writes `original.pdf`, then calls `extract_text`.
5. `extract_text` (`extractor.py:73-82`) dispatches `.pdf` to `extract_pdf_text`; `PdfReader` and page extraction are lines 42-53.
6. React then POSTs `/index`; `index_document` (`app.py:130-146`) calls `index_uploaded_document`.
7. Indexer reopens safe original and re-extracts (`indexer.py:32-34`), builds a document record with ID prefix (`36-41`), then calls shared `chunk_document` (`42`).
8. Chunk texts go to `embed_texts` and `normalize_embeddings` (`indexer.py:46-49`); exact Ollama embedding is `embedder.py:14`.
9. `rebuild_chroma_collection` (`indexer.py:50-59`) opens `uploads/<id>/chroma`, creates cosine collection and upserts explicit chunks/vectors (`vector_store.py:50-102`).
10. Metadata becomes `indexed` with count/model/dimension (`indexer.py:63-73`). Đây là indexing, không phải training.

### Flow 2 — Document question → Qwen → cited answer

1. `App.sendMessage` snapshots active doc and calls `askDocument` (`App.jsx:132-146`).
2. HTTP POST reaches `ask_indexed_document` (`app.py:149-158`) then `answer_uploaded_document`.
3. QA calls `retrieve_uploaded_document` (`question_answering.py:24`). Retriever verifies status/model/dimension and loads exactly that per-ID collection.
4. `query_chroma_collection` embeds question with all-minilm (`vector_store.py:162`), sends explicit vector and `where document_id` to Chroma (`retriever.py:42-48`; `vector_store.py:164-171`).
5. Chroma distance becomes similarity `1-distance` and ordered Top-K records (`vector_store.py:181-205`).
6. QA takes Top-1 and checks `>= 0.30` (`question_answering.py:25-27`). Failure skips all generation.
7. Pass calls `build_uploaded_document_prompt` (`34-37`), which creates `[SOURCE filename | document_id#chunk-NNN]` blocks.
8. `generate_answer` branches to Ollama by default (`generator.py:63-64`); `_generate_with_ollama` calls Qwen (`22-33`).
9. Text returns at `question_answering.py:38`; `answerable` compares it with exact refusal (`45`).
10. API returns answer plus retrieval flags/IDs (`app.py:174-202`); React renders answer and evidence (`App.jsx:147-156`, `32-70`).

### Flow 3 — Same document question with Gemini provider

Steps 1-7 above are **identical**: same upload, all-minilm query vector, same Chroma, Top-K, gate and prompt. Only at `generate_answer` (`generator.py:63-66`) does `LLM_PROVIDER == "gemini"` select `_generate_with_gemini` instead of Ollama. That function creates `genai.Client` and calls `generate_content` (`45-54`) using the configured `gemini-3.5-flash` default. The Agent is not involved and does not switch to Gemini.

### Flow 4 — General “Calculate 10 * 25”

1. No active document → `App.jsx:157-169` calls `sendChatMessage` → `/chat`.
2. `app.py:54-69` calls `run_agent_with_metadata`, which calls shared `_run_agent`.
3. Qwen receives messages + schemas at `agent_loop.py:83-92`; calculator instruction forces a tool result.
4. Qwen returns structured tool call; code reads `tool_calls` (`93-94`) and parses name/arguments (`108-110`).
5. `execute_tool` (`115`) checks name against `TOOL_REGISTRY`, validates schema, then calls `calculator(expression="10 * 25")` (`tool_registry.py:65-108`).
6. Calculator parses restricted AST and returns `250` (`calculator_tool.py:61-76`).
7. Result is JSON observation (`agent_loop.py:119-126`), appended with role `tool` (`128-134`).
8. Next loop sends all messages to Qwen again; with result 250, Qwen returns final content branch (`96-103`).
9. API returns answer/steps/tools; React renders `Tools used: calculator`.

### Flow 5 — General request requiring static knowledge

1. `/chat` enters same Qwen loop.
2. Qwen sees `search_knowledge` schema and selects it.
3. Registry executes `knowledge_tool.search_knowledge` (`knowledge_tool.py:28-47`).
4. Tool calls static `vector_store.search` (`line 30`), which loads `storage/chroma/static_knowledge` and performs all-minilm + Chroma Top-K.
5. Tool returns chunk texts/IDs/scores/highest/gate only—no RAG generator call.
6. Structured observation returns to Qwen. Agent instruction requires exact observed chunk ID citation (`agent_loop.py:24-29`).
7. Qwen generates final Agent answer from observation. This is “RAG as a tool”: retrieval is a tool, while Agent Qwen performs the final generation.

### Terminology check for this repository

| Term | Exact meaning here | Code evidence |
|---|---|---|
| Training | Updating model weights from examples. **This project performs none.** | No optimizer/backprop/fine-tuning code or training dependency. |
| Inference | Running an already-trained model. | `ollama.embed`, `ollama.chat`, Gemini `generate_content`. |
| Chunking | Splitting extracted text into overlapping 180-word windows. | `chunker.py:1-40`. |
| Embedding | Mapping each chunk/question to a numeric semantic vector (384 dimensions observed for all-minilm). | `embedder.py:7-31`. |
| Indexing | Pairing chunks/metadata with normalized vectors and storing them in Chroma. | `vector_store.py:50-102`. |
| Retrieval | Embed question and ask Chroma for nearest Top-K chunks. | `vector_store.py:133-205`. |
| Generation | Use Qwen/Gemini to write natural-language answer from labelled context. | `generator.py:17-74`. |

Nếu giảng viên nói không chính xác “train the document”, hãy lịch sự nói: trong project này document **không retrain Qwen/Gemini**. Code thật là `extract → chunk → embed → index`; lúc hỏi là `embed query → retrieve → generate`.

### Model / database responsibility table

| Component | Responsibility | Runs where | Input → output | RAG / Agent role |
|---|---|---|---|---|
| `all-minilm:latest` | Semantic embedding only | Local Ollama | text(s) → 384-number vector(s) | Index chunks and embed queries; no answer generation, no Agent decisions. |
| Chroma | Persist records/vectors and cosine nearest-neighbor query | Local filesystem (`storage/chroma` or per-upload `/chroma`) | explicit vectors+metadata → stored collection; query vector → distances/records | Vector database for static and uploaded RAG; no LLM. |
| `qwen3:4b-instruct` | Natural language generation and native Agent tool selection | Local Ollama | messages/context/schemas → answer or structured tool calls | Default RAG generator; only Agent controller. |
| `gemini-3.5-flash` | Optional cloud grounded answer generator | Google cloud through GenAI SDK | same RAG prompt → answer text | RAG generation only; never controls Agent. |

### Exact 50 exam locations

| # | File : current line(s) | Short exact snippet | One-sentence purpose |
|---:|---|---|---|
| 1 | `frontend/src/services/chatApi.js:4-11` | `fetch(CHAT_URL, ... body: JSON.stringify({ message }))` | React sends normal chat JSON to `/chat`. |
| 2 | `frontend/src/services/documentApi.js:53-60` | ``fetch(`${API_BASE_URL}/documents/${encodeURIComponent(documentId)}/ask`, ...)`` | React sends one uploaded-document question. |
| 3 | `src/api/app.py:54-57` | `@app.post("/chat"...); result = run_agent_with_metadata(...)` | FastAPI receives normal Agent chat. |
| 4 | `src/api/app.py:72-83` | `@app.post("/documents/upload"...); metadata = create_document(...)` | FastAPI receives, bounds and stores one upload. |
| 5 | `src/api/app.py:126-132` | `@app.post("/documents/{document_id}/index"...); index_uploaded_document(...)` | FastAPI starts per-document indexing. |
| 6 | `src/api/app.py:149-158` | `@app.post("/documents/{document_id}/ask"...); answer_uploaded_document(...)` | FastAPI receives uploaded-document Q&A. |
| 7 | `src/documents/extractor.py:31-39,42-53,56-70` | `extract_txt_text`; `extract_pdf_text`; `extract_docx_text` | Format-specific TXT/PDF/DOCX text extraction. |
| 8 | `src/chunker.py:21-38` | `while start_word < len(words): ... start_word += step` | Builds overlapping word-window chunks. |
| 9 | `config.py:15` | `CHUNK_SIZE_WORDS = 180` | Configures maximum full chunk size. |
| 10 | `config.py:16` | `CHUNK_OVERLAP_WORDS = 30` | Configures shared boundary overlap. |
| 11 | `config.py:18` | `EMBEDDING_MODEL = "all-minilm:latest"` | Selects document/query embedding model. |
| 12 | `src/embedder.py:14` | `ollama.embed(model=EMBEDDING_MODEL, input=texts)` | Exact local embedding inference call. |
| 13 | `src/embedder.py:40-47` | `normalized = (matrix / vector_lengths)...` | L2-normalizes and verifies vectors. |
| 14 | `src/vector_store.py:16-22` | `chromadb.PersistentClient(path=..., settings=...)` | Opens local persistent Chroma. |
| 15 | `src/vector_store.py:72-77` | `client.create_collection(... space: cosine ... embedding_function=None)` | Creates explicit cosine collection without hidden embedder. |
| 16 | `src/vector_store.py:94-99` | `collection.upsert(... embeddings=embeddings.tolist())` | Exact Chroma write of aligned chunks/vectors. |
| 17 | `src/vector_store.py:164-171` | `response = collection.query(**query_arguments)` | Exact Chroma Top-K query with explicit query vector. |
| 18 | `config.py:31` | `TOP_K = 3` | Default retrieval count. |
| 19 | `src/vector_store.py:187-194` | `similarity = 1.0 - distance; "score": similarity` | Converts Chroma cosine distance to project similarity. |
| 20 | `src/documents/retriever.py:42-52` | `where={"document_id": document_id}` + identity check | Enforces uploaded-document isolation in addition to separate DB path. |
| 21 | `config.py:32` | `MIN_SIMILARITY = 0.30` | Configures educational relevance gate. |
| 22 | `src/documents/question_answering.py:25-27` | `retrieval_gate_passed = highest_similarity >= MIN_SIMILARITY` | Decides if retrieved evidence may reach an LLM. |
| 23 | `src/documents/question_answering.py:32-39` | `generation_called = False ... generate_answer(...) ... True` | Records whether provider was actually invoked. |
| 24 | `src/documents/question_answering.py:43-45` | `answerable = answer != INSUFFICIENT_INFORMATION_MESSAGE` | Distinguishes gate pass from final supported answer. |
| 25 | `src/prompt_builder.py:49-67` | `context_blocks.append(f"[SOURCE {label}]\n{chunk['text']}")` | Turns Top-K chunks into labelled context. |
| 26 | `src/prompt_builder.py:75-83` | `user_message = (... CONTEXT ... QUESTION ...)` | Constructs grounded user prompt and returns both messages. |
| 27 | `config.py:20` | `LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")...` | Reads RAG generation-provider selection at startup. |
| 28 | `config.py:21` | `DEFAULT_QWEN_MODEL = "qwen3:4b-instruct"` | Configures default Qwen model for RAG and Agent. |
| 29 | `src/generator.py:22-33` | `response = ollama.chat(model=OLLAMA_GENERATION_MODEL, ...)` | Exact local RAG generation call. |
| 30 | `config.py:25` | `GEMINI_MODEL = ... "gemini-3.5-flash"` | Configures the verified optional Gemini 3.5 Flash RAG generator. |
| 31 | `config.py:26` | `GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()` | Reads cloud secret only from environment. |
| 32 | `src/generator.py:45` | `with genai.Client(api_key=GEMINI_API_KEY) as client:` | Exact Gemini SDK client creation. |
| 33 | `src/generator.py:46-54` | `client.models.generate_content(model=GEMINI_MODEL, ...)` | Exact Gemini inference call. |
| 34 | `src/documents/question_answering.py:34-39` | `answer = generate_answer(system_message, user_message)` | Generated text returns into uploaded-document QA. |
| 35 | `src/prompt_builder.py:26-31`; `evaluate_rag.py:190-217` | `Never invent a citation...`; `valid_citations = ...` | Prompt requires citations; evaluation deterministically validates static answers. |
| 36 | `build_index.py:20-65` | `load_documents → chunk_documents → embed_texts → rebuild_chroma_collection` | Builds static knowledge base index. |
| 37 | `src/vector_store.py:234-239` | `return query_chroma_collection(load_static_collection(), query, top_k)` | Queries static knowledge through shared Chroma path. |
| 38 | `src/tools/knowledge_tool.py:28-47` | `def search_knowledge... _, top_results = search(...)` | Exposes retrieval-only static RAG to Agent. |
| 39 | `src/tools/__init__.py:11-16` and tool files | `TOOL_SCHEMAS = [SEARCH_..., CALCULATOR_..., ...]` | Assembles four Qwen-visible function schemas. |
| 40 | `src/agent/agent_loop.py:83-92` | `chat_callable(... tools=OLLAMA_TOOL_SCHEMAS, ...)` | Sends tool schemas and conversation to Qwen. |
| 41 | `src/agent/agent_loop.py:93-94` | `model_message = response.message; tool_calls = ...` | Reads Qwen's structured tool-call response. |
| 42 | `src/agent/tool_registry.py:65-91` | allowlist check + `_validate_arguments(...)` | Validates tool name and arguments before execution. |
| 43 | `src/agent/tool_registry.py:93-95` | `result = TOOL_REGISTRY[tool_name](**arguments)` | Python executes only the approved callable. |
| 44 | `src/agent/agent_loop.py:119-126` | `observation = json.dumps(tool_result, ...)` | Converts tool result/error into an observation. |
| 45 | `src/agent/agent_loop.py:128-134` | `messages.append({"role":"tool", ... "content": observation})` | Sends observation back to Qwen on next loop iteration. |
| 46 | `src/agent/agent_loop.py:81,136-142` | `for step in range(1, MAX_STEPS + 1)` | Enforces five-step bound and safe stop. |
| 47 | `src/tools/calculator_tool.py:25-58,68-76` | allowed AST operator maps + `_calculate_node(tree.body)` | Rejects code/names/calls and safely evaluates only basic arithmetic. |
| 48 | `src/tools/file_tools.py:54-70` | reject absolute/`..`; `resolved_path.relative_to(_PROJECT_ROOT)` | Prevents path traversal outside project. |
| 49 | `tests/test_questions.json:2-25` | four `"type": "answerable"` cases | Supplies relevant/source/paraphrase RAG evaluation cases. |
| 50 | `tests/test_questions.json:26-43` | three `"type": "unanswerable"` cases | Verifies threshold refusal on absent knowledge. |

### Core runtime files reviewed

33 core runtime/UI files were reviewed: six root config/entry pipelines; six frontend runtime/UI files; six shared RAG modules; two API modules; six document-package modules; three Agent modules; and four tool modules. Empty package markers `src/__init__.py` and `src/api/__init__.py` were also checked; they contain no runtime logic. Evaluation/inspection scripts, tests, README, final summary, dependencies and ignore rules were reviewed separately.

### Ten highest-priority oral-defense locations

1. `frontend/src/App.jsx:125-170` — UI chooses Agent vs uploaded-document RAG.
2. `src/api/app.py:54-69` — `/chat` delegates to the shared Agent.
3. `src/api/app.py:149-202` — document Q&A boundary and safe response metadata.
4. `src/embedder.py:7-48` — exact embedding inference and normalization.
5. `src/vector_store.py:50-102` — Chroma collection rebuild/upsert.
6. `src/vector_store.py:146-205` — exact Top-K query and distance→similarity.
7. `src/documents/question_answering.py:19-60` — gate, generation-called and answerable semantics.
8. `src/prompt_builder.py:36-108` — labelled context and grounding rules.
9. `src/generator.py:17-74` — provider switch; exact Ollama/Gemini calls.
10. `src/agent/agent_loop.py:81-142` — bounded tool-call/observation loop.

### Project-specific exam question bank

**Q1. Vì sao không dùng LangChain/LlamaIndex?**
A. Mục tiêu lab là nhìn thấy cơ chế. Loader/chunker/embedder/Chroma query/prompt/Agent loop đều hiện trực tiếp trong `src/`; không bị framework che.

**Q2. React làm gì và FastAPI làm gì?**
A. React giữ UI state tạm và gọi HTTP (`App.jsx`). FastAPI validates schemas, maps errors và delegates Agent/document services (`app.py`); model/tool/path secrets ở backend.

**Q3. Vì sao React không gọi Ollama/Gemini trực tiếp?**
A. Backend phải giữ key, prompt, allowlist, filesystem restrictions và consistent behavior. Frontend chỉ gọi `/chat` hoặc document endpoints.

**Q4. Khi nào request đi `/chat`, khi nào đi `/documents/{id}/ask`?**
A. `App.jsx:145-170`: active document ready thì `askDocument`; nếu không thì `sendChatMessage`.

**Q5. Local và cloud inference khác gì trong project?**
A. Ollama/all-minilm/Qwen chạy local, riêng tư/không phí API nhưng giới hạn máy/model 4B. Gemini gọi cloud, cần key/network, có thể khác chất lượng/latency/cost. Only RAG generation switchable.

**Q6. Đổi Qwen sang Gemini có rebuild Chroma không?**
A. Không. `generate_answer` switch xảy ra sau retrieval; embedding vẫn all-minilm. Đổi embedding model thì phải re-embed/re-index.

**Q7. Current Gemini model có đúng kiến trúc đích không?**
A. Có. `config.py:25` mặc định là `gemini-3.5-flash`, và README dùng cùng model ID; Gemini vẫn chỉ là optional RAG generator, không điều khiển Agent.

**Q8. Knowledge Base nằm đâu?**
A. Static source ở `knowledge/*.txt`, indexed vào `storage/chroma/static_knowledge`; upload có original/metadata/Chroma riêng dưới `uploads/<document_id>/`.

**Q9. Upload có phải RAG hay training ngay không?**
A. Upload chỉ validate/store/extract. Sau `/index` mới chunk/embed/store; không bước nào update Qwen/Gemini weights.

**Q10. PDF được đọc như thế nào, scanned PDF thì sao?**
A. `PdfReader` và `page.extract_text()` (`extractor.py:42-53`). Không text thì controlled error nói cần OCR; OCR không được implement.

**Q11. Chunk boundary tính ra sao?**
A. Size 180, overlap 30, step 150. Ranges đầu `[0,180)`, tiếp `[150,330)`; cuối dùng `min`.

**Q12. Tại sao cần overlap?**
A. Giữ câu/ý gần boundary xuất hiện ở hai chunk, giảm mất context; đổi lại tăng số vector và duplicate text.

**Q13. Một chunk giữ metadata nào?**
A. ID, source, index, text, `start_word`, `end_word`; upload thêm `document_id`. Chroma stores text + metadata + explicit vector.

**Q14. all-minilm làm gì?**
A. `ollama.embed` biến chunk hoặc question thành vector semantic 384 chiều quan sát trong index; không viết answer.

**Q15. Vì sao document và query phải cùng embedding model?**
A. Similarity chỉ có nghĩa trong cùng vector space/dimension. Loader validates metadata/model and `embed_query` checks dimension.

**Q16. Vì sao L2 normalize?**
A. Đưa norm mỗi vector về gần 1, làm cosine geometry rõ/ổn định; `_validate_embeddings` không cho vector chưa normalize vào DB.

**Q17. Chroma thay NumPy search ở đâu?**
A. `collection.query` ở `vector_store.py:171`; application gửi explicit query vector. NumPy chỉ còn conversion/normalization/inspection.

**Q18. Chroma lưu gì?**
A. ID, chunk document text, metadata, normalized embedding; collection metadata còn model/dimension/cosine/index kind/document ID.

**Q19. `embedding_function=None` để làm gì?**
A. Không cho Chroma tự chọn/ẩn embedder; project dùng all-minilm qua Ollama và supply vector explicit.

**Q20. Top-K là gì?**
A. K records gần query nhất trong collection; default K=3. Nó luôn trả cái “gần nhất” chứ không đảm bảo relevant.

**Q21. Distance và similarity liên hệ thế nào?**
A. Chroma cosine returns distance; code line 188 dùng `similarity = 1 - distance`. Cao hơn được xem là semantic closer.

**Q22. Làm sao verify retrieval?**
A. `inspect_retrieval.py` in query vector/all scores/Top-K; `evaluate_rag.py` kiểm tra expected source trong Top-3 và score ranges.

**Q23. Threshold 0.30 có phải confidence probability không?**
A. Không. Nó là heuristic calibrated trên 7 câu static; upload note cảnh báo có thể cần recalibrate.

**Q24. Tại sao unanswerable vẫn có Top-3?**
A. Nearest-neighbor ranking luôn chọn records gần nhất dù đều irrelevant. Gate cố giảm việc gửi evidence yếu cho LLM.

**Q25. Retrieval failure và generation failure khác gì?**
A. Retrieval failure: relevant chunk không được lấy/gate fail. Generation failure: evidence đúng đã vào prompt nhưng model trả sai, thiếu citation hoặc refuse sai.

**Q26. Grounding thực hiện thế nào?**
A. System prompt says context-only/no outside facts, retrieved text is data not instructions, exact citation/refusal; context contains labels.

**Q27. Citation có được guarantee ở production không?**
A. Prompt yêu cầu; static evaluation regex-checks. API path không có deterministic validator sau generation, nên không guarantee tuyệt đối.

**Q28. Agent khác fixed RAG thế nào?**
A. RAG luôn retrieve trước generation. Agent để Qwen quyết định direct answer/tool(s), có loop nhiều bước và có thể gọi RAG retrieval như tool.

**Q29. Tool schema là gì?**
A. JSON-compatible name/description/parameters cho model biết cách đề xuất call; Python registry vẫn validate, schema không tự execute.

**Q30. Ai thực sự chạy tool?**
A. `execute_tool` lookup allowlist và gọi Python function. Qwen chỉ trả structured name/arguments.

**Q31. Observation là gì?**
A. JSON chứa validation/success/result/error, appended as `role: tool`; vòng sau Qwen đọc để quyết định tiếp/final.

**Q32. Calculator an toàn hơn `eval` thế nào?**
A. AST walker chỉ nhận numeric constants và allowlisted operators; names, calls, attributes, power/import đều bị reject; còn giới hạn length/node count.

**Q33. File tool chống traversal thế nào?**
A. Reject absolute/drive/`..`, resolve path và require it remains relative to resolved project root; read còn kiểm tra extension/size/UTF-8.

**Q34. `MAX_STEPS` bảo vệ gì?**
A. `range(1,6)` chặn loop vô hạn. Hết 5 model turns trả deterministic stop message.

**Q35. `search_knowledge` có gọi Qwen không?**
A. Không. Nó chỉ calls static `search`, returns chunks/scores/gate. Agent Qwen dùng observation để answer.

**Q36. Uploaded-document isolation ở đâu?**
A. Mỗi ID có physical Chroma directory; loader checks collection metadata; query thêm `where document_id`; results rechecked.

**Q37. Tại sao re-index không duplicate chunks?**
A. Rebuild deletes same collection before recreate/upsert (`vector_store.py:60-61`).

**Q38. `retrieval_gate_passed`, `generation_called`, `answerable` khác nhau?**
A. Lần lượt: score đạt threshold; provider thật sự chạy; output cuối không phải exact refusal.

**Q39. Model đổi ở đâu?**
A. RAG provider/model env ở `config.py:20-26`, branch `generator.py:63-66`; Agent model cố định ở `config.py:29` và used `agent_loop.py:84`.

**Q40. Điểm hạn chế production chính?**
A. Corpus nhỏ, word chunking, raw cosine+heuristic, no reranker/hybrid/verification/auth/persistent chat, local 4B probabilistic, per-document single index, Gemini cloud dependency.

---

**Audit completion:** tài liệu này phản ánh repository sau final consistency cleanup. Chỉ default Gemini model và documentation wording được đồng bộ; không có thay đổi kiến trúc hoặc logic Agent/RAG/tool.
