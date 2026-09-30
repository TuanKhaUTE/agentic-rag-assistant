import { useEffect, useRef, useState } from "react";

import { sendChatMessage } from "./services/chatApi.js";
import {
  askDocument,
  DocumentApiError,
  indexDocument,
  uploadDocument,
} from "./services/documentApi.js";


const EXAMPLE_PROMPTS = [
  {
    label: "Study the knowledge base",
    text: "What is RAG? Use the study knowledge base.",
  },
  {
    label: "Run a calculation",
    text: "Calculate 10 * 25.",
  },
  {
    label: "Inspect project files",
    text: "List the files in the knowledge folder.",
  },
  {
    label: "Combine two tools",
    text: "Use the study knowledge base to explain AI agents, then calculate 3 * 2.",
  },
];
const SUPPORTED_EXTENSIONS = ["txt", "pdf", "docx"];
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;


function AppIcon({ name, size = 20 }) {
  const paths = {
    attach: <path d="M12 17.5 18.5 11a4.24 4.24 0 0 0-6-6L5 12.5a5.66 5.66 0 0 0 8 8l7-7a3.54 3.54 0 0 0-5-5L8.5 15a1.41 1.41 0 0 0 2 2l6-6" />,
    chat: <path d="M20 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h9a4 4 0 0 1 4 4Z" />,
    chevron: <path d="m9 18 6-6-6-6" />,
    document: <><path d="M6 3h8l4 4v14H6Z" /><path d="M14 3v5h4M9 13h6M9 17h6" /></>,
    plus: <path d="M12 5v14M5 12h14" />,
    send: <><path d="m22 2-7 20-4-9-9-4Z" /><path d="M22 2 11 13" /></>,
    spark: <><path d="m12 3 1.35 4.15L17.5 8.5l-4.15 1.35L12 14l-1.35-4.15L6.5 8.5l4.15-1.35Z" /><path d="m18.5 14 .75 2.25L21.5 17l-2.25.75L18.5 20l-.75-2.25L15.5 17l2.25-.75Z" /></>,
  };

  return (
    <svg
      aria-hidden="true"
      className={`icon icon-${name}`}
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
    >
      <g stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8">
        {paths[name]}
      </g>
    </svg>
  );
}


function AnswerText({ text }) {
  const parts = text.split(/(\[[^\]\n]+#chunk-\d{3}\])/g);
  return (
    <p className="message-text">
      {parts.map((part, index) =>
        /^\[[^\]\n]+#chunk-\d{3}\]$/.test(part) ? (
          <span className="citation" key={`${part}-${index}`}>{part}</span>
        ) : (
          part
        ),
      )}
    </p>
  );
}


function GeneralMetadata({ steps, toolsUsed }) {
  return (
    <div className="message-metadata" aria-label="Agent execution details">
      <span>Steps <strong>{steps}</strong></span>
      <span>Tools <strong>{toolsUsed.length ? toolsUsed.join(", ") : "None"}</strong></span>
    </div>
  );
}


function DocumentMetadata({ result }) {
  const chunkCount = result.retrieved_chunks.length;
  return (
    <details className="rag-details">
      <summary>
        <span className="rag-summary-copy">
          <strong>View RAG details</strong>
          <span>
            Sources · {chunkCount} {chunkCount === 1 ? "chunk" : "chunks"}
            <span aria-hidden="true"> · </span>
            Similarity · {result.highest_similarity.toFixed(3)}
          </span>
        </span>
        <AppIcon name="chevron" size={18} />
      </summary>
      <div className="rag-details-body">
        <dl className="answer-summary">
          <div><dt>Document</dt><dd>{result.filename}</dd></div>
          <div><dt>Highest similarity</dt><dd>{result.highest_similarity.toFixed(3)}</dd></div>
          <div><dt>Top-K</dt><dd>{result.top_k}</dd></div>
          <div><dt>Heuristic threshold</dt><dd>{result.similarity_threshold.toFixed(2)}</dd></div>
          <div><dt>Retrieval gate</dt><dd>{result.retrieval_gate_passed ? "Passed" : "Failed"}</dd></div>
          <div><dt>Generation called</dt><dd>{result.generation_called ? "Yes" : "No"}</dd></div>
          <div><dt>Answerable</dt><dd>{result.answerable ? "Yes" : "No"}</dd></div>
        </dl>
        <h3>Retrieved source chunks</h3>
        <ol className="source-list">
          {result.retrieved_chunks.map((chunk) => (
            <li key={chunk.chunk_id}>
              <span>{chunk.chunk_id}</span>
              <strong>{chunk.score.toFixed(3)}</strong>
            </li>
          ))}
        </ol>
      </div>
    </details>
  );
}


function validateSelectedFile(file) {
  if (!file) {
    return "Select a TXT, PDF, or DOCX file first.";
  }
  const extension = file.name.toLowerCase().split(".").pop();
  if (!SUPPORTED_EXTENSIONS.includes(extension)) {
    return "Unsupported file type. Select a TXT, PDF, or DOCX file.";
  }
  if (file.size === 0) {
    return "The selected file is empty.";
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return "The selected file is larger than the 10 MB upload limit.";
  }
  return "";
}


function friendlyDocumentError(error, fallbackMessage) {
  return error instanceof DocumentApiError ? error.message : fallbackMessage;
}


function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [documentStatus, setDocumentStatus] = useState("idle");
  const [documentInfo, setDocumentInfo] = useState(null);
  const [documentError, setDocumentError] = useState("");
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);
  const historyEndRef = useRef(null);

  const isPreparingDocument = ["uploading", "indexing"].includes(documentStatus);
  const activeDocument = documentStatus === "ready" ? documentInfo : null;
  const activeModeLabel = activeDocument
    ? `Document: ${activeDocument.filename}`
    : "General Assistant";
  const modeDescription = activeDocument
    ? "Answers are grounded in the attached document."
    : "Ask directly or let the Agent use approved tools.";

  useEffect(() => {
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    historyEndRef.current?.scrollIntoView({
      behavior: reducedMotion ? "auto" : "smooth",
      block: "nearest",
    });
  }, [messages, isLoading]);

  function chooseExample(prompt) {
    setInput(prompt);
    setError("");
    inputRef.current?.focus();
  }

  function startNewConversation() {
    if (isLoading || isPreparingDocument) {
      return;
    }
    setMessages([]);
    setInput("");
    setError("");
    setDocumentError("");
    inputRef.current?.focus();
  }

  async function sendMessage(event) {
    event?.preventDefault();
    const message = input.trim();
    if (!message || isLoading || isPreparingDocument) {
      return;
    }

    const documentAtSendTime = activeDocument;
    const modeLabel = documentAtSendTime
      ? `Document: ${documentAtSendTime.filename}`
      : "General Assistant";
    setMessages((current) => [
      ...current,
      { role: "user", text: message, modeLabel },
    ]);
    setInput("");
    setError("");
    setIsLoading(true);

    try {
      if (documentAtSendTime) {
        const result = await askDocument(documentAtSendTime.document_id, message);
        setMessages((current) => [
          ...current,
          {
            role: "assistant",
            text: result.answer,
            mode: "document",
            modeLabel,
            documentResult: result,
          },
        ]);
      } else {
        const result = await sendChatMessage(message);
        setMessages((current) => [
          ...current,
          {
            role: "assistant",
            text: result.answer,
            mode: "general",
            modeLabel,
            steps: result.steps,
            toolsUsed: result.tools_used,
          },
        ]);
      }
    } catch (caughtError) {
      setError(
        documentAtSendTime
          ? friendlyDocumentError(
              caughtError,
              "The document question failed. Check that the backend is running.",
            )
          : "The assistant could not be reached. Check that the backend is running.",
      );
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  async function prepareDocument(file) {
    const previousDocument = activeDocument;
    setSelectedFile(file);
    setDocumentInfo(null);
    setDocumentError("");
    setError("");
    setDocumentStatus("uploading");
    try {
      const uploaded = await uploadDocument(file);
      setDocumentInfo(uploaded);
      setDocumentStatus("indexing");
      const indexed = await indexDocument(uploaded.document_id);
      setDocumentInfo({ ...uploaded, ...indexed });
      setDocumentStatus("ready");
    } catch (caughtError) {
      setDocumentInfo(previousDocument);
      setSelectedFile(null);
      setDocumentStatus(previousDocument ? "ready" : "error");
      setDocumentError(
        friendlyDocumentError(
          caughtError,
          "The document could not be prepared. Check that the backend is running.",
        ),
      );
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  function selectDocument(event) {
    const file = event.target.files?.[0] ?? null;
    const validationError = validateSelectedFile(file);
    if (validationError) {
      setSelectedFile(null);
      if (!activeDocument) {
        setDocumentInfo(null);
        setDocumentStatus("error");
      }
      setDocumentError(validationError);
      event.target.value = "";
      return;
    }
    prepareDocument(file);
  }

  function detachDocument() {
    if (isLoading || isPreparingDocument) {
      return;
    }
    setSelectedFile(null);
    setDocumentInfo(null);
    setDocumentStatus("idle");
    setDocumentError("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    inputRef.current?.focus();
  }

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Assistant workspace">
        <div className="brand">
          <span className="brand-mark"><AppIcon name="spark" size={22} /></span>
          <span>
            <strong>Knowledge Lab</strong>
            <small>AI study workspace</small>
          </span>
        </div>

        <button
          className="new-conversation-button"
          type="button"
          onClick={startNewConversation}
          disabled={isLoading || isPreparingDocument || messages.length === 0}
        >
          <AppIcon name="plus" size={18} />
          New conversation
        </button>

        <section className="sidebar-section" aria-labelledby="mode-heading">
          <p className="sidebar-heading" id="mode-heading">Current mode</p>
          <div className={`mode-card ${activeDocument ? "document" : "general"}`}>
            <span className="mode-icon">
              <AppIcon name={activeDocument ? "document" : "chat"} size={20} />
            </span>
            <span>
              <strong>{activeDocument ? "Document grounded" : "General Assistant"}</strong>
              <small>{activeDocument ? "Isolated document RAG" : "Agent with approved tools"}</small>
            </span>
          </div>
        </section>

        <section className="sidebar-section document-section" aria-labelledby="document-heading">
          <p className="sidebar-heading" id="document-heading">Active document</p>
          {documentStatus === "uploading" && (
            <div className="document-card preparing" role="status">
              <span className="file-tile"><AppIcon name="document" size={20} /></span>
              <span><strong>{selectedFile?.name}</strong><small>Uploading...</small></span>
            </div>
          )}
          {documentStatus === "indexing" && (
            <div className="document-card preparing" role="status">
              <span className="file-tile"><AppIcon name="document" size={20} /></span>
              <span><strong>{selectedFile?.name}</strong><small>Extracting and indexing...</small></span>
            </div>
          )}
          {activeDocument && (
            <div className="document-card ready">
              <span className="file-tile"><AppIcon name="document" size={20} /></span>
              <span>
                <strong>{activeDocument.filename}</strong>
                <small><span className="status-dot" aria-hidden="true" />Indexed · Ready</small>
              </span>
            </div>
          )}
          {!activeDocument && !isPreparingDocument && (
            <p className="no-document">No document attached. General requests use the Agent.</p>
          )}
        </section>

        <dl className="runtime-list">
          <div><dt>AI runtime</dt><dd>Backend managed</dd></div>
          <div><dt>Knowledge</dt><dd>Local Chroma</dd></div>
        </dl>
      </aside>

      <main className="workspace">
        <header className="workspace-header">
          <div>
            <p className="eyebrow">University AI systems lab</p>
            <h1>AI Knowledge Assistant</h1>
          </div>
          <div className="mode-badge">
            <span className="status-dot" aria-hidden="true" />
            {activeDocument ? "Document grounded" : "General Assistant"}
          </div>
        </header>

        <section className="conversation-card" aria-label="Chat with the AI Knowledge Assistant">
          <div className="message-history" aria-live="polite" aria-busy={isLoading}>
            {messages.length === 0 && (
              <section className="welcome-state">
                <div className="welcome-icon"><AppIcon name="spark" size={28} /></div>
                <p className="eyebrow">{activeDocument ? "Document workspace" : "Knowledge workspace"}</p>
                <h2>{activeDocument ? `Ask about ${activeDocument.filename}` : "What would you like to understand?"}</h2>
                <p>
                  {activeDocument
                    ? "Answers will use only the indexed content of your attached document."
                    : "Explore the study knowledge base, use approved tools, or attach a document for isolated RAG."}
                </p>
                {!activeDocument && !isPreparingDocument && (
                  <div className="example-grid" aria-label="Example prompts">
                    {EXAMPLE_PROMPTS.map((prompt) => (
                      <button
                        className="example-button"
                        key={prompt.text}
                        type="button"
                        onClick={() => chooseExample(prompt.text)}
                        disabled={isLoading}
                      >
                        <span>{prompt.label}</span>
                        <small>{prompt.text}</small>
                        <AppIcon name="chevron" size={16} />
                      </button>
                    ))}
                  </div>
                )}
              </section>
            )}

            {messages.map((message, index) => (
              <article
                className={`message-row ${message.role}`}
                key={`${message.role}-${index}`}
              >
                <div className="message-avatar" aria-hidden="true">
                  {message.role === "user" ? "Y" : <AppIcon name="spark" size={16} />}
                </div>
                <div className="message-content">
                  <div className="message-heading">
                    <strong>{message.role === "user" ? "You" : "Knowledge Assistant"}</strong>
                    <span>{message.modeLabel}</span>
                  </div>
                  <div className="message-bubble">
                    <AnswerText text={message.text} />
                    {message.role === "assistant" && message.mode === "general" && (
                      <GeneralMetadata steps={message.steps} toolsUsed={message.toolsUsed} />
                    )}
                    {message.role === "assistant" && message.mode === "document" && (
                      <DocumentMetadata result={message.documentResult} />
                    )}
                  </div>
                </div>
              </article>
            ))}

            {isLoading && (
              <article className="message-row assistant" role="status">
                <div className="message-avatar" aria-hidden="true"><AppIcon name="spark" size={16} /></div>
                <div className="message-content">
                  <div className="message-heading"><strong>Knowledge Assistant</strong><span>{activeModeLabel}</span></div>
                  <div className="message-bubble thinking">
                    <span>Thinking</span>
                    <span className="thinking-dots" aria-hidden="true"><i /><i /><i /></span>
                  </div>
                </div>
              </article>
            )}
            <div ref={historyEndRef} />
          </div>

          <div className="composer-zone">
            {error && <p className="error-message" role="alert">{error}</p>}
            {documentError && <p className="error-message" role="alert">{documentError}</p>}

            {activeDocument && (
              <div className="active-document-chip">
                <span className="file-tile"><AppIcon name="document" size={18} /></span>
                <span className="chip-copy">
                  <strong>{activeDocument.filename}</strong>
                  <small>{activeDocument.words} words · {activeDocument.chunk_count} chunks · Ready</small>
                </span>
                <button type="button" onClick={detachDocument} disabled={isLoading} aria-label={`Detach ${activeDocument.filename}`}>
                  Detach
                </button>
              </div>
            )}

            {isPreparingDocument && (
              <div className="preparation-banner" role="status">
                <span className="progress-ring" aria-hidden="true" />
                <span>
                  <strong>{documentStatus === "uploading" ? "Uploading document" : "Extracting and indexing"}</strong>
                  <small>{selectedFile?.name}</small>
                </span>
              </div>
            )}

            <form className="composer" onSubmit={sendMessage}>
              <label className="sr-only" htmlFor="message-input">Your message</label>
              <label
                className={`attach-button ${isLoading || isPreparingDocument ? "disabled" : ""}`}
                htmlFor="document-file"
                title="Attach a TXT, PDF, or DOCX document"
              >
                <AppIcon name="attach" size={20} />
                <span className="sr-only">Attach document</span>
                <input
                  id="document-file"
                  ref={fileInputRef}
                  type="file"
                  accept=".txt,.pdf,.docx"
                  onChange={selectDocument}
                  disabled={isLoading || isPreparingDocument}
                />
              </label>
              <textarea
                id="message-input"
                ref={inputRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={activeDocument ? `Ask about ${activeDocument.filename}...` : "Ask a question or request a tool..."}
                rows="2"
                disabled={isLoading || isPreparingDocument}
                aria-describedby="composer-context"
              />
              <button
                className="send-button"
                type="submit"
                disabled={isLoading || isPreparingDocument || !input.trim()}
              >
                <span>{isLoading ? "Thinking" : "Send"}</span>
                <AppIcon name="send" size={18} />
              </button>
            </form>
            <div className="composer-footer" id="composer-context">
              <span><strong>{activeDocument ? "Document grounded" : "General Assistant"}</strong> · {modeDescription}</span>
              <span>Enter to send · Shift+Enter for a new line</span>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}


export default App;
