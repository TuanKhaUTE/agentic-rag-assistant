import { useEffect, useRef, useState } from "react";

import {
  askKnowledgeBase,
  DocumentApiError,
  getAssignedKnowledgeBase,
} from "./services/documentApi.js";


function AppIcon({ name, size = 20 }) {
  const paths = {
    attach: <path d="M12 17.5 18.5 11a4.24 4.24 0 0 0-6-6L5 12.5a5.66 5.66 0 0 0 8 8l7-7a3.54 3.54 0 0 0-5-5L8.5 15a1.41 1.41 0 0 0 2 2l6-6" />,
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


function DocumentMetadata({ result }) {
  const chunkCount = result.retrieved_chunks.length;
  const documentLabel = result.document_count > 1
    ? `${result.document_count} documents`
    : result.filename ?? result.filenames?.[0] ?? "1 document";
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
          <div><dt>Knowledge base</dt><dd>{documentLabel}</dd></div>
          <div><dt>Highest similarity</dt><dd>{result.highest_similarity.toFixed(3)}</dd></div>
          <div><dt>Top-K</dt><dd>{result.top_k}</dd></div>
          <div><dt>Candidate pool</dt><dd>{result.candidate_count}</dd></div>
          <div><dt>Query vector</dt><dd>{result.query_embedding_dimension} dimensions</dd></div>
          <div><dt>Heuristic threshold</dt><dd>{result.similarity_threshold.toFixed(2)}</dd></div>
          <div><dt>Retrieval gate</dt><dd>{result.retrieval_gate_passed ? "Passed" : "Failed"}</dd></div>
          <div><dt>Generation called</dt><dd>{result.generation_called ? "Yes" : "No"}</dd></div>
          <div><dt>Answerable</dt><dd>{result.answerable ? "Yes" : "No"}</dd></div>
        </dl>
        <h3>Retrieved source chunks</h3>
        <ol className="source-list">
          {result.retrieved_chunks.map((chunk) => (
            <li key={chunk.chunk_id}>
              <span><b>{chunk.source}</b><small>{chunk.chunk_id}</small></span>
              <strong>{chunk.score.toFixed(3)}</strong>
            </li>
          ))}
        </ol>
      </div>
    </details>
  );
}


function friendlyDocumentError(error, fallbackMessage) {
  return error instanceof DocumentApiError ? error.message : fallbackMessage;
}


function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeDocuments, setActiveDocuments] = useState([]);
  const [isKnowledgeLoading, setIsKnowledgeLoading] = useState(true);
  const [documentError, setDocumentError] = useState("");
  const inputRef = useRef(null);
  const historyEndRef = useRef(null);

  const isPreparingDocument = isKnowledgeLoading;
  const hasActiveDocuments = activeDocuments.length > 0;
  const totalChunks = activeDocuments.reduce(
    (total, document) => total + document.chunk_count,
    0,
  );
  const activeModeLabel = hasActiveDocuments
    ? "Lumen Institute — Student Handbook"
    : "Knowledge Base unavailable";
  const modeDescription = hasActiveDocuments
    ? "Answers use only the assigned handbook."
    : "Run the assigned knowledge-base initializer before asking questions.";

  useEffect(() => {
    let current = true;

    async function loadKnowledgeBase() {
      try {
        const knowledgeBase = await getAssignedKnowledgeBase();
        if (current) {
          setActiveDocuments([knowledgeBase]);
          setDocumentError("");
        }
      } catch (caughtError) {
        if (current) {
          setDocumentError(
            friendlyDocumentError(
              caughtError,
              "The assigned knowledge base is unavailable. Run the initializer first.",
            ),
          );
        }
      } finally {
        if (current) {
          setIsKnowledgeLoading(false);
        }
      }
    }

    loadKnowledgeBase();
    return () => {
      current = false;
    };
  }, []);

  useEffect(() => {
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    historyEndRef.current?.scrollIntoView({
      behavior: reducedMotion ? "auto" : "smooth",
      block: "nearest",
    });
  }, [messages, isLoading]);

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

    if (!hasActiveDocuments) {
      setError("The assigned Knowledge Base is not ready. Run scripts/initialize_kb.py.");
      inputRef.current?.focus();
      return;
    }

    const documentsAtSendTime = [...activeDocuments];
    const modeLabel = "Lumen Institute — Student Handbook";
    setMessages((current) => [
      ...current,
      { role: "user", text: message, modeLabel },
    ]);
    setInput("");
    setError("");
    setIsLoading(true);

    try {
      const result = await askKnowledgeBase(
        documentsAtSendTime.map((document) => document.document_id),
        message,
      );
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          text: result.answer,
          modeLabel,
          documentResult: result,
        },
      ]);
    } catch (caughtError) {
      setError(
        friendlyDocumentError(
          caughtError,
          "The knowledge-base question failed. Check that the backend is running.",
        ),
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

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Assistant workspace">
        <div className="brand">
          <span className="brand-mark"><AppIcon name="spark" size={22} /></span>
          <span>
            <strong>Lumen Institute</strong>
            <small>Student Handbook RAG</small>
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
          <div className="mode-card document">
            <span className="mode-icon">
              <AppIcon name="document" size={20} />
            </span>
            <span>
              <strong>{hasActiveDocuments ? "Assigned handbook" : "Knowledge Base unavailable"}</strong>
              <small>{hasActiveDocuments ? `${totalChunks} chunks · Ready` : "Run the initializer"}</small>
            </span>
          </div>
        </section>

        <section className="sidebar-section document-section" aria-labelledby="document-heading">
          <p className="sidebar-heading" id="document-heading">Assigned Knowledge Base</p>
          <div className="sidebar-document-list">
            {activeDocuments.map((document) => (
              <div className="document-card ready" key={document.document_id}>
                <span className="file-tile"><AppIcon name="document" size={20} /></span>
                <span>
                  <strong>{document.title ?? document.filename}</strong>
                  <small><span className="status-dot" aria-hidden="true" />{document.chunk_count} chunks · Ready</small>
                </span>
              </div>
            ))}
          </div>
          {hasActiveDocuments && (
            <p className="knowledge-summary">
              Assigned handbook · {totalChunks} chunks · Ready
            </p>
          )}
          {!hasActiveDocuments && !isPreparingDocument && (
            <p className="no-document">Run scripts/initialize_kb.py, then refresh this page.</p>
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
            <h1>Lumen Institute — Student Handbook</h1>
          </div>
          <div className="mode-badge">
            <span className="status-dot" aria-hidden="true" />
            {hasActiveDocuments ? "Assigned Knowledge Base · Ready" : "Knowledge Base unavailable"}
          </div>
        </header>

        <section className="conversation-card" aria-label="Chat with the document-grounded RAG assistant">
          <div className="message-history" aria-live="polite" aria-busy={isLoading}>
            {messages.length === 0 && (
              <section className="welcome-state">
                <div className="welcome-icon"><AppIcon name="spark" size={28} /></div>
                <p className="eyebrow">Assigned midterm knowledge base</p>
                <h2>{hasActiveDocuments ? "Ask the Lumen Student Handbook" : "Initialize the handbook to begin"}</h2>
                <p>
                  {hasActiveDocuments
                    ? "Answers use only the assigned handbook and include retrieved chunk citations."
                    : "Run the initialization command in the README, then refresh this page."}
                </p>
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
                    {message.role === "assistant" && (
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

            {hasActiveDocuments && (
              <div className="active-knowledge-panel">
                <div className="active-knowledge-heading">
                  <span>
                    <strong>Assigned Knowledge Base</strong>
                    <small>Lumen Institute — Student Handbook · {totalChunks} chunks · Ready</small>
                  </span>
                </div>
                <div className="active-document-list">
                  {activeDocuments.map((document) => (
                    <div className="active-document-chip" key={document.document_id}>
                      <span className="file-tile"><AppIcon name="document" size={18} /></span>
                      <span className="chip-copy">
                        <strong>{document.title ?? document.filename}</strong>
                        <small>{document.words} words · {document.chunk_count} chunks</small>
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {isPreparingDocument && (
              <div className="preparation-banner" role="status">
                <span className="progress-ring" aria-hidden="true" />
                <span>
                  <strong>Loading assigned knowledge base</strong>
                  <small>Lumen Institute — Student Handbook</small>
                </span>
              </div>
            )}

            <form className="composer" onSubmit={sendMessage}>
              <label className="sr-only" htmlFor="message-input">Your message</label>
              <textarea
                id="message-input"
                ref={inputRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={hasActiveDocuments ? "Ask a question about the Lumen Student Handbook..." : "Initialize the assigned handbook before asking..."}
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
              <span><strong>{hasActiveDocuments ? "Assigned Knowledge Base" : "Knowledge Base unavailable"}</strong> · {modeDescription}</span>
              <span>Enter to send · Shift+Enter for a new line</span>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}


export default App;
