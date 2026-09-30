import { useEffect, useRef, useState } from "react";

import { sendChatMessage } from "./services/chatApi.js";
import {
  askDocument,
  DocumentApiError,
  indexDocument,
  uploadDocument,
} from "./services/documentApi.js";


const EXAMPLE_PROMPTS = [
  "What is RAG? Use the study knowledge base.",
  "Calculate 10 * 25.",
  "List the files in the knowledge folder.",
  "Use the study knowledge base to explain AI agents, then calculate 3 * 2.",
];
const SUPPORTED_EXTENSIONS = ["txt", "pdf", "docx"];
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;


function GeneralMetadata({ steps, toolsUsed }) {
  return (
    <div className="message-metadata">
      <span>Steps: {steps}</span>
      <span>Tools used: {toolsUsed.length ? toolsUsed.join(", ") : "none"}</span>
    </div>
  );
}


function DocumentMetadata({ result }) {
  return (
    <div className="document-message-metadata">
      <dl className="answer-summary">
        <div><dt>Document</dt><dd>{result.filename}</dd></div>
        <div>
          <dt>Highest similarity</dt>
          <dd>{result.highest_similarity.toFixed(3)}</dd>
        </div>
        <div><dt>Top-K</dt><dd>{result.top_k}</dd></div>
        <div>
          <dt>Heuristic threshold</dt>
          <dd>{result.similarity_threshold.toFixed(2)}</dd>
        </div>
        <div>
          <dt>Retrieval gate</dt>
          <dd>{result.retrieval_gate_passed ? "Passed" : "Failed"}</dd>
        </div>
        <div>
          <dt>Generation called</dt>
          <dd>{result.generation_called ? "Yes" : "No"}</dd>
        </div>
        <div>
          <dt>Answerable</dt>
          <dd>{result.answerable ? "Yes" : "No"}</dd>
        </div>
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

  useEffect(() => {
    historyEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  function chooseExample(prompt) {
    setInput(prompt);
    setError("");
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
    <main className="app-shell">
      <header className="page-header">
        <p className="eyebrow">University homework lab</p>
        <h1>AI Study Chat Assistant</h1>
        <p className="subtitle">Local Agent chat + uploaded-document RAG</p>
      </header>

      {!activeDocument && !isPreparingDocument && (
        <section className="examples" aria-labelledby="examples-heading">
          <h2 id="examples-heading">Example prompts</h2>
          <div className="example-list">
            {EXAMPLE_PROMPTS.map((prompt) => (
              <button
                className="example-button"
                key={prompt}
                type="button"
                onClick={() => chooseExample(prompt)}
                disabled={isLoading}
              >
                {prompt}
              </button>
            ))}
          </div>
        </section>
      )}

      <section className="chat-panel" aria-label="Chat with the study assistant">
        <div className="message-history" aria-live="polite">
          {messages.length === 0 && (
            <p className="empty-state">
              Ask the General Assistant, or attach one document for isolated document Q&amp;A.
            </p>
          )}

          {messages.map((message, index) => (
            <article
              className={`message-row ${message.role}`}
              key={`${message.role}-${index}`}
            >
              <div className="message-bubble">
                <p className="message-label">
                  {message.role === "user" ? "You" : "Assistant"}
                </p>
                <p className="message-mode">{message.modeLabel}</p>
                <p className="message-text">{message.text}</p>
                {message.role === "assistant" && message.mode === "general" && (
                  <GeneralMetadata
                    steps={message.steps}
                    toolsUsed={message.toolsUsed}
                  />
                )}
                {message.role === "assistant" && message.mode === "document" && (
                  <DocumentMetadata result={message.documentResult} />
                )}
              </div>
            </article>
          ))}

          {isLoading && (
            <article className="message-row assistant">
              <div className="message-bubble thinking">
                <p className="message-label">Assistant</p>
                <p className="message-mode">{activeModeLabel}</p>
                Thinking...
              </div>
            </article>
          )}
          <div ref={historyEndRef} />
        </div>

        {error && <p className="error-message" role="alert">{error}</p>}
        {documentError && (
          <p className="error-message" role="alert">{documentError}</p>
        )}

        <div className="attachment-area" aria-live="polite">
          <div className="attachment-row">
            <label
              className={`attach-button ${isLoading || isPreparingDocument ? "disabled" : ""}`}
              htmlFor="document-file"
            >
              Attach document
              <input
                id="document-file"
                ref={fileInputRef}
                type="file"
                accept=".txt,.pdf,.docx"
                onChange={selectDocument}
                disabled={isLoading || isPreparingDocument}
              />
            </label>

            <div className="attachment-status">
              {documentStatus === "idle" && (
                <span>No document attached — messages use the General Assistant.</span>
              )}
              {documentStatus === "uploading" && (
                <span>Uploading {selectedFile?.name}...</span>
              )}
              {documentStatus === "indexing" && (
                <span>Extracting and indexing {selectedFile?.name}...</span>
              )}
              {activeDocument && (
                <>
                  <strong>{activeDocument.filename} ✓ Ready</strong>
                  <span>
                    {activeDocument.words} words · {activeDocument.chunk_count} chunks · {activeDocument.embedding_model}
                  </span>
                </>
              )}
            </div>

            {activeDocument && (
              <button
                className="detach-button"
                type="button"
                onClick={detachDocument}
                disabled={isLoading}
              >
                Detach document
              </button>
            )}
          </div>
          <p className="attachment-guidance">
            Attach one TXT, PDF, or DOCX file to route new messages to its isolated RAG index.
            Detaching returns new messages to the General Assistant.
          </p>
        </div>

        <form className="composer" onSubmit={sendMessage}>
          <label htmlFor="message-input">Your message · {activeModeLabel}</label>
          <div className="composer-row">
            <textarea
              id="message-input"
              ref={inputRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                activeDocument
                  ? `Ask about ${activeDocument.filename}...`
                  : "Ask anything..."
              }
              rows="3"
              disabled={isLoading || isPreparingDocument}
            />
            <button
              className="send-button"
              type="submit"
              disabled={isLoading || isPreparingDocument || !input.trim()}
            >
              {isLoading ? "Thinking..." : "Send"}
            </button>
          </div>
          <p className="input-hint">Enter to send · Shift+Enter for a new line</p>
        </form>
      </section>
    </main>
  );
}


export default App;
