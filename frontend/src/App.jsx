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


function AssistantMetadata({ steps, toolsUsed }) {
  return (
    <div className="message-metadata">
      <span>Steps: {steps}</span>
      <span>Tools used: {toolsUsed.length ? toolsUsed.join(", ") : "none"}</span>
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
  const [documentQuestion, setDocumentQuestion] = useState("");
  const [documentAnswer, setDocumentAnswer] = useState(null);
  const [isAskingDocument, setIsAskingDocument] = useState(false);
  const [questionError, setQuestionError] = useState("");
  const inputRef = useRef(null);
  const historyEndRef = useRef(null);
  const isPreparingDocument = ["uploading", "indexing"].includes(documentStatus);

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

    if (!message || isLoading) {
      return;
    }

    setMessages((current) => [...current, { role: "user", text: message }]);
    setInput("");
    setError("");
    setIsLoading(true);

    try {
      const result = await sendChatMessage(message);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          text: result.answer,
          steps: result.steps,
          toolsUsed: result.tools_used,
        },
      ]);
    } catch {
      setError("The assistant could not be reached. Check that the backend is running.");
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

  function selectDocument(event) {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setDocumentInfo(null);
    setDocumentAnswer(null);
    setDocumentQuestion("");
    setQuestionError("");
    setDocumentStatus("idle");
    setDocumentError(validateSelectedFile(file));
  }

  async function prepareDocument() {
    const validationError = validateSelectedFile(selectedFile);
    if (validationError || isPreparingDocument || isAskingDocument) {
      setDocumentError(validationError);
      return;
    }

    setDocumentError("");
    setDocumentAnswer(null);
    setDocumentStatus("uploading");
    try {
      const uploaded = await uploadDocument(selectedFile);
      setDocumentInfo(uploaded);
      setDocumentStatus("indexing");
      const indexed = await indexDocument(uploaded.document_id);
      setDocumentInfo({ ...uploaded, ...indexed });
      setDocumentStatus("ready");
    } catch (caughtError) {
      setDocumentStatus("error");
      setDocumentError(
        friendlyDocumentError(
          caughtError,
          "The document could not be prepared. Check that the backend is running.",
        ),
      );
    }
  }

  async function submitDocumentQuestion(event) {
    event?.preventDefault();
    const question = documentQuestion.trim();
    if (!question || !documentInfo?.document_id || isAskingDocument) {
      return;
    }

    setQuestionError("");
    setDocumentAnswer(null);
    setIsAskingDocument(true);
    try {
      const result = await askDocument(documentInfo.document_id, question);
      setDocumentAnswer(result);
    } catch (caughtError) {
      setQuestionError(
        friendlyDocumentError(
          caughtError,
          "The document question failed. Check that the backend is running.",
        ),
      );
    } finally {
      setIsAskingDocument(false);
    }
  }

  function handleDocumentQuestionKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitDocumentQuestion();
    }
  }

  return (
    <main className="app-shell">
      <header className="page-header">
        <p className="eyebrow">University homework lab</p>
        <h1>AI Study Chat Assistant</h1>
        <p className="subtitle">Local RAG + Agent demo</p>
      </header>

      <section className="path-heading general-path" aria-labelledby="general-heading">
        <p className="path-label">Path A</p>
        <h2 id="general-heading">General Assistant</h2>
        <p>Uses the Agent, approved tools, and the static study knowledge base.</p>
      </section>

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

      <section className="chat-panel" aria-label="Chat with the study assistant">
        <div className="message-history" aria-live="polite">
          {messages.length === 0 && (
            <p className="empty-state">
              Choose an example or ask a question to begin.
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
                <p className="message-text">{message.text}</p>
                {message.role === "assistant" && (
                  <AssistantMetadata
                    steps={message.steps}
                    toolsUsed={message.toolsUsed}
                  />
                )}
              </div>
            </article>
          ))}

          {isLoading && (
            <article className="message-row assistant">
              <div className="message-bubble thinking">Thinking...</div>
            </article>
          )}
          <div ref={historyEndRef} />
        </div>

        {error && <p className="error-message" role="alert">{error}</p>}

        <form className="composer" onSubmit={sendMessage}>
          <label htmlFor="message-input">Your message</label>
          <div className="composer-row">
            <textarea
              id="message-input"
              ref={inputRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about the study notes or request a calculation..."
              rows="3"
              disabled={isLoading}
            />
            <button
              className="send-button"
              type="submit"
              disabled={isLoading || !input.trim()}
            >
              {isLoading ? "Thinking..." : "Send"}
            </button>
          </div>
          <p className="input-hint">Enter to send · Shift+Enter for a new line</p>
        </form>
      </section>

      <section className="document-section" aria-labelledby="document-heading">
        <div className="path-heading document-path">
          <p className="path-label">Path B</p>
          <h2 id="document-heading">Ask an Uploaded Document</h2>
          <p>Uses only the selected document&apos;s isolated RAG index.</p>
        </div>

        <div className="document-panel">
          <p className="document-guidance">
            Upload a TXT, PDF, or DOCX file. The assistant will extract its text,
            split it into chunks, create embeddings, and use the most relevant
            chunks to answer your questions.
          </p>

          <div className="upload-controls">
            <label className="file-picker" htmlFor="document-file">
              Select document
              <input
                id="document-file"
                type="file"
                accept=".txt,.pdf,.docx"
                onChange={selectDocument}
                disabled={isPreparingDocument || isAskingDocument}
              />
            </label>
            <span className="selected-filename">
              {selectedFile?.name ?? "No file selected"}
            </span>
            <button
              className="primary-button"
              type="button"
              onClick={prepareDocument}
              disabled={!selectedFile || isPreparingDocument || isAskingDocument}
            >
              {documentStatus === "uploading" || documentStatus === "indexing"
                ? "Preparing..."
                : "Upload and prepare"}
            </button>
          </div>

          <div className="document-status" aria-live="polite">
            {documentStatus === "uploading" && <p>Uploading...</p>}
            {documentStatus === "indexing" && <p>Extracting and indexing...</p>}
            {documentStatus === "ready" && <p className="ready-status">Document ready</p>}
          </div>

          {documentError && (
            <p className="inline-error" role="alert">{documentError}</p>
          )}

          {documentStatus === "ready" && documentInfo && (
            <dl className="document-metadata">
              <div><dt>Filename</dt><dd>{documentInfo.filename}</dd></div>
              <div><dt>File type</dt><dd>{documentInfo.file_type.toUpperCase()}</dd></div>
              <div><dt>Word count</dt><dd>{documentInfo.words}</dd></div>
              <div><dt>Chunk count</dt><dd>{documentInfo.chunk_count}</dd></div>
              <div><dt>Embedding model</dt><dd>{documentInfo.embedding_model}</dd></div>
            </dl>
          )}

          <form className="document-question-form" onSubmit={submitDocumentQuestion}>
            <label htmlFor="document-question">Question about this document</label>
            <div className="composer-row">
              <textarea
                id="document-question"
                value={documentQuestion}
                onChange={(event) => setDocumentQuestion(event.target.value)}
                onKeyDown={handleDocumentQuestionKeyDown}
                placeholder="Ask a question about this document..."
                rows="3"
                disabled={documentStatus !== "ready" || isAskingDocument}
              />
              <button
                className="primary-button"
                type="submit"
                disabled={
                  documentStatus !== "ready" ||
                  isAskingDocument ||
                  !documentQuestion.trim()
                }
              >
                {isAskingDocument ? "Thinking..." : "Ask Document"}
              </button>
            </div>
            <p className="input-hint">Enter to ask · Shift+Enter for a new line</p>
          </form>

          {questionError && (
            <p className="inline-error" role="alert">{questionError}</p>
          )}

          {documentAnswer && (
            <article className="document-answer" aria-live="polite">
              <p className="message-label">Assistant answer</p>
              <p className="message-text">{documentAnswer.answer}</p>
              <dl className="answer-summary">
                <div><dt>Document</dt><dd>{documentAnswer.filename}</dd></div>
                <div>
                  <dt>Highest similarity</dt>
                  <dd>{documentAnswer.highest_similarity.toFixed(3)}</dd>
                </div>
                <div><dt>Top-K</dt><dd>{documentAnswer.top_k}</dd></div>
                <div>
                  <dt>Threshold</dt>
                  <dd>
                    {documentAnswer.similarity_threshold.toFixed(2)}
                    {documentAnswer.threshold_is_heuristic ? " (heuristic)" : ""}
                  </dd>
                </div>
              </dl>
              <h3>Source chunks</h3>
              <ol className="source-list">
                {documentAnswer.retrieved_chunks.map((chunk) => (
                  <li key={chunk.chunk_id}>
                    <span>{chunk.chunk_id}</span>
                    <strong>{chunk.score.toFixed(3)}</strong>
                  </li>
                ))}
              </ol>
            </article>
          )}
        </div>
      </section>
    </main>
  );
}


export default App;
