import { useEffect, useRef, useState } from "react";

import { sendChatMessage } from "./services/chatApi.js";


const EXAMPLE_PROMPTS = [
  "What is RAG? Use the study knowledge base.",
  "Calculate 10 * 25.",
  "List the files in the knowledge folder.",
  "Use the study knowledge base to explain AI agents, then calculate 3 * 2.",
];


function AssistantMetadata({ steps, toolsUsed }) {
  return (
    <div className="message-metadata">
      <span>Steps: {steps}</span>
      <span>Tools used: {toolsUsed.length ? toolsUsed.join(", ") : "none"}</span>
    </div>
  );
}


function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);
  const historyEndRef = useRef(null);

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

  return (
    <main className="app-shell">
      <header className="page-header">
        <p className="eyebrow">University homework lab</p>
        <h1>AI Study Chat Assistant</h1>
        <p className="subtitle">Local RAG + Agent demo</p>
      </header>

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
    </main>
  );
}


export default App;
