const API_BASE_URL = "http://127.0.0.1:8000";


export class DocumentApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = "DocumentApiError";
    this.status = status;
  }
}


async function controlledError(response, fallbackMessage) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string" && data.detail.trim()) {
      return new DocumentApiError(data.detail, response.status);
    }
  } catch {
    // Fall through to the friendly generic message.
  }
  return new DocumentApiError(fallbackMessage, response.status);
}


export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/documents/upload`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw await controlledError(response, "The document could not be uploaded.");
  }
  return response.json();
}


export async function indexDocument(documentId) {
  const response = await fetch(
    `${API_BASE_URL}/documents/${encodeURIComponent(documentId)}/index`,
    { method: "POST" },
  );
  if (!response.ok) {
    throw await controlledError(response, "The document could not be indexed.");
  }
  return response.json();
}


export async function askDocument(documentId, question) {
  const response = await fetch(
    `${API_BASE_URL}/documents/${encodeURIComponent(documentId)}/ask`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    },
  );
  if (!response.ok) {
    throw await controlledError(
      response,
      "The document question could not be completed.",
    );
  }
  return response.json();
}


export async function askDocuments(documentIds, question) {
  const response = await fetch(`${API_BASE_URL}/documents/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_ids: documentIds, question }),
  });
  if (!response.ok) {
    throw await controlledError(
      response,
      "The knowledge-base question could not be completed.",
    );
  }
  return response.json();
}
