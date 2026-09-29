const CHAT_URL = "http://127.0.0.1:8000/chat";


export async function sendChatMessage(message) {
  const response = await fetch(CHAT_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    throw new Error(`Backend request failed with status ${response.status}`);
  }

  const data = await response.json();
  if (
    typeof data.answer !== "string" ||
    !Number.isInteger(data.steps) ||
    !Array.isArray(data.tools_used)
  ) {
    throw new Error("Backend returned an unexpected response format");
  }

  return data;
}
