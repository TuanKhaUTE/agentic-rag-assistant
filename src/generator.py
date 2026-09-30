import ollama
from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OLLAMA_GENERATION_MODEL,
)


class GenerationConfigurationError(ValueError):
    """Report an invalid generation-provider configuration without secrets."""


def _generate_with_ollama(system_message: str, user_message: str) -> str:
    if not OLLAMA_GENERATION_MODEL:
        raise GenerationConfigurationError(
            "OLLAMA_GENERATION_MODEL must not be empty."
        )
    response = ollama.chat(
        model=OLLAMA_GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        options={
            "temperature": 0.1,
            "num_ctx": 4096,
            "num_predict": 512,
        },
    )
    return response.message.content.strip()


def _generate_with_gemini(system_message: str, user_message: str) -> str:
    if not GEMINI_API_KEY:
        raise GenerationConfigurationError(
            "LLM_PROVIDER is 'gemini', but GEMINI_API_KEY is not configured."
        )
    if not GEMINI_MODEL:
        raise GenerationConfigurationError("GEMINI_MODEL must not be empty.")

    with genai.Client(api_key=GEMINI_API_KEY) as client:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                temperature=0.1,
                max_output_tokens=512,
            ),
        )
    return (response.text or "").strip()


def generate_answer(system_message: str, user_message: str) -> str:
    """Generate one grounded answer with the configured RAG provider."""
    if not system_message.strip() or not user_message.strip():
        raise ValueError("System and user messages must not be empty.")

    if LLM_PROVIDER == "ollama":
        answer = _generate_with_ollama(system_message, user_message)
    elif LLM_PROVIDER == "gemini":
        answer = _generate_with_gemini(system_message, user_message)
    else:
        raise GenerationConfigurationError(
            f"Unsupported LLM_PROVIDER {LLM_PROVIDER!r}. Use 'ollama' or 'gemini'."
        )

    if not answer:
        raise ValueError("Generation model returned an empty answer.")
    return answer
