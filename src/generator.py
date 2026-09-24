import ollama

from config import GENERATION_MODEL


def generate_answer(system_message: str, user_message: str) -> str:
    """Generate one grounded answer; retrieval happens outside this module."""
    if not system_message.strip() or not user_message.strip():
        raise ValueError("System and user messages must not be empty")

    response = ollama.chat(
        model=GENERATION_MODEL,
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

    answer = response.message.content.strip()
    if not answer:
        raise ValueError("Generation model returned an empty answer")
    return answer
