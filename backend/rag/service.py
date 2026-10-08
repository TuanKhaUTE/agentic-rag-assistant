from typing import Any

from backend.config import MIN_SIMILARITY, TOP_K
from backend.rag.generator import generate_answer
from backend.rag.prompt_builder import (
    INSUFFICIENT_INFORMATION_MESSAGE,
    build_grounded_prompt,
)
from backend.rag.retriever import retrieve_documents


THRESHOLD_NOTE = (
    f"The {MIN_SIMILARITY:.2f} similarity threshold is an educational heuristic, "
    "not a universal relevance boundary."
)


def answer_question(
    document_ids: list[str],
    question: str,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    retrieval = retrieve_documents(document_ids, question, top_k=top_k)
    return _answer_from_retrieval(question, retrieval)


def _answer_from_retrieval(
    question: str,
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    top_results = retrieval["top_results"]
    highest_similarity = float(top_results[0]["score"])
    retrieval_gate_passed = highest_similarity >= MIN_SIMILARITY

    context = ""
    system_message = ""
    user_message = ""
    generation_called = False
    if retrieval_gate_passed:
        context, system_message, user_message = build_grounded_prompt(
            question,
            top_results,
        )
        answer = generate_answer(system_message, user_message)
        generation_called = True
    else:
        answer = INSUFFICIENT_INFORMATION_MESSAGE

    # After retrieval passes, the configured generator can still find that the
    # supplied chunks do not support an answer and return the required refusal.
    answerable = answer != INSUFFICIENT_INFORMATION_MESSAGE

    return {
        **retrieval,
        "question": question,
        "highest_similarity": highest_similarity,
        "similarity_threshold": MIN_SIMILARITY,
        "threshold_note": THRESHOLD_NOTE,
        "retrieval_gate_passed": retrieval_gate_passed,
        "sufficient_context": retrieval_gate_passed,
        "generation_called": generation_called,
        "answerable": answerable,
        "context": context,
        "system_message": system_message,
        "user_message": user_message,
        "answer": answer,
    }
