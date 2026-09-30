from typing import Any

from config import MIN_SIMILARITY, TOP_K
from src.documents.retriever import retrieve_uploaded_document
from src.generator import generate_answer
from src.prompt_builder import (
    INSUFFICIENT_INFORMATION_MESSAGE,
    build_uploaded_document_prompt,
)


UPLOADED_THRESHOLD_NOTE = (
    f"The {MIN_SIMILARITY:.2f} similarity threshold is an educational heuristic "
    "calibrated on the static homework corpus; uploaded documents may require "
    "recalibration."
)


def answer_uploaded_document(
    document_id: str,
    question: str,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    retrieval = retrieve_uploaded_document(document_id, question, top_k=top_k)
    top_results = retrieval["top_results"]
    highest_similarity = float(top_results[0]["score"])
    sufficient_context = highest_similarity >= MIN_SIMILARITY

    context = ""
    system_message = ""
    user_message = ""
    generation_called = False
    if sufficient_context:
        context, system_message, user_message = build_uploaded_document_prompt(
            question,
            top_results,
        )
        answer = generate_answer(system_message, user_message)
        generation_called = True
    else:
        answer = INSUFFICIENT_INFORMATION_MESSAGE

    return {
        **retrieval,
        "question": question,
        "highest_similarity": highest_similarity,
        "similarity_threshold": MIN_SIMILARITY,
        "threshold_note": UPLOADED_THRESHOLD_NOTE,
        "sufficient_context": sufficient_context,
        "generation_called": generation_called,
        "context": context,
        "system_message": system_message,
        "user_message": user_message,
        "answer": answer,
    }
