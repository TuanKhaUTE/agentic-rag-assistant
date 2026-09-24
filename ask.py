import sys

from config import TOP_K
from src.generator import generate_answer
from src.prompt_builder import build_grounded_prompt
from src.vector_store import search


DEFAULT_QUESTION = (
    "What are the main stages of a retrieval-augmented generation pipeline?"
)


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else DEFAULT_QUESTION

    # search() visibly performs index loading, query embedding, and cosine scoring.
    _, _, top_results = search(question, top_k=TOP_K)

    print("=== QUESTION ===")
    print(question)

    print(f"\n=== TOP-{TOP_K} RETRIEVAL ===")
    for result in top_results:
        print(f"Rank: {result['rank']}")
        print(f"Similarity score: {result['score']:.6f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Source: {result['source']}")
        print()

    context, system_message, user_message = build_grounded_prompt(
        question,
        top_results,
    )

    print("=== FINAL CONTEXT SENT TO LLM ===")
    print(context)

    print("\n=== SYSTEM PROMPT ===")
    print(system_message)

    print("\n=== USER PROMPT ===")
    print(user_message)

    answer = generate_answer(system_message, user_message)
    print("\n=== FINAL ANSWER ===")
    print(answer)


if __name__ == "__main__":
    main()
