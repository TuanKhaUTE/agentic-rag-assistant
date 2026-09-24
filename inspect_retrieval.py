import sys

import numpy as np

from config import EMBEDDING_MODEL, TOP_K
from src.vector_store import search


DEFAULT_QUESTION = (
    "What are the main stages of a retrieval-augmented generation pipeline?"
)


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else DEFAULT_QUESTION
    query_vector, all_results, top_results = search(question, top_k=TOP_K)

    print("=== QUESTION ===")
    print(question)

    print("\n=== QUERY EMBEDDING ===")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Dimension: {query_vector.shape[0]}")
    print(
        "First 8 values: "
        + np.array2string(
            query_vector[:8],
            precision=6,
            separator=", ",
            suppress_small=False,
        )
    )
    print(f"Vector norm: {np.linalg.norm(query_vector):.6f}")

    print("\n=== ALL SIMILARITY SCORES ===")
    for result in all_results:
        print(
            f"{result['score']:.6f} | "
            f"{result['chunk_id']} | {result['source']}"
        )

    print(f"\n=== TOP-{TOP_K} RETRIEVAL ===")
    for result in top_results:
        print("\n" + "-" * 80)
        print(f"Rank: {result['rank']}")
        print(f"Similarity score: {result['score']:.6f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Source: {result['source']}")
        print(f"Word range: [{result['start_word']}, {result['end_word']})")
        print("Full retrieved chunk text:")
        print(result["text"])

    top_result = top_results[0]
    print("\n=== VERIFICATION SUMMARY ===")
    print(f"Question: {question}")
    print(f"Top-1 chunk: {top_result['chunk_id']}")
    print(f"Top-1 source: {top_result['source']}")
    print(f"Top-1 score: {top_result['score']:.6f}")

    print("\n=== WHAT THIS OUTPUT MEANS ===")
    print("Higher cosine similarity means closer semantic meaning.")
    print(f"Top-K means the {TOP_K} most similar chunks.")
    print("Retrieval has selected text, but it has not generated an answer yet.")


if __name__ == "__main__":
    main()
