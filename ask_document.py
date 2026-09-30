import sys

import numpy as np

from config import EMBEDDING_MODEL
from src.documents import DocumentNotFoundError
from src.documents.indexer import DocumentIndexError
from src.documents.question_answering import answer_uploaded_document
from src.documents.retriever import DocumentNotIndexedError


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            'Usage: python ask_document.py <document_id> "question"'
        )

    document_id = sys.argv[1]
    question = " ".join(sys.argv[2:]).strip()
    try:
        result = answer_uploaded_document(document_id, question)
    except (
        DocumentNotFoundError,
        DocumentNotIndexedError,
        DocumentIndexError,
        ValueError,
    ) as error:
        raise SystemExit(f"Document question failed: {error}") from error

    metadata = result["metadata"]
    query_vector = result["query_vector"]
    top_results = result["top_results"]

    print("=== DOCUMENT ===")
    print(f"Document ID: {document_id}")
    print(f"Filename: {metadata['filename']}")

    print("\n=== QUESTION ===")
    print(question)

    print("\n=== QUERY EMBEDDING ===")
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Dimension: {query_vector.shape[0]}")
    print(f"Norm: {np.linalg.norm(query_vector):.6f}")

    print("\n=== TOP-K RETRIEVAL ===")
    for chunk in top_results:
        print(f"Rank: {chunk['rank']}")
        print(f"Score: {chunk['score']:.6f}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Text preview: {str(chunk['text'])[:180]}")
        print()

    gate_decision = "passed" if result["retrieval_gate_passed"] else "failed"
    print("=== RELEVANCE DECISION ===")
    print(f"Highest score: {result['highest_similarity']:.6f}")
    print(f"Threshold: {result['similarity_threshold']:.2f}")
    print(f"Retrieval gate: {gate_decision} (heuristic)")
    print(f"Generation called: {'yes' if result['generation_called'] else 'no'}")
    print(f"Answerable from context: {'yes' if result['answerable'] else 'no'}")
    print(f"Note: {result['threshold_note']}")

    print("\n=== CONTEXT SENT TO LLM ===")
    if result["generation_called"]:
        print(result["context"])
    else:
        print("No context was sent because generation was skipped.")

    print("\n=== FINAL ANSWER ===")
    print(result["answer"])


if __name__ == "__main__":
    main()
