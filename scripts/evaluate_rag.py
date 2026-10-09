import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (  # noqa: E402
    ASSIGNED_KB_PATH,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    LLM_PROVIDER,
    MIN_SIMILARITY,
    OLLAMA_GENERATION_MODEL,
    TOP_K,
)
from backend.knowledge_base import initialize_assigned_knowledge_base  # noqa: E402
from backend.rag.prompt_builder import INSUFFICIENT_INFORMATION_MESSAGE  # noqa: E402
from backend.rag.service import answer_question  # noqa: E402


CASES_PATH = PROJECT_ROOT / "tests" / "rag_cases.json"
CITATION_PATTERN = re.compile(r"\[([0-9a-f]{32}#chunk-\d{3})\]")


def _patterns_match(patterns: list[str], answer: str) -> bool:
    return all(
        re.search(pattern, answer, flags=re.IGNORECASE | re.DOTALL)
        for pattern in patterns
    )


def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    metadata = initialize_assigned_knowledge_base()
    document_id = str(metadata["document_id"])
    rows: list[dict[str, object]] = []

    print("=== LUMEN HANDBOOK RAG EVALUATION ===")
    print(f"Knowledge base: {metadata['title']}")
    print(f"Source file: {ASSIGNED_KB_PATH.name}")
    print(f"Document ID: {document_id}")

    for case in cases:
        question = str(case["question"])
        result = answer_question([document_id], question)
        top_results = result["top_results"]
        top_indexes = {int(chunk["chunk_index"]) for chunk in top_results}
        sources_valid = all(
            str(chunk["source"]) == ASSIGNED_KB_PATH.name for chunk in top_results
        )
        citations = CITATION_PATTERN.findall(str(result["answer"]))
        allowed_ids = {str(chunk["chunk_id"]) for chunk in top_results}
        citations_valid = bool(citations) and set(citations).issubset(allowed_ids)
        required_indexes = {
            int(value) for value in case.get("required_chunk_indexes", [])
        }
        required_chunks_retrieved = required_indexes.issubset(top_indexes)
        patterns_ok = _patterns_match(
            case.get("required_patterns", []), str(result["answer"])
        )

        if case["type"] == "answerable":
            passed = bool(
                result["generation_called"]
                and result["answerable"]
                and citations_valid
                and patterns_ok
                and required_chunks_retrieved
                and sources_valid
            )
        else:
            passed = str(result["answer"]) == INSUFFICIENT_INFORMATION_MESSAGE

        rows.append(
            {
                "id": case["id"],
                "type": case["type"],
                "score": float(result["highest_similarity"]),
                "generated": bool(result["generation_called"]),
                "passed": passed,
            }
        )

        print(f"\n{case['id']}. {question}")
        print(f"Query embedding dimension: {result['query_vector'].shape[0]}")
        print(f"Candidate pool: {result['candidate_count']}")
        print(f"Global Top-{TOP_K} retrieved chunks:")
        for chunk in top_results:
            preview = " ".join(str(chunk["text"]).split())[:180]
            print(
                f"  {chunk['rank']}. {float(chunk['score']):.6f} | "
                f"{chunk['chunk_id']} | {chunk['source']}"
            )
            print(f"     {preview}...")
        print(f"Highest similarity: {result['highest_similarity']:.6f}")
        print(f"Threshold: {result['similarity_threshold']:.2f} (heuristic)")
        print(
            "Gate: "
            + ("passed" if result["retrieval_gate_passed"] else "failed")
        )
        print(f"Generation called: {'yes' if result['generation_called'] else 'no'}")
        print(f"Answer: {result['answer']}")
        print(f"Assigned source only: {'yes' if sources_valid else 'no'}")
        if case["type"] == "answerable":
            print(f"Citations valid: {'yes' if citations_valid else 'no'}")
            print(f"Expected facts present: {'yes' if patterns_ok else 'no'}")
            if required_indexes:
                print(
                    "Required amendment chunk retrieved: "
                    + ("yes" if required_chunks_retrieved else "no")
                )
        print(f"Result: {'PASS' if passed else 'FAIL'}")

    passed_count = sum(bool(row["passed"]) for row in rows)
    answerable_count = sum(row["type"] == "answerable" for row in rows)
    answerable_passed = sum(
        bool(row["passed"]) and row["type"] == "answerable" for row in rows
    )
    unanswerable_count = len(rows) - answerable_count
    unanswerable_passed = sum(
        bool(row["passed"]) and row["type"] == "unanswerable" for row in rows
    )
    generation_model = (
        GEMINI_MODEL if LLM_PROVIDER == "gemini" else OLLAMA_GENERATION_MODEL
    )

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Knowledge base documents: 1 ({ASSIGNED_KB_PATH.name})")
    print(f"Chunks: {metadata['chunk_count']}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Retrieval: one query embedding, Chroma cosine, Global Top-{TOP_K}")
    print(f"Similarity threshold: {MIN_SIMILARITY:.2f} (heuristic)")
    print(f"Generation provider/model: {LLM_PROVIDER}/{generation_model}")
    print(f"Answerable tests passed: {answerable_passed}/{answerable_count}")
    print(f"Unanswerable tests passed: {unanswerable_passed}/{unanswerable_count}")
    print(f"Total tests passed: {passed_count}/{len(rows)}")

    if passed_count != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
