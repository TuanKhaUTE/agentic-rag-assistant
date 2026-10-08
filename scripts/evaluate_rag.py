import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (  # noqa: E402
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    LLM_PROVIDER,
    MIN_SIMILARITY,
    OLLAMA_GENERATION_MODEL,
    TOP_K,
)
from backend.documents.indexer import index_uploaded_document  # noqa: E402
from backend.documents.storage import create_document  # noqa: E402
from backend.rag.prompt_builder import INSUFFICIENT_INFORMATION_MESSAGE  # noqa: E402
from backend.rag.service import answer_question  # noqa: E402


FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"
CASES_PATH = PROJECT_ROOT / "tests" / "rag_cases.json"
CITATION_PATTERN = re.compile(r"\[([0-9a-f]{32}#chunk-\d{3})\]")


def build_fixture_indexes() -> tuple[list[str], dict[str, str]]:
    document_ids: list[str] = []
    sources_by_id: dict[str, str] = {}
    for path in sorted(FIXTURE_DIR.iterdir()):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        uploaded = create_document(path.read_bytes(), path.name)
        document_id = str(uploaded["document_id"])
        index_uploaded_document(document_id)
        document_ids.append(document_id)
        sources_by_id[document_id] = path.name
    return document_ids, sources_by_id


def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    document_ids, sources_by_id = build_fixture_indexes()
    rows: list[dict[str, object]] = []

    print("=== DOCUMENT-GROUNDED RAG EVALUATION ===")
    for case in cases:
        result = answer_question(document_ids, str(case["question"]))
        top_sources = [str(chunk["source"]) for chunk in result["top_results"]]
        expected_sources = set(case["expected_sources"])
        citations = CITATION_PATTERN.findall(str(result["answer"]))
        allowed_ids = {str(chunk["chunk_id"]) for chunk in result["top_results"]}
        citations_valid = bool(citations) and set(citations).issubset(allowed_ids)

        if case["type"] == "answerable":
            expected_retrieved = expected_sources.issubset(set(top_sources))
            passed = bool(
                expected_retrieved
                and result["generation_called"]
                and result["answerable"]
                and citations_valid
            )
        else:
            expected_retrieved = True
            passed = str(result["answer"]) == INSUFFICIENT_INFORMATION_MESSAGE

        rows.append(
            {
                "id": case["id"],
                "type": case["type"],
                "score": float(result["highest_similarity"]),
                "sources": top_sources,
                "generated": bool(result["generation_called"]),
                "passed": passed,
            }
        )

        print(f"\n{case['id']}. {case['question']}")
        print(f"Top-{TOP_K}: {', '.join(top_sources)}")
        print(f"Candidate pool: {result['candidate_count']}")
        print(f"Highest similarity: {result['highest_similarity']:.6f}")
        print(
            "Decision: "
            + ("sufficient context" if result["retrieval_gate_passed"] else "insufficient context")
        )
        print(f"Generation called: {'yes' if result['generation_called'] else 'no'}")
        print(f"Answer: {result['answer']}")
        print(f"Citations valid: {'yes' if citations_valid else 'n/a' if not citations else 'no'}")
        print(f"Result: {'PASS' if passed else 'FAIL'}")

    passed_count = sum(bool(row["passed"]) for row in rows)
    generation_model = GEMINI_MODEL if LLM_PROVIDER == "gemini" else OLLAMA_GENERATION_MODEL
    print("\n=== SUMMARY ===")
    print(f"Documents: {len(document_ids)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Retrieval: one query embedding, per-document Chroma, global Top-{TOP_K}")
    print(f"Similarity threshold: {MIN_SIMILARITY:.2f} (heuristic)")
    print(f"Generation provider/model: {LLM_PROVIDER}/{generation_model}")
    print(f"Tests passed: {passed_count}/{len(rows)}")
    print("Indexed fixture IDs:")
    for document_id in document_ids:
        print(f"- {sources_by_id[document_id]}: {document_id}")

    if passed_count != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
