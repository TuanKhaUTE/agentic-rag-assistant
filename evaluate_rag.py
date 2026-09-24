import json
import re
from pathlib import Path

from config import (
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    MIN_SIMILARITY,
    PROJECT_ROOT,
    TOP_K,
)
from src.generator import generate_answer
from src.prompt_builder import (
    INSUFFICIENT_INFORMATION_MESSAGE,
    build_grounded_prompt,
)
from src.vector_store import load_index
from src.vector_store import search


TEST_QUESTIONS_PATH = PROJECT_ROOT / "tests" / "test_questions.json"


def load_test_questions(path: Path = TEST_QUESTIONS_PATH) -> list[dict[str, object]]:
    tests = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(tests, list) or not tests:
        raise ValueError("Test question file must contain a non-empty list")
    return tests


def collect_retrieval_results(
    tests: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for test in tests:
        _, _, top_results = search(str(test["question"]), top_k=TOP_K)
        top_sources = [str(result["source"]) for result in top_results]
        expected_source = test["expected_source"]
        expected_in_top_k = (
            expected_source in top_sources
            if expected_source is not None
            else None
        )
        rows.append(
            {
                **test,
                "top_results": top_results,
                "top_1_chunk": top_results[0]["chunk_id"],
                "top_1_source": top_results[0]["source"],
                "top_1_score": top_results[0]["score"],
                "top_sources": top_sources,
                "expected_in_top_k": expected_in_top_k,
            }
        )
    return rows


def print_retrieval_table(rows: list[dict[str, object]]) -> None:
    print("=== RETRIEVAL EVALUATION BEFORE THRESHOLD ===")
    print(
        "ID | Type         | Top-1 score | Top-1 chunk"
        "                              | Top-3 sources"
        "                                                    | Expected in Top-3"
    )
    print("-" * 170)
    for row in rows:
        top_sources = ", ".join(row["top_sources"])
        expected = (
            "N/A"
            if row["expected_in_top_k"] is None
            else "YES" if row["expected_in_top_k"] else "NO"
        )
        print(
            f"{row['id']:<2} | {row['type']:<12} | "
            f"{row['top_1_score']:>11.6f} | "
            f"{row['top_1_chunk']:<40} | "
            f"{top_sources:<64} | {expected}"
        )

    print("\n=== HIGHEST SIMILARITY SCORE FOR EVERY QUESTION ===")
    for row in rows:
        print(f"{row['id']} | {row['top_1_score']:.6f} | {row['question']}")


def print_score_ranges(rows: list[dict[str, object]]) -> None:
    answerable_scores = [
        float(row["top_1_score"])
        for row in rows
        if row["type"] == "answerable"
    ]
    unanswerable_scores = [
        float(row["top_1_score"])
        for row in rows
        if row["type"] == "unanswerable"
    ]

    print("\n=== ANSWERABLE TOP-1 SCORE RANGE ===")
    print(f"Minimum: {min(answerable_scores):.6f}")
    print(f"Maximum: {max(answerable_scores):.6f}")

    print("\n=== UNANSWERABLE TOP-1 SCORE RANGE ===")
    print(f"Minimum: {min(unanswerable_scores):.6f}")
    print(f"Maximum: {max(unanswerable_scores):.6f}")

    print("\n=== SEPARATION CHECK ===")
    if max(unanswerable_scores) < min(answerable_scores):
        print("Clean separation observed: YES")
        print(
            "A candidate threshold can be chosen between "
            f"{max(unanswerable_scores):.6f} and {min(answerable_scores):.6f}."
        )
    else:
        print("Clean separation observed: NO")
        print(
            "Raw cosine similarity alone does not perfectly separate "
            "answerable and unanswerable questions in this test set."
        )


def print_threshold_calibration(rows: list[dict[str, object]]) -> None:
    answerable_scores = [
        float(row["top_1_score"])
        for row in rows
        if row["type"] == "answerable"
    ]
    unanswerable_scores = [
        float(row["top_1_score"])
        for row in rows
        if row["type"] == "unanswerable"
    ]
    answerable_minimum = min(answerable_scores)
    unanswerable_maximum = max(unanswerable_scores)

    print("\n=== THRESHOLD CALIBRATION ===")
    print(f"Observed unanswerable maximum: {unanswerable_maximum:.6f}")
    print(f"Observed answerable minimum: {answerable_minimum:.6f}")
    print(f"Configured MIN_SIMILARITY: {MIN_SIMILARITY:.2f}")
    if unanswerable_maximum < MIN_SIMILARITY < answerable_minimum:
        print("Reason: 0.30 lies inside the clean observed score gap.")
        print(
            "Margin above unanswerable maximum: "
            f"{MIN_SIMILARITY - unanswerable_maximum:.6f}"
        )
        print(
            "Margin below answerable minimum: "
            f"{answerable_minimum - MIN_SIMILARITY:.6f}"
        )
    else:
        print("WARNING: configured threshold is not inside a clean observed gap.")
    print(
        "This threshold is a heuristic for this corpus and test set, "
        "not a universal rule."
    )


def extract_citations(answer: str) -> list[str]:
    return re.findall(r"\[([A-Za-z0-9_.-]+\.txt#chunk-\d{3})\]", answer)


def run_final_pipeline(rows: list[dict[str, object]]) -> None:
    print("\n=== FINAL RAG PIPELINE RUNS ===")
    for row in rows:
        top_results = row["top_results"]
        highest_similarity = float(row["top_1_score"])
        sufficient_context = highest_similarity >= MIN_SIMILARITY
        decision = "sufficient context" if sufficient_context else "insufficient context"

        print(f"\n--- {row['id']}: {row['question']} ---")
        for result in top_results:
            print(
                f"Top-{result['rank']}: {result['score']:.6f} | "
                f"{result['chunk_id']} | {result['source']}"
            )
        print(f"Highest similarity: {highest_similarity:.6f}")
        print(f"Threshold: {MIN_SIMILARITY:.2f}")
        print(f"Decision: {decision}")

        generation_called = False
        if sufficient_context:
            _, system_message, user_message = build_grounded_prompt(
                str(row["question"]),
                top_results,
            )
            answer = generate_answer(system_message, user_message)
            generation_called = True
        else:
            answer = INSUFFICIENT_INFORMATION_MESSAGE

        citations = extract_citations(answer)
        allowed_chunk_ids = {
            str(result["chunk_id"])
            for result in top_results
        }
        valid_citations = bool(citations) and all(
            citation in allowed_chunk_ids
            for citation in citations
        )

        expected_source = row["expected_source"]
        expected_source_cited = (
            any(
                citation.startswith(f"{expected_source}#")
                for citation in citations
            )
            if expected_source is not None
            else None
        )

        if row["type"] == "answerable":
            passed = bool(
                row["expected_in_top_k"]
                and sufficient_context
                and generation_called
                and valid_citations
                and expected_source_cited
            )
            result_description = (
                "grounded answer with valid expected-source citation"
                if passed
                else "answerable-case requirement failed"
            )
        else:
            passed = bool(
                not sufficient_context
                and not generation_called
                and answer == INSUFFICIENT_INFORMATION_MESSAGE
            )
            result_description = (
                "exact refusal before generation"
                if passed
                else "unanswerable-case requirement failed"
            )

        row["threshold_decision"] = decision
        row["generation_called"] = generation_called
        row["answer"] = answer
        row["citations"] = citations
        row["result_description"] = result_description
        row["passed"] = passed

        print(f"Generation called: {'YES' if generation_called else 'NO'}")
        print(f"Answer: {answer}")
        print(f"Citations: {', '.join(citations) if citations else 'none'}")
        print(f"Result: {result_description}")
        print(f"Pass/Fail: {'PASS' if passed else 'FAIL'}")


def print_final_evaluation_table(rows: list[dict[str, object]]) -> None:
    print("\n=== FINAL EVALUATION TABLE ===")
    print(
        "Question | Type | Top-1 source | Top-1 score | "
        "Expected in Top-3 | Decision             | Generated | Result                                      | Pass"
    )
    print("-" * 220)
    for row in rows:
        expected = (
            "N/A"
            if row["expected_in_top_k"] is None
            else "YES" if row["expected_in_top_k"] else "NO"
        )
        print(
            f"{row['question']} | "
            f"{row['type']} | "
            f"{row['top_1_source']} | "
            f"{row['top_1_score']:.6f} | "
            f"{expected} | "
            f"{row['threshold_decision']} | "
            f"{'YES' if row['generation_called'] else 'NO'} | "
            f"{row['result_description']} | "
            f"{'PASS' if row['passed'] else 'FAIL'}"
        )


def print_pipeline_summary(rows: list[dict[str, object]]) -> None:
    chunks, embeddings = load_index()
    document_count = len({str(chunk["source"]) for chunk in chunks})
    answerable_rows = [row for row in rows if row["type"] == "answerable"]
    unanswerable_rows = [row for row in rows if row["type"] == "unanswerable"]

    print("\n=== RAG PIPELINE COMPLETE ===")
    print(f"Documents: {document_count}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print("Retrieval: NumPy cosine similarity via normalized dot product")
    print(f"Top-K: {TOP_K}")
    print(f"Generation model: {GENERATION_MODEL}")
    print(f"Similarity threshold: {MIN_SIMILARITY:.2f} (heuristic)")
    print(
        "Answerable tests passed: "
        f"{sum(bool(row['passed']) for row in answerable_rows)}/{len(answerable_rows)}"
    )
    print(
        "Unanswerable tests passed: "
        f"{sum(bool(row['passed']) for row in unanswerable_rows)}/{len(unanswerable_rows)}"
    )


def main() -> None:
    tests = load_test_questions()
    rows = collect_retrieval_results(tests)
    print_retrieval_table(rows)
    print_score_ranges(rows)
    print_threshold_calibration(rows)
    run_final_pipeline(rows)
    print_final_evaluation_table(rows)
    print_pipeline_summary(rows)


if __name__ == "__main__":
    main()
