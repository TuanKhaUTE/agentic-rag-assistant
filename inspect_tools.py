import json

from src.tools import (
    TOOL_SCHEMAS,
    calculator,
    list_files,
    read_file,
    search_knowledge,
)


def main() -> None:
    print("=== TOOL SCHEMAS ===")
    print(json.dumps(TOOL_SCHEMAS, indent=2))

    print("\n=== SEARCH_KNOWLEDGE ===")
    question = "What is an AI agent and how does it use tools?"
    print(f"Query: {question}")
    knowledge_result = search_knowledge(question)
    print(f"Highest similarity: {knowledge_result['highest_similarity']:.6f}")
    print(f"Sufficient context: {knowledge_result['sufficient_context']}")
    for result in knowledge_result["results"]:
        preview = " ".join(str(result["text"]).split())
        if len(preview) > 140:
            preview = preview[:137] + "..."
        print(
            f"Rank {result['rank']} | score={result['score']:.6f} | "
            f"{result['chunk_id']} | {result['source']}"
        )
        print(f"Preview: {preview}")

    print("\n=== CALCULATOR ===")
    for expression in ("25 * 4", "(100 + 20) / 3"):
        print(f"{expression} = {calculator(expression)}")

    unsafe_expression = "__import__('os').system('whoami')"
    print(f"Unsafe expression: {unsafe_expression}")
    try:
        calculator(unsafe_expression)
    except ValueError as error:
        print(f"Rejected: {error}")

    print("\n=== LIST_FILES ===")
    print("Directory: knowledge")
    for path in list_files("knowledge"):
        print(path)

    print("\n=== READ_FILE ===")
    file_path = "knowledge/rag.txt"
    content = read_file(file_path)
    preview = content[:300].replace("\n", " ")
    print(f"Path: {file_path}")
    print(f"Characters read: {len(content)}")
    print(f"Preview: {preview}...")

    traversal_path = "../../something.txt"
    print(f"Traversal attempt: {traversal_path}")
    try:
        read_file(traversal_path)
    except ValueError as error:
        print(f"Rejected: {error}")


if __name__ == "__main__":
    main()
