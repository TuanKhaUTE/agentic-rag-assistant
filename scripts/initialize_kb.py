import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (  # noqa: E402
    ASSIGNED_KB_PATH,
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    EMBEDDING_MODEL,
)
from backend.knowledge_base import initialize_assigned_knowledge_base  # noqa: E402


def main() -> None:
    metadata = initialize_assigned_knowledge_base()
    print("=== ASSIGNED KNOWLEDGE BASE READY ===")
    print(f"Title: {metadata['title']}")
    print(f"Source: {ASSIGNED_KB_PATH.name}")
    print(f"Document ID: {metadata['document_id']}")
    print(f"Status: {metadata['status']}")
    print(f"Words: {metadata['words']}")
    print(f"Chunks: {metadata['chunk_count']}")
    print(f"Chunking: {CHUNK_SIZE_WORDS} words, {CHUNK_OVERLAP_WORDS} overlap")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding dimension: {metadata['embedding_dimension']}")
    print("Vector database: Chroma")
    print("Runtime knowledge restriction: assigned handbook only")


if __name__ == "__main__":
    main()
