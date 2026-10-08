import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.documents import DocumentNotFoundError
from backend.documents.indexer import DocumentIndexError, load_uploaded_collection
from backend.rag.vector_store import get_collection_records


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/inspect_uploaded_index.py <document_id>")

    document_id = sys.argv[1]
    try:
        metadata, collection = load_uploaded_collection(document_id)
    except (DocumentNotFoundError, DocumentIndexError) as error:
        raise SystemExit(f"Inspection failed: {error}") from error

    chunks = get_collection_records(collection, include_embeddings=True)
    embeddings = np.vstack([chunk["embedding"] for chunk in chunks])
    vector_norms = np.linalg.norm(embeddings, axis=1)

    print("=== UPLOADED DOCUMENT INDEX ===")
    print(f"Document ID: {metadata['document_id']}")
    print(f"Filename: {metadata['filename']}")
    print(f"Status: {metadata['status']}")
    print(f"Chunk count: {len(chunks)}")
    print(f"Embedding model: {metadata['embedding_model']}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Chroma collection: {collection.name}")
    print(f"Stored vector shape: {embeddings.shape}")

    for chunk, vector in zip(chunks[:3], embeddings[:3], strict=True):
        preview = str(chunk["text"])[:160]
        first_values = np.array2string(vector[:8], precision=6, separator=", ")
        print("\n--- CHUNK ---")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Word range: [{chunk['start_word']}, {chunk['end_word']})")
        print(f"Text preview: {preview}")
        print(f"First 8 embedding values: {first_values}")
        print(f"Vector norm: {np.linalg.norm(vector):.6f}")

    print("\n=== VERIFICATION ===")
    if len(chunks) != embeddings.shape[0]:
        raise SystemExit("FAIL: Chroma chunk count does not match vector count")
    print("PASS: one stored Chroma vector per chunk")

    if not np.allclose(vector_norms, 1.0, atol=1e-5):
        raise SystemExit("FAIL: one or more vectors are not normalized")
    print("PASS: all vectors have norm approximately 1.0")


if __name__ == "__main__":
    main()
