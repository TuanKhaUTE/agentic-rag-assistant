import sys

import numpy as np

from src.documents import DocumentNotFoundError
from src.documents.indexer import DocumentIndexError, load_uploaded_index


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python inspect_uploaded_index.py <document_id>")

    document_id = sys.argv[1]
    try:
        metadata, chunks, embeddings = load_uploaded_index(document_id)
    except (DocumentNotFoundError, DocumentIndexError) as error:
        raise SystemExit(f"Inspection failed: {error}") from error

    vector_norms = np.linalg.norm(embeddings, axis=1)

    print("=== UPLOADED DOCUMENT INDEX ===")
    print(f"Document ID: {metadata['document_id']}")
    print(f"Filename: {metadata['filename']}")
    print(f"Status: {metadata['status']}")
    print(f"Chunk count: {len(chunks)}")
    print(f"Embedding model: {metadata['embedding_model']}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Matrix shape: {embeddings.shape}")

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
        raise SystemExit("FAIL: chunk count does not match vector row count")
    print("PASS: one vector row per chunk")

    if not np.allclose(vector_norms, 1.0, atol=1e-5):
        raise SystemExit("FAIL: one or more vectors are not normalized")
    print("PASS: all vectors have norm approximately 1.0")


if __name__ == "__main__":
    main()
