import numpy as np

from config import EMBEDDING_MODEL
from src.vector_store import get_collection_records, load_static_collection


def main() -> None:
    collection = load_static_collection()
    chunks = get_collection_records(collection, include_embeddings=True)
    embeddings = np.vstack([chunk["embedding"] for chunk in chunks])

    print("=== CHROMA INDEX INSPECTION ===")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Chroma collection: {collection.name}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Stored vector shape: {embeddings.shape}")
    print(f"Data type: {embeddings.dtype}")
    print()
    print(f"Stored chunk count: {len(chunks)}")
    print(f"Stored embedding count: {embeddings.shape[0]}")

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            "FAIL: Chroma chunk count does not match embedding count"
        )
    print("PASS: one Chroma embedding corresponds to one chunk")

    for index, (chunk, vector) in enumerate(
        zip(chunks[:3], embeddings[:3]),
        start=1,
    ):
        preview = " ".join(str(chunk["text"]).split())
        if len(preview) > 100:
            preview = preview[:97] + "..."

        print()
        print(f"--- Chunk {index} / Embedding row {index - 1} ---")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Short text preview: {preview}")
        print(
            "First 8 embedding values: "
            + np.array2string(
                vector[:8],
                precision=6,
                separator=", ",
                suppress_small=False,
            )
        )
        print(f"Vector norm: {np.linalg.norm(vector):.6f}")


if __name__ == "__main__":
    main()
