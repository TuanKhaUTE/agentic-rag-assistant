import json

import numpy as np

from config import CHUNKS_PATH, EMBEDDING_MODEL, EMBEDDINGS_PATH


def main() -> None:
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=False)

    if not isinstance(chunks, list):
        raise ValueError("chunks.json must contain a list")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, received shape {embeddings.shape}")
    if embeddings.dtype != np.float32:
        raise ValueError(f"Expected float32 vectors, received {embeddings.dtype}")

    print("=== EMBEDDING INSPECTION ===")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Matrix shape: {embeddings.shape}")
    print(f"Data type: {embeddings.dtype}")
    print()
    print(f"chunks count: {len(chunks)}")
    print(f"embedding matrix row count: {embeddings.shape[0]}")

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            "FAIL: chunk count does not match embedding matrix row count"
        )
    print("PASS: one embedding row corresponds to one chunk")

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
