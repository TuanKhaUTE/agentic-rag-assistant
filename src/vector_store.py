import json

import numpy as np

from config import CHUNKS_PATH, EMBEDDINGS_PATH, TOP_K
from src.embedder import embed_texts


def load_index() -> tuple[list[dict[str, object]], np.ndarray]:
    """Load and validate the chunk metadata and its aligned vector matrix."""
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=False)

    if not isinstance(chunks, list):
        raise ValueError("chunks.json must contain a list")
    if embeddings.ndim != 2:
        raise ValueError(f"Stored embeddings must be 2D, received {embeddings.shape}")
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Index alignment failure: {len(chunks)} chunks but "
            f"{embeddings.shape[0]} embedding rows"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Stored embeddings contain non-finite values")

    vector_norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(vector_norms, 1.0, atol=1e-5):
        raise ValueError("Stored embedding vectors are not L2-normalized")

    return chunks, embeddings


def embed_query(query: str, expected_dimension: int) -> np.ndarray:
    """Embed and L2-normalize one query using the configured embedding model."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    query_vector = embed_texts([query])[0]
    if query_vector.shape[0] != expected_dimension:
        raise ValueError(
            f"Query dimension {query_vector.shape[0]} does not match "
            f"stored dimension {expected_dimension}"
        )

    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        raise ValueError("Cannot normalize a zero-length query vector")

    normalized_query = (query_vector / query_norm).astype(np.float32, copy=False)
    if not np.isclose(np.linalg.norm(normalized_query), 1.0, atol=1e-5):
        raise ValueError("Query L2 normalization verification failed")

    return normalized_query


def search(
    query: str,
    top_k: int = TOP_K,
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    """Return the query vector, every ranked result, and the Top-K results."""
    chunks, embeddings = load_index()
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")
    if top_k <= 0 or top_k > len(chunks):
        raise ValueError(f"top_k must be between 1 and {len(chunks)}")

    query_vector = embed_query(query, expected_dimension=embeddings.shape[1])

    # Unit vectors make this dot product exactly the cosine-similarity formula.
    scores = embeddings @ query_vector
    sorted_indices = np.argsort(-scores, kind="stable")

    ranked_results: list[dict[str, object]] = []
    for rank, row_index in enumerate(sorted_indices, start=1):
        chunk = chunks[int(row_index)]
        ranked_results.append(
            {
                "rank": rank,
                "score": float(scores[row_index]),
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "start_word": chunk["start_word"],
                "end_word": chunk["end_word"],
            }
        )

    return query_vector, ranked_results, ranked_results[:top_k]
