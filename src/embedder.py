import numpy as np
import ollama

from config import EMBEDDING_MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    """Convert an ordered list of texts into an ordered float32 matrix."""
    if not texts:
        raise ValueError("At least one text is required for embedding")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every embedding input must be a non-empty string")

    response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    returned_embeddings = response.embeddings

    if len(returned_embeddings) != len(texts):
        raise ValueError(
            "Embedding count does not match input count: "
            f"received {len(returned_embeddings)} for {len(texts)} texts"
        )

    dimensions = {len(embedding) for embedding in returned_embeddings}
    if len(dimensions) != 1:
        raise ValueError(f"Embeddings have inconsistent dimensions: {dimensions}")

    matrix = np.asarray(returned_embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, received shape {matrix.shape}")

    return matrix


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize an embedding matrix and return float32 vectors."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, received shape {matrix.shape}")

    vector_lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(vector_lengths == 0):
        raise ValueError("Cannot normalize a zero-length embedding vector")

    normalized = (matrix / vector_lengths).astype(np.float32, copy=False)
    normalized_lengths = np.linalg.norm(normalized, axis=1)
    if not np.allclose(normalized_lengths, 1.0, atol=1e-5):
        raise ValueError("L2 normalization verification failed")
    return normalized
