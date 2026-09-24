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
