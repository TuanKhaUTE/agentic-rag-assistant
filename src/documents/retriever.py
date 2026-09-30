from typing import Any

from config import EMBEDDING_MODEL, TOP_K
from src.documents.indexer import DocumentIndexError, load_uploaded_index
from src.documents.storage import get_document_metadata
from src.vector_store import embed_query, rank_chunks, validate_vector_index


class DocumentNotIndexedError(ValueError):
    pass


def retrieve_uploaded_document(
    document_id: str,
    question: str,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Retrieve Top-K chunks from exactly one uploaded document index."""
    metadata = get_document_metadata(document_id)
    if metadata.get("status") != "indexed":
        raise DocumentNotIndexedError(
            "The document must be indexed before questions can be asked."
        )
    if metadata.get("embedding_model") != EMBEDDING_MODEL:
        raise DocumentIndexError(
            "The document index uses a different embedding model."
        )

    metadata, chunks, embeddings = load_uploaded_index(document_id)
    try:
        validated_chunks = validate_vector_index(chunks, embeddings)
    except ValueError as error:
        raise DocumentIndexError(str(error)) from error

    stored_dimension = metadata.get("embedding_dimension")
    if stored_dimension != embeddings.shape[1]:
        raise DocumentIndexError(
            "Document metadata does not match the stored embedding dimension."
        )

    for chunk in validated_chunks:
        if str(chunk.get("document_id", "")) != document_id:
            raise DocumentIndexError(
                "Document index contains a chunk from another document."
            )

    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    actual_top_k = min(top_k, len(validated_chunks))

    query_vector = embed_query(question, expected_dimension=embeddings.shape[1])
    ranked_results = rank_chunks(validated_chunks, embeddings, query_vector)

    return {
        "metadata": metadata,
        "query_vector": query_vector,
        "ranked_results": ranked_results,
        "top_results": ranked_results[:actual_top_k],
    }
