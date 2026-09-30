from typing import Any

from config import EMBEDDING_MODEL, TOP_K
from src.documents.indexer import DocumentIndexError, load_uploaded_collection
from src.documents.storage import get_document_metadata
from src.vector_store import query_chroma_collection


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

    metadata, collection = load_uploaded_collection(document_id)
    stored_dimension = metadata.get("embedding_dimension")
    collection_dimension = (collection.metadata or {}).get("embedding_dimension")
    if stored_dimension != collection_dimension:
        raise DocumentIndexError(
            "Document metadata does not match the Chroma embedding dimension."
        )

    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    actual_top_k = min(top_k, collection.count())
    query_vector, top_results = query_chroma_collection(
        collection,
        question,
        top_k=actual_top_k,
        # Physical per-document persistence plus this filter provide isolation.
        where={"document_id": document_id},
    )
    if not top_results:
        raise DocumentIndexError("The document Chroma query returned no chunks.")
    if any(result.get("document_id") != document_id for result in top_results):
        raise DocumentIndexError("Chroma returned a chunk from another document.")

    return {
        "metadata": metadata,
        "query_vector": query_vector,
        "top_results": top_results,
    }
