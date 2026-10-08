from typing import Any

from backend.config import EMBEDDING_MODEL, TOP_K
from backend.documents.indexer import DocumentIndexError, load_uploaded_collection
from backend.documents.storage import get_document_metadata
from backend.rag.vector_store import (
    embed_query,
    query_chroma_collection_with_vector,
)


class DocumentNotIndexedError(ValueError):
    pass


def retrieve_documents(
    document_ids: list[str],
    question: str,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Embed once, query each selected index, then return one global Top-K."""
    if not isinstance(document_ids, list) or not document_ids:
        raise ValueError("At least one document ID is required.")
    if any(not isinstance(document_id, str) for document_id in document_ids):
        raise TypeError("Every document ID must be a string.")
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Document IDs must not contain duplicates.")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    indexed_documents: list[tuple[dict[str, object], Any]] = []
    expected_dimension: int | None = None
    for document_id in document_ids:
        metadata = get_document_metadata(document_id)
        if metadata.get("status") != "indexed":
            raise DocumentNotIndexedError(
                f"Document {document_id} must be indexed before questions can be asked."
            )
        if metadata.get("embedding_model") != EMBEDDING_MODEL:
            raise DocumentIndexError(
                f"Document {document_id} uses a different embedding model."
            )

        metadata, collection = load_uploaded_collection(document_id)
        stored_dimension = int(metadata.get("embedding_dimension", 0))
        collection_dimension = int(
            (collection.metadata or {}).get("embedding_dimension", 0)
        )
        if stored_dimension <= 0 or stored_dimension != collection_dimension:
            raise DocumentIndexError(
                f"Document {document_id} has inconsistent embedding dimensions."
            )
        if expected_dimension is None:
            expected_dimension = stored_dimension
        elif stored_dimension != expected_dimension:
            raise DocumentIndexError(
                "Selected document indexes use different embedding dimensions."
            )
        indexed_documents.append((metadata, collection))

    if expected_dimension is None:
        raise DocumentIndexError("No indexed documents were loaded.")
    query_vector = embed_query(question, expected_dimension)

    candidates: list[dict[str, object]] = []
    for metadata, collection in indexed_documents:
        document_id = str(metadata["document_id"])
        _, document_candidates = query_chroma_collection_with_vector(
            collection,
            query_vector,
            top_k=min(top_k, collection.count()),
            where={"document_id": document_id},
        )
        if any(
            result.get("document_id") != document_id
            for result in document_candidates
        ):
            raise DocumentIndexError(
                f"Chroma returned a chunk outside document {document_id}."
            )
        candidates.extend(document_candidates)

    if not candidates:
        raise DocumentIndexError("The selected Chroma indexes returned no chunks.")

    candidates.sort(key=lambda result: float(result["score"]), reverse=True)
    top_results = candidates[:top_k]
    for rank, result in enumerate(top_results, start=1):
        result["rank"] = rank

    return {
        "metadata": [metadata for metadata, _ in indexed_documents],
        "query_vector": query_vector,
        "candidate_count": len(candidates),
        "top_results": top_results,
    }
