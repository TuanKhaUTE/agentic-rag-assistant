from pathlib import Path
from typing import Any

from backend.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    EMBEDDING_MODEL,
    UPLOADED_CHROMA_COLLECTION,
)
from backend.documents.extractor import extract_text
from backend.documents.storage import (
    get_document_directory,
    get_document_metadata,
    get_original_file_path,
    update_document_metadata,
)
from backend.rag.chunker import chunk_document
from backend.rag.embedder import embed_texts, normalize_embeddings
from backend.rag.vector_store import load_chroma_collection, rebuild_chroma_collection


class DocumentIndexError(ValueError):
    pass


def uploaded_chroma_directory(document_id: str) -> Path:
    return get_document_directory(document_id) / "chroma"


def index_uploaded_document(document_id: str) -> dict[str, object]:
    """Extract, chunk, embed, and rebuild one isolated local Chroma index."""
    metadata = get_document_metadata(document_id)
    original_path = get_original_file_path(document_id)
    text = extract_text(original_path)

    document = {
        "document_id": document_id,
        "chunk_id_prefix": document_id,
        "source": str(metadata["filename"]),
        "text": text,
    }
    chunks = chunk_document(document, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
    if not chunks:
        raise DocumentIndexError("Document extraction produced no chunks.")

    try:
        embeddings = normalize_embeddings(
            embed_texts([str(chunk["text"]) for chunk in chunks])
        )
        collection = rebuild_chroma_collection(
            uploaded_chroma_directory(document_id),
            UPLOADED_CHROMA_COLLECTION,
            chunks,
            embeddings,
            {
                "index_kind": "uploaded_document",
                "document_id": document_id,
            },
        )
    except (OSError, ValueError) as error:
        raise DocumentIndexError(str(error)) from error

    return update_document_metadata(
        document_id,
        {
            "status": "indexed",
            "chunk_count": collection.count(),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": int(embeddings.shape[1]),
            "vector_database": "Chroma",
            "chroma_collection": UPLOADED_CHROMA_COLLECTION,
        },
    )


def load_uploaded_collection(
    document_id: str,
) -> tuple[dict[str, object], Any]:
    metadata = get_document_metadata(document_id)
    try:
        collection = load_chroma_collection(
            uploaded_chroma_directory(document_id),
            UPLOADED_CHROMA_COLLECTION,
        )
    except (OSError, ValueError) as error:
        raise DocumentIndexError("Document Chroma index is unavailable.") from error

    collection_metadata = collection.metadata or {}
    if collection_metadata.get("document_id") != document_id:
        raise DocumentIndexError("Document index identity does not match.")
    if int(metadata.get("chunk_count", -1)) != collection.count():
        raise DocumentIndexError("Document metadata and Chroma chunk count differ.")
    return metadata, collection
