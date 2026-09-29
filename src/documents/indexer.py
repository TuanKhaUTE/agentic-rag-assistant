import json
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np

from config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS, EMBEDDING_MODEL
from src.chunker import chunk_document
from src.documents.extractor import extract_text
from src.documents.storage import (
    get_document_directory,
    get_document_metadata,
    get_original_file_path,
    update_document_metadata,
)
from src.embedder import embed_texts, normalize_embeddings


class DocumentIndexError(ValueError):
    pass


def _remove_generated_directory(path: Path, document_directory: Path) -> None:
    if not path.exists():
        return
    resolved_path = path.resolve()
    if resolved_path.parent != document_directory.resolve():
        raise DocumentIndexError("Refusing to remove an unsafe index path.")
    shutil.rmtree(resolved_path)


def index_uploaded_document(document_id: str) -> dict[str, object]:
    metadata = get_document_metadata(document_id)
    document_directory = get_document_directory(document_id)
    original_path = get_original_file_path(document_id)
    text = extract_text(original_path)

    document = {
        "document_id": document_id,
        "chunk_id_prefix": document_id,
        "source": str(metadata["filename"]),
        "text": text,
    }
    chunks = chunk_document(
        document,
        CHUNK_SIZE_WORDS,
        CHUNK_OVERLAP_WORDS,
    )
    if not chunks:
        raise DocumentIndexError("Document extraction produced no chunks.")

    chunk_texts = [str(chunk["text"]) for chunk in chunks]
    embeddings = normalize_embeddings(embed_texts(chunk_texts))
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise DocumentIndexError(
            f"Alignment failure: {len(chunks)} chunks but "
            f"embedding shape {embeddings.shape}."
        )

    vector_norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(vector_norms, 1.0, atol=1e-5):
        raise DocumentIndexError("Stored vectors must be L2-normalized.")

    chunk_fields = (
        "chunk_id",
        "document_id",
        "source",
        "chunk_index",
        "text",
        "start_word",
        "end_word",
    )
    chunk_metadata = [
        {field: chunk[field] for field in chunk_fields}
        for chunk in chunks
    ]

    index_directory = document_directory / "index"
    build_directory = document_directory / f".index-build-{uuid4().hex}"
    backup_directory = document_directory / f".index-backup-{uuid4().hex}"
    build_directory.mkdir()

    index_installed = False
    previous_index_backed_up = False
    try:
        (build_directory / "chunks.json").write_text(
            json.dumps(chunk_metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        np.save(
            build_directory / "embeddings.npy",
            embeddings,
            allow_pickle=False,
        )

        if index_directory.exists():
            index_directory.rename(backup_directory)
            previous_index_backed_up = True
        build_directory.rename(index_directory)
        index_installed = True

        updated_metadata = update_document_metadata(
            document_id,
            {
                "status": "indexed",
                "chunk_count": len(chunk_metadata),
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimension": int(embeddings.shape[1]),
            },
        )
    except Exception:
        if index_installed and index_directory.exists():
            _remove_generated_directory(index_directory, document_directory)
        if previous_index_backed_up and backup_directory.exists():
            backup_directory.rename(index_directory)
        raise
    finally:
        if build_directory.exists():
            _remove_generated_directory(build_directory, document_directory)

    if backup_directory.exists():
        _remove_generated_directory(backup_directory, document_directory)

    return updated_metadata


def load_uploaded_index(
    document_id: str,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    metadata = get_document_metadata(document_id)
    index_directory = get_document_directory(document_id) / "index"
    chunks_path = index_directory / "chunks.json"
    embeddings_path = index_directory / "embeddings.npy"
    if not chunks_path.is_file() or not embeddings_path.is_file():
        raise DocumentIndexError("Document index is unavailable.")

    try:
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        embeddings = np.load(embeddings_path, allow_pickle=False)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise DocumentIndexError("Document index could not be read.") from error

    if not isinstance(chunks, list):
        raise DocumentIndexError("Chunk metadata must be a list.")
    if embeddings.ndim != 2 or len(chunks) != embeddings.shape[0]:
        raise DocumentIndexError("Chunk metadata and embedding rows are not aligned.")
    return metadata, chunks, embeddings
