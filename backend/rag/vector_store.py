import math
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings

from backend.config import EMBEDDING_MODEL, TOP_K
from backend.rag.embedder import embed_texts, normalize_embeddings


COSINE_DISTANCE_METRIC = "cosine"


def create_chroma_client(persist_directory: Path) -> Any:
    """Open a local persistent Chroma database with telemetry disabled."""
    persist_directory.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_directory),
        settings=Settings(anonymized_telemetry=False),
    )


def _collection_names(client: Any) -> set[str]:
    return {
        collection if isinstance(collection, str) else collection.name
        for collection in client.list_collections()
    }


def _validate_embeddings(
    chunks: list[dict[str, object]], embeddings: np.ndarray
) -> None:
    if not chunks:
        raise ValueError("The index must contain at least one chunk.")
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, received {embeddings.shape}.")
    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            f"Index alignment failure: {len(chunks)} chunks but "
            f"{embeddings.shape[0]} embedding rows."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values.")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
        raise ValueError("Embedding vectors must be L2-normalized.")


def rebuild_chroma_collection(
    persist_directory: Path,
    collection_name: str,
    chunks: list[dict[str, object]],
    embeddings: np.ndarray,
    extra_collection_metadata: dict[str, str | int] | None = None,
) -> Any:
    """Replace one collection and store explicitly supplied chunk embeddings."""
    _validate_embeddings(chunks, embeddings)
    client = create_chroma_client(persist_directory)
    if collection_name in _collection_names(client):
        client.delete_collection(collection_name)

    collection_metadata: dict[str, str | int] = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": int(embeddings.shape[1]),
        "distance_metric": COSINE_DISTANCE_METRIC,
    }
    if extra_collection_metadata:
        collection_metadata.update(extra_collection_metadata)

    # embedding_function=None prevents Chroma from silently choosing a model.
    collection = client.create_collection(
        name=collection_name,
        configuration={"hnsw": {"space": COSINE_DISTANCE_METRIC}},
        metadata=collection_metadata,
        embedding_function=None,
    )

    ids = [str(chunk["chunk_id"]) for chunk in chunks]
    documents = [str(chunk["text"]) for chunk in chunks]
    metadatas: list[dict[str, str | int | float | bool]] = []
    for chunk in chunks:
        record: dict[str, str | int | float | bool] = {
            "chunk_id": str(chunk["chunk_id"]),
            "source": str(chunk["source"]),
            "chunk_index": int(chunk["chunk_index"]),
            "start_word": int(chunk["start_word"]),
            "end_word": int(chunk["end_word"]),
        }
        if "document_id" in chunk:
            record["document_id"] = str(chunk["document_id"])
        metadatas.append(record)

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )
    if collection.count() != len(chunks):
        raise ValueError("Chroma did not store exactly one record per chunk.")
    return collection


def load_chroma_collection(persist_directory: Path, collection_name: str) -> Any:
    """Load and validate a collection created by this project."""
    client = create_chroma_client(persist_directory)
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=None,
        )
    except Exception as error:
        raise ValueError("The Chroma index is unavailable. Build it first.") from error

    metadata = collection.metadata or {}
    if metadata.get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError("The Chroma index uses a different embedding model.")
    if metadata.get("distance_metric") != COSINE_DISTANCE_METRIC:
        raise ValueError("The Chroma index is not configured for cosine distance.")
    hnsw_configuration = (collection.configuration or {}).get("hnsw", {})
    if hnsw_configuration.get("space") != COSINE_DISTANCE_METRIC:
        raise ValueError("The Chroma collection configuration is not cosine.")
    if collection.count() <= 0:
        raise ValueError("The Chroma index contains no chunks.")
    return collection


def embed_query(query: str, expected_dimension: int) -> np.ndarray:
    """Embed and L2-normalize one query using the configured Ollama model."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    query_vector = embed_texts([query])[0]
    if query_vector.shape[0] != expected_dimension:
        raise ValueError(
            f"Query dimension {query_vector.shape[0]} does not match "
            f"stored dimension {expected_dimension}."
        )
    return normalize_embeddings(query_vector.reshape(1, -1))[0]


def query_chroma_collection_with_vector(
    collection: Any,
    query_vector: np.ndarray,
    top_k: int = TOP_K,
    where: dict[str, object] | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Query Chroma with one already embedded and normalized question vector."""
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    metadata = collection.metadata or {}
    expected_dimension = int(metadata.get("embedding_dimension", 0))
    if expected_dimension <= 0:
        raise ValueError("The Chroma index has no valid embedding dimension.")
    query_vector = np.asarray(query_vector, dtype=np.float32)
    if query_vector.ndim != 1 or query_vector.shape[0] != expected_dimension:
        raise ValueError(
            "Query vector dimension does not match the Chroma collection."
        )
    if not np.isfinite(query_vector).all():
        raise ValueError("Query vector contains non-finite values.")
    if not np.isclose(np.linalg.norm(query_vector), 1.0, atol=1e-5):
        raise ValueError("Query vector must be L2-normalized.")

    query_arguments: dict[str, object] = {
        "query_embeddings": [query_vector.tolist()],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        query_arguments["where"] = where
    response = collection.query(**query_arguments)

    ids = (response.get("ids") or [[]])[0]
    documents = (response.get("documents") or [[]])[0]
    metadatas = (response.get("metadatas") or [[]])[0]
    distances = (response.get("distances") or [[]])[0]
    if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
        raise ValueError("Chroma returned misaligned retrieval fields.")

    results: list[dict[str, object]] = []
    for rank, (chunk_id, text, chunk_metadata, distance_value) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        distance = float(distance_value)
        if not math.isfinite(distance):
            raise ValueError("Chroma returned a non-finite distance.")
        # For Chroma cosine space: distance = 1 - cosine similarity.
        similarity = 1.0 - distance
        item: dict[str, object] = {
            "rank": rank,
            "distance": distance,
            "similarity": similarity,
            # The service exposes score as the cosine similarity value.
            "score": similarity,
            "chunk_id": str(chunk_id),
            "source": str(chunk_metadata["source"]),
            "chunk_index": int(chunk_metadata["chunk_index"]),
            "text": str(text),
            "start_word": int(chunk_metadata["start_word"]),
            "end_word": int(chunk_metadata["end_word"]),
        }
        if "document_id" in chunk_metadata:
            item["document_id"] = str(chunk_metadata["document_id"])
        results.append(item)
    return query_vector, results


def get_collection_records(
    collection: Any, include_embeddings: bool = False
) -> list[dict[str, object]]:
    """Read records for inspection only; retrieval always uses Chroma query()."""
    include = ["documents", "metadatas"]
    if include_embeddings:
        include.append("embeddings")
    response = collection.get(include=include)
    ids = response.get("ids") or []
    documents = response.get("documents") or []
    metadatas = response.get("metadatas") or []
    embeddings = response.get("embeddings") if include_embeddings else None

    records: list[dict[str, object]] = []
    for index, chunk_id in enumerate(ids):
        record: dict[str, object] = {
            "chunk_id": str(chunk_id),
            "text": str(documents[index]),
            **dict(metadatas[index]),
        }
        if embeddings is not None:
            record["embedding"] = np.asarray(embeddings[index], dtype=np.float32)
        records.append(record)
    return records
