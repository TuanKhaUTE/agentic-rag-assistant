import json

import numpy as np

from config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    KNOWLEDGE_DIR,
    STORAGE_DIR,
)
from src.chunker import chunk_documents
from src.document_loader import load_documents
from src.embedder import embed_texts, normalize_embeddings


def main() -> None:
    print("=== BUILDING EMBEDDING INDEX ===")

    print("1. Loading knowledge documents...")
    documents = load_documents(KNOWLEDGE_DIR)
    if not documents:
        raise ValueError(f"No .txt documents found in {KNOWLEDGE_DIR}")
    print(f"   Loaded documents: {len(documents)}")

    print("2. Chunking documents...")
    chunks = chunk_documents(
        documents,
        CHUNK_SIZE_WORDS,
        CHUNK_OVERLAP_WORDS,
    )
    if not chunks:
        raise ValueError("Document loading produced no chunks")
    print(f"   Created chunks: {len(chunks)}")

    print(f"3. Embedding chunk texts with {EMBEDDING_MODEL}...")
    chunk_texts = [str(chunk["text"]) for chunk in chunks]
    embeddings = embed_texts(chunk_texts)
    print(f"   Raw matrix shape: {embeddings.shape}")
    print(f"   Raw matrix data type: {embeddings.dtype}")

    print("4. Verifying chunk-to-row alignment...")
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Alignment failure: {len(chunks)} chunks but "
            f"{embeddings.shape[0]} embedding rows"
        )
    print(f"   PASS: {len(chunks)} chunks match {embeddings.shape[0]} rows")

    print("5. Applying L2 normalization...")
    normalized_embeddings = normalize_embeddings(embeddings)
    normalized_lengths = np.linalg.norm(normalized_embeddings, axis=1)
    print(
        "   Vector norm range after normalization: "
        f"{normalized_lengths.min():.6f} to {normalized_lengths.max():.6f}"
    )

    print("6. Saving chunk metadata and vector matrix...")
    metadata_fields = (
        "chunk_id",
        "source",
        "chunk_index",
        "text",
        "start_word",
        "end_word",
    )
    chunk_metadata = [
        {field: chunk[field] for field in metadata_fields}
        for chunk in chunks
    ]

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(
        json.dumps(chunk_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(EMBEDDINGS_PATH, normalized_embeddings, allow_pickle=False)

    print(f"   Saved metadata: {CHUNKS_PATH}")
    print(f"   Saved vectors:  {EMBEDDINGS_PATH}")
    print("=== INDEX BUILD COMPLETE ===")


if __name__ == "__main__":
    main()
