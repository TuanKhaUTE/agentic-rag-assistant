import numpy as np

from config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    KNOWLEDGE_DIR,
    STATIC_CHROMA_COLLECTION,
)
from src.chunker import chunk_documents
from src.document_loader import load_documents
from src.embedder import embed_texts, normalize_embeddings
from src.vector_store import rebuild_chroma_collection


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

    print("6. Rebuilding the persistent Chroma collection...")
    collection = rebuild_chroma_collection(
        CHROMA_DIR,
        STATIC_CHROMA_COLLECTION,
        chunks,
        normalized_embeddings,
        {"index_kind": "static_knowledge"},
    )
    print(f"   Document directory: {KNOWLEDGE_DIR}")
    print(f"   Embedding model: {EMBEDDING_MODEL}")
    print(f"   Chunk count: {collection.count()}")
    print(f"   Chroma collection: {collection.name}")
    print(f"   Local Chroma persistence: {CHROMA_DIR}")
    print("   Stored chunk IDs:")
    for chunk_id in collection.get(include=[])["ids"]:
        print(f"     - {chunk_id}")
    print("=== INDEX BUILD COMPLETE ===")


if __name__ == "__main__":
    main()
