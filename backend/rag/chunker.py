def chunk_document(
    document: dict[str, object],
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> list[dict[str, object]]:
    """Split one document using 0-based, end-exclusive word positions."""
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than zero")
    if chunk_overlap_words < 0:
        raise ValueError("chunk_overlap_words cannot be negative")
    if chunk_overlap_words >= chunk_size_words:
        raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")

    source = str(document["source"])
    chunk_id_prefix = str(document.get("chunk_id_prefix", source))
    words = str(document["text"]).split()
    step = chunk_size_words - chunk_overlap_words
    chunks: list[dict[str, object]] = []
    start_word = 0

    while start_word < len(words):
        end_word = min(start_word + chunk_size_words, len(words))
        chunk_index = len(chunks) + 1
        chunk: dict[str, object] = {
            "chunk_id": f"{chunk_id_prefix}#chunk-{chunk_index:03d}",
            "source": source,
            "chunk_index": chunk_index,
            "text": " ".join(words[start_word:end_word]),
            "start_word": start_word,
            "end_word": end_word,
        }
        if "document_id" in document:
            chunk["document_id"] = str(document["document_id"])
        chunks.append(chunk)

        if end_word == len(words):
            break
        start_word += step

    return chunks


def chunk_documents(
    documents: list[dict[str, object]],
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for document in documents:
        chunks.extend(
            chunk_document(document, chunk_size_words, chunk_overlap_words)
        )
    return chunks
