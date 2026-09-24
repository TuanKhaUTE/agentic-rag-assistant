from config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS, KNOWLEDGE_DIR
from src.chunker import chunk_document
from src.document_loader import load_documents


def main() -> None:
    documents = load_documents(KNOWLEDGE_DIR)
    total_chunks = 0

    print("=== PHASE 1A: DOCUMENT LOADING AND CHUNKING ===")
    print(f"Knowledge directory: {KNOWLEDGE_DIR.resolve()}")
    print(f"Chunk size: {CHUNK_SIZE_WORDS} words")
    print(f"Configured overlap: {CHUNK_OVERLAP_WORDS} words")
    print(f"Chunk start step: {CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS} words")

    for document_number, document in enumerate(documents, start=1):
        print("\n" + "=" * 80)
        print(f"DOCUMENT {document_number}")
        print(f"Path: {document['path']}")
        print(f"Title: {document['title']}")
        print(f"Word count: {document['word_count']}")

        chunks = chunk_document(
            document,
            CHUNK_SIZE_WORDS,
            CHUNK_OVERLAP_WORDS,
        )
        previous_end = 0

        for chunk in chunks:
            overlap = max(0, previous_end - int(chunk["start_word"]))
            print("\n" + "-" * 80)
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Source: {chunk['source']}")
            print(f"Chunk index: {chunk['chunk_index']}")
            print(
                "Word range: "
                f"[{chunk['start_word']}, {chunk['end_word']}) "
                "(0-based, end-exclusive)"
            )
            print(f"Words in chunk: {int(chunk['end_word']) - int(chunk['start_word'])}")
            print(f"Overlap with previous chunk: {overlap} words")
            print("Full chunk text:")
            print(chunk["text"])
            previous_end = int(chunk["end_word"])

        total_chunks += len(chunks)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"Total documents: {len(documents)}")
    print(f"Total chunks: {total_chunks}")


if __name__ == "__main__":
    main()
