from config import MIN_SIMILARITY, TOP_K
from src.vector_store import search


SEARCH_KNOWLEDGE_SCHEMA = {
    "name": "search_knowledge",
    "description": "Retrieve relevant chunks from the local study knowledge base.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question or topic to search for.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of highest-scoring chunks to return.",
                "default": TOP_K,
                "minimum": 1,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def search_knowledge(query: str, top_k: int = TOP_K) -> dict[str, object]:
    """Retrieve structured chunk results without generating an answer."""
    _, _, top_results = search(query, top_k=top_k)
    results = [
        {
            "rank": result["rank"],
            "score": result["score"],
            "chunk_id": result["chunk_id"],
            "source": result["source"],
            "text": result["text"],
        }
        for result in top_results
    ]
    highest_similarity = float(results[0]["score"])

    return {
        "highest_similarity": highest_similarity,
        "sufficient_context": highest_similarity >= MIN_SIMILARITY,
        "results": results,
    }
