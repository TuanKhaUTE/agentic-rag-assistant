import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import MAX_UPLOAD_BYTES
from backend.documents import (
    DocumentExtractionError,
    DocumentNotFoundError,
    EmptyUploadError,
    UnsupportedDocumentTypeError,
    UploadTooLargeError,
    create_document,
    get_document_metadata,
    supported_extension,
)
from backend.documents.indexer import DocumentIndexError, index_uploaded_document
from backend.rag.generator import GenerationConfigurationError
from backend.rag.retriever import DocumentNotIndexedError
from backend.rag.service import answer_question
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentIndexResponse,
    DocumentMetadataResponse,
    DocumentUploadResponse,
)


logger = logging.getLogger(__name__)

app = FastAPI(title="Document-Grounded RAG Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    response_model_exclude_none=True,
)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    filename = file.filename or ""
    try:
        supported_extension(filename)
        file_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        metadata = create_document(file_bytes, filename)
    except UnsupportedDocumentTypeError as error:
        raise HTTPException(status_code=415, detail=str(error)) from error
    except EmptyUploadError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except UploadTooLargeError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except DocumentExtractionError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Document upload failed: %s", type(error).__name__)
        raise HTTPException(
            status_code=500,
            detail="The document could not be uploaded.",
        ) from error
    finally:
        await file.close()
    return DocumentUploadResponse(**metadata)


@app.get(
    "/documents/{document_id}",
    response_model=DocumentMetadataResponse,
    response_model_exclude_none=True,
)
def document_metadata(document_id: str) -> DocumentMetadataResponse:
    try:
        metadata = get_document_metadata(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        logger.exception("Document metadata lookup failed: %s", type(error).__name__)
        raise HTTPException(
            status_code=500,
            detail="Document metadata could not be retrieved.",
        ) from error
    return DocumentMetadataResponse(**metadata)


@app.post(
    "/documents/{document_id}/index",
    response_model=DocumentIndexResponse,
)
def index_document(document_id: str) -> DocumentIndexResponse:
    try:
        metadata = index_uploaded_document(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except DocumentExtractionError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except DocumentIndexError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:
        logger.exception("Document indexing failed: %s", type(error).__name__)
        raise HTTPException(
            status_code=500,
            detail="The document could not be indexed.",
        ) from error
    return DocumentIndexResponse(**metadata)


def _retrieved_chunk_responses(result: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "rank": chunk["rank"],
            "score": chunk["score"],
            "similarity": chunk["similarity"],
            "distance": chunk["distance"],
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "document_id": chunk["document_id"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in result["top_results"]
    ]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer from selected uploaded documents; no general-LLM fallback exists."""
    try:
        result = answer_question(request.document_ids, request.question)
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except DocumentNotIndexedError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except GenerationConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except DocumentIndexError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Document-grounded chat failed: %s", type(error).__name__)
        raise HTTPException(
            status_code=500,
            detail="The document question could not be completed.",
        ) from error

    metadata = result["metadata"]
    retrieved_chunks = _retrieved_chunk_responses(result)
    query_vector = result["query_vector"]
    return ChatResponse(
        document_ids=[str(item["document_id"]) for item in metadata],
        filenames=[str(item["filename"]) for item in metadata],
        document_count=len(metadata),
        answer=str(result["answer"]),
        top_k=len(retrieved_chunks),
        candidate_count=int(result["candidate_count"]),
        query_embedding_dimension=int(query_vector.shape[0]),
        highest_similarity=float(result["highest_similarity"]),
        similarity_threshold=float(result["similarity_threshold"]),
        threshold_is_heuristic=True,
        threshold_note=str(result["threshold_note"]),
        retrieval_gate_passed=bool(result["retrieval_gate_passed"]),
        sufficient_context=bool(result["sufficient_context"]),
        generation_called=bool(result["generation_called"]),
        answerable=bool(result["answerable"]),
        retrieved_chunks=retrieved_chunks,
    )
