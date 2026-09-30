import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.agent import run_agent_with_metadata
from config import MAX_UPLOAD_BYTES
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentIndexResponse,
    DocumentMetadataResponse,
    DocumentQuestionRequest,
    DocumentQuestionResponse,
    DocumentUploadResponse,
)
from src.documents import (
    DocumentExtractionError,
    DocumentNotFoundError,
    EmptyUploadError,
    UnsupportedDocumentTypeError,
    UploadTooLargeError,
    create_document,
    get_document_metadata,
    supported_extension,
)
from src.documents.indexer import DocumentIndexError, index_uploaded_document
from src.documents.question_answering import answer_uploaded_document
from src.documents.retriever import DocumentNotIndexedError


logger = logging.getLogger(__name__)

app = FastAPI(title="AI Homework Lab API")

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


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = run_agent_with_metadata(request.message)
    except Exception as error:
        logger.exception("Chat request failed: %s", type(error).__name__)
        raise HTTPException(
            status_code=500,
            detail="The chat request could not be completed.",
        ) from error

    return ChatResponse(
        answer=result.answer,
        steps=result.steps,
        tools_used=result.tools_used,
    )


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


@app.post(
    "/documents/{document_id}/ask",
    response_model=DocumentQuestionResponse,
)
def ask_indexed_document(
    document_id: str,
    request: DocumentQuestionRequest,
) -> DocumentQuestionResponse:
    try:
        result = answer_uploaded_document(document_id, request.question)
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except DocumentNotIndexedError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except DocumentIndexError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:
        logger.exception("Uploaded-document question failed: %s", type(error).__name__)
        raise HTTPException(
            status_code=500,
            detail="The document question could not be completed.",
        ) from error

    metadata = result["metadata"]
    retrieved_chunks = [
        {
            "rank": chunk["rank"],
            "score": chunk["score"],
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
        }
        for chunk in result["top_results"]
    ]
    return DocumentQuestionResponse(
        document_id=document_id,
        filename=str(metadata["filename"]),
        answer=str(result["answer"]),
        top_k=len(retrieved_chunks),
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
