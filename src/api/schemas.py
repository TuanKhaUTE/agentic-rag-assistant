from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints


class ChatRequest(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ChatResponse(BaseModel):
    answer: str
    steps: int
    tools_used: list[str]


class DocumentMetadataResponse(BaseModel):
    document_id: str
    filename: str
    file_type: Literal["txt", "pdf", "docx"]
    characters: int
    words: int
    status: Literal["extracted", "indexed"]
    chunk_count: int | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None


class DocumentUploadResponse(DocumentMetadataResponse):
    text_preview: str


class DocumentIndexResponse(BaseModel):
    document_id: str
    filename: str
    status: Literal["indexed"]
    chunk_count: int
    embedding_model: str
    embedding_dimension: int


class DocumentQuestionRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RetrievedDocumentChunkResponse(BaseModel):
    rank: int
    score: float
    chunk_id: str
    source: str


class DocumentQuestionResponse(BaseModel):
    document_id: str
    filename: str
    answer: str
    top_k: int
    highest_similarity: float
    similarity_threshold: float
    threshold_is_heuristic: Literal[True]
    threshold_note: str
    retrieval_gate_passed: bool
    sufficient_context: bool
    generation_called: bool
    answerable: bool
    retrieved_chunks: list[RetrievedDocumentChunkResponse]
