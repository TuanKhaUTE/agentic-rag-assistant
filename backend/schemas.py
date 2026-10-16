from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class DocumentMetadataResponse(BaseModel):
    document_id: str
    filename: str
    file_type: Literal["txt", "md", "pdf", "docx"]
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


class AssignedKnowledgeBaseResponse(DocumentMetadataResponse):
    title: str


class ChatRequest(BaseModel):
    question: NonEmptyText
    document_ids: list[str] = Field(min_length=1)


class RetrievedChunkResponse(BaseModel):
    rank: int
    score: float
    similarity: float
    distance: float
    chunk_id: str
    source: str
    document_id: str
    chunk_index: int
    text: str


class ChatResponse(BaseModel):
    document_ids: list[str]
    filenames: list[str]
    document_count: int
    answer: str
    top_k: int
    candidate_count: int
    query_embedding_dimension: int
    highest_similarity: float
    similarity_threshold: float
    threshold_is_heuristic: Literal[True]
    threshold_note: str
    retrieval_gate_passed: bool
    sufficient_context: bool
    generation_called: bool
    answerable: bool
    retrieved_chunks: list[RetrievedChunkResponse]
