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
    status: Literal["extracted"]


class DocumentUploadResponse(DocumentMetadataResponse):
    text_preview: str
