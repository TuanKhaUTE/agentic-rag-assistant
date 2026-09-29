import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.agent import run_agent_with_metadata
from src.api.schemas import ChatRequest, ChatResponse


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
