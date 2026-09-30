import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
STATIC_CHROMA_COLLECTION = "static_knowledge"
UPLOADED_CHROMA_COLLECTION = "uploaded_document"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
TEXT_PREVIEW_CHARACTERS = 500

CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 30

EMBEDDING_MODEL = "all-minilm:latest"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
DEFAULT_QWEN_MODEL = "qwen3:4b-instruct"
OLLAMA_GENERATION_MODEL = os.getenv(
    "OLLAMA_GENERATION_MODEL", DEFAULT_QWEN_MODEL
).strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# The Agent remains Ollama/Qwen-only and retains its existing configuration.
GENERATION_MODEL = DEFAULT_QWEN_MODEL

TOP_K = 3
MIN_SIMILARITY = 0.30
