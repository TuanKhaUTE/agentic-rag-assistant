import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

ASSIGNED_KB_PATH = (
    PROJECT_ROOT / "knowledge" / "KB-A_Lumen-Institute-Student-Handbook.md"
)
ASSIGNED_KB_TITLE = "Lumen Institute — Student Handbook"
ASSIGNED_KB_INDEX_VERSION = 2

UPLOADED_CHROMA_COLLECTION = "uploaded_document"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
ASSIGNED_KB_STATE_PATH = UPLOADS_DIR / "assigned_knowledge_base.json"
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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

TOP_K = 3
MIN_SIMILARITY = 0.30
