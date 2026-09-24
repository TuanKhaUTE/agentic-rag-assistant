from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
STORAGE_DIR = PROJECT_ROOT / "storage"

CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 30
