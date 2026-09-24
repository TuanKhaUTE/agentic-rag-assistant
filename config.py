from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
STORAGE_DIR = PROJECT_ROOT / "storage"
CHUNKS_PATH = STORAGE_DIR / "chunks.json"
EMBEDDINGS_PATH = STORAGE_DIR / "embeddings.npy"

CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 30

EMBEDDING_MODEL = "all-minilm:latest"
