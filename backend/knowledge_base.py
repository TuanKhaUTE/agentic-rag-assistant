import hashlib
import json

from backend.config import (
    ASSIGNED_KB_INDEX_VERSION,
    ASSIGNED_KB_PATH,
    ASSIGNED_KB_STATE_PATH,
    ASSIGNED_KB_TITLE,
)
from backend.documents.indexer import (
    DocumentIndexError,
    index_uploaded_document,
    load_uploaded_collection,
)
from backend.documents.storage import (
    DocumentNotFoundError,
    create_document,
    get_document_metadata,
)


class AssignedKnowledgeBaseError(ValueError):
    """Raised when the fixed midterm knowledge base is unavailable or stale."""


def _source_bytes_and_hash() -> tuple[bytes, str]:
    if not ASSIGNED_KB_PATH.is_file():
        raise AssignedKnowledgeBaseError(
            f"Assigned knowledge base is missing: {ASSIGNED_KB_PATH.name}"
        )
    source_bytes = ASSIGNED_KB_PATH.read_bytes()
    if not source_bytes:
        raise AssignedKnowledgeBaseError("Assigned knowledge base is empty.")
    return source_bytes, hashlib.sha256(source_bytes).hexdigest()


def _load_state() -> dict[str, str]:
    if not ASSIGNED_KB_STATE_PATH.is_file():
        raise AssignedKnowledgeBaseError(
            "The assigned knowledge base is not initialized. "
            "Run scripts/initialize_kb.py first."
        )
    try:
        state = json.loads(ASSIGNED_KB_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AssignedKnowledgeBaseError(
            "The assigned knowledge-base state is unreadable. Reinitialize it."
        ) from error
    if not isinstance(state, dict):
        raise AssignedKnowledgeBaseError(
            "The assigned knowledge-base state is invalid. Reinitialize it."
        )
    return {str(key): str(value) for key, value in state.items()}


def _write_state(document_id: str, source_hash: str) -> None:
    ASSIGNED_KB_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = ASSIGNED_KB_STATE_PATH.with_suffix(".json.tmp")
    state = {
        "document_id": document_id,
        "index_version": ASSIGNED_KB_INDEX_VERSION,
        "source_filename": ASSIGNED_KB_PATH.name,
        "source_sha256": source_hash,
    }
    try:
        temporary_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(ASSIGNED_KB_STATE_PATH)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _with_title(metadata: dict[str, object]) -> dict[str, object]:
    return {**metadata, "title": ASSIGNED_KB_TITLE}


def get_assigned_knowledge_base() -> dict[str, object]:
    """Load and validate the one knowledge base allowed by the midterm runtime."""
    _, current_hash = _source_bytes_and_hash()
    state = _load_state()
    if state.get("source_filename") != ASSIGNED_KB_PATH.name:
        raise AssignedKnowledgeBaseError(
            "The assigned knowledge-base state references the wrong source file."
        )
    if state.get("index_version") != str(ASSIGNED_KB_INDEX_VERSION):
        raise AssignedKnowledgeBaseError(
            "The assigned knowledge-base index format changed. Reinitialize it."
        )
    if state.get("source_sha256") != current_hash:
        raise AssignedKnowledgeBaseError(
            "The assigned handbook changed after indexing. Reinitialize it."
        )

    document_id = state.get("document_id", "")
    try:
        metadata = get_document_metadata(document_id)
        if metadata.get("status") != "indexed":
            raise AssignedKnowledgeBaseError(
                "The assigned knowledge base has not finished indexing."
            )
        load_uploaded_collection(document_id)
    except (DocumentNotFoundError, DocumentIndexError) as error:
        raise AssignedKnowledgeBaseError(
            "The assigned knowledge-base files are unavailable. Reinitialize it."
        ) from error
    return _with_title(metadata)


def initialize_assigned_knowledge_base() -> dict[str, object]:
    """Idempotently index the exact handbook assigned for the midterm."""
    source_bytes, current_hash = _source_bytes_and_hash()

    try:
        return get_assigned_knowledge_base()
    except AssignedKnowledgeBaseError:
        pass

    try:
        state = _load_state()
        document_id = state.get("document_id", "")
        if (
            state.get("source_sha256") == current_hash
            and state.get("index_version") == str(ASSIGNED_KB_INDEX_VERSION)
        ):
            metadata = get_document_metadata(document_id)
            if metadata.get("status") != "indexed":
                metadata = index_uploaded_document(document_id)
            else:
                load_uploaded_collection(document_id)
            _write_state(document_id, current_hash)
            return _with_title(metadata)
    except (
        AssignedKnowledgeBaseError,
        DocumentNotFoundError,
        DocumentIndexError,
    ):
        pass

    uploaded = create_document(source_bytes, ASSIGNED_KB_PATH.name)
    document_id = str(uploaded["document_id"])
    metadata = index_uploaded_document(document_id)
    _write_state(document_id, current_hash)
    return _with_title(metadata)
