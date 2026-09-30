import json
import re
from pathlib import Path, PurePosixPath
from uuid import uuid4

from config import MAX_UPLOAD_BYTES, TEXT_PREVIEW_CHARACTERS, UPLOADS_DIR
from src.documents.extractor import extract_text, supported_extension


DOCUMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class EmptyUploadError(ValueError):
    pass


class UploadTooLargeError(ValueError):
    pass


class DocumentNotFoundError(LookupError):
    pass


def _safe_metadata_filename(filename: str) -> str:
    normalized = filename.replace("\\", "/")
    basename = PurePosixPath(normalized).name.strip()
    if not basename or basename in {".", ".."}:
        raise ValueError("A valid filename is required.")
    return basename


def _document_directory(document_id: str) -> Path:
    if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
        raise DocumentNotFoundError("Document not found.")
    return UPLOADS_DIR / document_id


def _discard_incomplete_document(document_directory: Path) -> None:
    if not document_directory.exists():
        return
    for child in document_directory.iterdir():
        if child.is_file():
            child.unlink()
    document_directory.rmdir()


def create_document(file_bytes: bytes, original_filename: str) -> dict[str, object]:
    if not file_bytes:
        raise EmptyUploadError("The uploaded file is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise UploadTooLargeError(
            f"The uploaded file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )

    filename = _safe_metadata_filename(original_filename)
    extension = supported_extension(filename)
    document_id = uuid4().hex
    document_directory = UPLOADS_DIR / document_id
    stored_path = document_directory / f"original{extension}"

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    document_directory.mkdir()

    try:
        stored_path.write_bytes(file_bytes)
        text = extract_text(stored_path)
        metadata: dict[str, object] = {
            "document_id": document_id,
            "filename": filename,
            "file_type": extension.removeprefix("."),
            "characters": len(text),
            "words": len(text.split()),
            "status": "extracted",
        }
        metadata_path = document_directory / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        _discard_incomplete_document(document_directory)
        raise

    return {
        **metadata,
        "text_preview": text[:TEXT_PREVIEW_CHARACTERS],
    }


def get_document_metadata(document_id: str) -> dict[str, object]:
    metadata_path = _document_directory(document_id) / "metadata.json"
    if not metadata_path.is_file():
        raise DocumentNotFoundError("Document not found.")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DocumentNotFoundError("Document metadata is unavailable.") from error

    return metadata


def get_document_directory(document_id: str) -> Path:
    document_directory = _document_directory(document_id)
    if not (document_directory / "metadata.json").is_file():
        raise DocumentNotFoundError("Document not found.")
    return document_directory


def get_original_file_path(document_id: str) -> Path:
    metadata = get_document_metadata(document_id)
    file_type = str(metadata.get("file_type", ""))
    if file_type not in {"txt", "md", "pdf", "docx"}:
        raise DocumentNotFoundError("Stored document type is invalid.")

    original_path = get_document_directory(document_id) / f"original.{file_type}"
    if not original_path.is_file():
        raise DocumentNotFoundError("Stored original document is unavailable.")
    return original_path


def update_document_metadata(
    document_id: str,
    updates: dict[str, object],
) -> dict[str, object]:
    metadata = get_document_metadata(document_id)
    metadata.update(updates)

    metadata_path = get_document_directory(document_id) / "metadata.json"
    temporary_path = metadata_path.with_name("metadata.json.tmp")
    try:
        temporary_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(metadata_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return metadata
