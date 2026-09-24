from pathlib import Path

from config import PROJECT_ROOT


LIST_FILES_SCHEMA = {
    "name": "list_files",
    "description": "List files in a directory inside the project.",
    "parameters": {
        "type": "object",
        "properties": {
            "relative_directory": {
                "type": "string",
                "description": "Project-relative directory, such as knowledge.",
                "default": ".",
            }
        },
        "additionalProperties": False,
    },
}

READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": "Read a UTF-8 text file inside the project.",
    "parameters": {
        "type": "object",
        "properties": {
            "relative_path": {
                "type": "string",
                "description": "Project-relative path to an approved text file.",
            }
        },
        "required": ["relative_path"],
        "additionalProperties": False,
    },
}


_PROJECT_ROOT = PROJECT_ROOT.resolve()
_MAX_TEXT_FILE_BYTES = 1_000_000
_TEXT_EXTENSIONS = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}


def _resolve_project_path(relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("Path must be a non-empty project-relative string")

    requested_path = Path(relative_path)
    if requested_path.is_absolute() or requested_path.drive:
        raise ValueError("Absolute paths are not allowed")
    if ".." in requested_path.parts:
        raise ValueError("Parent-directory traversal is not allowed")

    resolved_path = (_PROJECT_ROOT / requested_path).resolve()
    try:
        resolved_path.relative_to(_PROJECT_ROOT)
    except ValueError as error:
        raise ValueError("Path traversal outside the project is not allowed") from error

    return resolved_path


def list_files(relative_directory: str = ".") -> list[str]:
    """List direct child files and return project-relative paths."""
    directory = _resolve_project_path(relative_directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {relative_directory}")
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {relative_directory}")

    return [
        path.relative_to(_PROJECT_ROOT).as_posix()
        for path in sorted(directory.iterdir(), key=lambda item: item.name.lower())
        if path.is_file()
    ]


def read_file(relative_path: str) -> str:
    """Read one size-limited UTF-8 text file inside the project."""
    path = _resolve_project_path(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {relative_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {relative_path}")
    if path.suffix.lower() not in _TEXT_EXTENSIONS:
        raise ValueError(f"Unsupported text-file extension: {path.suffix or '(none)'}")

    size_bytes = path.stat().st_size
    if size_bytes > _MAX_TEXT_FILE_BYTES:
        raise ValueError(
            f"File is too large: {size_bytes} bytes; "
            f"maximum is {_MAX_TEXT_FILE_BYTES} bytes"
        )

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("File is not valid UTF-8 text") from error
