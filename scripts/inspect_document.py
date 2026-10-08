import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.documents import DocumentExtractionError, extract_text, supported_extension


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/inspect_document.py <path>")

    file_path = Path(sys.argv[1])
    try:
        extension = supported_extension(file_path.name)
        text = extract_text(file_path)
    except (DocumentExtractionError, OSError) as error:
        raise SystemExit(f"Extraction failed: {error}") from error

    print("=== DOCUMENT INSPECTION ===")
    print(f"Filename: {file_path.name}")
    print(f"Type: {extension.removeprefix('.')}")
    print(f"Characters: {len(text)}")
    print(f"Words: {len(text.split())}")
    print("\n=== FIRST 500 CHARACTERS ===")
    print(text[:500])


if __name__ == "__main__":
    main()
