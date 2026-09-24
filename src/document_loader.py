from pathlib import Path


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        title = line.strip().lstrip("#").strip()
        if title:
            return title
    return fallback


def load_documents(knowledge_dir: Path) -> list[dict[str, object]]:
    """Load every .txt file in deterministic filename order."""
    if not knowledge_dir.is_dir():
        raise FileNotFoundError(f"Knowledge directory not found: {knowledge_dir}")

    documents: list[dict[str, object]] = []
    for path in sorted(knowledge_dir.glob("*.txt"), key=lambda item: item.name.lower()):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        documents.append(
            {
                "path": str(path.resolve()),
                "source": path.name,
                "title": _extract_title(text, path.stem),
                "text": text,
                "word_count": len(text.split()),
            }
        )

    return documents
