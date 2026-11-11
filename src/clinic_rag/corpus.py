"""Load the synthetic clinic handouts."""

from pathlib import Path

from clinic_rag.paths import corpus_dir
from clinic_rag.types import Document


def _parse_frontmatter(raw: str, path: Path) -> Document:
    if not raw.startswith("---\n"):
        raise ValueError(f"{path} is missing frontmatter")
    end = raw.find("\n---\n", 4)
    if end < 0:
        raise ValueError(f"{path} frontmatter is not closed")
    header = raw[4:end]
    body = raw[end + 5 :].strip()
    fields: dict[str, str] = {}
    for line in header.splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    doc_id = fields.get("id", "")
    title = fields.get("title", "")
    if not doc_id or not title or not body:
        raise ValueError(f"{path} needs id, title, and a body")
    return Document(doc_id=doc_id, title=title, text=body, source_path=str(path))


def load_documents(directory: Path | None = None) -> list[Document]:
    root = directory or corpus_dir()
    paths = sorted(root.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No markdown documents in {root}")
    documents = [_parse_frontmatter(path.read_text(encoding="utf-8"), path) for path in paths]
    ids = [doc.doc_id for doc in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Document ids must be unique")
    return documents
