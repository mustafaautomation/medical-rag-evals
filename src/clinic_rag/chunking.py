"""Paragraph chunking with stable ids.

Handouts in this demo are short on purpose. The default size keeps each
handout as one chunk so reference ids stay stable.
"""

from clinic_rag.types import Chunk, Document


def chunk_document(document: Document, max_chars: int = 4000) -> list[Chunk]:
    if max_chars < 200:
        raise ValueError("max_chars must be at least 200")
    paragraphs = [part.strip() for part in document.text.split("\n\n") if part.strip()]
    if not paragraphs:
        paragraphs = [document.text.strip()]
    chunks: list[Chunk] = []
    buffer: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        extra = len(paragraph) + (2 if buffer else 0)
        if buffer and current_len + extra > max_chars:
            chunks.append(_make_chunk(document, len(chunks), buffer))
            buffer = [paragraph]
            current_len = len(paragraph)
        else:
            buffer.append(paragraph)
            current_len += extra
    if buffer:
        chunks.append(_make_chunk(document, len(chunks), buffer))
    return chunks


def chunk_documents(documents: list[Document], max_chars: int = 4000) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, max_chars=max_chars))
    return chunks


def _make_chunk(document: Document, index: int, paragraphs: list[str]) -> Chunk:
    body = "\n\n".join(paragraphs)
    text = f"{document.title}\n\n{body}"
    return Chunk(
        chunk_id=f"{document.doc_id}-{index}",
        doc_id=document.doc_id,
        title=document.title,
        text=text,
    )
