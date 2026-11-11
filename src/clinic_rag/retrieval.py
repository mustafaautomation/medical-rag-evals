"""FAISS and pgvector retrieval with the same hybrid ranking."""

from typing import Any, Protocol, cast

import numpy as np

from clinic_rag.embeddings import TfidfEmbedder, tokenize
from clinic_rag.types import Chunk, ScoredChunk


class HybridRetriever(Protocol):
    chunks: list[Chunk]

    def search(self, question: str, k: int = 4, *, worst: bool = False) -> list[ScoredChunk]:
        """Return the best chunks, or the worst chunks when worst is true."""
        ...


LEXICAL_WEIGHT = 0.45
EMBEDDING_WEIGHT = 0.55


def lexical_overlap(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    hits = sum(1 for token in query_tokens if token in text_tokens)
    return hits / len(query_tokens)


def hybrid_score(cosine: float, lexical: float) -> float:
    return (EMBEDDING_WEIGHT * cosine) + (LEXICAL_WEIGHT * lexical)


class FaissHybridRetriever:
    """Inner-product FAISS search over the full handout index, then hybrid re-rank.

    The corpus is small, so every chunk is a FAISS candidate. The final order
    mixes cosine similarity with token overlap. Both backends share that mix.
    """

    def __init__(self, chunks: list[Chunk], embedder: TfidfEmbedder) -> None:
        import faiss

        if embedder.dimension == 0:
            raise ValueError("Embedder is not fit")
        self.chunks = list(chunks)
        self.embedder = embedder
        matrix = embedder.embed_many([chunk.text for chunk in chunks]).astype(np.float32)
        if matrix.shape != (len(chunks), embedder.dimension):
            raise ValueError("Embedding matrix shape does not match the chunk list")
        self._matrix = np.ascontiguousarray(matrix)
        index = faiss.IndexFlatIP(embedder.dimension)
        index.add(self._matrix)
        self.index = index

    def search(self, question: str, k: int = 4, *, worst: bool = False) -> list[ScoredChunk]:
        if k < 1:
            raise ValueError("k must be at least 1")
        query = self.embedder.embed(question).astype(np.float32).reshape(1, -1)
        if float(np.linalg.norm(query)) == 0.0:
            return []
        total = len(self.chunks)
        _scores, indices = self.index.search(np.ascontiguousarray(query), total)
        scored: list[ScoredChunk] = []
        seen: set[int] = set()
        for raw_index in indices[0]:
            position = int(raw_index)
            if position < 0 or position in seen or position >= total:
                continue
            seen.add(position)
            chunk = self.chunks[position]
            cosine = float(np.dot(query[0], self._matrix[position]))
            lexical = lexical_overlap(question, chunk.text)
            scored.append(ScoredChunk(chunk=chunk, score=hybrid_score(cosine, lexical)))
        scored.sort(key=lambda item: item.score, reverse=not worst)
        return scored[:k]


def connect_pgvector(dsn: str | None = None) -> object:
    """Open the local docker-compose Postgres. The password is for local use only."""

    import os

    import psycopg

    url = dsn or os.environ.get(
        "DATABASE_URL",
        "postgresql://clinic:clinic@localhost:5432/clinic",
    )
    return psycopg.connect(url)


def pgvector_schema_sql(dimension: int) -> str:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS clinic_chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector({dimension}) NOT NULL
);
""".strip()


class PgVectorRetriever:
    """Postgres + pgvector candidate lookup, then the same hybrid re-rank as FAISS.

    The connection is opened by the caller. Tests that do not have Docker use
    FaissHybridRetriever instead.
    """

    def __init__(self, connection: object, chunks: list[Chunk], embedder: TfidfEmbedder) -> None:
        self.connection = connection
        self.chunks = list(chunks)
        self.embedder = embedder
        self._by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def rebuild(self) -> None:
        from pgvector.psycopg import register_vector

        register_vector(cast(Any, self.connection))
        cursor = self.connection.cursor()  # type: ignore[attr-defined]
        cursor.execute(pgvector_schema_sql(self.embedder.dimension))
        cursor.execute("TRUNCATE clinic_chunks")
        for chunk in self.chunks:
            vector = self.embedder.embed(chunk.text)
            cursor.execute(
                "INSERT INTO clinic_chunks (chunk_id, doc_id, title, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s)",
                (chunk.chunk_id, chunk.doc_id, chunk.title, chunk.text, vector.tolist()),
            )
        self.connection.commit()  # type: ignore[attr-defined]

    def search(self, question: str, k: int = 4, *, worst: bool = False) -> list[ScoredChunk]:
        query = self.embedder.embed(question)
        if float(np.linalg.norm(query)) == 0.0:
            return []
        cursor = self.connection.cursor()  # type: ignore[attr-defined]
        cursor.execute(
            "SELECT chunk_id, embedding <=> %s::vector AS distance "
            "FROM clinic_chunks "
            "ORDER BY distance ASC",
            (query.tolist(),),
        )
        rows = cursor.fetchall()
        scored: list[ScoredChunk] = []
        for chunk_id, distance in rows:
            chunk = self._by_id[str(chunk_id)]
            cosine = 1.0 - float(distance)
            lexical = lexical_overlap(question, chunk.text)
            scored.append(ScoredChunk(chunk=chunk, score=hybrid_score(cosine, lexical)))
        scored.sort(key=lambda item: item.score, reverse=not worst)
        return scored[:k]
