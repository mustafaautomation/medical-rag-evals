"""Shared data types for documents, retrieval, and evaluation."""

from dataclasses import dataclass, field
from typing import Literal

ExpectedBehavior = Literal["answer", "refuse", "emergency", "unknown"]
VariantName = Literal["baseline", "bad-retriever", "no-guardrail"]


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    source_path: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    reference_answer: str
    reference_doc_ids: list[str]
    expected_behavior: ExpectedBehavior
    category: str
    required_phrases: list[str]
    forbidden_phrases: list[str] = field(default_factory=list)
    score_retrieval: bool = False


@dataclass(frozen=True)
class Answer:
    question_id: str
    question: str
    text: str
    retrieved_doc_ids: list[str]
    retrieved_contexts: list[str]
    behavior: str
    variant: str
