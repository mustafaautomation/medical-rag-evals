"""Labelled questions for the clinic assistant."""

import json
from typing import cast

from clinic_rag.paths import questions_path
from clinic_rag.types import ExpectedBehavior, Question

_BEHAVIORS = {"answer", "refuse", "emergency", "unknown"}


def load_questions(path: object | None = None) -> list[Question]:
    source = questions_path() if path is None else path
    questions: list[Question] = []
    text = source.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        behavior = str(payload["expected_behavior"])
        if behavior not in _BEHAVIORS:
            raise ValueError(f"Line {line_number} has unknown behavior {behavior!r}")
        question = Question(
            id=str(payload["id"]),
            question=str(payload["question"]),
            reference_answer=str(payload["reference_answer"]),
            reference_doc_ids=[str(item) for item in payload.get("reference_doc_ids", [])],
            expected_behavior=cast(ExpectedBehavior, behavior),
            category=str(payload["category"]),
            required_phrases=[str(item) for item in payload["required_phrases"]],
            forbidden_phrases=[str(item) for item in payload.get("forbidden_phrases", [])],
            score_retrieval=bool(payload.get("score_retrieval", False)),
        )
        if not question.required_phrases:
            raise ValueError(f"{question.id} needs at least one required phrase")
        if question.score_retrieval and not question.reference_doc_ids:
            raise ValueError(f"{question.id} scores retrieval but has no reference docs")
        questions.append(question)
    ids = [item.id for item in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Question ids must be unique")
    return questions
