"""Recorded pipeline outputs for the offline gate.

The snapshot is produced by the deterministic fake provider. It stores
answers and retrieved handouts. It does not store LLM-judge scores.
"""

import json
from collections.abc import Sequence
from pathlib import Path

from clinic_rag.paths import snapshot_path
from clinic_rag.pipeline import Pipeline
from clinic_rag.types import Answer, Question


def answer_to_record(question: Question, answer: Answer) -> dict[str, object]:
    return {
        "id": question.id,
        "question": question.question,
        "reference_answer": question.reference_answer,
        "response": answer.text,
        "retrieved_contexts": answer.retrieved_contexts,
        "retrieved_doc_ids": answer.retrieved_doc_ids,
        "behavior": answer.behavior,
        "variant": answer.variant,
        "provider": "fake",
        "judge_scores": None,
        "note": (
            "Recorded from the deterministic offline pipeline. "
            "judge_scores is null because no live RAGAS judge was run for this snapshot."
        ),
    }


def record_pipeline(pipeline: Pipeline, questions: Sequence[Question]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for question in questions:
        answer = pipeline.answer(question.question, question.id)
        records.append(answer_to_record(question, answer))
    return records


def write_jsonl(records: Sequence[dict[str, object]], path: Path) -> None:
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path | None = None) -> list[dict[str, object]]:
    source = path or snapshot_path()
    records: list[dict[str, object]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            records.append(payload)
    return records
