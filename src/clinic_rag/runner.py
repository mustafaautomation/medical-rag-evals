"""Run the offline gate and, when requested, the live RAGAS judges."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import yaml

from clinic_rag.metrics import (
    hit_and_mrr,
    live_key_available,
    mean,
    non_llm_context_recall,
    refusal_correct,
    score_live_judgements,
    string_presence,
)
from clinic_rag.paths import thresholds_path
from clinic_rag.pipeline import Pipeline, build_pipeline
from clinic_rag.types import Answer, Chunk, Question, VariantName

OFFLINE_METRICS = (
    "string_presence",
    "non_llm_context_recall",
    "refusal_correctness",
    "hit_rate",
    "mrr",
)
LIVE_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


@dataclass
class QuestionScore:
    question: Question
    answer: Answer
    string_presence: float
    non_llm_context_recall: float | None
    refusal_correctness: float
    hit: float | None
    mrr: float | None

    @property
    def failures(self) -> list[str]:
        reasons: list[str] = []
        if self.string_presence < 1.0:
            reasons.append("string_presence")
        if self.refusal_correctness < 1.0:
            reasons.append("refusal_correctness")
        if self.hit is not None and self.hit < 1.0:
            reasons.append("hit_rate")
        if self.non_llm_context_recall is not None and self.non_llm_context_recall < 1.0:
            reasons.append("non_llm_context_recall")
        return reasons


@dataclass
class EvalResult:
    variant: str
    mode: str
    metrics: dict[str, float]
    thresholds: dict[str, float]
    question_scores: list[QuestionScore] = field(default_factory=list)
    judge_scores_included: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def failures(self) -> dict[str, float]:
        return {
            name: score
            for name, score in self.metrics.items()
            if name in self.thresholds and score < self.thresholds[name]
        }

    @property
    def passed(self) -> bool:
        return not self.failures


def load_thresholds() -> dict[str, dict[str, float]]:
    payload = yaml.safe_load(thresholds_path().read_text(encoding="utf-8"))
    offline = {str(key): float(value) for key, value in payload["offline"].items()}
    live = {str(key): float(value) for key, value in payload["live"].items()}
    return {"offline": offline, "live": live}


def reference_contexts(question: Question, chunks: Sequence[Chunk]) -> list[str]:
    by_doc: dict[str, list[str]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk.text)
    contexts: list[str] = []
    for doc_id in question.reference_doc_ids:
        contexts.extend(by_doc.get(doc_id, []))
    return contexts


def score_answer(question: Question, answer: Answer, chunks: Sequence[Chunk]) -> QuestionScore:
    presence = string_presence(answer.text, question.required_phrases)
    refusal = refusal_correct(question, answer.text)
    recall: float | None = None
    hit: float | None = None
    reciprocal: float | None = None
    if question.score_retrieval:
        recall = non_llm_context_recall(
            answer.retrieved_contexts,
            reference_contexts(question, chunks),
        )
        hit, reciprocal = hit_and_mrr(answer.retrieved_doc_ids, question.reference_doc_ids)
    return QuestionScore(
        question=question,
        answer=answer,
        string_presence=presence,
        non_llm_context_recall=recall,
        refusal_correctness=refusal,
        hit=hit,
        mrr=reciprocal,
    )


def aggregate(scores: Sequence[QuestionScore]) -> dict[str, float]:
    recall_values = [
        item.non_llm_context_recall for item in scores if item.non_llm_context_recall is not None
    ]
    hit_values = [item.hit for item in scores if item.hit is not None]
    mrr_values = [item.mrr for item in scores if item.mrr is not None]
    return {
        "string_presence": mean([item.string_presence for item in scores]),
        "non_llm_context_recall": mean([value for value in recall_values if value is not None]),
        "refusal_correctness": mean([item.refusal_correctness for item in scores]),
        "hit_rate": mean([value for value in hit_values if value is not None]),
        "mrr": mean([value for value in mrr_values if value is not None]),
    }


def evaluate_pipeline(
    pipeline: Pipeline,
    questions: Sequence[Question],
    *,
    mode: str = "offline",
) -> EvalResult:
    thresholds = load_thresholds()
    active = dict(thresholds["offline"])
    notes = [
        "Offline metrics were computed locally. No LLM judge scores are included.",
    ]
    if mode == "live":
        if not live_key_available():
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Live RAGAS eval was not run "
                "and no judge scores were invented."
            )
        active.update(thresholds["live"])
    chunks = pipeline.chunks
    question_scores: list[QuestionScore] = []
    live_rows: list[tuple[Question, str, list[str]]] = []
    for question in questions:
        answer = pipeline.answer(question.question, question.id)
        question_scores.append(score_answer(question, answer, chunks))
        live_rows.append((question, answer.text, list(answer.retrieved_contexts)))
    metrics = aggregate(question_scores)
    judge_scores_included = False
    if mode == "live":
        metrics.update(score_live_judgements(live_rows))
        judge_scores_included = True
        notes = ["Live mode ran RAGAS LLM judges with OPENAI_API_KEY."]
    return EvalResult(
        variant=pipeline.variant,
        mode=mode,
        metrics=metrics,
        thresholds=active,
        question_scores=question_scores,
        judge_scores_included=judge_scores_included,
        notes=notes,
    )


def evaluate_variant(
    variant: VariantName,
    questions: Sequence[Question],
    *,
    mode: str = "offline",
    provider_name: str | None = None,
) -> EvalResult:
    pipeline = build_pipeline(variant, provider_name=provider_name)
    return evaluate_pipeline(pipeline, questions, mode=mode)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        raise TypeError("Expected a list of strings in the snapshot")
    return [str(item) for item in value]


def evaluate_recorded(
    records: Sequence[dict[str, object]],
    questions: Sequence[Question],
    chunks: Sequence[Chunk],
    *,
    variant: str = "baseline",
) -> EvalResult:
    """Score committed pipeline outputs. Does not call a chat model or a judge."""

    by_id = {question.id: question for question in questions}
    if len(records) != len(questions):
        raise ValueError(
            f"Snapshot has {len(records)} rows but the dataset has {len(questions)} questions"
        )
    question_scores: list[QuestionScore] = []
    for record in records:
        question_id = str(record["id"])
        question = by_id[question_id]
        answer = Answer(
            question_id=question.id,
            question=question.question,
            text=str(record["response"]),
            retrieved_doc_ids=_string_list(record["retrieved_doc_ids"]),
            retrieved_contexts=_string_list(record["retrieved_contexts"]),
            behavior=str(record["behavior"]),
            variant=str(record["variant"]),
        )
        question_scores.append(score_answer(question, answer, chunks))
    thresholds = load_thresholds()["offline"]
    return EvalResult(
        variant=variant,
        mode="offline",
        metrics=aggregate(question_scores),
        thresholds=thresholds,
        question_scores=question_scores,
        judge_scores_included=False,
        notes=[
            "Scored the committed offline snapshot. judge_scores in that file are null.",
            "No LLM-judged RAGAS metric was computed.",
        ],
    )
