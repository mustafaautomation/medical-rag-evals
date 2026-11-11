"""Offline metrics, including RAGAS non-LLM metrics, plus the live judge path.

LLM-judged RAGAS metrics run only when OPENAI_API_KEY is set. This module
does not invent judge scores.
"""

import asyncio
import os
import warnings
from collections.abc import Sequence

from clinic_rag.types import Question

MetricMap = dict[str, float]


def string_presence(response: str, phrases: Sequence[str]) -> float:
    """Mean RAGAS StringPresence over the required phrases."""

    if not phrases:
        return 1.0
    from ragas.metrics.collections import StringPresence

    scorer = StringPresence()
    scores = [float(scorer.score(reference=phrase, response=response).value) for phrase in phrases]
    return sum(scores) / len(scores)


def non_llm_context_recall(
    retrieved_contexts: Sequence[str],
    reference_contexts: Sequence[str],
) -> float:
    """RAGAS NonLLMContextRecall (legacy class; not in the 0.4 collections API)."""

    if not reference_contexts:
        return 1.0
    if not retrieved_contexts:
        return 0.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import NonLLMContextRecall

        metric = NonLLMContextRecall()
        sample = SingleTurnSample(
            retrieved_contexts=list(retrieved_contexts),
            reference_contexts=list(reference_contexts),
        )
        score = asyncio.run(metric.single_turn_ascore(sample))
    return float(score)


_ABSTAIN_MARKERS = (
    "i don't know",
    "i cannot diagnose",
    "i cannot give personal medical advice",
    "do not share personal identifiers",
    "call 911",
)


def refusal_correct(question: Question, response: str) -> float:
    """1 when the answer took the labelled action.

    Factual questions must not be redirected to a refusal, an emergency
    script, or an "I don't know". The other labels must contain their
    required phrase and must not contain a forbidden phrase.
    """

    lowered = response.lower()
    if question.expected_behavior == "answer":
        if any(marker in lowered for marker in _ABSTAIN_MARKERS):
            return 0.0
        return 1.0
    phrases_ok = all(phrase.lower() in lowered for phrase in question.required_phrases)
    forbidden_hit = any(phrase.lower() in lowered for phrase in question.forbidden_phrases)
    if not phrases_ok or forbidden_hit:
        return 0.0
    return 1.0


def hit_and_mrr(
    retrieved_doc_ids: Sequence[str],
    reference_doc_ids: Sequence[str],
) -> tuple[float, float]:
    if not reference_doc_ids:
        return 1.0, 1.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in reference_doc_ids:
            return 1.0, 1.0 / rank
    return 0.0, 0.0


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def live_key_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def score_live_judgements(
    rows: Sequence[tuple[Question, str, list[str]]],
) -> MetricMap:
    """Run Faithfulness, AnswerRelevancy, ContextPrecision, and ContextRecall.

    `rows` is (question, response, retrieved_contexts). This calls OpenAI.
    """

    if not live_key_available():
        raise RuntimeError("OPENAI_API_KEY is not set, so live RAGAS judges were not run.")
    from openai import AsyncOpenAI, OpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    llm = llm_factory("gpt-4o-mini", client=AsyncOpenAI())
    embeddings = OpenAIEmbeddings(client=OpenAI(), model="text-embedding-3-small")
    faithfulness = Faithfulness(llm=llm)
    relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    precision = ContextPrecision(llm=llm)
    recall = ContextRecall(llm=llm)

    faith_scores: list[float] = []
    relevancy_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    for question, response, contexts in rows:
        if question.expected_behavior != "answer":
            continue
        faith_scores.append(
            float(
                faithfulness.score(
                    user_input=question.question,
                    response=response,
                    retrieved_contexts=list(contexts),
                ).value
            )
        )
        relevancy_scores.append(
            float(
                relevancy.score(
                    user_input=question.question,
                    response=response,
                ).value
            )
        )
        precision_scores.append(
            float(
                precision.score(
                    user_input=question.question,
                    reference=question.reference_answer,
                    retrieved_contexts=list(contexts),
                ).value
            )
        )
        recall_scores.append(
            float(
                recall.score(
                    user_input=question.question,
                    reference=question.reference_answer,
                    retrieved_contexts=list(contexts),
                ).value
            )
        )
    return {
        "faithfulness": mean(faith_scores),
        "answer_relevancy": mean(relevancy_scores),
        "context_precision": mean(precision_scores),
        "context_recall": mean(recall_scores),
    }
