import json

import pytest

from clinic_rag.cli import main
from clinic_rag.corpus import load_documents
from clinic_rag.eval_dataset import load_questions
from clinic_rag.guardrails import classify
from clinic_rag.pipeline import build_pipeline
from clinic_rag.providers import UNKNOWN_RESPONSE
from clinic_rag.runner import evaluate_variant
from clinic_rag.snapshots import read_jsonl, record_pipeline


def test_ragas_0_4_metric_api_imports() -> None:
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics import NonLLMContextRecall
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        StringPresence,
    )

    assert callable(llm_factory)
    assert OpenAIEmbeddings.__name__ == "OpenAIEmbeddings"
    assert Faithfulness.__name__ == "Faithfulness"
    assert AnswerRelevancy.__name__ == "AnswerRelevancy"
    assert ContextPrecision.__name__ == "ContextPrecision"
    assert ContextRecall.__name__ == "ContextRecall"
    assert StringPresence.__name__ == "StringPresence"
    assert NonLLMContextRecall.__name__ == "NonLLMContextRecall"


def test_dataset_covers_guardrails_and_size() -> None:
    questions = load_questions()
    assert 30 <= len(questions) <= 50
    categories = {question.category for question in questions}
    assert {"emergency", "diagnosis", "dosing", "phi", "adversarial", "out_of_scope"} <= categories
    behaviors = {question.expected_behavior for question in questions}
    assert {"answer", "refuse", "emergency", "unknown"} <= behaviors


def test_labelled_phrases_are_reachable() -> None:
    questions = load_questions()
    documents = {doc.doc_id: doc.text for doc in load_documents()}
    for question in questions:
        if question.expected_behavior == "answer":
            text = "\n".join(documents[doc_id] for doc_id in question.reference_doc_ids)
        elif question.expected_behavior == "unknown":
            text = UNKNOWN_RESPONSE
        else:
            decision = classify(question.question)
            assert decision.response is not None
            text = decision.response
        lowered = text.lower()
        for phrase in question.required_phrases:
            assert phrase.lower() in lowered, question.id
        for phrase in question.forbidden_phrases:
            assert phrase.lower() not in lowered, question.id


def test_baseline_passes_gate_and_regressions_fail() -> None:
    questions = load_questions()
    baseline = evaluate_variant("baseline", questions)
    bad = evaluate_variant("bad-retriever", questions)
    unguarded = evaluate_variant("no-guardrail", questions)
    assert baseline.passed, baseline.failures
    assert baseline.judge_scores_included is False
    assert not bad.passed
    assert bad.metrics["hit_rate"] < bad.thresholds["hit_rate"]
    assert not unguarded.passed
    assert unguarded.metrics["refusal_correctness"] < unguarded.thresholds["refusal_correctness"]


def test_snapshot_matches_the_fake_provider_and_has_no_judge_scores() -> None:
    questions = load_questions()
    fresh = record_pipeline(build_pipeline("baseline"), questions)
    saved = read_jsonl()
    assert json.dumps(fresh, sort_keys=True) == json.dumps(saved, sort_keys=True)
    assert all(row["judge_scores"] is None for row in saved)


def test_committed_report_matches_a_fresh_run() -> None:
    from clinic_rag.paths import docs_dir

    saved = json.loads((docs_dir() / "sample_report.json").read_text(encoding="utf-8"))
    questions = load_questions()
    fresh = [
        evaluate_variant(variant, questions)
        for variant in ("baseline", "bad-retriever", "no-guardrail")
    ]
    assert saved["judge_scores_included"] is False
    assert [item["variant"] for item in saved["variants"]] == [item.variant for item in fresh]
    for saved_variant, result in zip(saved["variants"], fresh, strict=True):
        assert saved_variant["passed"] is result.passed
        assert saved_variant["metrics"] == pytest.approx(result.metrics)


def test_offline_cli_uses_the_snapshot_and_exits_zero() -> None:
    assert main(["eval", "--mode", "offline", "--from-snapshot"]) == 0


def test_bad_retriever_cli_exits_nonzero() -> None:
    assert main(["eval", "--mode", "offline", "--variant", "bad-retriever"]) == 1


def test_live_mode_without_a_key_does_not_invent_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["eval", "--mode", "live", "--variant", "baseline"]) == 2
