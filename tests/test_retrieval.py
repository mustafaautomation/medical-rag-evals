from clinic_rag.embeddings import TfidfEmbedder
from clinic_rag.eval_dataset import load_questions
from clinic_rag.pipeline import build_index
from clinic_rag.retrieval import pgvector_schema_sql


def test_embeddings_are_deterministic_and_normalized() -> None:
    _chunks, embedder, _retriever = build_index()
    assert isinstance(embedder, TfidfEmbedder)
    first = embedder.embed("weekday hours Monday Friday")
    second = embedder.embed("weekday hours Monday Friday")
    assert first.shape == (embedder.dimension,)
    assert first.tolist() == second.tolist()
    norm = float((first @ first) ** 0.5)
    assert abs(norm - 1.0) < 1e-5


def test_factual_questions_rank_the_reference_doc_first() -> None:
    _chunks, _embedder, retriever = build_index()
    misses: list[str] = []
    for question in load_questions():
        if not question.score_retrieval:
            continue
        hits = retriever.search(question.question, k=4)
        top = hits[0].chunk.doc_id if hits else "(none)"
        if top not in question.reference_doc_ids:
            misses.append(f"{question.id}: got {top}, wanted {question.reference_doc_ids}")
    assert misses == []


def test_out_of_scope_questions_abstain() -> None:
    from clinic_rag.pipeline import build_pipeline

    pipeline = build_pipeline("baseline")
    for question in load_questions():
        if question.expected_behavior != "unknown":
            continue
        answer = pipeline.answer(question.question, question.id)
        assert answer.behavior == "unknown", question.id
        assert answer.retrieved_contexts == []


def test_bad_retriever_does_not_lead_with_the_reference() -> None:
    _chunks, _embedder, retriever = build_index()
    question = next(item for item in load_questions() if item.id == "q01")
    hits = retriever.search(question.question, k=1, worst=True)
    assert hits
    assert hits[0].chunk.doc_id not in question.reference_doc_ids


def test_pgvector_schema_uses_the_fit_dimension() -> None:
    _chunks, embedder, _retriever = build_index()
    sql = pgvector_schema_sql(embedder.dimension)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert f"vector({embedder.dimension})" in sql
