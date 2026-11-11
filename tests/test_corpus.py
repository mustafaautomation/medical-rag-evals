from clinic_rag.chunking import chunk_documents
from clinic_rag.corpus import load_documents


def test_corpus_size_and_unique_ids() -> None:
    documents = load_documents()
    assert 15 <= len(documents) <= 30
    assert len({doc.doc_id for doc in documents}) == len(documents)


def test_each_handout_is_one_chunk() -> None:
    chunks = chunk_documents(load_documents())
    assert len(chunks) == len(load_documents())
    assert all(chunk.chunk_id.endswith("-0") for chunk in chunks)


def test_medication_leaflets_are_marked_synthetic() -> None:
    documents = {doc.doc_id: doc for doc in load_documents()}
    for doc_id in (
        "leaflet-cedarcalm",
        "leaflet-harborpain",
        "leaflet-rivermycin",
        "leaflet-glucoflat",
    ):
        assert "not a real medication label" in documents[doc_id].text.lower()
