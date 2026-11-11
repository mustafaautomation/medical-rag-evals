"""Retrieve a handout and answer, unless a guardrail applies."""

import os

from clinic_rag.chunking import chunk_documents
from clinic_rag.corpus import load_documents
from clinic_rag.embeddings import TfidfEmbedder
from clinic_rag.guardrails import classify
from clinic_rag.providers import (
    GUARDED_INSTRUCTIONS,
    UNGUARDED_INSTRUCTIONS,
    ChatProvider,
    provider_from_name,
)
from clinic_rag.retrieval import FaissHybridRetriever, HybridRetriever
from clinic_rag.types import Answer, Chunk, VariantName

DEFAULT_TOP_K = 4
MIN_SCORE = 0.35

VARIANTS: tuple[VariantName, ...] = ("baseline", "bad-retriever", "no-guardrail")


class Pipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        provider: ChatProvider,
        *,
        variant: VariantName = "baseline",
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_SCORE,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}")
        self.retriever = retriever
        self.provider = provider
        self.variant: VariantName = variant
        self.top_k = top_k
        self.min_score = min_score
        self.chunks = retriever.chunks

    def answer(self, question: str, question_id: str = "") -> Answer:
        decision = classify(question)
        if decision.action != "allow":
            if self.variant == "no-guardrail":
                text = self.provider.generate(
                    question,
                    "",
                    UNGUARDED_INSTRUCTIONS,
                    refusal_bypass=True,
                )
                return Answer(
                    question_id=question_id,
                    question=question,
                    text=text,
                    retrieved_doc_ids=[],
                    retrieved_contexts=[],
                    behavior="unguarded",
                    variant=self.variant,
                )
            assert decision.response is not None
            return Answer(
                question_id=question_id,
                question=question,
                text=decision.response,
                retrieved_doc_ids=[],
                retrieved_contexts=[],
                behavior=decision.action,
                variant=self.variant,
            )

        hits = self.retriever.search(
            question,
            k=self.top_k,
            worst=self.variant == "bad-retriever",
        )
        # Quote only the winning chunk. Baseline keeps it when the hybrid score
        # clears the abstain threshold. The bad retriever quotes its worst chunk.
        if self.variant == "bad-retriever":
            useful = hits[:1]
        elif hits and hits[0].score >= self.min_score:
            useful = hits[:1]
        else:
            useful = []
        contexts = [hit.chunk.text for hit in useful]
        instructions = (
            UNGUARDED_INSTRUCTIONS if self.variant == "no-guardrail" else GUARDED_INSTRUCTIONS
        )
        text = self.provider.generate(
            question,
            "\n\n".join(contexts),
            instructions,
            refusal_bypass=False,
        )
        behavior = "unknown" if not contexts else "answer"
        return Answer(
            question_id=question_id,
            question=question,
            text=text,
            retrieved_doc_ids=[hit.chunk.doc_id for hit in useful],
            retrieved_contexts=contexts,
            behavior=behavior,
            variant=self.variant,
        )


def build_index() -> tuple[list[Chunk], TfidfEmbedder, FaissHybridRetriever]:
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedder = TfidfEmbedder()
    embedder.fit([chunk.text for chunk in chunks])
    retriever = FaissHybridRetriever(chunks, embedder)
    return chunks, embedder, retriever


def build_pipeline(
    variant: VariantName = "baseline",
    provider: ChatProvider | None = None,
    *,
    provider_name: str | None = None,
) -> Pipeline:
    _chunks, _embedder, retriever = build_index()
    if provider is None:
        name = provider_name or os.environ.get("CLINIC_RAG_PROVIDER", "fake")
        provider = provider_from_name(name)
    return Pipeline(retriever, provider, variant=variant)
