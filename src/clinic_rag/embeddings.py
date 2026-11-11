"""Local TF-IDF embeddings.

Vectors are fit on the clinic handouts, L2-normalized, and deterministic.
No model download and no API key are required. Live generation can still
use a hosted chat model; retrieval stays on this index so CI and a laptop
search the same space.
"""

import math
import re
from collections import Counter

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "and",
        "or",
        "for",
        "on",
        "in",
        "is",
        "are",
        "was",
        "be",
        "as",
        "at",
        "by",
        "from",
        "with",
        "that",
        "this",
        "it",
        "if",
        "do",
        "does",
        "did",
        "what",
        "when",
        "where",
        "who",
        "how",
        "which",
        "can",
        "i",
        "you",
        "your",
        "my",
        "me",
        "we",
        "our",
        "please",
        "about",
        "into",
        "its",
    }
)


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


class TfidfEmbedder:
    """Fit once on the chunk corpus, then embed queries in that vocabulary."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf = np.zeros(0, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return len(self.vocab)

    def fit(self, texts: list[str]) -> None:
        docs = [tokenize(text) for text in texts]
        df: Counter[str] = Counter()
        for tokens in docs:
            df.update(set(tokens))
        self.vocab = {token: index for index, token in enumerate(sorted(df))}
        n_docs = max(len(docs), 1)
        idf = np.ones(len(self.vocab), dtype=np.float32)
        for token, index in self.vocab.items():
            idf[index] = math.log((n_docs + 1) / (df[token] + 1)) + 1.0
        self.idf = idf

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = tokenize(text)
        if not tokens or self.dimension == 0:
            return vector
        counts = Counter(tokens)
        length = float(len(tokens))
        for token, count in counts.items():
            index = self.vocab.get(token)
            if index is None:
                continue
            vector[index] = (count / length) * self.idf[index]
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self.embed(text) for text in texts])
