from __future__ import annotations

from typing import Literal  # noqa: F401

import numpy as np
import pytest

from autoreview.evaluation.claim_dedup import (
    compute_cosine_similarity_matrix,
    deduplicate_claims,
)
from autoreview.evaluation.models import Claim

ClaimCategory = Literal["empirical", "methodological", "contextual", "synthesis", "limitation"]


def _make_claim(text: str, category: ClaimCategory = "empirical") -> Claim:
    return Claim(text=text, category=category)


class MockEmbedder:
    """Returns deterministic embeddings based on word overlap."""

    async def embed(self, texts: list[str]) -> np.ndarray:
        all_words: set[str] = set()
        for t in texts:
            all_words.update(t.lower().split())
        vocab = sorted(all_words)
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        embeddings = np.zeros((len(texts), len(vocab)))
        for i, t in enumerate(texts):
            for word in t.lower().split():
                embeddings[i, word_to_idx[word]] = 1.0
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        return embeddings


@pytest.fixture
def embedder():
    return MockEmbedder()


def test_cosine_similarity_matrix():
    a = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    b = np.array([[1, 0, 0], [0, 0, 1]], dtype=float)
    sim = compute_cosine_similarity_matrix(a, b)
    assert sim.shape == (2, 2)
    assert sim[0, 0] == pytest.approx(1.0)
    assert sim[0, 1] == pytest.approx(0.0)
    assert sim[1, 0] == pytest.approx(0.0)
    assert sim[1, 1] == pytest.approx(0.0)


async def test_deduplicate_identical_claims(embedder):
    target = [_make_claim("Drug A reduces blood pressure")]
    baseline = [_make_claim("Drug A reduces blood pressure")]
    novel, shared = await deduplicate_claims(target, baseline, embedder, threshold=0.90)
    assert len(novel) == 0
    assert len(shared) == 1


async def test_deduplicate_distinct_claims(embedder):
    target = [_make_claim("Novel gene therapy shows promise")]
    baseline = [_make_claim("Traditional chemotherapy has limitations")]
    novel, shared = await deduplicate_claims(target, baseline, embedder, threshold=0.90)
    assert len(novel) == 1
    assert len(shared) == 0


async def test_deduplicate_empty_baseline(embedder):
    target = [_make_claim("Claim one"), _make_claim("Claim two")]
    baseline = []
    novel, shared = await deduplicate_claims(target, baseline, embedder, threshold=0.90)
    assert len(novel) == 2
    assert len(shared) == 0


async def test_deduplicate_empty_target(embedder):
    target = []
    baseline = [_make_claim("Claim one")]
    novel, shared = await deduplicate_claims(target, baseline, embedder, threshold=0.90)
    assert len(novel) == 0
    assert len(shared) == 0


async def test_deduplicate_mixed(embedder):
    target = [
        _make_claim("Drug A reduces blood pressure significantly"),
        _make_claim("A completely novel finding about genetics"),
    ]
    baseline = [
        _make_claim("Drug A reduces blood pressure"),
    ]
    novel, shared = await deduplicate_claims(target, baseline, embedder, threshold=0.50)
    assert len(novel) + len(shared) == 2
