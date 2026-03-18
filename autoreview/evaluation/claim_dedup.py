from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import structlog

from autoreview.evaluation.models import Claim

logger = structlog.get_logger()


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> np.ndarray: ...


def compute_cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of a and rows of b.

    Both a and b should be 2D arrays with shape (n, dim) and (m, dim).
    Returns (n, m) similarity matrix.
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.asarray(a_norm @ b_norm.T)


async def deduplicate_claims(
    target_claims: list[Claim],
    baseline_claims: list[Claim],
    embedder: Any,
    threshold: float = 0.90,
) -> tuple[list[Claim], list[Claim]]:
    """Identify novel and shared claims in target vs. baseline.

    Baseline can be a union of claims from multiple shallower depths.
    For medium->deep: pass union(low_claims, medium_claims) as baseline.

    Returns:
        (novel_in_target, shared_with_baseline)
    """
    if not target_claims:
        return [], []
    if not baseline_claims:
        return list(target_claims), []

    target_texts = [c.text for c in target_claims]
    baseline_texts = [c.text for c in baseline_claims]

    all_emb = await embedder.embed(target_texts + baseline_texts)
    target_emb = all_emb[: len(target_texts)]
    baseline_emb = all_emb[len(target_texts) :]

    sim_matrix = compute_cosine_similarity_matrix(target_emb, baseline_emb)
    max_sims = sim_matrix.max(axis=1)

    novel: list[Claim] = []
    shared: list[Claim] = []

    for i, claim in enumerate(target_claims):
        if max_sims[i] >= threshold:
            shared.append(claim)
        else:
            novel.append(claim)

    logger.info(
        "claim_dedup.complete",
        n_target=len(target_claims),
        n_baseline=len(baseline_claims),
        n_novel=len(novel),
        n_shared=len(shared),
        threshold=threshold,
    )
    return novel, shared
