"""Benchmark scoring module for comparing programmatic extractions against LLM ground truth.

Compares PaperExtraction outputs using word-overlap matching by default.
Optionally uses embedding-based similarity (sentence-transformers) for more
accurate key_findings matching against LLM paraphrases.

Does NOT require sentence-transformers at runtime — falls back gracefully.
"""

from __future__ import annotations

from statistics import correlation

import structlog

from autoreview.extraction.models import EvidenceStrength, PaperExtraction, StudyDesign

logger = structlog.get_logger()

# Ordinal mapping for evidence strength partial credit scoring.
# Adjacent levels get partial credit instead of binary exact match.
_EVIDENCE_ORDINAL: dict[str, int] = {
    "strong": 3,
    "moderate": 2,
    "weak": 1,
    "preliminary": 0,
}


def _evidence_strength_score(pred: str | EvidenceStrength, gold: str | EvidenceStrength) -> float:
    """Score evidence strength with partial credit for adjacent levels.

    Returns:
        1.0 for exact match, 0.5 for one level apart, 0.0 for 2+ levels apart.
    """
    p = _EVIDENCE_ORDINAL.get(str(pred), -1)
    g = _EVIDENCE_ORDINAL.get(str(gold), -1)
    if p < 0 or g < 0:
        return 0.0
    diff = abs(p - g)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.5
    return 0.0


def _word_overlap_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity based on word overlap (case-insensitive)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length (for ROUGE-L). 1D DP."""
    if not a or not b:
        return 0
    # Use 1D DP: prev row approach
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


# Study design similarity groups for partial credit scoring.
# Designs within the same group get 0.5 credit when mismatched.
_STUDY_DESIGN_GROUPS: dict[str, int] = {
    # Group 0: Review-type designs
    "systematic_review": 0,
    "narrative_review": 0,
    "meta_analysis": 0,
    # Group 1: Empirical/observational designs
    "cross_sectional": 1,
    "cohort": 1,
    "case_control": 1,
    "case_series": 1,
    "case_report": 1,
    # Group 2: Experimental/computational designs
    "computational": 2,
    "rct": 2,
    "in_vitro": 2,
    # Group 3: Catch-all
    "other": 3,
}


def _study_design_score(pred: StudyDesign | str | None, gold: StudyDesign | str | None) -> float:
    """Score study design with partial credit for related designs.

    Returns:
        1.0 for exact match, 0.5 for same group or computational<->other,
        0.0 for unrelated designs.
    """
    p = str(pred) if pred is not None else "other"
    g = str(gold) if gold is not None else "other"
    if p == g:
        return 1.0
    # Computational <-> other is a common borderline case, give partial credit
    if {p, g} == {"computational", "other"}:
        return 0.5
    # Same group gets partial credit
    p_group = _STUDY_DESIGN_GROUPS.get(p, -1)
    g_group = _STUDY_DESIGN_GROUPS.get(g, -2)
    if p_group >= 0 and p_group == g_group:
        return 0.5
    return 0.0


def rouge_l_f1(prediction: str, reference: str) -> float:
    """ROUGE-L F1 score between two strings. Word-level LCS."""
    pred_words = prediction.lower().split()
    ref_words = reference.lower().split()
    if not pred_words or not ref_words:
        return 0.0
    lcs_len = _lcs_length(pred_words, ref_words)
    if lcs_len == 0:
        return 0.0
    precision = lcs_len / len(pred_words)
    recall = lcs_len / len(ref_words)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_study_design_accuracy(
    predicted: list[StudyDesign | None],
    ground_truth: list[StudyDesign | None],
) -> float:
    """Exact match percentage."""
    if not predicted:
        return 0.0
    matches = sum(1 for p, g in zip(predicted, ground_truth, strict=False) if p == g)
    return matches / len(predicted)


def compute_sample_size_accuracy(
    predicted: list[int | None],
    ground_truth: list[int | None],
    tolerance: float = 0.10,
) -> float:
    """Accuracy with 10% tolerance. None-None = match, None-value = 0."""
    if not predicted:
        return 0.0
    matches = 0
    for p, g in zip(predicted, ground_truth, strict=False):
        if p is None and g is None:
            matches += 1
        elif p is not None and g is not None:
            if g == 0:
                # Exact match required when ground truth is 0
                if p == 0:
                    matches += 1
            elif abs(p - g) / abs(g) <= tolerance:
                matches += 1
        # else: None vs value = no match
    return matches / len(predicted)


def compute_quality_score_correlation(
    predicted: list[float | None],
    ground_truth: list[float | None],
) -> float:
    """Pearson correlation normalized to [0,1]. Returns 0.5 if <3 pairs or no variance."""
    # Filter to pairs where both are non-None
    pairs = [
        (p, g)
        for p, g in zip(predicted, ground_truth, strict=False)
        if p is not None and g is not None
    ]
    if len(pairs) < 3:
        return 0.5
    pred_vals = [p for p, _ in pairs]
    gold_vals = [g for _, g in pairs]
    # Check for zero variance
    if len(set(pred_vals)) <= 1 or len(set(gold_vals)) <= 1:
        return 0.5
    try:
        r = correlation(pred_vals, gold_vals)
    except (ValueError, ZeroDivisionError):
        return 0.5
    # Normalize from [-1, 1] to [0, 1]
    return (r + 1.0) / 2.0


def compute_composite_score(field_scores: dict[str, float]) -> float:
    """Weighted composite score.

    Weights:
        key_findings=0.40, evidence_strength=0.05, quantitative_result=0.05,
        methods_summary=0.15, limitations=0.10, study_design=0.10,
        quality_score=0.05, sample_size=0.10.
    """
    weights = {
        "key_findings": 0.40,
        "evidence_strength": 0.05,
        "quantitative_result": 0.05,
        "methods_summary": 0.15,
        "limitations": 0.10,
        "study_design": 0.10,
        "quality_score": 0.05,
        "sample_size": 0.10,
    }
    total = 0.0
    for field, weight in weights.items():
        total += field_scores.get(field, 0.0) * weight
    return total


def compute_dual_composite(
    similarity_scores: dict[str, float],
    factual_scores: dict[str, float],
    alpha: float = 0.5,
) -> dict[str, float]:
    """Compute dual-layer composite score.

    Args:
        similarity_scores: Per-field embedding similarity scores.
        factual_scores: Per-field factual accuracy scores (with pass-through
            fields copied from similarity_scores).
        alpha: Blend weight. 1.0 = similarity only, 0.0 = factual only.

    Returns:
        Dict with 'similarity', 'factual', and 'combined' composite scores.
    """
    sim_composite = compute_composite_score(similarity_scores)
    fact_composite = compute_composite_score(factual_scores)
    combined = alpha * sim_composite + (1 - alpha) * fact_composite
    return {
        "similarity": sim_composite,
        "factual": fact_composite,
        "combined": combined,
    }


def score_extraction_pair(
    predicted: PaperExtraction,
    ground_truth: PaperExtraction,
) -> dict[str, float]:
    """Score a single extraction pair. Returns per-field scores dict.

    Fields scored:
        key_findings: Greedy word-overlap matching between claim sets.
        evidence_strength: Exact match on paired findings.
        quantitative_result: Token overlap on paired findings' quantitative_result fields.
        methods_summary: ROUGE-L F1.
        limitations: ROUGE-L F1.
        study_design: Exact match (1.0 or 0.0).
        quality_score: 1.0 - abs(pred - gold), handle None.
        sample_size: Tolerance match via compute_sample_size_accuracy.
    """
    scores: dict[str, float] = {}

    # --- key_findings: greedy word-overlap matching ---
    pred_claims = [f.claim for f in predicted.key_findings]
    gold_claims = [f.claim for f in ground_truth.key_findings]

    # matched_pairs: list of (pred_idx, gold_idx, similarity) from greedy matching
    matched_pairs: list[tuple[int, int, float]] = []

    if not gold_claims:
        scores["key_findings"] = 1.0 if not pred_claims else 0.0
    else:
        used_pred: set[int] = set()
        matched_sims: list[float] = []
        for g_idx, g_claim in enumerate(gold_claims):
            best_sim = 0.0
            best_idx = -1
            for i, p_claim in enumerate(pred_claims):
                if i in used_pred:
                    continue
                sim = _word_overlap_similarity(g_claim, p_claim)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            if best_idx >= 0:
                used_pred.add(best_idx)
                matched_pairs.append((best_idx, g_idx, best_sim))
            matched_sims.append(best_sim)
        scores["key_findings"] = sum(matched_sims) / len(matched_sims)

    # --- evidence_strength: partial credit on matched finding pairs ---
    # Use greedy-matched pairs (with similarity > 0.3) instead of positional zip.
    # Partial credit: exact match = 1.0, one level apart = 0.5, 2+ = 0.0.
    pred_findings = predicted.key_findings
    gold_findings = ground_truth.key_findings
    valid_pairs = [(pi, gi) for pi, gi, sim in matched_pairs if sim > 0.3]
    if not valid_pairs:
        scores["evidence_strength"] = 1.0 if not gold_findings else 0.0
    else:
        ev_score_sum = sum(
            _evidence_strength_score(
                pred_findings[pi].evidence_strength,
                gold_findings[gi].evidence_strength,
            )
            for pi, gi in valid_pairs
        )
        scores["evidence_strength"] = ev_score_sum / len(valid_pairs)

    # --- quantitative_result: token overlap on matched finding pairs ---
    quant_scores: list[float] = []
    for pi, gi, sim in matched_pairs:
        if sim <= 0.3:
            continue
        p_q = pred_findings[pi].quantitative_result or ""
        g_q = gold_findings[gi].quantitative_result or ""
        if not p_q and not g_q:
            quant_scores.append(1.0)
        elif p_q and g_q:
            quant_scores.append(_word_overlap_similarity(p_q, g_q))
        else:
            quant_scores.append(0.0)
    scores["quantitative_result"] = sum(quant_scores) / len(quant_scores) if quant_scores else 0.0

    # --- methods_summary: ROUGE-L F1 ---
    scores["methods_summary"] = rouge_l_f1(predicted.methods_summary, ground_truth.methods_summary)

    # --- limitations: ROUGE-L F1 ---
    scores["limitations"] = rouge_l_f1(predicted.limitations, ground_truth.limitations)

    # --- study_design: match with partial credit for related designs ---
    scores["study_design"] = _study_design_score(predicted.study_design, ground_truth.study_design)

    # --- quality_score: 1.0 - abs(pred - gold) ---
    p_qs = predicted.quality_score
    g_qs = ground_truth.quality_score
    if p_qs is not None and g_qs is not None:
        scores["quality_score"] = 1.0 - abs(p_qs - g_qs)
    elif p_qs is None and g_qs is None:
        scores["quality_score"] = 1.0
    else:
        scores["quality_score"] = 0.0

    # --- sample_size: tolerance match ---
    scores["sample_size"] = compute_sample_size_accuracy(
        [predicted.sample_size], [ground_truth.sample_size]
    )

    return scores


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> object | None:
    """Load a sentence-transformers model. Returns None if unavailable.

    Args:
        model_name: HuggingFace model name for sentence embeddings.

    Returns:
        SentenceTransformer model instance, or None if import fails.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        logger.info("embedding_model_loaded", model=model_name)
        return model
    except ImportError:
        logger.warning("sentence_transformers_not_available", fallback="word_overlap")
        return None


def _embedding_key_findings_score(
    pred_claims: list[str],
    gold_claims: list[str],
    model: object,
) -> tuple[dict[str, float], list[tuple[int, int, float]]]:
    """Score key_findings using embedding cosine similarity + Hungarian matching.

    Args:
        pred_claims: Predicted claim strings.
        gold_claims: Ground truth claim strings.
        model: A loaded SentenceTransformer model.

    Returns:
        Tuple of (scores_dict, matched_pairs) where:
        - scores_dict has keys: key_findings, key_findings_precision, key_findings_recall
        - matched_pairs is a list of (pred_idx, gold_idx, similarity) tuples
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not gold_claims:
        score = 1.0 if not pred_claims else 0.0
        return (
            {
                "key_findings": score,
                "key_findings_precision": score,
                "key_findings_recall": score,
            },
            [],
        )
    if not pred_claims:
        return (
            {
                "key_findings": 0.0,
                "key_findings_precision": 0.0,
                "key_findings_recall": 0.0,
            },
            [],
        )

    # Encode all claims
    all_texts = pred_claims + gold_claims
    embeddings = model.encode(all_texts, convert_to_numpy=True)  # type: ignore[attr-defined]
    pred_emb = embeddings[: len(pred_claims)]
    gold_emb = embeddings[len(pred_claims) :]

    # Cosine similarity matrix: (n_gold, n_pred)
    # Normalize embeddings
    pred_norm = pred_emb / (np.linalg.norm(pred_emb, axis=1, keepdims=True) + 1e-9)
    gold_norm = gold_emb / (np.linalg.norm(gold_emb, axis=1, keepdims=True) + 1e-9)
    sim_matrix = gold_norm @ pred_norm.T  # shape: (n_gold, n_pred)

    # Hungarian algorithm on cost matrix (1 - similarity)
    cost_matrix = 1.0 - sim_matrix
    gold_idx, pred_idx = linear_sum_assignment(cost_matrix)

    matched_sims = sim_matrix[gold_idx, pred_idx]
    mean_sim = float(np.mean(matched_sims)) if len(matched_sims) > 0 else 0.0

    # Build matched pairs list: (pred_idx, gold_idx, similarity)
    matched_pairs: list[tuple[int, int, float]] = [
        (int(pi), int(gi), float(sim_matrix[gi, pi]))
        for gi, pi in zip(gold_idx, pred_idx, strict=False)
    ]

    # Precision: fraction of predicted claims matched with similarity > 0.5
    match_threshold = 0.5
    # Build full similarity profile for each predicted claim (best match to any gold)
    best_sim_per_pred = np.max(sim_matrix, axis=0)  # shape: (n_pred,)
    precision = float(np.mean(best_sim_per_pred > match_threshold))

    # Recall: fraction of gold claims matched with similarity > 0.5
    best_sim_per_gold = np.max(sim_matrix, axis=1)  # shape: (n_gold,)
    recall = float(np.mean(best_sim_per_gold > match_threshold))

    return (
        {
            "key_findings": mean_sim,
            "key_findings_precision": precision,
            "key_findings_recall": recall,
        },
        matched_pairs,
    )


def _embedding_text_similarity(text_a: str, text_b: str, model: object) -> float:
    """Compute cosine similarity between two texts using embedding model.

    Args:
        text_a: First text string.
        text_b: Second text string.
        model: A loaded SentenceTransformer model.

    Returns:
        Cosine similarity in [0, 1] (clamped from [-1, 1]).
    """
    import numpy as np

    if not text_a.strip() or not text_b.strip():
        return 0.0

    embeddings = model.encode([text_a, text_b], convert_to_numpy=True)  # type: ignore[attr-defined]
    emb_a = embeddings[0]
    emb_b = embeddings[1]

    # Normalize
    norm_a = np.linalg.norm(emb_a) + 1e-9
    norm_b = np.linalg.norm(emb_b) + 1e-9
    similarity = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))

    # Clamp to [0, 1] — negative similarity means unrelated, treat as 0
    return max(0.0, similarity)


def score_extraction_pair_with_embeddings(
    predicted: PaperExtraction,
    ground_truth: PaperExtraction,
    model: object | None = None,
) -> dict[str, float]:
    """Score using embedding similarity for key_findings, methods_summary, and limitations.

    When a model is provided, overrides the following with embedding-based scores:
    - key_findings: Hungarian-matched cosine similarity between claim sets
    - methods_summary: cosine similarity between predicted and ground truth text
    - limitations: cosine similarity between predicted and ground truth text

    Falls back to word-overlap/ROUGE-L scoring if sentence-transformers model is
    not provided.

    Args:
        predicted: The programmatic extraction to evaluate.
        ground_truth: The LLM extraction used as reference.
        model: A pre-loaded SentenceTransformer model. Pass None to fall back
            to word-overlap scoring via score_extraction_pair.

    Returns:
        Dict of per-field scores. When embeddings are used, includes additional
        key_findings_precision and key_findings_recall fields.
    """
    if model is None:
        logger.debug("embedding_model_not_provided", fallback="word_overlap")
        return score_extraction_pair(predicted, ground_truth)

    # Start with word-overlap scores for all fields
    scores = score_extraction_pair(predicted, ground_truth)

    # Override key_findings with embedding-based score
    pred_claims = [f.claim for f in predicted.key_findings]
    gold_claims = [f.claim for f in ground_truth.key_findings]

    embedding_scores, matched_pairs = _embedding_key_findings_score(pred_claims, gold_claims, model)
    scores.update(embedding_scores)

    # Override evidence_strength using Hungarian-matched pairs (sim > 0.5)
    # Uses partial credit: exact match = 1.0, one level apart = 0.5, 2+ = 0.0
    pred_findings = predicted.key_findings
    gold_findings = ground_truth.key_findings
    valid_pairs = [(pi, gi) for pi, gi, sim in matched_pairs if sim > 0.5]
    if not valid_pairs:
        scores["evidence_strength"] = 1.0 if not gold_findings else 0.0
    else:
        ev_score_sum = sum(
            _evidence_strength_score(
                pred_findings[pi].evidence_strength,
                gold_findings[gi].evidence_strength,
            )
            for pi, gi in valid_pairs
        )
        scores["evidence_strength"] = ev_score_sum / len(valid_pairs)

    # Override quantitative_result using Hungarian-matched pairs (sim > 0.5)
    quant_scores: list[float] = []
    for pi, gi, sim in matched_pairs:
        if sim <= 0.5:
            continue
        p_q = pred_findings[pi].quantitative_result or ""
        g_q = gold_findings[gi].quantitative_result or ""
        if not p_q and not g_q:
            quant_scores.append(1.0)
        elif p_q and g_q:
            quant_scores.append(_word_overlap_similarity(p_q, g_q))
        else:
            quant_scores.append(0.0)
    scores["quantitative_result"] = sum(quant_scores) / len(quant_scores) if quant_scores else 0.0

    # Override methods_summary with embedding similarity
    scores["methods_summary"] = _embedding_text_similarity(
        predicted.methods_summary, ground_truth.methods_summary, model
    )

    # Override limitations with embedding similarity
    scores["limitations"] = _embedding_text_similarity(
        predicted.limitations, ground_truth.limitations, model
    )

    return scores
