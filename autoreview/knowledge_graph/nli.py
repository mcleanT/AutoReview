"""Cross-claim Natural Language Inference pipeline for the knowledge graph.

Detects contradictions between graph-edge claims using a DeBERTa NLI model and
updates Beta-Binomial confidence posteriors in-place on the graph.

Public API
----------
classify_cross_claims(graph, config)  -> CrossClaimNLIResult
    Main entry point.  Mutates graph edge data with confidence_mean,
    controversy_score, _nli_alpha, _nli_beta, _nli_cross_beta.

diagnose_evidence_directions(graph, config)  -> EvidenceDiagnosticResult
    Read-only diagnostic: runs NLI on evidence→claim pairs and returns
    a label distribution summary.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Any

import networkx as nx
import structlog

from autoreview.knowledge_graph.confidence import EVIDENCE_WEIGHTS
from autoreview.models.base import AutoReviewModel

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Predicate opposition families
# ---------------------------------------------------------------------------

_OPPOSING_PREDICATES: list[tuple[frozenset[str], frozenset[str]]] = [
    (
        frozenset(
            {
                "induces",
                "activates",
                "triggers",
                "initiates",
                "promotes",
                "stimulates",
                "upregulates",
            }
        ),
        frozenset(
            {
                "inhibits",
                "suppresses",
                "blocks",
                "represses",
                "downregulates",
                "prevents",
                "attenuates",
            }
        ),
    ),
    (
        frozenset(
            {
                "is_required_for",
                "is_necessary_for",
                "is_essential_for",
                "is_critical_for",
            }
        ),
        frozenset(
            {
                "is_not_required_for",
                "is_not_necessary_for",
                "is_not_essential_for",
            }
        ),
    ),
    (
        frozenset({"contains", "expresses", "produces"}),
        frozenset({"does_not_contain", "does_not_express", "lacks"}),
    ),
    (
        frozenset({"regulates", "modulates", "controls", "mediates"}),
        frozenset(
            {
                "does_not_regulate",
                "does_not_affect",
                "does_not_modulate",
            }
        ),
    ),
    (
        frozenset(
            {
                "generates",
                "gives_rise_to",
                "develops_into",
                "differentiates_into",
            }
        ),
        frozenset(
            {
                "does_not_generate",
                "fails_to_generate",
                "fail_to_generate",
            }
        ),
    ),
    (
        frozenset({"affects", "influences", "impacts", "alters"}),
        frozenset(
            {
                "does_not_affect",
                "does_not_alter",
                "does_not_influence",
                "does_not_impact",
            }
        ),
    ),
]


# ---------------------------------------------------------------------------
# Config and result models
# ---------------------------------------------------------------------------


class NLIConfig(AutoReviewModel):
    """Configuration for the cross-claim NLI pipeline.

    Attributes:
        model_name: HuggingFace model identifier for the NLI cross-encoder.
        batch_size: Number of (premise, hypothesis) pairs per inference batch.
        max_length: Maximum tokenized sequence length.
        contradiction_threshold: Minimum p_contradiction to include a pair in
            results and apply Beta updates.
        filter_parallel_assertions: If True, skip pairs that are parallel
            assertions (same subject+predicate, different object, or vice versa).
        use_predicate_opposition: If True, resolve structurally opposing
            predicates without running NLI.
        max_shared_entities_per_pair: Cap on how many shared-entity links are
            stored per claim pair.
        device: Compute device — "auto", "mps", "cuda", or "cpu".
        context_mismatch_discount: Discount factor applied to p_contradiction
            when claims come from different experimental contexts (organism,
            model_system, in_vitro). 0.3 means the contradiction probability
            is multiplied by 0.3.
    """

    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli-ling-wanli"
    batch_size: int = 64
    max_length: int = 256
    contradiction_threshold: float = 0.3
    filter_parallel_assertions: bool = True
    use_predicate_opposition: bool = True
    max_shared_entities_per_pair: int = 5
    device: str = "auto"
    context_mismatch_discount: float = 0.3


class NLIPairResult(AutoReviewModel):
    """NLI result for a single claim pair.

    Attributes:
        claim_a_id: Edge key string for claim A.
        claim_b_id: Edge key string for claim B.
        p_contradiction: Probability of contradiction label.
        p_entailment: Probability of entailment label.
        p_neutral: Probability of neutral label.
        method: How the classification was produced — "nli",
            "parallel_skip", "predicate_opposition", or
            "nli_context_discounted".
        shared_entities: Canonical entity names shared by both claims.
        context_mismatch: Description of the experimental context mismatch
            that triggered a discount, or None if no discount was applied.
        original_p_contradiction: Pre-discount p_contradiction value, or None
            if no discount was applied.
    """

    claim_a_id: str
    claim_b_id: str
    p_contradiction: float
    p_entailment: float
    p_neutral: float
    method: str
    shared_entities: list[str]
    context_mismatch: str | None = None
    original_p_contradiction: float | None = None
    contradiction_type: str | None = None


class CrossClaimNLIResult(AutoReviewModel):
    """Aggregate result from classify_cross_claims.

    Attributes:
        total_pairs: Total claim pairs examined.
        parallel_skipped: Pairs skipped as parallel assertions.
        structural_resolved: Pairs resolved via predicate opposition rules.
        nli_classified: Pairs sent through the DeBERTa model.
        contradictions_p05: Pairs with p_contradiction >= 0.5.
        contradictions_p08: Pairs with p_contradiction >= 0.8.
        claims_updated: Number of graph edges whose posteriors were updated.
        context_discounted: Pairs whose p_contradiction was discounted due to
            experimental context mismatch.
        pair_results: Per-pair results for all pairs above contradiction_threshold.
    """

    total_pairs: int
    parallel_skipped: int
    structural_resolved: int
    nli_classified: int
    contradictions_p05: int
    contradictions_p08: int
    claims_updated: int
    context_discounted: int = 0
    pair_results: list[NLIPairResult]


class EvidenceDiagnosticResult(AutoReviewModel):
    """Result from diagnose_evidence_directions.

    Attributes:
        total_pairs: Total (evidence, claim) pairs evaluated.
        label_distribution: Counts keyed by "contradiction", "entailment",
            "neutral".
        results: Per-pair dicts with keys edge_key, evidence_id, premise,
            hypothesis, label, p_contradiction, p_entailment, p_neutral.
    """

    total_pairs: int
    label_distribution: dict[str, int]
    results: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_device(preference: str) -> Any:  # returns torch.device
    """Resolve the compute device based on a preference string.

    Args:
        preference: One of "auto", "mps", "cuda", or "cpu".

    Returns:
        A torch.device object for the best available device.
    """
    import torch  # lazy import

    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # "auto" — prefer CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def _load_model(
    model_name: str, device_str: str
) -> tuple[Any, Any, Any]:  # (model, tokenizer, device)
    """Load the NLI model and tokenizer, cached for the process lifetime.

    Args:
        model_name: HuggingFace model identifier.
        device_str: String representation of the target device (used as cache
            key — the actual device object is constructed inside).

    Returns:
        Tuple of (model, tokenizer, torch.device).
    """
    import torch  # lazy
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # lazy

    device = torch.device(device_str)
    log.info("nli.load_model", model=model_name, device=str(device))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _build_claims(
    graph: nx.MultiDiGraph,
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    """Build claim text and entity index from graph edges.

    Each edge becomes a claim. Text is sourced from (in priority order):
    1. ``natural_language`` field from the KGEdge (full sentence)
    2. Enriched triple with model_system/organism/conditions context
    3. Bare triple: ``"{subject_name} {predicate} {object_name}"``

    The subject and object canonical names are resolved from node attributes
    on the graph.

    Args:
        graph: The knowledge graph.

    Returns:
        A two-element tuple:
            - claims_by_key: Mapping from edge_key str to a dict with keys
              edge_key, subj_id, obj_id, predicate, text,
              subject_name, object_name, edge_data.
            - entity_to_claim_ids: Mapping from canonical entity name to the
              set of edge_key strings that involve that entity.
    """
    claims_by_key: dict[str, dict[str, Any]] = {}
    entity_to_claim_ids: dict[str, set[str]] = defaultdict(set)

    for u, v, k, data in graph.edges(keys=True, data=True):
        predicate: str = data.get("predicate", "relates_to")
        subj_name: str = graph.nodes[u].get("canonical_name", str(u))
        obj_name: str = graph.nodes[v].get("canonical_name", str(v))
        edge_key = f"{u}__{predicate}__{v}__{k}"

        # Prefer natural_language > enriched triple > bare triple
        kg_edge = data.get("_kg_edge")
        natural_lang = None
        if kg_edge:
            natural_lang = getattr(kg_edge, "natural_language", None)
        if not natural_lang:
            natural_lang = data.get("natural_language")

        if natural_lang:
            claim_text = natural_lang
        else:
            # Build enriched text with available context
            parts = [f"{subj_name} {predicate} {obj_name}"]
            model_sys = (getattr(kg_edge, "model_system", None) if kg_edge else None) or data.get(
                "model_system"
            )
            organism = (getattr(kg_edge, "organism", None) if kg_edge else None) or data.get(
                "organism"
            )
            conditions = (getattr(kg_edge, "conditions", None) if kg_edge else None) or data.get(
                "conditions"
            )
            if model_sys:
                parts.append(f"in {model_sys}")
            if organism:
                parts.append(f"({organism})")
            if conditions and isinstance(conditions, dict):
                cond_strs = [f"{k}: {v}" for k, v in conditions.items() if v]
                if cond_strs:
                    parts.append(f"[{', '.join(cond_strs)}]")
            claim_text = " ".join(parts)

        claims_by_key[edge_key] = {
            "edge_key": edge_key,
            "subj_id": u,
            "obj_id": v,
            "key": k,
            "predicate": predicate,
            "text": claim_text,
            "subject_name": subj_name,
            "object_name": obj_name,
            "edge_data": data,
            "condition_signature": data.get("condition_signature"),
        }
        entity_to_claim_ids[subj_name].add(edge_key)
        entity_to_claim_ids[obj_name].add(edge_key)

    return claims_by_key, dict(entity_to_claim_ids)


def _find_shared_entity_pairs(
    entity_to_claims: dict[str, set[str]],
    max_entities: int,
) -> dict[tuple[str, str], list[str]]:
    """Generate canonical claim-pair keys that share at least one entity.

    Args:
        entity_to_claims: Mapping from entity name to set of edge_key strings.
        max_entities: Maximum number of shared entity names to record per pair.

    Returns:
        Mapping from (claim_a_id, claim_b_id) tuple (alphabetically sorted) to
        the list of shared entity canonical names (capped at max_entities).
    """
    pair_entities: dict[tuple[str, str], list[str]] = defaultdict(list)

    for entity_name, claim_ids in entity_to_claims.items():
        claim_list = sorted(claim_ids)  # deterministic ordering
        for i in range(len(claim_list)):
            for j in range(i + 1, len(claim_list)):
                a, b = claim_list[i], claim_list[j]
                key = (a, b)
                if len(pair_entities[key]) < max_entities:
                    pair_entities[key].append(entity_name)

    return dict(pair_entities)


def _is_parallel_assertion(claim_a: dict[str, Any], claim_b: dict[str, Any]) -> bool:
    """Return True if two claims are parallel assertions.

    Parallel assertions share the same predicate and subject (but differ in
    object), or share the same predicate and object (but differ in subject).
    Such pairs describe related but non-contradictory statements and should
    be skipped.

    Args:
        claim_a: Claim dict with keys subj_id, obj_id, predicate.
        claim_b: Claim dict with keys subj_id, obj_id, predicate.

    Returns:
        True if the pair is a parallel assertion, False otherwise.
    """
    if claim_a["predicate"] != claim_b["predicate"]:
        return False
    # Same subject, different object
    if claim_a["subj_id"] == claim_b["subj_id"] and claim_a["obj_id"] != claim_b["obj_id"]:
        return True
    # Same object, different subject
    return claim_a["obj_id"] == claim_b["obj_id"] and claim_a["subj_id"] != claim_b["subj_id"]


def _predicates_oppose(pred_a: str, pred_b: str) -> float | None:
    """Check whether two predicates are structurally opposing.

    Args:
        pred_a: Predicate string for claim A.
        pred_b: Predicate string for claim B.

    Returns:
        1.0 if the predicates belong to opposing sets in
        _OPPOSING_PREDICATES, None otherwise.
    """
    for pos_set, neg_set in _OPPOSING_PREDICATES:
        if (pred_a in pos_set and pred_b in neg_set) or (pred_a in neg_set and pred_b in pos_set):
            return 1.0
    return None


def _contexts_mismatch(claim_a: dict[str, Any], claim_b: dict[str, Any]) -> str | None:
    """Check if two claims come from different experimental contexts.

    Returns a string describing the mismatch type, or None if contexts are
    compatible (same or unknown).

    Args:
        claim_a: Claim dict with edge_data containing v5 fields.
        claim_b: Claim dict with edge_data containing v5 fields.

    Returns:
        Mismatch description string, or None if compatible.
    """
    data_a = claim_a["edge_data"]
    data_b = claim_b["edge_data"]

    # Check organism mismatch
    org_a = data_a.get("organism")
    org_b = data_b.get("organism")
    if org_a and org_b and org_a.lower() != org_b.lower():
        return f"organism: {org_a} vs {org_b}"

    # Check in_vitro vs in_vivo mismatch
    vitro_a = data_a.get("in_vitro")
    vitro_b = data_b.get("in_vitro")
    if vitro_a is not None and vitro_b is not None and vitro_a != vitro_b:
        return f"in_vitro: {vitro_a} vs {vitro_b}"

    # Check model_system mismatch (fuzzy — only flag clearly different systems)
    ms_a = data_a.get("model_system")
    ms_b = data_b.get("model_system")
    if ms_a and ms_b:
        # Normalize for comparison
        ms_a_lower = ms_a.lower().strip()
        ms_b_lower = ms_b.lower().strip()
        if ms_a_lower != ms_b_lower:
            # Only flag as mismatch if they don't share key terms
            tokens_a = set(ms_a_lower.split())
            tokens_b = set(ms_b_lower.split())
            overlap = tokens_a & tokens_b
            # If less than 30% overlap, consider them different systems
            min_size = min(len(tokens_a), len(tokens_b))
            if min_size > 0 and len(overlap) / min_size < 0.3:
                return f"model_system: {ms_a} vs {ms_b}"

    return None


def _classify_contradiction_type(
    claim_a: dict[str, Any],
    claim_b: dict[str, Any],
) -> str:
    """Classify the type of contradiction between two claims.

    Uses condition signatures and predicate relationships to determine
    whether a contradiction is within-context, cross-context, structural,
    or NLI-semantic.

    Args:
        claim_a: Claim dict with subj_id, obj_id, predicate, edge_data.
        claim_b: Claim dict with subj_id, obj_id, predicate, edge_data.

    Returns:
        One of: "within_context", "cross_context", "structural", "nli_semantic".
    """
    same_subject = claim_a["subj_id"] == claim_b["subj_id"]
    same_object = claim_a["obj_id"] == claim_b["obj_id"]
    same_predicate = claim_a["predicate"] == claim_b["predicate"]

    sig_a = claim_a.get("edge_data", {}).get("condition_signature")
    sig_b = claim_b.get("edge_data", {}).get("condition_signature")
    same_condition = sig_a is not None and sig_b is not None and sig_a == sig_b

    if same_subject and same_object:
        # Check for structural opposition (opposing predicates)
        if not same_predicate and same_condition:
            opposition = _predicates_oppose(claim_a["predicate"], claim_b["predicate"])
            if opposition is not None:
                return "structural"

        if same_predicate:
            if same_condition:
                return "within_context"
            if sig_a is not None and sig_b is not None:
                return "cross_context"

    return "nli_semantic"


def _batch_nli_classify(
    pairs: list[tuple[str, str]],
    claims: dict[str, dict[str, Any]],
    model: Any,
    tokenizer: Any,
    device: Any,
    config: NLIConfig,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Run batched NLI inference for a list of claim-pair keys.

    Label indices are auto-detected from ``model.config.id2label`` to handle
    both cross-encoder and MoritzLaurer label orders without hardcoding.

    Args:
        pairs: List of (claim_a_id, claim_b_id) tuples to classify.
        claims: Mapping from edge_key to claim dict.
        model: Loaded HuggingFace sequence-classification model.
        tokenizer: Corresponding tokenizer.
        device: torch.device to run inference on.
        config: NLI configuration (batch_size, max_length).

    Returns:
        Mapping from (claim_a_id, claim_b_id) to a dict with keys
        p_contra, p_entail, p_neutral, method.
    """
    import torch  # lazy

    results: dict[tuple[str, str], dict[str, Any]] = {}

    # Auto-detect label indices from model config (handles both cross-encoder
    # and MoritzLaurer label orders without hardcoding)
    id2label = getattr(model.config, "id2label", {})
    contra_idx = next((int(k) for k, v in id2label.items() if "contradiction" in v.lower()), 0)
    entail_idx = next((int(k) for k, v in id2label.items() if "entailment" in v.lower()), 1)
    neutral_idx = next((int(k) for k, v in id2label.items() if "neutral" in v.lower()), 2)
    log.debug(
        "nli.label_indices", contradiction=contra_idx, entailment=entail_idx, neutral=neutral_idx
    )

    for batch_start in range(0, len(pairs), config.batch_size):
        batch_pairs = pairs[batch_start : batch_start + config.batch_size]
        premises = [claims[a]["text"] for a, _ in batch_pairs]
        hypotheses = [claims[b]["text"] for _, b in batch_pairs]

        encoding = tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_length,
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = model(**encoding).logits  # (batch, 3)
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

        for (a_id, b_id), prob_row in zip(batch_pairs, probs, strict=False):
            results[(a_id, b_id)] = {
                "p_contra": float(prob_row[contra_idx]),
                "p_entail": float(prob_row[entail_idx]),
                "p_neutral": float(prob_row[neutral_idx]),
                "method": "nli",
            }

        log.debug(
            "nli.batch_complete",
            batch_end=batch_start + len(batch_pairs),
            total_pairs=len(pairs),
        )

    return results


def _update_graph_posteriors(
    graph: nx.MultiDiGraph,
    nli_results: dict[tuple[str, str], dict[str, Any]],
    claims: dict[str, dict[str, Any]],
    contradiction_threshold: float,
) -> int:
    """Apply Beta-Binomial updates to graph edges from NLI contradiction scores.

    For each pair where p_contra >= threshold:
    - B's evidence becomes counter-evidence for A (beta_updates[A] += p_contra * weight)
    - A's evidence becomes counter-evidence for B (beta_updates[B] += p_contra * weight)

    Final per-edge update:
    - base_alpha = 1.0 + sum(EVIDENCE_WEIGHTS[ev.evidence_strength] for each evidence_link)
    - base_beta  = 1.0
    - new_beta   = base_beta + beta_updates[edge_key]
    - confidence_mean    = alpha / (alpha + new_beta)
    - controversy_score  = min(alpha, new_beta) / max(alpha, new_beta)
    - Stored on edge data: confidence_mean, controversy_score, _nli_alpha, _nli_beta,
      _nli_cross_beta

    Args:
        graph: The knowledge graph (mutated in-place).
        nli_results: NLI output dict from _batch_nli_classify or structural
            resolution, keyed by (claim_a_id, claim_b_id).
        claims: Mapping from edge_key to claim dict (includes edge_data ref).
        contradiction_threshold: Minimum p_contra to apply an update.

    Returns:
        Number of graph edges that received a posterior update.
    """
    beta_updates: dict[str, float] = defaultdict(float)

    for (c1, c2), nli_out in nli_results.items():
        p_contra = nli_out["p_contra"]
        if p_contra < contradiction_threshold:
            continue

        claim_a = claims[c1]
        claim_b = claims[c2]

        # B's evidence → counter-evidence for A
        kg_edge_b = claim_b["edge_data"].get("_kg_edge")
        evidence_links_b = kg_edge_b.evidence_links if kg_edge_b else []
        for ev in evidence_links_b:
            strength = (
                ev.evidence_strength
                if hasattr(ev, "evidence_strength")
                else ev.get("evidence_strength", "expert_opinion")
            )
            weight = EVIDENCE_WEIGHTS.get(str(strength), 0.2)
            beta_updates[c1] += p_contra * weight

        # A's evidence → counter-evidence for B
        kg_edge_a = claim_a["edge_data"].get("_kg_edge")
        evidence_links_a = kg_edge_a.evidence_links if kg_edge_a else []
        for ev in evidence_links_a:
            strength = (
                ev.evidence_strength
                if hasattr(ev, "evidence_strength")
                else ev.get("evidence_strength", "expert_opinion")
            )
            weight = EVIDENCE_WEIGHTS.get(str(strength), 0.2)
            beta_updates[c2] += p_contra * weight

    updated_count = 0
    keys_to_update = set(beta_updates.keys())

    for edge_key, claim in claims.items():
        if edge_key not in keys_to_update:
            continue

        u = claim["subj_id"]
        v = claim["obj_id"]
        k = claim["key"]
        data = claim["edge_data"]

        kg_edge = data.get("_kg_edge")
        evidence_links = kg_edge.evidence_links if kg_edge else []
        base_alpha = 1.0
        for ev in evidence_links:
            strength = (
                ev.evidence_strength
                if hasattr(ev, "evidence_strength")
                else ev.get("evidence_strength", "expert_opinion")
            )
            base_alpha += EVIDENCE_WEIGHTS.get(str(strength), 0.2)

        base_beta = 1.0
        cross_beta = beta_updates[edge_key]
        new_beta = base_beta + cross_beta

        confidence_mean = base_alpha / (base_alpha + new_beta)
        controversy_score = min(base_alpha, new_beta) / max(base_alpha, new_beta)

        graph[u][v][k]["confidence_mean"] = confidence_mean
        graph[u][v][k]["controversy_score"] = controversy_score
        graph[u][v][k]["_nli_alpha"] = base_alpha
        graph[u][v][k]["_nli_beta"] = new_beta
        graph[u][v][k]["_nli_cross_beta"] = cross_beta

        updated_count += 1

    return updated_count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_cross_claims(
    graph: nx.MultiDiGraph,
    config: NLIConfig | None = None,
) -> CrossClaimNLIResult:
    """Detect contradictions between graph-edge claims and update confidence posteriors.

    Mutates graph in-place: adds confidence_mean, controversy_score,
    _nli_alpha, _nli_beta, _nli_cross_beta to edge data dicts for all edges
    that receive Beta updates.

    Pipeline steps:
        1. Build a claim dict and entity→claims index from graph edges.
        2. Generate candidate pairs via shared-entity lookup.
        3. Pre-filter parallel assertions (skip with p_contra=0.0).
        4. Pre-filter structurally opposing predicates (set p_contra=1.0).
        5. Run batched DeBERTa NLI on remaining pairs.
        6. Update Beta-Binomial posteriors on graph edges.

    Args:
        graph: The knowledge graph (NetworkX MultiDiGraph).
        config: NLI configuration.  Defaults to NLIConfig().

    Returns:
        CrossClaimNLIResult with aggregate statistics and per-pair results
        for all pairs above contradiction_threshold.
    """
    if config is None:
        config = NLIConfig()

    log.info(
        "nli.classify_cross_claims.start",
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
    )

    claims, entity_to_claims = _build_claims(graph)
    if not claims:
        log.warning("nli.classify_cross_claims.no_claims")
        return CrossClaimNLIResult(
            total_pairs=0,
            parallel_skipped=0,
            structural_resolved=0,
            nli_classified=0,
            contradictions_p05=0,
            contradictions_p08=0,
            claims_updated=0,
            pair_results=[],
        )

    pair_shared: dict[tuple[str, str], list[str]] = _find_shared_entity_pairs(
        entity_to_claims, config.max_shared_entities_per_pair
    )

    total_pairs = len(pair_shared)
    log.info("nli.classify_cross_claims.pairs_found", total=total_pairs)

    parallel_skipped = 0
    structural_resolved = 0
    nli_pairs: list[tuple[str, str]] = []
    all_nli_results: dict[tuple[str, str], dict[str, Any]] = {}

    for (a_id, b_id), _shared_entities in pair_shared.items():
        claim_a = claims[a_id]
        claim_b = claims[b_id]

        if config.filter_parallel_assertions and _is_parallel_assertion(claim_a, claim_b):
            all_nli_results[(a_id, b_id)] = {
                "p_contra": 0.0,
                "p_entail": 1.0,
                "p_neutral": 0.0,
                "method": "parallel_skip",
            }
            parallel_skipped += 1
            continue

        if config.use_predicate_opposition:
            opposition_score = _predicates_oppose(claim_a["predicate"], claim_b["predicate"])
            if opposition_score is not None:
                all_nli_results[(a_id, b_id)] = {
                    "p_contra": opposition_score,
                    "p_entail": 0.0,
                    "p_neutral": 0.0,
                    "method": "predicate_opposition",
                }
                structural_resolved += 1
                continue

        nli_pairs.append((a_id, b_id))

    nli_classified = len(nli_pairs)
    if nli_pairs:
        device = _resolve_device(config.device)
        model, tokenizer, device = _load_model(config.model_name, str(device))
        nli_batch_results = _batch_nli_classify(nli_pairs, claims, model, tokenizer, device, config)
        all_nli_results.update(nli_batch_results)

    log.info(
        "nli.classify_cross_claims.inference_done",
        parallel_skipped=parallel_skipped,
        structural_resolved=structural_resolved,
        nli_classified=nli_classified,
    )

    # Apply context mismatch discount
    context_discounted = 0
    for a_id, b_id in list(all_nli_results.keys()):
        nli_out = all_nli_results[(a_id, b_id)]
        if nli_out["method"] in ("parallel_skip", "predicate_opposition"):
            continue
        claim_a = claims.get(a_id)
        claim_b = claims.get(b_id)
        if claim_a and claim_b:
            mismatch = _contexts_mismatch(claim_a, claim_b)
            if mismatch:
                original_p = nli_out["p_contra"]
                discounted_p = original_p * config.context_mismatch_discount
                all_nli_results[(a_id, b_id)] = {
                    **nli_out,
                    "p_contra": discounted_p,
                    "method": "nli_context_discounted",
                    "context_mismatch": mismatch,
                    "original_p_contra": original_p,
                }
                context_discounted += 1

    log.info(
        "nli.classify_cross_claims.context_discount",
        context_discounted=context_discounted,
    )

    updated_count = _update_graph_posteriors(
        graph, all_nli_results, claims, config.contradiction_threshold
    )

    # Build pair_results (only above threshold, excluding parallel_skip)
    pair_results: list[NLIPairResult] = []
    contradictions_p05 = 0
    contradictions_p08 = 0

    for (a_id, b_id), nli_out in all_nli_results.items():
        p_contra = nli_out["p_contra"]
        method = nli_out["method"]

        if p_contra >= 0.5:
            contradictions_p05 += 1
        if p_contra >= 0.8:
            contradictions_p08 += 1

        if p_contra >= config.contradiction_threshold and method != "parallel_skip":
            shared = pair_shared.get((a_id, b_id), [])
            contradiction_type = _classify_contradiction_type(claims[a_id], claims[b_id])
            pair_results.append(
                NLIPairResult(
                    claim_a_id=a_id,
                    claim_b_id=b_id,
                    p_contradiction=p_contra,
                    p_entailment=nli_out["p_entail"],
                    p_neutral=nli_out["p_neutral"],
                    method=method,
                    shared_entities=shared,
                    context_mismatch=nli_out.get("context_mismatch"),
                    original_p_contradiction=nli_out.get("original_p_contra"),
                    contradiction_type=contradiction_type,
                )
            )

    pair_results.sort(key=lambda r: r.p_contradiction, reverse=True)

    log.info(
        "nli.classify_cross_claims.done",
        contradictions_p05=contradictions_p05,
        contradictions_p08=contradictions_p08,
        claims_updated=updated_count,
    )

    return CrossClaimNLIResult(
        total_pairs=total_pairs,
        parallel_skipped=parallel_skipped,
        structural_resolved=structural_resolved,
        nli_classified=nli_classified,
        contradictions_p05=contradictions_p05,
        contradictions_p08=contradictions_p08,
        claims_updated=updated_count,
        context_discounted=context_discounted,
        pair_results=pair_results,
    )


def diagnose_evidence_directions(
    graph: nx.MultiDiGraph,
    config: NLIConfig | None = None,
) -> EvidenceDiagnosticResult:
    """Run NLI on evidence→claim pairs to diagnose evidence label distribution.

    Does NOT modify the graph.  For each edge's evidence_links, the evidence
    experiment_summary is used as the NLI premise and the edge claim text as
    the hypothesis.

    Args:
        graph: The knowledge graph (read-only).
        config: NLI configuration.  Defaults to NLIConfig().

    Returns:
        EvidenceDiagnosticResult with total pair count, label distribution
        counts, and per-pair result dicts.
    """
    import torch  # lazy

    if config is None:
        config = NLIConfig()

    claims, _ = _build_claims(graph)

    premises: list[str] = []
    hypotheses: list[str] = []
    metadata: list[dict[str, str]] = []

    for edge_key, claim in claims.items():
        evidence_links = claim["edge_data"].get("evidence_links", [])
        for ev in evidence_links:
            evidence_id = (
                ev.evidence_id if hasattr(ev, "evidence_id") else ev.get("evidence_id", "")
            )
            summary = (
                ev.experiment_summary
                if hasattr(ev, "experiment_summary")
                else ev.get("experiment_summary", "")
            )
            if not summary:
                continue
            premises.append(summary)
            hypotheses.append(claim["text"])
            metadata.append({"edge_key": edge_key, "evidence_id": str(evidence_id)})

    total_pairs = len(premises)
    log.info("nli.diagnose_evidence_directions.start", total_pairs=total_pairs)

    if total_pairs == 0:
        return EvidenceDiagnosticResult(
            total_pairs=0,
            label_distribution={"contradiction": 0, "entailment": 0, "neutral": 0},
            results=[],
        )

    device = _resolve_device(config.device)
    model, tokenizer, device = _load_model(config.model_name, str(device))

    label_map = {0: "contradiction", 1: "entailment", 2: "neutral"}
    label_distribution: dict[str, int] = {"contradiction": 0, "entailment": 0, "neutral": 0}
    per_pair_results: list[dict[str, Any]] = []

    for batch_start in range(0, total_pairs, config.batch_size):
        batch_end = batch_start + config.batch_size
        batch_premises = premises[batch_start:batch_end]
        batch_hyp = hypotheses[batch_start:batch_end]
        batch_meta = metadata[batch_start:batch_end]

        encoding = tokenizer(
            batch_premises,
            batch_hyp,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_length,
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = model(**encoding).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

        for meta, prob_row, premise, hyp in zip(  # noqa: B905
            batch_meta, probs, batch_premises, batch_hyp, strict=False
        ):
            label_idx = int(max(range(3), key=lambda i: prob_row[i]))
            label_name = label_map[label_idx]
            label_distribution[label_name] += 1
            per_pair_results.append(
                {
                    "edge_key": meta["edge_key"],
                    "evidence_id": meta["evidence_id"],
                    "premise": premise,
                    "hypothesis": hyp,
                    "label": label_name,
                    "p_contradiction": float(prob_row[0]),
                    "p_entailment": float(prob_row[1]),
                    "p_neutral": float(prob_row[2]),
                }
            )

    log.info(
        "nli.diagnose_evidence_directions.done",
        distribution=label_distribution,
    )

    return EvidenceDiagnosticResult(
        total_pairs=total_pairs,
        label_distribution=label_distribution,
        results=per_pair_results,
    )
