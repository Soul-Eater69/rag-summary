"""
Card-level candidate extraction (V5 architecture).

Generates candidate value streams from the new card's own content,
populating the 'summary' and 'chunk' CandidateEvidence source slots
that would otherwise remain zero.

Two extraction paths:

summary_candidates
    Derived from the new card's structured summary fields:
    - capability_tags → looked up in capability map → promoted VS names
    - direct_functions_canonical / implied_functions_canonical → looked up
      in capability cluster canonical_functions → promoted VS names
    Score: 0.5–0.8 depending on match type (capability_tag > function)

chunk_candidates
    Derived from scanning the raw cleaned card text:
    - capability map direct_cues and indirect_cues scanned in the text
    - matching cue density → per-VS chunk score
    Score: 0.3–0.7 depending on cue match density

These are intentionally lightweight:
- They do not call the LLM
- They do not replace KG, historical, or capability sources
- They exist to ensure the summary and chunk source slots are non-zero
  when the new card explicitly mentions or implies VS-relevant signals
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "capability_map.yaml"
_SIGNAL_RULES_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "stream_signal_rules.yaml"

# Minimum cue matches required to produce a chunk candidate
_MIN_CHUNK_CUES = 1
# Score for a capability_tag direct hit (high confidence)
_CAP_TAG_SCORE = 0.72
# Score for a canonical function hit (moderate confidence)
_FUNC_HIT_SCORE = 0.55
# Score scale for chunk candidates (max at high cue density)
_CHUNK_MAX_SCORE = 0.65
_CHUNK_BASE_SCORE = 0.30
# Minimum final score to emit a candidate after signal penalty
_MIN_EMIT_SCORE = 0.15

# Maps capability_map cluster names → taxonomy family names for signal rule lookup.
# Clusters without a family mapping get no signal adjustment.
_CLUSTER_FAMILY_MAP: Dict[str, str] = {
    "compliance_privacy_audit": "compliance_and_risk",
    "enterprise_risk_governance": "compliance_and_risk",
    "vendor_partner_onboarding": "partner_management",
    "billing_order_to_cash": "finance",
    "enrollment_quoting": "enrollment",
    "product_launch_commercialization": "product_and_pricing",
    "member_engagement_outreach": "marketing",
    "care_management_clinical": "care_management",
    "claims_adjudication": "claims",
    "portal_request_resolution": "member_services",
    "analytics_reporting": "data_and_analytics",
    "provider_network": "provider_network",
    "utilization_management_administration": "utilization",
    "quality_management": "quality",
    "health_management_programs": "care_management",
    "member_care_delivery": "care_management",
    "pharmacy_prescriptions": "pharmacy",
    "community_health": "care_management",
    "medicare_enrollment": "enrollment",
    "medicaid_enrollment": "enrollment",
    "individual_enrollment": "enrollment",
    "overpayment_recovery": "recovery",
    "payment_integrity": "compliance_and_risk",
    "appeals_grievances": "appeals_and_grievances",
    "audit_management": "compliance_and_risk",
    "regulatory_reporting": "compliance_and_risk",
    "privacy_incident": "compliance_and_risk",
    "it_strategy": "it_and_operations",
    "it_incident_management": "it_and_operations",
    "it_support": "it_and_operations",
    "hr_career_development": "it_and_operations",
    "strategy_planning": "strategy",
    "workforce_management": "it_and_operations",
    "hr_onboarding": "it_and_operations",
    "payroll": "it_and_operations",
}


def _load_signal_rules() -> Dict[str, Dict[str, Any]]:
    """Load stream signal rules keyed by family name."""
    if not _SIGNAL_RULES_PATH.exists():
        return {}
    try:
        with _SIGNAL_RULES_PATH.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        return dict(payload.get("families") or {})
    except Exception as exc:
        logger.warning("card_candidates: failed to load signal rules: %s", exc)
        return {}


def _compute_signal_adjustment(
    lower_text: str,
    family: str,
    signal_rules: Dict[str, Dict[str, Any]],
) -> float:
    """
    Compute score adjustment based on per-family signal rules.

    Returns a positive boost if enough positive signals are found,
    a negative penalty if only weak-only signals are present without
    any positive signals, or 0.0 if no rules apply.
    """
    rules = signal_rules.get(family)
    if not rules:
        return 0.0

    positive_signals: List[str] = rules.get("positive_signals") or []
    weak_only: List[str] = rules.get("weak_only_not_enough") or []
    min_positive = int(rules.get("min_positive_required") or 1)
    score_boost = float(rules.get("score_boost") or 0.0)
    score_penalty = float(rules.get("score_penalty") or 0.0)

    positive_matches = sum(1 for s in positive_signals if s in lower_text)
    if positive_matches >= min_positive:
        return score_boost

    # Only weak terms present — apply penalty
    weak_matches = sum(1 for s in weak_only if s in lower_text)
    if weak_matches > 0 and positive_matches == 0:
        return -score_penalty

    return 0.0


def _load_capability_map() -> Dict[str, Dict[str, Any]]:
    """Load capability map clusters keyed by cluster name."""
    if not _CONFIG_PATH.exists():
        return {}
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        caps = payload.get("capabilities") or {}
        result = {}
        for name, cluster in caps.items():
            if not isinstance(cluster, dict):
                continue
            promote = cluster.get("promote_value_streams") or cluster.get("promoted_value_streams") or []
            if not promote:
                continue
            result[name] = {
                "direct_cues": set(cluster.get("direct_cues") or []),
                "indirect_cues": set(cluster.get("indirect_cues") or []),
                "canonical_functions": set(cluster.get("canonical_functions") or []),
                "promote_value_streams": list(promote),
                "related_value_streams": list(cluster.get("related_value_streams") or []),
                "weight": float(cluster.get("weight") or 1.0),
            }
        return result
    except Exception as exc:
        logger.warning("card_candidates: failed to load capability map: %s", exc)
        return {}


def extract_summary_candidates(
    new_card_summary: Dict[str, Any],
    *,
    allowed_names: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate summary-source candidates from the new card's structured summary.

    Two inference paths:
    1. capability_tags field → if a tag matches a capability cluster name,
       promote that cluster's VS names with a high confidence score.
    2. direct/implied canonical functions → if a function appears in a
       cluster's canonical_functions, promote promoted VS names with a
       moderate score.

    Returns candidates shaped for build_candidate_evidence's summary source.
    """
    capability_map = _load_capability_map()
    if not capability_map:
        return []

    cap_tags: Set[str] = set(new_card_summary.get("capability_tags") or [])
    canon_funcs: Set[str] = set()
    for field in ("direct_functions_canonical", "implied_functions_canonical",
                  "direct_functions", "implied_functions"):
        for fn in new_card_summary.get(field) or []:
            if fn:
                canon_funcs.add(fn.lower().strip())

    candidates: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for cluster_name, cluster in capability_map.items():
        # Path 1: capability_tag direct match
        if cluster_name in cap_tags:
            score = _CAP_TAG_SCORE * cluster.get("weight", 1.0)
            for vs_name in cluster["promote_value_streams"]:
                if vs_name not in seen and (allowed_names is None or vs_name in allowed_names):
                    candidates.append({
                        "entity_name": vs_name,
                        "score": round(min(0.85, score), 3),
                        "source": "summary",
                        "match_reason": f"capability_tag:{cluster_name}",
                    })
                    seen.add(vs_name)
            # Related streams get attenuated score
            for vs_name in cluster.get("related_value_streams", []):
                if vs_name not in seen and (allowed_names is None or vs_name in allowed_names):
                    candidates.append({
                        "entity_name": vs_name,
                        "score": round(min(0.60, score * 0.6), 3),
                        "source": "summary",
                        "match_reason": f"capability_tag_related:{cluster_name}",
                    })
                    seen.add(vs_name)
            continue

        # Path 2: canonical function match
        cluster_funcs = cluster.get("canonical_functions", set())
        matched_funcs = canon_funcs & {f.lower() for f in cluster_funcs}
        if matched_funcs:
            score = _FUNC_HIT_SCORE * min(1.0, len(matched_funcs) / 2.0) * cluster.get("weight", 1.0)
            for vs_name in cluster["promote_value_streams"]:
                if vs_name not in seen and (allowed_names is None or vs_name in allowed_names):
                    candidates.append({
                        "entity_name": vs_name,
                        "score": round(min(0.75, score), 3),
                        "source": "summary",
                        "match_reason": f"canonical_function:{','.join(sorted(matched_funcs))}",
                    })
                    seen.add(vs_name)

    return candidates


def extract_chunk_candidates(
    cleaned_text: str,
    *,
    allowed_names: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate chunk-source candidates by scanning cleaned card text.

    Scans for direct and indirect cue terms from the capability map.
    Clusters with enough cue signal produce candidates with a chunk score
    proportional to match density.

    Returns candidates shaped for build_candidate_evidence's chunk source.
    """
    if not cleaned_text or not cleaned_text.strip():
        return []

    capability_map = _load_capability_map()
    if not capability_map:
        return []

    lower_text = cleaned_text.lower()
    signal_rules = _load_signal_rules()
    candidates: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for cluster_name, cluster in capability_map.items():
        matched_direct = [t for t in cluster.get("direct_cues", set()) if t in lower_text]
        matched_indirect = [t for t in cluster.get("indirect_cues", set()) if t in lower_text]

        total_matches = len(matched_direct) + len(matched_indirect)
        if total_matches < _MIN_CHUNK_CUES:
            continue

        # Chunk score: direct hits weight more than indirect
        direct_score = len(matched_direct) * 1.0
        indirect_score = len(matched_indirect) * 0.5
        raw_strength = (direct_score + indirect_score) / 5.0
        chunk_score = _CHUNK_BASE_SCORE + (_CHUNK_MAX_SCORE - _CHUNK_BASE_SCORE) * min(1.0, raw_strength)
        chunk_score *= cluster.get("weight", 1.0)

        # Apply signal rule boost/penalty for the cluster's family
        family = _CLUSTER_FAMILY_MAP.get(cluster_name, "")
        signal_adj = _compute_signal_adjustment(lower_text, family, signal_rules) if family else 0.0

        for vs_name in cluster["promote_value_streams"]:
            if vs_name in seen:
                continue
            if allowed_names is not None and vs_name not in allowed_names:
                continue
            final_score = round(min(_CHUNK_MAX_SCORE, chunk_score + signal_adj), 3)
            if final_score < _MIN_EMIT_SCORE:
                continue
            candidates.append({
                "entity_name": vs_name,
                "score": final_score,
                "source": "chunk",
                "match_reason": (
                    f"cues:{','.join((matched_direct + matched_indirect)[:4])}"
                    + (f",signal_adj:{signal_adj:+.2f}" if signal_adj != 0.0 else "")
                ),
            })
            seen.add(vs_name)

    return candidates


def extract_card_attachment_candidates(
    cleaned_text: str,
    *,
    allowed_names: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate attachment-source candidates from card-native attachment signals.

    The current pipeline receives idea cards as extracted text (from PPT/PDF).
    Attachments embedded in or referenced by the card are already present in
    ``cleaned_text`` — their content has been extracted during ingestion.

    This function detects attachment-indicator patterns in the card text:
    exhibits, appendices, tables, budget documents, scope documents,
    roadmaps, and file references. When such signals co-occur with
    domain-relevant cue terms from the capability map, we treat them as
    evidence that the card contains detailed supporting material for those
    domains — which warrants a moderate attachment-source score.

    This is card-native (not analog-proxy): it reasons about the NEW card's
    own content structure, not about what historical analogs happened to say.

    Score range: 0.35–0.58 (lower than chunk/summary; attachment signal is
    structural rather than semantic, so confidence is bounded).
    """
    if not cleaned_text or not cleaned_text.strip():
        return []

    capability_map = _load_capability_map()
    if not capability_map:
        return []

    lower_text = cleaned_text.lower()

    # Attachment indicator terms — suggest detailed supporting material is present
    _ATTACHMENT_INDICATORS = {
        "attachment", "attached", "appendix", "appendices", "exhibit",
        "table", "figure", "budget", "scope", "roadmap", "implementation plan",
        "project plan", "specification", "spec", "requirement",
        ".xlsx", ".xls", ".pdf", ".pptx", ".docx", "spreadsheet",
        "see below", "refer to", "as shown", "as described",
    }

    # Check how many attachment indicators appear in the card text
    attachment_signal_count = sum(1 for ind in _ATTACHMENT_INDICATORS if ind in lower_text)
    if attachment_signal_count == 0:
        return []

    # Attachment signal strength: 0.0–1.0 based on indicator density
    attachment_strength = min(1.0, attachment_signal_count / 4.0)

    _ATTACH_BASE_SCORE = 0.35
    _ATTACH_MAX_SCORE = 0.58

    signal_rules = _load_signal_rules()
    candidates: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for cluster_name, cluster in capability_map.items():
        # Check for domain cues co-occurring with attachment signals
        direct_hits = [t for t in cluster.get("direct_cues", set()) if t in lower_text]
        indirect_hits = [t for t in cluster.get("indirect_cues", set()) if t in lower_text]

        if not direct_hits and not indirect_hits:
            continue

        # Score: attachment strength × domain cue density × cluster weight
        cue_strength = (len(direct_hits) + 0.5 * len(indirect_hits)) / 4.0
        combined = attachment_strength * min(1.0, cue_strength)
        attach_score = _ATTACH_BASE_SCORE + (_ATTACH_MAX_SCORE - _ATTACH_BASE_SCORE) * combined
        attach_score *= cluster.get("weight", 1.0)

        # Apply signal rule adjustment — attachment boost signals get additional weight
        family = _CLUSTER_FAMILY_MAP.get(cluster_name, "")
        if family:
            rules = signal_rules.get(family, {})
            attach_boost_signals = rules.get("attachment_boost_signals") or []
            if any(s in lower_text for s in attach_boost_signals):
                attach_score += float(rules.get("score_boost") or 0.0) * 1.5
            else:
                # Apply regular signal adjustment (may penalize)
                adj = _compute_signal_adjustment(lower_text, family, signal_rules)
                attach_score += adj

        for vs_name in cluster["promote_value_streams"]:
            if vs_name in seen:
                continue
            if allowed_names is not None and vs_name not in allowed_names:
                continue
            final_score = round(min(_ATTACH_MAX_SCORE, attach_score), 3)
            if final_score < _MIN_EMIT_SCORE:
                continue
            candidates.append({
                "entity_name": vs_name,
                "score": final_score,
                "source": "attachment",
                "sub_source": "attachment_heuristic",
                "match_reason": (
                    f"card_attachment_signal:{attachment_signal_count}_indicators,"
                    f"cues:{','.join((direct_hits + indirect_hits)[:3])}"
                ),
            })
            seen.add(vs_name)

    return candidates


def extract_historical_footprint_candidates(
    analog_tickets: List[Dict[str, Any]],
    *,
    allowed_names: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate additional pattern candidates by exploiting richer FAISS metadata.

    After analog retrieval, each analog now carries capability_tags and
    operational_footprint. Cross-reference those with the capability map to
    promote additional VS names that the analogs collectively imply, even
    if those VS names weren't in their explicit value_stream_labels.

    This is how 'history as footprint memory' differs from 'history as
    label list': we infer downstream implications from what the analogs
    were doing, not just from what they were tagged with.

    Returns candidates shaped for build_candidate_evidence's historical source.
    """
    capability_map = _load_capability_map()
    if not capability_map or not analog_tickets:
        return []

    # Accumulate capability tags across all analogs weighted by similarity score
    cluster_signal: Dict[str, float] = {}
    for analog in analog_tickets:
        score = float(analog.get("score", 0.0))
        for tag in analog.get("capability_tags") or []:
            if tag in capability_map:
                cluster_signal[tag] = max(cluster_signal.get(tag, 0.0), score)

    candidates: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for cluster_name, analog_score in cluster_signal.items():
        cluster = capability_map.get(cluster_name, {})
        # Attenuate: these are pattern-inferred from analogs, not from the new card
        pattern_score = round(analog_score * 0.55, 3)

        for vs_name in cluster.get("promote_value_streams", []):
            if vs_name in seen:
                continue
            if allowed_names is not None and vs_name not in allowed_names:
                continue
            candidates.append({
                "entity_name": vs_name,
                "score": min(0.65, pattern_score),
                "source": "historical",
                "match_reason": f"analog_capability_tag:{cluster_name}",
            })
            seen.add(vs_name)

    return candidates
