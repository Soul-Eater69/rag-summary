"""
Capability mapping and candidate enrichment for summary-RAG runtime.

This layer bridges business-language cues to value-stream candidates.
It is intended to run after analog/KG evidence gathering and before
final LLM selection.
"""

from __future__ import annotations

from copy import deepcopy
import logging
import pathlib
from typing import Any, Dict, List, Optional, Set

import yaml

from core.text import normalize_for_search

logger = logging.getLogger(__name__)

_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "capability_map.yaml"

_SUMMARY_FIELDS_FOR_CUES = [
    "short_summary",
    "business_goal",
    "actors",
    "direct_functions",
    "implied_functions",
    "change_types",
    "domain_tags",
    "evidence_sentences",
]

_FALLBACK_CAPABILITY_CLUSTERS = [
    {
        "capability_cluster": "compliance_privacy_audit",
        "capability": "Compliance / Privacy / Audit",
        "description": "Privacy, auditability, governance controls, and regulatory obligations.",
        "direct_cues": {
            "privacy", "pii", "audit", "controls", "balancing", "regulatory",
            "compliance", "hipaa", "consent", "data governance", "security controls",
        },
        "indirect_cues": {
            "vendor/member data handling", "governed data sharing", "review and approval controls",
            "oversight", "policy enforcement",
        },
        "promoted_value_streams": ["Ensure Compliance"],
        "related_value_streams": ["Manage Enterprise Risk"],
        "strength": 1.0,
        "score_boost": 0.35,
    },
    {
        "capability_cluster": "enterprise_risk_governance",
        "capability": "Enterprise Risk / Governance",
        "description": "Enterprise risk posture, governance processes, and risk controls.",
        "direct_cues": {
            "risk", "governance", "control framework", "oversight", "enterprise risk",
            "policy", "regulatory risk", "audit readiness",
        },
        "indirect_cues": {"mitigation plan", "governance review", "risk gating"},
        "promoted_value_streams": ["Manage Enterprise Risk"],
        "related_value_streams": ["Ensure Compliance"],
        "strength": 0.9,
        "score_boost": 0.25,
    },
    {
        "capability_cluster": "provider_onboarding_network",
        "capability": "Provider Onboarding / Network",
        "description": "Provider setup, contracting, credentialing, and network expansion.",
        "direct_cues": {
            "provider setup", "provider onboarding", "contracting", "network expansion",
            "credentialing", "provider network", "provider program",
        },
        "indirect_cues": {"new provider group", "network growth", "provider enablement"},
        "promoted_value_streams": ["Establish Provider Network", "Establish Provider Program"],
        "related_value_streams": [],
        "strength": 0.8,
        "score_boost": 0.25,
    },
    {
        "capability_cluster": "billing_order_to_cash",
        "capability": "Billing / Order-to-Cash",
        "description": "Invoicing, payment lifecycle, collections, and remittance operations.",
        "direct_cues": {
            "invoice", "invoicing", "billing", "payment", "collections", "remittance",
            "order to cash", "accounts receivable",
        },
        "indirect_cues": {"billing operations", "payment processing", "revenue cycle"},
        "promoted_value_streams": ["Order to Cash", "Manage Invoice and Payment"],
        "related_value_streams": [],
        "strength": 0.8,
        "score_boost": 0.25,
    },
    {
        "capability_cluster": "enrollment_quoting",
        "capability": "Enrollment / Quoting",
        "description": "Quoting, pricing configuration, and enrollment-related business flow.",
        "direct_cues": {
            "enrollment", "enrolment", "quoting", "quote", "pricing", "rate card",
            "sell", "configure price", "configure quote",
        },
        "indirect_cues": {"plan selection", "quote workflow", "broker enrollment"},
        "promoted_value_streams": ["Sell and Enroll", "Configure Price and Quote"],
        "related_value_streams": [],
        "strength": 0.8,
        "score_boost": 0.25,
    },
]

def _load_capability_clusters() -> List[Dict[str, Any]]:
    if not _CONFIG_PATH.exists():
        logger.info("Capability map config not found at %s; using fallback clusters.", _CONFIG_PATH)
        return _FALLBACK_CAPABILITY_CLUSTERS

    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            payload = yaml.safe_load(config_file) or {}
        clusters = payload.get("clusters")
        if not isinstance(clusters, list) or not clusters:
            logger.warning("Capability map at %s has no clusters; using fallback.", _CONFIG_PATH)
            return _FALLBACK_CAPABILITY_CLUSTERS
        normalized_clusters: List[Dict[str, Any]] = []
        for cluster in clusters:
            if not isinstance(cluster, dict):
                continue
            normalized_clusters.append(
                {
                    "capability_cluster": cluster.get("capability_cluster") or cluster.get("capability") or "",
                    "capability": cluster.get("capability") or cluster.get("capability_cluster") or "",
                    "description": cluster.get("description") or "",
                    "direct_cues": set(cluster.get("direct_cues") or []),
                    "indirect_cues": set(cluster.get("indirect_cues") or []),
                    "promoted_value_streams": list(cluster.get("promote_value_streams") or cluster.get("promoted_value_streams") or []),
                    "related_value_streams": list(cluster.get("related_value_streams") or []),
                    "strength": float(cluster.get("strength") or 1.0),
                    "score_boost": float(cluster.get("score_boost") or 0.25),
                }
            )
        filtered_clusters = [
            cluster
            for cluster in normalized_clusters
            if cluster["capability"] and cluster["promoted_value_streams"]
        ]
        if not filtered_clusters:
            logger.warning("Capability map at %s produced no valid clusters; using fallback.", _CONFIG_PATH)
            return _FALLBACK_CAPABILITY_CLUSTERS
        return filtered_clusters
    except Exception as exc:
        logger.warning("Failed to load capability map from %s (%s); using fallback.", _CONFIG_PATH, exc)
        return _FALLBACK_CAPABILITY_CLUSTERS

def _norm(value: str) -> str:
    return normalize_for_search((value or "").strip())

def _to_allowed_set(allowed_value_stream_names: Optional[List[str]]) -> Optional[Set[str]]:
    if not allowed_value_stream_names:
        return None
    return {_norm(name) for name in allowed_value_stream_names if name}

def _build_cue_text(new_card_summary: Dict[str, Any], cleaned_text: Optional[str]) -> str:
    parts: List[str] = []
    for field in _SUMMARY_FIELDS_FOR_CUES:
        value = new_card_summary.get(field)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item)
        elif isinstance(value, str) and value.strip():
            parts.append(value)
    if cleaned_text:
        parts.append(cleaned_text)
    return "\n".join(parts).lower()

def _compute_capability_hits(cue_text: str, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for cluster in clusters:
        matched_direct_terms = sorted(term for term in cluster.get("direct_cues", set()) if term in cue_text)
        matched_indirect_terms = sorted(term for term in cluster.get("indirect_cues", set()) if term in cue_text)
        if not matched_direct_terms and not matched_indirect_terms:
            continue
        direct_score = len(matched_direct_terms) * 1.0
        indirect_score = len(matched_indirect_terms) * 0.6
        base_strength = min(1.0, (direct_score + indirect_score) / 5.0)
        adjusted_strength = min(1.0, base_strength * float(cluster.get("strength") or 1.0))
        hits.append(
            {
                "capability_cluster": cluster["capability_cluster"],
                "capability": cluster["capability"],
                "matched_terms": sorted(matched_direct_terms + matched_indirect_terms),
                "matched_direct_terms": matched_direct_terms,
                "matched_indirect_terms": matched_indirect_terms,
                "strength": round(adjusted_strength, 3),
            }
        )
    return hits

def _inject_vs_support_candidates(
    candidates: List[Dict[str, Any]],
    vs_support: List[Dict[str, Any]],
) -> None:
    existing = {_norm(candidate.get("entity_name", "")) for candidate in candidates}
    for support in vs_support or []:
        name = (support.get("entity_name") or "").strip()
        key = _norm(name)
        if not name or key in existing:
            continue
        candidates.append(
            {
                "entity_id": "",
                "entity_name": name,
                "description": f"Historical support from {int(support.get('support_count', 0))} analog tickets.",
                "score": float(support.get("best_score") or 0.0),
                "source": "historical_support",
                "support_count": int(support.get("support_count") or 0),
            }
        )
        existing.add(key)

def map_capabilities_to_candidates(
    *,
    new_card_summary: Dict[str, Any],
    cleaned_text: Optional[str],
    vs_support: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    allowed_value_stream_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Apply capability mapping and return enriched/promoted candidate payload.

    Output shape:
      - capability_hits
      - promoted_value_streams
      - enriched_candidates
    """
    clusters = _load_capability_clusters()
    allowed_set = _to_allowed_set(allowed_value_stream_names)
    enriched_candidates = deepcopy(candidates or [])

    # Ensure analog support can contribute to recall before mapping boosts.
    _inject_vs_support_candidates(enriched_candidates, vs_support)
    
    cue_text = _build_cue_text(new_card_summary, cleaned_text)
    capability_hits = _compute_capability_hits(cue_text, clusters)
    
    by_name = {_norm(candidate.get("entity_name", "")): candidate for candidate in enriched_candidates}
    promoted_value_streams: List[Dict[str, Any]] = []

    for hit in capability_hits:
        cluster = next(item for item in clusters if item["capability"] == hit["capability"])
        strength = float(hit["strength"])
        base_boost = float(cluster["score_boost"])
        dynamic_boost = round(base_boost * max(0.6, strength), 3)

        for stream_name in cluster["promoted_value_streams"]:
            key = _norm(stream_name)
            if allowed_set and key not in allowed_set:
                continue
            
            if key in by_name:
                existing = by_name[key]
                current_score = float(existing.get("score") or existing.get("best_score") or 0.0)
                boosted_score = min(0.95, max(current_score + dynamic_boost, current_score))
                existing["score"] = round(boosted_score, 4)
                existing["source"] = existing.get("source") or "capability_mapping"
                existing["promotion_reason"] = "capability_mapping"
            else:
                seeded_score = min(0.9, 0.55 + dynamic_boost)
                injected = {
                    "entity_id": "",
                    "entity_name": stream_name,
                    "description": f"Promoted from capability hit: {hit['capability']}.",
                    "score": round(seeded_score, 4),
                    "source": "capability_mapping",
                    "promotion_reason": "capability_mapping",
                }
                enriched_candidates.append(injected)
                by_name[key] = injected

            promoted_value_streams.append(
                {
                    "entity_name": stream_name,
                    "promotion_reason": "capability_mapping",
                    "score_boost": dynamic_boost,
                    "capability_cluster": cluster.get("capability_cluster", ""),
                    "capability": hit["capability"],
                }
            )

    # Optional post-filter if callers expect strict allow-list output.
    if allowed_set:
        enriched_candidates = [
            candidate
            for candidate in enriched_candidates
            if _norm(candidate.get("entity_name", "")) in allowed_set
        ]

    enriched_candidates.sort(
        key=lambda item: float(item.get("score") or item.get("best_score") or 0.0),
        reverse=True,
    )

    return {
        "config_path": str(_CONFIG_PATH),
        "capability_hits": capability_hits,
        "promoted_value_streams": promoted_value_streams,
        "enriched_candidates": enriched_candidates,
    }
