"""
Capability mapping and candidate enrichment for summary-RAG runtime (V5).

This layer bridges business-language cues to value-stream candidates.
It runs after new-card understanding and before final candidate selection.

V5 changes:
- Loads from capabilities dict keyed by cluster name (not flat list)
- Supports canonical_functions matching from function normalizer
- Uses min_signal_strength thresholds
- Emits capability source scores for CandidateEvidence
- Distinguishes promote_value_streams vs related_value_streams with different boosts
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
    "direct_functions_raw",
    "direct_functions_canonical",
    "implied_functions_raw",
    "implied_functions_canonical",
    # Legacy compat
    "direct_functions",
    "implied_functions",
    "change_types",
    "domain_tags",
    "evidence_sentences",
]

# Default score boosts
_DEFAULT_PROMOTE_BOOST = 0.30
_DEFAULT_RELATED_BOOST = 0.15


def _load_capability_map() -> Dict[str, Dict[str, Any]]:
    """Load capability map from YAML config, returning dict keyed by cluster name."""
    if not _CONFIG_PATH.exists():
        logger.info("Capability map config not found at %s; using empty map.", _CONFIG_PATH)
        return {}

    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            payload = yaml.safe_load(config_file) or {}

        caps = payload.get("capabilities")
        if not isinstance(caps, dict) or not caps:
            logger.warning("Capability map at %s has no capabilities; using empty map.", _CONFIG_PATH)
            return {}

        normalized: Dict[str, Dict[str, Any]] = {}
        for cluster_name, cluster in caps.items():
            if not isinstance(cluster, dict):
                continue
            promote = cluster.get("promote_value_streams") or cluster.get("promoted_value_streams") or []
            if not promote:
                continue
            normalized[cluster_name] = {
                "description": cluster.get("description", ""),
                "direct_cues": set(cluster.get("direct_cues") or []),
                "indirect_cues": set(cluster.get("indirect_cues") or []),
                "canonical_functions": set(cluster.get("canonical_functions") or []),
                "promote_value_streams": list(promote),
                "related_value_streams": list(cluster.get("related_value_streams") or []),
                "weight": float(cluster.get("weight") or 1.0),
                "min_signal_strength": float(cluster.get("min_signal_strength") or 0.5),
            }

        if not normalized:
            logger.warning("Capability map at %s produced no valid clusters.", _CONFIG_PATH)
        return normalized

    except Exception as exc:
        logger.warning("Failed to load capability map from %s (%s); using empty map.", _CONFIG_PATH, exc)
        return {}


def _norm(value: str) -> str:
    return normalize_for_search((value or "").strip())


def _to_allowed_set(allowed_value_stream_names: Optional[List[str]]) -> Optional[Set[str]]:
    if not allowed_value_stream_names:
        return None
    return {_norm(name) for name in allowed_value_stream_names if name}


def _build_cue_text(new_card_summary: Dict[str, Any], cleaned_text: Optional[str]) -> str:
    """Build lowercased text from summary fields for cue matching."""
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


def _extract_canonical_functions(new_card_summary: Dict[str, Any]) -> Set[str]:
    """Extract canonical function labels from the summary for function-based matching."""
    funcs: Set[str] = set()
    for field in ("direct_functions_canonical", "implied_functions_canonical",
                  "direct_functions", "implied_functions"):
        for f in new_card_summary.get(field, []) or []:
            if f:
                funcs.add(f.lower().strip())
    return funcs


def _compute_capability_hits(
    cue_text: str,
    canonical_functions: Set[str],
    capability_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Score each capability cluster against the cue text and canonical functions.

    Returns only clusters that exceed their min_signal_strength threshold.
    """
    hits: List[Dict[str, Any]] = []

    for cluster_name, cluster in capability_map.items():
        # Direct cue matches
        matched_direct = sorted(
            term for term in cluster.get("direct_cues", set()) if term in cue_text
        )
        # Indirect cue matches
        matched_indirect = sorted(
            term for term in cluster.get("indirect_cues", set()) if term in cue_text
        )
        # Canonical function matches
        matched_functions = sorted(
            fn for fn in cluster.get("canonical_functions", set())
            if fn.lower() in canonical_functions
        )

        if not matched_direct and not matched_indirect and not matched_functions:
            continue

        # Score: direct=1.0, indirect=0.6, function=0.8
        direct_score = len(matched_direct) * 1.0
        indirect_score = len(matched_indirect) * 0.6
        function_score = len(matched_functions) * 0.8
        raw_strength = (direct_score + indirect_score + function_score) / 5.0
        base_strength = min(1.0, raw_strength)
        adjusted_strength = min(1.0, base_strength * float(cluster.get("weight", 1.0)))

        # Check threshold
        min_threshold = cluster.get("min_signal_strength", 0.5)
        if adjusted_strength < min_threshold:
            continue

        hits.append({
            "capability_cluster": cluster_name,
            "description": cluster.get("description", ""),
            "matched_direct_cues": matched_direct,
            "matched_indirect_cues": matched_indirect,
            "matched_canonical_functions": matched_functions,
            "strength": round(adjusted_strength, 3),
            "promote_value_streams": cluster["promote_value_streams"],
            "related_value_streams": cluster.get("related_value_streams", []),
        })

    return hits


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
      - config_path: str
      - capability_hits: list of cluster hit details
      - promoted_value_streams: list of promoted stream details
      - capability_candidates: list of candidates from capability source (for CandidateEvidence)
      - enriched_candidates: list of all candidates with boosted scores
    """
    capability_map = _load_capability_map()
    allowed_set = _to_allowed_set(allowed_value_stream_names)
    enriched_candidates = deepcopy(candidates or [])

    # Inject historical support candidates that aren't already present
    _inject_vs_support_candidates(enriched_candidates, vs_support)

    # Build cue text and extract canonical functions
    cue_text = _build_cue_text(new_card_summary, cleaned_text)
    canonical_functions = _extract_canonical_functions(new_card_summary)

    # Compute capability hits
    capability_hits = _compute_capability_hits(cue_text, canonical_functions, capability_map)

    by_name = {_norm(c.get("entity_name", "")): c for c in enriched_candidates}
    promoted_value_streams: List[Dict[str, Any]] = []
    capability_candidates: List[Dict[str, Any]] = []

    for hit in capability_hits:
        strength = float(hit["strength"])

        # Process promoted streams (full boost)
        for stream_name in hit["promote_value_streams"]:
            key = _norm(stream_name)
            if allowed_set and key not in allowed_set:
                continue

            promote_boost = round(_DEFAULT_PROMOTE_BOOST * max(0.6, strength), 3)
            # Capability score for CandidateEvidence: based on signal strength
            cap_score = round(min(0.9, 0.4 + strength * 0.5), 3)

            if key in by_name:
                existing = by_name[key]
                current = float(existing.get("score") or existing.get("best_score") or 0.0)
                existing["score"] = round(min(0.95, current + promote_boost), 4)
                existing.setdefault("source", "capability_mapping")
                existing["promotion_reason"] = "capability_mapping"
            else:
                injected = {
                    "entity_id": "",
                    "entity_name": stream_name,
                    "description": f"Promoted from capability: {hit['capability_cluster']}",
                    "score": round(min(0.9, 0.55 + promote_boost), 4),
                    "source": "capability_mapping",
                    "promotion_reason": "capability_mapping",
                }
                enriched_candidates.append(injected)
                by_name[key] = injected

            # Track for CandidateEvidence builder
            capability_candidates.append({
                "entity_name": stream_name,
                "score": cap_score,
                "source": "capability",
                "capability_cluster": hit["capability_cluster"],
            })

            promoted_value_streams.append({
                "entity_name": stream_name,
                "promotion_reason": "capability_mapping",
                "score_boost": promote_boost,
                "capability_cluster": hit["capability_cluster"],
                "strength": strength,
            })

        # Process related streams (weaker boost)
        for stream_name in hit.get("related_value_streams", []):
            key = _norm(stream_name)
            if allowed_set and key not in allowed_set:
                continue

            related_boost = round(_DEFAULT_RELATED_BOOST * max(0.5, strength), 3)
            cap_score = round(min(0.7, 0.25 + strength * 0.3), 3)

            if key in by_name:
                existing = by_name[key]
                current = float(existing.get("score") or existing.get("best_score") or 0.0)
                existing["score"] = round(min(0.90, current + related_boost), 4)
            else:
                injected = {
                    "entity_id": "",
                    "entity_name": stream_name,
                    "description": f"Related via capability: {hit['capability_cluster']}",
                    "score": round(min(0.75, 0.40 + related_boost), 4),
                    "source": "capability_mapping_related",
                    "promotion_reason": "capability_related",
                }
                enriched_candidates.append(injected)
                by_name[key] = injected

            capability_candidates.append({
                "entity_name": stream_name,
                "score": cap_score,
                "source": "capability",
                "capability_cluster": hit["capability_cluster"],
                "relation": "related",
            })

    # Filter to allowed set if specified
    if allowed_set:
        enriched_candidates = [
            c for c in enriched_candidates
            if _norm(c.get("entity_name", "")) in allowed_set
        ]

    enriched_candidates.sort(
        key=lambda item: float(item.get("score") or item.get("best_score") or 0.0),
        reverse=True,
    )

    return {
        "config_path": str(_CONFIG_PATH),
        "capability_hits": capability_hits,
        "promoted_value_streams": promoted_value_streams,
        "capability_candidates": capability_candidates,
        "enriched_candidates": enriched_candidates,
    }


def _inject_vs_support_candidates(
    candidates: List[Dict[str, Any]],
    vs_support: List[Dict[str, Any]],
) -> None:
    """Inject analog VS support entries not already present in candidates."""
    existing = {_norm(c.get("entity_name", "")) for c in candidates}
    for support in vs_support or []:
        name = (support.get("entity_name") or "").strip()
        key = _norm(name)
        if not name or key in existing:
            continue
        candidates.append({
            "entity_id": "",
            "entity_name": name,
            "description": f"Historical support from {int(support.get('support_count', 0))} analog tickets.",
            "score": float(support.get("best_score") or 0.0),
            "source": "historical_support",
            "support_count": int(support.get("support_count") or 0),
        })
        existing.add(key)
