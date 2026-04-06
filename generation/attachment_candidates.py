"""
Attachment-native candidate extraction (V6).

Generates VALUE STREAM candidates from explicitly parsed attachment sections.
Unlike attachment_heuristic (which detects structural indicators in card text),
these candidates are derived from actual section *content* — making them the
highest-quality attachment signal.

sub_source = "attachment_native"

Scoring:
  Budget / requirements / scope sections get higher baseline scores (0.55–0.72)
  than heading / body sections (0.40–0.55), because they tend to be more
  semantically focused.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "capability_map.yaml"

# Per-section-type score ceilings
_SECTION_TYPE_CEILINGS: Dict[str, float] = {
    "budget": 0.72,
    "scope": 0.70,
    "requirements": 0.68,
    "exhibit": 0.65,
    "appendix": 0.62,
    "table": 0.60,
    "roadmap": 0.60,
    "heading": 0.55,
    "body": 0.50,
}
_DEFAULT_CEILING = 0.50
_BASE_SCORE = 0.35


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
            promote = (
                cluster.get("promote_value_streams")
                or cluster.get("promoted_value_streams")
                or []
            )
            if not promote:
                continue
            result[name] = {
                "direct_cues": set(cluster.get("direct_cues") or []),
                "indirect_cues": set(cluster.get("indirect_cues") or []),
                "canonical_functions": set(cluster.get("canonical_functions") or []),
                "promote_value_streams": list(promote),
                "weight": float(cluster.get("weight") or 1.0),
            }
        return result
    except Exception as exc:
        logger.warning("attachment_candidates: failed to load capability map: %s", exc)
        return {}


def extract_attachment_native_candidates(
    attachment_docs: List[Dict[str, Any]],
    *,
    allowed_names: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate attachment-native candidates from parsed attachment section dicts.

    Each section is scanned for capability map cues. Sections with strong cue
    matches produce candidates with sub_source="attachment_native" and scores
    proportional to section type quality × cue density.

    attachment_docs: list of ParsedAttachment.to_dict() outputs, each containing
      a "sections" list of {"section_id", "section_title", "section_type", "content"}.

    Returns candidates shaped for build_candidate_evidence's attachment source.
    """
    if not attachment_docs:
        return []

    capability_map = _load_capability_map()
    if not capability_map:
        return []

    candidates: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for doc in attachment_docs:
        attachment_id = doc.get("attachment_id", "")
        for section in doc.get("sections", []):
            section_id = section.get("section_id", "")
            section_title = section.get("section_title", "")
            section_type = section.get("section_type", "body")
            content = section.get("content", "").lower()

            if not content.strip():
                continue

            ceiling = _SECTION_TYPE_CEILINGS.get(section_type, _DEFAULT_CEILING)

            for cluster_name, cluster in capability_map.items():
                direct_hits = [c for c in cluster["direct_cues"] if c in content]
                indirect_hits = [c for c in cluster["indirect_cues"] if c in content]

                if not direct_hits and not indirect_hits:
                    continue

                cue_strength = (len(direct_hits) + 0.5 * len(indirect_hits)) / 4.0
                score = _BASE_SCORE + (ceiling - _BASE_SCORE) * min(1.0, cue_strength)
                score *= cluster.get("weight", 1.0)
                score = round(min(ceiling, score), 3)

                for vs_name in cluster["promote_value_streams"]:
                    if allowed_names is not None and vs_name not in allowed_names:
                        continue
                    # Keep highest score per (vs_name, attachment_id, section_id) triple
                    dedup_key = f"{vs_name}|{attachment_id}|{section_id}"
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    candidates.append({
                        "entity_name": vs_name,
                        "score": score,
                        "source": "attachment",
                        "sub_source": "attachment_native",
                        "attachment_id": attachment_id,
                        "section_id": section_id,
                        "section_title": section_title,
                        "section_type": section_type,
                        "match_reason": (
                            f"section:{section_type},"
                            f"cues:{','.join((direct_hits + indirect_hits)[:3])}"
                        ),
                    })

    return candidates
