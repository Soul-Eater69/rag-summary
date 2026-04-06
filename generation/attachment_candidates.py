"""
Attachment-native candidate extraction (V6).

Generates VALUE STREAM candidates from explicitly parsed attachment sections.
Unlike attachment_heuristic (which detects structural indicators in card text),
these candidates are derived from actual section *content* — making them the
highest-quality attachment signal.

sub_source = "attachment_native"

Two scoring layers:
  1. Capability-map cue scan (all section types) — standard cue density × ceiling
  2. Content-signal extraction (section-type-aware):
     - Budget sections:      financial terms ($, million, cost, spend, premium) → billing/payment VS boost
     - Scope/requirements:   action-verb phrases (implement, configure, integrate) → functional VS boost
     - Table sections:       column header keywords → multi-VS signals
     These content signals add a secondary score increment on top of cue scan.

Per-section-type score ceilings:
  budget=0.72, scope/requirements=0.68, exhibit/appendix=0.65, table=0.60, heading=0.55, body=0.50
"""

from __future__ import annotations

import logging
import pathlib
import re
from typing import Any, Dict, List, Optional, Set, Tuple

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


# ---------------------------------------------------------------------------
# Content-signal extractors (section-type-aware)
# ---------------------------------------------------------------------------

# Financial signals found in budget sections — suggest billing/payment VS
_FINANCIAL_RE = re.compile(
    r"(\$[\d,]+|\d+[\s]*million|\d+[\s]*m\b|budget|cost estimate|spend|investment|premium|roi)",
    re.IGNORECASE,
)

# Action-verb phrases found in scope/requirements — suggest functional VS
_ACTION_VERB_RE = re.compile(
    r"\b(implement|configure|integrate|enable|deploy|automate|migrate|build|support|"
    r"process|manage|establish|onboard|enroll|adjudicate|authorize|report)\b",
    re.IGNORECASE,
)

# Financial VS cue terms for content-signal boosting
_FINANCIAL_VS_CUES = {
    "billing", "invoice", "payment", "remittance", "premium",
    "cost", "budget", "financial", "revenue",
}


def _extract_budget_signals(content: str) -> Tuple[bool, List[str]]:
    """Return (has_financial_content, matched_terms)."""
    matches = _FINANCIAL_RE.findall(content)
    return bool(matches), [m.strip().lower() for m in matches[:5]]


def _extract_scope_signals(content: str) -> Tuple[int, List[str]]:
    """Return (action_count, matched_verbs) from scope/requirements content."""
    matches = _ACTION_VERB_RE.findall(content)
    deduped = list(dict.fromkeys(m.lower() for m in matches))
    return len(deduped), deduped[:6]


def _content_score_boost(section_type: str, content: str) -> Tuple[float, str]:
    """
    Compute a content-signal-based score increment for high-value section types.

    Returns (boost, reason_suffix). Boost is added on top of cue-scan score,
    capped so the total never exceeds the section ceiling.
    """
    if section_type in ("budget",):
        has_financial, terms = _extract_budget_signals(content)
        if has_financial:
            boost = min(0.08, 0.02 * len(terms))
            return boost, f"financial_signals:{','.join(terms[:3])}"

    if section_type in ("scope", "requirements"):
        count, verbs = _extract_scope_signals(content)
        if count >= 2:
            boost = min(0.07, 0.015 * count)
            return boost, f"action_verbs:{','.join(verbs[:3])}"

    if section_type in ("table",):
        # Tables with multiple domain words → richer signal
        words = set(re.findall(r"\b\w{4,}\b", content.lower()))
        domain_hits = words & {
            "claim", "claims", "billing", "enrollment", "member", "provider",
            "payment", "eligibility", "authorization", "referral", "network",
        }
        if len(domain_hits) >= 2:
            return 0.05, f"table_domain_terms:{','.join(sorted(domain_hits)[:3])}"

    return 0.0, ""


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

                # Content-signal boost (section-type-aware, runs once per section)
            boost, boost_reason = _content_score_boost(section_type, content)

            for vs_name in cluster["promote_value_streams"]:
                    if allowed_names is not None and vs_name not in allowed_names:
                        continue
                    # Keep highest score per (vs_name, attachment_id, section_id) triple
                    dedup_key = f"{vs_name}|{attachment_id}|{section_id}"
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    final_score = round(min(ceiling, score + boost), 3)
                    reason = f"section:{section_type},cues:{','.join((direct_hits + indirect_hits)[:3])}"
                    if boost_reason:
                        reason += f",{boost_reason}"
                    candidates.append({
                        "entity_name": vs_name,
                        "score": final_score,
                        "source": "attachment",
                        "sub_source": "attachment_native",
                        "attachment_id": attachment_id,
                        "section_id": section_id,
                        "section_title": section_title,
                        "section_type": section_type,
                        "match_reason": reason,
                    })

    return candidates
