"""
KeywordThemeService — always-active, zero-setup theme service (V6).

Uses the capability map's direct_cues and cue_phrases to score query text
without requiring any offline index build or embedding service. This replaces
the _NoopThemeService as the default, so the theme evidence slot is never
left at zero in a fresh repo.

Scoring per VS:
    cue_score = weighted match count / normalizer
    candidate_score = cue_score * cluster_weight * BASE_SCALE

Scores are deliberately capped below 0.50 (max ~0.45) so keyword-matched
themes sit below FAISS-matched themes in the ranking and never inflate
confidence above pattern_inferred level. When a real FaissThemeRetrievalService
is available (via auto-discovery or explicit injection), it takes precedence.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "capability_map.yaml"

# Keyword theme scores are capped well below 0.50 to stay in pattern territory
_KEYWORD_THEME_MAX = 0.45
_KEYWORD_THEME_BASE = 0.10

# Normalizer: how many cue matches = "full signal"
_CUE_NORM = 5.0


def _load_capability_map() -> Dict[str, Dict[str, Any]]:
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
                "direct_cues": list(cluster.get("direct_cues") or []),
                "indirect_cues": list(cluster.get("indirect_cues") or []),
                "cue_phrases": list(cluster.get("cue_phrases") or []),
                "promote_value_streams": list(promote),
                "weight": float(cluster.get("weight") or 1.0),
            }
        return result
    except Exception as exc:
        logger.warning("[KeywordThemeService] Failed to load capability map: %s", exc)
        return {}


class KeywordThemeService:
    """
    Always-active theme service backed by capability map keyword matching.

    Activated automatically by node_retrieve_themes when no FAISS index exists.
    Produces theme candidates with source="theme" at modest scores (≤ 0.45)
    so they contribute to pattern_inferred classification without over-inflating
    confidence.

    When a FaissThemeRetrievalService is available (index built), that service
    produces higher-quality scores that supersede these keyword signals.
    """

    def __init__(self) -> None:
        self._capability_map: Optional[Dict[str, Dict[str, Any]]] = None

    def _get_map(self) -> Dict[str, Dict[str, Any]]:
        if self._capability_map is None:
            self._capability_map = _load_capability_map()
        return self._capability_map

    def retrieve_theme_candidates(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        allowed_names: Optional[List[str]] = None,
        cutoff_date: Optional[str] = None,  # accepted but unused (no temporal data here)
    ) -> List[Dict[str, Any]]:
        """
        Score query_text against capability map cues.

        Returns per-VS candidates sorted by score descending, capped at top_k.
        """
        if not query_text or not query_text.strip():
            return []

        capability_map = self._get_map()
        if not capability_map:
            return []

        lower_query = query_text.lower()
        candidates: Dict[str, Dict[str, Any]] = {}  # vs_name → best candidate

        for cluster_name, cluster in capability_map.items():
            all_cues = (
                cluster.get("direct_cues", [])
                + cluster.get("indirect_cues", [])
                + cluster.get("cue_phrases", [])
            )

            # Direct cues weighted 1.0, indirect/cue_phrases 0.5
            direct_set = set(cluster.get("direct_cues", []))
            direct_hits = [c for c in all_cues if c in lower_query and c in direct_set]
            other_hits = [c for c in all_cues if c in lower_query and c not in direct_set]

            total_signal = len(direct_hits) * 1.0 + len(other_hits) * 0.5
            if total_signal == 0:
                continue

            cue_score = min(1.0, total_signal / _CUE_NORM)
            score = round(
                _KEYWORD_THEME_BASE
                + (_KEYWORD_THEME_MAX - _KEYWORD_THEME_BASE) * cue_score * cluster["weight"],
                4,
            )

            matched_cues = (direct_hits + other_hits)[:5]
            evidence_snippets = [
                f"keyword-theme:{cluster_name}, cue:{c}" for c in matched_cues[:3]
            ]

            for vs_name in cluster["promote_value_streams"]:
                if allowed_names is not None and vs_name not in allowed_names:
                    continue
                existing = candidates.get(vs_name)
                if existing is None or score > existing.get("score", 0.0):
                    candidates[vs_name] = {
                        "entity_name": vs_name,
                        "candidate_name": vs_name,
                        "source": "theme",
                        "sub_source": "keyword_theme",
                        "score": score,
                        "theme_id": f"kw:{cluster_name}",
                        "theme_label": cluster_name,
                        "vs_support_fraction": cue_score,
                        "cue_phrases": matched_cues,
                        "capability_tags": [],
                        "member_count": 0,
                        "evidence_snippets": evidence_snippets,
                        "snippets": evidence_snippets,
                        "theme_match_count": 1,
                    }

        result = sorted(candidates.values(), key=lambda c: -c["score"])[:top_k]
        if result:
            logger.debug(
                "[KeywordThemeService] %d VS candidates (keyword-matched, max_score=%.3f)",
                len(result), result[0]["score"] if result else 0,
            )
        return result
