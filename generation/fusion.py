"""
Source-aware fused ranking (V6 architecture).

Computes a weighted fused score for each CandidateEvidence object,
incorporating source-specific weights, a diversity bonus, quality multipliers,
and a theme promotion bonus.

Formula:
    S_fused(v) = sum(w_source * S_adj_source(v))
                 + bonus_source_diversity
                 + theme_promotion_bonus
                 - no_evidence_penalty

where S_adj_source applies quality multipliers to the attachment score.

Profile selection
-----------------
compute_fused_scores() accepts optional profile_hints describing runtime
conditions (analog count, native attachment count, theme candidate count).
It selects the highest-priority profile whose trigger is satisfied and
uses that profile's weights instead of the base DEFAULT_WEIGHTS.

Score diagnostics
-----------------
Each candidate receives a `fusion_breakdown` dict with per-component
contributions, and `fusion_profile` naming the selected weight profile.
These are written to debug artifacts by pipeline.py.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from .candidate_evidence import ALL_SOURCES

logger = logging.getLogger(__name__)

_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "source_weights.yaml"

# -----------------------------------------------------------------------
# Defaults (used when config YAML is unavailable)
# -----------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "chunk": 0.20,
    "summary": 0.15,
    "attachment": 0.18,
    "theme": 0.08,
    "kg": 0.18,
    "historical": 0.12,
    "capability": 0.09,
}
DEFAULT_DIVERSITY_BONUS_PER_SOURCE = 0.03
DEFAULT_MAX_DIVERSITY_BONUS = 0.15
DEFAULT_NO_EVIDENCE_PENALTY = -0.10
DEFAULT_THEME_PROMOTION_MIN_SCORE = 0.50
DEFAULT_THEME_PROMOTION_BONUS = 0.04


# -----------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------

def _load_weights_config() -> Dict[str, Any]:
    """Load source_weights.yaml; return empty dict on failure."""
    if not _CONFIG_PATH.exists():
        return {}
    try:
        import yaml  # type: ignore
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("[fusion] Failed to load source_weights.yaml: %s", exc)
        return {}


def _select_profile(
    config: Dict[str, Any],
    *,
    analog_count: int,
    attachment_native_count: int,
    theme_candidate_count: int,
) -> Tuple[str, Dict[str, float]]:
    """
    Select the highest-priority weight profile whose trigger fires.

    Returns (profile_name, weights_dict).
    Falls back to base weights if no profile trigger matches.
    """
    profiles = config.get("profiles", {})
    base_weights = dict(config.get("weights", DEFAULT_WEIGHTS))

    # Collect triggered profiles sorted by priority (lower = higher priority)
    triggered = []
    for name, profile in profiles.items():
        trigger = profile.get("trigger", {})
        fired = True

        if "max_analog_count" in trigger and analog_count > trigger["max_analog_count"]:
            fired = False
        if "min_attachment_native_count" in trigger and attachment_native_count < trigger["min_attachment_native_count"]:
            fired = False
        if "min_theme_candidate_count" in trigger and theme_candidate_count < trigger["min_theme_candidate_count"]:
            fired = False

        if fired:
            priority = profile.get("priority", 99)
            profile_weights = dict(profile.get("weights", base_weights))
            triggered.append((priority, name, profile_weights))

    if not triggered:
        return "default", base_weights

    triggered.sort(key=lambda x: x[0], reverse=True)  # highest priority number wins
    _, name, weights = triggered[0]
    return name, weights


# -----------------------------------------------------------------------
# Main scoring function
# -----------------------------------------------------------------------

def compute_fused_scores(
    candidates: List[Dict[str, Any]],
    *,
    weights: Optional[Dict[str, float]] = None,
    diversity_bonus_per_source: float = DEFAULT_DIVERSITY_BONUS_PER_SOURCE,
    max_diversity_bonus: float = DEFAULT_MAX_DIVERSITY_BONUS,
    no_evidence_penalty: float = DEFAULT_NO_EVIDENCE_PENALTY,
    profile_hints: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute fused scores for CandidateEvidence objects and sort by score.

    Mutates candidates in-place (adds/updates fused_score, support_confidence,
    fusion_breakdown, fusion_profile) and returns them sorted descending.

    Args:
        candidates:             List of CandidateEvidence dicts.
        weights:                Explicit weight override (skips profile selection).
        diversity_bonus_per_source: Bonus per active source beyond first.
        max_diversity_bonus:    Cap on total diversity bonus.
        no_evidence_penalty:    Penalty for zero active sources.
        profile_hints:          Runtime conditions for profile selection:
                                  analog_count, attachment_native_count,
                                  theme_candidate_count.
    """
    config = _load_weights_config()

    # Derive config-level parameters
    div_cfg = config.get("diversity_bonus", {})
    div_per = div_cfg.get("per_source", diversity_bonus_per_source)
    div_max = div_cfg.get("max_total", max_diversity_bonus)
    pen_cfg = config.get("penalties", {})
    no_ev_pen = pen_cfg.get("no_evidence", no_evidence_penalty)
    theme_cfg = config.get("theme_promotion", {})
    theme_min = theme_cfg.get("min_score", DEFAULT_THEME_PROMOTION_MIN_SCORE)
    theme_bonus_val = theme_cfg.get("bonus", DEFAULT_THEME_PROMOTION_BONUS)
    qm = config.get("quality_multipliers", {})

    # Profile selection
    if weights is not None:
        w = weights
        profile_name = "override"
    else:
        hints = profile_hints or {}
        profile_name, w = _select_profile(
            config,
            analog_count=hints.get("analog_count", 99),
            attachment_native_count=hints.get("attachment_native_count", 0),
            theme_candidate_count=hints.get("theme_candidate_count", 0),
        )

    if profile_name != "default":
        logger.info("[fusion] Selected weight profile: %s", profile_name)

    for cand in candidates:
        scores = cand.get("source_scores", {})

        # Quality multiplier for attachment sub_source
        adj_scores = dict(scores)
        attachment_sub_sources = cand.get("_attachment_sub_sources") or set()
        if attachment_sub_sources and adj_scores.get("attachment", 0.0) > 0.0:
            if "attachment_native" in attachment_sub_sources:
                multiplier = qm.get("attachment_native", 1.00)
            elif "attachment_proxy" in attachment_sub_sources:
                multiplier = qm.get("attachment_proxy", 0.75)
            else:
                multiplier = qm.get("attachment_heuristic", 0.55)
            adj_scores["attachment"] = round(adj_scores["attachment"] * multiplier, 4)

        # Weighted sum
        weighted = round(sum(
            w.get(source, 0.0) * adj_scores.get(source, 0.0)
            for source in ALL_SOURCES
        ), 4)

        # Diversity bonus
        active_count = cand.get("source_diversity_count", 0)
        diversity = round(
            min((active_count - 1) * div_per, div_max) if active_count > 1 else 0.0,
            4,
        )

        # No-evidence penalty
        penalty = no_ev_pen if active_count == 0 else 0.0

        # Theme promotion bonus
        theme_score = scores.get("theme", 0.0)
        theme_bonus = theme_bonus_val if theme_score >= theme_min else 0.0

        fused = max(0.0, min(1.0, weighted + diversity + penalty + theme_bonus))
        cand["fused_score"] = round(fused, 4)

        # Support confidence: blend of fused score and diversity signal
        confidence = fused * (0.7 + 0.3 * min(1.0, active_count / 3.0))
        cand["support_confidence"] = round(confidence, 4)

        # Diagnostics
        cand["fusion_profile"] = profile_name
        cand["fusion_breakdown"] = {
            "weighted": weighted,
            "diversity": diversity,
            "penalty": penalty,
            "theme_bonus": theme_bonus,
            "fused": cand["fused_score"],
            "active_sources": active_count,
            "adj_attachment": round(adj_scores.get("attachment", 0.0), 4),
        }

    candidates.sort(key=lambda c: c.get("fused_score", 0.0), reverse=True)
    return candidates


# -----------------------------------------------------------------------
# Candidate floor guardrail
# -----------------------------------------------------------------------

def apply_candidate_floor(
    candidates: List[Dict[str, Any]],
    *,
    min_candidates: int = 8,
) -> List[Dict[str, Any]]:
    """
    Ensure at least min_candidates survive for the verifier stage.

    This is a recall guardrail: even if fused scores are low, preserve
    enough candidates so the LLM verifier has material to work with.
    """
    if len(candidates) <= min_candidates:
        return candidates

    above_zero = [c for c in candidates if c.get("fused_score", 0.0) > 0.0]
    if len(above_zero) >= min_candidates:
        return above_zero

    remaining = [c for c in candidates if c.get("fused_score", 0.0) <= 0.0]
    needed = min_candidates - len(above_zero)
    return above_zero + remaining[:needed]
