"""
Source-aware fused ranking (V5 architecture).

Computes a weighted fused score for each CandidateEvidence object,
incorporating source-specific weights and a diversity bonus.

Formula:
    S_fused(v) = sum(w_source * S_source(v)) + bonus_source_diversity

This module is intentionally separate from candidate_evidence.py so that
weights can be tuned independently.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .candidate_evidence import ALL_SOURCES

logger = logging.getLogger(__name__)

# Default source weights -- tunable
DEFAULT_WEIGHTS: Dict[str, float] = {
    "chunk": 0.20,
    "summary": 0.15,
    "attachment": 0.18,
    "theme": 0.08,
    "kg": 0.18,
    "historical": 0.12,
    "capability": 0.09,
}

# Bonus per additional source beyond the first
DEFAULT_DIVERSITY_BONUS_PER_SOURCE = 0.03

# Maximum total diversity bonus
DEFAULT_MAX_DIVERSITY_BONUS = 0.15

# Penalty for candidates with zero evidence sources
DEFAULT_NO_EVIDENCE_PENALTY = -0.10


def compute_fused_scores(
    candidates: List[Dict[str, Any]],
    *,
    weights: Dict[str, float] | None = None,
    diversity_bonus_per_source: float = DEFAULT_DIVERSITY_BONUS_PER_SOURCE,
    max_diversity_bonus: float = DEFAULT_MAX_DIVERSITY_BONUS,
    no_evidence_penalty: float = DEFAULT_NO_EVIDENCE_PENALTY,
) -> List[Dict[str, Any]]:
    """
    Compute fused scores for CandidateEvidence objects and sort by score.

    Mutates candidates in-place (adds/updates fused_score, support_confidence)
    and returns them sorted descending by fused_score.
    """
    w = weights or DEFAULT_WEIGHTS

    for cand in candidates:
        scores = cand.get("source_scores", {})

        # Weighted sum
        weighted = sum(
            w.get(source, 0.0) * scores.get(source, 0.0)
            for source in ALL_SOURCES
        )

        # Diversity bonus
        active_count = cand.get("source_diversity_count", 0)
        if active_count > 1:
            diversity = min(
                (active_count - 1) * diversity_bonus_per_source,
                max_diversity_bonus,
            )
        else:
            diversity = 0.0

        # Penalty for no evidence
        penalty = no_evidence_penalty if active_count == 0 else 0.0

        fused = max(0.0, min(1.0, weighted + diversity + penalty))
        cand["fused_score"] = round(fused, 4)

        # Support confidence: blend of fused score and diversity signal
        confidence = fused * (0.7 + 0.3 * min(1.0, active_count / 3.0))
        cand["support_confidence"] = round(confidence, 4)

    candidates.sort(key=lambda c: c.get("fused_score", 0.0), reverse=True)
    return candidates


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

    # All candidates with fused_score > 0 pass; pad from remainder if needed
    above_zero = [c for c in candidates if c.get("fused_score", 0.0) > 0.0]
    if len(above_zero) >= min_candidates:
        return above_zero

    # Pad with top remaining candidates by original ordering
    remaining = [c for c in candidates if c.get("fused_score", 0.0) <= 0.0]
    needed = min_candidates - len(above_zero)
    return above_zero + remaining[:needed]
