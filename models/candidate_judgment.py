"""
Pydantic CandidateJudgment model — Pass 1 verifier output (V5 architecture).

Pass 1 returns one judgment per candidate. Pass 2 consumes this list and
produces the final SelectionResult. Keeping these separate makes the
two-pass design genuinely distinct rather than a re-formatting loop.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


BucketLabel = Literal["directly_supported", "pattern_inferred", "no_evidence"]


class CandidateJudgment(BaseModel):
    """
    Evidence judgment for a single candidate value stream.

    Returned as a flat list by Pass 1 (SelectorVerifyChain v2).
    Consumed by Pass 2 (SelectorFinalizeChain v2) to produce SelectionResult.
    """

    entity_name: str
    bucket: BucketLabel
    confidence: float = 0.0
    rationale: str = ""


class JudgmentList(BaseModel):
    """Wrapper for a list of CandidateJudgments, for structured output binding."""

    judgments: List[CandidateJudgment] = Field(default_factory=list)
