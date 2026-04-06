"""
Pydantic CandidateJudgment / VerificationResult models — Pass 1 verifier output.

Pass 1 returns one judgment per candidate as a VerificationResult.
Pass 2 consumes this and produces the final SelectionResult.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


BucketLabel = Literal["directly_supported", "pattern_inferred", "no_evidence"]


class CandidateJudgment(BaseModel):
    """
    Evidence judgment for a single candidate value stream.

    Returned as a flat list by Pass 1 (SelectorVerifyChain).
    Consumed by Pass 2 (SelectorFinalizeChain) to produce SelectionResult.
    """

    entity_name: str
    bucket: BucketLabel
    confidence: float = 0.0
    rationale: str = ""


class VerificationResult(BaseModel):
    """
    Typed output of Pass 1 (SelectorVerifyChain).

    Contains one CandidateJudgment per candidate — a flat list, not
    pre-grouped into buckets. Pass 2 is responsible for grouping.
    """

    judgments: List[CandidateJudgment] = Field(default_factory=list)


# Backward-compat alias
JudgmentList = VerificationResult
