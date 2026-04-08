"""
Pydantic CandidateJudgment / VerificationResult models - Pass 1 verifier output.

Pass 1 returns one judgment per candidate as a VerificationResult.
Pass 2 consumes this and produces the final SelectionResult.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

BucketLabel = Literal["directly_supported", "pattern_inferred", "no_evidence"]

class CandidateJudgment(BaseModel):
    """
    Evidence judgment for a single candidate value stream.

    Returned as a flat list by Pass 1 (SelectorVerifyChain).
    Consumed by Pass 2 (SelectorFinalizeChain) to produce SelectionResult.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    entity_name: str = Field(validation_alias=AliasChoices("entity_name", "candidate", "name"))
    bucket: BucketLabel = Field(validation_alias=AliasChoices("bucket", "label", "judgment"))
    confidence: float = Field(default=0.0, validation_alias=AliasChoices("confidence", "score"))
    rationale: str = Field(default="", validation_alias=AliasChoices("rationale", "reason", "evidence"))

class VerificationResult(BaseModel):
    """
    Typed output of Pass 1 (SelectorVerifyChain).

    Contains one CandidateJudgment per candidate - a flat list, not
    pre-grouped into buckets. Pass 2 is responsible for grouping.
    """

    judgments: List[CandidateJudgment] = Field(default_factory=list)

# Backward-compat alias
JudgmentList = VerificationResult
