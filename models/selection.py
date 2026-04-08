"""
Pydantic selection result models (V5 architecture).

SelectionResult is the final pipeline output, bucketing candidates into
three evidence-quality tiers.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class SupportedStream(BaseModel):
    """A value stream with sufficient evidence to support it."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    entity_name: str = Field(validation_alias=AliasChoices("entity_name", "name"))
    entity_id: str = Field(default="", validation_alias=AliasChoices("entity_id", "id"))
    confidence: float
    evidence: str = Field(default="", validation_alias=AliasChoices("evidence", "reason", "rationale"))
    # V6: pattern basis for pattern_inferred streams
    pattern_basis: Optional[str] = None       # "analog_similarity" | "bundle_pattern" |
                                              # "downstream_chain" | "capability_overlap" | "theme"
    supporting_analog_ids: Optional[List[str]] = None  # ticket IDs that drove this inference


class UnsupportedStream(BaseModel):
    """A candidate value stream lacking sufficient evidence."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    entity_name: str = Field(validation_alias=AliasChoices("entity_name", "name"))
    reason: str = Field(default="", validation_alias=AliasChoices("reason", "evidence", "rationale"))


class SelectionResult(BaseModel):
    """
    Three-class output from the LLM verifier.

    - directly_supported: strong direct evidence (card text, summary, attachments, KG)
    - pattern_inferred: supported mainly by historical analogs or capability mapping
    - no_evidence: insufficient evidence for prediction
    """

    directly_supported: List[SupportedStream] = Field(default_factory=list)
    pattern_inferred: List[SupportedStream] = Field(default_factory=list)
    no_evidence: List[UnsupportedStream] = Field(default_factory=list)

    def selected_value_streams(self) -> List[dict]:
        """Compat helper: union of directly_supported + pattern_inferred."""
        result = []
        for vs in self.directly_supported:
            result.append({
                "entity_id": vs.entity_id,
                "entity_name": vs.entity_name,
                "confidence": vs.confidence,
                "reason": vs.evidence,
                "support_type": "direct",
            })
        for vs in self.pattern_inferred:
            result.append({
                "entity_id": vs.entity_id,
                "entity_name": vs.entity_name,
                "confidence": vs.confidence,
                "reason": vs.evidence,
                "support_type": "pattern",
            })
        return result
