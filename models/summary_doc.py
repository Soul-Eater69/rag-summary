"""Pydantic models for new-card summaries and historical ticket summaries."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


class CardSummaryDoc(BaseModel):
    """Structured summary for a new idea card before any value-stream prediction."""

    # Stable identifiers
    doc_id: str = ""
    ticket_id: str = ""
    title: str = ""

    # Core semantic fields
    short_summary: str = ""
    business_goal: str = ""
    actors: List[str] = Field(default_factory=list)
    change_types: List[str] = Field(default_factory=list)
    domain_tags: List[str] = Field(default_factory=list)
    evidence_sentences: List[str] = Field(default_factory=list)

    # V5: raw LLM-extracted functions (free-form)
    direct_functions_raw: List[str] = Field(default_factory=list)
    implied_functions_raw: List[str] = Field(default_factory=list)

    # V5: normalized to canonical vocabulary
    direct_functions_canonical: List[str] = Field(default_factory=list)
    implied_functions_canonical: List[str] = Field(default_factory=list)

    # Legacy compat aliases (populated from canonical)
    direct_functions: List[str] = Field(default_factory=list)
    implied_functions: List[str] = Field(default_factory=list)

    # V5: capability and operational metadata
    capability_tags: List[str] = Field(default_factory=list)
    operational_footprint: List[str] = Field(default_factory=list)

    # V5: short evidence snippets justifying major functions / themes
    supporting_evidence: List[str] = Field(default_factory=list)

    # V5: compact co-occurrence bundle hinted inside the card text itself
    co_occurrence_bundle: List[str] = Field(default_factory=list)

    # Packed retrieval text
    retrieval_text: str = ""

    @field_validator(
        "actors",
        "change_types",
        "domain_tags",
        "evidence_sentences",
        "direct_functions_raw",
        "implied_functions_raw",
        "direct_functions_canonical",
        "implied_functions_canonical",
        "direct_functions",
        "implied_functions",
        "capability_tags",
        "operational_footprint",
        "supporting_evidence",
        "co_occurrence_bundle",
        mode="before",
    )
    @classmethod
    def _coerce_list_fields(cls, value: Any) -> List[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        if isinstance(value, dict):
            coerced = cls._coerce_list_item(value)
            return [coerced] if coerced else []
        if isinstance(value, list):
            items = [cls._coerce_list_item(item) for item in value]
            return [item for item in items if item]
        return []

    @staticmethod
    def _coerce_list_item(item: Any) -> str:
        if item is None:
            return ""
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            for key in ("description", "name", "label", "value", "id"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return str(item).strip() if not isinstance(item, (list, tuple, set)) else ""

    class Config:
        extra = "allow"


class SummaryDoc(CardSummaryDoc):
    """Historical/indexed summary document with known value-stream metadata."""

    value_stream_labels: List[str] = Field(default_factory=list)
    value_stream_ids: List[str] = Field(default_factory=list)
    stream_support_type: Dict[str, str] = Field(default_factory=dict)

    @field_validator("value_stream_labels", "value_stream_ids", mode="before")
    @classmethod
    def _coerce_vs_list_fields(cls, value: Any) -> List[str]:
        return cls._coerce_list_fields(value)

    @field_validator("stream_support_type", mode="before")
    @classmethod
    def _coerce_stream_support_type(cls, value: Any) -> Dict[str, str]:
        if isinstance(value, dict):
            return {
                str(k): str(v)
                for k, v in value.items()
                if str(k).strip() and str(v).strip()
            }
        return {}
