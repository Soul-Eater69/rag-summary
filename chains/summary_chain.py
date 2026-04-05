"""
LangChain-style summary generation chain (V5 architecture).

Wraps prompt loading + LLM call + JSON parsing for ticket/card summarization.
Uses prompt YAML specs from prompts/summary/.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from summary_rag.ingestion.adapters import LLMService, get_default_llm, safe_json_extract
from summary_rag.ingestion.function_normalizer import normalize_functions
from .prompt_loader import load_prompt, render_prompt

logger = logging.getLogger(__name__)


def _enrich_with_canonical(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize raw function lists to canonical vocabulary."""
    raw_direct = summary.get("direct_functions_raw") or []
    raw_implied = summary.get("implied_functions_raw") or []
    canon_direct = normalize_functions(raw_direct)
    canon_implied = normalize_functions(raw_implied)
    summary["direct_functions_canonical"] = canon_direct
    summary["implied_functions_canonical"] = canon_implied
    # Legacy compat
    summary.setdefault("direct_functions", canon_direct)
    summary.setdefault("implied_functions", canon_implied)
    return summary


class SummaryChain:
    """
    Chain for generating structured semantic summaries.

    Usage:
        chain = SummaryChain(llm=my_llm)
        summary = chain.run_ticket(ticket_id="IDMT-123", title="...", ticket_text="...", vs_labels=[...])
        card_summary = chain.run_card(card_text="...")
    """

    def __init__(
        self,
        *,
        llm: Optional[LLMService] = None,
        prompt_version: str = "v1",
    ) -> None:
        self._llm = llm
        self._prompt_version = prompt_version
        self._historical_prompt = load_prompt("historical_summary", version=prompt_version)
        self._card_prompt = load_prompt("idea_card_summary", version=prompt_version)

    @property
    def llm(self) -> LLMService:
        if self._llm is None:
            self._llm = get_default_llm()
        return self._llm

    def run_ticket(
        self,
        *,
        ticket_id: str,
        title: str,
        ticket_text: str,
        vs_labels: list,
    ) -> Dict[str, Any]:
        """
        Generate a structured summary for a historical ticket.

        Returns a SummaryDoc-shaped dict.
        """
        system = self._historical_prompt["system"]
        user = render_prompt(
            self._historical_prompt,
            {
                "ticket_id": ticket_id,
                "title": title,
                "ticket_text": ticket_text,
                "vs_labels": ", ".join(vs_labels) if vs_labels else "unknown",
            },
            role="user",
        )

        raw_response = self._call_llm(system=system, user=user)
        parsed = safe_json_extract(raw_response) or {}

        summary: Dict[str, Any] = {
            "doc_id": f"summary_{ticket_id}",
            "ticket_id": ticket_id,
            "title": title,
            "short_summary": parsed.get("short_summary", ""),
            "business_goal": parsed.get("business_goal", ""),
            "actors": parsed.get("actors") or [],
            "change_types": parsed.get("change_types") or [],
            "domain_tags": parsed.get("domain_tags") or [],
            "evidence_sentences": parsed.get("evidence_sentences") or [],
            "direct_functions_raw": parsed.get("direct_functions_raw") or [],
            "implied_functions_raw": parsed.get("implied_functions_raw") or [],
            "capability_tags": parsed.get("capability_tags") or [],
            "operational_footprint": parsed.get("operational_footprint") or [],
            "value_stream_labels": vs_labels or parsed.get("value_stream_labels") or [],
            "stream_support_type": parsed.get("stream_support_type") or {},
            "supporting_evidence": [],
            "co_occurrence_bundle": [],
            "direct_functions": [],
            "implied_functions": [],
            "direct_functions_canonical": [],
            "implied_functions_canonical": [],
            "retrieval_text": "",
        }
        _enrich_with_canonical(summary)
        return summary

    def run_card(self, *, card_text: str) -> Dict[str, Any]:
        """
        Generate a structured summary for a new idea card.

        Returns a SummaryDoc-shaped dict (no value_stream_labels).
        """
        system = self._card_prompt["system"]
        user = render_prompt(
            self._card_prompt,
            {"card_text": card_text},
            role="user",
        )

        raw_response = self._call_llm(system=system, user=user)
        parsed = safe_json_extract(raw_response) or {}

        summary: Dict[str, Any] = {
            "doc_id": "new_idea_card_summary",
            "ticket_id": "",
            "title": "",
            "short_summary": parsed.get("short_summary", ""),
            "business_goal": parsed.get("business_goal", ""),
            "actors": parsed.get("actors") or [],
            "change_types": parsed.get("change_types") or [],
            "domain_tags": parsed.get("domain_tags") or [],
            "evidence_sentences": parsed.get("evidence_sentences") or [],
            "direct_functions_raw": parsed.get("direct_functions_raw") or [],
            "implied_functions_raw": parsed.get("implied_functions_raw") or [],
            "capability_tags": parsed.get("capability_tags") or [],
            "operational_footprint": parsed.get("operational_footprint") or [],
            "value_stream_labels": [],
            "stream_support_type": {},
            "supporting_evidence": [],
            "co_occurrence_bundle": [],
            "direct_functions": [],
            "implied_functions": [],
            "direct_functions_canonical": [],
            "implied_functions_canonical": [],
            "retrieval_text": "",
        }
        _enrich_with_canonical(summary)
        return summary

    def _call_llm(self, *, system: str, user: str) -> str:
        try:
            reply = self.llm.generate(
                query=user,
                context="",
                system_prompt=system,
            )
            return reply.content or ""
        except Exception as exc:
            logger.error("[SummaryChain] LLM call failed: %s", exc)
            return ""
