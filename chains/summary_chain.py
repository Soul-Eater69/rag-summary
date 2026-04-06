"""
LangChain-style summary generation chain (V5 architecture).

Wraps prompt loading + LLM call + Pydantic validation for ticket/card summarization.
Returns SummaryDoc (Pydantic model) rather than raw dicts.

Primary path: provider-native structured output via structured_generate().
Fallback: text generation + JSON parse + model_validate().

Prompt version hierarchy:
  v3 (default): schema-light, relies on structured output
  v2: schema in prompt body (used if v3 not available)
  v1: legacy full JSON skeleton
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from rag_summary.ingestion.adapters import (
    LLMService,
    get_default_llm,
    structured_generate,
)
from rag_summary.ingestion.function_normalizer import normalize_functions
from rag_summary.models.summary_doc import SummaryDoc
from .prompt_loader import load_prompt, render_prompt

logger = logging.getLogger(__name__)


def _enrich_with_canonical(summary: SummaryDoc) -> SummaryDoc:
    """Normalize raw function lists to canonical vocabulary, in-place."""
    canon_direct = normalize_functions(summary.direct_functions_raw)
    canon_implied = normalize_functions(summary.implied_functions_raw)
    summary.direct_functions_canonical = canon_direct
    summary.implied_functions_canonical = canon_implied
    # Legacy compat fields
    if not summary.direct_functions:
        summary.direct_functions = canon_direct
    if not summary.implied_functions:
        summary.implied_functions = canon_implied
    return summary


class SummaryChain:
    """
    Chain for generating structured semantic summaries.

    Returns SummaryDoc (Pydantic model) for both ticket and card variants.

    Usage:
        chain = SummaryChain(llm=my_llm)
        doc: SummaryDoc = chain.run_ticket(ticket_id=..., title=..., ticket_text=..., vs_labels=[...])
        card: SummaryDoc = chain.run_card(card_text=...)
    """

    def __init__(
        self,
        *,
        llm: Optional[LLMService] = None,
        prompt_version: str = "v3",
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
        vs_labels: List[str],
    ) -> SummaryDoc:
        """Generate a structured summary for a historical ticket."""
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

        doc = self._call_structured(system=system, user=user)

        # Overlay stable identifiers not extracted by LLM
        doc.doc_id = f"summary_{ticket_id}"
        doc.ticket_id = ticket_id
        doc.title = title
        if vs_labels:
            doc.value_stream_labels = vs_labels

        _enrich_with_canonical(doc)
        return doc

    def run_card(self, *, card_text: str) -> SummaryDoc:
        """Generate a structured summary for a new idea card."""
        system = self._card_prompt["system"]
        user = render_prompt(
            self._card_prompt,
            {"card_text": card_text},
            role="user",
        )

        doc = self._call_structured(system=system, user=user)
        doc.doc_id = "new_idea_card_summary"
        doc.ticket_id = ""

        _enrich_with_canonical(doc)
        return doc

    def _call_structured(self, *, system: str, user: str) -> SummaryDoc:
        """Call LLM and return a validated SummaryDoc. Falls back to empty doc on failure."""
        try:
            return structured_generate(
                self.llm,
                user,
                SummaryDoc,
                context="",
                system_prompt=system,
            )
        except Exception as exc:
            logger.error("[SummaryChain] structured_generate failed: %s", exc)
            return SummaryDoc()
