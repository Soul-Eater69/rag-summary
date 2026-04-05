"""
LangChain-style selector verification chain — Pass 1 (V5 architecture).

Calls the LLM to classify each candidate value stream into
directly_supported / pattern_inferred / no_evidence based on the
full evidence context.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from summary_rag.ingestion.adapters import LLMService, get_default_llm, safe_json_extract
from .prompt_loader import load_prompt, render_prompt

logger = logging.getLogger(__name__)


def _format_new_card_summary(summary: Dict[str, Any]) -> str:
    parts = []
    if summary.get("short_summary"):
        parts.append(f"Summary: {summary['short_summary']}")
    if summary.get("business_goal"):
        parts.append(f"Business goal: {summary['business_goal']}")
    if summary.get("actors"):
        parts.append(f"Actors: {', '.join(summary['actors'])}")
    raw_direct = summary.get("direct_functions_raw") or summary.get("direct_functions") or []
    canon_direct = summary.get("direct_functions_canonical") or []
    if raw_direct:
        parts.append(f"Direct functions (raw): {', '.join(raw_direct)}")
    if canon_direct:
        parts.append(f"Direct functions (canonical): {', '.join(canon_direct)}")
    raw_implied = summary.get("implied_functions_raw") or summary.get("implied_functions") or []
    canon_implied = summary.get("implied_functions_canonical") or []
    if raw_implied:
        parts.append(f"Implied functions (raw): {', '.join(raw_implied)}")
    if canon_implied:
        parts.append(f"Implied functions (canonical): {', '.join(canon_implied)}")
    if summary.get("capability_tags"):
        parts.append(f"Capability tags: {', '.join(summary['capability_tags'])}")
    if summary.get("change_types"):
        parts.append(f"Change types: {', '.join(summary['change_types'])}")
    if summary.get("domain_tags"):
        parts.append(f"Domain: {', '.join(summary['domain_tags'])}")
    return "\n".join(parts)


def _format_analog_summaries(analogs: List[Dict[str, Any]], limit: int = 5) -> str:
    if not analogs:
        return "No historical analogs found."
    parts = []
    for i, analog in enumerate(analogs[:limit], 1):
        lines = [f"### Analog {i}: {analog.get('ticket_id', '?')} (score: {analog.get('score', 0):.4f})"]
        if analog.get("short_summary"):
            lines.append(f"  Summary: {analog['short_summary']}")
        funcs = analog.get("direct_functions_canonical") or analog.get("direct_functions") or []
        if funcs:
            lines.append(f"  Functions: {', '.join(funcs)}")
        if analog.get("value_stream_labels"):
            lines.append(f"  Mapped VS: {', '.join(analog['value_stream_labels'])}")
        if analog.get("capability_tags"):
            lines.append(f"  Capabilities: {', '.join(analog['capability_tags'])}")
        footprint = analog.get("operational_footprint") or []
        if footprint:
            lines.append(f"  Footprint: {', '.join(footprint[:6])}")
        sst = analog.get("stream_support_type") or {}
        if sst:
            sst_summary = ", ".join(f"{k}:{v}" for k, v in list(sst.items())[:4])
            lines.append(f"  Support types: {sst_summary}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _format_candidate_evidence(candidates: List[Dict[str, Any]], limit: int = 20) -> str:
    if not candidates:
        return "No candidates available."
    parts = []
    for cand in candidates[:limit]:
        name = cand.get("candidate_name") or cand.get("entity_name") or ""
        fused = cand.get("fused_score", 0.0)
        support_type = cand.get("support_type", "unknown")
        diversity = cand.get("source_diversity_count", 0)
        source_scores = cand.get("source_scores", {})
        active_scores = {k: round(v, 2) for k, v in source_scores.items() if v > 0}
        line = f"- {name} [fused: {fused:.3f}, type: {support_type}, sources: {diversity}]"
        if active_scores:
            line += f"\n  Scores: {active_scores}"
        desc = cand.get("description", "")
        if desc:
            line += f"\n  {desc[:150]}"
        parts.append(line)
    return "\n".join(parts)


def _format_raw_evidence(evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return ""
    parts = ["## RAW EVIDENCE SNIPPETS (for verification)\n"]
    for ev in evidence[:8]:
        parts.append(f"- [{ev.get('ticket_id', '?')}] {ev.get('snippet', '')}")
    return "\n".join(parts)


class SelectorVerifyChain:
    """
    Pass 1 of the two-pass LLM verifier.

    Classifies each candidate value stream as directly_supported,
    pattern_inferred, or no_evidence.
    """

    def __init__(
        self,
        *,
        llm: Optional[LLMService] = None,
        prompt_version: str = "v1",
        max_retries: int = 2,
    ) -> None:
        self._llm = llm
        self._max_retries = max_retries
        self._prompt = load_prompt("verify_candidates", version=prompt_version)

    @property
    def llm(self) -> LLMService:
        if self._llm is None:
            self._llm = get_default_llm()
        return self._llm

    def run(
        self,
        *,
        new_card_summary: Dict[str, Any],
        analog_tickets: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        raw_evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run pass 1 verification.

        Returns dict with: directly_supported, pattern_inferred, no_evidence,
        raw_response, prompt_system, prompt_user.
        """
        system = self._prompt["system"]
        user = render_prompt(
            self._prompt,
            {
                "new_card_summary": _format_new_card_summary(new_card_summary),
                "analog_summaries": _format_analog_summaries(analog_tickets),
                "candidate_evidence": _format_candidate_evidence(candidates),
                "raw_evidence_section": _format_raw_evidence(raw_evidence or []),
            },
            role="user",
        )

        parsed: Optional[Dict] = None
        raw_response: Optional[str] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                reply = self.llm.generate(query=user, context="", system_prompt=system)
                raw_response = reply.content
                parsed = safe_json_extract(raw_response)
                if isinstance(parsed, dict) and any(
                    k in parsed for k in ("directly_supported", "pattern_inferred", "no_evidence")
                ):
                    break
                parsed = None
            except Exception as exc:
                logger.error("[VerifyChain] Attempt %d failed: %s", attempt, exc)
                if attempt < self._max_retries:
                    time.sleep(3 * attempt)

        if not parsed:
            parsed = {"directly_supported": [], "pattern_inferred": [], "no_evidence": []}

        return {
            "directly_supported": parsed.get("directly_supported", []),
            "pattern_inferred": parsed.get("pattern_inferred", []),
            "no_evidence": parsed.get("no_evidence", []),
            "raw_response": raw_response,
            "prompt_system": system,
            "prompt_user": user,
        }
