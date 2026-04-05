"""
LangChain-style selector finalize chain — Pass 2 (V5 architecture).

V2 behaviour: consumes List[CandidateJudgment] from Pass 1 and applies:
  - deduplication of semantically overlapping streams
  - calibration (confidence bands)
  - contradiction cleanup
  - final shaping into SelectionResult

Falls back to converting judgments directly to SelectionResult if LLM fails.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from summary_rag.ingestion.adapters import LLMService, get_default_llm, safe_json_extract
from summary_rag.models.candidate_judgment import CandidateJudgment
from summary_rag.models.selection import SelectionResult, SupportedStream, UnsupportedStream
from .prompt_loader import load_prompt, render_prompt
from .selector_verify_chain import _format_new_card_summary

logger = logging.getLogger(__name__)


def _format_judgments(judgments: List[CandidateJudgment]) -> str:
    """Format pass-1 CandidateJudgment list for the finalize prompt."""
    if not judgments:
        return "No judgments available."

    lines = []
    for j in judgments:
        conf_str = f" (conf: {j.confidence:.2f})" if j.confidence > 0 else ""
        rat_str = f" — {j.rationale}" if j.rationale else ""
        lines.append(f"- [{j.bucket}]{conf_str} {j.entity_name}{rat_str}")
    return "\n".join(lines)


def _format_fused_scores(fused_candidates: Optional[List[Dict[str, Any]]]) -> str:
    """Format fused scores section for the finalize prompt."""
    if not fused_candidates:
        return "Fused scores not available."
    lines = []
    for c in sorted(fused_candidates, key=lambda x: -x.get("fused_score", 0))[:15]:
        name = c.get("candidate_name") or c.get("entity_name") or ""
        fused = c.get("fused_score", 0.0)
        diversity = c.get("source_diversity_count", 0)
        lines.append(f"- {name}: fused={fused:.3f}, sources={diversity}")
    return "\n".join(lines)


def _judgments_to_selection_result(judgments: List[CandidateJudgment]) -> SelectionResult:
    """
    Convert judgment list directly to SelectionResult without LLM.
    Used as fallback when Pass 2 LLM call fails.
    """
    directly: List[SupportedStream] = []
    pattern: List[SupportedStream] = []
    no_ev: List[UnsupportedStream] = []

    for j in judgments:
        if j.bucket == "directly_supported":
            directly.append(SupportedStream(
                entity_name=j.entity_name,
                confidence=j.confidence if j.confidence > 0 else 0.70,
                evidence=j.rationale,
            ))
        elif j.bucket == "pattern_inferred":
            pattern.append(SupportedStream(
                entity_name=j.entity_name,
                confidence=j.confidence if j.confidence > 0 else 0.50,
                evidence=j.rationale,
            ))
        else:
            no_ev.append(UnsupportedStream(
                entity_name=j.entity_name,
                reason=j.rationale or "No evidence.",
            ))

    return SelectionResult(
        directly_supported=directly,
        pattern_inferred=pattern,
        no_evidence=no_ev,
    )


class SelectorFinalizeChain:
    """
    Pass 2 of the two-pass LLM verifier.

    Consumes List[CandidateJudgment] from SelectorVerifyChain v2, applies
    deduplication / calibration / contradiction cleanup, and produces
    SelectionResult (Pydantic model).

    Falls back to direct judgment conversion if LLM call fails.
    """

    def __init__(
        self,
        *,
        llm: Optional[LLMService] = None,
        prompt_version: str = "v2",
        max_retries: int = 2,
    ) -> None:
        self._llm = llm
        self._max_retries = max_retries
        self._prompt = load_prompt("finalize_selection", version=prompt_version)

    @property
    def llm(self) -> LLMService:
        if self._llm is None:
            self._llm = get_default_llm()
        return self._llm

    def run(
        self,
        *,
        new_card_summary: Dict[str, Any],
        judgments: List[CandidateJudgment],
        fused_candidates: Optional[List[Dict[str, Any]]] = None,
        # v1 compat: accept preliminary_classification dict
        preliminary_classification: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run Pass 2 finalization.

        Returns:
          - selection_result: SelectionResult (Pydantic model)
          - directly_supported, pattern_inferred, no_evidence: dicts (compat)
          - raw_response, prompt_system, prompt_user
        """
        # Accept v1 legacy dict input
        if not judgments and preliminary_classification:
            judgments = self._judgments_from_legacy(preliminary_classification)

        system = self._prompt["system"]
        user = render_prompt(
            self._prompt,
            {
                "new_card_summary": _format_new_card_summary(new_card_summary),
                "candidate_judgments": _format_judgments(judgments),
                "fused_scores_section": _format_fused_scores(fused_candidates),
                # v1 compat variable name
                "preliminary_classification": _format_judgments(judgments),
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
                logger.error("[FinalizeChain] Attempt %d failed: %s", attempt, exc)
                if attempt < self._max_retries:
                    time.sleep(3 * attempt)

        if not parsed:
            logger.warning("[FinalizeChain] LLM failed, converting judgments directly")
            selection_result = _judgments_to_selection_result(judgments)
        else:
            selection_result = self._parse_selection_result(parsed)

        return {
            "selection_result": selection_result,
            "directly_supported": [s.model_dump() for s in selection_result.directly_supported],
            "pattern_inferred": [s.model_dump() for s in selection_result.pattern_inferred],
            "no_evidence": [s.model_dump() for s in selection_result.no_evidence],
            "raw_response": raw_response,
            "prompt_system": system,
            "prompt_user": user,
        }

    def _parse_selection_result(self, parsed: Dict[str, Any]) -> SelectionResult:
        directly = [
            SupportedStream(
                entity_name=item.get("entity_name", ""),
                entity_id=item.get("entity_id", ""),
                confidence=float(item.get("confidence", 0.75)),
                evidence=item.get("evidence", ""),
            )
            for item in parsed.get("directly_supported", [])
            if item.get("entity_name")
        ]
        pattern = [
            SupportedStream(
                entity_name=item.get("entity_name", ""),
                entity_id=item.get("entity_id", ""),
                confidence=float(item.get("confidence", 0.55)),
                evidence=item.get("evidence", ""),
            )
            for item in parsed.get("pattern_inferred", [])
            if item.get("entity_name")
        ]
        no_ev = [
            UnsupportedStream(
                entity_name=item.get("entity_name", ""),
                reason=item.get("reason", ""),
            )
            for item in parsed.get("no_evidence", [])
            if item.get("entity_name")
        ]
        return SelectionResult(
            directly_supported=directly,
            pattern_inferred=pattern,
            no_evidence=no_ev,
        )

    def _judgments_from_legacy(self, preliminary: Dict[str, Any]) -> List[CandidateJudgment]:
        """Convert v1 bucket dict to CandidateJudgment list."""
        result = []
        for item in preliminary.get("directly_supported", []):
            result.append(CandidateJudgment(
                entity_name=item.get("entity_name", ""),
                bucket="directly_supported",
                confidence=float(item.get("confidence", 0.75)),
                rationale=item.get("evidence", ""),
            ))
        for item in preliminary.get("pattern_inferred", []):
            result.append(CandidateJudgment(
                entity_name=item.get("entity_name", ""),
                bucket="pattern_inferred",
                confidence=float(item.get("confidence", 0.55)),
                rationale=item.get("evidence", ""),
            ))
        for item in preliminary.get("no_evidence", []):
            result.append(CandidateJudgment(
                entity_name=item.get("entity_name", ""),
                bucket="no_evidence",
                confidence=0.0,
                rationale=item.get("reason", ""),
            ))
        return result
