"""
LangChain-style selector finalize chain — Pass 2 (V5 architecture).

Consumes VerificationResult (List[CandidateJudgment]) from Pass 1 and produces
a validated SelectionResult (Pydantic model).

Responsibilities: deduplication, calibration, contradiction cleanup,
fused-score reconciliation, demotion of weak predictions.

Primary path: provider-native structured output via structured_generate().
Fallback: direct judgment-to-result conversion without LLM.

Prompt version hierarchy:
  v3 (default): schema-light, relies on structured output
  v2 / v1: legacy variants with JSON skeleton
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from summary_rag.ingestion.adapters import (
    LLMService,
    get_default_llm,
    structured_generate,
)
from summary_rag.models.candidate_judgment import CandidateJudgment, VerificationResult
from summary_rag.models.selection import SelectionResult, SupportedStream, UnsupportedStream
from .prompt_loader import load_prompt, render_prompt
from .selector_verify_chain import _format_new_card_summary

logger = logging.getLogger(__name__)


def _format_judgments(judgments: List[CandidateJudgment]) -> str:
    if not judgments:
        return "No judgments available."
    lines = []
    for j in judgments:
        conf_str = f" (conf:{j.confidence:.2f})" if j.confidence > 0 else ""
        rat_str = f" — {j.rationale}" if j.rationale else ""
        lines.append(f"- [{j.bucket}]{conf_str} {j.entity_name}{rat_str}")
    return "\n".join(lines)


def _format_fused_scores(fused_candidates: Optional[List[Dict[str, Any]]]) -> str:
    if not fused_candidates:
        return "Not available."
    top = sorted(fused_candidates, key=lambda x: -x.get("fused_score", 0))[:12]
    return "\n".join(
        f"- {c.get('candidate_name') or c.get('entity_name', '')}: "
        f"fused={c.get('fused_score', 0):.3f}, sources={c.get('source_diversity_count', 0)}"
        for c in top
    )


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

    Consumes VerificationResult from SelectorVerifyChain, applies
    dedup / calibration / contradiction cleanup, and returns SelectionResult.

    Falls back to direct judgment-to-result conversion if LLM fails.
    """

    def __init__(
        self,
        *,
        llm: Optional[LLMService] = None,
        prompt_version: str = "v3",
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
        judgments: Optional[List[CandidateJudgment]] = None,
        fused_candidates: Optional[List[Dict[str, Any]]] = None,
        # v1/v2 compat: accept preliminary_classification dict
        preliminary_classification: Optional[Dict[str, Any]] = None,
        verification_result: Optional[VerificationResult] = None,
    ) -> SelectionResult:
        """
        Run Pass 2 finalization.

        Returns validated SelectionResult (Pydantic model).

        Accepts:
          - judgments: List[CandidateJudgment] from Pass 1
          - verification_result: VerificationResult from Pass 1 (preferred)
          - preliminary_classification: legacy dict input (v1/v2 compat)
        """
        # Normalise inputs
        if verification_result is not None:
            judgments = verification_result.judgments
        elif judgments is None and preliminary_classification:
            judgments = _judgments_from_legacy(preliminary_classification)
        elif judgments is None:
            judgments = []

        system = self._prompt["system"]
        user = render_prompt(
            self._prompt,
            {
                "new_card_summary": _format_new_card_summary(new_card_summary),
                "candidate_judgments": _format_judgments(judgments),
                "fused_scores_section": _format_fused_scores(fused_candidates),
                # v1/v2 compat variable name
                "preliminary_classification": _format_judgments(judgments),
            },
            role="user",
        )

        for attempt in range(1, self._max_retries + 1):
            try:
                result = structured_generate(
                    self.llm,
                    user,
                    SelectionResult,
                    context="",
                    system_prompt=system,
                )
                if (
                    result.directly_supported
                    or result.pattern_inferred
                    or result.no_evidence
                ):
                    return result
            except Exception as exc:
                logger.error("[FinalizeChain] Attempt %d failed: %s", attempt, exc)
                if attempt < self._max_retries:
                    time.sleep(3 * attempt)

        logger.warning("[FinalizeChain] LLM failed, converting judgments directly")
        return _judgments_to_selection_result(judgments)


def _judgments_from_legacy(preliminary: Dict[str, Any]) -> List[CandidateJudgment]:
    """Convert v1/v2 bucket dict to CandidateJudgment list."""
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
