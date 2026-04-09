"""
Selector finalize chain - Pass 2 (V6 / Phase 4 architecture).

Consumes the output of taxonomy_policy_rerank (preferred) or raw
VerificationResult (List[CandidateJudgment]) from Pass 1, and produces
a validated SelectionResult (Pydantic model).

Phase 4 input priority:
  1. taxonomy_reranked_candidates — eligibility-filtered, sibling-resolved,
     prior-adjusted candidates from node_taxonomy_policy_rerank (preferred)
  2. verify_judgments — raw Pass 1 judgments (fallback when reranker absent)

Responsibilities: convert reranked candidates into three-class output
(directly_supported / pattern_inferred / no_evidence), apply deduplication,
calibration, and pattern metadata annotation.

Primary path: provider-native structured output via structured_generate().
Fallback: direct judgment-to-result conversion without LLM.

Prompt version hierarchy:
  v3 (default): schema-light, relies on structured output
  v2 / v1: legacy variants with JSON skeleton
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import json
import re

from rag_summary.ingestion.adapters import (
    LLMService,
    get_default_llm,
    structured_generate,
)
from rag_summary.models.candidate_judgment import CandidateJudgment, VerificationResult
from rag_summary.models.selection import SelectionResult, SupportedStream, UnsupportedStream
from .prompt_loader import load_prompt, render_prompt
from .selector_verify_chain import _format_new_card_summary

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"\b(?:IDMT|CP|SP|CC)-\d+\b|\bIDMT-\d+\b", re.IGNORECASE)


def _is_gateway_timeout_error(exc: Exception) -> bool:
    stack = [exc]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)

        status_code = getattr(current, "status_code", None)
        if status_code == 504:
            return True

        response = getattr(current, "response", None)
        if response is not None and getattr(response, "status_code", None) == 504:
            return True

        message = str(current).lower()
        if "504" in message and ("timeout" in message or "gateway" in message):
            return True
        if "gateway timeout" in message:
            return True

        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if isinstance(cause, Exception):
            stack.append(cause)
        if isinstance(context, Exception):
            stack.append(context)
    return False


def _infer_pattern_basis(evidence: str) -> str:
    text = (evidence or "").lower()
    if "bundle" in text or "co-occurrence" in text:
        return "bundle_pattern"
    if "downstream" in text:
        return "downstream_chain"
    if "theme" in text:
        return "theme"
    if "capability" in text or "overlap" in text:
        return "capability_overlap"
    return "analog_similarity"


def _extract_analog_ids(evidence: str, limit: int = 2) -> List[str]:
    ids = []
    for match in _ID_RE.findall(evidence or ""):
        cleaned = match.upper()
        if cleaned not in ids:
            ids.append(cleaned)
        if len(ids) >= limit:
            break
    return ids


def _ensure_pattern_metadata(result: SelectionResult) -> SelectionResult:
    patched: List[SupportedStream] = []
    for stream in result.pattern_inferred:
        basis = stream.pattern_basis or _infer_pattern_basis(stream.evidence)
        analog_ids = stream.supporting_analog_ids
        if not analog_ids:
            extracted = _extract_analog_ids(stream.evidence)
            analog_ids = extracted if extracted else []
        patched.append(
            stream.model_copy(
                update={
                    "pattern_basis": basis,
                    "supporting_analog_ids": analog_ids,
                }
            )
        )
    return result.model_copy(update={"pattern_inferred": patched})


def _format_judgments(judgments: List[CandidateJudgment]) -> str:
    if not judgments:
        return "No judgments available."
    lines = []
    for j in judgments:
        conf_str = f" (conf:{j.confidence:.2f})" if j.confidence > 0 else ""
        rat_str = f" - {j.rationale}" if j.rationale else ""
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

    return _ensure_pattern_metadata(SelectionResult(
        directly_supported=directly,
        pattern_inferred=pattern,
        no_evidence=no_ev,
    ))


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
        self.last_prompt_payload: Dict[str, Any] = {}

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
        on_prompt: Optional[Callable[[Dict[str, Any]], None]] = None,
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
        self.last_prompt_payload = {
            "system": system,
            "user": user,
        }

        with open("finalize_chain_last_prompt.json", "w", encoding="utf-8") as f:
            json.dump(self.last_prompt_payload, f, ensure_ascii=False, indent=2)

        if on_prompt:
            try:
                on_prompt(dict(self.last_prompt_payload))
            except Exception as exc:
                logger.warning("[FinalizeChain] on_prompt callback failed: %s", exc)

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
                    return _ensure_pattern_metadata(result)
            except Exception as exc:
                if _is_gateway_timeout_error(exc):
                    logger.error("[FinalizeChain] Gateway timeout (504) detected; failing fast")
                    raise
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
