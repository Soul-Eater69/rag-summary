"""
LangChain-style selector verification chain — Pass 1 (V3 architecture).

Returns VerificationResult (flat List[CandidateJudgment]) — one judgment per
candidate. Pass 2 (SelectorFinalizeChain) groups and calibrates these into the
final SelectionResult.

Primary path: provider-native structured output via structured_generate().
Fallback: text generation + JSON parse + model_validate().

Prompt version hierarchy:
  v3 (default): schema-light, relies on structured output
  v2: schema in prompt, per-candidate flat list
  v1: legacy grouped bucket output (compat)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from rag_summary.ingestion.adapters import (
    LLMService,
    get_default_llm,
    structured_generate,
)
from rag_summary.models.candidate_judgment import (
    CandidateJudgment,
    VerificationResult,
    BucketLabel,
)

from .prompt_loader import load_prompt, render_prompt

logger = logging.getLogger(__name__)

_VALID_BUCKETS: set = {"directly_supported", "pattern_inferred", "no_evidence"}


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


# ----------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------

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
        parts.append(f"Direct functions: {', '.join(raw_direct)}")
        if canon_direct != raw_direct:
            parts.append(f"Canonical functions: {', '.join(canon_direct)}")

    raw_implied = summary.get("implied_functions_raw") or summary.get("implied_functions") or []
    if raw_implied:
        parts.append(f"Implied functions: {', '.join(raw_implied)}")

    if summary.get("capability_tags"):
        parts.append(f"Capability tags: {', '.join(summary['capability_tags'])}")
    if summary.get("operational_footprint"):
        parts.append(f"Footprint: {', '.join(summary['operational_footprint'][:4])}")
    if summary.get("change_types"):
        parts.append(f"Change types: {', '.join(summary['change_types'])}")
    if summary.get("domain_tags"):
        parts.append(f"Domain: {', '.join(summary['domain_tags'])}")

    if not parts:
        return "Summary unavailable: use candidate evidence, analog summaries, and fused scores as primary grounding."
    return "\n".join(parts)


def _format_analog_summaries(analogs: List[Dict[str, Any]], limit: int = 5) -> str:
    if not analogs:
        return "No historical analogs found."

    parts = []
    for i, analog in enumerate(analogs[:limit], 1):
        lines = [f"#{i} Analog {analog.get('ticket_id', '?')} (score: {analog.get('score', 0):.3f})"]
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
            lines.append(f"  Footprint: {', '.join(footprint[:5])}")

        sst = analog.get("stream_support_type") or {}
        if sst:
            lines.append(f"  Support types: {', '.join(f'{k}:{v}' for k, v in list(sst.items())[:3])}")

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
        active = {k: round(v, 2) for k, v in source_scores.items() if v > 0}
        line = f"{name} | fused:{fused:.3f}, type:{support_type}, sources:{diversity}"
        if active:
            line += f" | Active: {active}"
        parts.append(line)

    return "\n".join(parts)


def _format_raw_evidence(evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return ""

    parts = ["## RAW EVIDENCE SNIPPETS\n"]
    for ev in evidence[:8]:
        parts.append(f"- [{ev.get('ticket_id', '?')}] {ev.get('snippet', '')}")
    return "\n".join(parts)


def _fallback_judgments_from_candidates(
    candidates: List[Dict[str, Any]],
    candidate_names: List[str],
) -> VerificationResult:
    """
    Degraded-mode fallback when verifier LLM fails.

    Uses fused ranking as weak evidence so we avoid blanket rejection while
    still being conservative about confidence and bucket assignment.
    """
    if not candidates and not candidate_names:
        return VerificationResult(judgments=[])

    ranked: List[Dict[str, Any]] = sorted(
        candidates,
        key=lambda c: float(c.get("fused_score", 0.0)),
        reverse=True,
    )

    max_fused = max((float(c.get("fused_score", 0.0)) for c in ranked), default=0.0)
    judgments: List[CandidateJudgment] = []
    seen: set[str] = set()

    for rank, cand in enumerate(ranked, start=1):
        name = (cand.get("candidate_name") or cand.get("entity_name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)

        fused = float(cand.get("fused_score", 0.0))
        diversity = int(cand.get("source_diversity_count", 0) or 0)
        rel = (fused / max_fused) if max_fused > 0 else 0.0

        if rel >= 0.92 and diversity >= 2:
            bucket: BucketLabel = "directly_supported"
            confidence = max(0.62, min(0.78, 0.60 + 0.10 * rel + 0.02 * diversity))
        elif rel >= 0.70 and rank <= 8:
            bucket = "pattern_inferred"
            confidence = max(0.46, min(0.64, 0.42 + 0.12 * rel + 0.015 * diversity))
        else:
            bucket = "no_evidence"
            confidence = 0.0

        rationale = (
            "Verifier unavailable; fallback used fused ranking "
            f"(rank {rank}, fused={fused:.3f}, sources={diversity})."
        )
        if bucket == "no_evidence":
            rationale = (
                "Verifier unavailable; candidate remained below fallback threshold "
                f"(rank {rank}, fused={fused:.3f})."
            )

        judgments.append(
            CandidateJudgment(
                entity_name=name,
                bucket=bucket,
                confidence=confidence,
                rationale=rationale,
            )
        )

    # Ensure any candidates present only in name list are covered.
    seen_lower = {j.entity_name.lower() for j in judgments}
    for name in candidate_names:
        if not name:
            continue
        key = name.lower()
        if key in seen_lower:
            continue
        judgments.append(
            CandidateJudgment(
                entity_name=name,
                bucket="no_evidence",
                confidence=0.0,
                rationale="Verifier unavailable; candidate missing from fused ranking.",
            )
        )

    return VerificationResult(judgments=judgments)


# ----------------------------------------------------------------------
# Chain
# ----------------------------------------------------------------------

class SelectorVerifyChain:
    """
    Pass 1 of the two-pass LLM verifier.

    Returns VerificationResult (flat List[CandidateJudgment]) — one per candidate.
    Pass 2 (SelectorFinalizeChain) groups and calibrates into SelectionResult.
    """

    def __init__(
        self,
        llm: Optional[LLMService] = None,
        prompt_version: str = "v3",
        max_retries: int = 2,
    ) -> None:
        self._llm = llm
        self._max_retries = max_retries
        self._prompt_version = prompt_version
        self._prompt = load_prompt("verify_candidates", version=prompt_version)
        self.last_prompt_payload: Dict[str, str] = {}

    @property
    def llm(self) -> LLMService:
        if self._llm is None:
            self._llm = get_default_llm()
        return self._llm

    def run(
        self,
        new_card_summary: Dict[str, Any],
        analog_tickets: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        raw_evidence: Optional[List[Dict[str, Any]]] = None,
        on_prompt: Optional[Any] = None,
    ) -> VerificationResult:
        """
        Run Pass 1 verification.

        Returns VerificationResult with one CandidateJudgment per candidate.
        Any candidates missing from LLM output are covered with no_evidence fallback.
        """
        candidate_names = [
            (c.get("candidate_name") or c.get("entity_name") or "").strip()
            for c in candidates
        ]

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

        self.last_prompt_payload = {"system": system, "user": user}
        if on_prompt:
            try:
                on_prompt(dict(self.last_prompt_payload))
            except Exception as exc:
                logger.warning("[VerifyChain] on_prompt callback failed: %s", exc)

        for attempt in range(1, self._max_retries + 1):
            try:
                result = structured_generate(
                    self.llm,
                    user,
                    VerificationResult,
                    context="",
                    system_prompt=system,
                )
                if result.judgments:
                    return self._ensure_coverage(result, candidate_names)
            except Exception as exc:
                if _is_gateway_timeout_error(exc):
                    logger.error("[VerifyChain] Gateway timeout (504) detected; failing fast")
                    raise
                logger.error("[VerifyChain] Attempt %d failed: %s", attempt, exc)
                if attempt < self._max_retries:
                    time.sleep(3 * attempt)

        logger.warning(
            "[VerifyChain] All attempts failed, using fused-score fallback judgments"
        )
        return _fallback_judgments_from_candidates(candidates, candidate_names)

    def _ensure_coverage(
        self,
        result: VerificationResult,
        candidate_names: List[str],
    ) -> VerificationResult:
        """
        Ensure every candidate has exactly one judgment.

        Adds no_evidence fallback for candidates missing from LLM output.
        Normalises invalid bucket labels.
        """
        seen: Dict[str, CandidateJudgment] = {}
        for j in result.judgments:
            name = j.entity_name.strip()
            if not name:
                continue

            # Clamp invalid bucket labels
            if j.bucket not in _VALID_BUCKETS:
                j = j.model_copy(update={"bucket": "no_evidence"})
            seen[name] = j

        final = list(seen.values())
        seen_lower = {n.lower() for n in seen}

        for name in candidate_names:
            if name and name.lower() not in seen_lower:
                final.append(
                    CandidateJudgment(
                        entity_name=name,
                        bucket="no_evidence",
                        confidence=0.0,
                        rationale="Not covered in verification pass.",
                    )
                )

        return VerificationResult(judgments=final)
