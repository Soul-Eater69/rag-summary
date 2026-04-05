"""
LangChain-style selector finalize chain — Pass 2 (V5 architecture).

Takes the preliminary classification from Pass 1 and produces the final
calibrated three-class output.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from summary_rag.ingestion.adapters import LLMService, get_default_llm, safe_json_extract
from .prompt_loader import load_prompt, render_prompt
from .selector_verify_chain import _format_new_card_summary

logger = logging.getLogger(__name__)


def _format_preliminary(preliminary: Dict[str, Any]) -> str:
    """Format pass-1 output for the finalize prompt."""
    lines = []

    direct = preliminary.get("directly_supported", [])
    if direct:
        lines.append("DIRECTLY SUPPORTED:")
        for item in direct:
            conf = item.get("confidence", 0)
            lines.append(f"  - {item.get('entity_name', '')} (conf: {conf:.2f}): {item.get('evidence', '')}")

    pattern = preliminary.get("pattern_inferred", [])
    if pattern:
        lines.append("PATTERN INFERRED:")
        for item in pattern:
            conf = item.get("confidence", 0)
            lines.append(f"  - {item.get('entity_name', '')} (conf: {conf:.2f}): {item.get('evidence', '')}")

    no_ev = preliminary.get("no_evidence", [])
    if no_ev:
        lines.append("NO EVIDENCE:")
        for item in no_ev:
            lines.append(f"  - {item.get('entity_name', '')}: {item.get('reason', '')}")

    return "\n".join(lines) if lines else "No preliminary classification available."


class SelectorFinalizeChain:
    """
    Pass 2 of the two-pass LLM verifier.

    Reviews preliminary classifications and produces the final calibrated output.
    Falls back to pass-1 output if LLM call fails.
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
        preliminary_classification: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run pass 2 finalization.

        Returns dict with: directly_supported, pattern_inferred, no_evidence,
        raw_response, prompt_system, prompt_user.
        Falls back to preliminary_classification on failure.
        """
        system = self._prompt["system"]
        user = render_prompt(
            self._prompt,
            {
                "new_card_summary": _format_new_card_summary(new_card_summary),
                "preliminary_classification": _format_preliminary(preliminary_classification),
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
            # Fall back to preliminary pass-1 output
            logger.warning("[FinalizeChain] LLM failed, using pass-1 output as final")
            parsed = preliminary_classification

        return {
            "directly_supported": parsed.get("directly_supported", []),
            "pattern_inferred": parsed.get("pattern_inferred", []),
            "no_evidence": parsed.get("no_evidence", []),
            "raw_response": raw_response,
            "prompt_system": system,
            "prompt_user": user,
        }
