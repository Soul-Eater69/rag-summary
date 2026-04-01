"""
LLM value-stream selector: builds a compact evidence package and asks the LLM
to pick the best-matching value streams.

Input:
  - new card semantic summary
  - KG candidate value streams
  - top historical analog summaries (from FAISS)
  - optional raw evidence snippets

Output:
  - selected_value_streams with confidence + reason
  - rejected_candidates with reason
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from core.constants import CANONICAL_VALUE_STREAMS
from core.prompts import safe_json_extract
from core.text import normalize_for_search
from src.services.generation_service import GenerationService

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert at classifying healthcare idea cards into existing HCSC value streams.

You will receive:
1. A semantic summary of a new idea card
2. Top analogous historical tickets (from a summary index)
3. Candidate value streams with descriptions

Your job: select the value streams that best match the new idea card.

Rules:
- Choose ONLY from the candidate value streams provided.
- Do NOT invent, rename, or merge value streams.
- Prefer recall over precision - include a value stream if there is reasonable evidence.
- Use the historical analog summaries as supporting evidence, not as the sole basis.
- Return valid JSON only.
"""

_USER_PROMPT_TEMPLATE = """
## NEW IDEA CARD SUMMARY

{new_card_summary}

## TOP ANALOGOUS HISTORICAL TICKETS

{analog_summaries}

## CANDIDATE VALUE STREAMS

{candidate_list}

{raw_evidence_section}

## INSTRUCTIONS

Select the best-matching value streams for this idea card from the candidates above.
Consider what the historical analogs were mapped to as supporting evidence.

Return JSON exactly in this format:
{{
  "selected_value_streams": [
    {{
      "entity_id": "string",
      "entity_name": "string",
      "confidence": 0.0,
      "reason": "short explanation citing evidence"
    }}
  ]
}}
"""

def _format_new_card_summary(summary: Dict[str, Any]) -> str:
    """Format new card summary for the prompt."""
    parts = []
    if summary.get("short_summary"):
        parts.append(f"Summary: {summary['short_summary']}")
    if summary.get("business_goal"):
        parts.append(f"Business goal: {summary['business_goal']}")
    if summary.get("actors"):
        parts.append(f"Actors: {', '.join(summary['actors'])}")
    if summary.get("direct_functions"):
        parts.append(f"Direct functions: {', '.join(summary['direct_functions'])}")
    if summary.get("implied_functions"):
        parts.append(f"Implied functions: {', '.join(summary['implied_functions'])}")
    if summary.get("change_types"):
        parts.append(f"Change types: {', '.join(summary['change_types'])}")
    if summary.get("domain_tags"):
        parts.append(f"Domain: {', '.join(summary['domain_tags'])}")
    return "\n".join(parts)

def _format_analog_summaries(analogs: List[Dict[str, Any]], limit: int = 5) -> str:
    """Format analog ticket summaries for the prompt."""
    if not analogs:
        return "No historical analogs found."

    parts = []
    for i, analog in enumerate(analogs[:limit], 1):
        lines = [f"### Analog {i}: {analog.get('ticket_id', '?')} (score: {analog.get('score', 0):.4f})"]
        if analog.get("short_summary"):
            lines.append(f"  Summary: {analog['short_summary']}")
        if analog.get("direct_functions"):
            lines.append(f"  Functions: {', '.join(analog['direct_functions'])}")
        if analog.get("value_stream_labels"):
            lines.append(f"  Mapped VS: {', '.join(analog['value_stream_labels'])}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)

def _format_candidates(candidates: List[Dict[str, Any]], limit: int = 15) -> str:
    """Format KG candidate value streams for the prompt."""
    if not candidates:
        return "No candidates available."

    parts = []
    for cand in candidates[:limit]:
        name = cand.get("entity_name") or cand.get("name") or ""
        eid = cand.get("entity_id") or cand.get("id") or ""
        desc = (cand.get("description") or cand.get("value_proposition") or "")[:200]
        score = float(cand.get("score") or cand.get("best_score") or 0.0)
        parts.append(f"- {name} ({eid}) [score: {score:.4f}]\n  {desc}")

    return "\n".join(parts)

def _format_raw_evidence(evidence: List[Dict[str, Any]]) -> str:
    """Format raw evidence snippets, if any."""
    if not evidence:
        return ""

    parts = ["## RAW EVIDENCE SNIPPETS (for verification)\n"]
    for ev in evidence[:6]:
        parts.append(f"[{ev.get('ticket_id', '?')}] {ev.get('snippet', '')}")

    return "\n".join(parts)

def _norm(name: str) -> str:
    return normalize_for_search((name or "").strip())

def select_value_streams(
    new_card_summary: Dict[str, Any],
    analog_tickets: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    *,
    raw_evidence: Optional[List[Dict[str, Any]]] = None,
    vs_support: Optional[List[Dict[str, Any]]] = None,
    allowed_names: Optional[List[str]] = None,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Build compact evidence package and ask LLM to select value streams.

    Returns dict with selected_value_streams, rejected_candidates, raw_response.
    """
    t_start = time.time()

    # Inject VS-support candidates that aren't in KG results
    if vs_support:
        _inject_support_candidates(candidates, vs_support)

    # Filter to allowed if specified
    if allowed_names:
        allowed_set = {_norm(n) for n in allowed_names}
        candidates = [c for c in candidates if _norm(c.get("entity_name", "")) in allowed_set]

    # Build prompt
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        new_card_summary=_format_new_card_summary(new_card_summary),
        analog_summaries=_format_analog_summaries(analog_tickets),
        candidate_list=_format_candidates(candidates),
        raw_evidence_section=_format_raw_evidence(raw_evidence or []),
    )

    gen_svc = GenerationService()
    parsed: Optional[Dict] = None
    raw_response: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:
            reply = gen_svc.generate(
                query=user_prompt,
                context="",
                system_prompt=_SYSTEM_PROMPT,
            )
            raw_response = reply.content
            parsed = safe_json_extract(raw_response)

            if isinstance(parsed, list):
                parsed = {"selected_value_streams": parsed, "rejected_candidates": []}
            elif not isinstance(parsed, dict):
                parsed = {"selected_value_streams": [], "rejected_candidates": []}

            break
        except Exception as exc:
            logger.error("[SummaryRAG-LLM] Attempt %d failed: %s", attempt, exc)
            if attempt < max_retries:
                time.sleep(3 * attempt)

    if not parsed or not parsed.get("selected_value_streams"):
        logger.warning("[SummaryRAG] LLM returned no selections, using fallback")
        parsed = _fallback_selection(vs_support or [], candidates)

    # Filter to allowed
    if allowed_names:
        allowed_set = {_norm(n) for n in allowed_names}
        parsed["selected_value_streams"] = [
            vs for vs in parsed["selected_value_streams"]
            if _norm(vs.get("entity_name", "")) in allowed_set
        ]

    elapsed = time.time() - t_start
    logger.info(
        "[SummaryRAG] Selection done in %.2fs | selected=%d",
        elapsed, len(parsed.get("selected_value_streams", [])),
    )

    return {
        "selected_value_streams": parsed.get("selected_value_streams", []),
        "rejected_candidates": parsed.get("rejected_candidates", []),
        "raw_response": raw_response,
    }

def _inject_support_candidates(
    candidates: List[Dict[str, Any]],
    vs_support: List[Dict[str, Any]],
) -> int:
    """Inject VS support entries that aren't already in KG candidates."""
    existing = {_norm(c.get("entity_name", "")) for c in candidates}
    injected = 0

    for entry in vs_support:
        name = entry.get("entity_name", "").strip()
        if not name or _norm(name) in existing:
            continue
        candidates.append({
            "entity_id": "",
            "entity_name": name,
            "description": f"Historical support from {entry.get('support_count', 0)} analog tickets.",
            "score": float(entry.get("best_score", 0.0)),
            "source": "historical_support",
        })
        existing.add(_norm(name))
        injected += 1

    return injected

def _fallback_selection(
    vs_support: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    top_n: int = 8,
) -> Dict[str, Any]:
    """Fallback: pick top VS support or top KG candidates."""
    selected = []
    # Prefer VS support
    for entry in sorted(vs_support, key=lambda x: -x.get("support_count", 0))[:top_n]:
        selected.append({
            "entity_id": "",
            "entity_name": entry.get("entity_name", ""),
            "confidence": 0.5,
            "reason": f"Fallback from {entry.get('support_count', 0)} historical analog tickets.",
        })

    # Fill from candidates if needed
    if len(selected) < top_n:
        existing = {_norm(s.get("entity_name", "")) for s in selected}
        for cand in candidates[:top_n * 2]:
            if _norm(cand.get("entity_name", "")) in existing:
                continue
            selected.append({
                "entity_id": cand.get("entity_id", ""),
                "entity_name": cand.get("entity_name", ""),
                "confidence": 0.4,
                "reason": "Fallback from top KG candidates.",
            })
            if len(selected) >= top_n:
                break
    return {"selected_value_streams": selected, "rejected_candidates": []}