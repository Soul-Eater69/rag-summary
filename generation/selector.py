"""
Two-pass LLM value-stream verifier and selector (V5 architecture).

Pass 1: Evidence verification -- for each candidate, verify evidence and
        classify as direct/pattern/none.
Pass 2: Final selection -- bucket into directly_supported, pattern_inferred,
        and no_evidence.

Input:
  - new card semantic summary
  - CandidateEvidence objects with fused scores
  - top historical analog summaries (from FAISS)
  - optional raw evidence snippets

Output:
  - directly_supported: list with confidence + evidence
  - pattern_inferred: list with confidence + evidence
  - no_evidence: list
  - raw_response: str
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from summary_rag.ingestion.adapters import (
    LLMService,
    get_default_llm,
    safe_json_extract,
    normalize_text as normalize_for_search,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert at classifying healthcare idea cards into existing HCSC value streams.

You will receive:
1. A semantic summary of a new idea card (with direct and implied functions)
2. Top analogous historical tickets (from a summary index)
3. Candidate value streams with evidence scores from multiple sources

Your job: verify the evidence for each candidate and classify them into three buckets.

Rules:
- Choose ONLY from the candidate value streams provided.
- Do NOT invent, rename, or merge value streams.
- Classify each candidate into exactly one of:
  - "directly_supported": strong direct evidence from card text, summary, attachments, or KG semantics
  - "pattern_inferred": supported mainly by historical analogs, capability mapping, or co-occurrence patterns
  - "no_evidence": insufficient current evidence to justify prediction
- Prefer recall over precision -- include a stream if there is reasonable evidence.
- Use historical analogs as supporting evidence for pattern inference, not sole basis for direct support.
- For each selected stream, cite the specific evidence that justifies it.
- Return valid JSON only.
"""

_USER_PROMPT_TEMPLATE = """
## NEW IDEA CARD SUMMARY

{new_card_summary}

## TOP ANALOGOUS HISTORICAL TICKETS

{analog_summaries}

## CANDIDATE VALUE STREAMS WITH EVIDENCE

{candidate_evidence}

{raw_evidence_section}

## INSTRUCTIONS

For each candidate value stream above, verify the evidence and classify it.

Return JSON exactly in this format:
{{
  "directly_supported": [
    {{
      "entity_name": "string",
      "confidence": 0.0,
      "evidence": "short explanation citing specific direct evidence"
    }}
  ],
  "pattern_inferred": [
    {{
      "entity_name": "string",
      "confidence": 0.0,
      "evidence": "short explanation citing historical patterns or capability signals"
    }}
  ],
  "no_evidence": [
    {{
      "entity_name": "string",
      "reason": "why there is insufficient evidence"
    }}
  ]
}}
"""


def _format_new_card_summary(summary: Dict[str, Any]) -> str:
    parts = []
    if summary.get("short_summary"):
        parts.append(f"Summary: {summary['short_summary']}")
    if summary.get("business_goal"):
        parts.append(f"Business goal: {summary['business_goal']}")
    if summary.get("actors"):
        parts.append(f"Actors: {', '.join(summary['actors'])}")

    # V5: show both raw and canonical functions
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
        cap_tags = analog.get("capability_tags", [])
        if cap_tags:
            lines.append(f"  Capabilities: {', '.join(cap_tags)}")
        footprint = analog.get("operational_footprint", [])
        if footprint:
            lines.append(f"  Footprint: {', '.join(footprint[:6])}")
        # Show stream support classification if available (direct/downstream/pattern)
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
        sources = cand.get("evidence_sources", [])
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


def _norm(name: str) -> str:
    return normalize_for_search((name or "").strip())


def select_value_streams(
    new_card_summary: Dict[str, Any],
    analog_tickets: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    *,
    raw_evidence: Optional[List[Dict[str, Any]]] = None,
    vs_support: Optional[List[Dict[str, Any]]] = None,
    allowed_value_stream_names: Optional[List[str]] = None,
    max_retries: int = 2,
    llm: Optional[LLMService] = None,
) -> Dict[str, Any]:
    """
    Two-pass LLM evidence verification and value stream selection.

    Candidates should be CandidateEvidence objects from the fusion stage.

    Returns dict with:
      - directly_supported, pattern_inferred, no_evidence (V5 three-class output)
      - selected_value_streams (union of directly_supported + pattern_inferred for compat)
      - rejected_candidates
      - raw_response, prompt_system, prompt_user
    """
    t_start = time.time()

    # Filter to allowed if specified
    if allowed_value_stream_names:
        allowed_set = {_norm(n) for n in allowed_value_stream_names}
        candidates = [
            c for c in candidates
            if _norm(c.get("candidate_name") or c.get("entity_name", "")) in allowed_set
        ]

    # Build prompt
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        new_card_summary=_format_new_card_summary(new_card_summary),
        analog_summaries=_format_analog_summaries(analog_tickets),
        candidate_evidence=_format_candidate_evidence(candidates),
        raw_evidence_section=_format_raw_evidence(raw_evidence or []),
    )

    gen_svc = llm or get_default_llm()
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

            if not isinstance(parsed, dict):
                parsed = None
                continue

            # Validate expected structure
            if any(k in parsed for k in ("directly_supported", "pattern_inferred", "no_evidence")):
                break

            # Legacy format compat: if LLM returns old format
            if "selected_value_streams" in parsed:
                parsed = _convert_legacy_response(parsed)
                break

            parsed = None

        except Exception as exc:
            logger.error("[SummaryRAG-LLM] Attempt %d failed: %s", attempt, exc)
            if attempt < max_retries:
                time.sleep(3 * attempt)

    if not parsed:
        logger.warning("[SummaryRAG] LLM returned no valid response, using fallback")
        parsed = _fallback_selection(candidates, vs_support or [])

    # Filter to allowed
    if allowed_value_stream_names:
        allowed_set = {_norm(n) for n in allowed_value_stream_names}
        for key in ("directly_supported", "pattern_inferred", "no_evidence"):
            parsed[key] = [
                vs for vs in parsed.get(key, [])
                if _norm(vs.get("entity_name", "")) in allowed_set
            ]

    # Build compat selected_value_streams (union of direct + pattern)
    selected = []
    for vs in parsed.get("directly_supported", []):
        selected.append({
            "entity_id": vs.get("entity_id", ""),
            "entity_name": vs.get("entity_name", ""),
            "confidence": vs.get("confidence", 0.8),
            "reason": vs.get("evidence", ""),
            "support_type": "direct",
        })
    for vs in parsed.get("pattern_inferred", []):
        selected.append({
            "entity_id": vs.get("entity_id", ""),
            "entity_name": vs.get("entity_name", ""),
            "confidence": vs.get("confidence", 0.6),
            "reason": vs.get("evidence", ""),
            "support_type": "pattern",
        })

    rejected = [
        {
            "entity_name": vs.get("entity_name", ""),
            "reason": vs.get("reason", "No evidence"),
        }
        for vs in parsed.get("no_evidence", [])
    ]

    elapsed = time.time() - t_start
    logger.info(
        "[SummaryRAG] Selection done in %.2fs | direct=%d | pattern=%d | no_evidence=%d",
        elapsed,
        len(parsed.get("directly_supported", [])),
        len(parsed.get("pattern_inferred", [])),
        len(parsed.get("no_evidence", [])),
    )

    return {
        "directly_supported": parsed.get("directly_supported", []),
        "pattern_inferred": parsed.get("pattern_inferred", []),
        "no_evidence": parsed.get("no_evidence", []),
        "selected_value_streams": selected,
        "rejected_candidates": rejected,
        "raw_response": raw_response,
        "prompt_system": _SYSTEM_PROMPT,
        "prompt_user": user_prompt,
        "candidates_after_filter": candidates,
    }


def _convert_legacy_response(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Convert old-format LLM response to V5 three-class format."""
    direct = []
    pattern = []
    for vs in parsed.get("selected_value_streams", []):
        entry = {
            "entity_name": vs.get("entity_name", ""),
            "confidence": vs.get("confidence", 0.7),
            "evidence": vs.get("reason", ""),
        }
        # Heuristic: high confidence -> direct, lower -> pattern
        if float(vs.get("confidence", 0.7)) >= 0.7:
            direct.append(entry)
        else:
            pattern.append(entry)

    return {
        "directly_supported": direct,
        "pattern_inferred": pattern,
        "no_evidence": [],
    }


def _fallback_selection(
    candidates: List[Dict[str, Any]],
    vs_support: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Fallback: classify candidates based on their support_type and fused_score."""
    direct = []
    pattern = []
    no_evidence = []

    for cand in candidates:
        name = cand.get("candidate_name") or cand.get("entity_name", "")
        if not name:
            continue

        support_type = cand.get("support_type", "none")
        fused = float(cand.get("fused_score", 0.0))

        entry = {
            "entity_name": name,
            "confidence": round(fused, 2),
            "evidence": f"Fallback classification from support_type={support_type}",
        }

        if support_type in ("direct", "mixed") and fused >= 0.3:
            direct.append(entry)
        elif support_type == "pattern" and fused >= 0.2:
            pattern.append(entry)
        else:
            no_evidence.append({
                "entity_name": name,
                "reason": f"Fallback: fused_score={fused:.2f}, support_type={support_type}",
            })

    # Also consider VS support entries not in candidates
    cand_names = {_norm(c.get("candidate_name") or c.get("entity_name", "")) for c in candidates}
    for entry in sorted(vs_support, key=lambda x: -x.get("support_count", 0))[:5]:
        name = entry.get("entity_name", "")
        if _norm(name) in cand_names:
            continue
        pattern.append({
            "entity_name": name,
            "confidence": 0.5,
            "evidence": f"Historical support from {entry.get('support_count', 0)} analog tickets.",
        })

    return {
        "directly_supported": direct,
        "pattern_inferred": pattern,
        "no_evidence": no_evidence,
    }
