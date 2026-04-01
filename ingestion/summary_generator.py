"""
Summary generator: Uses LLM to create a structured semantic summary of a
historical ticket or a new idea card.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.services.generation_service import GenerationService
from core.prompts import safe_json_extract

logger = logging.getLogger(__name__)

_HISTORICAL_SUMMARY_SYSTEM = """You are an expert at summarizing healthcare enterprise project tickets.

Your goal is to create a dense, semantic summary that captures:
1. What the project actually did (the "what")
2. Who it affected (actors)
3. The business intent (the "why")
4. Functional domains (direct and implied)

This summary will be used for vector retrieval to find analogous projects.
"""

_HISTORICAL_SUMMARY_USER = """
## TICKET DATA
ID: {ticket_id}
TITLE: {title}
KNOWN VALUE STREAMS: {vs_labels}

## CONTENT
{ticket_text}

## INSTRUCTIONS
Extract a structured summary. Be precise.
- short_summary: 1-2 sentences of the core technical/business change.
- business_goal: The primary objective or pain point addressed.
- actors: List of stakeholders (e.g., Member, Provider, Broker, Internal Ops).
- direct_functions: Explicit capabilities mentioned (e.g., Claims processing, Enrollment).
- implied_functions: Capabilities logically required but not explicitly named.
- change_types: e.g., New Capability, Regulatory Compliance, Process Improvement, Migration.
- domain_tags: e.g., Clinical, Financial, IT, Operational.
- evidence_sentences: 2-3 key sentences from the text that justify the summary.

Return valid JSON ONLY.
"""

_IDEA_CARD_SUMMARY_SYSTEM = """You are a healthcare business analyst.
Your goal is to extract a structured semantic summary from an "Idea Card" PPT.
Focus on business intent, functional impact, and affected stakeholders.
"""

_IDEA_CARD_SUMMARY_USER = """
## IDEA CARD CONTENT
{ppt_text}

## INSTRUCTIONS
Extract a structured summary.
- short_summary: 1-2 sentence high-level summary.
- business_goal: The core problem or opportunity.
- actors: Affected stakeholders.
- direct_functions: Explicitly mentioned capabilities.
- implied_functions: Logically required capabilities.
- change_types: Type of change (e.g., New Product, Efficiency).
- domain_tags: Relevant healthcare domains.

Return valid JSON ONLY.
"""

def generate_historical_ticket_summary(
    ticket_text: str,
    ticket_id: str,
    title: str,
    value_stream_labels: List[str],
    model: str = "gpt-5-mini-idp",
) -> Dict[str, Any]:
    """Use LLM to generate a structured summary of a historical ticket."""
    prompt = _HISTORICAL_SUMMARY_USER.format(
        ticket_id=ticket_id,
        title=title,
        vs_labels=", ".join(value_stream_labels),
        ticket_text=ticket_text[:6000], # Truncate to avoid context limits
    )

    gen_svc = GenerationService()
    try:
        reply = gen_svc.generate(
            query=prompt,
            context="",
            system_prompt=_HISTORICAL_SUMMARY_SYSTEM,
            # model=model # Uncomment if your service supports dynamic model selection
        )
        parsed = safe_json_extract(reply.content)
        
        # Add metadata
        parsed["ticket_id"] = ticket_id
        parsed["title"] = title
        parsed["value_stream_labels"] = value_stream_labels
        parsed["doc_id"] = f"summary_{ticket_id}"
        
        return parsed
    except Exception as exc:
        logger.error("Failed to generate summary for %s: %s", ticket_id, exc)
        raise

def generate_idea_card_semantic_summary(
    ppt_text: str,
    model: str = "gpt-5-mini-idp",
) -> Dict[str, Any]:
    """Use LLM to generate a structured summary of a new idea card."""
    prompt = _IDEA_CARD_SUMMARY_USER.format(
        ppt_text=ppt_text[:6000]
    )

    gen_svc = GenerationService()
    try:
        reply = gen_svc.generate(
            query=prompt,
            context="",
            system_prompt=_IDEA_CARD_SUMMARY_SYSTEM,
        )
        parsed = safe_json_extract(reply.content)
        parsed["doc_id"] = "new_idea_card_summary"
        return parsed
    except Exception as exc:
        logger.error("Failed to generate idea card summary: %s", exc)
        raise

def build_retrieval_text(summary: Dict[str, Any]) -> str:
    """
    Concatenate summary fields into a single block of text optimized
    for vector embedding and retrieval.
    """
    parts = []
    
    # Core identifying info
    if summary.get("title"):
        parts.append(f"Title: {summary['title']}")
    
    # Semantic content
    if summary.get("short_summary"):
        parts.append(f"Summary: {summary['short_summary']}")
    if summary.get("business_goal"):
        parts.append(f"Goal: {summary['business_goal']}")
        
    # Categorical tags
    if summary.get("actors"):
        parts.append(f"Actors: {', '.join(summary['actors'])}")
    if summary.get("direct_functions"):
        parts.append(f"Functions: {', '.join(summary['direct_functions'])}")
    if summary.get("implied_functions"):
        parts.append(f"Implied: {', '.join(summary['implied_functions'])}")
    if summary.get("domain_tags"):
        parts.append(f"Domains: {', '.join(summary['domain_tags'])}")
    if summary.get("value_stream_labels"):
        parts.append(f"Value Streams: {', '.join(summary['value_stream_labels'])}")
        
    return "\n".join(parts)