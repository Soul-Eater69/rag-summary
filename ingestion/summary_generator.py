"""
Summary generator (V5 architecture): Uses LLM to create structured semantic
summaries of historical tickets or new idea cards.

V5 changes:
- Extracts raw direct/implied functions separately
- Normalizes to canonical vocabulary via function_normalizer
- Richer retrieval text including capability tags and operational footprint
- Populates V5 schema fields
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.services.generation_service import GenerationService
from core.prompts import safe_json_extract
from .schema import SummaryDoc
from .function_normalizer import normalize_functions

logger = logging.getLogger(__name__)

_HISTORICAL_SUMMARY_SYSTEM = """You are an expert at summarizing healthcare enterprise project tickets.

Your goal is to create a dense, semantic summary that captures:
1. What the project actually did (the "what")
2. Who it affected (actors)
3. The business intent (the "why")
4. Functional domains (direct and implied)
5. Operational footprint and capability signals

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
- direct_functions_raw: Explicit capabilities mentioned in the text, using the exact phrasing found.
  Examples: "vendor eligibility handoff", "outreach campaign support", "payment execution"
- implied_functions_raw: Capabilities logically required but not explicitly named.
  Examples: "billing adjustment", "portal inquiry handling", "onboarding"
- change_types: e.g., New Capability, Regulatory Compliance, Process Improvement, Migration.
- domain_tags: e.g., Clinical, Financial, IT, Operational.
- capability_tags: Business capability families. Choose from:
  product_launch_commercialization, billing_order_to_cash, vendor_partner_onboarding,
  portal_request_resolution, compliance_privacy_audit, analytics_reporting,
  care_management_clinical, claims_adjudication, enrollment_quoting,
  member_engagement_outreach, provider_network, enterprise_risk_governance,
  leads_opportunities
- operational_footprint: Concrete operational activities. Examples:
  vendor integration, outreach, reporting assets, payment execution,
  billing adjustment, portal handling, quote/rating, eligibility handoff
- evidence_sentences: 2-3 key sentences from the text that justify the summary.

Return valid JSON ONLY.
"""

_IDEA_CARD_SUMMARY_SYSTEM = """You are a healthcare business analyst.
Your goal is to extract a structured semantic summary from an "Idea Card" PPT.
Focus on business intent, functional impact, affected stakeholders, and
downstream operational implications.
"""

_IDEA_CARD_SUMMARY_USER = """
## IDEA CARD CONTENT
{ppt_text}

## INSTRUCTIONS
Extract a structured summary.
- short_summary: 1-2 sentence high-level summary.
- business_goal: The core problem or opportunity.
- actors: Affected stakeholders.
- direct_functions_raw: Explicitly mentioned capabilities, using exact phrasing.
- implied_functions_raw: Logically required capabilities not explicitly stated.
- change_types: Type of change (e.g., New Product, Efficiency, Regulatory).
- domain_tags: Relevant healthcare domains.
- capability_tags: Business capability families relevant to this card. Choose from:
  product_launch_commercialization, billing_order_to_cash, vendor_partner_onboarding,
  portal_request_resolution, compliance_privacy_audit, analytics_reporting,
  care_management_clinical, claims_adjudication, enrollment_quoting,
  member_engagement_outreach, provider_network, enterprise_risk_governance,
  leads_opportunities
- operational_footprint: Concrete operational activities implied by this card.

Return valid JSON ONLY.
"""


def _enrich_with_canonical(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Add canonical function fields by normalizing raw functions."""
    raw_direct = parsed.get("direct_functions_raw") or parsed.get("direct_functions") or []
    raw_implied = parsed.get("implied_functions_raw") or parsed.get("implied_functions") or []

    parsed["direct_functions_raw"] = raw_direct
    parsed["implied_functions_raw"] = raw_implied
    parsed["direct_functions_canonical"] = normalize_functions(raw_direct)
    parsed["implied_functions_canonical"] = normalize_functions(raw_implied)

    # Legacy compat: populate flat fields from canonical
    parsed["direct_functions"] = parsed["direct_functions_canonical"]
    parsed["implied_functions"] = parsed["implied_functions_canonical"]

    # Ensure V5 fields have defaults
    parsed.setdefault("capability_tags", [])
    parsed.setdefault("operational_footprint", [])
    parsed.setdefault("evidence_sentences", [])

    return parsed


def generate_ticket_summary(
    ticket_text: str,
    ticket_id: str,
    title: str,
    value_stream_labels: List[str],
    model: str = "gpt-5-mini-idp",
) -> SummaryDoc:
    """Use LLM to generate a structured summary of a historical ticket."""
    prompt = _HISTORICAL_SUMMARY_USER.format(
        ticket_id=ticket_id,
        title=title,
        vs_labels=", ".join(value_stream_labels),
        ticket_text=ticket_text[:6000],
    )

    gen_svc = GenerationService()
    try:
        reply = gen_svc.generate(
            query=prompt,
            context="",
            system_prompt=_HISTORICAL_SUMMARY_SYSTEM,
        )
        parsed = safe_json_extract(reply.content)

        # V5: normalize functions
        _enrich_with_canonical(parsed)

        # Add metadata
        parsed["ticket_id"] = ticket_id
        parsed["title"] = title
        parsed["value_stream_labels"] = value_stream_labels
        parsed["doc_id"] = f"summary_{ticket_id}"

        return parsed
    except Exception as exc:
        logger.error("Failed to generate summary for %s: %s", ticket_id, exc)
        raise


def generate_new_card_summary(
    ppt_text: str,
    model: str = "gpt-5-mini-idp",
) -> SummaryDoc:
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

        # V5: normalize functions
        _enrich_with_canonical(parsed)

        parsed["doc_id"] = "new_idea_card_summary"
        return parsed
    except Exception as exc:
        logger.error("Failed to generate idea card summary: %s", exc)
        raise


def build_retrieval_text(summary: SummaryDoc) -> str:
    """
    Concatenate summary fields into a single block of text optimized
    for vector embedding and retrieval.

    V5: includes capability tags, operational footprint, and both
    raw and canonical function representations.
    """
    parts = []

    # Core identifying info
    if summary.get("title"):
        parts.append(f"Title: {summary['title']}")

    # Semantic content
    if summary.get("short_summary"):
        parts.append(f"Summary: {summary['short_summary']}")
    if summary.get("business_goal"):
        parts.append(f"Business goal: {summary['business_goal']}")

    # Actors
    if summary.get("actors"):
        parts.append(f"Actors: {', '.join(summary['actors'])}")

    # Functions: prefer canonical, include raw for richer embedding
    direct_canon = summary.get("direct_functions_canonical") or summary.get("direct_functions") or []
    if direct_canon:
        parts.append(f"Direct functions: {', '.join(direct_canon)}")

    implied_canon = summary.get("implied_functions_canonical") or summary.get("implied_functions") or []
    if implied_canon:
        parts.append(f"Implied functions: {', '.join(implied_canon)}")

    # V5: capability tags
    if summary.get("capability_tags"):
        parts.append(f"Capabilities: {', '.join(summary['capability_tags'])}")

    # V5: operational footprint
    if summary.get("operational_footprint"):
        parts.append(f"Operational footprint: {', '.join(summary['operational_footprint'])}")

    # Domain tags
    if summary.get("domain_tags"):
        parts.append(f"Domains: {', '.join(summary['domain_tags'])}")

    # Value streams (historical tickets only)
    if summary.get("value_stream_labels"):
        parts.append(f"Value streams: {', '.join(summary['value_stream_labels'])}")

    return "\n".join(parts)
