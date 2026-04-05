"""
Function normalization layer (V5 architecture).

Maps raw LLM-extracted function phrases to a controlled canonical vocabulary.
Used during:
- historical document creation
- new-card understanding
- before capability mapping

The normalizer uses fuzzy token-overlap matching so that phrases like
"vendor eligibility handoff" map to "vendor integration" without requiring
an exact string match.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical function vocabulary
# ---------------------------------------------------------------------------

FUNCTION_VOCAB: List[str] = [
    "product setup",
    "outreach",
    "reporting",
    "vendor integration",
    "billing",
    "payment",
    "onboarding",
    "portal access",
    "request handling",
    "quote/rating",
    "care workflow",
    "compliance",
    "analytics",
    "eligibility",
    "claims processing",
    "enrollment",
    "authorization",
    "grievance/appeals",
    "pharmacy",
    "risk adjustment",
    "credentialing",
    "network management",
    "data migration",
    "system integration",
]

# ---------------------------------------------------------------------------
# Mapping rules: raw phrase tokens -> canonical function
# ---------------------------------------------------------------------------
# Each entry maps a set of trigger tokens to a canonical function.
# If ALL trigger tokens appear in the raw phrase (case-insensitive),
# the canonical function is matched.

_PHRASE_RULES: List[Dict[str, object]] = [
    # vendor / partner
    {"triggers": ["vendor"], "canonical": "vendor integration"},
    {"triggers": ["partner"], "canonical": "vendor integration"},
    {"triggers": ["eligibility", "handoff"], "canonical": "eligibility"},
    {"triggers": ["eligibility"], "canonical": "eligibility"},

    # billing / payment / financial
    {"triggers": ["billing"], "canonical": "billing"},
    {"triggers": ["invoice"], "canonical": "billing"},
    {"triggers": ["payment"], "canonical": "payment"},
    {"triggers": ["remittance"], "canonical": "payment"},
    {"triggers": ["collections"], "canonical": "billing"},
    {"triggers": ["premium"], "canonical": "billing"},
    {"triggers": ["settlement"], "canonical": "payment"},
    {"triggers": ["financial"], "canonical": "payment"},
    {"triggers": ["revenue"], "canonical": "billing"},

    # outreach / engagement
    {"triggers": ["outreach"], "canonical": "outreach"},
    {"triggers": ["engagement"], "canonical": "outreach"},
    {"triggers": ["campaign"], "canonical": "outreach"},
    {"triggers": ["communication"], "canonical": "outreach"},

    # reporting / analytics
    {"triggers": ["reporting"], "canonical": "reporting"},
    {"triggers": ["report"], "canonical": "reporting"},
    {"triggers": ["analytics"], "canonical": "analytics"},
    {"triggers": ["business", "insights"], "canonical": "analytics"},
    {"triggers": ["dashboard"], "canonical": "analytics"},

    # care / clinical
    {"triggers": ["care", "workflow"], "canonical": "care workflow"},
    {"triggers": ["care", "management"], "canonical": "care workflow"},
    {"triggers": ["clinical"], "canonical": "care workflow"},
    {"triggers": ["care", "coordination"], "canonical": "care workflow"},

    # onboarding
    {"triggers": ["onboarding"], "canonical": "onboarding"},
    {"triggers": ["setup"], "canonical": "onboarding"},

    # portal / request
    {"triggers": ["portal"], "canonical": "portal access"},
    {"triggers": ["request", "handling"], "canonical": "request handling"},
    {"triggers": ["inquiry"], "canonical": "request handling"},
    {"triggers": ["self-service"], "canonical": "portal access"},

    # quote / pricing
    {"triggers": ["quote"], "canonical": "quote/rating"},
    {"triggers": ["quoting"], "canonical": "quote/rating"},
    {"triggers": ["pricing"], "canonical": "quote/rating"},
    {"triggers": ["rating"], "canonical": "quote/rating"},
    {"triggers": ["configure", "price"], "canonical": "quote/rating"},

    # compliance / regulatory
    {"triggers": ["compliance"], "canonical": "compliance"},
    {"triggers": ["regulatory"], "canonical": "compliance"},
    {"triggers": ["audit"], "canonical": "compliance"},
    {"triggers": ["privacy"], "canonical": "compliance"},
    {"triggers": ["hipaa"], "canonical": "compliance"},
    {"triggers": ["governance"], "canonical": "compliance"},

    # product
    {"triggers": ["product", "launch"], "canonical": "product setup"},
    {"triggers": ["product", "setup"], "canonical": "product setup"},
    {"triggers": ["product", "offering"], "canonical": "product setup"},
    {"triggers": ["commercialization"], "canonical": "product setup"},

    # claims
    {"triggers": ["claims"], "canonical": "claims processing"},
    {"triggers": ["adjudication"], "canonical": "claims processing"},

    # enrollment
    {"triggers": ["enrollment"], "canonical": "enrollment"},
    {"triggers": ["enrolment"], "canonical": "enrollment"},
    {"triggers": ["member", "registration"], "canonical": "enrollment"},

    # authorization
    {"triggers": ["authorization"], "canonical": "authorization"},
    {"triggers": ["prior", "auth"], "canonical": "authorization"},
    {"triggers": ["referral"], "canonical": "authorization"},

    # grievance / appeals
    {"triggers": ["grievance"], "canonical": "grievance/appeals"},
    {"triggers": ["appeal"], "canonical": "grievance/appeals"},

    # pharmacy
    {"triggers": ["pharmacy"], "canonical": "pharmacy"},
    {"triggers": ["formulary"], "canonical": "pharmacy"},

    # risk adjustment
    {"triggers": ["risk", "adjustment"], "canonical": "risk adjustment"},

    # credentialing
    {"triggers": ["credentialing"], "canonical": "credentialing"},
    {"triggers": ["contracting"], "canonical": "credentialing"},

    # network
    {"triggers": ["network"], "canonical": "network management"},
    {"triggers": ["provider", "network"], "canonical": "network management"},

    # data / system
    {"triggers": ["migration"], "canonical": "data migration"},
    {"triggers": ["data", "migration"], "canonical": "data migration"},
    {"triggers": ["integration"], "canonical": "system integration"},
    {"triggers": ["api"], "canonical": "system integration"},
]


def _tokenize(text: str) -> Set[str]:
    """Lowercase split into word tokens."""
    return set(text.lower().split())


def normalize_function(raw_phrase: str) -> Optional[str]:
    """
    Map a single raw function phrase to the best canonical function.

    Returns the canonical label, or None if no rule matches.
    """
    if not raw_phrase or not raw_phrase.strip():
        return None

    lower = raw_phrase.lower().strip()

    # Exact match against vocab first
    if lower in {v.lower() for v in FUNCTION_VOCAB}:
        for v in FUNCTION_VOCAB:
            if v.lower() == lower:
                return v
        return None

    # Rule-based matching: find the most specific match (most trigger tokens)
    tokens = _tokenize(lower)
    best_match: Optional[str] = None
    best_specificity = 0

    for rule in _PHRASE_RULES:
        triggers = rule["triggers"]
        if all(t in tokens or t in lower for t in triggers):
            specificity = len(triggers)
            if specificity > best_specificity:
                best_specificity = specificity
                best_match = rule["canonical"]

    return best_match


def normalize_functions(raw_phrases: Sequence[str]) -> List[str]:
    """
    Normalize a list of raw function phrases to canonical vocabulary.

    Deduplicates and preserves order of first occurrence.
    """
    seen: Set[str] = set()
    result: List[str] = []

    for phrase in raw_phrases:
        canonical = normalize_function(phrase)
        if canonical and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)

    return result
