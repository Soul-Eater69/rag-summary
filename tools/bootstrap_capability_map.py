"""
Bootstrap capability map from Azure Value Stream index (V5 architecture).

This is an OFFLINE / BUILD-TIME tool, not a runtime component.

Flow:
1. Fetch all canonical VS entries from Azure AI Search index
2. Normalize and export as VS corpus
3. Draft capability clusters from VS semantics
4. Enrich with historical ticket patterns (if available)
5. Generate coverage report
6. Write capability_map.yaml

Usage:
    python -m rag_summary.tools.bootstrap_capability_map \
      --output-dir data \
      --config-path config/capability_map.yaml

    # Dry run (no writes):
    python -m rag_summary.tools.bootstrap_capability_map --dry-run

    # With historical enrichment:
    python -m rag_summary.tools.bootstrap_capability_map \
      --ticket-chunks-dir ticket_chunks \
      --index-dir local_ticket_summary_faiss
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)

# Default paths relative to repo root
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data"
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "capability_map.yaml"

# Predefined cluster templates based on common VS groupings.
# These are used as seeds when bootstrapping from Azure index.
_CLUSTER_TEMPLATES = {
    "compliance_privacy_audit": {
        "description": "Privacy, auditability, regulatory obligations, consent, controls, and governance requirements.",
        "seed_streams": ["Ensure Compliance"],
        "related_patterns": ["Manage Enterprise Risk"],
        "cue_families": ["privacy", "audit", "compliance", " ", "hipaa", "consent", "pii", "controls"],
    },
    "enterprise_risk_governance": {
        "description": "Enterprise risk posture, governance processes, and risk controls.",
        "seed_streams": ["Manage Enterprise Risk"],
        "related_patterns": ["Ensure Compliance"],
        "cue_families": ["risk", "governance", "oversight", "policy", "control framework"],
    },
    "vendor_partner_onboarding": {
        "description": "Onboarding, partner/vendor integration, setup, contracting, and external ecosystem participation.",
        "seed_streams": ["Onboard Partner", "Establish Provider Program"],
        "related_patterns": ["Establish Provider Network"],
        "cue_families": ["vendor onboarding", "partner onboarding", "contracting", "vendor integration", "eligibility handoff"],
    },
    "billing_order_to_cash": {
        "description": "Billing, invoicing, payment, remittance, financial operations, and commercialization transaction flows.",
        "seed_streams": ["Order to Cash for Group Coverage", "Manage Invoice and Payment Receipt", "Issue Payment"],
        "related_patterns": ["Configure Price and Quote"],
        "cue_families": ["billing", "invoice", "payment", "remittance", "collections", "premium"],
    },
    "enrollment_quoting": {
        "description": "Quoting, pricing configuration, enrollment, and plan selection flows.",
        "seed_streams": ["Sell and Enroll", "Configure Price and Quote"],
        "related_patterns": ["Manage Leads and Opportunities"],
        "cue_families": ["enrollment", "quoting", "pricing", "rate card", "configure price"],
    },
    "product_launch_commercialization": {
        "description": "New product launch, product setup, commercialization, and go-to-market capability rollouts.",
        "seed_streams": ["Establish Product Offering"],
        "related_patterns": ["Manage Leads and Opportunities", "Sell and Enroll"],
        "cue_families": ["product launch", "commercialization", "product offering", "go-to-market"],
    },
    "member_engagement_outreach": {
        "description": "Member outreach, engagement programs, and communication campaigns.",
        "seed_streams": ["Perform Engagement"],
        "related_patterns": ["Manage Member Care"],
        "cue_families": ["outreach", "engagement", "campaign", "member communication"],
    },
    "care_management_clinical": {
        "description": "Care management workflows, clinical programs, and health services coordination.",
        "seed_streams": ["Manage Member Care"],
        "related_patterns": ["Perform Engagement"],
        "cue_families": ["care management", "clinical", "utilization management", "care coordination"],
    },
    "claims_adjudication": {
        "description": "Claims processing, adjudication, and payment determination.",
        "seed_streams": ["Process Claims"],
        "related_patterns": ["Issue Payment", "Manage Invoice and Payment Receipt"],
        "cue_families": ["claims processing", "adjudication", "claim submission"],
    },
    "portal_request_resolution": {
        "description": "Portal access, self-service, request handling, and inquiry resolution.",
        "seed_streams": ["Resolve Request Inquiry"],
        "related_patterns": [],
        "cue_families": ["portal", "self-service", "request handling", "inquiry"],
    },
    "analytics_reporting": {
        "description": "Business analytics, reporting, data insights, and decision support.",
        "seed_streams": ["Discover Business Insights"],
        "related_patterns": [],
        "cue_families": ["analytics", "reporting", "business insights", "dashboard"],
    },
    "provider_network": {
        "description": "Provider network establishment, adequacy, credentialing, and directory management.",
        "seed_streams": ["Establish Provider Network"],
        "related_patterns": ["Establish Provider Program", "Onboard Partner"],
        "cue_families": ["provider network", "credentialing", "network adequacy", "provider directory"],
    },
    "leads_opportunities": {
        "description": "Sales pipeline, lead management, and opportunity tracking.",
        "seed_streams": ["Manage Leads and Opportunities"],
        "related_patterns": ["Sell and Enroll", "Configure Price and Quote"],
        "cue_families": ["leads", "opportunities", "sales pipeline", "rfp", "prospect"],
    },
}

def fetch_azure_vs_corpus() -> List[Dict[str, Any]]:
    """
    Fetch canonical VS entries from Azure AI Search index.

    Returns list of dicts with: entity_id, entity_name, description,
    value_proposition, aliases, category.

    NOTE: This function requires the Azure retrieval pipeline to be
    available. If not available, returns an empty list and logs a warning.
    """
    try:
        from src.pipelines.value_stream.retrieval_pipeline import (
            retrieve_kg_candidates,
        )

        # Broad query to get all VS entries
        candidates = retrieve_kg_candidates("", top_k=100)
        corpus = []
        seen: Set[str] = set()
        for c in candidates:
            name = (c.get("entity_name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            corpus.append({
                "entity_id": c.get("entity_id", ""),
                "entity_name": name,
                "description": c.get("description", ""),
                "value_proposition": c.get("value_proposition", ""),
                "aliases": c.get("aliases", []),
                "category": c.get("category", ""),
            })

        logger.info("Fetched %d VS entries from Azure index", len(corpus))
        return corpus

    except Exception as exc:
        logger.warning("Could not fetch Azure VS corpus: %s", exc)
        return []

def build_coverage_report(
    corpus: List[Dict[str, Any]],
    capability_map: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Check that every VS from the corpus appears in at least one
    promote_value_streams or related_value_streams.
    """
    all_vs_names = {entry["entity_name"] for entry in corpus}

    covered_promote: Set[str] = set()
    covered_related: Set[str] = set()

    for cluster_name, cluster in capability_map.get("capabilities", {}).items():
        for vs in cluster.get("promote_value_streams", []):
            covered_promote.add(vs)
        for vs in cluster.get("related_value_streams", []):
            covered_related.add(vs)

    covered = covered_promote | covered_related
    uncovered = all_vs_names - covered

    return {
        "total_vs_in_corpus": len(all_vs_names),
        "covered_by_promote": len(covered_promote),
        "covered_by_related": len(covered_related),
        "total_covered": len(covered),
        "uncovered_count": len(uncovered),
        "uncovered_streams": sorted(uncovered),
        "coverage_pct": round(len(covered) / max(1, len(all_vs_names)) * 100, 1),
    }

def draft_capability_map_from_templates(
    corpus: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Draft a capability_map.yaml from cluster templates.

    If a corpus is provided, enrich cues from VS descriptions.
    """
    capabilities: Dict[str, Any] = {}

    for cluster_name, template in _CLUSTER_TEMPLATES.items():
        direct_cues = list(template["cue_families"])
        indirect_cues: List[str] = []

        # Enrich from corpus descriptions if available
        if corpus:
            for vs_name in template["seed_streams"] + template.get("related_patterns", []):
                for entry in corpus:
                    if entry["entity_name"] == vs_name:
                        desc = (entry.get("description") or "").lower()
                        # Extract 2-3 word phrases as indirect cues
                        words = desc.split()
                        for i in range(len(words) - 1):
                            bigram = f"{words[i]} {words[i+1]}"
                            if len(bigram) > 5 and bigram not in direct_cues:
                                indirect_cues.append(bigram)

        # Deduplicate indirect cues and limit
        seen = set(direct_cues)
        unique_indirect = []
        for cue in indirect_cues[:10]:
            if cue not in seen:
                seen.add(cue)
                unique_indirect.append(cue)

        capabilities[cluster_name] = {
            "description": template["description"],
            "direct_cues": direct_cues,
            "indirect_cues": unique_indirect,
            "canonical_functions": [],
            "promote_value_streams": template["seed_streams"],
            "related_value_streams": template.get("related_patterns", []),
            "weight": 0.9,
            "min_signal_strength": 0.5,
        }

    return {"version": 1, "capabilities": capabilities}

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Bootstrap capability map from Azure VS index"
    )

    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Directory for output artifacts",
    )
    parser.add_argument(
        "--config-path",
        default=str(_DEFAULT_CONFIG_PATH),
        help="Path to write capability_map.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print draft map without writing files",
    )
    parser.add_argument(
        "--skip-azure",
        action="store_true",
        help="Skip Azure fetch and use templates only",
    )

    args = parser.parse_args()

    # Step 1: Fetch VS corpus
    corpus: List[Dict[str, Any]] = []
    if not args.skip_azure:
        corpus = fetch_azure_vs_corpus()
        if not corpus:
            logger.warning("No VS corpus fetched; proceeding with templates only")

    # Step 2: Draft capability map
    draft_map = draft_capability_map_from_templates(corpus or None)

    # Step 3: Coverage report
    if corpus:
        report = build_coverage_report(corpus, draft_map)
        logger.info("Coverage: %s", json.dumps(report, indent=2))
    else:
        report = {"note": "No corpus available; coverage not computed"}

    if args.dry_run:
        print("--- Draft Capability Map ---")
        print(yaml.dump(draft_map, default_flow_style=False, sort_keys=False))
        if corpus:
            print("--- Coverage Report ---")
            print(json.dumps(report, indent=2))
        return

    # Step 4: Write outputs
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if corpus:
        corpus_path = output_dir / "value_stream_corpus.json"
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        logger.info("Wrote VS corpus to %s", corpus_path)

    report_path = output_dir / "capability_map_coverage_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Wrote coverage report to %s", report_path)

    config_path = pathlib.Path(args.config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(draft_map, f, default_flow_style=False, sort_keys=False)
    logger.info("Wrote capability map to %s", config_path)

if __name__ == "__main__":
    main()
