from __future__ import annotations


def _simulate_finalize_selection(state: dict) -> dict:
    """
    Replicate node_finalize_selection's source preference logic without importing graph/nodes.py.
    """
    reranked = state.get("taxonomy_reranked_candidates") or []
    if reranked:
        judgments = [
            {
                "entity_name": r["entity_name"],
                "bucket": r.get("bucket", "no_evidence"),
                "confidence": float(r.get("eligibility_score") or r.get("confidence") or 0.0),
            }
            for r in reranked
        ]
        source = "taxonomy_reranked"
    else:
        judgments = state.get("verify_judgments", [])
        source = "verify_judgments"

    return {"source": source, "judgments": judgments}


def test_finalize_prefers_taxonomy_reranked_when_available():
    out = _simulate_finalize_selection(
        {
            "verify_judgments": [
                {"entity_name": "Order to Cash", "bucket": "directly_supported", "confidence": 0.45}
            ],
            "taxonomy_reranked_candidates": [
                {"entity_name": "Order to Cash", "bucket": "directly_supported", "eligibility_score": 0.86}
            ],
        }
    )
    assert out["source"] == "taxonomy_reranked"
    assert out["judgments"][0]["confidence"] == 0.86


def test_finalize_falls_back_to_verify_judgments_when_reranked_absent():
    out = _simulate_finalize_selection(
        {
            "verify_judgments": [
                {"entity_name": "Issue Payment", "bucket": "pattern_inferred", "confidence": 0.52}
            ],
            "taxonomy_reranked_candidates": [],
        }
    )
    assert out["source"] == "verify_judgments"
    assert out["judgments"][0]["entity_name"] == "Issue Payment"
