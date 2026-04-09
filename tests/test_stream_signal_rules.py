"""
Tests for stream signal rules (Phase 2).

Covers:
  - signal_rules YAML loads and has expected families
  - _compute_signal_adjustment returns boost when positive signals match
  - _compute_signal_adjustment returns penalty when only weak signals match
  - _compute_signal_adjustment returns 0 when neither
  - min_positive_required enforcement
  - extract_chunk_candidates applies boosts for CPQ text
  - extract_chunk_candidates applies penalty for bare weak terms
  - extract_card_attachment_candidates uses attachment_boost_signals
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

# Ensure rag_summary is importable (symlink or path adjustment)
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIGNAL_RULES_PATH = _REPO_ROOT / "config" / "stream_signal_rules.yaml"
_CAPABILITY_MAP_PATH = _REPO_ROOT / "config" / "capability_map.yaml"


def _load_signal_rules():
    with _SIGNAL_RULES_PATH.open("r") as f:
        payload = yaml.safe_load(f)
    return payload.get("families", {})


# ---------------------------------------------------------------------------
# Test: YAML structure
# ---------------------------------------------------------------------------

class TestSignalRulesYAML:
    def test_file_exists(self):
        assert _SIGNAL_RULES_PATH.exists(), "stream_signal_rules.yaml must exist"

    def test_top_level_keys(self):
        with _SIGNAL_RULES_PATH.open("r") as f:
            payload = yaml.safe_load(f)
        assert "version" in payload
        assert "families" in payload
        assert isinstance(payload["families"], dict)

    def test_expected_families_present(self):
        rules = _load_signal_rules()
        for family in [
            "product_and_pricing",
            "member_services",
            "finance",
            "enrollment",
            "provider_network",
            "compliance_and_risk",
            "partner_management",
            "strategy",
            "care_management",
            "claims",
            "utilization",
            "pharmacy",
            "quality",
        ]:
            assert family in rules, f"Family '{family}' missing from signal rules"

    def test_each_family_has_required_fields(self):
        rules = _load_signal_rules()
        required_fields = {
            "positive_signals",
            "weak_only_not_enough",
            "attachment_boost_signals",
            "historical_promotion_hints",
            "min_positive_required",
            "score_boost",
            "score_penalty",
        }
        for family, rule in rules.items():
            missing = required_fields - set(rule.keys())
            assert not missing, f"Family '{family}' missing fields: {missing}"

    def test_score_boost_in_range(self):
        rules = _load_signal_rules()
        for family, rule in rules.items():
            boost = float(rule["score_boost"])
            assert 0.0 <= boost <= 1.0, f"Family '{family}' score_boost out of range: {boost}"

    def test_score_penalty_in_range(self):
        rules = _load_signal_rules()
        for family, rule in rules.items():
            penalty = float(rule["score_penalty"])
            assert 0.0 <= penalty <= 1.0, f"Family '{family}' score_penalty out of range: {penalty}"

    def test_min_positive_required_positive_int(self):
        rules = _load_signal_rules()
        for family, rule in rules.items():
            val = int(rule["min_positive_required"])
            assert val >= 1, f"Family '{family}' min_positive_required must be >= 1"

    def test_positive_signals_are_lists(self):
        rules = _load_signal_rules()
        for family, rule in rules.items():
            assert isinstance(rule["positive_signals"], list), \
                f"Family '{family}' positive_signals must be a list"

    def test_weak_only_not_enough_are_lists(self):
        rules = _load_signal_rules()
        for family, rule in rules.items():
            assert isinstance(rule["weak_only_not_enough"], list), \
                f"Family '{family}' weak_only_not_enough must be a list"


# ---------------------------------------------------------------------------
# Test: _compute_signal_adjustment logic
# ---------------------------------------------------------------------------

class TestComputeSignalAdjustment:
    """Import and test the internal helper directly."""

    @pytest.fixture(autouse=True)
    def _import_helper(self):
        # Dynamically import to avoid module-level import issues
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "card_candidates",
            str(_REPO_ROOT / "generation" / "card_candidates.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._fn = mod._compute_signal_adjustment
        self._load = mod._load_signal_rules

    def test_boost_when_positive_signal_matches(self):
        rules = self._load()
        # "cpq" is a positive signal for product_and_pricing
        adj = self._fn("this card is about cpq implementation", "product_and_pricing", rules)
        assert adj > 0.0, "Should return positive boost for CPQ text"

    def test_no_boost_for_unknown_family(self):
        rules = self._load()
        adj = self._fn("cpq pricing quote", "nonexistent_family", rules)
        assert adj == 0.0

    def test_penalty_for_weak_only_terms(self):
        rules = self._load()
        # "quote" alone is weak-only for product_and_pricing; no positive signals
        adj = self._fn("we need a better quote system", "product_and_pricing", rules)
        assert adj < 0.0, "Should return penalty when only weak signals present"

    def test_zero_when_neither(self):
        rules = self._load()
        adj = self._fn("this is about weather forecasting", "product_and_pricing", rules)
        assert adj == 0.0

    def test_boost_overrides_penalty(self):
        """Positive signals + weak signals → boost wins."""
        rules = self._load()
        # Both "cpq" (positive) and "quote" (weak) present
        adj = self._fn("cpq and quote system", "product_and_pricing", rules)
        assert adj > 0.0

    def test_strategy_requires_two_positive_signals(self):
        """Strategy family has min_positive_required=2."""
        rules = self._load()
        # One positive signal — should not boost (need 2)
        adj_one = self._fn("strategic plan", "strategy", rules)
        # Two positive signals
        adj_two = self._fn("strategic plan and corporate strategy", "strategy", rules)
        assert adj_one <= 0.0, "One positive should not meet min_positive_required=2 for strategy"
        assert adj_two > 0.0, "Two positives should boost strategy family"

    def test_finance_positive_boost(self):
        rules = self._load()
        adj = self._fn("invoice management and billing reconciliation", "finance", rules)
        assert adj > 0.0

    def test_finance_payment_alone_is_penalty(self):
        rules = self._load()
        adj = self._fn("process the payment", "finance", rules)
        assert adj < 0.0, "'payment' alone should be penalized for finance family"

    def test_member_services_positive_boost(self):
        rules = self._load()
        adj = self._fn("member inquiry resolution at the contact center", "member_services", rules)
        assert adj > 0.0

    def test_pharmacy_positive_boost(self):
        rules = self._load()
        adj = self._fn("pharmacy benefit management with formulary", "pharmacy", rules)
        assert adj > 0.0

    def test_provider_network_credentialing_boost(self):
        rules = self._load()
        adj = self._fn("provider credentialing workflow for new clinicians", "provider_network", rules)
        assert adj > 0.0


# ---------------------------------------------------------------------------
# Test: extract_chunk_candidates uses signal rules
# ---------------------------------------------------------------------------

class TestChunkCandidatesSignalRules:
    @pytest.fixture(autouse=True)
    def _import_module(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "card_candidates",
            str(_REPO_ROOT / "generation" / "card_candidates.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._extract_chunk = mod.extract_chunk_candidates
        self._cluster_map = mod._CLUSTER_FAMILY_MAP

    def test_cpq_text_produces_cpq_candidate(self):
        text = (
            "This initiative requires configure, price, and quote (CPQ) capability "
            "to support quoting engine and pricing rules for all product lines."
        )
        candidates = self._extract_chunk(text)
        names = [c["entity_name"] for c in candidates]
        assert "Configure, Price, and Quote" in names, (
            f"CPQ text should produce Configure,Price,Quote candidate. Got: {names}"
        )

    def test_cpq_candidate_has_reasonable_score(self):
        """CPQ-rich text should produce a CPQ candidate with a meaningful score."""
        text = (
            "This initiative requires configure, price, and quote (CPQ) capability "
            "to support quoting engine and pricing rules for all product lines."
        )
        candidates = self._extract_chunk(text)
        cpq_cands = [c for c in candidates if c["entity_name"] == "Configure, Price, and Quote"]
        assert cpq_cands, "CPQ text must produce a Configure,Price,Quote candidate"
        assert cpq_cands[0]["score"] >= 0.30, (
            f"CPQ candidate score too low: {cpq_cands[0]['score']}"
        )

    def test_bare_payment_alone_does_not_produce_finance_candidate(self):
        """'payment' alone is weak-only for finance; should be penalized or dropped."""
        text = "we need to improve the payment process"
        candidates = self._extract_chunk(text)
        # If any finance candidates exist, they should have reduced scores
        finance_cands = [
            c for c in candidates
            if c["entity_name"] in ("Order to Cash", "Manage Invoice and Payment Receipt", "Issue Payment")
        ]
        for c in finance_cands:
            assert c["score"] < 0.50, (
                f"Finance candidate from bare 'payment' should have low score. Got: {c['score']}"
            )

    def test_invoice_text_produces_finance_candidates(self):
        text = (
            "This project addresses invoice management and billing reconciliation "
            "for group coverage premium billing cycles."
        )
        candidates = self._extract_chunk(text)
        names = [c["entity_name"] for c in candidates]
        finance_streams = {"Order to Cash", "Manage Invoice and Payment Receipt", "Issue Payment"}
        found = finance_streams & set(names)
        assert found, f"Invoice text should produce finance candidates. Got: {names}"

    def test_cluster_family_map_covers_common_clusters(self):
        """All major clusters should have a family entry."""
        expected_clusters = [
            "billing_order_to_cash",
            "enrollment_quoting",
            "portal_request_resolution",
            "claims_adjudication",
            "provider_network",
            "compliance_privacy_audit",
            "pharmacy_prescriptions",
        ]
        for cluster in expected_clusters:
            assert cluster in self._cluster_map, \
                f"Cluster '{cluster}' missing from _CLUSTER_FAMILY_MAP"

    def test_returns_list(self):
        candidates = self._extract_chunk("some random text about claims processing")
        assert isinstance(candidates, list)

    def test_empty_text_returns_empty(self):
        candidates = self._extract_chunk("")
        assert candidates == []

    def test_allowed_names_filters_candidates(self):
        text = (
            "This initiative requires configure, price, and quote (CPQ) capability "
            "to support quoting engine and pricing rules. Also involves billing "
            "and invoice management."
        )
        allowed = {"Configure, Price, and Quote"}
        candidates = self._extract_chunk(text, allowed_names=allowed)
        names = set(c["entity_name"] for c in candidates)
        assert names <= allowed, f"Candidates outside allowed set: {names - allowed}"


# ---------------------------------------------------------------------------
# Test: taxonomy_registry_coverage — verify all capability_map VS names
#       are in the taxonomy registry
# ---------------------------------------------------------------------------

class TestTaxonomyRegistryCoverage:
    @pytest.fixture(autouse=True)
    def _load_data(self):
        with _CAPABILITY_MAP_PATH.open("r") as f:
            cap_payload = yaml.safe_load(f)
        self._capabilities = cap_payload.get("capabilities", {})

        with (pathlib.Path(__file__).resolve().parent.parent / "config" / "taxonomy_registry.yaml").open("r") as f:
            tax_payload = yaml.safe_load(f)
        self._taxonomy_names = {
            stream["canonical_name"]
            for stream in tax_payload.get("streams", [])
        }

    def test_all_promoted_vs_names_in_taxonomy(self):
        """
        Every promote_value_streams entry in capability_map.yaml should match
        a canonical_name in taxonomy_registry.yaml.
        """
        missing = []
        for cluster_name, cluster in self._capabilities.items():
            for vs_name in cluster.get("promote_value_streams") or []:
                if vs_name not in self._taxonomy_names:
                    missing.append((cluster_name, vs_name))
        assert not missing, (
            "capability_map promote_value_streams entries not in taxonomy registry:\n"
            + "\n".join(f"  cluster={c}, vs={v}" for c, v in missing)
        )

    def test_taxonomy_has_49_streams(self):
        """Registry should have all 49 streams (25 original + 24 added in Phase 2)."""
        assert len(self._taxonomy_names) == 49, (
            f"Expected 49 taxonomy streams, found {len(self._taxonomy_names)}"
        )

    def test_key_under_recovered_streams_in_taxonomy(self):
        """Spot-check the streams identified as under-recovered in Phase 2."""
        for name in [
            "Configure, Price, and Quote",
            "Resolve Request Inquiry",
            "Order to Cash",
            "Manage Invoice and Payment Receipt",
            "Issue Payment",
            "Onboard Partner",
            "Manage Pharmacy Benefit",
            "Manage Quality Program",
            "Manage Care Coordination",
            "Manage Population Health",
            "Credential Provider",
            "Contract Provider",
            "Manage Fraud, Waste, and Abuse",
            "Manage COB",
            "Manage Subrogation",
        ]:
            assert name in self._taxonomy_names, f"'{name}' missing from taxonomy registry"
