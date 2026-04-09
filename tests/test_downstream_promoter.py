"""
Tests for downstream candidate promotion (Phase 3).

Covers:
  - Rules YAML loads and validates
  - _load_promotion_rules, _build_historical_score_map, _build_bundle_score_map
  - promote_downstream_candidates core promotion logic
  - historical_score_min threshold enforcement
  - historical_bundle_min threshold enforcement
  - downstream_chain_trigger enforcement
  - max_score cap
  - skip when target already well-covered
  - allowed_names filter
  - score max-pooling across rules
  - node_promote_downstream_candidates integration
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))

_RULES_PATH = _REPO_ROOT / "config" / "downstream_promotion_rules.yaml"

# ---------------------------------------------------------------------------
# Import module under test
# ---------------------------------------------------------------------------

import importlib.util as _ilu


def _load_module(name: str, path: pathlib.Path):
    spec = _ilu.spec_from_file_location(name, str(path))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_promoter = _load_module("downstream_promoter", _REPO_ROOT / "generation" / "downstream_promoter.py")

promote_downstream_candidates = _promoter.promote_downstream_candidates
_load_promotion_rules = _promoter._load_promotion_rules
_build_historical_score_map = _promoter._build_historical_score_map
_build_bundle_score_map = _promoter._build_bundle_score_map
_build_downstream_set = _promoter._build_downstream_set


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _hist(name: str, score: float) -> dict:
    return {"entity_name": name, "score": score}


def _bundle(primary: str, bundled: str, fraction: float) -> dict:
    return {"primary_vs": primary, "bundled_vs": bundled, "co_occurrence_fraction": fraction}


def _chain(upstream: str, downstream: str, count: int = 3) -> dict:
    return {"upstream_vs": upstream, "downstream_vs": downstream, "analog_count": count}


# ---------------------------------------------------------------------------
# YAML structure tests
# ---------------------------------------------------------------------------

class TestRulesYAML:
    def test_file_exists(self):
        assert _RULES_PATH.exists()

    def test_has_version_and_rules(self):
        with _RULES_PATH.open() as f:
            payload = yaml.safe_load(f)
        assert "version" in payload
        assert "rules" in payload
        assert isinstance(payload["rules"], list)
        assert len(payload["rules"]) > 0

    def test_each_rule_has_required_fields(self):
        rules = _load_promotion_rules()
        required = {"name", "requires_any", "promotes", "max_score"}
        for rule in rules:
            missing = required - set(rule.keys())
            assert not missing, f"Rule '{rule.get('name')}' missing fields: {missing}"

    def test_max_score_in_range(self):
        rules = _load_promotion_rules()
        for rule in rules:
            ms = float(rule["max_score"])
            assert 0.0 < ms <= 1.0, f"Rule '{rule['name']}' max_score out of range: {ms}"

    def test_requires_any_is_list(self):
        rules = _load_promotion_rules()
        for rule in rules:
            assert isinstance(rule["requires_any"], list), \
                f"Rule '{rule['name']}' requires_any must be list"

    def test_promotes_is_list(self):
        rules = _load_promotion_rules()
        for rule in rules:
            assert isinstance(rule["promotes"], list), \
                f"Rule '{rule['name']}' promotes must be list"

    def test_key_rules_present(self):
        rules = _load_promotion_rules()
        names = {r["name"] for r in rules}
        for expected in [
            "o2c_to_issue_payment",
            "invoice_to_issue_payment",
            "provider_network_to_credentialing",
            "enrollment_to_eligibility",
            "claims_to_cob",
            "population_health_to_care_coordination",
        ]:
            assert expected in names, f"Rule '{expected}' missing from rules YAML"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_historical_score_map_basic(self):
        entries = [
            _hist("Order to Cash", 0.72),
            _hist("Issue Payment", 0.45),
        ]
        score_map = _build_historical_score_map(entries)
        assert score_map["order to cash"] == pytest.approx(0.72)
        assert score_map["issue payment"] == pytest.approx(0.45)

    def test_historical_score_map_max_pooling(self):
        entries = [_hist("Order to Cash", 0.50), _hist("Order to Cash", 0.80)]
        score_map = _build_historical_score_map(entries)
        assert score_map["order to cash"] == pytest.approx(0.80)

    def test_historical_score_map_empty(self):
        assert _build_historical_score_map([]) == {}

    def test_bundle_score_map_primary_and_bundled(self):
        bundles = [_bundle("Order to Cash", "Issue Payment", 0.70)]
        bmap = _build_bundle_score_map(bundles)
        assert bmap["order to cash"] == pytest.approx(0.70)
        assert bmap["issue payment"] == pytest.approx(0.70)

    def test_downstream_set_contains_upstream_names(self):
        chains = [_chain("Order to Cash", "Issue Payment")]
        upstream_set = _build_downstream_set(chains)
        assert "order to cash" in upstream_set

    def test_downstream_set_empty(self):
        assert _build_downstream_set([]) == set()


# ---------------------------------------------------------------------------
# Core promotion logic tests
# ---------------------------------------------------------------------------

class TestPromoteDownstreamCandidates:

    def test_o2c_historical_triggers_issue_payment(self):
        """High historical O2C score should promote Issue Payment."""
        candidates = promote_downstream_candidates(
            historical_value_stream_support=[_hist("Order to Cash", 0.75)],
            bundle_patterns=[],
            downstream_chains=[],
        )
        names = [c["entity_name"] for c in candidates]
        assert "Issue Payment" in names, f"Expected Issue Payment in {names}"

    def test_o2c_historical_triggers_manage_invoice(self):
        """High historical O2C should also promote Manage Invoice and Payment Receipt."""
        candidates = promote_downstream_candidates(
            historical_value_stream_support=[_hist("Order to Cash", 0.75)],
            bundle_patterns=[],
            downstream_chains=[],
        )
        names = [c["entity_name"] for c in candidates]
        assert "Manage Invoice and Payment Receipt" in names

    def test_below_threshold_no_promotion(self):
        """Low historical score should not trigger promotion."""
        candidates = promote_downstream_candidates(
            historical_value_stream_support=[_hist("Order to Cash", 0.10)],
            bundle_patterns=[],
            downstream_chains=[],
        )
        names = [c["entity_name"] for c in candidates]
        assert "Issue Payment" not in names

    def test_bundle_pattern_triggers_promotion(self):
        """High bundle fraction satisfies rule even without historical score."""
        candidates = promote_downstream_candidates(
            historical_value_stream_support=[],
            bundle_patterns=[_bundle("Order to Cash", "Other VS", 0.65)],
            downstream_chains=[],
        )
        # Order to Cash has bundle_fraction=0.65 >= historical_bundle_min=0.50
        names = [c["entity_name"] for c in candidates]
        assert "Issue Payment" in names

    def test_downstream_chain_triggers_promotion(self):
        """A downstream chain entry for a requires_any stream triggers promotion."""
        candidates = promote_downstream_candidates(
            historical_value_stream_support=[_hist("Order to Cash", 0.45)],
            bundle_patterns=[],
            downstream_chains=[_chain("Order to Cash", "Some Downstream VS")],
        )
        names = [c["entity_name"] for c in candidates]
        assert "Issue Payment" in names

    def test_score_below_max_score_cap(self):
        """Promoted score must not exceed the rule's max_score."""
        candidates = promote_downstream_candidates(
            historical_value_stream_support=[_hist("Order to Cash", 0.99)],
            bundle_patterns=[],
            downstream_chains=[],
        )
        issue_pay_cands = [c for c in candidates if c["entity_name"] == "Issue Payment"]
        assert issue_pay_cands, "Issue Payment should be promoted"
        for c in issue_pay_cands:
            assert c["score"] <= 0.62 + 1e-6, (
                f"Issue Payment score {c['score']} exceeds max_score 0.62"
            )

    def test_promoted_score_is_attenuated(self):
        """Promoted score = trigger_score * 0.65, capped at max_score."""
        trigger = 0.70
        candidates = promote_downstream_candidates(
            historical_value_stream_support=[_hist("Order to Cash", trigger)],
            bundle_patterns=[],
            downstream_chains=[],
        )
        issue_pay = [c for c in candidates if c["entity_name"] == "Issue Payment"]
        assert issue_pay
        expected_max = min(0.62, round(trigger * 0.65, 3))
        assert issue_pay[0]["score"] <= expected_max + 1e-6

    def test_skip_when_target_already_well_covered(self):
        """If Issue Payment already has high historical score, skip promotion."""
        candidates = promote_downstream_candidates(
            historical_value_stream_support=[
                _hist("Order to Cash", 0.80),
                _hist("Issue Payment", 0.90),   # already well covered
            ],
            bundle_patterns=[],
            downstream_chains=[],
        )
        issue_pay = [c for c in candidates if c["entity_name"] == "Issue Payment"]
        assert not issue_pay, (
            "Issue Payment should be skipped when already at high historical score"
        )

    def test_allowed_names_filter(self):
        """Candidates outside allowed_names should not be emitted."""
        candidates = promote_downstream_candidates(
            historical_value_stream_support=[_hist("Order to Cash", 0.80)],
            bundle_patterns=[],
            downstream_chains=[],
            allowed_names={"Manage Invoice and Payment Receipt"},  # exclude Issue Payment
        )
        names = [c["entity_name"] for c in candidates]
        assert "Issue Payment" not in names
        assert "Manage Invoice and Payment Receipt" in names

    def test_max_pooling_across_rules(self):
        """Same target promoted by multiple rules uses max score."""
        # o2c_to_issue_payment AND invoice_to_issue_payment both promote Issue Payment
        candidates = promote_downstream_candidates(
            historical_value_stream_support=[
                _hist("Order to Cash", 0.80),
                _hist("Manage Invoice and Payment Receipt", 0.70),
            ],
            bundle_patterns=[],
            downstream_chains=[],
        )
        issue_pay = [c for c in candidates if c["entity_name"] == "Issue Payment"]
        assert len(issue_pay) == 1, "Should be max-pooled to a single entry"

    def test_support_type_is_pattern(self):
        candidates = promote_downstream_candidates(
            historical_value_stream_support=[_hist("Order to Cash", 0.75)],
            bundle_patterns=[],
            downstream_chains=[],
        )
        for c in candidates:
            assert c["support_type"] == "pattern", (
                f"Downstream promoted candidate should have support_type=pattern, got {c['support_type']}"
            )

    def test_sub_source_is_downstream_promotion(self):
        candidates = promote_downstream_candidates(
            historical_value_stream_support=[_hist("Order to Cash", 0.75)],
            bundle_patterns=[],
            downstream_chains=[],
        )
        for c in candidates:
            assert c["sub_source"] == "downstream_promotion"

    def test_provider_network_promotes_credentialing(self):
        candidates = promote_downstream_candidates(
            historical_value_stream_support=[_hist("Manage Provider Network", 0.65)],
            bundle_patterns=[],
            downstream_chains=[],
        )
        names = [c["entity_name"] for c in candidates]
        assert "Credential Provider" in names

    def test_enrollment_promotes_eligibility(self):
        candidates = promote_downstream_candidates(
            historical_value_stream_support=[_hist("Manage Enrollment", 0.70)],
            bundle_patterns=[],
            downstream_chains=[],
        )
        names = [c["entity_name"] for c in candidates]
        assert "Manage Eligibility" in names

    def test_claims_promotes_cob(self):
        candidates = promote_downstream_candidates(
            historical_value_stream_support=[_hist("Adjudicate Claim", 0.70)],
            bundle_patterns=[],
            downstream_chains=[],
        )
        names = [c["entity_name"] for c in candidates]
        assert "Manage COB" in names

    def test_empty_inputs_returns_empty(self):
        candidates = promote_downstream_candidates(
            historical_value_stream_support=[],
            bundle_patterns=[],
            downstream_chains=[],
        )
        assert candidates == []

    def test_missing_rules_file_returns_empty(self):
        candidates = promote_downstream_candidates(
            historical_value_stream_support=[_hist("Order to Cash", 0.80)],
            bundle_patterns=[],
            downstream_chains=[],
            rules_path=pathlib.Path("/nonexistent/path/rules.yaml"),
        )
        assert candidates == []

    def test_match_reason_contains_rule_name(self):
        candidates = promote_downstream_candidates(
            historical_value_stream_support=[_hist("Order to Cash", 0.75)],
            bundle_patterns=[],
            downstream_chains=[],
        )
        issue_pay = [c for c in candidates if c["entity_name"] == "Issue Payment"]
        assert issue_pay
        assert "o2c_to_issue_payment" in issue_pay[0]["match_reason"]


# ---------------------------------------------------------------------------
# Graph node smoke test
# ---------------------------------------------------------------------------

class TestNodePromoteDownstreamCandidatesSmoke:
    """
    Smoke tests for node_promote_downstream_candidates.

    The node is a thin wrapper around promote_downstream_candidates() that:
      1. Reads state keys: historical_value_stream_support, bundle_patterns,
         downstream_chains, allowed_value_stream_names
      2. Calls promote_downstream_candidates()
      3. Returns {"downstream_promoted_candidates": [...]}

    Full behavioral coverage is in TestPromoteDownstreamCandidates above.
    Here we only test the node contract (input/output keys) using the
    promoter function directly to avoid importing nodes.py (which requires
    langchain_core to be installed).
    """

    def _simulate_node(self, state: dict) -> dict:
        """Replicate the node logic without importing nodes.py."""
        allowed = state.get("allowed_value_stream_names")
        allowed_set = set(allowed) if allowed else None
        promoted = promote_downstream_candidates(
            historical_value_stream_support=state.get("historical_value_stream_support", []),
            bundle_patterns=state.get("bundle_patterns", []),
            downstream_chains=state.get("downstream_chains", []),
            allowed_names=allowed_set,
        )
        return {"downstream_promoted_candidates": promoted}

    def test_returns_correct_key(self):
        result = self._simulate_node({})
        assert "downstream_promoted_candidates" in result

    def test_returns_list(self):
        result = self._simulate_node({})
        assert isinstance(result["downstream_promoted_candidates"], list)

    def test_promotes_issue_payment_for_o2c(self):
        state = {"historical_value_stream_support": [_hist("Order to Cash", 0.75)]}
        result = self._simulate_node(state)
        names = [c["entity_name"] for c in result["downstream_promoted_candidates"]]
        assert "Issue Payment" in names

    def test_empty_state_returns_empty(self):
        result = self._simulate_node({})
        assert result["downstream_promoted_candidates"] == []

    def test_allowed_names_filter(self):
        state = {
            "historical_value_stream_support": [_hist("Order to Cash", 0.80)],
            "allowed_value_stream_names": ["Issue Payment"],
        }
        result = self._simulate_node(state)
        names = [c["entity_name"] for c in result["downstream_promoted_candidates"]]
        for name in names:
            assert name in {"Issue Payment"}
