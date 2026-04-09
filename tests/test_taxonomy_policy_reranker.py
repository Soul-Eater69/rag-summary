"""
Tests for the taxonomy policy reranker (Phase 4).

Covers:
  - YAML configs load correctly (policy rules + historical priors)
  - _load_policy_rules / _load_historical_priors
  - _compute_policy_adjustment signal logic
  - rerank_candidates_by_taxonomy_policy:
      - canonicalization via registry
      - suppress_if_preferred + suppress_if_only_semantic
      - sibling dominance suppression
      - historical prior adjacent-weak penalty
      - eligibility_boost application
      - output keys and sorting
  - Edge cases: empty inputs, missing configs, no registry
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))

_POLICY_RULES_PATH = _REPO_ROOT / "config" / "taxonomy_policy_rules.yaml"
_PRIORS_PATH = _REPO_ROOT / "config" / "historical_label_priors.yaml"

# ---------------------------------------------------------------------------
# Import module under test
# ---------------------------------------------------------------------------

import importlib.util as _ilu


def _load_module(name: str, path: pathlib.Path):
    spec = _ilu.spec_from_file_location(name, str(path))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_reranker = _load_module(
    "policy_reranker",
    _REPO_ROOT / "taxonomy" / "policy_reranker.py",
)

rerank_candidates_by_taxonomy_policy = _reranker.rerank_candidates_by_taxonomy_policy
_load_policy_rules = _reranker._load_policy_rules
_load_historical_priors = _reranker._load_historical_priors
_compute_policy_adjustment = _reranker._compute_policy_adjustment
_count_signal_matches = _reranker._count_signal_matches


# ---------------------------------------------------------------------------
# Minimal stub registry
# ---------------------------------------------------------------------------

class _Stream:
    def __init__(self, canonical_name, family="", preferred_over=None,
                 suppress_if_preferred=False, aliases=None):
        self.canonical_name = canonical_name
        self.family = family
        self.preferred_over = preferred_over or []
        self.suppress_if_preferred = suppress_if_preferred
        self.aliases = aliases or []


class _Registry:
    def __init__(self, streams):
        self.streams = streams
        self._map = {}
        self.canonical_label_map = {}
        for s in streams:
            self._map[s.canonical_name.lower()] = s.canonical_name
            self.canonical_label_map[s.canonical_name.lower()] = s.canonical_name
            for a in s.aliases:
                self._map[a.lower()] = s.canonical_name
                self.canonical_label_map[a.lower()] = s.canonical_name

    def canonicalize(self, name: str) -> str:
        return self._map.get(name.lower(), name)

    def get_stream(self, name: str):
        canon = self.canonicalize(name)
        for s in self.streams:
            if s.canonical_name == canon:
                return s
        return None


def _make_judgment(name, bucket="directly_supported", confidence=0.70, rationale=""):
    return {
        "entity_name": name,
        "bucket": bucket,
        "confidence": confidence,
        "rationale": rationale,
    }


def _make_evidence(name, fused_score=0.60):
    return {
        "candidate_name": name,
        "fused_score": fused_score,
    }


# ---------------------------------------------------------------------------
# YAML structure tests
# ---------------------------------------------------------------------------

class TestPolicyRulesYAML:
    def test_file_exists(self):
        assert _POLICY_RULES_PATH.exists()

    def test_has_version_and_streams(self):
        with _POLICY_RULES_PATH.open() as f:
            payload = yaml.safe_load(f)
        assert "version" in payload
        assert "streams" in payload
        assert isinstance(payload["streams"], dict)
        assert len(payload["streams"]) > 0

    def test_key_streams_present(self):
        rules = _load_policy_rules()
        for expected in [
            "Order to Cash",
            "Issue Payment",
            "Configure, Price, and Quote",
            "Resolve Request Inquiry",
            "Manage Enrollment",
            "Adjudicate Claim",
            "Manage Care Coordination",
            "Promote Community Health",
            "Develop Mission, Vision, and Strategy",
            "Onboard Partner",
        ]:
            assert expected in rules, f"Expected rule for '{expected}'"

    def test_each_rule_has_family(self):
        rules = _load_policy_rules()
        for name, rule in rules.items():
            assert "family" in rule, f"Rule '{name}' missing 'family'"

    def test_requires_any_is_list(self):
        rules = _load_policy_rules()
        for name, rule in rules.items():
            if "requires_any" in rule:
                assert isinstance(rule["requires_any"], list), \
                    f"Rule '{name}' requires_any must be list"

    def test_eligibility_boost_numeric(self):
        rules = _load_policy_rules()
        for name, rule in rules.items():
            if "eligibility_boost" in rule:
                assert isinstance(rule["eligibility_boost"], (int, float)), \
                    f"Rule '{name}' eligibility_boost must be numeric"


class TestHistoricalPriorsYAML:
    def test_file_exists(self):
        assert _PRIORS_PATH.exists()

    def test_has_version_and_families(self):
        with _PRIORS_PATH.open() as f:
            payload = yaml.safe_load(f)
        assert "version" in payload
        assert "families" in payload
        assert isinstance(payload["families"], dict)
        assert len(payload["families"]) > 0

    def test_key_families_present(self):
        priors = _load_historical_priors()
        for expected in ["finance", "enrollment", "claims", "care_management"]:
            assert expected in priors, f"Expected priors for family '{expected}'"

    def test_dominant_labels_are_lists(self):
        priors = _load_historical_priors()
        for fam, prior in priors.items():
            assert isinstance(prior.get("dominant_labels", []), list), \
                f"Family '{fam}' dominant_labels must be list"


# ---------------------------------------------------------------------------
# _compute_policy_adjustment tests
# ---------------------------------------------------------------------------

class TestComputePolicyAdjustment:
    def _rule(self, requires_any=None, weak_only=None, boost=0.10):
        return {
            "requires_any": requires_any or [],
            "weak_only_not_enough": weak_only or [],
            "eligibility_boost": boost,
        }

    def test_positive_signal_returns_boost(self):
        rule = self._rule(requires_any=["order to cash"], boost=0.08)
        adj = _compute_policy_adjustment("order to cash billing", rule)
        assert adj == pytest.approx(0.08)

    def test_weak_only_returns_penalty(self):
        rule = self._rule(requires_any=["order to cash"], weak_only=["billing"], boost=0.08)
        adj = _compute_policy_adjustment("billing process only", rule)
        assert adj < 0.0

    def test_no_signals_returns_zero(self):
        rule = self._rule(requires_any=["order to cash"], boost=0.08)
        adj = _compute_policy_adjustment("unrelated text here", rule)
        assert adj == pytest.approx(0.0)

    def test_positive_signal_wins_over_weak(self):
        rule = self._rule(requires_any=["cpq", "configure price"], weak_only=["price"], boost=0.12)
        adj = _compute_policy_adjustment("cpq configuration with price", rule)
        assert adj == pytest.approx(0.12)  # positive signal present → boost

    def test_empty_rule_returns_zero(self):
        assert _compute_policy_adjustment("any text", {}) == pytest.approx(0.0)

    def test_zero_boost_weak_penalty_is_small(self):
        rule = self._rule(requires_any=["credentialing"], weak_only=["provider"], boost=0.0)
        adj = _compute_policy_adjustment("provider data", rule)
        assert adj <= 0.0  # weak-only with no boost gives small default penalty


# ---------------------------------------------------------------------------
# Core reranker tests
# ---------------------------------------------------------------------------

class TestRerankerOutputKeys:
    def test_returns_required_keys(self):
        result = rerank_candidates_by_taxonomy_policy(
            verify_judgments=[],
            candidate_evidence=[],
            taxonomy_registry=None,
        )
        assert "taxonomy_reranked_candidates" in result
        assert "taxonomy_suppressed_candidates" in result
        assert "taxonomy_decisions" in result

    def test_empty_inputs(self):
        result = rerank_candidates_by_taxonomy_policy(
            verify_judgments=[],
            candidate_evidence=[],
            taxonomy_registry=None,
        )
        assert result["taxonomy_reranked_candidates"] == []
        assert result["taxonomy_suppressed_candidates"] == []
        assert result["taxonomy_decisions"] == []


class TestRerankerCanonicalization:
    def test_alias_resolved_to_canonical(self):
        registry = _Registry([
            _Stream("Order to Cash", family="finance", aliases=["O2C"]),
        ])
        judgments = [_make_judgment("O2C", confidence=0.70)]
        evidence = [_make_evidence("Order to Cash", fused_score=0.65)]
        result = rerank_candidates_by_taxonomy_policy(
            verify_judgments=judgments,
            candidate_evidence=evidence,
            taxonomy_registry=registry,
        )
        reranked = result["taxonomy_reranked_candidates"]
        assert len(reranked) == 1
        assert reranked[0]["entity_name"] == "Order to Cash"

    def test_unknown_name_passes_through(self):
        registry = _Registry([])
        judgments = [_make_judgment("Unknown Stream")]
        result = rerank_candidates_by_taxonomy_policy(
            verify_judgments=judgments,
            candidate_evidence=[],
            taxonomy_registry=registry,
        )
        assert result["taxonomy_reranked_candidates"][0]["entity_name"] == "Unknown Stream"


class TestSuppressIfPreferred:
    def test_suppressible_dominated_stream_suppressed(self):
        """When Configure, Price, and Quote (preferred) scores much higher than a
        suppressible sibling, the sibling is suppressed."""
        registry = _Registry([
            _Stream(
                "Configure, Price, and Quote",
                family="product_and_pricing",
                preferred_over=["Manage Product and Service Inventory"],
            ),
            _Stream(
                "Manage Product and Service Inventory",
                family="product_and_pricing",
                suppress_if_preferred=True,
            ),
        ])
        judgments = [
            _make_judgment("Configure, Price, and Quote", confidence=0.80),
            _make_judgment("Manage Product and Service Inventory", confidence=0.40),
        ]
        evidence = [
            _make_evidence("Configure, Price, and Quote", fused_score=0.80),
            _make_evidence("Manage Product and Service Inventory", fused_score=0.40),
        ]
        result = rerank_candidates_by_taxonomy_policy(
            verify_judgments=judgments,
            candidate_evidence=evidence,
            taxonomy_registry=registry,
        )
        suppressed_names = [s["entity_name"] for s in result["taxonomy_suppressed_candidates"]]
        reranked_names = [r["entity_name"] for r in result["taxonomy_reranked_candidates"]]
        assert "Manage Product and Service Inventory" in suppressed_names
        assert "Configure, Price, and Quote" in reranked_names

    def test_not_suppressed_when_gap_too_small(self):
        """When preferred stream's score advantage is below threshold, no suppression."""
        registry = _Registry([
            _Stream(
                "Configure, Price, and Quote",
                family="product_and_pricing",
                preferred_over=["Manage Product and Service Inventory"],
            ),
            _Stream(
                "Manage Product and Service Inventory",
                family="product_and_pricing",
                suppress_if_preferred=True,
            ),
        ])
        judgments = [
            _make_judgment("Configure, Price, and Quote", confidence=0.65),
            _make_judgment("Manage Product and Service Inventory", confidence=0.60),
        ]
        evidence = [
            _make_evidence("Configure, Price, and Quote", fused_score=0.65),
            _make_evidence("Manage Product and Service Inventory", fused_score=0.60),
        ]
        result = rerank_candidates_by_taxonomy_policy(
            verify_judgments=judgments,
            candidate_evidence=evidence,
            taxonomy_registry=registry,
        )
        suppressed_names = [s["entity_name"] for s in result["taxonomy_suppressed_candidates"]]
        assert "Manage Product and Service Inventory" not in suppressed_names


class TestSemanticOnlySuppression:
    def _policy_rules_with_semantic_flag(self):
        return {
            "Promote Community Health": {
                "family": "care_management",
                "requires_any": ["community health", "health equity", "public health"],
                "weak_only_not_enough": ["community", "health"],
                "suppress_if_only_semantic": True,
                "minimum_support_type": "direct",
                "eligibility_boost": -0.05,
            }
        }

    def test_semantic_only_suppressed_when_no_signals_and_pattern(self):
        """Pattern-inferred Promote Community Health with no text signals → suppressed."""
        result = rerank_candidates_by_taxonomy_policy(
            verify_judgments=[
                _make_judgment(
                    "Promote Community Health",
                    bucket="pattern_inferred",
                    confidence=0.45,
                )
            ],
            candidate_evidence=[_make_evidence("Promote Community Health", 0.45)],
            taxonomy_registry=None,
            taxonomy_policy_rules=self._policy_rules_with_semantic_flag(),
            lower_card_text="this card is about improving health outcomes broadly",
        )
        suppressed = [s["entity_name"] for s in result["taxonomy_suppressed_candidates"]]
        assert "Promote Community Health" in suppressed

    def test_not_suppressed_when_positive_signal_present(self):
        """When community health signal is in text, no suppression."""
        result = rerank_candidates_by_taxonomy_policy(
            verify_judgments=[
                _make_judgment("Promote Community Health", bucket="directly_supported", confidence=0.70)
            ],
            candidate_evidence=[_make_evidence("Promote Community Health", 0.70)],
            taxonomy_registry=None,
            taxonomy_policy_rules=self._policy_rules_with_semantic_flag(),
            lower_card_text="this card addresses community health equity programs",
        )
        suppressed = [s["entity_name"] for s in result["taxonomy_suppressed_candidates"]]
        assert "Promote Community Health" not in suppressed


class TestEligibilityBoost:
    def test_positive_boost_applied(self):
        """When text confirms a required signal, eligibility_score > fused_score."""
        rules = {
            "Manage COB": {
                "family": "recovery",
                "requires_any": ["coordination of benefits", "cob"],
                "weak_only_not_enough": [],
                "suppress_if_only_semantic": False,
                "eligibility_boost": 0.08,
            }
        }
        result = rerank_candidates_by_taxonomy_policy(
            verify_judgments=[_make_judgment("Manage COB", confidence=0.60)],
            candidate_evidence=[_make_evidence("Manage COB", fused_score=0.60)],
            taxonomy_registry=None,
            taxonomy_policy_rules=rules,
            lower_card_text="coordination of benefits workflow for primary and secondary payer",
        )
        reranked = result["taxonomy_reranked_candidates"]
        assert reranked
        cob = reranked[0]
        assert cob["eligibility_score"] > cob["fused_score"]

    def test_negative_adjustment_applied(self):
        """When only weak signals match, eligibility_score <= fused_score."""
        rules = {
            "Develop Mission, Vision, and Strategy": {
                "family": "strategy",
                "requires_any": ["strategic planning", "enterprise strategy", "strategic roadmap"],
                "weak_only_not_enough": ["strategy", "strategic", "planning"],
                "suppress_if_only_semantic": True,
                "minimum_support_type": "direct",
                "eligibility_boost": -0.10,
            }
        }
        result = rerank_candidates_by_taxonomy_policy(
            verify_judgments=[_make_judgment(
                "Develop Mission, Vision, and Strategy",
                bucket="pattern_inferred",
                confidence=0.50,
            )],
            candidate_evidence=[_make_evidence("Develop Mission, Vision, and Strategy", 0.50)],
            taxonomy_registry=None,
            taxonomy_policy_rules=rules,
            # Only weak signals present (no "strategic planning", "enterprise strategy", etc.)
            lower_card_text="we need a better strategy and planning for our platform",
        )
        # suppress_if_only_semantic + minimum_support_type=direct means pattern_inferred
        # candidates with no positive signals should be suppressed.
        suppressed = [s["entity_name"] for s in result["taxonomy_suppressed_candidates"]]
        assert "Develop Mission, Vision, and Strategy" in suppressed


class TestSortingOrder:
    def test_directly_supported_before_pattern(self):
        """directly_supported candidates come before pattern_inferred in output."""
        result = rerank_candidates_by_taxonomy_policy(
            verify_judgments=[
                _make_judgment("Stream A", bucket="pattern_inferred", confidence=0.80),
                _make_judgment("Stream B", bucket="directly_supported", confidence=0.50),
            ],
            candidate_evidence=[
                _make_evidence("Stream A", 0.80),
                _make_evidence("Stream B", 0.50),
            ],
            taxonomy_registry=None,
        )
        reranked = result["taxonomy_reranked_candidates"]
        buckets = [r["bucket"] for r in reranked]
        direct_idx = buckets.index("directly_supported")
        pattern_idx = buckets.index("pattern_inferred")
        assert direct_idx < pattern_idx

    def test_higher_eligibility_first_within_bucket(self):
        """Within same bucket, higher eligibility_score comes first."""
        result = rerank_candidates_by_taxonomy_policy(
            verify_judgments=[
                _make_judgment("Low Score", bucket="directly_supported", confidence=0.40),
                _make_judgment("High Score", bucket="directly_supported", confidence=0.80),
            ],
            candidate_evidence=[
                _make_evidence("Low Score", 0.40),
                _make_evidence("High Score", 0.80),
            ],
            taxonomy_registry=None,
        )
        reranked = result["taxonomy_reranked_candidates"]
        names = [r["entity_name"] for r in reranked]
        assert names.index("High Score") < names.index("Low Score")


class TestMissingConfig:
    def test_missing_policy_rules_file_returns_empty(self):
        rules = _load_policy_rules(pathlib.Path("/nonexistent/path/rules.yaml"))
        assert rules == {}

    def test_missing_priors_file_returns_empty(self):
        priors = _load_historical_priors(pathlib.Path("/nonexistent/path/priors.yaml"))
        assert priors == {}

    def test_reranker_with_no_config_passes_through(self):
        """Without policy files or registry, judgments pass through unchanged."""
        judgments = [
            _make_judgment("Order to Cash", confidence=0.75),
            _make_judgment("Issue Payment", confidence=0.55),
        ]
        result = rerank_candidates_by_taxonomy_policy(
            verify_judgments=judgments,
            candidate_evidence=[
                _make_evidence("Order to Cash", 0.75),
                _make_evidence("Issue Payment", 0.55),
            ],
            taxonomy_registry=None,
            taxonomy_policy_rules={},
            historical_label_priors={},
        )
        reranked_names = [r["entity_name"] for r in result["taxonomy_reranked_candidates"]]
        assert "Order to Cash" in reranked_names
        assert "Issue Payment" in reranked_names
        assert result["taxonomy_suppressed_candidates"] == []


class TestDecisionLog:
    def test_decisions_contain_all_candidates(self):
        """taxonomy_decisions should contain every input candidate."""
        judgments = [
            _make_judgment("Order to Cash"),
            _make_judgment("Issue Payment"),
        ]
        result = rerank_candidates_by_taxonomy_policy(
            verify_judgments=judgments,
            candidate_evidence=[],
            taxonomy_registry=None,
        )
        decision_names = {d["entity_name"] for d in result["taxonomy_decisions"]}
        assert "Order to Cash" in decision_names
        assert "Issue Payment" in decision_names

    def test_decision_has_required_fields(self):
        judgments = [_make_judgment("Order to Cash")]
        result = rerank_candidates_by_taxonomy_policy(
            verify_judgments=judgments,
            candidate_evidence=[],
            taxonomy_registry=None,
        )
        decision = result["taxonomy_decisions"][0]
        for field in ("entity_name", "bucket", "fused_score", "eligibility_score",
                      "policy_adjustment", "suppressed", "suppression_reason"):
            assert field in decision, f"Decision missing field '{field}'"


class TestRealConfigs:
    """Integration tests using the actual YAML config files."""

    def test_o2c_no_suppression_with_signals(self):
        """Order to Cash with strong text signals should not be suppressed."""
        result = rerank_candidates_by_taxonomy_policy(
            verify_judgments=[_make_judgment("Order to Cash", confidence=0.75)],
            candidate_evidence=[_make_evidence("Order to Cash", 0.75)],
            taxonomy_registry=None,
            lower_card_text="order to cash process including invoice and payment collection",
        )
        suppressed_names = [s["entity_name"] for s in result["taxonomy_suppressed_candidates"]]
        assert "Order to Cash" not in suppressed_names

    def test_strategy_weak_signal_suppressed(self):
        """Develop Mission, Vision, and Strategy with only weak 'strategy' signal
        and pattern_inferred bucket should be suppressed."""
        result = rerank_candidates_by_taxonomy_policy(
            verify_judgments=[
                _make_judgment(
                    "Develop Mission, Vision, and Strategy",
                    bucket="pattern_inferred",
                    confidence=0.45,
                )
            ],
            candidate_evidence=[
                _make_evidence("Develop Mission, Vision, and Strategy", 0.45)
            ],
            taxonomy_registry=None,
            lower_card_text="this is a strategy to improve our claims platform",
        )
        suppressed_names = [s["entity_name"] for s in result["taxonomy_suppressed_candidates"]]
        assert "Develop Mission, Vision, and Strategy" in suppressed_names

    def test_cpq_with_strong_signal_not_suppressed(self):
        """Configure, Price, and Quote with cpq signal text not suppressed."""
        result = rerank_candidates_by_taxonomy_policy(
            verify_judgments=[
                _make_judgment("Configure, Price, and Quote", bucket="directly_supported", confidence=0.70)
            ],
            candidate_evidence=[_make_evidence("Configure, Price, and Quote", 0.70)],
            taxonomy_registry=None,
            lower_card_text="we need a cpq solution with quoting engine and rate card generation",
        )
        suppressed_names = [s["entity_name"] for s in result["taxonomy_suppressed_candidates"]]
        assert "Configure, Price, and Quote" not in suppressed_names
