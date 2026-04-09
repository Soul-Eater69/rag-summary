from __future__ import annotations

import pathlib
import sys
import importlib.util as _ilu
import types


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))


def _load_module(name: str, path: pathlib.Path):
    spec = _ilu.spec_from_file_location(name, str(path))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_canon_mod = _load_module("canonicalize_predictions", _REPO_ROOT / "eval" / "canonicalize_predictions.py")
_metrics_mod = _load_module("eval_taxonomy_metrics", _REPO_ROOT / "eval" / "eval_taxonomy_metrics.py")

# Bridge absolute imports used inside eval_taxonomy_metrics without requiring full package install.
rag_summary_pkg = types.ModuleType("rag_summary")
rag_summary_eval_pkg = types.ModuleType("rag_summary.eval")
rag_summary_eval_pkg.canonicalize_predictions = _canon_mod
rag_summary_pkg.eval = rag_summary_eval_pkg
sys.modules.setdefault("rag_summary", rag_summary_pkg)
sys.modules.setdefault("rag_summary.eval", rag_summary_eval_pkg)
sys.modules.setdefault("rag_summary.eval.canonicalize_predictions", _canon_mod)


canonicalize_predictions = _canon_mod.canonicalize_predictions
extract_predicted_names = _canon_mod.extract_predicted_names
compute_exact_metrics = _metrics_mod.compute_exact_metrics
compute_family_metrics = _metrics_mod.compute_family_metrics
evaluate_batch = _metrics_mod.evaluate_batch


class _Stream:
    def __init__(self, canonical_name: str, family: str, aliases=None):
        self.canonical_name = canonical_name
        self.family = family
        self.aliases = aliases or []


class _Registry:
    def __init__(self):
        self.streams = [
            _Stream("Order to Cash", "finance", aliases=["Order-to-Cash", "O2C"]),
            _Stream("Issue Payment", "finance"),
            _Stream("Configure, Price, and Quote", "product_and_pricing", aliases=["CPQ"]),
        ]
        self.canonical_label_map = {}
        for s in self.streams:
            self.canonical_label_map[s.canonical_name.lower()] = s.canonical_name
            for a in s.aliases:
                self.canonical_label_map[a.lower()] = s.canonical_name

    def canonicalize(self, name: str) -> str:
        return self.canonical_label_map.get(name.lower(), name)

    def get_stream(self, name: str):
        canon = self.canonicalize(name)
        for s in self.streams:
            if s.canonical_name == canon:
                return s
        return None


def test_canonicalize_alias_and_punctuation_variants():
    registry = _Registry()
    output = canonicalize_predictions(
        {
            "directly_supported": [{"entity_name": "Order-to-Cash", "confidence": 0.8}],
            "pattern_inferred": [{"entity_name": "CPQ", "confidence": 0.6}],
            "taxonomy_suppressed_candidates": [{"entity_name": "O2C"}],
        },
        registry=registry,
    )
    names = extract_predicted_names(output, include_pattern=True)
    assert "Order to Cash" in names
    assert "Configure, Price, and Quote" in names
    assert output["taxonomy_suppressed"] == ["Order to Cash"]


def test_eval_metrics_exact_vs_family_aware():
    registry = _Registry()
    exact = compute_exact_metrics(["Issue Payment"], ["Order to Cash"])
    family = compute_family_metrics(["Issue Payment"], ["Order to Cash"], registry)
    assert exact["f1"] == 0.0
    assert family["family_f1"] > 0.0
    assert "finance" in family["matched_families"]


def test_evaluate_batch_returns_expected_aggregates():
    registry = _Registry()
    predictions = [
        canonicalize_predictions(
            {"directly_supported": [{"entity_name": "O2C", "confidence": 0.9}]},
            registry=registry,
        ),
        canonicalize_predictions(
            {"pattern_inferred": [{"entity_name": "Issue Payment", "confidence": 0.55}]},
            registry=registry,
        ),
    ]
    gts = [["Order to Cash"], ["Order to Cash"]]
    report = evaluate_batch(predictions, gts, registry=registry)
    assert report["n_cards"] == 2
    assert "macro_avg" in report
    assert "micro_avg" in report
    assert "family_agg" in report
