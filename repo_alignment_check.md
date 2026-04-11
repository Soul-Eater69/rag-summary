## `Soul-Eater69/rag-summary` — architecture alignment check (rechecked)

## Current status at a glance

The repository is now **materially aligned** with the intended V5 architecture, but it is best described as **~85–90% complete** rather than fully finished.

---

## What is truly implemented now

### 1. V5-style orchestration is real in runtime
The runtime flow now goes beyond the old “summary + KG + pick labels” shape and includes:
- card text cleaning/normalization
- structured summary generation
- FAISS analog retrieval
- historical value-stream evidence collection
- KG candidate retrieval
- capability mapping enrichment path
- candidate evidence build
- source-aware fused ranking
- three-class output contract

The runtime output now centers on:
- `directly_supported`
- `pattern_inferred`
- `no_evidence`

---

### 2. Historical initiative memory is much richer
The FAISS historical document format now carries more than semantic summaries. It includes:
- raw + canonical direct/implied functions
- capability tags
- operational footprint
- mapped value streams
- support type
- supporting evidence

This is a major move toward true historical pattern memory.

---

### 3. Capability mapping is live runtime logic
Capability mapping is no longer architectural intent only. Runtime behavior now includes:
- loading `config/capability_map.yaml`
- cue matching via direct/indirect signals
- canonical function matching
- stream promotion and candidate emission

The capability map itself is now substantial (with meaningful clusters rather than a stub ontology).

---

### 4. `CandidateEvidence` is now a concrete runtime object
Per-candidate evidence modeling exists and carries fields such as:
- source scores
- evidence sources
- evidence snippets
- support type
- source diversity count
- fused/confidence placeholders

This meaningfully improves debuggability and auditability.

---

### 5. Source-aware fusion and three-class contract are in code
The system now has explicit fusion mechanics (weighted sources, penalties/guards) and a selector contract that outputs three classes instead of a flat legacy list.

---

### 6. Capability-map bootstrap path exists offline
A bootstrap tool exists for:
- Azure value-stream corpus fetching
- template-guided draft map generation
- coverage reporting
- writing `capability_map.yaml`

This provides a practical build-time path, even if it is still a first-pass generator.

---

## Key mismatches still open

### 1. “Two-pass selector” is not yet truly two-pass
Architecture language implies a genuine two-stage verifier/select flow, but implementation behavior is still effectively a single LLM classification call, not a strict pass-1 verification artifact feeding an independent pass-2 selection step.

### 2. `enriched_candidates` is only partially honored downstream
Pipeline enrichment exists, but enriched KG candidate effects are not consistently propagated as the dominant downstream KG evidence/ranking path. Capability mapping is active, but full enriched-candidate influence remains partial.

### 3. Theme source remains mostly schema-level
`theme` exists in evidence schema and fusion weighting, but the pipeline still does not consistently pass live `theme_candidates` into evidence construction. It is present in design surfaces, not fully alive as a runtime source.

### 4. Attachment evidence is present but still shallow
Attachment candidates are still largely analog-snippet-derived, rather than driven by a deep new-card attachment understanding path. This is improvement, but not the attachment-first model described by the architecture.

### 5. Selector prompt is not yet maximally evidence-grounded
Candidate packaging into the LLM is improved, but it still does not always inject a deeply structured, per-candidate evidence bundle with tightly bound snippets and support rationale for every candidate.

### 6. Bootstrap remains template-led, not learned from history
Bootstrap tooling is useful, but still draft-oriented. It does not yet robustly mine historical ticket patterns to auto-learn and validate capability clusters at production quality.

---

## Updated verdict

### Strongly implemented
- richer historical store
- raw + canonical function extraction
- runtime capability mapping
- concrete `CandidateEvidence`
- fused scoring framework
- three-class output contract
- offline bootstrap tooling for capability map generation

### Still incomplete
- true two-pass selector implementation
- live theme candidate source in evidence build
- full downstream exploitation of `enriched_candidates`
- deeper attachment-first reasoning
- stronger per-candidate evidence grounding into LLM selection
- richer bootstrap enrichment from historical ticket learning

---

## One-line summary

`rag-summary` has crossed into **real V5 architecture implementation**, but remains a **highly aligned (≈85–90%) system in progress**, not the final fully realized evidence-fusion runtime.
