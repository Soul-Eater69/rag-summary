"""
Offline theme index builder (V6).

Clusters historical ticket summaries into themes, then builds a FAISS index
over theme centroid embeddings for fast runtime lookup.

Output (written to --output-dir):
  theme_index.faiss   — FAISS flat IP index (unit-normalised for cosine)
  theme_docs.json     — List[ThemeDoc] serialized (without embeddings)
  theme_manifest.json — ThemeIndexManifest with build metadata

Usage:
    python -m rag_summary.tools.build_theme_index \\
        --summary-dir summaries/ \\
        --output-dir config/theme_index/ \\
        [--n-clusters 40] \\
        [--min-cohesion 0.35] \\
        [--min-vs-support-fraction 0.30] \\
        [--cutoff-date 2024-01-01]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_ticket_summaries(summary_dir: str, cutoff_date: Optional[str]) -> List[Dict[str, Any]]:
    """
    Load all JSON ticket summary files from summary_dir.

    Each file must be a JSON object with at least:
      ticket_id, value_stream_labels (list), retrieval_text (str)
    Optional: capability_tags, canonical_functions, ingested_at / created_at

    Returns only tickets whose temporal marker is before cutoff_date (if given).
    """
    summaries = []
    if not os.path.isdir(summary_dir):
        logger.error("summary_dir does not exist: %s", summary_dir)
        sys.exit(1)

    for fname in sorted(os.listdir(summary_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(summary_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as exc:
            logger.warning("Skipping %s: %s", fname, exc)
            continue

        # Temporal filter
        if cutoff_date:
            ticket_date = doc.get("ingested_at") or doc.get("created_at") or ""
            if ticket_date and ticket_date > cutoff_date:
                continue

        summaries.append(doc)

    logger.info("Loaded %d ticket summaries from %s", len(summaries), summary_dir)
    return summaries


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _embed_texts(texts: List[str], embedding_svc) -> "np.ndarray":
    """Embed a list of texts, returning a float32 numpy array (n, dim)."""
    import numpy as np

    embeddings = []
    for text in texts:
        try:
            vec = embedding_svc.embed_query(text)
            embeddings.append(vec)
        except Exception as exc:
            logger.warning("Embedding failed for text snippet: %s", exc)
            embeddings.append([0.0] * 1536)  # placeholder dim; will be dropped by normalisation

    arr = np.array(embeddings, dtype="float32")
    return arr


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _kmeans_cluster(
    embeddings: "np.ndarray",
    n_clusters: int,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """
    Run K-means via faiss.Kmeans.

    Returns (labels, centroids) where labels[i] is the cluster id for embedding i.
    """
    import faiss
    import numpy as np

    n, dim = embeddings.shape
    k = min(n_clusters, n)

    # Normalise for cosine clustering
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = (embeddings / norms).astype("float32")

    kmeans = faiss.Kmeans(dim, k, niter=40, verbose=False, spherical=True)
    kmeans.train(normed)

    _, labels = kmeans.index.search(normed, 1)
    labels = labels.flatten()
    centroids = kmeans.centroids  # (k, dim) already unit-normalised

    return labels, centroids


# ---------------------------------------------------------------------------
# Cohesion scoring
# ---------------------------------------------------------------------------

def _cohesion_score(member_embeddings: "np.ndarray", centroid: "np.ndarray") -> float:
    """Mean cosine similarity of member embeddings to cluster centroid."""
    import numpy as np

    if len(member_embeddings) == 0:
        return 0.0
    norms = np.linalg.norm(member_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = member_embeddings / norms
    c = centroid / (np.linalg.norm(centroid) + 1e-9)
    sims = normed @ c
    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# Theme doc construction
# ---------------------------------------------------------------------------

def _build_theme_doc(
    cluster_id: int,
    member_tickets: List[Dict[str, Any]],
    member_embeddings: "np.ndarray",
    centroid: "np.ndarray",
    *,
    min_vs_support_fraction: float,
) -> Optional[Dict[str, Any]]:
    """Build a ThemeDoc dict from a cluster of tickets."""
    if not member_tickets:
        return None

    ticket_ids = [t.get("ticket_id", "") for t in member_tickets]
    member_count = len(ticket_ids)

    # VS support counts
    vs_counts: Counter = Counter()
    for ticket in member_tickets:
        for vs in ticket.get("value_stream_labels", []):
            vs_counts[vs] += 1

    vs_support_fractions = {
        vs: round(cnt / member_count, 4)
        for vs, cnt in vs_counts.items()
    }
    qualifying_vs = [
        vs for vs, frac in vs_support_fractions.items()
        if frac >= min_vs_support_fraction
    ]
    if not qualifying_vs:
        return None  # discard cluster with no qualifying VS

    # Capability tags and canonical functions: union across members (top-10 by freq)
    cap_counter: Counter = Counter()
    func_counter: Counter = Counter()
    for ticket in member_tickets:
        for tag in ticket.get("capability_tags", []):
            cap_counter[tag] += 1
        for fn in ticket.get("canonical_functions", []) or ticket.get("direct_functions_canonical", []):
            func_counter[fn] += 1

    capability_tags = [t for t, _ in cap_counter.most_common(10)]
    canonical_functions = [f for f, _ in func_counter.most_common(10)]

    # Cue phrases: top recurring ngrams from retrieval_text (simple word freq)
    word_counter: Counter = Counter()
    for ticket in member_tickets:
        words = ticket.get("retrieval_text", "").lower().split()
        for w in words:
            if len(w) > 4:
                word_counter[w] += 1
    cue_phrases = [w for w, _ in word_counter.most_common(10)]

    # Theme label: most common VS + top cue
    top_vs = qualifying_vs[0] if qualifying_vs else "unknown"
    top_cue = cue_phrases[0] if cue_phrases else "pattern"
    theme_label = f"{top_vs.lower().replace(' ', '-')}-{top_cue}"[:64]
    theme_description = (
        f"Cluster of {member_count} tickets dominated by {top_vs}. "
        f"Top capability tags: {', '.join(capability_tags[:3])}."
    )

    # Latest ticket date in cluster
    dates = [
        t.get("ingested_at") or t.get("created_at") or ""
        for t in member_tickets
    ]
    last_ingested = max((d for d in dates if d), default="")

    cohesion = _cohesion_score(member_embeddings, centroid)

    # Retrieval text: concatenate label, VS names, cue phrases for embedding
    retrieval_text = " ".join([theme_label] + qualifying_vs[:5] + cue_phrases[:5])

    return {
        "theme_id": str(uuid.uuid4()),
        "theme_label": theme_label,
        "theme_description": theme_description,
        "member_ticket_ids": ticket_ids,
        "member_count": member_count,
        "value_stream_names": qualifying_vs,
        "vs_support_counts": dict(vs_counts),
        "vs_support_fractions": vs_support_fractions,
        "canonical_functions": canonical_functions,
        "capability_tags": capability_tags,
        "cue_phrases": cue_phrases,
        "retrieval_text": retrieval_text,
        "last_ticket_ingested_at": last_ingested,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "cohesion_score": round(cohesion, 4),
        "min_vs_support_fraction": min_vs_support_fraction,
    }


# ---------------------------------------------------------------------------
# FAISS index builder
# ---------------------------------------------------------------------------

def _build_faiss_index(theme_embeddings: "np.ndarray") -> "faiss.Index":
    """Build a flat L2 FAISS index over unit-normalised theme embeddings."""
    import faiss
    import numpy as np

    n, dim = theme_embeddings.shape
    norms = np.linalg.norm(theme_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = (theme_embeddings / norms).astype("float32")

    index = faiss.IndexFlatL2(dim)
    index.add(normed)
    return index


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_theme_index(
    summary_dir: str,
    output_dir: str,
    *,
    n_clusters: int = 40,
    min_cohesion: float = 0.35,
    min_vs_support_fraction: float = 0.30,
    cutoff_date: Optional[str] = None,
    embedding_svc=None,
) -> None:
    """
    End-to-end theme index builder.

    1. Load ticket summaries from summary_dir
    2. Embed retrieval_text for each ticket
    3. K-means cluster embeddings
    4. Build ThemeDoc per cluster (discard low-cohesion or no-VS clusters)
    5. Embed theme retrieval_text for centroid index
    6. Write theme_index.faiss, theme_docs.json, theme_manifest.json
    """
    import numpy as np

    if embedding_svc is None:
        from rag_summary.ingestion.adapters import get_default_embedding
        embedding_svc = get_default_embedding()

    # 1. Load
    tickets = _load_ticket_summaries(summary_dir, cutoff_date)
    if len(tickets) < 2:
        logger.error("Need at least 2 tickets to cluster. Found: %d", len(tickets))
        sys.exit(1)

    # 2. Embed tickets
    logger.info("Embedding %d ticket retrieval texts…", len(tickets))
    texts = [t.get("retrieval_text", t.get("title", "")) for t in tickets]
    ticket_embeddings = _embed_texts(texts, embedding_svc)

    # 3. Cluster
    logger.info("Clustering into up to %d themes…", n_clusters)
    labels, centroids = _kmeans_cluster(ticket_embeddings, n_clusters)

    # 4. Group members
    cluster_members: Dict[int, List[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_members[int(label)].append(i)

    # 5. Build ThemeDocs
    theme_docs = []
    discarded = 0
    for cluster_id, member_indices in cluster_members.items():
        member_tickets = [tickets[i] for i in member_indices]
        member_embs = ticket_embeddings[member_indices]
        centroid = centroids[cluster_id]

        doc = _build_theme_doc(
            cluster_id,
            member_tickets,
            member_embs,
            centroid,
            min_vs_support_fraction=min_vs_support_fraction,
        )
        if doc is None:
            discarded += 1
            continue

        cohesion = doc["cohesion_score"]
        if cohesion < min_cohesion:
            logger.debug(
                "Discarding cluster %d (cohesion=%.3f < %.3f)",
                cluster_id, cohesion, min_cohesion,
            )
            discarded += 1
            continue

        theme_docs.append(doc)

    logger.info(
        "Built %d theme docs (%d clusters discarded)", len(theme_docs), discarded
    )
    if not theme_docs:
        logger.error("No theme docs produced — check cohesion threshold or VS data.")
        sys.exit(1)

    # 6. Embed theme retrieval texts
    logger.info("Embedding %d theme retrieval texts for FAISS index…", len(theme_docs))
    theme_texts = [d["retrieval_text"] for d in theme_docs]
    theme_embeddings = _embed_texts(theme_texts, embedding_svc)

    # 7. Build FAISS index
    faiss_index = _build_faiss_index(theme_embeddings)

    # 8. VS coverage stats for manifest
    vs_coverage: Dict[str, int] = Counter()
    all_vs: set = set()
    for doc in theme_docs:
        for vs in doc["value_stream_names"]:
            vs_coverage[vs] += 1
            all_vs.add(vs)

    # 9. Write outputs
    os.makedirs(output_dir, exist_ok=True)

    import faiss as faiss_lib
    index_path = os.path.join(output_dir, "theme_index.faiss")
    faiss_lib.write_index(faiss_index, index_path)
    logger.info("Wrote FAISS index → %s (%d vectors)", index_path, faiss_index.ntotal)

    docs_path = os.path.join(output_dir, "theme_docs.json")
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(theme_docs, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Wrote theme docs → %s", docs_path)

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_faiss_dir": summary_dir,
        "source_ticket_count": len(tickets),
        "theme_count": len(theme_docs),
        "discarded_clusters": discarded,
        "vs_coverage": dict(vs_coverage),
        "uncovered_vs_names": [],  # filled in by caller if known universe provided
        "build_params": {
            "n_clusters": n_clusters,
            "min_cohesion": min_cohesion,
            "min_vs_support_fraction": min_vs_support_fraction,
            "cutoff_date": cutoff_date or "",
        },
    }
    manifest_path = os.path.join(output_dir, "theme_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("Wrote manifest → %s", manifest_path)
    logger.info(
        "Done. %d themes, %d VS covered, %d clusters discarded.",
        len(theme_docs), len(all_vs), discarded,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build offline theme FAISS index (V6)")
    p.add_argument("--summary-dir", required=True, help="Directory of ticket JSON summaries")
    p.add_argument("--output-dir", required=True, help="Output directory for theme index files")
    p.add_argument("--n-clusters", type=int, default=40, help="Number of K-means clusters (default: 40)")
    p.add_argument("--min-cohesion", type=float, default=0.35, help="Min mean cosine cohesion to keep a cluster (default: 0.35)")
    p.add_argument("--min-vs-support-fraction", type=float, default=0.30, help="Min VS support fraction to include VS in theme (default: 0.30)")
    p.add_argument("--cutoff-date", default=None, help="ISO-8601 date; exclude tickets after this date (leakage prevention)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_theme_index(
        summary_dir=args.summary_dir,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        min_cohesion=args.min_cohesion,
        min_vs_support_fraction=args.min_vs_support_fraction,
        cutoff_date=args.cutoff_date,
    )
