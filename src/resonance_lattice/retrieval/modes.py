# SPDX-License-Identifier: BUSL-1.1
"""Unified retrieval-mode dispatcher — the 4 production modes in one place.

Promoted from `benchmarks/bench_new_arch_beir.py:_run_eval` (Horizon 1
board item 236a). Modes now live as first-class library calls so `rlat
search --retrieval-mode <mode>` (board item 236b) and the build-time
probe (board item 236c) can both reuse the same orchestration the
measurement harness used in 2026-04-20.

Modes:
  - field_only               — dense retrieval, no post-processing.
  - plus_cross_encoder       — dense + cross-encoder rerank (inline).
  - plus_cross_encoder_expanded — CE with B1 passage expansion.
  - plus_hybrid              — dense + B3 ripgrep lexical rerank.
  - bm25_only                — BM25 sparse retrieval only.
  - plus_rrf                 — dense (chunk→doc) ⊕ BM25 via RRF fusion.
  - plus_full_stack          — RRF candidate set → cross-encoder rerank.

Semantic contract: given the same `(lattice, query, mode)` inputs, this
function returns identical doc-level rankings to what the 2026-04-20
5-BEIR measurement produced. Behavior must not drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from resonance_lattice.retrieval.bm25 import BM25Query, reciprocal_rank_fusion
from resonance_lattice.retrieval.lexical import ScoredHit, lexical_rerank

if TYPE_CHECKING:
    from resonance_lattice.lattice import Lattice
    from resonance_lattice.reranker import CrossEncoderReranker

VALID_MODES = (
    "field_only",
    "plus_cross_encoder",
    "plus_cross_encoder_expanded",
    "plus_hybrid",
    "bm25_only",
    "plus_rrf",
    "plus_full_stack",
)


def retrieve(
    lattice: Lattice,
    query: str,
    *,
    mode: str,
    top_k: int = 100,
    source_root: Path | str | None = None,
    bm25: BM25Query | None = None,
    cross_encoder: CrossEncoderReranker | None = None,
) -> dict[str, float]:
    """Run `mode` against `lattice` for `query`, return doc-level scores.

    Returns a `{doc_id: score}` dict ranked descending (though dict order
    is not a ranking — callers that need an ordered list should sort).

    Args:
        lattice: loaded Lattice instance.
        query: query text.
        mode: one of VALID_MODES.
        top_k: top-K per upstream signal + final cut for fused modes.
        source_root: directory with per-doc text files (required for
            plus_hybrid, plus_full_stack when BM25-only candidates need
            text resolution).
        bm25: preloaded BM25Query (required for bm25_only, plus_rrf,
            plus_full_stack). Lets the caller amortise BM25 load across
            many queries.
        cross_encoder: preloaded CrossEncoderReranker (required for
            plus_full_stack; optional for plus_cross_encoder — will lazy-
            initialize). Lets the caller amortise model load.

    Raises:
        ValueError: on unknown `mode`.
        RuntimeError: on missing required artifacts (bm25 for fused
            modes, source_root for plus_full_stack text resolution).
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"Unknown retrieval mode: {mode!r}. "
            f"Valid modes: {', '.join(VALID_MODES)}"
        )

    # Shortcut: bm25_only skips the dense path entirely.
    if mode == "bm25_only":
        if bm25 is None:
            raise RuntimeError(
                f"Mode {mode!r} requires a loaded BM25Query. "
                f"Load with: BM25Query.load('<cartridge>.bm25')"
            )
        doc_ids, scores = bm25.query(query, top_k=top_k)
        return {d: s for d, s in zip(doc_ids, scores)}

    # Dense query kwargs. enable_lexical is the OLD in-field lexical
    # injection path — kept OFF universally; B3 and BM25 sidecar are
    # the post-retrieval replacements.
    query_kwargs = dict(
        top_k=top_k,
        enable_cascade=False,
        enable_contradictions=False,
        enable_lexical=False,
        enable_rerank=False,
    )
    if mode in ("plus_cross_encoder", "plus_cross_encoder_expanded"):
        query_kwargs["enable_cross_encoder"] = True
        if mode == "plus_cross_encoder_expanded":
            query_kwargs["cross_encoder_expand"] = True
    # plus_full_stack uses cross-encoder AFTER RRF, not inline — so
    # enable_cross_encoder stays False here.

    enriched = lattice.enriched_query(query, **query_kwargs)

    # Extract (source_file, score, char_offset, text) per hit.
    raw_hits: list[tuple[str, float, int, str]] = []
    for r in enriched.results:
        meta = (r.content.metadata or {}) if r.content else {}
        source_file = meta.get("source_file", "") or ""
        char_offset = int(meta.get("char_offset") or 0)
        text = (r.content.full_text if r.content else "") or ""
        raw_hits.append((source_file, float(r.score), char_offset, text))

    if mode == "plus_hybrid":
        sh = [
            ScoredHit(
                source_file=sf, char_offset=co,
                char_length=len(t), text=t, score=s,
            )
            for (sf, s, co, t) in raw_hits
        ]
        reranked = lexical_rerank(sh, query, source_root=source_root)
        return _aggregate_chunks_to_docs_max(
            reranked,
            score_getter=lambda h: h.score,
            path_getter=lambda h: h.source_file,
        )

    if mode in ("plus_rrf", "plus_full_stack"):
        if bm25 is None:
            raise RuntimeError(
                f"Mode {mode!r} requires a loaded BM25Query. "
                f"Load with: BM25Query.load('<cartridge>.bm25')"
            )

        # Dense (chunk→doc max) ranking.
        dense_per_doc: dict[str, float] = {}
        for sf, s, _co, _t in raw_hits:
            if not sf:
                continue
            did = Path(sf).stem
            if did not in dense_per_doc or s > dense_per_doc[did]:
                dense_per_doc[did] = s
        dense_ranking = [
            d for d, _ in sorted(
                dense_per_doc.items(), key=lambda kv: kv[1], reverse=True
            )
        ]

        bm25_ranking, _bm25_scores = bm25.query(query, top_k=top_k)

        fused = reciprocal_rank_fusion(
            [dense_ranking, bm25_ranking], k=60, top_k=top_k,
        )

        if mode == "plus_rrf":
            return {did: score for did, score in fused}

        # plus_full_stack: cross-encoder rerank the top-50 fused candidates.
        return _cross_encoder_over_fused(
            query=query,
            fused=fused,
            raw_hits=raw_hits,
            source_root=source_root,
            cross_encoder=cross_encoder,
            top_k=top_k,
        )

    # field_only, plus_cross_encoder, plus_cross_encoder_expanded — all
    # handled entirely by enriched_query. Collapse chunks to docs by max.
    return _aggregate_chunks_to_docs_max_tuple(raw_hits)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _aggregate_chunks_to_docs_max(
    hits,
    *,
    score_getter,
    path_getter,
) -> dict[str, float]:
    """Collapse chunk-level hits to doc-level scores (max aggregation)."""
    per_doc: dict[str, float] = {}
    for h in hits:
        source_file = path_getter(h)
        if not source_file:
            continue
        doc_id = Path(source_file).stem
        score = float(score_getter(h))
        prev = per_doc.get(doc_id)
        if prev is None or score > prev:
            per_doc[doc_id] = score
    return per_doc


def _aggregate_chunks_to_docs_max_tuple(
    raw_hits: list[tuple[str, float, int, str]],
) -> dict[str, float]:
    """Same as _aggregate_chunks_to_docs_max but for the (sf, s, co, t) tuple form."""
    per_doc: dict[str, float] = {}
    for sf, s, _co, _t in raw_hits:
        if not sf:
            continue
        did = Path(sf).stem
        if did not in per_doc or s > per_doc[did]:
            per_doc[did] = s
    return per_doc


def _cross_encoder_over_fused(
    *,
    query: str,
    fused: list[tuple[str, float]],
    raw_hits: list[tuple[str, float, int, str]],
    source_root: Path | str | None,
    cross_encoder: CrossEncoderReranker | None,
    top_k: int,
) -> dict[str, float]:
    """Rerank the top-50 of a RRF-fused candidate list with a cross-encoder.

    Passage text comes from the original dense hits when present; falls
    back to a source-file read for BM25-only docs in the fused top-50.
    """
    from resonance_lattice.lattice import MaterialisedResult
    from resonance_lattice.reranker import CrossEncoderReranker
    from resonance_lattice.store import SourceContent

    # Text lookup from dense hits (cheap — at most top_k entries).
    text_by_doc: dict[str, tuple[str, int, str]] = {}
    for sf, _s, co, t in raw_hits:
        did = Path(sf).stem if sf else ""
        if did and did not in text_by_doc:
            text_by_doc[did] = (sf, co, t)

    fused_ids = [did for did, _ in fused[:50]]

    ce_items: list[tuple[str, MaterialisedResult]] = []
    for did in fused_ids:
        entry = text_by_doc.get(did)
        if entry is not None:
            sf, co, t = entry
        else:
            # BM25-only candidate: resolve via source_root.
            if source_root is None:
                continue
            sr = Path(source_root)
            p = sr / f"{did}.md"
            if not p.exists():
                p = sr / f"{did}.txt"
            if not p.exists():
                continue
            try:
                t = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            sf, co = str(p), 0

        content = SourceContent(
            source_id=did,
            summary=t[:200],
            full_text=t,
            metadata={"source_file": sf, "char_offset": co},
        )
        mat = MaterialisedResult(
            source_id=did, score=0.0, raw_score=None,
            band_scores=None, content=content,
        )
        ce_items.append((did, mat))

    # Lazy-init cross-encoder if not injected.
    if cross_encoder is None:
        cross_encoder = CrossEncoderReranker()

    reranked_results, _ms = cross_encoder.rerank(
        query,
        [m for _, m in ce_items],
        top_k=top_k,
        skip_margin=0.0,
        source_root=source_root,
    )
    return {r.source_id: float(r.score) for r in reranked_results}
