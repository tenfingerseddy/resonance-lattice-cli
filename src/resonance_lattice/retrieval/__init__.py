# SPDX-License-Identifier: BUSL-1.1
"""Retrieval layer — sits between the field (router) and the reader
(synthesiser). Each module here is a pure function over `(chunk, source)`:

- `expand` (B1): grow a retrieved chunk to a natural boundary so the
  reader sees enough surrounding context to resolve references,
  pronouns, definitions, etc.
- `expand.rechunk` (B2): serve the chunk at a named granularity
  (sentence / passage / section) chosen at query time, independent of
  how the chunker sliced the corpus.
- `lexical` (B3): run a second-pass lexical query (ripgrep) inside the
  field-retrieved neighbourhood and blend the signal with dense scores.

The field tells us *which region of the corpus is relevant*. These
modules decide *what bytes we actually return to the reader* given that
region. They must never mutate the field or the registry — the
semantic layer's integrity depends on that separation.
"""

from resonance_lattice.retrieval.bm25 import (
    BM25Query,
    build_bm25_index,
    reciprocal_rank_fusion,
)
from resonance_lattice.retrieval.expand import (
    ExpandedChunk,
    Granularity,
    expand_chunk,
    rechunk,
)
from resonance_lattice.retrieval.lexical import ScoredHit, lexical_rerank
from resonance_lattice.retrieval.modes import VALID_MODES, retrieve
from resonance_lattice.retrieval.probe import (
    ProbeResult,
    load_qrels_tsv,
    load_queries_jsonl,
    ndcg_at_k,
    probe_modes,
)

__all__ = [
    "BM25Query",
    "ExpandedChunk",
    "Granularity",
    "ProbeResult",
    "ScoredHit",
    "VALID_MODES",
    "build_bm25_index",
    "expand_chunk",
    "lexical_rerank",
    "load_qrels_tsv",
    "load_queries_jsonl",
    "ndcg_at_k",
    "probe_modes",
    "rechunk",
    "reciprocal_rank_fusion",
    "retrieve",
]
