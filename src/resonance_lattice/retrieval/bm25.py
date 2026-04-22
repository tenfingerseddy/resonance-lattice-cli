# SPDX-License-Identifier: BUSL-1.1
"""BM25 sidecar index for hybrid retrieval.

Promoted from `benchmarks/bm25_index.py` (Horizon 1 board item 236a) so
the 4-mode retrieval stack can live in the library instead of the bench
harness. API unchanged from the bench version — same `build_bm25_index`,
`BM25Query`, `reciprocal_rank_fusion` contracts — so the bench can
import from here without behavior drift.

Design:

- **Doc-level indexing.** BEIR qrels score at the doc level and our
  chunk→doc aggregation takes the max. Indexing at the doc level skips
  that aggregation step and stays faithful to how BEIR evaluates. Each
  source file under `source_root` becomes one document; the stem
  becomes the doc_id.

- **Sidecar file.** Index is stored as `<knowledge model>.bm25` next to the
  `.rlat` file — same portability story, independent format, no `.rlat`
  header changes required.

- **bm25s backend.** Pure-Python, ~10× faster than rank_bm25, matches
  the "no heavy deps" bias of the rest of the codebase.

Usage:
    from resonance_lattice.retrieval.bm25 import (
        build_bm25_index, BM25Query, reciprocal_rank_fusion
    )

    build_bm25_index(
        Path("benchmark_data/nfcorpus_docs"),
        Path("bench_out/nfcorpus_new_arch.rlat.bm25"),
    )

    bm25 = BM25Query.load("bench_out/nfcorpus_new_arch.rlat.bm25")
    doc_ids, scores = bm25.query("breast cancer risk", top_k=100)
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


def _read_corpus_files(source_root: Path) -> tuple[list[str], list[str]]:
    paths = sorted(
        p for p in source_root.rglob("*")
        if p.is_file() and p.suffix.lower() in (".md", ".txt", ".markdown", ".mdx")
    )
    doc_ids: list[str] = []
    doc_texts: list[str] = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        doc_ids.append(p.stem)
        doc_texts.append(text)
    return doc_ids, doc_texts


def build_bm25_index(
    source_root: Path | str,
    output_path: Path | str,
    *,
    stopwords: str = "en",
) -> dict:
    """Build a BM25 index from source files, save alongside knowledge model."""
    import bm25s

    source_root = Path(source_root).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    doc_ids, doc_texts = _read_corpus_files(source_root)

    if not doc_texts:
        raise RuntimeError(f"No .md/.txt files found under {source_root}")

    tokens = bm25s.tokenize(doc_texts, stopwords=stopwords, show_progress=False)

    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)

    index_dir = output_path.with_suffix(".bm25s_dir")
    if index_dir.exists():
        import shutil
        shutil.rmtree(index_dir)
    retriever.save(str(index_dir))
    meta_path = output_path.with_suffix(".bm25.meta.json")
    meta = {
        "schema_version": 1,
        "source_root": str(source_root),
        "doc_count": len(doc_ids),
        "doc_ids": doc_ids,
        "stopwords": stopwords,
        "build_time_s": round(time.perf_counter() - t0, 2),
        "tokenizer": "bm25s.tokenize",
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    output_path.write_text(
        json.dumps({"index_dir": str(index_dir.name), "meta": meta_path.name}),
        encoding="utf-8",
    )
    return meta


@dataclass
class BM25Query:
    """Loaded BM25 index with doc_id mapping."""
    retriever: object  # bm25s.BM25, typed as object so import stays lazy
    doc_ids: list[str]
    stopwords: str = "en"

    @classmethod
    def load(cls, index_path: Path | str) -> BM25Query:
        import bm25s

        index_path = Path(index_path).resolve()
        stub = json.loads(index_path.read_text(encoding="utf-8"))
        index_dir = index_path.parent / stub["index_dir"]
        meta_path = index_path.parent / stub["meta"]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        retriever = bm25s.BM25.load(str(index_dir))
        return cls(
            retriever=retriever,
            doc_ids=meta["doc_ids"],
            stopwords=meta.get("stopwords", "en"),
        )

    def query(self, text: str, top_k: int = 100) -> tuple[list[str], list[float]]:
        """Return (doc_ids, scores) for the top-K docs, descending by BM25 score."""
        import bm25s

        tokens = bm25s.tokenize([text], stopwords=self.stopwords, show_progress=False)
        doc_indices, scores = self.retriever.retrieve(
            tokens, k=min(top_k, len(self.doc_ids)), show_progress=False,
        )
        idx_row = doc_indices[0].tolist()
        score_row = scores[0].tolist()
        return (
            [self.doc_ids[i] for i in idx_row],
            [float(s) for s in score_row],
        )


def reciprocal_rank_fusion(
    rankings: Iterable[list[str]],
    *,
    k: int = 60,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """Fuse multiple rankings via RRF. Returns [(doc_id, score), ...].

    RRF is rank-based (1/(k+rank)) so it's robust to score scale
    differences — dense cosine, BM25 weights, and cross-encoder logits
    fuse without normalisation calibration. k=60 is the standard
    hyperparameter from the 2009 paper.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[:top_k]
