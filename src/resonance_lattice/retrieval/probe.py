# SPDX-License-Identifier: BUSL-1.1
"""Build-time retrieval-mode probe — measure each mode on a held-out
query set, pick the winner, return a config for `__retrieval_config__`.

Part of Horizon 1 board item 236c. `rlat build --probe-qrels <path>
--probe-queries <path>` runs this probe after knowledge model assembly and
writes the winning mode into the knowledge model's `__retrieval_config__`
entry. `rlat search --retrieval-mode auto` then reads that back at
query time so each knowledge model ships with its measured winning mode.

Design goals:
- **Library-only deps.** Probe ships as part of the core; no
  dependence on `beir` / `pytrec_eval` (those are bench-only deps).
  nDCG@10 is computed inline from the standard definition.
- **Single source of truth.** Probe runs every query through
  `resonance_lattice.retrieval.modes.retrieve` — exactly the same
  dispatch the shipping `rlat search` will use. No drift between
  probe measurement and production retrieval.
- **Gracefully skips fused modes** when BM25 sidecar isn't supplied.
  `rlat build --probe-modes field_only,plus_cross_encoder` is a valid
  probe on a knowledge model without BM25.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from resonance_lattice.retrieval.bm25 import BM25Query
from resonance_lattice.retrieval.modes import VALID_MODES, retrieve

if TYPE_CHECKING:
    from resonance_lattice.lattice import Lattice
    from resonance_lattice.reranker import CrossEncoderReranker


def ndcg_at_k(
    ranking: list[str],
    qrels_for_query: dict[str, int],
    k: int = 10,
) -> float:
    """Standard nDCG@k with graded-relevance qrels.

    ranking: list of doc_ids ordered by retrieved score, descending.
    qrels_for_query: {doc_id: relevance_grade} (0, 1, 2, ...).
    k: truncation depth.

    Matches the `pytrec_eval` / BEIR evaluator's output to within
    floating-point noise on standard qrels shapes.
    """
    if not ranking or not qrels_for_query:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(ranking[:k], start=1):
        rel = qrels_for_query.get(doc_id, 0)
        if rel > 0:
            # Standard exponential-gain formulation used in BEIR:
            # gain = (2**rel - 1); discount = log2(rank + 1).
            dcg += (2 ** rel - 1) / math.log2(rank + 1)

    # Ideal DCG: sort qrels by relevance descending, take top-k.
    ideal_rels = sorted(qrels_for_query.values(), reverse=True)[:k]
    idcg = 0.0
    for rank, rel in enumerate(ideal_rels, start=1):
        if rel > 0:
            idcg += (2 ** rel - 1) / math.log2(rank + 1)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def load_qrels_tsv(path: Path | str) -> dict[str, dict[str, int]]:
    """Load qrels from BEIR's qrels/test.tsv format."""
    qrels: dict[str, dict[str, int]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row["query-id"]
            did = row["corpus-id"]
            score = int(row["score"])
            qrels.setdefault(qid, {})[did] = score
    return qrels


def load_queries_jsonl(
    path: Path | str,
    qids: set[str] | None = None,
) -> dict[str, str]:
    """Load queries from BEIR's queries.jsonl format.

    If `qids` is provided, keep only queries whose id appears in the
    set (matches the test-qrels subset).
    """
    out: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("_id") or rec.get("id")
            text = rec.get("text") or ""
            if qid and (qids is None or qid in qids):
                out[qid] = text
    return out


@dataclass
class ProbeResult:
    """Outcome of probing retrieval modes against a held-out query set.

    Written to the knowledge model as `__retrieval_config__` JSON.
    """
    default_mode: str  # the winning mode, by avg nDCG@10
    scores: dict[str, float] = field(default_factory=dict)  # {mode: ndcg@10}
    reranker_model: str | None = None  # set by 238 reranker routing
    n_queries: int = 0
    modes_probed: list[str] = field(default_factory=list)
    schema_version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


_CE_USING_MODES = frozenset(
    {"plus_cross_encoder", "plus_cross_encoder_expanded", "plus_full_stack"}
)


def probe_modes(
    lattice: Lattice,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    *,
    modes: list[str] | None = None,
    source_root: Path | str | None = None,
    bm25_path: Path | str | None = None,
    cross_encoder: CrossEncoderReranker | None = None,
    reranker_candidates: list[str] | None = None,
    top_k: int = 100,
    k_for_ndcg: int = 10,
) -> ProbeResult:
    """Run every mode in `modes` against the probe queries, return a
    ProbeResult naming the (mode, reranker) combo with the highest avg
    nDCG@`k_for_ndcg`.

    If `bm25_path` is None, fused modes (plus_rrf, plus_full_stack,
    bm25_only) are skipped automatically with a note in the result.

    Args:
        lattice: loaded Lattice instance.
        queries: {qid: query_text}. Usually the test subset of
            queries.jsonl filtered by qrels-present qids.
        qrels: {qid: {doc_id: relevance_grade}}.
        modes: which modes to probe. Defaults to all 7 VALID_MODES
            when bm25 is available, else the 3 dense modes.
        source_root: directory with per-doc text files. Required for
            plus_hybrid and plus_full_stack (when BM25-only candidates
            need text resolution).
        bm25_path: path to a BM25 sidecar index.
        cross_encoder: preloaded CrossEncoderReranker for plus_full_stack
            (lazy-init if None). Overridden per-candidate when
            `reranker_candidates` is set.
        reranker_candidates: list of HuggingFace model IDs to test for
            CE-using modes (board item 238). When set, the probe runs
            each (mode × reranker) combo and records the global winner
            in ProbeResult.reranker_model. When None, the default
            reranker is used and reranker_model stays None — that's the
            236c behavior.
        top_k: retrieval cutoff.
        k_for_ndcg: nDCG truncation depth.

    Returns:
        ProbeResult with default_mode set to the argmax of the probed
        modes' avg nDCG@k. When reranker_candidates is set,
        reranker_model is the model ID that won alongside default_mode.
    """
    bm25: BM25Query | None = None
    if bm25_path is not None and Path(bm25_path).is_file():
        bm25 = BM25Query.load(bm25_path)

    # Default mode set: dense trio if no BM25, full 7 if BM25 loaded.
    if modes is None:
        if bm25 is None:
            modes = ["field_only", "plus_cross_encoder",
                     "plus_cross_encoder_expanded"]
        else:
            modes = list(VALID_MODES)

    # Filter out fused modes when BM25 missing, rather than erroring.
    if bm25 is None:
        filtered = [
            m for m in modes
            if m not in ("bm25_only", "plus_rrf", "plus_full_stack")
        ]
        modes = filtered

    unknown = [m for m in modes if m not in VALID_MODES]
    if unknown:
        raise ValueError(
            f"Unknown probe mode(s): {unknown}. "
            f"Valid: {', '.join(VALID_MODES)}"
        )

    # `None` sentinel = use the lattice/default reranker. Real model IDs
    # cause a per-combo CE swap. Empty list collapses to None.
    candidates: list[str | None]
    if not reranker_candidates:
        candidates = [None]
    else:
        candidates = [r for r in reranker_candidates if r and r.strip()] or [None]

    # Pick the CE device once up front. CrossEncoderReranker defaults to
    # device="cpu"; on a CUDA box (runpod A100) that would stall the
    # probe at ~0% GPU utilization while shredding CPU for hours. Detect
    # CUDA here and pass it explicitly so every per-combo CE we build
    # below lands on the GPU. OpenVINO auto-attach still takes priority
    # if the Intel IR is available — the `device` kwarg is only used by
    # the torch fallback path.
    ce_device = "cpu"
    try:
        import torch  # Optional at import time; only needed for CUDA check.
        if torch.cuda.is_available():
            ce_device = "cuda"
    except ImportError:
        pass

    # detailed_scores[(mode, reranker_or_None)] = avg nDCG.
    # mode_scores[mode] = best score across rerankers tested for that mode.
    detailed_scores: dict[tuple[str, str | None], float] = {}
    mode_scores: dict[str, float] = {}

    # 238: stash the lattice's existing _cross_encoder (if any) so we
    # can restore it after probing. Setting `lattice._cross_encoder`
    # directly is how we route enriched_query to a non-default reranker
    # — it consults the attribute before lazy-instantiating.
    saved_ce = getattr(lattice, "_cross_encoder", None)
    try:
        for mode in modes:
            # Non-CE modes are reranker-invariant — score once with the
            # lattice's existing CE (or default) and reuse for the
            # combos table so reporting stays consistent.
            mode_candidates: list[str | None]
            if mode in _CE_USING_MODES:
                mode_candidates = candidates
            else:
                mode_candidates = [None]

            for reranker in mode_candidates:
                local_ce: CrossEncoderReranker | None = cross_encoder
                if mode in _CE_USING_MODES and reranker is not None:
                    from resonance_lattice.reranker import (
                        CrossEncoderReranker,
                    )
                    local_ce = CrossEncoderReranker(
                        model_name=reranker, device=ce_device,
                    )
                    lattice._cross_encoder = local_ce
                elif mode in _CE_USING_MODES and reranker is None:
                    # Default-reranker probe path: if nothing's on the
                    # lattice yet, lazy-instantiate one on the right
                    # device. Otherwise keep whatever the caller set.
                    if not hasattr(lattice, "_cross_encoder"):
                        from resonance_lattice.reranker import (
                            CrossEncoderReranker,
                        )
                        lattice._cross_encoder = CrossEncoderReranker(
                            device=ce_device,
                        )

                per_query_ndcg: list[float] = []
                for qid, qtext in queries.items():
                    q_qrels = qrels.get(qid)
                    if not q_qrels:
                        continue
                    per_doc = retrieve(
                        lattice,
                        qtext,
                        mode=mode,
                        top_k=top_k,
                        source_root=source_root,
                        bm25=bm25,
                        cross_encoder=local_ce,
                    )
                    ranking = [
                        d for d, _ in sorted(
                            per_doc.items(),
                            key=lambda kv: kv[1], reverse=True,
                        )
                    ]
                    per_query_ndcg.append(
                        ndcg_at_k(ranking, q_qrels, k=k_for_ndcg)
                    )

                score = (
                    sum(per_query_ndcg) / len(per_query_ndcg)
                    if per_query_ndcg else 0.0
                )
                detailed_scores[(mode, reranker)] = score
                if mode not in mode_scores or score > mode_scores[mode]:
                    mode_scores[mode] = score
    finally:
        # Restore prior state. If the lattice had no CE attached
        # before, drop the attribute we created so future code paths
        # lazy-init normally.
        if saved_ce is None:
            if hasattr(lattice, "_cross_encoder"):
                try:
                    delattr(lattice, "_cross_encoder")
                except AttributeError:
                    pass
        else:
            lattice._cross_encoder = saved_ce

    # Winner = argmax over (mode, reranker) combos.
    if not detailed_scores:
        return ProbeResult(
            default_mode="field_only",
            n_queries=len(queries),
            modes_probed=modes,
        )

    (best_mode, best_reranker), _best_score = max(
        detailed_scores.items(), key=lambda kv: kv[1]
    )

    return ProbeResult(
        default_mode=best_mode,
        scores={m: round(v, 5) for m, v in mode_scores.items()},
        reranker_model=best_reranker,
        n_queries=len(queries),
        modes_probed=modes,
    )
