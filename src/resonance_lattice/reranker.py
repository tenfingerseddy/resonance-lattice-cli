# SPDX-License-Identifier: BUSL-1.1
"""Cross-encoder reranker for domain-agnostic result quality.

Scores query-passage pairs using a cross-encoder transformer. Unlike the
existing keyword-overlap reranking, this generalizes across domains because
it reads the full text of both query and passage.

The model is loaded lazily on first use to avoid import-time overhead.

Backend selection (on first use):
    1. `openvino_ir_dir` / `openvino_device` constructor args — explicit.
    2. Auto-detect: if OpenVINO is installed AND RLAT_OPENVINO is not `off`
       AND an Arc/NPU device is visible, export-or-reuse an OpenVINO IR in
       the standard cache and run there. ~40× torch CPU on Intel Arc iGPU
       (same ratio measured for the main encoder — see bench_encoder_throughput).
    3. Fallback: sentence_transformers.CrossEncoder on the given `device`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from resonance_lattice.lattice import MaterialisedResult

logger = logging.getLogger(__name__)

# BGE-reranker-v2-m3 (2024, BAAI) — 568M params, multilingual, 8k context.
# Consistently top-tier on MTEB rerank benchmarks and has a stable
# sentence_transformers.CrossEncoder interface. Replaces the 2023-vintage
# mxbai-rerank-base-v1 default (~184M params) as part of the v1.0.0
# semantic-layer SOTA push. OpenVINO export works unchanged — same
# num_labels=1 sequence-classification shape.
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class _PassageExpander:
    """Expand chunk text to its natural structural boundary using the
    external store's source files at query time.

    This is the "lossless store enables SOTA reranking" mechanism:
    cross-encoders score (query, passage) pairs, and were trained on
    passage-length inputs (~200-400 tokens). When the field returns
    sub-passage chunks, the cross-encoder's input is a truncated view
    of the candidate — it underperforms its training distribution
    and regresses vs dense retrieval (seen in April-17 phase0 `reranked`
    mode on NFCorpus/FiQA/SciDocs).

    With the external store + A3 char_offset metadata, we can rehydrate
    each candidate to the full enclosing section / function / paragraph
    BEFORE the cross-encoder sees it — a strict improvement on input
    quality without model changes.

    A per-file read cache amortizes the cost across a top-K rerank pass
    (20-100 candidates often touch < 50 unique files on BEIR corpora).
    Fail-soft: any I/O error or missing metadata returns chunk text.
    """

    def __init__(self, source_root, expand_mode: str) -> None:
        self._root = Path(source_root).resolve() if source_root else None
        self._mode = expand_mode
        self._file_cache: dict[Path, str] = {}

    def expand(self, materialised, chunk_text: str) -> str:
        if self._root is None or not chunk_text:
            return chunk_text
        if materialised is None or materialised.content is None:
            return chunk_text
        meta = materialised.content.metadata or {}
        source_file = meta.get("source_file") or ""
        if not source_file:
            return chunk_text
        char_offset = int(meta.get("char_offset") or 0)

        src_path = self._resolve(source_file)
        if src_path is None:
            return chunk_text

        source_text = self._read(src_path)
        if not source_text:
            return chunk_text

        try:
            from resonance_lattice.retrieval import expand_chunk
            expanded = expand_chunk(
                chunk_text,
                char_offset,
                source_text,
                mode=self._mode,
                source_file=str(src_path),
            )
        except Exception:  # noqa: BLE001 — strictly fail-soft
            return chunk_text

        # expand_chunk is safe: returns the chunk itself on no-match,
        # with expansion_kind="none". Use the expanded text unconditionally.
        return expanded.text or chunk_text

    def _resolve(self, source_file: str):
        p = Path(source_file)
        if p.is_absolute():
            return p if p.exists() else None
        # Manifest paths normalize to forward-slash (A2); try both.
        cand = (self._root / source_file)
        if cand.exists():
            return cand.resolve()
        cand = (self._root / Path(*source_file.replace("\\", "/").split("/")))
        return cand.resolve() if cand.exists() else None

    def _read(self, path: Path) -> str:
        cached = self._file_cache.get(path)
        if cached is not None:
            return cached
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        self._file_cache[path] = text
        return text


def _default_torch_device() -> str:
    """Auto-detect CUDA for the torch fallback. Callers can still pass
    device explicitly to override. Kept conservative: torch may not be
    importable at module-load time in some minimal environments, so
    guard the import.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class CrossEncoderReranker:
    """Rerank retrieval results using a cross-encoder model.

    Args:
        model_name: HuggingFace model ID for the cross-encoder.
        batch_size: Number of query-passage pairs to score per forward pass.
        device: PyTorch device ("cpu", "cuda", etc.) when the torch fallback
            is used. Default None = auto-detect CUDA (fall through to "cpu"
            when unavailable). Ignored when OpenVINO binds — OpenVINO uses
            its own device selector ("CPU" / "GPU" / "NPU" / "AUTO").
        openvino_ir_dir: Directory containing an OpenVINO IR export of the
            cross-encoder model. When set, skips auto-detect. None means
            auto-detect (fall through to torch on any failure).
        openvino_device: OpenVINO device name ("AUTO" / "CPU" / "GPU" /
            "NPU"). Default None → RLAT_OPENVINO_DEVICE env var, else
            `preferred_device()` (GPU > NPU > CPU).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        device: str | None = None,
        *,
        openvino_ir_dir: str | None = None,
        openvino_device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        # Auto-detect CUDA when caller didn't specify. Previous default of
        # "cpu" was a silent trap on CUDA pods — the torch fallback would
        # stall the run at 0% GPU utilization (observed 2026-04-21 on
        # runpod A100 during 239 probe). OpenVINO auto-attach still wins
        # on Intel hosts; this only affects the torch fallback path.
        self.device = device if device is not None else _default_torch_device()
        self.openvino_ir_dir = openvino_ir_dir
        self.openvino_device = openvino_device
        self._model = None
        self._backend: str | None = None  # "openvino" | "torch", set at load

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        # 1. Explicit OpenVINO IR path from the constructor.
        if self.openvino_ir_dir is not None:
            self._load_openvino(self.openvino_ir_dir)
            return

        # 2. Auto-detect. Skips itself when RLAT_OPENVINO=off, OpenVINO isn't
        #    installed, no Arc/NPU device is visible, or export fails. All
        #    failures fall through to torch.
        try:
            from resonance_lattice import reranker_openvino as rov
            ov_dir = rov.auto_get_or_export(self.model_name)
        except Exception as exc:  # noqa: BLE001 — best-effort auto-detect
            logger.debug("auto-openvino-reranker probe failed: %s", exc)
            ov_dir = None
        if ov_dir is not None:
            try:
                self._load_openvino(ov_dir)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "openvino reranker attach failed at %s (%s) — falling back to torch",
                    ov_dir, exc,
                )

        # 3. Torch fallback (original path).
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name, device=self.device)
        self._backend = "torch"

    def _load_openvino(self, ir_dir) -> None:
        """Load the OpenVINO backend. Raises on failure (caller decides fallback)."""
        from resonance_lattice import reranker_openvino as rov

        device = self.openvino_device
        if device is None:
            import os
            device = os.environ.get("RLAT_OPENVINO_DEVICE") or rov.preferred_device() or "AUTO"

        self._model = rov.OpenVinoReranker(ir_dir, device=device)
        self._backend = "openvino"
        logger.info(
            "CrossEncoderReranker using OpenVINO backend (device=%s, model=%s)",
            device, self.model_name,
        )

    def rerank(
        self,
        query: str,
        results: list[MaterialisedResult],
        top_k: int = 20,
        skip_margin: float = 0.15,
        blend_alpha: float = 0.7,
        *,
        source_root: str | Path | None = None,
        expand_mode: str = "natural",
    ) -> tuple[list[MaterialisedResult], float]:
        """Rerank results by cross-encoder relevance score blended with retrieval.

        Blends cross-encoder scores with the prior stage's retrieval scores
        rather than replacing them entirely. This prevents cross-encoder
        errors from completely overriding a good field+keyword ordering,
        which is most valuable on out-of-domain queries.

        When `source_root` is provided (external-mode store available),
        each candidate's text is expanded to its natural structural
        boundary (section / function / paragraph) via B1 before being
        scored by the cross-encoder. This is the lossless-store-enables-
        SOTA lever: cross-encoders are trained on passage-length inputs
        and score them better than sub-passage chunks. Prior art (mxbai
        on BEIR prose) consistently regresses vs dense because the
        reranker sees sub-chunk fragments. Expanded context fixes that
        input mismatch without changing the model.

        Args:
            query: The query text.
            results: Candidate results with content.full_text populated.
            top_k: Number of results to return.
            skip_margin: Skip reranking when the top-1 retrieval score
                exceeds top-2 by this relative margin (saves ~800ms when
                the retrieval ranking is already confident).
            blend_alpha: Weight for cross-encoder score (0-1). The prior
                stage's score gets weight (1 - blend_alpha). Default 0.7
                (cross-encoder dominant, field/keyword contributes).
            source_root: External store's source_root. When provided,
                expansion fires; when None, bare chunk text is used
                (original behaviour, back-compat).
            expand_mode: Passed through to expand_chunk when expansion
                is enabled. "natural" grows to smallest enclosing unit;
                "max" grows further. "off" disables expansion.

        Returns:
            (reranked_results, latency_ms)
        """
        if not results:
            return results, 0.0

        t0 = time.perf_counter()

        # Selective reranking: skip when top-1 clearly dominates top-2
        if len(results) >= 2 and skip_margin > 0:
            s1, s2 = results[0].score, results[1].score
            if s2 > 0 and (s1 - s2) / s2 > skip_margin:
                latency_ms = (time.perf_counter() - t0) * 1000
                return results[:top_k], latency_ms

        self._ensure_loaded()

        # Pre-resolve the expansion context. When source_root is
        # absent, this stays as the no-op fallback (chunk text).
        expander = _PassageExpander(source_root, expand_mode) if (
            source_root is not None and expand_mode != "off"
        ) else None

        # Build query-passage pairs
        pairs = []
        valid_indices = []
        for i, r in enumerate(results):
            text = ""
            if r.content:
                text = r.content.full_text or r.content.summary or ""
            # B1 expansion — swap in the natural-boundary text when the
            # lossless store can resolve the source. Silent fallback on
            # any failure: the reranker still works on chunk text.
            if text and expander is not None:
                text = expander.expand(r, text)
            if text:
                pairs.append((query, text))
                valid_indices.append(i)

        if not pairs:
            return results[:top_k], (time.perf_counter() - t0) * 1000

        # Score all pairs
        scores = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False,
        )

        # Normalize cross-encoder scores to [0,1] for blending
        ce_scores = [float(s) for s in scores]
        ce_min = min(ce_scores) if ce_scores else 0.0
        ce_max = max(ce_scores) if ce_scores else 1.0
        ce_range = ce_max - ce_min if ce_max > ce_min else 1.0

        # Blend cross-encoder with prior stage score
        reranked = []
        for idx, ce_score in zip(valid_indices, ce_scores):
            r = results[idx]
            ce_norm = (ce_score - ce_min) / ce_range
            prior_score = r.score  # Already [0,1] from _lexical_rerank
            blended = blend_alpha * ce_norm + (1.0 - blend_alpha) * prior_score
            reranked.append(replace(
                r,
                score=blended,
                raw_score=r.raw_score if r.raw_score is not None else r.score,
            ))

        # Add back results without text (keep at bottom)
        seen = set(valid_indices)
        for i, r in enumerate(results):
            if i not in seen:
                reranked.append(replace(r, score=-1e9))

        reranked.sort(key=lambda r: r.score, reverse=True)
        latency_ms = (time.perf_counter() - t0) * 1000

        return reranked[:top_k], latency_ms
