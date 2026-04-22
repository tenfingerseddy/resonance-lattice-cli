# SPDX-License-Identifier: BUSL-1.1
"""OpenVINO cross-encoder reranker — CPU, Intel Arc iGPU, or NPU.

Mirrors encoder_openvino.py for the cross-encoder model. Cross-encoders are
sequence-classification models (one scalar relevance logit per (query,
passage) pair), so this uses OVModelForSequenceClassification rather than
OVModelForFeatureExtraction.

Default reranker: mixedbread-ai/mxbai-rerank-base-v1 (num_labels=1, sigmoid
activation). The wrapper exposes a .predict(pairs, batch_size=..., ...)
interface compatible with sentence_transformers.CrossEncoder so callers in
reranker.py don't care which backend is running.

Usage:
    from resonance_lattice.reranker_openvino import (
        default_cache_dir, auto_get_or_export, OpenVinoReranker,
    )
    ov_dir = auto_get_or_export("mixedbread-ai/mxbai-rerank-base-v1")
    if ov_dir is not None:
        scorer = OpenVinoReranker(ov_dir, device="GPU")
        scores = scorer.predict([(q, p) for p in passages])

Device selection (same semantics as encoder_openvino):
    - "AUTO"  — OpenVINO picks (prefers GPU > CPU)
    - "CPU"   — Intel CPU
    - "GPU"   — default GPU (Intel Arc 140V on this host)
    - "NPU"   — Intel AI Boost NPU; requires static-shape export
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _model_slug(model_name: str) -> str:
    """Filesystem-safe slug: mixedbread-ai/mxbai-rerank-base-v1 → mixedbread-ai__mxbai-rerank-base-v1"""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_name).strip("_") or "model"


def default_cache_dir(model_name: str, static_seq_len: int | None = None) -> Path:
    """Preferred cache directory for an OpenVINO IR export of a reranker model.

    Lookup/write order (mirrors encoder_openvino.default_cache_dir):
      1. `$RLAT_OPENVINO_CACHE_DIR/<slug>` if env set
      2. `<project>/.cache/rlat/openvino/<slug>` — project-local (gitignored)
      3. `~/.cache/rlat/openvino/<slug>` — user-wide fallback
    """
    slug = _model_slug(model_name)
    if static_seq_len:
        slug = f"{slug}__static{static_seq_len}"

    override = os.environ.get("RLAT_OPENVINO_CACHE_DIR")
    if override:
        return Path(override) / slug

    project = Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())
    project_cache = project / ".cache" / "rlat" / "openvino" / slug
    if project_cache.exists():
        return project_cache
    user_cache = Path(os.path.expanduser("~")) / ".cache" / "rlat" / "openvino" / slug
    if user_cache.exists():
        return user_cache
    return project_cache


def _ir_present(path: Path) -> bool:
    return path.is_dir() and (path / "openvino_model.xml").exists()


def auto_enabled() -> bool:
    """Whether auto-detect should run. `RLAT_OPENVINO=off|0|false|no` disables."""
    v = (os.environ.get("RLAT_OPENVINO") or "").strip().lower()
    return v not in {"off", "0", "false", "no"}


def is_available() -> tuple[bool, str]:
    """Returns (available, diagnostic). Empty diagnostic when available."""
    try:
        import openvino  # noqa: F401
    except ImportError as exc:
        return False, f"openvino not installed: {exc}"
    try:
        from optimum.intel import OVModelForSequenceClassification  # noqa: F401
    except ImportError as exc:
        return False, f"optimum-intel not installed: {exc}"
    return True, ""


def available_devices() -> list[str]:
    """Return the list of OpenVINO-visible devices on this host."""
    try:
        import openvino as ov
        return list(ov.Core().available_devices)
    except Exception:
        return []


def preferred_device() -> str | None:
    """Pick the best-performing device for reranker inference.

    Order: GPU (Intel Arc / iGPU) > NPU > CPU. Returns None when OpenVINO
    is not installed.
    """
    devices = available_devices()
    if not devices:
        return None
    for choice in ("GPU", "NPU", "CPU"):
        if choice in devices:
            return choice
    return devices[0]


def auto_get_or_export(model_name: str, *, static_seq_len: int | None = None) -> Path | None:
    """Find (or produce) an OpenVINO IR dir for the named reranker model.

    Returns the IR directory on success, or None on any failure. Callers MUST
    handle None as "skip OpenVINO, fall through to the existing torch path".

    First call exports the model (~15-30s); later calls reuse the cached IR.
    """
    if not auto_enabled():
        return None
    ok, why = is_available()
    if not ok:
        logger.debug("auto-openvino-reranker skipped: %s", why)
        return None
    devs = available_devices()
    if not ({"GPU", "NPU"} & set(devs)):
        logger.debug(
            "auto-openvino-reranker skipped: no GPU/NPU device visible (devices=%s)",
            devs,
        )
        return None

    target = default_cache_dir(model_name, static_seq_len=static_seq_len)
    if _ir_present(target):
        return target
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        export_reranker_openvino(model_name, target, static_seq_len=static_seq_len)
        if _ir_present(target):
            return target
        logger.warning("auto-openvino-reranker: export completed but IR missing at %s", target)
        return None
    except Exception as exc:  # noqa: BLE001 — best-effort auto-detect
        logger.warning("auto-openvino-reranker export failed: %s", exc)
        return None


def export_reranker_openvino(
    model_name: str,
    output_dir: str | Path,
    *,
    static_seq_len: int | None = None,
    static_batch_size: int = 1,
) -> Path:
    """Export a cross-encoder model to OpenVINO IR format.

    Args:
        model_name: HuggingFace model id of the cross-encoder (e.g.
            "mixedbread-ai/mxbai-rerank-base-v1").
        output_dir: Directory to save the OpenVINO IR into.
        static_seq_len: If set, reshape to a fixed sequence length. Required
            for NPU; CPU/GPU work fine with dynamic shapes (leave None).
        static_batch_size: Only used when static_seq_len is set. Default 1.

    Returns:
        Path to the directory containing openvino_model.xml/.bin + tokenizer.
    """
    ok, why = is_available()
    if not ok:
        raise RuntimeError(why)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from optimum.intel import OVModelForSequenceClassification
    from transformers import AutoTokenizer

    model = OVModelForSequenceClassification.from_pretrained(model_name, export=True)
    model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    del model, tokenizer
    import gc
    gc.collect()

    if static_seq_len is not None:
        # NPU: fully-static shapes required. Same in-place-write dance as
        # encoder_openvino — replace IR with a reshaped sibling atomically to
        # sidestep Windows "file still open" errors on .bin.
        import openvino as ov
        xml = output_dir / "openvino_model.xml"
        bin_path = output_dir / "openvino_model.bin"
        core = ov.Core()
        ov_model = core.read_model(str(xml))
        ov_model.reshape({
            inp.any_name: [static_batch_size, static_seq_len] for inp in ov_model.inputs
        })
        tmp_xml = output_dir / "openvino_model.static.xml"
        tmp_bin = output_dir / "openvino_model.static.bin"
        ov.save_model(ov_model, str(tmp_xml))
        del ov_model
        gc.collect()
        if xml.exists():
            xml.unlink()
        if bin_path.exists():
            bin_path.unlink()
        tmp_xml.replace(xml)
        tmp_bin.replace(bin_path)

    logger.info(
        "Exported reranker %s to OpenVINO IR at %s (static_seq_len=%s, batch=%s)",
        model_name, output_dir, static_seq_len,
        static_batch_size if static_seq_len else "dynamic",
    )
    return output_dir


class OpenVinoReranker:
    """OpenVINO-backed cross-encoder scorer.

    Exposes a `.predict(pairs, batch_size=..., show_progress_bar=...)` method
    matching sentence_transformers.CrossEncoder so reranker.py can swap
    backends without touching its own predict-call site.

    Output: per-pair relevance score, list[float]. Applies sigmoid when the
    model has num_labels=1 (regression-style cross-encoder like mxbai), else
    returns raw logits (argmax class).
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "AUTO",
        *,
        max_length: int = 512,
        static_batch_size: int | None = None,
        static_seq_len: int | None = None,
    ) -> None:
        from optimum.intel import OVModelForSequenceClassification
        from transformers import AutoTokenizer

        model_dir = Path(model_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        if static_seq_len is not None:
            self._model = OVModelForSequenceClassification.from_pretrained(
                str(model_dir), compile=False,
            )
            batch = int(static_batch_size or 1)
            self._model.reshape(batch_size=batch, sequence_length=int(static_seq_len))
            self._model.to(device)
            self._model.compile()
            self._static_batch = batch
            self._static_seq = int(static_seq_len)
        else:
            self._model = OVModelForSequenceClassification.from_pretrained(
                str(model_dir), device=device,
            )
            self._static_batch = None
            self._static_seq = None

        self._max_length = int(max_length)
        self._device_label = device
        # num_labels == 1 ⇒ regression-style (sigmoid output ∈ [0,1])
        self._num_labels = int(getattr(self._model.config, "num_labels", 1))
        logger.info(
            "OpenVINO reranker loaded from %s on device=%s (static=%sx%s, num_labels=%d)",
            model_dir, device,
            self._static_batch or "dynamic", self._static_seq or "dynamic",
            self._num_labels,
        )

    def predict(
        self,
        sentences: list[tuple[str, str]] | list[list[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,  # noqa: ARG002 — kept for API compat
        **_kwargs,
    ) -> list[float]:
        """Score a list of (query, passage) pairs. Returns a list of floats.

        API matches sentence_transformers.CrossEncoder.predict — callers in
        reranker.py already consume the result as `[float(s) for s in scores]`.

        Static-shape paths pad each batch to `static_batch_size × static_seq_len`
        and strip padded rows from the output before returning.
        """
        import numpy as np

        if not sentences:
            return []

        # Normalise pair shape ([q, p] and (q, p) both accepted).
        pairs = [(p[0], p[1]) for p in sentences]

        # For the static path, lock tokenizer padding to static_seq_len.
        pad_length = self._static_seq if self._static_seq is not None else self._max_length
        effective_batch = (
            self._static_batch if self._static_batch is not None else int(batch_size)
        )

        scores: list[float] = []
        for start in range(0, len(pairs), effective_batch):
            chunk = pairs[start:start + effective_batch]
            rows = len(chunk)
            if self._static_batch is not None and rows < self._static_batch:
                # Pad the chunk with dummy pairs — we'll drop the scores.
                chunk = chunk + [("", "")] * (self._static_batch - rows)

            enc = self._tokenizer(
                [q for q, _ in chunk],
                [p for _, p in chunk],
                padding="max_length" if self._static_seq is not None else True,
                truncation=True,
                max_length=pad_length,
                return_tensors="np",
            )
            inputs = {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            }
            if "token_type_ids" in enc:
                inputs["token_type_ids"] = enc["token_type_ids"]

            out = self._model(**inputs)
            logits = out.logits if hasattr(out, "logits") else out[0]
            arr = np.asarray(logits)

            if self._num_labels == 1:
                # Regression-style: sigmoid → [0,1]
                arr = 1.0 / (1.0 + np.exp(-arr))
                batch_scores = arr.squeeze(-1).tolist()
            elif self._num_labels == 2:
                # Binary classifier: take the positive-class softmax probability.
                ex = np.exp(arr - arr.max(axis=-1, keepdims=True))
                probs = ex / ex.sum(axis=-1, keepdims=True)
                batch_scores = probs[:, 1].tolist()
            else:
                # Multi-class: return the max-logit class index's score.
                batch_scores = arr.max(axis=-1).tolist()

            # Drop padded-row scores when we over-batched for static shape.
            scores.extend(batch_scores[:rows])

        return scores


def attach_openvino_reranker(
    reranker,
    model_dir: str | Path,
    *,
    device: str = "AUTO",
) -> None:
    """Replace a CrossEncoderReranker's internal model with OpenVINO.

    Mirrors attach_openvino_backbone for the encoder: auto-detects static vs
    dynamic shapes, loads, and swaps `reranker._model` with an OpenVinoReranker
    that exposes the same `.predict()` contract.
    """
    ok, why = is_available()
    if not ok:
        raise RuntimeError(why)

    b, s = _infer_static_shape(model_dir)
    ov_scorer = OpenVinoReranker(
        model_dir, device=device,
        static_batch_size=b, static_seq_len=s,
    )
    reranker._model = ov_scorer
    logger.info(
        "Reranker model replaced with OpenVINO on device=%s (static=%sx%s)",
        device, b or "dynamic", s or "dynamic",
    )


def _infer_static_shape(model_dir: str | Path) -> tuple[int | None, int | None]:
    """Peek at the on-disk IR to detect persisted static shapes.

    Returns (batch_size, seq_len) when the IR was reshaped to static dims,
    else (None, None).
    """
    try:
        import openvino as ov
        model = ov.Core().read_model(str(Path(model_dir) / "openvino_model.xml"))
        shapes = {inp.any_name: inp.partial_shape for inp in model.inputs}
        ids = shapes.get("input_ids")
        if ids is None or len(ids) != 2:
            return None, None
        b = ids[0].get_length() if not ids[0].is_dynamic else None
        s = ids[1].get_length() if not ids[1].is_dynamic else None
        return b, s
    except Exception:
        return None, None
