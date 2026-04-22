# SPDX-License-Identifier: BUSL-1.1
"""OpenVINO backbone for encoder inference — CPU, Intel Arc iGPU, or NPU.

Replaces the PyTorch backbone forward pass with an OpenVINO inference request,
hitting the Intel Arc GPU (or integrated GPU, or NPU) when available. Measured
at 40× stock torch CPU on Intel Arc 140V 16GB iGPU running E5-large-v2
(benchmarks/results/encoder_throughput/latest.json).

Usage (mirrors the ONNX pattern):
    encoder = Encoder.from_backbone("intfloat/e5-large-v2")
    ov_path = export_backbone_openvino(encoder, "backbone_ov/")
    attach_openvino_backbone(encoder, ov_path, device="GPU")
    # Now encode_texts uses OpenVINO on the named device.

Device selection:
    - "AUTO"  — OpenVINO picks (prefers GPU > CPU)
    - "CPU"   — Intel CPU
    - "GPU"   — the default GPU (on this machine, Intel Arc 140V)
    - "NPU"   — Intel AI Boost NPU (requires static-shape export; see note)

NPU caveat: the default dynamic-shape OpenVINO export does NOT compile on NPU
(dynamic `seq_len` with upper bound `INT64_MAX` hits the VPUX compiler). For
NPU use, call `export_backbone_openvino(..., static_seq_len=512)` which issues
`model.reshape({...})` with fixed dims before save.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _backbone_slug(model_name: str) -> str:
    """Filesystem-safe slug for a HuggingFace model id.

    intfloat/e5-large-v2 → intfloat__e5-large-v2
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_name).strip("_") or "model"


def default_cache_dir(model_name: str, static_seq_len: int | None = None) -> Path:
    """Return the preferred cache directory for an OpenVINO IR export.

    Lookup/write order:
      1. `$RLAT_OPENVINO_CACHE_DIR/<slug>` if env set
      2. `<project>/.cache/rlat/openvino/<slug>` — project-local (gitignored).
         Project root is CLAUDE_PROJECT_DIR or cwd.
      3. `~/.cache/rlat/openvino/<slug>` — user-wide fallback

    `default_cache_dir` is a *resolver* — it picks the dir without reading or
    writing anything. Callers should test existence and export if missing.
    """
    slug = _backbone_slug(model_name)
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
    # Neither exists — prefer project-local for the new export (keeps things
    # visible, matches the existing `.cache/` convention).
    return project_cache


def _ir_present(path: Path) -> bool:
    return path.is_dir() and (path / "openvino_model.xml").exists()


def auto_enabled() -> bool:
    """Whether auto-detect should run. `RLAT_OPENVINO=off|0|false|no` disables."""
    v = (os.environ.get("RLAT_OPENVINO") or "").strip().lower()
    return v not in {"off", "0", "false", "no"}


def auto_get_or_export(encoder, *, static_seq_len: int | None = None) -> Path | None:
    """Find (or produce) an OpenVINO IR dir for this encoder's backbone.

    Returns the IR directory on success, or None on any failure (missing deps,
    no Arc/NPU device, encoder without a backbone name, export error). Callers
    MUST handle None as "skip OpenVINO, fall through to the existing path".

    First call with no cached IR triggers a one-time export (~15-30s). Later
    calls reuse the cached IR.
    """
    if not auto_enabled():
        return None
    ok, why = is_available()
    if not ok:
        logger.debug("auto-openvino skipped: %s", why)
        return None
    # Require at least one accelerator device. Pure CPU-only OpenVINO is
    # on-par with torch CPU on this hardware (per bench_encoder_throughput.py);
    # the auto path exists specifically to opportunistically light up Arc/NPU.
    devs = available_devices()
    if not ({"GPU", "NPU"} & set(devs)):
        logger.debug("auto-openvino skipped: no GPU/NPU device visible (devices=%s)", devs)
        return None
    model_name = getattr(getattr(encoder, "config", None), "backbone", None)
    if not model_name:
        logger.debug("auto-openvino skipped: encoder has no backbone name")
        return None

    target = default_cache_dir(model_name, static_seq_len=static_seq_len)
    if _ir_present(target):
        return target
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        export_backbone_openvino(encoder, target, static_seq_len=static_seq_len)
        if _ir_present(target):
            return target
        logger.warning("auto-openvino: export completed but IR missing at %s", target)
        return None
    except Exception as exc:  # noqa: BLE001 — best-effort auto-detect
        logger.warning("auto-openvino export failed: %s", exc)
        return None


def is_available() -> tuple[bool, str]:
    """Returns (available, diagnostic). Empty diagnostic when available."""
    try:
        import openvino  # noqa: F401
    except ImportError as exc:
        return False, f"openvino not installed: {exc}"
    try:
        from optimum.intel import OVModelForFeatureExtraction  # noqa: F401
    except ImportError as exc:
        return False, f"optimum-intel not installed: {exc}"
    return True, ""


def available_devices() -> list[str]:
    """Return the list of OpenVINO-visible devices on this host.

    Empty list when OpenVINO is not installed. Typical values: ['CPU'],
    ['CPU', 'GPU'], ['CPU', 'GPU', 'NPU'].
    """
    try:
        import openvino as ov
        return list(ov.Core().available_devices)
    except Exception:
        return []


def preferred_device() -> str | None:
    """Pick the best-performing device for encoder inference.

    Order: GPU (Intel Arc / iGPU) > NPU > CPU. Returns None when OpenVINO
    is not installed. Callers should also honour RLAT_OPENVINO_DEVICE env var.
    """
    devices = available_devices()
    if not devices:
        return None
    for choice in ("GPU", "NPU", "CPU"):
        if choice in devices:
            return choice
    return devices[0]


def export_backbone_openvino(
    encoder,
    output_dir: str | Path,
    *,
    static_seq_len: int | None = None,
    static_batch_size: int = 1,
) -> Path:
    """Export the encoder's backbone to OpenVINO IR format.

    Args:
        encoder: An Encoder instance with a loaded backbone.
        output_dir: Directory to save the OpenVINO model into.
        static_seq_len: If set, reshape the model to a fixed sequence length.
            Required for NPU deployment (NPU compile rejects dynamic shapes;
            even batch_size=-1 is rejected as "negative shape dim bound"). CPU
            and GPU work fine with dynamic shapes and should leave this as None.
        static_batch_size: Only used when static_seq_len is set. Defaults to 1
            (works on every target). Bigger batches improve GPU/CPU throughput
            but must match what the caller feeds at inference time.

    Returns:
        Path to the directory containing openvino_model.xml/.bin + tokenizer.
    """
    ok, why = is_available()
    if not ok:
        raise RuntimeError(why)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = encoder.config.backbone
    if not model_name:
        raise RuntimeError("Encoder has no backbone model name")

    from optimum.intel import OVModelForFeatureExtraction

    model = OVModelForFeatureExtraction.from_pretrained(model_name, export=True)
    model.save_pretrained(output_dir)
    if encoder._tokenizer is not None:
        encoder._tokenizer.save_pretrained(output_dir)
    del model
    import gc
    gc.collect()

    if static_seq_len is not None:
        # NPU compile requires FULLY static shapes. optimum-intel's high-level
        # .reshape() does NOT persist to the saved XML (tested 2026-04-19 on
        # optimum-intel 1.x + openvino 2026.1.0). Use the low-level ov API to
        # reload, reshape, and save to a sibling path, then atomically replace.
        # Saving over the open .bin in-place fails on Windows with "Can't open
        # bin file" — go via a _static-shaped_ sibling first.
        import openvino as ov
        xml = output_dir / "openvino_model.xml"
        bin_path = output_dir / "openvino_model.bin"
        core = ov.Core()
        ov_model = core.read_model(str(xml))
        ov_model.reshape({inp.any_name: [static_batch_size, static_seq_len] for inp in ov_model.inputs})
        tmp_xml = output_dir / "openvino_model.static.xml"
        tmp_bin = output_dir / "openvino_model.static.bin"
        ov.save_model(ov_model, str(tmp_xml))
        del ov_model
        gc.collect()
        # Replace originals with the reshaped copies.
        if xml.exists():
            xml.unlink()
        if bin_path.exists():
            bin_path.unlink()
        tmp_xml.replace(xml)
        tmp_bin.replace(bin_path)

    logger.info("Exported backbone to OpenVINO IR at %s (static_seq_len=%s, batch=%s)",
                output_dir, static_seq_len, static_batch_size if static_seq_len else "dynamic")
    return output_dir


class _StaticOutput:
    """Minimal HF-compatible output holder (mirrors `BaseModelOutput`)."""
    __slots__ = ("last_hidden_state",)

    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


def _infer_static_shape(model_dir: str | Path) -> tuple[int | None, int | None]:
    """Peek at the on-disk IR to detect persisted static shapes.

    Returns (batch_size, seq_len) when the IR was reshaped to static dims,
    else (None, None). Used to decide whether the attach path needs the
    static-shape adapter (per-sample iteration + max_length padding) vs the
    fast dynamic path.
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


class OpenVinoBackbone:
    """Drop-in replacement for the PyTorch backbone using OpenVINO.

    Two inference paths:
      - **Dynamic** — IR has `[?, ?]` input shapes (CPU/GPU default). One
        forward pass handles the whole batch; output is the raw HF model
        return value (`.last_hidden_state` already a torch tensor).
      - **Static** — IR has `[B_static, S_static]` (NPU requires this;
        CPU/GPU accept it). The encoder feeds variable batches at variable
        token lengths, so this adapter: (1) re-tokenises via padding to
        `max_length=S_static` and truncation, then (2) iterates the batch
        in chunks of `B_static`, appending zero-pad rows when the final
        chunk is short. Outputs are concatenated back to the caller's
        original batch size.

    `attach_openvino_backbone` auto-detects which path to use by peeking at
    the persisted IR shapes, so callers don't need to know the difference.
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "AUTO",
        *,
        static_batch_size: int | None = None,
        static_seq_len: int | None = None,
    ) -> None:
        from optimum.intel import OVModelForFeatureExtraction

        if static_seq_len is not None:
            # Static path: load without compile so we can reshape the in-memory
            # ov.Model (optimum's loader re-dynamizes on load — reshape undoes
            # that), then target the device and compile.
            self._model = OVModelForFeatureExtraction.from_pretrained(str(model_dir), compile=False)
            batch = int(static_batch_size or 1)
            self._model.reshape(batch_size=batch, sequence_length=int(static_seq_len))
            self._model.to(device)
            self._model.compile()
            self._static_batch = batch
            self._static_seq = int(static_seq_len)
            # A tokenizer is needed to enforce max_length padding on inference.
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        else:
            # Dynamic path — as before. OV model auto-compiles on the device.
            self._model = OVModelForFeatureExtraction.from_pretrained(str(model_dir), device=device)
            self._static_batch = None
            self._static_seq = None
            self._tokenizer = None
        self._device_label = device
        logger.info(
            "OpenVINO backbone loaded from %s on device=%s (static=%sx%s)",
            model_dir, device,
            self._static_batch or "dynamic", self._static_seq or "dynamic",
        )

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ):
        if self._static_seq is None:
            # Dynamic path: one forward pass.
            return self._model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # Static path: the encoder tokenised to the longest-in-batch length S.
        # The OV backbone expects exactly static_seq (Ss) and static_batch (Bs).
        # Pad/truncate S → Ss, chunk B over Bs, run each chunk, then trim the
        # output back to S so the encoder's mean-pool sees a matching shape.
        import torch as _torch

        B, S = int(input_ids.shape[0]), int(input_ids.shape[1])
        Ss = self._static_seq
        effective_S = min(S, Ss)  # if encoder batch was longer than Ss, we truncated
        if S > Ss:
            input_ids = input_ids[:, :Ss]
            attention_mask = attention_mask[:, :Ss]
            token_type_ids = kwargs.get("token_type_ids")
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids[:, :Ss]
        elif S < Ss:
            pad_w = Ss - S
            input_ids = _torch.nn.functional.pad(input_ids, (0, pad_w), value=0)
            attention_mask = _torch.nn.functional.pad(attention_mask, (0, pad_w), value=0)
            token_type_ids = kwargs.get("token_type_ids")
            if token_type_ids is not None:
                kwargs["token_type_ids"] = _torch.nn.functional.pad(token_type_ids, (0, pad_w), value=0)

        Bs = self._static_batch
        outs = []
        for start in range(0, B, Bs):
            end = min(start + Bs, B)
            rows = end - start
            sub_ids = input_ids[start:end]
            sub_mask = attention_mask[start:end]
            sub_kwargs = {k: v[start:end] for k, v in kwargs.items() if hasattr(v, "__getitem__")}
            if rows < Bs:
                pad_rows = Bs - rows
                sub_ids = _torch.nn.functional.pad(sub_ids, (0, 0, 0, pad_rows), value=0)
                sub_mask = _torch.nn.functional.pad(sub_mask, (0, 0, 0, pad_rows), value=0)
                sub_kwargs = {
                    k: _torch.nn.functional.pad(v, (0, 0, 0, pad_rows), value=0)
                    for k, v in sub_kwargs.items()
                }
            out = self._model(input_ids=sub_ids, attention_mask=sub_mask, **sub_kwargs)
            hid = out.last_hidden_state  # (Bs, Ss, D)
            outs.append(hid[:rows])  # drop zero-pad rows
        merged = _torch.cat(outs, dim=0)  # (B, Ss, D)
        # Trim seq-dim back to what the caller gave us (effective_S) so the
        # encoder's own attention_mask (shape (B, S, 1)) aligns at pool time.
        merged = merged[:, :effective_S, :]
        return _StaticOutput(last_hidden_state=merged)


def attach_openvino_backbone(
    encoder,
    model_dir: str | Path,
    *,
    device: str = "AUTO",
) -> None:
    """Replace the encoder's PyTorch backbone with an OpenVINO inference request.

    Auto-detects static vs dynamic shapes from the persisted IR. Static IRs
    (required for NPU) are loaded with compile=False + reshape + device + compile,
    and inputs are re-padded / batch-chunked on every call.

    `device` accepts any OpenVINO device name: "CPU", "GPU", "NPU", "AUTO".
    """
    ok, why = is_available()
    if not ok:
        raise RuntimeError(why)

    b, s = _infer_static_shape(model_dir)
    ov_backbone = OpenVinoBackbone(
        model_dir, device=device,
        static_batch_size=b, static_seq_len=s,
    )
    # Bypass nn.Module.__setattr__ which rejects non-Module values.
    object.__setattr__(encoder, "_backbone", ov_backbone)
    # Keep _device as CPU — the tokenizer .to(cpu) call still works; OV handles
    # the actual device placement internally.
    object.__setattr__(encoder, "_device", torch.device("cpu"))
    logger.info(
        "Encoder backbone replaced with OpenVINO on device=%s (static=%sx%s)",
        device, b or "dynamic", s or "dynamic",
    )
