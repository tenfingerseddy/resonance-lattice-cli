# SPDX-License-Identifier: BUSL-1.1
"""ONNX Runtime backbone for faster encoder inference.

Replaces the PyTorch backbone forward pass with an ONNX Runtime session,
giving 2-5x speedup on CPU via graph optimization and operator fusion.

Usage:
    encoder = Encoder.from_backbone("intfloat/e5-large-v2")
    onnx_path = export_backbone_onnx(encoder, "backbone.onnx")
    attach_onnx_backbone(encoder, onnx_path)
    # Now encode_query uses ONNX instead of PyTorch
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Return True if onnxruntime is installed."""
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def export_backbone_onnx(
    encoder,
    output_dir: str | Path,
) -> Path:
    """Export the encoder's backbone to ONNX format using HF Optimum.

    Args:
        encoder: An Encoder instance with a loaded backbone.
        output_dir: Directory to save the ONNX model into.

    Returns:
        Path to the exported model directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = encoder.config.backbone
    if not model_name:
        raise RuntimeError("Encoder has no backbone model name")

    from optimum.onnxruntime import ORTModelForFeatureExtraction

    model = ORTModelForFeatureExtraction.from_pretrained(
        model_name, export=True,
    )
    model.save_pretrained(output_dir)
    # Also copy the tokenizer
    encoder._tokenizer.save_pretrained(output_dir)

    logger.info("Exported backbone to %s", output_dir)
    return output_dir


class OnnxBackbone:
    """Drop-in replacement for the PyTorch backbone using ONNX Runtime.

    Uses the HuggingFace Optimum ORTModel for seamless integration.
    """

    def __init__(self, model_dir: str | Path) -> None:
        from optimum.onnxruntime import ORTModelForFeatureExtraction

        self._model = ORTModelForFeatureExtraction.from_pretrained(str(model_dir))
        logger.info("ONNX backbone loaded from %s", model_dir)

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ):
        """Run inference matching the HuggingFace model interface."""
        return self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )


def attach_onnx_backbone(encoder, model_dir: str | Path) -> None:
    """Replace the encoder's PyTorch backbone with an ONNX session.

    After this call, all encode_query/encode_passage calls use ONNX
    Runtime instead of PyTorch, giving 2-5x speedup on CPU.
    """
    onnx_backbone = OnnxBackbone(model_dir)
    # Bypass nn.Module.__setattr__ which rejects non-Module values
    object.__setattr__(encoder, "_backbone", onnx_backbone)
    object.__setattr__(encoder, "_device", torch.device("cpu"))
    logger.info("Encoder backbone replaced with ONNX Runtime session")
