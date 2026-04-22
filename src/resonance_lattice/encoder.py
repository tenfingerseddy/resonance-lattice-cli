# SPDX-License-Identifier: BUSL-1.1
"""Multi-scale encoder for the Resonance Lattice.

Wraps a pre-trained sentence encoder backbone (e.g. E5-large) and adds B
projection heads that produce sparse phase vectors at different abstraction
levels. Each head projects the backbone's [CLS] representation to a D-dimensional
sparse vector via a linear layer + top-k sparsification.

MVP: 2 bands (Topic + Entity) with random projection heads.
Full: 5 bands with trained projection heads (Phase 3).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from resonance_lattice.config import EncoderConfig


class PhaseSpectrum(NamedTuple):
    """The multi-scale phase spectrum for a single source or query.

    Attributes:
        vectors: Shape (B, D) — sparse phase vectors, one per band.
        sparsity: Actual sparsity (fraction of non-zero) per band.
    """
    vectors: NDArray[np.float32]  # (B, D)
    sparsity: NDArray[np.float32]  # (B,)


class KeyValuePhaseSpectrum(NamedTuple):
    """Asymmetric key/value phase spectrum for a single source or query.

    Separates the matching signal (key) from the retrieval payload (value).
    Keys optimize for query discrimination; values optimize for evidence quality.

    Attributes:
        key_vectors: Shape (B, D_key) — sparse key vectors for matching.
        value_vectors: Shape (B, D_value) — sparse value vectors for retrieval.
        key_sparsity: Actual key sparsity per band.
        value_sparsity: Actual value sparsity per band.
    """
    key_vectors: NDArray[np.float32]    # (B, D_key)
    value_vectors: NDArray[np.float32]  # (B, D_value)
    key_sparsity: NDArray[np.float32]   # (B,)
    value_sparsity: NDArray[np.float32] # (B,)


def _sparsemax(z: torch.Tensor) -> torch.Tensor:
    """Sparsemax: exactly sparse softmax alternative (Martins & Astudillo 2016).

    Maps R^D -> sparse probability simplex. Unlike hard top-k, the sparsity
    pattern is learned from the value distribution — semantically similar
    inputs naturally get similar support sets.

    Args:
        z: (batch, D) input logits.

    Returns:
        (batch, D) sparse, non-negative output summing to 1 per row.
    """
    z_sorted, _ = z.sort(dim=-1, descending=True)
    z_cumsum = z_sorted.cumsum(dim=-1)
    D = z.shape[-1]
    k = torch.arange(1, D + 1, device=z.device, dtype=z.dtype)
    # Support: largest k such that 1 + k*z_sorted[k] > cumsum[k]
    support = ((1 + k * z_sorted) > z_cumsum).sum(dim=-1, keepdim=True)
    # Threshold tau
    tau = (z_cumsum.gather(-1, (support - 1).clamp(min=0)) - 1) / support.float().clamp(min=1)
    return torch.clamp(z - tau, min=0)


class ProjectionHead(nn.Module):
    """Single-band projection head: backbone_dim -> D with configurable sparsification.

    Sparsification modes:
        "threshold" (default): Learnable threshold (ISTA-style). Produces
            genuinely variable sparsity with data-dependent support sets.
            +5.5% dense nDCG@10 over hard top-k on SciFact.
        "topk": Hard top-k by magnitude. Fast, simple, but fragile
            sparsity patterns — small magnitude changes flip the support set.
        "sparsemax": Exactly sparse output where the boundary is determined by
            value distribution, not rank order. Semantically similar inputs
            get similar support patterns (higher overlap in dot products).
        "soft_topk": Sigmoid-smoothed top-k. Near-threshold values get partial
            weight. Differentiable but not exactly sparse.

    Optional enhancements:
        learned_temperature: learnable per-band scaling applied before
            sparsification.
        soft_topk: legacy flag, equivalent to sparsify_mode="soft_topk".
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sparsity: float,
        learned_temperature: bool = False,
        soft_topk: bool = False,
        tau: float = 10.0,
        sparsify_mode: str = "threshold",
        sparsemax_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.k = max(1, int(sparsity * output_dim))
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

        # Learned temperature: exp(log_temp) scales activations before sparsification
        self.learned_temperature = learned_temperature
        if learned_temperature:
            self.log_temp = nn.Parameter(torch.zeros(1))  # init at temp=1.0

        # Soft top-k: sigmoid mask steepness
        self.soft_topk = soft_topk
        self.tau = tau

        # Sparsification mode
        self.sparsify_mode = "soft_topk" if soft_topk else sparsify_mode

        # Sparsemax scale: controls sparsity level. Higher = sparser.
        if sparsemax_scale is not None:
            self._sparsemax_scale = sparsemax_scale
        else:
            self._sparsemax_scale = 8.0 / max(sparsity, 0.01)

        # Learned threshold for "threshold" mode (soft thresholding from
        # compressed sensing / ISTA). Initialized near zero so all values
        # pass initially; the sparsity penalty during training pushes it up
        # to the right level.
        if sparsify_mode == "threshold":
            self.threshold = nn.Parameter(torch.tensor(0.001))

    def forward(self, x: torch.Tensor, dense_query: bool = False) -> torch.Tensor:
        """Project and sparsify.

        Args:
            x: Shape (batch, input_dim).
            dense_query: If True, skip sparsification entirely.

        Returns:
            Shape (batch, output_dim) — sparse, L2-normalised vectors.
        """
        h = self.linear(x)  # (batch, output_dim)

        # Learned temperature scaling
        if self.learned_temperature:
            h = h * torch.exp(self.log_temp)

        if dense_query:
            return F.normalize(h, p=2, dim=-1)

        if self.sparsify_mode == "threshold":
            # Learned soft thresholding (ISTA-style).
            # h_sparse = sign(h) * max(|h| - theta, 0)
            # Exactly sparse, differentiable, preserves signs.
            theta = self.threshold.abs()  # ensure non-negative threshold
            h = h.sign() * torch.relu(h.abs() - theta)

        elif self.sparsify_mode == "sparsemax":
            # Sparsemax on absolute values, then restore signs
            abs_h = h.abs() * self._sparsemax_scale
            sparse_weights = _sparsemax(abs_h)
            h = h.sign() * sparse_weights

        elif self.sparsify_mode == "soft_topk":
            abs_h = h.abs()
            topk_vals, _ = torch.topk(abs_h, self.k, dim=-1)
            threshold = topk_vals[:, -1:].detach()
            mask = torch.sigmoid(self.tau * (abs_h - threshold))
            h = h * mask

        else:
            # Default: hard top-k
            abs_h = h.abs()
            _, top_indices = torch.topk(abs_h, self.k, dim=-1)
            mask = torch.zeros_like(h)
            mask.scatter_(-1, top_indices, 1.0)
            h = h * mask

        # L2 normalise
        h = F.normalize(h, p=2, dim=-1)
        return h


class Encoder(nn.Module):
    """Multi-scale encoder producing B sparse phase vectors per input.

    Architecture:
        Input text -> Backbone (frozen) -> [CLS] -> B projection heads -> PhaseSpectrum
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self._backbone = None
        self._tokenizer = None
        self._backbone_dim: int = config.backbone_dim or 1024  # placeholder
        self._device = torch.device("cpu")

        # Create projection heads
        self.heads = nn.ModuleList([
            ProjectionHead(
                input_dim=self._backbone_dim,
                output_dim=config.dim,
                sparsity=config.get_sparsity(b),
            )
            for b in range(config.bands)
        ])

    @classmethod
    def from_backbone(
        cls,
        model_name: str = "intfloat/e5-large-v2",
        bands: int = 2,
        dim: int = 2048,
        sparsity: tuple[float, ...] | None = None,
        device: str = "cpu",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        pooling: str = "mean",
        max_length: int = 512,
        sparsify_mode: str = "threshold",
        soft_topk_tau: float | None = None,
        sparsemax_scale: float | None = None,
        **kwargs,
    ) -> Encoder:
        """Create an encoder by loading a HuggingFace backbone.

        The backbone is frozen; only the projection heads are trainable.

        Args:
            sparsify_mode: ProjectionHead sparsification mode. One of
                "threshold" (default), "topk", "sparsemax", "soft_topk".
                Research/advanced — see Phase 3 ablate_sparsification.
            soft_topk_tau: sigmoid sharpness for sparsify_mode="soft_topk"
                (passed as ProjectionHead.tau). None → head default 10.0.
            sparsemax_scale: pre-sparsemax magnitude scaling for
                sparsify_mode="sparsemax". None → head's per-sparsity default.
        """
        from transformers import AutoModel, AutoTokenizer

        trust_rc = kwargs.pop("trust_remote_code", True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_rc)
        backbone = AutoModel.from_pretrained(model_name, trust_remote_code=trust_rc)
        backbone_dim = backbone.config.hidden_size

        config = EncoderConfig(
            backbone=model_name,
            bands=bands,
            dim=dim,
            sparsity=sparsity,
            backbone_dim=backbone_dim,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            pooling=pooling,
            max_length=max_length,
        )

        encoder = cls(config)
        encoder._backbone = backbone
        encoder._tokenizer = tokenizer
        encoder._backbone_dim = backbone_dim
        encoder._device = torch.device(device)

        # Rebuild heads with correct input dim. **Deterministic** head init:
        # the random projection heads must be identical across processes so
        # phases encoded in one run (e.g. a cached Kaggle kernel output) match
        # phases encoded in another run against the same lattice. Without this
        # seeding, two `from_backbone(...)` calls produce orthogonal phase
        # subspaces and cached retrieval degrades to near-random.
        import hashlib
        _seed_key = f"{model_name}|{bands}|{dim}|{sparsity}".encode()
        seed = int(hashlib.md5(_seed_key).hexdigest()[:8], 16)
        _rng_state = torch.get_rng_state()
        try:
            torch.manual_seed(seed)
            head_extra = {}
            if soft_topk_tau is not None:
                head_extra["tau"] = soft_topk_tau
            if sparsemax_scale is not None:
                head_extra["sparsemax_scale"] = sparsemax_scale
            encoder.heads = nn.ModuleList([
                ProjectionHead(
                    input_dim=backbone_dim,
                    output_dim=dim,
                    sparsity=config.get_sparsity(b),
                    sparsify_mode=sparsify_mode,
                    **head_extra,
                )
                for b in range(bands)
            ])
        finally:
            torch.set_rng_state(_rng_state)

        # Freeze backbone, leave heads trainable
        for param in backbone.parameters():
            param.requires_grad = False

        encoder.to(encoder._device)
        backbone.to(encoder._device)

        return encoder

    def enable_soft_topk(self, tau: float = 10.0) -> None:
        """Enable soft top-k sparsification on all projection heads.

        Replaces hard binary masks with sigmoid-based soft masks that
        preserve near-threshold values. Call after loading an encoder
        to improve cross-corpus retrieval quality.
        """
        for head in self.heads:
            head.soft_topk = True
            head.tau = tau

    @classmethod
    def random(
        cls,
        bands: int = 2,
        dim: int = 2048,
        input_dim: int = 768,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        pooling: str = "cls",
        sparsify_mode: str = "threshold",
        soft_topk_tau: float | None = None,
        sparsemax_scale: float | None = None,
    ) -> Encoder:
        """Create an encoder with random projection heads (no backbone).

        Useful for testing field machinery without downloading a large model.
        Accepts raw vectors of shape (input_dim,) instead of text.

        Args:
            sparsify_mode: ProjectionHead sparsification mode. One of
                "threshold" (default), "topk", "sparsemax", "soft_topk".
            soft_topk_tau: sigmoid sharpness for soft_topk mode.
            sparsemax_scale: magnitude scaling for sparsemax mode.
        """
        config = EncoderConfig(
            backbone="random",
            bands=bands,
            dim=dim,
            backbone_dim=input_dim,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            pooling=pooling,
        )
        encoder = cls(config)
        encoder._backbone_dim = input_dim
        # Rebuild heads with requested sparsify_mode (Encoder.__init__ always
        # uses the default). Keeps parity with from_backbone.
        needs_rebuild = (
            sparsify_mode != "threshold"
            or soft_topk_tau is not None
            or sparsemax_scale is not None
        )
        if needs_rebuild:
            head_extra = {}
            if soft_topk_tau is not None:
                head_extra["tau"] = soft_topk_tau
            if sparsemax_scale is not None:
                head_extra["sparsemax_scale"] = sparsemax_scale
            encoder.heads = nn.ModuleList([
                ProjectionHead(
                    input_dim=input_dim,
                    output_dim=dim,
                    sparsity=config.get_sparsity(b),
                    sparsify_mode=sparsify_mode,
                    **head_extra,
                )
                for b in range(bands)
            ])
        return encoder

    def get_config(self) -> dict:
        """Return a JSON-serializable dict describing how to reconstruct this encoder.

        Stored in the .rlat SourceStore so knowledge models are self-describing.
        """
        sparsities = [head.sparsity for head in self.heads]
        encoder_type = "random"
        if self._backbone is not None:
            encoder_type = "from_backbone"

        # Detect sparsify_mode from heads (all heads share the same mode)
        head0 = self.heads[0] if self.heads else None
        sparsify_mode = getattr(head0, "sparsify_mode", "threshold") if head0 else "threshold"
        soft_topk_tau = getattr(head0, "tau", None) if head0 is not None else None
        sparsemax_scale = getattr(head0, "_sparsemax_scale", None) if head0 is not None else None

        config = {
            "encoder_type": encoder_type,
            "backbone": self.config.backbone,
            "bands": self.config.bands,
            "dim": self.config.dim,
            "backbone_dim": self._backbone_dim,
            "sparsities": sparsities,
            "sparsify_mode": sparsify_mode,
            "soft_topk_tau": soft_topk_tau,
            "sparsemax_scale": sparsemax_scale,
            "query_prefix": self.config.query_prefix,
            "passage_prefix": self.config.passage_prefix,
            "pooling": self.config.pooling,
            "max_length": self.config.max_length,
        }

        return config

    @classmethod
    def from_config(cls, config: dict) -> Encoder:
        """Reconstruct an encoder from a config dict (stored in .rlat).

        Args:
            config: Dict from get_config().

        Returns:
            Reconstructed Encoder instance.
        """
        encoder_type = config.get("encoder_type", "random")
        bands = config["bands"]
        dim = config["dim"]
        backbone_dim = config.get("backbone_dim", 1024)

        # Protocol fields — legacy cartridges lack these fields and were built
        # with bare encode() (no prefix, CLS pooling), so the fallback must
        # match that contract to avoid silently changing retrieval behavior.
        query_prefix = config.get("query_prefix", "")
        passage_prefix = config.get("passage_prefix", "")
        pooling = config.get("pooling", "cls")
        max_length = config.get("max_length", 512)

        # Restore sparsify_mode from stored config; legacy cartridges used "topk"
        sparsify_mode = config.get("sparsify_mode", "topk")
        soft_topk_tau = config.get("soft_topk_tau")
        sparsemax_scale = config.get("sparsemax_scale")

        if encoder_type == "from_backbone":
            backbone = config.get("backbone", "intfloat/e5-large-v2")
            sparsities = tuple(config.get("sparsities", []))
            enc = cls.from_backbone(
                model_name=backbone,
                bands=bands,
                dim=dim,
                sparsity=sparsities or None,
                query_prefix=query_prefix,
                passage_prefix=passage_prefix,
                pooling=pooling,
                max_length=max_length,
            )
            # Rebuild heads with stored sparsify_mode / scale / tau to match build-time
            head0 = enc.heads[0] if enc.heads else None
            mode_mismatch = sparsify_mode != getattr(head0, "sparsify_mode", "threshold")
            tau_mismatch = soft_topk_tau is not None and soft_topk_tau != getattr(head0, "tau", None)
            scale_mismatch = sparsemax_scale is not None and sparsemax_scale != getattr(head0, "_sparsemax_scale", None)
            if mode_mismatch or tau_mismatch or scale_mismatch:
                head_extra = {}
                if soft_topk_tau is not None:
                    head_extra["tau"] = soft_topk_tau
                if sparsemax_scale is not None:
                    head_extra["sparsemax_scale"] = sparsemax_scale
                enc.heads = nn.ModuleList([
                    ProjectionHead(
                        input_dim=enc._backbone_dim,
                        output_dim=dim,
                        sparsity=sparsities[b] if b < len(sparsities) else enc.config.get_sparsity(b),
                        sparsify_mode=sparsify_mode,
                        **head_extra,
                    )
                    for b in range(bands)
                ])
                enc.to(enc._device)
            return enc

        # Default: random encoder
        return cls.random(
            bands=bands, dim=dim, input_dim=backbone_dim,
            query_prefix=query_prefix, passage_prefix=passage_prefix,
            pooling=pooling,
        )

    # ── Embedded head weight serialization ──────────────────────────────

    _RLHW_MAGIC = b"RLHW"  # Resonance Lattice Head Weights
    _RLHW_VERSION = 1
    _RLHW_PRECISION = {0: np.float16, 1: np.float16, 2: np.float32}  # 0=f16,1=bf16,2=f32
    _RLHW_PRECISION_ID = {"f16": 0, "bf16": 1, "f32": 2}

    def get_head_weights_blob(self, precision: str = "f16") -> bytes:
        """Serialize projection head weights into a compact binary blob.

        Format: [magic 4B][version 1B][precision 1B][num_heads 1B][pad 1B]
                [input_dim 4B][output_dim 4B] = 16 bytes header
                + concatenated weight matrices in row-major order
        """
        import struct
        import zlib

        prec_id = self._RLHW_PRECISION_ID.get(precision, 0)
        dtype = self._RLHW_PRECISION.get(prec_id, np.float16)
        num_heads = len(self.heads)

        # Collect weight matrices
        weight_arrays = []
        for head in self.heads:
            w = head.linear.weight.detach().cpu().numpy().astype(dtype)
            weight_arrays.append(w)

        input_dim = weight_arrays[0].shape[1] if weight_arrays else self._backbone_dim
        output_dim = weight_arrays[0].shape[0] if weight_arrays else self.config.dim

        header = struct.pack(
            "<4sBBBxII",
            self._RLHW_MAGIC,
            self._RLHW_VERSION,
            prec_id,
            num_heads,
            input_dim,
            output_dim,
        )

        raw_weights = b"".join(w.tobytes() for w in weight_arrays)
        compressed = zlib.compress(raw_weights, level=6)

        return header + compressed

    def load_head_weights_blob(self, blob: bytes) -> None:
        """Deserialize and load projection head weights from a binary blob."""
        import struct
        import zlib

        header_size = 16
        magic, version, prec_id, num_heads, input_dim, output_dim = struct.unpack(
            "<4sBBBxII", blob[:header_size],
        )
        if magic != self._RLHW_MAGIC:
            raise ValueError(f"Invalid head weights magic: {magic!r}")

        dtype = self._RLHW_PRECISION.get(prec_id, np.float16)
        raw_weights = zlib.decompress(blob[header_size:])

        bytes_per_head = input_dim * output_dim * np.dtype(dtype).itemsize
        for b, head in enumerate(self.heads):
            offset = b * bytes_per_head
            w = np.frombuffer(raw_weights, dtype=dtype, count=input_dim * output_dim,
                              offset=offset).reshape(output_dim, input_dim)
            head.linear.weight.data.copy_(torch.from_numpy(w.astype(np.float32)))

    @classmethod
    def from_config_with_weights(cls, config: dict, weights_blob: bytes) -> Encoder:
        """Reconstruct an encoder from config + embedded head weights.

        Loads the backbone from HuggingFace but uses the embedded head weights
        instead of freshly-seeded random ones.
        """
        encoder_type = config.get("encoder_type", "random")
        bands = config["bands"]
        dim = config["dim"]
        backbone_dim = config.get("backbone_dim", 1024)
        query_prefix = config.get("query_prefix", "")
        passage_prefix = config.get("passage_prefix", "")
        pooling = config.get("pooling", "cls")
        max_length = config.get("max_length", 512)
        backbone = config.get("backbone", "intfloat/e5-large-v2")

        sparsities = tuple(config.get("sparsities", []))
        # Restore the sparsify_mode that was used at build time.
        # Legacy cartridges without this field used "topk".
        sparsify_mode = config.get("sparsify_mode", "topk")
        soft_topk_tau = config.get("soft_topk_tau")
        sparsemax_scale = config.get("sparsemax_scale")

        if encoder_type == "random" or backbone in ("random", "none", ""):
            enc = cls.random(
                bands=bands, dim=dim, input_dim=backbone_dim,
                query_prefix=query_prefix, passage_prefix=passage_prefix,
                pooling=pooling,
            )
        else:
            enc = cls.from_backbone(
                model_name=backbone,
                bands=bands,
                dim=dim,
                sparsity=sparsities or None,
                query_prefix=query_prefix,
                passage_prefix=passage_prefix,
                pooling=pooling,
                max_length=max_length,
            )

        head0 = enc.heads[0] if enc.heads else None
        mode_mismatch = sparsify_mode != getattr(head0, "sparsify_mode", "threshold")
        tau_mismatch = soft_topk_tau is not None and soft_topk_tau != getattr(head0, "tau", None)
        scale_mismatch = sparsemax_scale is not None and sparsemax_scale != getattr(head0, "_sparsemax_scale", None)
        if mode_mismatch or tau_mismatch or scale_mismatch:
            head_extra = {}
            if soft_topk_tau is not None:
                head_extra["tau"] = soft_topk_tau
            if sparsemax_scale is not None:
                head_extra["sparsemax_scale"] = sparsemax_scale
            enc.heads = nn.ModuleList([
                ProjectionHead(
                    input_dim=backbone_dim,
                    output_dim=dim,
                    sparsity=sparsities[b] if b < len(sparsities) else enc.config.get_sparsity(b),
                    sparsify_mode=sparsify_mode,
                    **head_extra,
                )
                for b in range(bands)
            ])

        enc.load_head_weights_blob(weights_blob)
        return enc

    @property
    def has_backbone(self) -> bool:
        return self._backbone is not None

    def _get_backbone_embedding(self, texts: list[str], pooling: str | None = None) -> torch.Tensor:
        """Run the backbone to get pooled embeddings.

        Args:
            texts: List of input strings.
            pooling: Override pooling strategy ("cls", "mean", or "last").
                If None, uses self.config.pooling.

        Returns:
            Shape (batch, backbone_dim).
        """
        pool = pooling or self.config.pooling

        if self._tokenizer is None or self._backbone is None:
            # Fallback: deterministic hash-based embedding for text
            # This allows the random encoder to accept text inputs for testing
            return self._hash_embed(texts)

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self._device)

        # FP16 autocast on CUDA for ~2x throughput on tensor-core GPUs (T4, A100)
        use_amp = self._device != "cpu" and torch.cuda.is_available()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            outputs = self._backbone(**inputs)

            if pool == "mean":
                # Attention-masked mean pooling (E5 recipe)
                token_embeddings = outputs.last_hidden_state  # (batch, seq, dim)
                attention_mask = inputs["attention_mask"].unsqueeze(-1).float()  # (batch, seq, 1)
                summed = (token_embeddings * attention_mask).sum(dim=1)  # (batch, dim)
                counts = attention_mask.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
                pooled = summed / counts
            elif pool == "last":
                # Last non-padding token pooling for decoder-only LMs
                # (Qwen3-Embedding, E5-mistral, GritLM, NV-Embed). Mean/CLS
                # pooling on a causal LM dilutes or discards the position
                # with full left-context — measured 7x nDCG@10 collapse on
                # fiqa with Qwen3-8B + mean pool (2026-04-21 baseline).
                token_embeddings = outputs.last_hidden_state  # (batch, seq, dim)
                attention_mask = inputs["attention_mask"]  # (batch, seq)
                seq_lens = attention_mask.sum(dim=1) - 1  # (batch,)
                batch_idx = torch.arange(
                    token_embeddings.size(0), device=token_embeddings.device
                )
                pooled = token_embeddings[batch_idx, seq_lens]
            else:
                # CLS token pooling (backward-compatible default)
                pooled = outputs.last_hidden_state[:, 0, :]  # (batch, backbone_dim)

        return pooled.float()  # ensure fp32 output regardless of amp

    def _hash_embed(self, texts: list[str]) -> torch.Tensor:
        """Deterministic hash-based embedding for testing without a backbone.

        Uses character n-gram hashing to produce a fixed-size vector.
        Not semantic — but deterministic and fast, so same text always
        gives the same embedding. Useful for testing field machinery.
        """
        import hashlib

        embeddings = []
        for text in texts:
            vec = np.zeros(self._backbone_dim, dtype=np.float32)
            # Hash overlapping 3-grams into the vector
            words = text.lower().split()
            for i in range(len(words)):
                for n in range(1, 4):  # 1-gram, 2-gram, 3-gram
                    if i + n <= len(words):
                        gram = " ".join(words[i:i+n])
                        h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
                        idx = h % self._backbone_dim
                        sign = 1.0 if (h // self._backbone_dim) % 2 == 0 else -1.0
                        vec[idx] += sign * (1.0 / n)  # Weight by n-gram length
            # L2 normalise
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec /= norm
            embeddings.append(vec)

        return torch.tensor(np.array(embeddings), device=self._device)

    def encode_texts(self, texts: list[str]) -> list[PhaseSpectrum]:
        """Encode a list of texts into phase spectra.

        Args:
            texts: Input text strings.

        Returns:
            List of PhaseSpectrum, one per input text.
        """
        embeddings = self._get_backbone_embedding(texts)
        return self._project(embeddings)

    def encode(self, text: str) -> PhaseSpectrum:
        """Encode a single text into a phase spectrum (no prefix, backward compat)."""
        return self.encode_texts([text])[0]

    _QUERY_CACHE_SIZE = 512

    def __init_query_cache(self) -> None:
        if not hasattr(self, "_query_cache"):
            self._query_cache: dict[tuple[str, bool], PhaseSpectrum] = {}

    def encode_query(self, text: str, asymmetric: bool = False) -> PhaseSpectrum:
        """Encode a single query text with the query prefix.

        Results are cached by (text, asymmetric) key — repeated queries
        skip the backbone forward pass entirely.
        """
        self.__init_query_cache()
        key = (text.strip().lower(), asymmetric)
        cached = self._query_cache.get(key)
        if cached is not None:
            return cached
        result = self.encode_queries([text], asymmetric=asymmetric)[0]
        self._query_cache[key] = result
        # Evict oldest if over capacity
        while len(self._query_cache) > self._QUERY_CACHE_SIZE:
            self._query_cache.pop(next(iter(self._query_cache)))
        return result

    def encode_passage(self, text: str) -> PhaseSpectrum:
        """Encode a single passage text with the passage prefix."""
        return self.encode_passages([text])[0]

    def encode_queries(
        self, texts: list[str], asymmetric: bool = False,
    ) -> list[PhaseSpectrum]:
        """Encode a list of query texts with the query prefix.

        Args:
            texts: Input query strings.
            asymmetric: If True, produce dense (non-sparsified) query vectors.
                Used for asymmetric retrieval where queries are dense but
                corpus phases remain sparse.

        Returns:
            List of PhaseSpectrum, one per input text.
        """
        prefixed = [self.config.query_prefix + t for t in texts]
        embeddings = self._get_backbone_embedding(prefixed)
        return self._project(embeddings, dense_query=asymmetric)

    def encode_passages(
        self, texts: list[str], batch_size: int = 64,
    ) -> list[PhaseSpectrum]:
        """Encode a list of passage texts with the passage prefix.

        Args:
            texts: Input passage strings.
            batch_size: Number of texts to encode per backbone forward pass.
                Larger = faster (better GPU utilisation) but more VRAM.
                Default 64 fits comfortably on T4 (16 GB).

        Returns:
            List of PhaseSpectrum, one per input text.
        """
        if len(texts) <= batch_size:
            prefixed = [self.config.passage_prefix + t for t in texts]
            embeddings = self._get_backbone_embedding(prefixed)
            return self._project(embeddings)

        # Mini-batch to avoid OOM on large corpora
        all_results: list[PhaseSpectrum] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            prefixed = [self.config.passage_prefix + t for t in batch]
            embeddings = self._get_backbone_embedding(prefixed)
            all_results.extend(self._project(embeddings))
        return all_results

    def encode_vectors(self, vectors: NDArray[np.float32]) -> list[PhaseSpectrum]:
        """Encode raw vectors (bypassing backbone) through projection heads.

        Useful with the random encoder or pre-computed embeddings.

        Args:
            vectors: Shape (batch, input_dim).

        Returns:
            List of PhaseSpectrum.
        """
        x = torch.from_numpy(vectors).to(self._device)
        return self._project(x)

    def encode_vector(self, vector: NDArray[np.float32]) -> PhaseSpectrum:
        """Encode a single raw vector through projection heads."""
        return self.encode_vectors(vector[np.newaxis, :])[0]

    def _project(
        self, embeddings: torch.Tensor, dense_query: bool = False,
    ) -> list[PhaseSpectrum]:
        """Run projection heads on backbone embeddings.

        Args:
            embeddings: Shape (batch, backbone_dim).
            dense_query: If True, skip top-k sparsification in heads.

        Returns:
            List of PhaseSpectrum.
        """
        batch_size = embeddings.shape[0]
        results = []

        # Run each head
        band_outputs = []
        for head in self.heads:
            band_outputs.append(head(embeddings, dense_query=dense_query))  # (batch, D)

        # Stack into (batch, B, D) and convert to numpy
        stacked = torch.stack(band_outputs, dim=1)  # (batch, B, D)
        stacked_np = stacked.detach().cpu().numpy()

        for i in range(batch_size):
            vectors = stacked_np[i]  # (B, D)
            # Compute actual sparsity per band
            sparsity = np.array([
                np.count_nonzero(vectors[b]) / vectors.shape[1]
                for b in range(vectors.shape[0])
            ], dtype=np.float32)
            results.append(PhaseSpectrum(vectors=vectors, sparsity=sparsity))

        return results

    # ── Asymmetric key/value encoding ─────────────────────────────────

    def init_value_heads(
        self,
        dim_value: int | tuple[int, ...] | None = None,
        sparsity: tuple[float, ...] | None = None,
    ) -> None:
        """Create value-space projection heads for asymmetric encoding.

        After calling this, the existing `heads` become key heads (matching),
        and the new `value_heads` produce value vectors (retrieval).

        Args:
            dim_value: Output dimensionality for value heads. If None, uses
                the same dim as key heads. Can be a tuple for per-band sizing.
            sparsity: Per-band sparsity for value heads. If None, uses same
                as key heads.
        """
        bands = self.config.bands

        if dim_value is None:
            dims = [self.config.dim] * bands
        elif isinstance(dim_value, int):
            dims = [dim_value] * bands
        else:
            dims = list(dim_value)
            if len(dims) != bands:
                raise ValueError(
                    f"dim_value tuple length ({len(dims)}) must equal bands ({bands})"
                )

        self.value_heads = nn.ModuleList([
            ProjectionHead(
                input_dim=self._backbone_dim,
                output_dim=dims[b],
                sparsity=sparsity[b] if sparsity else self.config.get_sparsity(b),
            )
            for b in range(bands)
        ])
        self._dim_value = tuple(dims)

    @property
    def has_value_heads(self) -> bool:
        """Whether this encoder has separate value heads for asymmetric encoding."""
        return hasattr(self, "value_heads") and self.value_heads is not None

    def encode_kv_passage(self, text: str) -> KeyValuePhaseSpectrum:
        """Encode a passage into separate key and value phase spectra."""
        return self.encode_kv_passages([text])[0]

    def encode_kv_passages(
        self, texts: list[str], batch_size: int = 64,
    ) -> list[KeyValuePhaseSpectrum]:
        """Encode passages into key/value phase spectra.

        Key vectors come from `self.heads` (the standard projection heads).
        Value vectors come from `self.value_heads`.

        Args:
            texts: Input passage strings.
            batch_size: Texts per backbone forward pass.

        Returns:
            List of KeyValuePhaseSpectrum.
        """
        if not self.has_value_heads:
            raise RuntimeError(
                "Encoder has no value heads. Call init_value_heads() first."
            )

        if len(texts) <= batch_size:
            prefixed = [self.config.passage_prefix + t for t in texts]
            embeddings = self._get_backbone_embedding(prefixed)
            return self._project_kv(embeddings)

        all_results: list[KeyValuePhaseSpectrum] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            prefixed = [self.config.passage_prefix + t for t in batch]
            embeddings = self._get_backbone_embedding(prefixed)
            all_results.extend(self._project_kv(embeddings))
        return all_results

    def encode_kv_query(self, text: str) -> KeyValuePhaseSpectrum:
        """Encode a query into key/value phase spectra (key is used for matching)."""
        return self.encode_kv_queries([text])[0]

    def encode_kv_queries(
        self, texts: list[str], dense_key: bool = False,
    ) -> list[KeyValuePhaseSpectrum]:
        """Encode queries into key/value phase spectra.

        At query time, only the key vectors are used for field resonance.
        Value vectors are returned for completeness but not used in matching.

        Args:
            texts: Input query strings.
            dense_key: If True, skip sparsification on key vectors.
        """
        if not self.has_value_heads:
            raise RuntimeError(
                "Encoder has no value heads. Call init_value_heads() first."
            )
        prefixed = [self.config.query_prefix + t for t in texts]
        embeddings = self._get_backbone_embedding(prefixed)
        return self._project_kv(embeddings, dense_key=dense_key)

    def _project_kv(
        self, embeddings: torch.Tensor, dense_key: bool = False,
    ) -> list[KeyValuePhaseSpectrum]:
        """Run both key and value heads on backbone embeddings.

        Args:
            embeddings: Shape (batch, backbone_dim).
            dense_key: If True, skip sparsification on key heads.

        Returns:
            List of KeyValuePhaseSpectrum.
        """
        batch_size = embeddings.shape[0]

        # Key heads (existing heads)
        key_outputs = []
        for head in self.heads:
            key_outputs.append(head(embeddings, dense_query=dense_key))
        key_stacked = torch.stack(key_outputs, dim=1).detach().cpu().numpy()

        # Value heads
        val_outputs = []
        for head in self.value_heads:
            val_outputs.append(head(embeddings, dense_query=False))
        val_stacked = torch.stack(val_outputs, dim=1).detach().cpu().numpy()

        results = []
        for i in range(batch_size):
            kv = key_stacked[i]  # (B, D_key)
            vv = val_stacked[i]  # (B, D_value)
            key_sp = np.array([
                np.count_nonzero(kv[b]) / kv.shape[1]
                for b in range(kv.shape[0])
            ], dtype=np.float32)
            val_sp = np.array([
                np.count_nonzero(vv[b]) / vv.shape[1]
                for b in range(vv.shape[0])
            ], dtype=np.float32)
            results.append(KeyValuePhaseSpectrum(
                key_vectors=kv,
                value_vectors=vv,
                key_sparsity=key_sp,
                value_sparsity=val_sp,
            ))

        return results
