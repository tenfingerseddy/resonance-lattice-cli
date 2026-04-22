# SPDX-License-Identifier: BUSL-1.1
"""Configuration dataclasses for the Resonance Lattice."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class FieldType(StrEnum):
    """Storage backend for the interference field tensor."""
    DENSE = "dense"
    ASYMMETRIC_DENSE = "asymmetric_dense"
    FACTORED = "factored"
    PQ = "pq"
    MULTI_VECTOR = "multi_vector"


class Precision(StrEnum):
    """Floating-point precision for field storage."""
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"


class Compression(StrEnum):
    """Compression algorithm for serialisation."""
    NONE = "none"
    ZSTD = "zstd"
    LZ4 = "lz4"


class StoreMode(StrEnum):
    """How source content is stored in the knowledge model."""
    EMBEDDED = "embedded"   # Full store inside .rlat (default)
    EXTERNAL = "external"   # No store — resolve source_ids to local files at query time


@dataclass(frozen=True)
class LatticeConfig:
    """Configuration for a Resonance Lattice instance.

    Attributes:
        bands: Number of frequency bands (Omega_1 through Omega_B).
        dim: Dimensionality per band (D). Phase vectors are D-dimensional.
            For symmetric field types (dense, factored, pq), D is used for both
            key and value. For asymmetric_dense, see dim_key and dim_value.
        field_type: Storage backend — dense, asymmetric_dense, factored (SVD),
            or pq (product-quantised).
        dim_key: Key dimensionality for asymmetric field. Either a single int
            (uniform across bands) or a tuple of length B (per-band sizing).
            Only used when field_type="asymmetric_dense". If None, falls back to dim.
        dim_value: Value dimensionality for asymmetric field. Either a single int
            or a tuple of length B. Only used when field_type="asymmetric_dense".
            If None, falls back to dim.
        pq_subspaces: M — number of PQ subspaces (only used when field_type="pq").
        pq_codebook_size: K — centroids per subspace (only used when field_type="pq").
        svd_rank: K — rank for factored SVD field (only used when field_type="factored").
        precision: Floating-point precision for field storage.
        compression: Compression for .rlat serialisation.
        temporal_windows: Number of time-horizon field tensors (e.g. recent/medium/archive).
    """
    bands: int = 5
    dim: int = 2048
    field_type: FieldType = FieldType.DENSE
    dim_key: int | tuple[int, ...] | None = None
    dim_value: int | tuple[int, ...] | None = None
    pq_subspaces: int = 8
    pq_codebook_size: int = 1024
    svd_rank: int = 512
    precision: Precision = Precision.F16
    compression: Compression = Compression.ZSTD
    temporal_windows: int = 3

    def __post_init__(self) -> None:
        if self.bands < 1:
            raise ValueError(f"bands must be >= 1, got {self.bands}")
        if self.dim < 64:
            raise ValueError(f"dim must be >= 64, got {self.dim}")
        if self.field_type == FieldType.PQ:
            if self.dim % self.pq_subspaces != 0:
                raise ValueError(
                    f"dim ({self.dim}) must be divisible by pq_subspaces ({self.pq_subspaces})"
                )
        if self.field_type == FieldType.ASYMMETRIC_DENSE:
            dk = self.dim_key if self.dim_key is not None else self.dim
            dv = self.dim_value if self.dim_value is not None else self.dim
            if isinstance(dk, tuple) and len(dk) != self.bands:
                raise ValueError(
                    f"dim_key tuple length ({len(dk)}) must equal bands ({self.bands})"
                )
            if isinstance(dv, tuple) and len(dv) != self.bands:
                raise ValueError(
                    f"dim_value tuple length ({len(dv)}) must equal bands ({self.bands})"
                )

    @property
    def subspace_dim(self) -> int:
        """Dimensionality of each PQ subspace (D / M)."""
        return self.dim // self.pq_subspaces

    @property
    def field_size_bytes(self) -> int:
        """Estimated field tensor size in bytes."""
        bytes_per_elem = 2 if self.precision in (Precision.F16, Precision.BF16) else 4
        if self.field_type == FieldType.DENSE:
            return self.bands * self.dim * self.dim * bytes_per_elem
        elif self.field_type == FieldType.ASYMMETRIC_DENSE:
            dk = self.dim_key if self.dim_key is not None else self.dim
            dv = self.dim_value if self.dim_value is not None else self.dim
            if isinstance(dk, tuple) and isinstance(dv, tuple):
                return sum(k * v for k, v in zip(dk, dv)) * bytes_per_elem
            dk_int = dk if isinstance(dk, int) else dk[0]
            dv_int = dv if isinstance(dv, int) else dv[0]
            return self.bands * dk_int * dv_int * bytes_per_elem
        elif self.field_type == FieldType.FACTORED:
            # U[D x K] + Sigma[K] + V[D x K] per band
            return self.bands * (2 * self.dim * self.svd_rank + self.svd_rank) * bytes_per_elem
        elif self.field_type == FieldType.PQ:
            # Per band, per subspace: codebook[K x D/M] + quantised_field[K x K]
            M = self.pq_subspaces
            K = self.pq_codebook_size
            d = self.subspace_dim
            per_band = M * (K * d + K * K) * bytes_per_elem
            return self.bands * per_band
        return 0

    @property
    def field_size_mb(self) -> float:
        """Estimated field tensor size in megabytes."""
        return self.field_size_bytes / (1024 * 1024)


# Default sparsity targets per band from the spec (Section 5.1).
# Relaxed from original spec: Topic 8%→25%, Entity 15%→37.5%.
# Denser representations let the field carry richer signal, improving
# field-only retrieval quality without relying on external indexes.
DEFAULT_SPARSITY_TARGETS = {
    1: 0.05,    # Omega_1 (Domain): 5%
    2: 0.25,    # Omega_2 (Topic): 25%  (was 8%)
    3: 0.10,    # Omega_3 (Relations): 10%
    4: 0.375,   # Omega_4 (Entities): 37.5%  (was 15%)
    5: 0.20,    # Omega_5 (Verbatim): 20%
}


# ── Encoder presets ────────────────────────────────────────────────
# Named configurations for supported backbones.  Each preset maps a
# short CLI name to the full set of encoder parameters the backbone
# requires.  Users select a preset with ``--encoder <name>`` or pass
# a raw HuggingFace model ID for unlisted models.
#
# Three encoders are measured and well-supported on BEIR-5 (see
# docs/ENCODER_CHOICE.md for the decision guide):
#   - ``bge-large-en-v1.5`` — starting-point default (portable, 335M).
#   - ``e5-large-v2``       — opt-in; wins on ArguAna-class corpora.
#   - ``qwen3-8b``          — opt-in; frontier quality, needs 16 GB GPU.
# Other presets (bge-m3, qwen3-0.6b/4b, arctic-embed-2, etc.) are
# available but not currently covered by the 5-BEIR measurement matrix.

_QWEN3_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)

ENCODER_PRESETS: dict[str, dict] = {
    "e5-large-v2": {
        "backbone": "intfloat/e5-large-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "pooling": "mean",
        "max_length": 512,
    },
    "qwen3-0.6b": {
        "backbone": "Qwen/Qwen3-Embedding-0.6B",
        "query_prefix": _QWEN3_QUERY_PREFIX,
        "passage_prefix": "",
        "pooling": "last",
        "max_length": 32768,
    },
    "qwen3-4b": {
        "backbone": "Qwen/Qwen3-Embedding-4B",
        "query_prefix": _QWEN3_QUERY_PREFIX,
        "passage_prefix": "",
        "pooling": "last",
        "max_length": 32768,
    },
    "qwen3-8b": {
        "backbone": "Qwen/Qwen3-Embedding-8B",
        "query_prefix": _QWEN3_QUERY_PREFIX,
        "passage_prefix": "",
        "pooling": "last",
        "max_length": 32768,
    },
    "arctic-embed-2": {
        "backbone": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "query_prefix": "query: ",
        "passage_prefix": "",
        "pooling": "cls",
        "max_length": 8192,
    },
    "nemotron-1b": {
        "backbone": "nvidia/llama-nemotron-embed-1b-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "pooling": "mean",
        "max_length": 8192,
    },
    "nomic-v2": {
        "backbone": "nomic-ai/nomic-embed-text-v2-moe",
        "query_prefix": "search_query: ",
        "passage_prefix": "search_document: ",
        "pooling": "mean",
        "max_length": 8192,
    },
    "bge-m3": {
        "backbone": "BAAI/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
        "pooling": "cls",
        "max_length": 8192,
    },
    "bge-large-en-v1.5": {
        "backbone": "BAAI/bge-large-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "pooling": "cls",
        "max_length": 512,
    },
    # Stage 1 fine-tune of BGE-large on MS MARCO (Tevatron/msmarco-passage-aug,
    # 500K queries × 7 hard negs, 1 epoch, bf16, batch 64, A100 80GB).
    # Trained 2026-04-20 → 2026-04-21 on RunPod — see board item 235 and
    # .claude/plans/encoder-training-stage1.md. Checkpoint lives outside the
    # HuggingFace cache, so the preset's "backbone" key is a local path; set
    # RLAT_STAGE1_MODEL_PATH to override (e.g. point at /workspace/... on a
    # pod or at a different local copy).
    "ft-bge-large-v1-stage1": {
        "backbone": os.environ.get(
            "RLAT_STAGE1_MODEL_PATH",
            str(
                Path(__file__).resolve().parents[2]
                / ".cache"
                / "rlat"
                / "finetuned"
                / "bge-large-en-v1.5-ms-marco-stage1"
                / "final"
            ),
        ),
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "pooling": "cls",
        "max_length": 512,
    },
    # ── Code-strong presets ──────────────────────────────────────────
    "gte-large": {
        "backbone": "Alibaba-NLP/gte-large-en-v1.5",
        "query_prefix": "",
        "passage_prefix": "",
        "pooling": "mean",
        "max_length": 8192,
    },
    "gte-qwen2-1.5b": {
        "backbone": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "query_prefix": "",
        "passage_prefix": "",
        "pooling": "mean",
        "max_length": 8192,
        "trust_remote_code": False,
    },
    "jina-v3": {
        "backbone": "jinaai/jina-embeddings-v3",
        "query_prefix": "",
        "passage_prefix": "",
        "pooling": "mean",
        "max_length": 8192,
    },
}


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for the multi-scale encoder.

    Attributes:
        backbone: HuggingFace model name for the shared transformer backbone.
        bands: Number of frequency bands to produce.
        dim: Output dimensionality per band (D).
        sparsity: Target sparsity (fraction of non-zero components) per band.
            Length must equal bands. If None, uses defaults from spec.
        backbone_dim: Hidden dimension of the backbone (auto-detected if None).
        query_prefix: Text prepended to queries before encoding (E5: "query: ").
        passage_prefix: Text prepended to passages before encoding (E5: "passage: ").
        pooling: Pooling strategy for backbone output. "mean" uses attention-masked
            mean pooling (E5 recipe, default for new knowledge models). "cls" uses [CLS]
            token. Legacy knowledge models loaded without protocol metadata fall back to
            "cls" for backward compatibility.
        max_length: Maximum token length for backbone tokenization. Defaults to 512
            for backward compatibility; presets override this per model.
    """
    backbone: str = "intfloat/e5-large-v2"
    bands: int = 2  # MVP default: topic + entity
    dim: int = 2048
    sparsity: tuple[float, ...] | None = None
    backbone_dim: int | None = None  # Auto-detected from model
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    pooling: str = "mean"  # "mean" (E5 recipe) or "cls"; legacy fallback is "cls"
    max_length: int = 512

    def __post_init__(self) -> None:
        if self.sparsity is not None and len(self.sparsity) != self.bands:
            raise ValueError(
                f"sparsity length ({len(self.sparsity)}) must equal bands ({self.bands})"
            )

    def get_sparsity(self, band_index: int) -> float:
        """Get the sparsity target for a given band (0-indexed)."""
        if self.sparsity is not None:
            return self.sparsity[band_index]
        # Map to 1-indexed band numbers for the default lookup
        band_num = band_index + 1
        if self.bands == 2:
            # MVP: band 0 = Topic (Omega_2), band 1 = Entity (Omega_4)
            return DEFAULT_SPARSITY_TARGETS[2 if band_index == 0 else 4]
        return DEFAULT_SPARSITY_TARGETS.get(band_num, 0.10)


@dataclass(frozen=True)
class MaterialiserConfig:
    """Configuration for the context materialiser.

    Attributes:
        token_budget: Total token budget for the assembled context.
        landscape_tokens: Tokens allocated to landscape context (Omega_1-2).
        structure_tokens: Tokens allocated to structural context (Omega_3).
        evidence_tokens: Tokens allocated to evidence passages (Omega_4-5).
    """
    token_budget: int = 3000
    landscape_tokens: int = 300
    structure_tokens: int = 400
    evidence_tokens: int = 2000

    def __post_init__(self) -> None:
        allocated = self.landscape_tokens + self.structure_tokens + self.evidence_tokens
        if allocated > self.token_budget:
            raise ValueError(
                f"Allocated tokens ({allocated}) exceed budget ({self.token_budget})"
            )


@dataclass
class SalienceWeights:
    """Weights for computing source salience (alpha_i).

    salience = recency * authority * density * novelty
    Each component is in [0, 1].
    """
    recency: float = 1.0
    authority: float = 1.0
    density: float = 1.0
    novelty: float = 1.0

    @property
    def value(self) -> float:
        """Compute the combined salience weight."""
        return self.recency * self.authority * self.density * self.novelty


# ─────────────────────────────────────────────────────────────────────
# C6: ReaderConfig — project-level defaults for `rlat ask --reader ...`
# ─────────────────────────────────────────────────────────────────────
#
# Stored in .rlat.toml alongside other project settings so a team
# can set a default reader (e.g. anthropic for the main project,
# local for privacy-sensitive skills) without every developer having
# to remember the flag set. CLI args override config values; config
# values override built-in defaults.


READER_TOML_SECTION = "reader"


@dataclass(frozen=True)
class ReaderConfig:
    """Project-level defaults for `rlat ask` reader synthesis.

    Attributes are the same names as the `rlat ask` CLI flags (with
    underscores), so readers of .rlat.toml can map between them
    without a translation table.

    Attributes:
        reader:      "off" | "context" | "llm". Default reader mode.
        backend:     "auto" | "local" | "anthropic" | "openai".
        model:       Backend-specific model id; None to use backend default.
        max_tokens:  Generation cap for --reader llm.
        temperature: Sampling temperature for --reader llm.
        system_prompt: Optional override for the grounded-assistant
            system prompt. Use to specialise rlat ask for a particular
            task (code review, document QA, etc.).
        expand:      "off" | "natural" | "max". Retrieval-side context
            expansion; matches --expand on rlat search.
        hybrid:      "off" | "on" | "auto". Hybrid retrieval; matches
            --hybrid on rlat search.

    Validation is deliberately lenient — unknown keys in the TOML
    file are ignored (forward-compat), but values that don't match
    the expected enum raise at load time so typos surface early.
    """
    reader: str = "off"
    backend: str = "auto"
    model: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.3
    system_prompt: str | None = None
    expand: str = "natural"
    hybrid: str = "auto"

    def __post_init__(self) -> None:
        if self.reader not in ("off", "context", "llm"):
            raise ValueError(
                f"ReaderConfig.reader must be off|context|llm, got {self.reader!r}"
            )
        if self.backend not in ("auto", "local", "anthropic", "openai"):
            raise ValueError(
                f"ReaderConfig.backend must be auto|local|anthropic|openai, "
                f"got {self.backend!r}"
            )
        if self.expand not in ("off", "natural", "max"):
            raise ValueError(
                f"ReaderConfig.expand must be off|natural|max, got {self.expand!r}"
            )
        if self.hybrid not in ("off", "on", "auto"):
            raise ValueError(
                f"ReaderConfig.hybrid must be off|on|auto, got {self.hybrid!r}"
            )
        if self.max_tokens <= 0:
            raise ValueError(
                f"ReaderConfig.max_tokens must be positive, got {self.max_tokens}"
            )
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(
                f"ReaderConfig.temperature must be in [0, 2], got {self.temperature}"
            )


def find_rlat_toml(start: Path | str | None = None) -> Path | None:
    """Walk upward from `start` looking for `.rlat.toml`.

    Stops at the filesystem root. Returns None when no file is found,
    so callers can treat "no config" and "config explicitly empty" the
    same way (fall back to defaults).

    Import is local so `config.py` doesn't eagerly import pathlib at
    module load — it's a data module first and foremost.
    """
    from pathlib import Path as _Path
    start_path = _Path(start) if start else _Path.cwd()
    start_path = start_path.resolve()
    if start_path.is_file():
        start_path = start_path.parent
    current = start_path
    while True:
        candidate = current / ".rlat.toml"
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def load_reader_config(
    path: Path | str | None = None,
    *,
    search_from: Path | str | None = None,
) -> ReaderConfig:
    """Load the `[reader]` section from a `.rlat.toml` file.

    Args:
        path: Explicit path to a TOML file. If provided, must exist.
        search_from: Starting dir for auto-discovery. Ignored when
            `path` is given. Defaults to current working directory.

    Returns:
        A ReaderConfig. Defaults when no file is found or the file
        has no `[reader]` section — never raises for "missing config"
        cases, only for malformed ones (invalid TOML or invalid
        enum values).
    """
    import tomllib
    from pathlib import Path as _Path

    if path is not None:
        toml_path = _Path(path)
        if not toml_path.is_file():
            raise FileNotFoundError(f".rlat.toml not found at: {toml_path}")
    else:
        toml_path = find_rlat_toml(search_from)

    if toml_path is None:
        return ReaderConfig()

    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"malformed .rlat.toml at {toml_path}: {e}") from e

    section = data.get(READER_TOML_SECTION) or {}
    if not isinstance(section, dict):
        # A scalar under [reader] is a config error — surface it.
        raise ValueError(
            f"{toml_path}: [{READER_TOML_SECTION}] must be a table"
        )

    # Only pass keys that match ReaderConfig fields; silently drop
    # unknown keys for forward-compat.
    field_names = {f for f in ReaderConfig.__dataclass_fields__}
    kwargs = {k: v for k, v in section.items() if k in field_names}
    return ReaderConfig(**kwargs)
