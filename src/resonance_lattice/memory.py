# SPDX-License-Identifier: BUSL-1.1
"""ResonanceMemory: Field-based corpus memory for LLM augmentation.

The core primitive of the "Field as LLM Hippocampus" architecture.

Instead of RAG (retrieve passages → stuff into context → hope):
    Query → Field → Resonance vector → Memory tokens → LLM

The resonance vector r = F @ q is a synthesised embedding of everything
the corpus knows about the query. It's not a pointer to documents — it's
the field's "memory" of the query, blended across all sources.

Two projection modes:
1. TextProjector: Converts resonance to a compressed natural language
   summary (works with any LLM, no fine-tuning). ~100-200 tokens.
2. EmbeddingProjector: Maps resonance vectors to LLM embedding space
   as soft prompt tokens (requires LLM fine-tuning). 5-10 tokens.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class MemoryReadout(NamedTuple):
    """Result of reading from the resonance memory."""
    resonance_vectors: NDArray   # (B, D) — per-band resonance
    fused: NDArray               # (D,) — weighted fusion across bands
    band_energies: NDArray       # (B,) — energy per band
    top_source_indices: NDArray  # (K,) — indices of top-K resonating sources
    top_source_scores: NDArray   # (K,) — scores of top-K sources
    whitened: bool               # Whether whitening was applied


class ResonanceMemory:
    """Fixed-size corpus memory that produces LLM-injectable context.

    Wraps a dense field with whitening and source scoring to produce
    memory readouts that can be projected into LLM context.

    Usage:
        memory = ResonanceMemory.from_phases(passage_phases)
        readout = memory.query(query_phase)
        # Then use a projector to convert readout → LLM context
    """

    def __init__(
        self,
        field_tensor: NDArray,
        passage_phases: NDArray,
        whitening_alpha: float = 0.7,
        epsilon: float = 1e-6,
        whitening_mode: str = "power_law",
        eml_params: tuple[float, float, float, float] | None = None,
    ) -> None:
        """
        Args:
            field_tensor: Shape (B, D, D) — the dense field.
            passage_phases: Shape (N, B, D) — all source phase vectors.
            whitening_alpha: Power-law whitening exponent (0.7 = ATH).
            epsilon: Regularisation for whitening.
            whitening_mode: "power_law" (default) or "eml".
            eml_params: (a, b, c, d) for EML whitening.
                f(λ) = exp(a·λ + b) - ln(max(c·λ + d, ε)).
                Only used when whitening_mode="eml".
        """
        self.B, self.D, _ = field_tensor.shape
        self.N = passage_phases.shape[0]
        self.field = field_tensor
        self.passage_phases = passage_phases
        self.alpha = whitening_alpha

        # Pre-compute whitening matrices
        self.whitening = np.zeros((self.B, self.D, self.D), dtype=np.float32)
        for b in range(self.B):
            cov = field_tensor[b] / self.N
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 0)

            if whitening_mode == "eml" and eml_params is not None:
                a, b_param, c, d = eml_params
                # EML whitening: scale = eml_filter(λ)
                # f(λ) = exp(a·λ + b) - ln(max(c·λ + d, ε))
                exp_term = np.exp(a * eigvals + b_param)
                log_arg = np.maximum(c * eigvals + d, epsilon)
                scale = exp_term - np.log(log_arg)
                # Ensure positive scale for valid whitening
                scale = np.maximum(scale, epsilon)
            else:
                # Original power-law whitening
                scale = 1.0 / (eigvals ** whitening_alpha + epsilon)

            self.whitening[b] = (
                eigvecs @ np.diag(scale.astype(np.float32)) @ eigvecs.T
            )

    @classmethod
    def from_phases(
        cls,
        passage_phases: NDArray,
        whitening_alpha: float = 0.7,
        whitening_mode: str = "power_law",
        eml_params: tuple[float, float, float, float] | None = None,
    ) -> ResonanceMemory:
        """Build memory from raw phase vectors.

        Args:
            passage_phases: Shape (N, B, D).
            whitening_alpha: Whitening strength.
            whitening_mode: "power_law" (default) or "eml".
            eml_params: (a, b, c, d) for EML whitening.
        """
        N, B, D = passage_phases.shape
        field = np.zeros((B, D, D), dtype=np.float32)
        for b in range(B):
            X = passage_phases[:, b, :]
            field[b] = X.T @ X
        return cls(field, passage_phases, whitening_alpha,
                   whitening_mode=whitening_mode, eml_params=eml_params)

    @property
    def size_mb(self) -> float:
        return self.field.nbytes / (1024 * 1024)

    def query(
        self,
        query_phase: NDArray,
        top_k: int = 10,
        use_whitening: bool = True,
    ) -> MemoryReadout:
        """Query the memory and produce a readout.

        Args:
            query_phase: Shape (B, D) — query phase spectrum.
            top_k: Number of top sources to identify.
            use_whitening: Whether to apply whitening.

        Returns:
            MemoryReadout with resonance vectors and top source indices.
        """
        resonance = np.zeros((self.B, self.D), dtype=np.float32)
        band_energies = np.zeros(self.B, dtype=np.float32)

        for b in range(self.B):
            if use_whitening:
                q_w = self.whitening[b] @ query_phase[b]
                r_b = self.field[b] @ q_w
            else:
                r_b = self.field[b] @ query_phase[b]

            resonance[b] = r_b
            band_energies[b] = np.linalg.norm(r_b)

        # Fuse across bands (energy-weighted)
        total_energy = band_energies.sum() + 1e-12
        band_weights = band_energies / total_energy
        fused = np.zeros(self.D, dtype=np.float32)
        for b in range(self.B):
            fused += band_weights[b] * resonance[b]

        # Score all sources against the resonance
        scores = np.zeros(self.N, dtype=np.float32)
        for b in range(self.B):
            scores += self.passage_phases[:, b, :] @ resonance[b]

        # Top-K sources
        top_idx = np.argsort(scores)[-top_k:][::-1]
        top_scores = scores[top_idx]

        return MemoryReadout(
            resonance_vectors=resonance,
            fused=fused,
            band_energies=band_energies,
            top_source_indices=top_idx,
            top_source_scores=top_scores,
            whitened=use_whitening,
        )

    def resonance_fingerprint(self, readout: MemoryReadout) -> dict:
        """Extract interpretable diagnostics from a memory readout.

        Returns per-band statistics that characterise what the memory
        "remembers" about this query.
        """
        bands = {}
        band_names = ["domain", "topic", "relations", "entity", "verbatim"]

        for b in range(self.B):
            r = readout.resonance_vectors[b]
            energy = readout.band_energies[b]

            # Sparsity of resonance (how focused is the response?)
            nonzero = np.count_nonzero(np.abs(r) > 1e-6)
            sparsity = nonzero / self.D

            # Concentration: how much energy in top-k components?
            top_k_energy = np.sum(np.sort(np.abs(r))[-20:] ** 2)
            total_energy = np.sum(r ** 2) + 1e-12
            concentration = top_k_energy / total_energy

            name = band_names[b] if b < len(band_names) else f"band_{b}"
            bands[name] = {
                "energy": float(energy),
                "sparsity": float(sparsity),
                "concentration_top20": float(concentration),
            }

        return bands
