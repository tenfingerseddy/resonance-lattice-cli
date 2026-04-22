# SPDX-License-Identifier: BUSL-1.1
"""Field X-Ray: corpus-level semantic diagnostics.

Surfaces the field's spectral, information-theoretic, and signal-quality
properties through curated RQL pipelines. Every metric here is uniquely
possible because the corpus IS a matrix, not a vector index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.field.dense import DenseField
from resonance_lattice.rql.cache import SpectralCache
from resonance_lattice.rql.info import InfoOps
from resonance_lattice.rql.signal_ops import SignalOps
from resonance_lattice.rql.spectral import SpectralOps
from resonance_lattice.rql.stats import StatsOps


@dataclass
class BandHealth:
    name: str
    band: int
    label: str  # "rich", "adequate", "thin", "noisy"
    effective_rank: float
    entropy: float
    spectral_gap: float
    snr: float
    condition: float
    purity: float
    vn_entropy: float
    energy_compaction_20: float
    n_signal_eigenvalues: int


@dataclass
class XRayResult:
    lattice_path: str
    source_count: int
    bands: int
    dim: int
    snr: float
    band_health: list[BandHealth]
    band_correlation: NDArray[np.float32]
    total_signal_eigenvalues: int
    saturation: float  # 0-1
    diagnostics: list[str]
    # Deep mode extras
    communities: dict | None = None
    topology: list | None = None

    def to_dict(self) -> dict:
        return {
            "lattice": self.lattice_path,
            "source_count": self.source_count,
            "bands": self.bands,
            "dim": self.dim,
            "snr": round(self.snr, 1),
            "saturation": round(self.saturation, 3),
            "total_signal_eigenvalues": self.total_signal_eigenvalues,
            "band_health": [
                {
                    "name": bh.name, "band": bh.band, "label": bh.label,
                    "effective_rank": round(bh.effective_rank, 1),
                    "entropy": round(bh.entropy, 4),
                    "spectral_gap": round(bh.spectral_gap, 4),
                    "snr": round(bh.snr, 1),
                    "condition": round(bh.condition, 1),
                    "purity": round(bh.purity, 6),
                    "vn_entropy": round(bh.vn_entropy, 4),
                    "energy_compaction_20": round(bh.energy_compaction_20, 4),
                    "n_signal_eigenvalues": bh.n_signal_eigenvalues,
                }
                for bh in self.band_health
            ],
            "band_correlation": self.band_correlation.tolist(),
            "diagnostics": self.diagnostics,
            "communities": self.communities,
        }

    def to_text(self) -> str:
        lines = [f"Field X-Ray: {self.lattice_path}"]
        lines.append(f"  Sources: {self.source_count:,} | Bands: {self.bands}x{self.dim} | SNR: {self.snr:.1f}")
        lines.append("")

        lines.append("  Per-Band Health")
        for bh in self.band_health:
            tag = {"rich": "RICH", "adequate": "ADEQ", "thin": "THIN", "noisy": "NOISY"}[bh.label]
            lines.append(
                f"    {bh.name:<12} [{tag:<5}]  eff_rank={bh.effective_rank:>6.1f}"
                f"  entropy={bh.entropy:.2f}  SNR={bh.snr:>5.1f}  gap={bh.spectral_gap:.4f}"
            )
        lines.append("")

        lines.append(f"  Signal: {self.total_signal_eigenvalues} eigenvalues above Marchenko-Pastur bulk")

        # Band coupling summary: top 2 correlations
        B = self.band_correlation.shape[0]
        pairs = []
        for i in range(B):
            for j in range(i + 1, B):
                pairs.append((self.band_correlation[i, j], i, j))
        pairs.sort(reverse=True)
        if pairs:
            coupling_parts = []
            for corr, i, j in pairs[:3]:
                coupling_parts.append(f"{self.band_health[i].name}-{self.band_health[j].name} r={corr:.2f}")
            lines.append(f"  Band coupling: {', '.join(coupling_parts)}")

        remaining = max(0, int(self.source_count * (1 / max(self.saturation, 0.01) - 1)))
        lines.append(f"  Saturation: {self.saturation:.0%} (field has capacity for ~{remaining:,} more sources)")
        lines.append("")

        if self.diagnostics:
            lines.append("  Diagnostics")
            for d in self.diagnostics:
                lines.append(f"    {d}")
            lines.append("")

        if self.communities and self.communities.get("communities"):
            lines.append(f"  Topic Communities ({len(self.communities['communities'])} detected)")
            for c in self.communities["communities"][:8]:
                pct = c["fraction"] * 100
                reps = c["representatives"][:3]
                lines.append(f"    {c['size']:>5} sources ({pct:4.1f}%)  coherence={c['coherence']:.2f}  {', '.join(reps)}")
            lines.append("")

        return "\n".join(lines)

    def to_prompt(self) -> str:
        lines = [f"## Corpus X-Ray: {self.lattice_path}"]
        lines.append(f"Sources: {self.source_count:,} | {self.bands} bands x {self.dim}d | SNR {self.snr:.1f}")
        lines.append("")

        rich = [bh for bh in self.band_health if bh.label == "rich"]
        weak = [bh for bh in self.band_health if bh.label in ("thin", "noisy")]
        if rich:
            lines.append(f"**Strengths**: {', '.join(bh.name for bh in rich)} bands are information-rich.")
        if weak:
            lines.append(f"**Weaknesses**: {', '.join(bh.name for bh in weak)} bands are {'noisy' if any(b.label == 'noisy' for b in weak) else 'thin'}.")

        remaining = max(0, int(self.source_count * (1 / max(self.saturation, 0.01) - 1)))
        lines.append(f"**Saturation**: {self.saturation:.0%} — room for ~{remaining:,} more sources.")
        lines.append(f"**Signal eigenvalues**: {self.total_signal_eigenvalues} above noise floor.")

        if self.diagnostics:
            lines.append("")
            for d in self.diagnostics:
                lines.append(f"- {d}")

        return "\n".join(lines)


class FieldXRay:

    @staticmethod
    def quick(field: DenseField, source_count: int, lattice_path: str = "") -> XRayResult:
        cache = SpectralCache(field)
        B = field.bands
        from resonance_lattice.lattice import BAND_NAMES
        band_names = list(BAND_NAMES[:B])
        if B > len(BAND_NAMES):
            band_names.extend(f"band_{i}" for i in range(len(BAND_NAMES), B))

        band_health: list[BandHealth] = []
        total_signal = 0
        total_eff_rank = 0.0
        diagnostics: list[str] = []

        for b in range(B):
            eff_rank = SpectralOps.effective_rank(field, b, cache).value
            entropy = SpectralOps.entropy(field, b, cache).value
            gap = SpectralOps.spectral_gap(field, b, cache).value
            snr = SignalOps.snr_estimate(field, b, cache).value
            cond = SpectralOps.condition_number(field, b, cache).value
            purity = InfoOps.purity(field, b, cache).value
            vn_ent = InfoOps.von_neumann_entropy(field, b, cache).value
            compaction = SignalOps.energy_compaction(field, 20, b, cache).value
            mp = StatsOps.marchenko_pastur_fit(field, b)
            n_signal = mp["n_signal_eigenvalues"]

            # Classify band health
            if snr >= 30 and eff_rank >= 100:
                label = "rich"
            elif snr >= 10 and eff_rank >= 30:
                label = "adequate"
            elif snr < 10:
                label = "noisy"
            else:
                label = "thin"

            band_health.append(BandHealth(
                name=band_names[b], band=b, label=label,
                effective_rank=eff_rank, entropy=entropy,
                spectral_gap=gap, snr=snr, condition=cond,
                purity=purity, vn_entropy=vn_ent,
                energy_compaction_20=compaction,
                n_signal_eigenvalues=n_signal,
            ))

            total_signal += n_signal
            total_eff_rank += eff_rank

            # Diagnostics
            if snr < 10:
                diagnostics.append(f"! {band_names[b]}: SNR below 10 — retrieval unreliable")
            if cond > 1e6:
                diagnostics.append(f"! {band_names[b]}: condition number {cond:.0f} — numerically unstable")
            if eff_rank < 20:
                diagnostics.append(f"! {band_names[b]}: low effective rank ({eff_rank:.0f}) — limited diversity")
            if gap > 0.03:
                diagnostics.append(f"* {band_names[b]}: healthy spectral gap — well-separated topics")

        # Band correlation matrix
        band_corr = InfoOps.band_correlation_matrix(field)

        # Saturation: mean(effective_rank / dim) across bands
        saturation = total_eff_rank / (B * field.dim) if field.dim > 0 else 0.0

        # Overall SNR
        overall_snr = float(np.mean([bh.snr for bh in band_health]))

        return XRayResult(
            lattice_path=lattice_path,
            source_count=source_count,
            bands=B,
            dim=field.dim,
            snr=overall_snr,
            band_health=band_health,
            band_correlation=band_corr,
            total_signal_eigenvalues=total_signal,
            saturation=saturation,
            diagnostics=diagnostics,
        )

    @staticmethod
    def deep(field: DenseField, source_count: int, lattice_path: str = "",
             lattice=None) -> XRayResult:
        result = FieldXRay.quick(field, source_count, lattice_path)

        # Community detection
        if lattice is not None:
            try:
                result.communities = lattice.detect_communities(n_communities=8, band=0)
            except Exception:
                pass

        # Topological analysis
        from resonance_lattice.topology import TopologicalAnalyzer
        result.topology = []
        for b in range(field.bands):
            try:
                topo = TopologicalAnalyzer.analyze(field, band=b)
                result.topology.append({
                    "band": b,
                    "knowledge_clusters": topo.knowledge_clusters,
                    "circular_patterns": topo.circular_patterns,
                    "robustness": round(topo.robustness_score, 4),
                })
            except Exception:
                pass

        return result
