# SPDX-License-Identifier: BUSL-1.1
"""Probe Recipes: curated RQL pipelines for quick semantic insights.

Each recipe composes 2-5 existing RQL operations into a named analysis
that answers a specific question about the field or a query's position in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.field.dense import DenseField
from resonance_lattice.rql.cache import SpectralCache
from resonance_lattice.rql.info import InfoOps
from resonance_lattice.rql.query_ops import QueryOps
from resonance_lattice.rql.signal_ops import SignalOps
from resonance_lattice.rql.spectral import SpectralOps
from resonance_lattice.rql.stats import StatsOps


def _band_names(B: int) -> list[str]:
    from resonance_lattice.lattice import BAND_NAMES
    names = list(BAND_NAMES[:B])
    if B > len(BAND_NAMES):
        names.extend(f"band_{i}" for i in range(len(BAND_NAMES), B))
    return names


def _bar(value: float, max_val: float, width: int = 10) -> str:
    import sys
    frac = min(1.0, max(0.0, value / (max_val + 1e-12)))
    filled = int(width * frac)
    try:
        "\u2588".encode(sys.stdout.encoding or "utf-8")
        return "\u2588" * filled + "\u2591" * (width - filled)
    except (UnicodeEncodeError, LookupError):
        return "#" * filled + "-" * (width - filled)


@dataclass
class ProbeResult:
    recipe: str
    metrics: dict[str, Any]
    interpretation: str
    detail: dict[str, Any]

    def to_dict(self) -> dict:
        return {"recipe": self.recipe, "metrics": self.metrics,
                "interpretation": self.interpretation, "detail": self.detail}

    def to_text(self) -> str:
        lines = [f"  {self.recipe.title()} Analysis", ""]
        for key, val in self.metrics.items():
            if isinstance(val, dict):
                lines.append(f"  {key}:")
                for k2, v2 in val.items():
                    lines.append(f"    {k2:<12} {v2}")
            elif isinstance(val, list):
                lines.append(f"  {key}: {val}")
            else:
                lines.append(f"  {key}: {val}")
        lines.append("")
        lines.append(f"  {self.interpretation}")
        return "\n".join(lines)

    def to_prompt(self) -> str:
        lines = [f"## Probe: {self.recipe}"]
        for key, val in self.metrics.items():
            if isinstance(val, dict):
                parts = [f"{k}: {v}" for k, v in val.items()]
                lines.append(f"**{key}**: {', '.join(parts)}")
            else:
                lines.append(f"**{key}**: {val}")
        lines.append("")
        lines.append(self.interpretation)
        return "\n".join(lines)


class ProbeRecipes:

    @staticmethod
    def health(field: DenseField) -> ProbeResult:
        """Marchenko-Pastur signal/noise split, SNR, effective rank."""
        cache = SpectralCache(field)
        B = field.bands
        names = _band_names(B)
        per_band = {}

        for b in range(B):
            mp = StatsOps.marchenko_pastur_fit(field, b)
            snr = SignalOps.snr_estimate(field, b, cache).value
            eff_rank = SpectralOps.effective_rank(field, b, cache).value
            noise = SignalOps.noise_floor(field, b, cache).value
            per_band[names[b]] = (
                f"signal={mp['n_signal_eigenvalues']:>3}  "
                f"bulk={mp['n_bulk_eigenvalues']:>4}  "
                f"SNR={snr:>5.1f}  eff_rank={eff_rank:>6.1f}  "
                f"noise_floor={noise:.4f}"
            )

        total_signal = sum(
            StatsOps.marchenko_pastur_fit(field, b)["n_signal_eigenvalues"]
            for b in range(B)
        )

        return ProbeResult(
            recipe="health",
            metrics={"per_band": per_band, "total_signal_eigenvalues": total_signal},
            interpretation=f"{total_signal} eigenvalues carry real semantic signal across {B} bands.",
            detail={"marchenko_pastur": {
                names[b]: StatsOps.marchenko_pastur_fit(field, b) for b in range(B)
            }},
        )

    @staticmethod
    def novelty(field: DenseField, phase_vectors: NDArray[np.float32],
                text: str = "") -> ProbeResult:
        """How novel is this content relative to the corpus?"""
        from resonance_lattice.algebra import FieldAlgebra
        result = FieldAlgebra.novelty(field, phase_vectors)
        B = field.bands
        names = _band_names(B)

        per_band = {}
        max(float(np.max(result.per_band)), 0.01)
        for b in range(B):
            nov = float(result.per_band[b])
            per_band[names[b]] = f"{_bar(nov, 1.0)}  {nov:.2f}"

        if result.score > 0.7:
            label = "mostly new to this corpus"
        elif result.score > 0.4:
            label = "partially covered"
        elif result.score > 0.15:
            label = "mostly familiar"
        else:
            label = "already well-covered"

        recommendation = "ADD — fills a real gap" if result.score > 0.5 else (
            "OPTIONAL — some new information" if result.score > 0.2 else
            "SKIP — redundant with existing content"
        )

        display = f'"{text}"' if text else "input"
        return ProbeResult(
            recipe="novelty",
            metrics={
                "overall": f"{result.score:.2f} ({label})",
                "per_band": per_band,
                "information_gain": f"{result.information_gain_estimate:.1f} bits",
                "recommendation": recommendation,
            },
            interpretation=f"Novelty {result.score:.2f} for {display}. {recommendation}.",
            detail={
                "score": result.score,
                "per_band": {names[b]: round(float(result.per_band[b]), 4) for b in range(B)},
                "self_energy": result.self_energy,
                "projection_energy": result.projection_energy,
                "information_gain": result.information_gain_estimate,
            },
        )

    @staticmethod
    def saturation(field: DenseField) -> ProbeResult:
        """Is the field saturated? How much capacity remains?"""
        cache = SpectralCache(field)
        B = field.bands
        names = _band_names(B)
        per_band = {}
        total_sat = 0.0

        for b in range(B):
            eff_rank = SpectralOps.effective_rank(field, b, cache).value
            InfoOps.channel_capacity(field, b, cache).value
            SpectralOps.entropy(field, b, cache).value
            sat = eff_rank / field.dim
            total_sat += sat
            pct = sat * 100
            per_band[names[b]] = f"{_bar(sat, 1.0, 20)}  {pct:>5.1f}%"

        avg_sat = total_sat / B
        remaining = max(0, int(field.source_count * (1 / max(avg_sat, 0.01) - 1)))

        return ProbeResult(
            recipe="saturation",
            metrics={
                "overall": f"{avg_sat:.0%} ({field.source_count:,} sources)",
                "per_band": per_band,
                "estimated_capacity_remaining": f"~{remaining:,} sources",
            },
            interpretation=(
                f"Field is {avg_sat:.0%} saturated. "
                f"{'Adding more sources would have diminishing returns.' if avg_sat > 0.8 else f'Room for ~{remaining:,} more sources.'}"
            ),
            detail={
                "saturation": avg_sat,
                "per_band": {
                    names[b]: round(SpectralOps.effective_rank(field, b, cache).value / field.dim, 4)
                    for b in range(B)
                },
            },
        )

    @staticmethod
    def band_flow(field: DenseField) -> ProbeResult:
        """Inter-band mutual information matrix."""
        B = field.bands
        names = _band_names(B)

        mi_matrix = np.zeros((B, B), dtype=np.float32)
        for i in range(B):
            for j in range(i + 1, B):
                mi = InfoOps.mutual_information_bands(field, i, j).value
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        corr = InfoOps.band_correlation_matrix(field)

        # Find strongest and weakest links
        pairs = []
        for i in range(B):
            for j in range(i + 1, B):
                pairs.append((mi_matrix[i, j], names[i], names[j]))
        pairs.sort(reverse=True)

        strongest = pairs[:3] if pairs else []
        weakest = pairs[-2:] if len(pairs) >= 2 else []

        summary = {}
        for mi_val, a, b in strongest:
            summary[f"{a}-{b}"] = f"MI={mi_val:.3f}  corr={corr[names.index(a), names.index(b)]:.3f}"

        return ProbeResult(
            recipe="band-flow",
            metrics={"strongest_links": summary},
            interpretation=(
                f"Strongest inter-band coupling: {strongest[0][1]}-{strongest[0][2]} (MI={strongest[0][0]:.3f}). "
                f"Weakest: {weakest[-1][1]}-{weakest[-1][2]} (MI={weakest[-1][0]:.3f})."
                if strongest and weakest else "Insufficient bands for flow analysis."
            ),
            detail={
                "mutual_information": mi_matrix.tolist(),
                "correlation": corr.tolist(),
                "band_names": names,
            },
        )

    @staticmethod
    def anti(field: DenseField, query_phase: NDArray[np.float32],
             text: str = "") -> ProbeResult:
        """What the field does NOT know about this query."""
        B = field.bands
        names = _band_names(B)

        anti = QueryOps.anti_resonate(field, query_phase)
        energies = QueryOps.energy_all_bands(field, query_phase)

        per_band = {}
        for b in range(B):
            anti_norm = float(np.linalg.norm(anti[b]))
            res_norm = float(np.linalg.norm(field.F[b] @ query_phase[b]))
            gap_ratio = anti_norm / (anti_norm + res_norm + 1e-12)
            per_band[names[b]] = f"{_bar(gap_ratio, 1.0)}  gap={gap_ratio:.0%}  energy={float(energies[b]):.2f}"

        total_anti = sum(float(np.linalg.norm(anti[b])) for b in range(B))
        total_res = sum(float(np.linalg.norm(field.F[b] @ query_phase[b])) for b in range(B))
        overall_gap = total_anti / (total_anti + total_res + 1e-12)

        display = f'"{text}"' if text else "query"
        return ProbeResult(
            recipe="anti",
            metrics={
                "overall_gap": f"{overall_gap:.0%}",
                "per_band": per_band,
            },
            interpretation=f"The field lacks {overall_gap:.0%} of {display}'s semantic content.",
            detail={
                "gap_ratio": overall_gap,
                "per_band_gap": {
                    names[b]: round(float(np.linalg.norm(anti[b])) / (float(np.linalg.norm(anti[b])) + float(np.linalg.norm(field.F[b] @ query_phase[b])) + 1e-12), 4)
                    for b in range(B)
                },
            },
        )

    @staticmethod
    def gaps(field: DenseField) -> ProbeResult:
        """Topological knowledge gap analysis."""
        from resonance_lattice.topology import TopologicalAnalyzer
        B = field.bands
        names = _band_names(B)
        per_band = {}

        for b in range(B):
            topo = TopologicalAnalyzer.analyze(field, band=b)
            per_band[names[b]] = (
                f"clusters={topo.knowledge_clusters:>3}  "
                f"loops={topo.circular_patterns}  "
                f"robustness={topo.robustness_score:.3f}"
            )

        return ProbeResult(
            recipe="gaps",
            metrics={"per_band": per_band},
            interpretation="Knowledge clusters indicate well-separated topic groups. Loops indicate potential circular reasoning.",
            detail={
                "per_band": {
                    names[b]: {
                        "clusters": TopologicalAnalyzer.analyze(field, band=b).knowledge_clusters,
                        "loops": TopologicalAnalyzer.analyze(field, band=b).circular_patterns,
                        "robustness": TopologicalAnalyzer.analyze(field, band=b).robustness_score,
                    }
                    for b in range(B)
                },
            },
        )


RECIPES = {
    "health": {"needs_query": False, "fn": "health"},
    "novelty": {"needs_query": True, "fn": "novelty"},
    "saturation": {"needs_query": False, "fn": "saturation"},
    "band-flow": {"needs_query": False, "fn": "band_flow"},
    "anti": {"needs_query": True, "fn": "anti"},
    "gaps": {"needs_query": False, "fn": "gaps"},
}
