# SPDX-License-Identifier: BUSL-1.1
"""Query Locator: position a query within the field's knowledge landscape.

Not search (which returns passages) — structural analysis of how a query
relates to the field's geometry: band energy, anti-resonance, uncertainty,
and expansion direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from resonance_lattice.field.dense import DenseField
from resonance_lattice.rql.info import InfoOps
from resonance_lattice.rql.query_ops import QueryOps
from resonance_lattice.rql.stats import StatsOps


def _safe_bar(fraction: float, width: int = 20) -> str:
    """Render a bar safe for the current terminal encoding."""
    import sys
    fill_len = int(width * max(0.0, min(1.0, fraction)))
    empty_len = width - fill_len
    try:
        "\u2588".encode(sys.stdout.encoding or "utf-8")
        return "\u2588" * fill_len + "\u2591" * empty_len
    except (UnicodeEncodeError, LookupError):
        return "#" * fill_len + "-" * empty_len


@dataclass
class QueryLocation:
    query: str
    band_names: list[str]
    band_energies: NDArray[np.float32]
    band_focus: str  # name of dominant band
    band_focus_pct: float  # percentage
    anti_resonance_ratio: float  # 0-1: fraction of query in gap
    coverage_label: str  # "strong", "partial", "edge", "gap"
    mahalanobis: float  # distance from corpus center
    uncertainty_per_band: list[float]
    fisher_per_band: list[float]
    expansion_hint: str | None  # nearest source via steepest ascent

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "band_energies": {n: round(float(e), 4) for n, e in zip(self.band_names, self.band_energies)},
            "band_focus": self.band_focus,
            "band_focus_pct": round(self.band_focus_pct, 1),
            "anti_resonance_ratio": round(self.anti_resonance_ratio, 4),
            "coverage": self.coverage_label,
            "mahalanobis": round(self.mahalanobis, 2),
            "uncertainty_per_band": [round(u, 4) for u in self.uncertainty_per_band],
            "fisher_per_band": [round(f, 4) for f in self.fisher_per_band],
            "expansion_hint": self.expansion_hint,
        }

    def to_text(self) -> str:
        lines = [f'  "{self.query}"', ""]

        lines.append(f"  Position: {self.coverage_label} (gap ratio {self.anti_resonance_ratio:.0%})")

        # Band focus summary
        total_e = float(np.sum(self.band_energies)) + 1e-12
        focus_parts = []
        for name, e in sorted(zip(self.band_names, self.band_energies), key=lambda x: -x[1]):
            pct = float(e) / total_e * 100
            if pct >= 5:
                focus_parts.append(f"{name} {pct:.0f}%")
        lines.append(f"  Band focus: {' | '.join(focus_parts)}")

        # Gap detail
        weakest_band = self.band_names[int(np.argmin(self.band_energies))]
        lines.append(f"  Knowledge gap: {self.anti_resonance_ratio:.0%} ({weakest_band} band — underrepresented)")
        lines.append("")

        # Energy bars
        lines.append("  Per-Band Energy")
        max_e = float(np.max(self.band_energies)) + 1e-12
        for name, e in zip(self.band_names, self.band_energies):
            bar = _safe_bar(float(e) / max_e)
            primary = "  <-- primary" if name == self.band_focus else ""
            lines.append(f"    {name:<12} {bar}  {float(e):.2f}{primary}")
        lines.append("")

        lines.append(f"  Uncertainty: Mahalanobis {self.mahalanobis:.1f}"
                      f" ({'within 2-sigma' if self.mahalanobis < 2 else 'peripheral' if self.mahalanobis < 4 else 'outlier'})")

        if self.expansion_hint:
            lines.append("")
            lines.append("  Nearby richer query:")
            lines.append(f"    {self.expansion_hint} (via steepest ascent)")

        return "\n".join(lines)

    def to_prompt(self) -> str:
        lines = [f'## Query Position: "{self.query}"']
        lines.append(f"**Coverage**: {self.coverage_label.title()} (gap ratio {self.anti_resonance_ratio:.0%}).")
        lines.append(f"**Band focus**: {self.band_focus} band carries {self.band_focus_pct:.0f}% of the answer.")

        weakest = self.band_names[int(np.argmin(self.band_energies))]
        lines.append(f"**Knowledge gap**: {self.anti_resonance_ratio:.0%} of query content falls outside field coverage, primarily on the {weakest} band.")
        lines.append(f"**Mahalanobis distance**: {self.mahalanobis:.1f} ({'within normal range' if self.mahalanobis < 2 else 'peripheral query'}).")

        if self.expansion_hint:
            lines.append(f"**Suggestion**: The field has denser coverage near \"{self.expansion_hint}\".")

        return "\n".join(lines)


class QueryLocator:

    @staticmethod
    def locate(
        field: DenseField,
        query_phase: NDArray[np.float32],
        query_text: str,
        registry=None,
        store=None,
    ) -> QueryLocation:
        from resonance_lattice.lattice import BAND_NAMES
        B = field.bands
        band_names = list(BAND_NAMES[:B])
        if B > len(BAND_NAMES):
            band_names.extend(f"band_{i}" for i in range(len(BAND_NAMES), B))

        # Per-band energy
        band_energies = QueryOps.energy_all_bands(field, query_phase)

        # Band focus
        total_e = float(np.sum(band_energies)) + 1e-12
        focus_idx = int(np.argmax(band_energies))
        focus_pct = float(band_energies[focus_idx]) / total_e * 100

        # Anti-resonance ratio
        anti = QueryOps.anti_resonate(field, query_phase)
        anti_energy = float(np.sum([np.linalg.norm(anti[b]) for b in range(B)]))
        res_energy = float(np.sum([np.linalg.norm(field.F[b] @ query_phase[b]) for b in range(B)]))
        anti_ratio = anti_energy / (anti_energy + res_energy + 1e-12)

        # Coverage classification
        if anti_ratio < 0.15:
            coverage = "strong coverage"
        elif anti_ratio < 0.35:
            coverage = "partial coverage"
        elif anti_ratio < 0.6:
            coverage = "edge of knowledge"
        else:
            coverage = "knowledge gap"

        # Mahalanobis distance (use the dominant band)
        maha = StatsOps.mahalanobis_distance(field, query_phase, band=focus_idx).value

        # Uncertainty per band
        uncertainties = []
        for b in range(B):
            _, sigma = QueryOps.uncertainty_resonate(field, query_phase, band=b)
            uncertainties.append(sigma)

        # Fisher information per band
        fishers = []
        for b in range(B):
            fi = InfoOps.fisher_information(field, query_phase, band=b).value
            fishers.append(fi)

        # Expansion hint via steepest ascent + registry lookup
        expansion_hint = None
        if registry is not None:
            expanded = QueryOps.steepest_ascent(field, query_phase, alpha=0.3)
            pointers = registry.lookup_bruteforce(query_phase=expanded, top_k=5)
            for ptr in pointers:
                # Find first source that differs from top direct results
                direct = registry.lookup_bruteforce(query_phase=query_phase, top_k=3)
                direct_ids = {p.source_id for p in direct}
                if ptr.source_id not in direct_ids:
                    if store is not None:
                        content = store.retrieve(ptr.source_id)
                        if content and content.summary:
                            expansion_hint = content.summary[:120]
                            break
                    else:
                        expansion_hint = ptr.source_id
                        break

        return QueryLocation(
            query=query_text,
            band_names=band_names,
            band_energies=band_energies,
            band_focus=band_names[focus_idx],
            band_focus_pct=focus_pct,
            anti_resonance_ratio=anti_ratio,
            coverage_label=coverage,
            mahalanobis=maha,
            uncertainty_per_band=uncertainties,
            fisher_per_band=fishers,
            expansion_hint=expansion_hint,
        )
