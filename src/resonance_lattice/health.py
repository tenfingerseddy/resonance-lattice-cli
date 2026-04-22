# SPDX-License-Identifier: BUSL-1.1
"""Knowledge Health: composed diagnostic pipeline.

Combines xray (per-band health, SNR, saturation) with optional
baseline comparison (diff, contradiction detection) into a single
health report.

Usage:
    report = HealthCheck.run(lattice)
    report = HealthCheck.run(lattice, baseline=older_lattice)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from resonance_lattice.lattice import Lattice


@dataclass
class ContradictionEntry:
    """A specific contradiction between two knowledge models."""
    text_a: str
    source_a: str
    text_b: str
    source_b: str


@dataclass
class HealthReport:
    """Composed health report for a knowledge model."""

    # Overall status
    status: str  # "HEALTHY", "WARNING", "CRITICAL"

    # From xray
    snr: float
    saturation: float
    remaining_capacity: int
    band_health: list[dict]  # [{name, label, effective_rank, snr}]
    diagnostics: list[str]

    # From baseline comparison (optional)
    has_baseline: bool = False
    contradiction_count: int = 0
    contradiction_ratio: float = 0.0
    contradictions: list[ContradictionEntry] = field(default_factory=list)
    coverage_changes: list[str] = field(default_factory=list)
    new_topics: list[str] = field(default_factory=list)
    lost_topics: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = ["Health Report"]
        lines.append(f"Status: {self.status}")
        lines.append("")

        # Overall metrics
        snr_label = "excellent" if self.snr > 20 else "good" if self.snr > 10 else "adequate" if self.snr > 5 else "poor"
        lines.append(f"Signal quality: {snr_label} (SNR {self.snr:.1f})")

        rank_labels = [bh for bh in self.band_health if bh["label"] == "rich"]
        if rank_labels:
            lines.append(f"Diversity: good ({len(rank_labels)}/{len(self.band_health)} bands rich)")
        else:
            lines.append("Diversity: limited (no rich bands)")
        lines.append(f"Capacity: {self.saturation:.0%} used (room for ~{self.remaining_capacity:,} more sources)")
        lines.append("")

        # Per-band health
        lines.append("Per-band health:")
        for bh in self.band_health:
            detail = ""
            if bh["label"] == "thin":
                detail = " — knowledge is sparse"
            elif bh["label"] == "noisy":
                detail = f" — signal is noisy (SNR {bh['snr']:.1f})"
            lines.append(f"  {bh['name']}: {bh['label']}{detail}")
        lines.append("")

        # Baseline comparison
        if self.has_baseline:
            lines.append("Changes vs baseline:")
            if self.coverage_changes:
                for change in self.coverage_changes:
                    lines.append(f"  {change}")
            else:
                lines.append("  No significant coverage changes")

            if self.contradiction_count > 0:
                ratio_label = "high" if self.contradiction_ratio > 0.15 else "moderate" if self.contradiction_ratio > 0.05 else "low"
                lines.append(f"Contradictions: {self.contradiction_count} ({ratio_label}, {self.contradiction_ratio:.1%} ratio)")
                for i, c in enumerate(self.contradictions[:5], 1):
                    lines.append(f"  {i}. \"{c.text_a}\" ({c.source_a})")
                    lines.append(f"     vs \"{c.text_b}\" ({c.source_b})")
            else:
                lines.append("Contradictions: none detected")

            if self.new_topics:
                lines.append(f"New topics: {', '.join(self.new_topics)}")
            if self.lost_topics:
                lines.append(f"Lost topics: {', '.join(self.lost_topics)}")
            lines.append("")

        # Diagnostics
        if self.diagnostics:
            lines.append("Diagnostics:")
            for d in self.diagnostics:
                lines.append(f"  {d}")

        return "\n".join(lines)


class HealthCheck:
    """Compose xray + diff + contradict into a health report."""

    @staticmethod
    def run(
        lattice: Lattice,
        baseline: Lattice | None = None,
        lattice_path: str = "",
    ) -> HealthReport:

        from resonance_lattice.field.dense import DenseField
        from resonance_lattice.xray import FieldXRay

        if not isinstance(lattice.field, DenseField):
            return HealthReport(
                status="UNKNOWN",
                snr=0, saturation=0, remaining_capacity=0,
                band_health=[], diagnostics=["Health requires DenseField"],
            )

        # ── Xray ──
        xray = FieldXRay.quick(
            lattice.field,
            lattice.source_count,
            lattice_path=lattice_path,
        )

        band_health = []
        for bh in xray.band_health:
            band_health.append({
                "name": bh.name,
                "label": bh.label,
                "effective_rank": round(bh.effective_rank, 1),
                "snr": round(bh.snr, 1),
            })

        remaining = max(0, int(lattice.source_count * (1 / max(xray.saturation, 0.01) - 1)))

        # ── Determine overall status ──
        noisy_count = sum(1 for bh in xray.band_health if bh.label == "noisy")
        thin_count = sum(1 for bh in xray.band_health if bh.label == "thin")

        status = "HEALTHY"
        if noisy_count >= 2 or xray.snr < 5:
            status = "CRITICAL"
        elif noisy_count >= 1 or thin_count >= 2 or xray.snr < 10:
            status = "WARNING"

        report = HealthReport(
            status=status,
            snr=xray.snr,
            saturation=xray.saturation,
            remaining_capacity=remaining,
            band_health=band_health,
            diagnostics=xray.diagnostics,
        )

        # ── Baseline comparison (optional) ──
        if baseline is not None and isinstance(baseline.field, DenseField):
            if baseline.field.bands == lattice.field.bands and baseline.field.dim == lattice.field.dim:
                report.has_baseline = True
                _add_baseline_comparison(report, lattice, baseline)

        return report


def _add_baseline_comparison(
    report: HealthReport,
    current: Lattice,
    baseline: Lattice,
) -> None:
    """Add diff + contradiction analysis to the health report."""
    import numpy as np

    from resonance_lattice.algebra import FieldAlgebra
    from resonance_lattice.lattice import BAND_NAMES

    # ── Coverage changes per band ──
    B = current.field.bands
    for b in range(B):
        energy_current = float(np.linalg.norm(current.field.F[b], "fro"))
        energy_baseline = float(np.linalg.norm(baseline.field.F[b], "fro"))
        if energy_baseline > 1e-6:
            pct_change = (energy_current - energy_baseline) / energy_baseline * 100
            name = BAND_NAMES[b] if b < len(BAND_NAMES) else f"band_{b}"
            if abs(pct_change) > 5:
                direction = "+" if pct_change > 0 else ""
                report.coverage_changes.append(
                    f"{name}: {direction}{pct_change:.0f}% energy"
                )

    # ── Contradiction detection ──
    try:
        contradiction = FieldAlgebra.contradict(current.field, baseline.field)
        report.contradiction_ratio = contradiction.contradiction_ratio

        if contradiction.contradiction_ratio > 0.01:
            # Surface specific contradictions
            contradictions = _find_contradiction_passages(
                contradiction, current, baseline, max_pairs=5,
            )
            report.contradictions = contradictions
            report.contradiction_count = len(contradictions)

            if report.contradiction_count > 0 and report.status == "HEALTHY":
                report.status = "WARNING"
            if report.contradiction_ratio > 0.15 and report.status != "CRITICAL":
                report.status = "CRITICAL"
    except Exception:
        pass  # contradiction detection is best-effort


def _find_contradiction_passages(
    contradiction,
    current: Lattice,
    baseline: Lattice,
    max_pairs: int = 5,
) -> list[ContradictionEntry]:
    """Find specific passages that contradict between current and baseline."""
    import numpy as np

    cfield = contradiction.contradiction_field
    entries = []

    if current.registry is None or baseline.registry is None:
        return entries
    if current.store is None or baseline.store is None:
        return entries

    def _top_sources(registry, n=5):
        scored = []
        for source_id, entry in registry._source_index.items():
            res = cfield.resonate(entry.phase_vectors)
            energy = float(np.sum(res.band_energies))
            if energy > 1e-6:
                scored.append((source_id, energy))
        scored.sort(key=lambda x: -x[1])
        return scored[:n]

    top_current = _top_sources(current.registry)
    top_baseline = _top_sources(baseline.registry)

    if not top_current or not top_baseline:
        return entries

    found = 0
    for sid_a, _ in top_current:
        content_a = current.store.retrieve(sid_a)
        if not content_a:
            continue
        text_a = (content_a.summary or (content_a.full_text or "")[:120]).strip()
        if not text_a:
            continue

        for sid_b, _ in top_baseline:
            content_b = baseline.store.retrieve(sid_b)
            if not content_b:
                continue
            text_b = (content_b.summary or (content_b.full_text or "")[:120]).strip()
            if not text_b:
                continue

            file_a = (content_a.metadata or {}).get("source_file", sid_a)
            file_b = (content_b.metadata or {}).get("source_file", sid_b)
            entries.append(ContradictionEntry(
                text_a=text_a, source_a=file_a,
                text_b=text_b, source_b=file_b,
            ))
            found += 1
            if found >= max_pairs:
                return entries
            break  # one match per current source

    return entries
