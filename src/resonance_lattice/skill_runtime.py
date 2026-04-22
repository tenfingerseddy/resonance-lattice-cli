# SPDX-License-Identifier: BUSL-1.1
"""SkillRuntime: stateful bridge between skill frontmatter and RL internals.

Owns skill discovery, knowledge model resolution, encoder management, and caching.
Without this, every skill command would duplicate: parse frontmatter, resolve
paths, check encoder compatibility, load lattices, encode once.

See docs/SKILL_INTEGRATION.md for the architecture.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from resonance_lattice.encoder import Encoder, PhaseSpectrum
    from resonance_lattice.field.dense import DenseField
    from resonance_lattice.lattice import Lattice
    from resonance_lattice.skill import SkillConfig


logger = logging.getLogger(__name__)


@dataclass
class SkillMatch:
    """A skill ranked by relevance to a query."""
    name: str
    energy: float
    coverage: str   # high, medium, low
    mode: str


class SkillRuntime:
    """Owns skill discovery, knowledge model resolution, encoder management, and caching.

    Lifecycle:
        rt = SkillRuntime(skills_root, project_root)
        rt.discover()
        rt.resolve_cartridges(skill)
        phase = rt.encode_query("some question")
        matches = rt.route(phase, top_n=3)
    """

    def __init__(
        self,
        skills_root: Path,
        project_root: Path,
        source_root: Path | str | None = None,
    ) -> None:
        self._skills_root = skills_root
        self._project_root = project_root
        # source_root is passed through to Lattice.load so external-mode
        # cartridges can resolve content at skill-inject time.  Without it,
        # external carts silently fall back to "no content" retrieval (T2/T3
        # return source_ids but zero body text).
        self._source_root: Path | None = Path(source_root) if source_root else None
        self._skills: dict[str, SkillConfig] = {}
        self._lattices: dict[str, Lattice] = {}       # path_str -> loaded lattice
        self._fields: dict[str, DenseField] = {}       # path_str -> field-only
        self._encoder: Encoder | None = None
        self._encoder_config: dict | None = None         # for compat checks
        self._discovered = False

    # ── Discovery ────────────────────────────────────────────────────

    def discover(self) -> list[SkillConfig]:
        """Discover and cache all skills with valid frontmatter."""
        from resonance_lattice.skill import discover_skills

        skills = discover_skills(self._skills_root)
        self._skills = {s.name: s for s in skills}
        self._discovered = True
        return skills

    def get_skill(self, name: str) -> SkillConfig | None:
        if not self._discovered:
            self.discover()
        return self._skills.get(name)

    def cartridge_skills(self) -> list[SkillConfig]:
        """Return only skills that have knowledge model integration."""
        if not self._discovered:
            self.discover()
        return [s for s in self._skills.values()
                if s.has_cartridges or s.local_cartridge_path().exists()]

    # ── Cartridge Resolution ─────────────────────────────────────────

    def resolve_cartridges(self, skill: SkillConfig) -> list[Path]:
        """Resolve, validate, and return all knowledge model paths for a skill.

        Checks: paths exist and have at least a full header (64 B),
        encoder compatibility across knowledge models.
        Raises ValueError on incompatible encoders.

        Zero-byte or truncated knowledge model files are silently skipped (with a
        warning). This covers the common case of `skill build` never having
        been run — the skill_projector's T2/T3 tiers will then see an empty
        path list and degrade gracefully to instruction-only (T1) injection,
        instead of crashing deep in `RlatHeader.from_bytes`.
        """
        from resonance_lattice.serialise import RlatHeader

        paths = skill.resolve_cartridge_paths(project_root=self._project_root)

        # Add skill-local cartridge if it exists
        local = skill.local_cartridge_path()
        if local.exists() and local not in paths:
            paths.append(local)

        usable: list[Path] = []
        for p in paths:
            if not p.exists():
                continue
            try:
                size = p.stat().st_size
            except OSError:
                logger.warning("skipping unreadable knowledge model: %s", p)
                continue
            if size < RlatHeader.SIZE:
                logger.warning(
                    "skipping empty/truncated knowledge model: %s (%d B; run `rlat skill build %s`)",
                    p, size, skill.name,
                )
                continue
            usable.append(p)

        if not usable:
            return []

        # Check encoder compatibility
        self._check_encoder_compatibility(usable)
        return usable

    def _check_encoder_compatibility(self, paths: list[Path]) -> None:
        """Verify all knowledge models share the same encoder architecture.

        Checks backbone, bands, dim, query/passage prefixes, pooling,
        max_length, and sparsity config. Different checkpoints or projection
        heads with the same base config would produce incompatible vectors.
        """
        if len(paths) <= 1:
            return

        configs = []
        for p in paths:
            cfg = self._read_encoder_config(p)
            if cfg:
                configs.append((p, cfg))

        if len(configs) < 2:
            return

        def _compat_key(cfg: dict) -> tuple:
            return (
                cfg.get("backbone", ""),
                cfg.get("bands", 0),
                cfg.get("dim", 0),
                cfg.get("query_prefix", ""),
                cfg.get("passage_prefix", ""),
                cfg.get("pooling", ""),
                cfg.get("max_length", 0),
                # Include sparsity config — different sparsities produce
                # different vector distributions even with same backbone
                tuple(cfg.get("sparsity_targets", [])),
            )

        ref_path, ref_cfg = configs[0]
        ref_key = _compat_key(ref_cfg)

        for other_path, other_cfg in configs[1:]:
            other_key = _compat_key(other_cfg)
            if ref_key != other_key:
                # Build a human-readable diff of what's different
                diff_fields = []
                ref_dict = dict(zip(
                    ["backbone", "bands", "dim", "query_prefix", "passage_prefix",
                     "pooling", "max_length", "sparsity"], ref_key))
                other_dict = dict(zip(
                    ["backbone", "bands", "dim", "query_prefix", "passage_prefix",
                     "pooling", "max_length", "sparsity"], other_key))
                for k in ref_dict:
                    if ref_dict[k] != other_dict[k]:
                        diff_fields.append(f"    {k}: {ref_dict[k]} vs {other_dict[k]}")

                raise ValueError(
                    f"Encoder mismatch between cartridges:\n"
                    f"  {ref_path.name} vs {other_path.name}\n"
                    f"  Differences:\n" + "\n".join(diff_fields) + "\n"
                    f"All cartridges in a skill must use the same encoder.\n"
                    f"Run: rlat build <source-dir> -o {other_path} "
                    f"to rebuild with a matching encoder."
                )

    def _read_encoder_config(self, cartridge_path: Path) -> dict | None:
        """Read encoder config from a knowledge model without loading the full lattice."""
        try:
            lattice = self._load_lattice(cartridge_path)
            content = lattice.store.retrieve("__encoder__")
            if content and content.full_text:
                return json.loads(content.full_text)
        except Exception:
            pass
        return None

    # ── Loading ──────────────────────────────────────────────────────

    def _load_lattice(self, path: Path) -> Lattice:
        """Load a lattice with caching. Restores encoder from first load."""
        key = str(path.resolve())
        if key in self._lattices:
            return self._lattices[key]

        from resonance_lattice.lattice import Lattice

        lattice = Lattice.load(path, restore_encoder=True, source_root=self._source_root)
        self._lattices[key] = lattice

        # Capture encoder from first loaded lattice
        if self._encoder is None and lattice.encoder is not None:
            self._encoder = lattice.encoder
            logger.debug("Captured encoder from %s", path.name)

        return lattice

    def load_lattice(self, path: Path) -> Lattice:
        """Public: load a full lattice (field + store). Cached."""
        return self._load_lattice(path)

    def load_field(self, path: Path) -> DenseField:
        """Load field-only for fast routing. Cached."""
        key = str(path.resolve())
        if key in self._fields:
            return self._fields[key]

        lattice = self._load_lattice(path)
        self._fields[key] = lattice.field
        return lattice.field

    def get_encoder(self) -> Encoder:
        """Return the shared encoder. Raises if none loaded."""
        if self._encoder is not None:
            return self._encoder
        # Try to load from any cartridge skill
        for skill in self.cartridge_skills():
            paths = self.resolve_cartridges(skill)
            if paths:
                self._load_lattice(paths[0])
                if self._encoder is not None:
                    return self._encoder
        raise RuntimeError(
            "No encoder available. Load at least one knowledge model first.\n"
            "Run: rlat skill build <skill-dir> to build a knowledge model."
        )

    # ── Query Encoding ───────────────────────────────────────────────

    def encode_query(self, text: str) -> PhaseSpectrum:
        """Encode a query once, reusable across all tiers/knowledge models."""
        encoder = self.get_encoder()
        return encoder.encode_query(text)

    # ── Routing ──────────────────────────────────────────────────────

    def route(
        self,
        query_text: str,
        top_n: int = 3,
    ) -> list[SkillMatch]:
        """Rank all knowledge model-backed skills by resonance energy for a query.

        Loads field-only for each skill's knowledge models (fast, cached).
        """
        from resonance_lattice.discover import auto_route_query

        skills = self.cartridge_skills()
        if not skills:
            return []

        phase = self.encode_query(query_text)

        # Build name -> field mapping across all skill cartridges
        skill_fields: dict[str, DenseField] = {}
        skill_for_cartridge: dict[str, SkillConfig] = {}

        for skill in skills:
            paths = self.resolve_cartridges(skill)
            for p in paths:
                field = self.load_field(p)
                cart_key = f"{skill.name}:{p.stem}"
                skill_fields[cart_key] = field
                skill_for_cartridge[cart_key] = skill

        if not skill_fields:
            return []

        # Route across all cartridge fields
        ranked = auto_route_query(phase.vectors, skill_fields, top_n=len(skill_fields))

        # Aggregate by skill name (sum energy across cartridges)
        skill_energy: dict[str, float] = {}
        for cart_key, energy in ranked:
            skill = skill_for_cartridge[cart_key]
            skill_energy[skill.name] = skill_energy.get(skill.name, 0.0) + energy

        # Sort and classify
        sorted_skills = sorted(skill_energy.items(), key=lambda x: -x[1])

        matches = []
        for name, energy in sorted_skills[:top_n]:
            skill = self._skills[name]
            if energy > 200:
                coverage = "high"
            elif energy > 50:
                coverage = "medium"
            else:
                coverage = "low"
            matches.append(SkillMatch(
                name=name,
                energy=energy,
                coverage=coverage,
                mode=skill.cartridge_mode,
            ))

        return matches

    # ── Search helpers ───────────────────────────────────────────────

    def search_cartridges(
        self,
        paths: list[Path],
        query_text: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Search one or more knowledge models, returning merged results.

        Each result dict contains: source_id, score, content, knowledge model, band_scores.
        """
        if len(paths) == 1:
            lattice = self._load_lattice(paths[0])
            result = lattice.enriched_query(
                text=query_text, top_k=top_k,
                enable_cascade=False, enable_contradictions=False,
            )
            return [
                {
                    "source_id": r.source_id,
                    "score": r.score,
                    "content": r.content,
                    "knowledge model": paths[0].stem,
                    "band_scores": r.band_scores,
                }
                for r in result.results
            ]

        # Multiple cartridges: compose and search with text query
        from resonance_lattice.composition.composed import ComposedCartridge

        lattices = {}
        for p in paths:
            lattices[p.stem] = self._load_lattice(p)

        composed = ComposedCartridge.merge(lattices)
        # ComposedCartridge.search takes query_text (str), encodes internally
        results = composed.search(query_text, top_k=top_k)
        # Convert MaterialisedResult objects to dicts matching single-cartridge path
        return [
            {
                "source_id": r.source_id,
                "score": r.score,
                "content": r.content,
                "knowledge model": getattr(r, "knowledge model", "?"),
                "band_scores": r.band_scores,
            }
            for r in results
        ]
