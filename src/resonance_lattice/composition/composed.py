# SPDX-License-Identifier: BUSL-1.1
"""ComposedCartridge: virtual composition of N knowledge models at query time.

The central runtime class for context composition. Composes field tensors
algebraically while keeping registries and stores separate.

Architecture:
    1. Compose field tensors via FieldAlgebra (exact, O(BD²))
    2. Resonate query through composed field (coverage/confidence)
    3. Dispatch registry lookups to each constituent independently
    4. Per-knowledge model score normalisation (prevent scale dominance)
    5. Re-rank combined pool using composed field resonance signal
    6. Content-hash dedup (same text across knowledge models)
    7. Materialise from each constituent's own store with provenance
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from resonance_lattice.algebra import FieldAlgebra
from resonance_lattice.field.dense import DenseField


@dataclass
class CompositionInfo:
    """Diagnostics about a composed knowledge model set."""
    constituent_names: list[str]
    total_sources: int
    composed_energy: float
    per_constituent_energy: dict[str, float]
    per_constituent_sources: dict[str, int]
    composition_type: str  # "merge", "project", "diff", etc.


class ComposedCartridge:
    """A virtual knowledge model formed by composing multiple loaded Lattices.

    Does NOT produce a new .rlat file. Instead:
    - Field algebra composes the semantic models
    - Registry lookups dispatch to each constituent independently
    - Results materialise from each constituent's own store
    - Provenance tracking records which knowledge model contributed each result

    Usage:
        from resonance_lattice.composition import ComposedCartridge
        from resonance_lattice.lattice import Lattice

        a = Lattice.load("docs.rlat")
        b = Lattice.load("code.rlat")

        composed = ComposedCartridge.merge({"docs": a, "code": b})
        results = composed.search("how does auth work?", top_k=5)
        # results[0].knowledge model == "docs" or "code"

        # Or with projection:
        composed = ComposedCartridge.project(
            source={"code": code_lattice},
            lens={"compliance": compliance_lattice},
        )
    """

    def __init__(
        self,
        constituents: dict[str, Lattice],  # noqa: F821 — avoid circular import
        composed_field: DenseField,
        composition_type: str = "merge",
        searchable: set[str] | None = None,
    ) -> None:

        self._constituents = constituents
        self._composed_field = composed_field
        self._composition_type = composition_type
        # Only these constituents are searched for results.
        # Lens/baseline cartridges contribute to field composition but
        # should NOT return their own documents in search results.
        self._searchable = searchable or set(constituents.keys())

        # Validate encoder compatibility
        fingerprints = {}
        for name, lattice in constituents.items():
            if lattice.encoder is not None:
                fp = self._encoder_fingerprint(lattice.encoder)
                fingerprints[name] = fp

        unique_fps = set(fingerprints.values())
        if len(unique_fps) > 1:
            raise ValueError(
                f"All cartridges must share the same encoder. "
                f"Found {len(unique_fps)} different encoders: "
                + ", ".join(f"{k}={v}" for k, v in fingerprints.items())
            )

        # Use the first lattice's encoder for query encoding
        first = next(iter(constituents.values()))
        self._encoder = first.encoder

    @staticmethod
    def _encoder_fingerprint(encoder) -> str:
        """Compute a stable fingerprint for encoder compatibility checks."""
        parts = [
            encoder.__class__.__name__,
            str(getattr(encoder, "backbone_name", "unknown")),
            str(getattr(encoder, "bands", "?")),
            str(getattr(encoder, "output_dim", "?")),
        ]
        return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]

    @property
    def composed_field(self) -> DenseField:
        return self._composed_field

    @property
    def encoder(self):
        return self._encoder

    @property
    def constituent_names(self) -> list[str]:
        return list(self._constituents.keys())

    @property
    def total_sources(self) -> int:
        return sum(l.source_count for l in self._constituents.values())

    def info(self) -> CompositionInfo:
        """Return diagnostics about this composition."""
        per_energy = {}
        per_sources = {}
        for name, lattice in self._constituents.items():
            per_energy[name] = float(sum(
                np.linalg.norm(lattice.field.F[b], "fro")
                for b in range(lattice.field.bands)
            ))
            per_sources[name] = lattice.source_count

        composed_energy = float(sum(
            np.linalg.norm(self._composed_field.F[b], "fro")
            for b in range(self._composed_field.bands)
        ))

        return CompositionInfo(
            constituent_names=self.constituent_names,
            total_sources=self.total_sources,
            composed_energy=composed_energy,
            per_constituent_energy=per_energy,
            per_constituent_sources=per_sources,
            composition_type=self._composition_type,
        )

    # ── Factory methods ──────────────────────────────────────

    @classmethod
    def merge(
        cls,
        constituents: dict[str, Lattice],  # noqa: F821
        weights: dict[str, float] | None = None,
    ) -> ComposedCartridge:
        """Create a composed knowledge model by merging fields.

        F_composed = Σ w_i * F_i

        Args:
            constituents: Named lattices to merge.
            weights: Per-constituent weights. Defaults to uniform.
        """
        names = list(constituents.keys())
        lattices = list(constituents.values())

        if weights is None:
            weights = {n: 1.0 for n in names}

        # Progressive merge
        composed = DenseField(
            bands=lattices[0].field.bands,
            dim=lattices[0].field.dim,
        )
        total_sources = 0
        for name, lattice in zip(names, lattices):
            w = weights.get(name, 1.0)
            composed.F += w * lattice.field.F
            total_sources += lattice.source_count
        composed._source_count = total_sources

        return cls(constituents, composed, composition_type="merge")

    @classmethod
    def project(
        cls,
        source: dict[str, Lattice],  # noqa: F821
        lens: dict[str, Lattice],  # noqa: F821
        k: int | None = None,
    ) -> ComposedCartridge:
        """Create a composed knowledge model by projecting source through lens.

        "Show me source's knowledge through lens's perspective."

        Args:
            source: Named lattices providing the knowledge.
            lens: Named lattice(s) defining the projection subspace.
                If multiple, their fields are merged first.
            k: Number of eigenvectors for projection. None = auto.
        """
        # Merge sources and lenses if multiple
        source_lattices = list(source.values())

        if len(source_lattices) == 1:
            source_field = source_lattices[0].field
        else:
            result = source_lattices[0].field
            for sl in source_lattices[1:]:
                merge_result = FieldAlgebra.merge(result, sl.field)
                result = merge_result.field
            source_field = result

        lens_lattices = list(lens.values())

        if len(lens_lattices) == 1:
            lens_field = lens_lattices[0].field
        else:
            result = lens_lattices[0].field
            for ll in lens_lattices[1:]:
                merge_result = FieldAlgebra.merge(result, ll.field)
                result = merge_result.field
            lens_field = result

        proj = FieldAlgebra.project(source_field, lens_field, k=k)

        all_constituents = {**source, **lens}
        # Only source cartridges are searchable — lens defines the subspace but
        # shouldn't contribute its own documents to results
        return cls(all_constituents, proj.projected_field,
                   composition_type="project", searchable=set(source.keys()))

    @classmethod
    def diff(
        cls,
        newer: dict[str, Lattice],  # noqa: F821
        older: dict[str, Lattice],  # noqa: F821
    ) -> ComposedCartridge:
        """Create a composed knowledge model from the difference between two sets.

        "What's new in the newer set vs the older set?"
        """
        newer_lattices = list(newer.values())
        older_lattices = list(older.values())

        # Merge each set if multiple
        if len(newer_lattices) == 1:
            newer_field = newer_lattices[0].field
        else:
            f = newer_lattices[0].field
            for nl in newer_lattices[1:]:
                f = FieldAlgebra.merge(f, nl.field).field
            newer_field = f

        if len(older_lattices) == 1:
            older_field = older_lattices[0].field
        else:
            f = older_lattices[0].field
            for ol in older_lattices[1:]:
                f = FieldAlgebra.merge(f, ol.field).field
            older_field = f

        diff_result = FieldAlgebra.diff(newer_field, older_field)

        all_constituents = {**newer, **older}
        # Only newer cartridges are searchable — older/baseline shouldn't
        # contribute documents to "what's new" results
        return cls(all_constituents, diff_result.delta_field,
                   composition_type="diff", searchable=set(newer.keys()))

    @classmethod
    def contradict(
        cls,
        set_a: dict[str, Lattice],  # noqa: F821
        set_b: dict[str, Lattice],  # noqa: F821
    ) -> ComposedCartridge:
        """Create a composed knowledge model from contradictions between two sets.

        "Where do A and B disagree?"
        """
        a_lattices = list(set_a.values())
        b_lattices = list(set_b.values())

        if len(a_lattices) == 1:
            a_field = a_lattices[0].field
        else:
            f = a_lattices[0].field
            for al in a_lattices[1:]:
                f = FieldAlgebra.merge(f, al.field).field
            a_field = f

        if len(b_lattices) == 1:
            b_field = b_lattices[0].field
        else:
            f = b_lattices[0].field
            for bl in b_lattices[1:]:
                f = FieldAlgebra.merge(f, bl.field).field
            b_field = f

        result = FieldAlgebra.contradict(a_field, b_field)

        all_constituents = {**set_a, **set_b}
        return cls(
            all_constituents,
            result.contradiction_field,
            composition_type="contradict",
        )

    # ── Topic sculpting ────────────────────────────────────────

    def sculpt_topics(
        self,
        boost_topics: list[str] | None = None,
        suppress_topics: list[str] | None = None,
        boost_strength: float = 0.5,
        suppress_strength: float = 0.3,
    ) -> ComposedCartridge:
        """Apply topic boost/suppress to the composed field (returns new instance).

        This modifies the composed field via rank-1 updates WITHOUT
        touching the original knowledge model files. Topics are encoded
        via the shared encoder, then sculpted into/out of the field.

        Args:
            boost_topics: List of topic strings to amplify.
            suppress_topics: List of topic strings to attenuate.
            boost_strength: Rank-1 boost magnitude (default 0.5).
            suppress_strength: Rank-1 suppress magnitude (default 0.3).

        Returns:
            New ComposedCartridge with sculpted field (original unchanged).
        """
        if self._encoder is None:
            raise ValueError("Topic sculpting requires an encoder.")

        sculpted = DenseField(
            bands=self._composed_field.bands,
            dim=self._composed_field.dim,
        )
        sculpted.F = self._composed_field.F.copy()
        sculpted._source_count = self._composed_field.source_count

        for topic in (boost_topics or []):
            phase = self._encoder.encode(topic)
            sculpted.superpose(phase.vectors, salience=boost_strength)

        for topic in (suppress_topics or []):
            phase = self._encoder.encode(topic)
            sculpted.remove(phase.vectors, salience=suppress_strength)

        return ComposedCartridge(
            self._constituents,
            sculpted,
            composition_type=self._composition_type,
            searchable=self._searchable,
        )

    # ── Per-cartridge injection modes ────────────────────────

    def set_injection_modes(
        self,
        modes: dict[str, str],
    ) -> None:
        """Set per-knowledge model injection modes for prompt formatting.

        When results are formatted for LLM consumption, each result's
        injection mode determines how the LLM should treat it:

        - "augment": Use as supplementary context — LLM can use its own
          knowledge AND cite this source.
        - "constrain": Answer ONLY from this source — LLM must not use
          its own knowledge for results from this knowledge model.
        - "knowledge": Base answer on this context, be transparent about gaps.

        Args:
            modes: Map of knowledge model name -> injection mode.
                   e.g. {"docs": "augment", "compliance": "constrain"}
        """
        self._injection_modes = modes

    def get_injection_mode(self, cartridge_name: str) -> str:
        """Get the injection mode for a specific knowledge model."""
        if not hasattr(self, "_injection_modes"):
            return "augment"  # default
        return self._injection_modes.get(cartridge_name, "augment")

    # ── Search ───────────────────────────────────────────────

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        retrieval_mode: str = "auto",
    ) -> list:
        """Search the composed knowledge model with full provenance tracking.

        Steps:
            1. Encode query via shared encoder
            2. Resonate through composed field (coverage/confidence)
            3. For each constituent: registry.lookup(query, top_k*2)
            4. Per-knowledge model score normalisation
            5. Re-rank using composed field resonance signal
            6. Content-hash dedup
            7. Materialise from each constituent's store with provenance

        Args:
            query_text: Natural language query.
            top_k: Number of results to return.
            retrieval_mode: "auto", "field", "registry", or "hybrid".

        Returns:
            list[MaterialisedResult] with knowledge model provenance.
            Each result also carries an `injection_mode` via its knowledge model name.
        """
        from resonance_lattice.lattice import MaterialisedResult

        if self._encoder is None:
            raise ValueError("No encoder available. Load knowledge models with encoders.")

        # 1. Encode query
        query_phase = self._encoder.encode_query(query_text)
        q_vectors = query_phase.vectors

        # 2. Resonate through composed field
        composed_resonance = self._composed_field.resonate(q_vectors)

        # 3. Dispatch registry lookups to searchable constituents only
        # (lens/baseline cartridges contributed to field composition but
        # should NOT return their own documents)
        over_retrieve = top_k * 2
        per_cartridge_results: list[tuple[str, list]] = []

        for name, lattice in self._constituents.items():
            if name not in self._searchable:
                continue
            retrieval = lattice.resonate(
                q_vectors,
                top_k=over_retrieve,
                retrieval_mode=retrieval_mode,
            )
            per_cartridge_results.append((name, retrieval.results))

        # 4. Per-cartridge score normalisation (prevent scale dominance)
        all_candidates: list[tuple[str, MaterialisedResult]] = []
        for name, results in per_cartridge_results:
            if not results:
                continue
            max_score = max(r.score for r in results) or 1.0
            for r in results:
                normalised = MaterialisedResult(
                    source_id=r.source_id,
                    score=r.score / max_score,  # normalise to [0, 1]
                    band_scores=r.band_scores,
                    content=r.content,
                    raw_score=r.score,
                    provenance=r.provenance,
                    cartridge=name,
                )
                all_candidates.append((name, normalised))

        # 5. Re-rank using composed field's resonance
        # Blend normalised score with composed resonance alignment
        for i, (name, result) in enumerate(all_candidates):
            if result.band_scores is not None:
                # Dot product of result's band scores with composed resonance
                composed_alignment = float(np.dot(
                    result.band_scores,
                    composed_resonance.band_energies[:len(result.band_scores)],
                ))
                # Blend: 70% normalised score + 30% composed alignment
                blended = 0.7 * result.score + 0.3 * composed_alignment
                all_candidates[i] = (name, MaterialisedResult(
                    source_id=result.source_id,
                    score=blended,
                    band_scores=result.band_scores,
                    content=result.content,
                    raw_score=result.raw_score,
                    provenance=result.provenance,
                    cartridge=name,
                ))

        # Sort by blended score
        all_candidates.sort(key=lambda x: x[1].score, reverse=True)

        # 6. Content-hash dedup (same text appearing in multiple cartridges)
        seen_hashes: set[str] = set()
        deduped: list[MaterialisedResult] = []

        for name, result in all_candidates:
            if result.content and result.content.full_text:
                # Hash first 200 chars of content for dedup
                content_hash = hashlib.md5(
                    result.content.full_text[:200].encode()
                ).hexdigest()[:8]
            else:
                content_hash = result.source_id

            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            deduped.append(result)

            if len(deduped) >= top_k:
                break

        return deduped

    def search_text(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> list:
        """Alias for search() — matches Lattice.resonate_text() convention."""
        return self.search(query_text, top_k=top_k)
