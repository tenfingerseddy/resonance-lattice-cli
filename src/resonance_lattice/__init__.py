# SPDX-License-Identifier: BUSL-1.1
"""
Resonance Lattice - A next-generation AI context retrieval layer.

Information is not stored as objects to be found. It is encoded as a field
to be excited. The query doesn't find the answer. The query IS the resonance
that makes the answer emerge.
"""

__version__ = "0.11.0"

# Static re-exports for type checkers (pyright, mypy). Imports are guarded
# by TYPE_CHECKING so they don't eagerly pull in torch/scipy at runtime —
# the real lookup still happens through __getattr__ below.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from resonance_lattice.calculus import FieldCalculus
    from resonance_lattice.cascade import ResonanceCascade
    from resonance_lattice.compiler import (
        AutoTune,
        Cascade,
        Chain,
        CrossBandCouple,
        ExpandQuery,
        Metabolise,
        Operator,
        Sculpt,
        SpectralFilter,
    )
    from resonance_lattice.config import (
        EncoderConfig,
        LatticeConfig,
        MaterialiserConfig,
    )
    from resonance_lattice.encoder import Encoder, PhaseSpectrum
    from resonance_lattice.field.dense import DenseField
    from resonance_lattice.field.factored import FactoredField
    from resonance_lattice.field.pq import PQField
    from resonance_lattice.lattice import Config, Lattice, MaterialisedResult
    from resonance_lattice.materialiser import MaterialisedContext, Materialiser
    from resonance_lattice.pattern_injection import InterferenceSculptor
    from resonance_lattice.registry import PhaseRegistry
    from resonance_lattice.store import SourceContent, SourceStore
    from resonance_lattice.subspace import SubspaceField, metabolise


# Lazy imports — heavy modules (torch, scipy) are only loaded when accessed.
# This keeps `python -m resonance_lattice.cli` startup fast for the warm
# worker path, which only needs stdlib.

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Production SDK (WS1)
    "Config": ("resonance_lattice.lattice", "Config"),
    "Lattice": ("resonance_lattice.lattice", "Lattice"),
    "MaterialisedResult": ("resonance_lattice.lattice", "MaterialisedResult"),
    # Config
    "LatticeConfig": ("resonance_lattice.config", "LatticeConfig"),
    "EncoderConfig": ("resonance_lattice.config", "EncoderConfig"),
    "MaterialiserConfig": ("resonance_lattice.config", "MaterialiserConfig"),
    # Encoder (pulls in torch)
    "Encoder": ("resonance_lattice.encoder", "Encoder"),
    "PhaseSpectrum": ("resonance_lattice.encoder", "PhaseSpectrum"),
    # Fields
    "DenseField": ("resonance_lattice.field.dense", "DenseField"),
    "FactoredField": ("resonance_lattice.field.factored", "FactoredField"),
    "PQField": ("resonance_lattice.field.pq", "PQField"),
    # Materialiser
    "Materialiser": ("resonance_lattice.materialiser", "Materialiser"),
    "MaterialisedContext": ("resonance_lattice.materialiser", "MaterialisedContext"),
    # Registry & Store
    "PhaseRegistry": ("resonance_lattice.registry", "PhaseRegistry"),
    "SourceStore": ("resonance_lattice.store", "SourceStore"),
    "SourceContent": ("resonance_lattice.store", "SourceContent"),
    # Dynamic Field Algebra
    "FieldCalculus": ("resonance_lattice.calculus", "FieldCalculus"),
    "InterferenceSculptor": ("resonance_lattice.pattern_injection", "InterferenceSculptor"),
    "metabolise": ("resonance_lattice.subspace", "metabolise"),
    "SubspaceField": ("resonance_lattice.subspace", "SubspaceField"),
    "ResonanceCascade": ("resonance_lattice.cascade", "ResonanceCascade"),
    # Compiler
    "Chain": ("resonance_lattice.compiler", "Chain"),
    "Operator": ("resonance_lattice.compiler", "Operator"),
    "AutoTune": ("resonance_lattice.compiler", "AutoTune"),
    "ExpandQuery": ("resonance_lattice.compiler", "ExpandQuery"),
    "Sculpt": ("resonance_lattice.compiler", "Sculpt"),
    "SpectralFilter": ("resonance_lattice.compiler", "SpectralFilter"),
    "CrossBandCouple": ("resonance_lattice.compiler", "CrossBandCouple"),
    "Metabolise": ("resonance_lattice.compiler", "Metabolise"),
    "Cascade": ("resonance_lattice.compiler", "Cascade"),
}

# __all__ is the single source of truth for the public API.
# Mutations require a CHANGELOG entry (see CONTRIBUTING.md).
__all__ = [
    # Production SDK
    "Config",
    "Lattice",
    "MaterialisedResult",
    # Config
    "LatticeConfig",
    "EncoderConfig",
    "MaterialiserConfig",
    # Encoder
    "Encoder",
    "PhaseSpectrum",
    # Fields
    "DenseField",
    "FactoredField",
    "PQField",
    # Materialiser
    "Materialiser",
    "MaterialisedContext",
    # Registry & Store
    "PhaseRegistry",
    "SourceStore",
    "SourceContent",
    # Dynamic Field Algebra
    "FieldCalculus",
    "InterferenceSculptor",
    "metabolise",
    "SubspaceField",
    "ResonanceCascade",
    # Compiler (RQL Python API — see docs/RQL_REFERENCE.md)
    "Chain",
    "Operator",
    "AutoTune",
    "ExpandQuery",
    "Sculpt",
    "SpectralFilter",
    "CrossBandCouple",
    "Metabolise",
    "Cascade",
]


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        mod = importlib.import_module(module_path)
        val = getattr(mod, attr)
        # Cache on the module so __getattr__ is not called again
        globals()[name] = val
        return val
    raise AttributeError(f"module 'resonance_lattice' has no attribute {name!r}")
