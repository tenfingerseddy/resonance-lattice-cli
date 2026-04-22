# SPDX-License-Identifier: BUSL-1.1
"""Context Composition: knowledge models as composable building blocks.

This package enables fine-grained context control by composing multiple
.rlat knowledge models at query time using field algebra.

The key insight: compose field tensors algebraically (exact, cheap),
but dispatch registry lookups to each constituent independently.
This preserves source provenance and avoids registry/store merging.

Primary API:
    ComposedCartridge — virtual composition of N knowledge models
    compose_search   — convenience function for CLI/MCP
"""

from resonance_lattice.composition.composed import ComposedCartridge
from resonance_lattice.composition.context_file import ContextConfig, load_context
from resonance_lattice.composition.diagnostics import (
    CompositionDiagnostics,
    diagnose_composition,
    format_diagnostics,
)
from resonance_lattice.composition.parser import (
    ExprNode,
    collect_cartridge_paths,
    parse_expression,
)

__all__ = [
    "ComposedCartridge",
    "CompositionDiagnostics",
    "ContextConfig",
    "ExprNode",
    "collect_cartridge_paths",
    "diagnose_composition",
    "format_diagnostics",
    "load_context",
    "parse_expression",
]
