# SPDX-License-Identifier: BUSL-1.1
"""RQL — Resonance Query Language.

A comprehensive mathematical operations library for knowledge fields,
organised into 10 domains with 271 total operations.

Domains:
    spectral  — Eigenvalue/eigenvector operations (functional calculus, filtering, metrics)
    algebra   — Field arithmetic, decompositions, matrix functions, norms
    info      — Information theory (entropy, divergence, capacity, purity)
    compare   — Distances, similarities, alignment between fields
    query     — Resonance and query operations
    dynamics  — Time evolution, flows, diffusion, decay
    stats     — Density matrices, moments, ensembles, covariance
    signal    — Filtering, denoising, compression
    (geo, compose — Wave 2/3)

Infrastructure:
    Field     — Fluent wrapper with arithmetic and pipe composition
    SpectralCache — Avoids redundant eigendecompositions
    types     — Scalar, Spectrum, EigenDecomp typed results
"""

from resonance_lattice.rql.algebra_ops import AlgebraOps
from resonance_lattice.rql.cache import SpectralCache
from resonance_lattice.rql.compare import CompareOps
from resonance_lattice.rql.compose import ComposeOps
from resonance_lattice.rql.dsl import (
    Field,
    autotune,
    boost,
    cascade,
    crossband,
    expand,
    get_encoder,
    metabolise,
    set_encoder,
    spectral_filter,
    suppress,
)
from resonance_lattice.rql.dynamics_ops import DynamicsOps
from resonance_lattice.rql.eml import EmlOps
from resonance_lattice.rql.geometric import GeoOps
from resonance_lattice.rql.info import InfoOps
from resonance_lattice.rql.query_ops import QueryOps
from resonance_lattice.rql.result_manifold import (
    ResultAssessment,
    assess_results,
    concentration,
    coverage,
    diversity,
    eigenbasis_scores,
    eml_result_contrast,
    eml_result_filter,
    eml_result_sharpen,
    iterative_deepen,
    refine_query,
    result_field,
    signal_strength,
)
from resonance_lattice.rql.signal_ops import SignalOps
from resonance_lattice.rql.spectral import SpectralOps
from resonance_lattice.rql.stats import StatsOps
from resonance_lattice.rql.types import EigenDecomp, Scalar, Spectrum

# Domain aliases for clean namespace
spectral = SpectralOps
algebra = AlgebraOps
info = InfoOps
compare = CompareOps
query = QueryOps
dynamics = DynamicsOps
stats = StatsOps
signal = SignalOps
geo = GeoOps
compose = ComposeOps
eml_ops = EmlOps

__all__ = [
    # DSL
    "Field", "boost", "suppress", "cascade", "metabolise",
    "autotune", "expand", "spectral_filter", "crossband",
    "set_encoder", "get_encoder",
    # Types
    "Scalar", "Spectrum", "EigenDecomp", "SpectralCache",
    # Domains
    "SpectralOps", "AlgebraOps", "InfoOps", "CompareOps",
    "QueryOps", "DynamicsOps", "StatsOps", "SignalOps",
    "EmlOps",
    # Aliases
    "spectral", "algebra", "info", "compare",
    "query", "dynamics", "stats", "signal",
    "eml_ops",
    # Result Manifold
    "result_field", "ResultAssessment", "assess_results",
    "diversity", "coverage", "concentration", "signal_strength",
    "eigenbasis_scores",
    "eml_result_contrast", "eml_result_sharpen", "eml_result_filter",
    "refine_query", "iterative_deepen",
]
