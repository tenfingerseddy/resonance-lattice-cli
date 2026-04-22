---
title: RQL Reference
slug: rql-reference
description: Complete reference for RQL operations, types, fluent DSL patterns, and extended field capabilities.
nav_group: Deep Dives
nav_order: 40
aliases:
---

# RQL — Resonance Query Language Reference

> Complete reference for all RQL operations, types, and usage patterns.

**Version**: 0.9.0 | **Operations**: ~271 across 10 domains + composition, lenses, temporal, access control

---

## What Is RQL?

RQL is the programmable query and transformation layer for Resonance Lattice fields. It exposes the semantic model directly instead of only through end-user commands.

## Why Should I Use It?

Use RQL when you need fine-grained control over field operations, composition, diagnostics, or experimental analysis that goes beyond the CLI.

## How Does It Work?

RQL organizes field operations into domains and exposes both direct function calls and a fluent `Field` DSL for composition.

## How Do I Use This Reference?

Start with the quick start and type system, then move through the operation domains. Use the extended capabilities sections when you need composition, temporal, access-control, or routing features.

## Quick Start

```python
from resonance_lattice.rql import (
    Field, spectral, algebra, info, compare, query,
    dynamics, stats, signal, geo, compose,
    SpectralCache, Scalar, Spectrum,
)
from resonance_lattice import DenseField

# Wrap a field
field = DenseField(bands=2, dim=64)
# ... superpose sources ...

# Use domain operations directly
entropy = info.von_neumann_entropy(field, band=0)
distance = compare.frobenius_distance(field_a, field_b)
denoised = signal.denoise_optimal(field, band=0)

# Fluent API
from resonance_lattice.rql import Field, boost, cascade, autotune
result = (
    Field(field)
    .pipe(boost(phase_vectors, 0.5))
    .pipe(cascade(3, 0.1))
    .pipe(autotune())
    .resonate(query_phase)
)

# Field arithmetic
merged = Field(field_a) + Field(field_b)   # merge
delta = Field(field_a) - Field(field_b)    # diff
scaled = 0.5 * Field(field_a)              # scale
```

---

## Type System

| Type | Description | Access |
|------|------------|--------|
| `Scalar` | Float with metadata | `.value`, `.name`, `.band` |
| `Spectrum` | Sorted eigenvalues | `.values`, `.top`, `.count` |
| `EigenDecomp` | Eigenvalues + eigenvectors | `.eigenvalues`, `.eigenvectors`, `.reconstruct()` |

### SpectralCache

Caches eigendecompositions to avoid redundant O(D³) computations:

```python
cache = SpectralCache(field)
eig = cache.get(band=0)      # Computed once
eig2 = cache.get(band=0)     # Cache hit (same object)
cache.invalidate()            # After field mutation
```

---

## Domain A: Spectral (`rql.spectral`)

Operations on eigenvalues and eigenvectors.

### Functional Calculus

| Operation | Formula | Cost |
|-----------|---------|------|
| `apply_fn(field, fn)` | F' = V diag(f(λ)) Vᵀ | O(BD³) |
| `power(field, p)` | F^p = V diag(λ^p) Vᵀ | O(BD³) |
| `exp(field)` | exp(F) = V diag(e^λ) Vᵀ | O(BD³) |
| `log(field)` | log(F) = V diag(ln λ) Vᵀ | O(BD³) |
| `sqrt(field)` | F^½ = V diag(√λ) Vᵀ | O(BD³) |
| `inv(field, ε)` | F⁻¹ = V diag(1/λ) Vᵀ | O(BD³) |
| `softmax(field, T)` | Redistribute energy via softmax | O(BD³) |

### Filtering

| Operation | Formula |
|-----------|---------|
| `threshold_hard(field, t)` | λ' = λ · (\|λ\| > t) |
| `threshold_soft(field, t)` | λ' = sign(λ) · max(\|λ\| - t, 0) |
| `bandpass(field, lo, hi)` | Keep λ ∈ [lo, hi] |
| `top_k(field, k)` | Keep top-k by magnitude |
| `bottom_k(field, k)` | Keep bottom-k |
| `power_law_decay(field, α)` | λ'_i = λ_i · (1 + rank)^{-α} |
| `exponential_decay(field, α)` | λ'_i = λ_i · e^{-α·rank} |

### Normalisation

| Operation | Formula |
|-----------|---------|
| `normalize_trace(field)` | F' = F / tr(F) |
| `normalize_spectral(field)` | F' = F / λ_max |
| `whiten(field)` | F^{-½} (decorrelate) |
| `standardize(field)` | λ' = (λ - μ) / σ |

### Metrics (return Scalar)

| Operation | Result |
|-----------|--------|
| `effective_rank(field, band)` | exp(H(\|λ\|/Σ\|λ\|)) |
| `entropy(field, band)` | Normalised spectral entropy [0,1] |
| `spectral_gap(field, band)` | λ₁ - λ₂ |
| `condition_number(field, band)` | λ_max / λ_min |
| `trace(field, band)` | Σλ |
| `participation_ratio(field, band)` | (Σλ)² / Σ(λ²) |
| `numerical_rank(field, band, ε)` | count(\|λ\| > ε) |
| `determinant(field, band)` | Πλ |

### Extraction

| Operation | Result |
|-----------|--------|
| `extract_top_k(field, k)` | V_k diag(λ_k) V_kᵀ |
| `extract_residual(field, k)` | F - extract_top_k(F, k) |
| `deflate(field)` | F - λ₁v₁v₁ᵀ |
| `split_pos_neg(field)` | (F_positive, F_negative) |
| `spectrum(field, band)` | Sorted eigenvalue array |

---

## Domain B: Algebra (`rql.algebra`)

| Operation | Formula | PSD-preserving |
|-----------|---------|---------------|
| `add(A, B)` | A + B | Yes |
| `subtract(A, B)` | A - B | No |
| `scale(F, α)` | αF | Yes (α ≥ 0) |
| `hadamard(A, B)` | A ⊙ B (element-wise) | No |
| `matmul(A, B)` | A @ B per band | No |
| `sandwich(F, A)` | A F Aᵀ (congruence) | Yes |
| `weighted_sum(fields, weights)` | Σ w_i F_i | Yes (w ≥ 0) |
| `lerp(A, B, t)` | (1-t)A + tB | Yes (t ∈ [0,1]) |
| `commutator(A, B)` | AB - BA | No (antisymmetric) |
| `anticommutator(A, B)` | AB + BA | Yes |
| `jordan_product(A, B)` | (AB + BA)/2 | Yes |
| `matrix_exp(F)` | V diag(e^λ) Vᵀ | Yes |
| `matrix_log(F)` | V diag(ln λ) Vᵀ | No |
| `matrix_sqrt(F)` | V diag(√λ) Vᵀ | Yes |
| `matrix_power(F, p)` | V diag(λ^p) Vᵀ | Yes (p > 0) |
| `project_psd(F)` | Clamp negative λ to 0 | Yes (output) |

Decompositions: `eigen(field, band)`, `svd(field, band)`, `low_rank_approx(field, k)`.

Norms: `frobenius_norm`, `spectral_norm`, `nuclear_norm`.

---

## Domain C: Information (`rql.info`)

| Operation | Formula | Range |
|-----------|---------|-------|
| `von_neumann_entropy(F, band)` | -tr(ρ log ρ) | [0, log D] |
| `renyi_entropy(F, α, band)` | (1/(1-α)) log(tr(ρ^α)) | [0, ∞) |
| `tsallis_entropy(F, q, band)` | (1 - tr(ρ^q)) / (q-1) | [0, ∞) |
| `kl_divergence(A, B, band)` | tr(ρ_a (log ρ_a - log ρ_b)) | [0, ∞) |
| `js_divergence(A, B, band)` | (KL(a\|\|m) + KL(b\|\|m))/2 | [0, log 2] |
| `mutual_information_bands(F, a, b)` | S(a) + S(b) - S(a,b) | [0, ∞) |
| `fidelity(A, B, band)` | (tr(√(√ρ_a ρ_b √ρ_a)))² | [0, 1] |
| `purity(F, band)` | tr(ρ²) | [1/D, 1] |
| `mixedness(F, band)` | 1 - purity | [0, 1-1/D] |
| `channel_capacity(F, band)` | λ_max | [0, ∞) |
| `fisher_information(F, q, band)` | \|\|F_b\|\|_F² | [0, ∞) |
| `information_content(F, band, α)` | log₂(det(I + αF)) | [0, ∞) |
| `redundancy(F, band)` | 1 - S/log(D) | [0, 1] |
| `cross_entropy(A, B, band)` | -tr(ρ_a log ρ_b) | [0, ∞) |
| `log_determinant(F, band)` | log(det(F)) | (-∞, ∞) |
| `band_correlation_matrix(F)` | C_ab = tr(F_a F_b)/(\|\|F_a\|\| \|\|F_b\|\|) | B×B matrix |

---

## Domain D: Geometric (`rql.geo`)

Operations on the Riemannian manifold of PSD matrices.

| Operation | Description |
|-----------|------------|
| `log_euclidean_distance(A, B, band)` | \|\|log(A) - log(B)\|\|_F |
| `affine_invariant_distance(A, B, band)` | \|\|log(A^{-½} B A^{-½})\|\|_F |
| `geodesic_interpolation(A, B, t)` | A^½ (A^{-½} B A^{-½})^t A^½ |
| `log_euclidean_mean(fields)` | exp(mean(log(F_i))) |
| `frechet_mean(fields, band)` | Iterative Riemannian mean |
| `stein_divergence(A, B, band)` | log det((A+B)/2) - ½ log det(AB) |
| `procrustes_align(A, B, band)` | Find rotation R: min \|\|A - RBRᵀ\|\| |
| `principal_angles(A, B, band, k)` | Angles between eigenspaces |
| `grassmann_distance(A, B, band, k)` | \|\|θ\|\|₂ |
| `subspace_overlap(A, B, band, k)` | Σcos²(θ)/k |
| `natural_gradient(F, ∇, band)` | F⁻¹ · ∇ |
| `exp_map(F, V, band)` | Riemannian exponential map |
| `log_map(A, B, band)` | Riemannian logarithmic map |
| `parallel_transport(F, V, G, band)` | Transport tangent V from F to G |
| `curvature_scalar(F, band)` | Effective scalar curvature |
| `tangent_vector(A, B)` | B - A (Euclidean approximation) |

---

## Domain E: Dynamics (`rql.dynamics`)

| Operation | Description |
|-----------|------------|
| `heat_diffusion(F, t)` | F(t) = V diag(e^{-λt}) Vᵀ |
| `exponential_decay(F, rate)` | F · e^{-rate} |
| `impulse(F, φ, strength)` | F + δ·(φ⊗φ) |
| `spectral_sharpening(F, factor)` | \|λ\|^{factor} (increase gaps) |
| `gradient_flow(F, ∇, dt)` | F - dt·∇ |
| `langevin_step(F, dt, T)` | F + √(2dt·T)·noise |
| `annealing_step(F, T, cooling)` | Spectral softmax at cooled T |
| `dissipative_step(F, F_eq, γ)` | (1-γ)F + γF_eq |
| `logistic_growth(F, r, K, dt)` | F + dt·r·F·(1 - F/K) |
| `spectral_erosion(F)` | Remove smallest eigenvalue |
| `mean_field_step(F, neighbors, dt)` | (1-dt)F + dt·mean(neighbors) |
| `lotka_volterra_bands(F, α, β, dt)` | Predator-prey between bands |
| `equilibrium_distance(F, F_eq)` | \|\|F - F_eq\|\|_F |
| `ode_step_rk4(F, rhs, dt)` | 4th-order Runge-Kutta |
| `conservation_check(F)` | Dict of trace and energy per band |

---

## Domain F: Statistics (`rql.stats`)

| Operation | Description |
|-----------|------------|
| `density_matrix(F, band)` | ρ = F / tr(F) |
| `covariance_bands(F)` | Cross-band covariance matrix |
| `correlation_bands(F)` | Normalised cross-band correlation |
| `ensemble_mean(fields)` | Element-wise mean |
| `ensemble_variance(fields)` | Element-wise variance |
| `energy_percentile(F, band, p)` | λ at p-th percentile of cumulative energy |
| `median_eigenvalue(F, band)` | Median of λ distribution |
| `skewness_spectrum(F, band)` | Third moment of λ distribution |
| `kurtosis_spectrum(F, band)` | Fourth moment (3.0 = Gaussian) |
| `moment(F, k, band)` | k-th spectral moment: mean(λ^k) |
| `mahalanobis_distance(F, q, band)` | √(qᵀF⁻¹q) |
| `marchenko_pastur_fit(F, band)` | Fit MP distribution (σ², γ, edges) |
| `outlier_eigenvalues(F, band)` | λ above MP bulk edge |

---

## Domain G: Signal (`rql.signal`)

| Operation | Description |
|-----------|------------|
| `lowpass(F, k)` | Keep top-k eigenvalues |
| `highpass(F, k)` | Remove top-k, keep rest |
| `compress(F, k)` | Rank-k + distortion metric |
| `denoise_hard(F, σ)` | Hard threshold at σ√(2 log D) |
| `denoise_soft(F, σ)` | Soft threshold |
| `denoise_optimal(F, band)` | Marchenko-Pastur threshold |
| `spectral_convolution(A, B)` | Pointwise eigenvalue product |
| `spectral_derivative(F, band)` | Finite difference on λ sequence |
| `noise_floor(F, band)` | Bottom quartile median |
| `moving_average_spectrum(F, window)` | Smooth eigenvalues |
| `energy_compaction(F, k, band)` | Top-k energy fraction |
| `snr_estimate(F, band)` | Signal vs bulk eigenvalues |

---

## Domain H: Comparison (`rql.compare`)

All binary operations return `Scalar`.

| Operation | Property |
|-----------|---------|
| `frobenius_distance(A, B)` | Metric. d(A,A)=0. Triangle inequality. |
| `spectral_distance(A, B, band)` | \|\|λ_a - λ_b\|\|₂ |
| `wasserstein_spectral(A, B, band)` | W₁ on sorted eigenvalues |
| `cosine_similarity(A, B)` | [-1, 1]. Self = 1. |
| `trace_distance(A, B, band)` | \|\|ρ_a - ρ_b\|\|₁ / 2 |
| `hellinger_distance(A, B, band)` | √(1 - fidelity) |
| `symmetrized_kl(A, B, band)` | (KL(a\|\|b) + KL(b\|\|a))/2. Symmetric. |
| `rank_correlation(A, B, band)` | Spearman on eigenvalue orderings |
| `intersection_over_union(A, B, band, k)` | IoU of top-k eigenvalue indices |
| `edit_distance(A, B, band)` | Nuclear norm of A-B |
| `relative_change(new, old)` | \|\|new-old\|\|/\|\|old\|\| |
| `energy_ratio(A, B)` | Σ\|\|A\|\| / Σ\|\|B\|\| |
| `max_eigenvalue_ratio(A, B, band)` | λ_max(A) / λ_max(B) |
| `drift_velocity(new, old, dt)` | \|\|new-old\|\|/dt |
| `band_similarity_matrix(A, B)` | B×B cosine similarity matrix |

---

## Domain I: Query (`rql.query`)

| Operation | Description |
|-----------|------------|
| `resonate(F, q, w)` | Standard: r_b = F_b @ q_b |
| `resonate_batch(F, Q)` | Batch N queries |
| `resonate_regularised(F, q, λ)` | (F + λI)⁻¹ F q |
| `optimal_query(F, band)` | q* = v₁ (dominant eigenvector) |
| `inverse_query(F, r, band, λ)` | Find q such that F@q ≈ r |
| `diversified_resonate(F, q, n, band)` | Resonate, deflate, repeat |
| `uncertainty_resonate(F, q, band)` | (r, σ) where σ = 1/√(qᵀFq) |
| `anti_resonate(F, q)` | What F does NOT know |
| `energy(F, q, band)` | qᵀ F_b q |
| `energy_all_bands(F, q)` | E_b per band |
| `gradient_at(F, q)` | ∇E = F@q |
| `steepest_ascent(F, q, α)` | q + α·norm(F@q) |
| `probe(F, q, threshold)` | Boolean: energy > threshold? |
| `similarity_under_field(F, q_a, q_b, band)` | q_aᵀ F_b q_b |

---

## Domain J: Compose (`rql.compose`)

| Operation | Description |
|-----------|------------|
| `map_bands(F, fn)` | Apply fn(F_b, b) independently per band |
| `reduce_bands(F, fn)` | Fold bands with binary fn |
| `zip_bands(A, B, fn)` | Pair bands, apply fn(A_b, B_b) |
| `parallel(F, chains, weights, q)` | Apply chains in parallel, weighted merge |
| `branch_merge(F, pred, chain_t, chain_f, q)` | Conditional chain selection |
| `conditional(F, metric, threshold, chain_above, chain_below, q)` | Metric-based branching |
| `fixed_point(F, chain, max_iter, tol, q)` | Iterate until convergence |
| `power_iterate(F, chain, iters, q)` | Return trajectory |
| `commutativity_test(F, chain_a, chain_b, q)` | \|\|AB(F) - BA(F)\|\|_F |

---

## Fluent API (`rql.dsl`)

The `Field` wrapper provides arithmetic and pipe composition:

```python
from resonance_lattice.rql import Field

# Arithmetic
merged = Field(a) + Field(b)     # merge
delta = Field(a) - Field(b)      # diff
scaled = 0.5 * Field(a)          # scale

# Pipe composition
result = (
    Field(my_field)
    .pipe(op1)
    .pipe(op2)
    .pipe(op3)
    .resonate(query_phase)
)

# Compile without resonating
compiled_field, context = Field(my_field).pipe(op1).compile(query_phase)
```

### Operator Factories

| Factory | Compiles To |
|---------|------------|
| `boost(phase, β)` | `Sculpt(phase, β)` |
| `suppress(phase, β)` | `Sculpt(phase, -β)` |
| `cascade(depth, α)` | `Cascade(depth, α)` |
| `metabolise(K, strategy)` | `Metabolise(K, strategy)` |
| `autotune(τ)` | `AutoTune(τ)` |
| `expand(steps, η)` | `ExpandQuery(steps, η)` |
| `spectral_filter(fn, label)` | `SpectralFilter(fn, label)` |
| `crossband(M)` | `CrossBandCouple(M)` |
| `lens.as_operator()` | `LensOp(lens)` |
| `eml_sharpen(strength)` | `EmlSharpen(strength)` |
| `eml_soften(strength)` | `EmlSoften(strength)` |
| `eml_contrast(background)` | `EmlContrast(background)` |
| `eml_tune(preset)` | `EmlTune(preset)` |

### EML Corpus Transforms

Nonlinear spectral transforms that reshape the field's eigenvalue spectrum. Unlike `boost`/`suppress` (rank-1, topic-specific), EML operates on the whole field without requiring topic names. All transforms are scale-invariant (normalise eigenvalues to `[0, 1]` before filtering, rescale by `λ_max` after).

```python
from resonance_lattice.rql import Field, eml_sharpen, eml_tune, eml_contrast

# Unsupervised sharpening for precise retrieval
result = (Field(corpus_field)
    .pipe(eml_sharpen(1.5))
    .resonate(query))

# Task-matched preset
result = Field(corpus_field).pipe(eml_tune("focus")).resonate(query)

# Asymmetric contrast: what's unique to A vs B?
result = Field(primary).pipe(eml_contrast(background)).resonate(query)
```

**Preset parameters** (`f(λ_norm) = exp(aλ + b) - ln(cλ + d)` on normalised eigenvalues):

| Preset | `(a, b, c, d)` | Use for |
|--------|----------------|---------|
| `focus` | `(2.0, 0, 0, 1)` | Factoid lookups, specific questions |
| `explore` | `(0, 0, 1, 1)` | Research queries, "what are the trade-offs of X?" |
| `denoise` | `(0.5, 0, 1, 0)` | Noisy corpora with lots of boilerplate |

---

# Context Composition and Extended Capabilities

The following modules extend the core RQL algebra with composition, temporal reasoning, access control, and adaptive routing.

---

## Context Composition

**Module:** `resonance_lattice.composition`

Compose multiple knowledge models at query time without modifying the underlying files. All composition modes produce queryable objects with full provenance tracking.

### ComposedCartridge

| Method | Purpose |
|--------|---------|
| `ComposedCartridge.merge(constituents, weights)` | Weighted union of knowledge |
| `ComposedCartridge.project(source, lens)` | View source through lens's semantic subspace |
| `ComposedCartridge.diff(newer, older)` | Queryable semantic difference |
| `ComposedCartridge.contradict(set_a, set_b)` | Surface disagreements between knowledge models |

```python
from resonance_lattice.composition import ComposedCartridge

# Merge with weights
composed = ComposedCartridge.merge(
    {"docs": lattice_docs, "code": lattice_code},
    weights={"docs": 0.7, "code": 0.3},
)
results = composed.search("how does auth work?", top_k=10)
# results[0].knowledge model == "docs" or "code"

# Project: view code through compliance's perspective
composed = ComposedCartridge.project(
    source={"code": code_lattice},
    lens={"compliance": compliance_lattice},
)

# Diff: what's new since baseline?
composed = ComposedCartridge.diff(
    newer={"current": current_lattice},
    older={"baseline": baseline_lattice},
)

# Contradict: where do docs and code disagree?
composed = ComposedCartridge.contradict(
    set_a={"docs": docs_lattice},
    set_b={"code": code_lattice},
)
```

### Topic Sculpting

Boost or suppress topics on any composed knowledge model without rebuilding.

```python
composed = composed.sculpt_topics(
    boost_topics=["security", "authentication"],
    suppress_topics=["marketing"],
    boost_strength=0.5,
    suppress_strength=0.3,
)
```

### Per-Knowledge Model Injection Modes

Control how each knowledge model's results are framed for an LLM.

```python
composed.set_injection_modes({
    "docs": "augment",        # LLM can blend its own knowledge
    "compliance": "constrain", # LLM must answer only from these sources
})
# Each result carries: composed.get_injection_mode(result.knowledge model)
```

### Expression Parser

Parse composition expressions into an evaluable AST.

```python
from resonance_lattice.composition import parse_expression, collect_cartridge_paths

ast = parse_expression("0.7 * docs.rlat + 0.3 * code.rlat")
paths = collect_cartridge_paths(ast)  # ["docs.rlat", "code.rlat"]
```

| Operator | Meaning | Example |
|----------|---------|---------|
| `+` | Merge | `docs + code` |
| `~` | Diff | `current ~ baseline` |
| `^` | Contradict | `docs ^ code` |
| `*` | Weight | `0.7 * docs` |
| `project(A, B)` | Projection | `project(code, compliance)` |
| `( )` | Grouping | `(docs + code) ~ baseline` |

### Context Files (.rctx)

YAML or JSON manifest declaring a reusable composition recipe.

```yaml
# team.rctx
name: team-context
knowledge models:
  docs: ./docs.rlat
  code: ./code.rlat
weights:
  docs: 0.7
  code: 0.3
suppress: ["meeting notes"]
injection_modes:
  docs: augment
  code: constrain
lens: sharpen
```

```bash
rlat compose team.rctx "how does auth work?"
```

---

## Algebra Extensions

### project(A, B, k)

Semantic projection of field A into B's eigenspace.

For each band *b*:
```
V_B = top-k eigenvectors of B_b
P   = V_B @ V_B^T                    (orthogonal projector)
project(A, B)_b = P @ A_b @ P        (congruence transform)
```

| Property | Value |
|----------|-------|
| Preserves PSD | Yes |
| Cost | O(B * D^3) |
| Idempotent | project(project(A, B), B) = project(A, B) |

```python
from resonance_lattice.algebra import FieldAlgebra

result = FieldAlgebra.project(code_field, compliance_field, k=32)
print(f"Retention: {result.retention_fraction:.1%}")
print(f"Subspace rank: {result.subspace_rank}")
```

Returns `ProjectionResult` with: `projected_field`, `retention_fraction`, `original_energy`, `projected_energy`, `per_band_retention`, `subspace_rank`.

### contradict(A, B)

Contradiction detection. Produces a queryable DenseField of disagreements.

```
For each band b:
    (A_b + B_b)/2 = V L V^T
    For each eigenvector v_k:
        c_k = |v_k^T A_b v_k - v_k^T B_b v_k| / max(...) * min(...)
    C_b = V diag(c) V^T
```

| Property | Value |
|----------|-------|
| Symmetric | contradict(A, B) = contradict(B, A) |
| Cost | O(B * D^3) |
| Self-contradiction | contradict(A, A) ~ 0 |

```python
result = FieldAlgebra.contradict(docs_field, code_field)
print(f"Contradiction ratio: {result.contradiction_ratio:.1%}")
# Search the contradiction field
resonance = result.contradiction_field.resonate(query_phase)
```

Returns `ContradictionResult` with: `contradiction_field`, `total_contradiction`, `total_agreement`, `contradiction_ratio`, `per_band_contradiction`.

---

## Knowledge Lenses

**Module:** `resonance_lattice.lens`

A lens transforms a field's semantic structure without altering its content. Lenses are composable, serializable, and usable in `.pipe()` chains.

### EigenspaceLens

Projects a field into a learned semantic subspace. Completeness guarantee: `lens.apply(F) + lens.invert(F) = F`.

```python
from resonance_lattice.lens import LensBuilder

# Build from exemplar phase vectors
lens = LensBuilder.from_exemplars("security", phase_vectors_list, k=16)

# Build from another field's eigenspace
lens = LensBuilder.from_field(compliance_field, name="compliance", k=32)

# Build from text (requires encoder)
lens = LensBuilder.from_text("security", ["authentication", "encryption", "vulnerability"], encoder, k=16)

# Apply
viewed = lens.apply(field)
hidden = lens.invert(field)  # complementary view
```

### SpectralLens

Reweights eigenvalues. Pre-built presets:

| Preset | Transform | Effect |
|--------|-----------|--------|
| `sharpen` | lambda^1.5 | Amplifies dominant themes |
| `flatten` | log(1 + lambda) | Equalises prominence |
| `denoise` | zero if < median | Removes low-energy noise |

```python
lens = LensBuilder.sharpen()
sharpened = lens.apply(field)
```

### CompoundLens

Compose multiple lenses in sequence.

```python
compound = lens_a.compose(lens_b)
result = compound.apply(field)
```

### Serialization

```python
lens.save("security")       # creates security.rlens + security.rlens.npz
loaded = Lens.load("security")
```

### Pipe Integration

```python
from resonance_lattice.rql import Field

result = (
    Field(my_field)
    .pipe(lens.as_operator())
    .pipe(autotune())
    .resonate(query_phase)
)
```

---

## Temporal Algebra

**Module:** `resonance_lattice.temporal_algebra`

Treats time-ordered field snapshots as a trajectory of knowledge states.

### temporal_derivative(newer, older, dt, mode)

Rate of semantic change. The derivative field is signed (not PSD): positive eigenvalues = growth, negative = loss.

```python
from resonance_lattice.temporal_algebra import temporal_derivative

deriv = temporal_derivative(this_week, last_week, dt=7.0)
print(f"Added: {deriv.added_energy:.2f}, Removed: {deriv.removed_energy:.2f}")
```

### knowledge_trend(snapshots, query)

Track whether a topic is growing, shrinking, or stable.

```python
from resonance_lattice.temporal_algebra import knowledge_trend

trend = knowledge_trend([v1, v2, v3, v4], query_phase)
print(f"{trend.trend_direction}: scores={trend.scores}")
```

### temporal_diff_chain(snapshots, labels)

Consecutive diffs, each independently searchable.

```python
from resonance_lattice.temporal_algebra import temporal_diff_chain

chain = temporal_diff_chain([v1, v2, v3], labels=["v1", "v2", "v3"])
for pair, diff_field in zip(chain.pairs, chain.diff_fields):
    print(f"{pair[0]} -> {pair[1]}: energy={np.linalg.norm(diff_field.F):.2f}")
```

### temporal_extrapolate(snapshots, dt)

Predict next knowledge state by linear extrapolation.

```python
from resonance_lattice.temporal_algebra import temporal_extrapolate

predicted = temporal_extrapolate([week1, week2, week3], dt=1.0)
```

---

## Cross-Knowledge Model Cascading

**Module:** `resonance_lattice.network`

Multi-hop search following an explicit route across knowledge model boundaries.

```python
from resonance_lattice.network import ResonanceNetwork

net = ResonanceNetwork()
net.add_node("code", code_field)
net.add_node("docs", docs_field)
net.add_node("incidents", incidents_field)

result = net.cascade_route(
    query_phase,
    route=["code", "docs", "incidents"],
    alpha=0.5,        # decay per hop
    threshold=0.01,   # minimum energy to continue
)

for hop in result.hops:
    print(f"{hop.node}: energy={hop.energy:.4f}")
```

Auto-connects nodes if no explicit channel exists. Energy gating stops the cascade early when signal attenuates below threshold.

---

## Algebraic Access Control

**Module:** `resonance_lattice.access`

Role-based knowledge views via orthogonal projection. Auditable: `visible + hidden = original`.

```python
from resonance_lattice.access import AccessPolicy

policy = AccessPolicy.from_exemplars("engineering", phase_vectors, k=10)

visible = policy.apply(field)         # what this role can see
hidden = policy.hidden_view(field)    # the complement
cert = policy.audit(field)            # audit certificate

print(f"Retention: {cert.retention_fraction:.1%}")
print(f"Verifiable: {cert.verifiable}")
print(f"Policy hash: {cert.policy_hash}")

# Compose policies
combined = policy_a.intersect(policy_b)  # only what both roles can see
```

---

## Adaptive Composition

**Module:** `resonance_lattice.adaptive`

Query-aware dynamic weighting across knowledge models.

```python
from resonance_lattice.adaptive import adaptive_weights

w = adaptive_weights(
    {"docs": docs_field, "code": code_field},
    query_phase,
    strategy="novelty",
)
print(w.global_weights)  # {"docs": 0.6, "code": 0.4}
```

| Strategy | Method | Best for |
|----------|--------|----------|
| `energy` | Weight by resonance strength | General-purpose |
| `novelty` | Weight by unique information (cosine distance from leave-one-out mean) | Diversity across sources |
| `band` | Per-band routing: each band selects the strongest knowledge model | Heterogeneous corpora |

---

## Updated Field DSL

### Arithmetic resolves pipe chains

`Field.__add__`, `__sub__`, `__mul__` now compile any pending `.pipe()` operators before executing:

```python
# Before fix: pipe was silently dropped
# After fix: spectral_filter is applied to field_a before merging
result = Field(field_a).pipe(spectral_filter(fn)) + Field(field_b)
```

### Text-based boost and suppress

```python
from resonance_lattice.rql import set_encoder, boost, suppress

set_encoder(encoder)  # register once

op = boost("security vulnerabilities", beta=0.5)   # encodes text automatically
op = suppress("marketing content", beta=0.3)
```

---

## Knowledge Model Discovery

**Module:** `resonance_lattice.discover`

Makes knowledge models discoverable to AI assistants without manual configuration.

### Manifest

The manifest at `.rlat/manifest.json` indexes all knowledge models in the project:

```python
from resonance_lattice.discover import Manifest

manifest = Manifest.load(".rlat/manifest.json")
for c in manifest.knowledge models:
    print(f"{c.name}: {c.sources} sources, domain={c.domain}")
```

Generated automatically by `init-project`, `build`, and `sync`.

### Auto-Integration

```bash
rlat init-project --auto-integrate
# Creates: .rlat/project.rlat, .rlat/manifest.json
# Updates: .mcp.json, CLAUDE.md
```

```python
from resonance_lattice.discover import inject_claude_md, update_mcp_json

update_mcp_json()       # wire MCP server to primary knowledge model
inject_claude_md()      # inject knowledge model section into CLAUDE.md
```

### Auto-Routing

```python
from resonance_lattice.discover import auto_route_query

routes = auto_route_query(query_phase, {"docs": docs_field, "code": code_field})
# [("docs", 4.2), ("code", 2.1)]  — docs is the better knowledge model for this query
```

### Freshness

```python
from resonance_lattice.discover import check_freshness

report = check_freshness(manifest.knowledge models[0], source_dir="./src")
print(f"{report.recommendation}")  # "fresh" or "stale -- rebuild recommended"
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `rlat_discover` | List all available knowledge models with domain, freshness, and primer paths |
| `rlat_freshness` | Check if a specific knowledge model needs rebuild |

---

## CLI Composition Flags

### search command

| Flag | Argument | Description |
|------|----------|-------------|
| `--with` | `PATH` | Merge with additional knowledge model (repeatable) |
| `--through` | `PATH` | Project through lens knowledge model |
| `--diff` | `PATH` | Semantic diff against baseline |
| `--boost` | `TEXT` | Boost topic (repeatable) |
| `--suppress` | `TEXT` | Suppress topic (repeatable) |
| `--sharpen` | `FLOAT` | EML sharpen the corpus (higher = more precise) |
| `--soften` | `FLOAT` | EML soften the corpus (higher = broader exploration) |
| `--contrast` | `PATH` | EML REML contrast against a background knowledge model |
| `--tune` | `focus\|explore\|denoise` | Task-matched EML retrieval preset |
| `--explain` | -- | Show composition diagnostics before searching |
| `--lens` | `NAME` | Apply lens: `sharpen`, `flatten`, `denoise`, or `.rlens` path |
| `--cascade-through` | `PATH...` | Multi-hop cascade through listed knowledge models |
| `--access` | `PATH` | Apply access policy from `.rlens` file |

### compose command

```bash
rlat compose "0.7 * docs.rlat + 0.3 * code.rlat" "query"
rlat compose "project(code, compliance)" "data handling"
rlat compose "(docs + code) ~ baseline" "what changed?"
rlat compose team.rctx "how does auth work?" --explain
```
