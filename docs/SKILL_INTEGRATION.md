# Skill Integration: Resonance Lattice as the Semantic Intelligence Layer

> **Status**: Phase 1 shipped, Phase 2-4 in design
> **Version**: 0.2 (2026-04-12)
> **Depends on**: RL 0.9.0+, Claude Code skills

---

## Thesis

Skills are the atom of AI specialization. Every workflow improvement, every domain expertise, every tool integration eventually becomes a skill. They are the primary adoption surface for AI assistants.

But skills today are static documents. A 500-line SKILL.md loads the same content regardless of the user's question. Most of that context is irrelevant to any given query. Skills can't tell Claude what they know vs. don't. Two skills can't merge knowledge for cross-domain questions.

**Resonance Lattice becomes the semantic intelligence layer that makes every skill fundamentally better.** A skill declares which cartridges it draws from, what foundational queries it always needs, and RL handles adaptive context injection at trigger time. The user gets precisely relevant context instead of a static dump. The skill author adds a few frontmatter fields. RL becomes invisible infrastructure.

### Design Constraints

- **Zero-friction for skill authors** — as simple as adding frontmatter fields
- **Backwards-compatible** — skills without cartridge fields work unchanged
- **Composable** — field algebra enables cross-skill and cross-cartridge queries
- **Measurable** — coverage, freshness, quality become first-class
- **Graceful degradation** — if any tier fails, the others still work

---

## What's Shipped (Phase 1)

Phase 1 is implemented in `src/resonance_lattice/skill.py` and `cli.py`.

### Shipped CLI Commands

```bash
rlat skill build .claude/skills/rlat/     # Build cartridge from references/
rlat skill sync .claude/skills/rlat/      # Incremental update (skip unchanged)
rlat skill search rlat "query"            # Search a skill's cartridges
rlat skill info                           # List all skills with cartridge status
rlat skill info rlat                      # Detailed config for one skill
```

### Shipped Modules

| Module | What it provides |
|--------|-----------------|
| `skill.py` | `SkillConfig` dataclass, `parse_skill_frontmatter()`, `discover_skills()`, `find_skill()`, `extract_skill_header()` |
| `cli.py` | `cmd_skill_build`, `cmd_skill_sync`, `cmd_skill_search`, `cmd_skill_info` |

### Shipped Capabilities

- Frontmatter parsing for all `cartridge-*` fields
- Skill discovery across `.claude/skills/`
- Build cartridge from `references/` or explicit `cartridge-sources`
- Incremental sync (skip unchanged files)
- Search single or multi-cartridge skills (auto-compose via `ComposedCartridge`)
- Primer generation alongside cartridge build
- Info display with cartridge size, status, config

### Known Limitations of Phase 1

- **YAML parser is minimal** — no nested objects, anchors, or full YAML spec. PyYAML should be adopted before the schema grows further. See "Open: YAML Parser" below.
- **No skill routing** — `skill search` requires explicit skill name, no auto-routing by query.
- **No adaptive injection** — search returns standard enriched results, not the four-tier injection pipeline.
- **No MCP integration** — skill commands are CLI-only.

---

## The Four-Tier Injection Model

When a cartridge-backed skill triggers, context loads in four tiers:

```
┌──────────────────────────────────────────────────────────────────┐
│ Tier 1: STATIC                                                   │
│ Source: SKILL.md header                                          │
│ Always loaded. No rlat involved.                                 │
│ Templates, scripts, workflow steps, decision trees.              │
│ "Here's HOW to use this skill"                                   │
│ Budget: ~500 tokens                                              │
├──────────────────────────────────────────────────────────────────┤
│ Tier 2: FOUNDATIONAL                                             │
│ Source: Skill-authored queries -> cartridges                     │
│ Same every trigger. Skill's baseline knowledge.                  │
│ Pre-built queries the skill ALWAYS needs answered.               │
│ "Here's WHAT you need to know every time"                        │
│ Budget: ~800 tokens (40% of dynamic budget)                      │
├──────────────────────────────────────────────────────────────────┤
│ Tier 3: SPECIFIC                                                 │
│ Source: User query -> cartridges                                 │
│ Different every trigger. This request's unique context.          │
│ The user's actual question resonated against the cartridges.     │
│ "Here's what's UNIQUE about this request"                        │
│ Budget: ~600 tokens (30% of dynamic budget)                      │
├──────────────────────────────────────────────────────────────────┤
│ Tier 4: DERIVED                                                  │
│ Source: Caller-supplied queries -> cartridges                     │
│ The skill's standing instructions tell Claude to generate 2-3    │
│ targeted search queries and pass them via --derived. Claude IS   │
│ the LLM -- no separate API call needed.                          │
│ "Here's what you DIDN'T KNOW TO ASK FOR"                         │
│ Budget: ~600 tokens (30% of dynamic budget)                      │
└──────────────────────────────────────────────────────────────────┘
```

### Budget Model

Budgets are in **tokens**, not lines. This matches the existing RL convention
(`MaterialiserConfig.token_budget`, `projector.py` char/token limits).
The `cartridge-budget` frontmatter field is in tokens (default 2000).
Internally, `_estimate_tokens()` and `_truncate_to_tokens()` from the
materialiser handle enforcement.

| Tier | Allocation | Default tokens |
|------|-----------|----------------|
| Tier 1 (static header) | Fixed | ~500 (SKILL.md body, outside budget) |
| Tier 2 (foundational) | 40% of `cartridge-budget` | ~800 |
| Tier 3 (user query) | 30% of `cartridge-budget` | ~600 |
| Tier 4 (derived) | 30% of `cartridge-budget` | ~600 |

If Tier 4 is disabled, its allocation shifts to Tier 3 (40/60 split).

### Why Four Tiers

| Configuration | What it catches | What it misses |
|---------------|-----------------|----------------|
| Tier 1 alone | Workflow structure | All domain knowledge |
| Tier 2 alone | Foundational domain knowledge | Nothing specific to this request |
| Tier 3 alone | User's explicit intent | Foundational knowledge + implicit needs |
| Tier 4 alone | Implicit needs the user didn't express | Foundation + explicit intent |
| **Tiers 1+2+3+4** | **Everything: structure + foundation + explicit + implicit** | -- |

---

## Concrete Example

User types: **"create a notebook for ingesting these tables from this REST API"**

The `fabric-notebook-ingest` skill triggers:

```
Tier 1 (static):
  SKILL.md header loads -- templates, notebook structure, workflow steps
  "Here's HOW you build a notebook response"

Tier 2 (foundational, skill-authored):
  Q: "How do you create a notebook in Fabric through the API"
    -> fabric-docs.rlat -> Fabric REST API, workspace setup, notebook lifecycle
  Q: "What are pyspark best practices for data ingestion"
    -> pyspark-docs.rlat -> DataFrame patterns, schema handling, write modes
  Q: "Fabric workspace authentication and authorization patterns"
    -> fabric-docs.rlat -> SPN auth, token refresh, workspace identity

Tier 3 (user query):
  Q: "create a notebook for ingesting these tables from this REST API"
    -> fabric-docs.rlat -> REST connector docs, notebook creation endpoints
    -> pyspark-docs.rlat -> requests library integration, JSON parsing in Spark

Tier 4 (derived, via --derived):
  Claude generates (from standing instructions):
  Q: "pyspark REST API pagination retry error handling"
    -> pyspark-docs.rlat -> pagination patterns, retry decorators, error recovery
  Q: "Delta Lake merge upsert incremental ingestion"
    -> pyspark-docs.rlat -> MERGE INTO syntax, watermark tracking, idempotency

All results merged, deduped, budget-capped -> ~2000 tokens of precisely targeted context
```

---

## Tier 4: Derived Queries (Detail)

Tier 4 accepts caller-supplied search queries via `--derived`. The key insight:
**Claude IS the LLM.** It's already running, already has the user's request in
context, already understands the skill's domain. There's no need for a separate
API call inside the projector.

### How It Works

The skill's Tier 1 standing instructions tell Claude to generate derived queries:

1. Claude reads the user's request
2. Claude generates 2-3 short, specific search queries targeting implicit needs
3. Claude passes them via `rlat skill inject <name> "query" --derived "q1;q2;q3"`
4. The projector searches each derived query against the cartridges
5. Results are deduped against Tiers 2/3 and budget-capped

This means:
- **Zero new dependencies** -- no API key, no anthropic SDK in the retrieval path
- **Zero extra latency** -- Claude generates queries as part of normal reasoning
- **Skill author controls the prompt** -- standing instructions shape what Claude asks for
- **SkillProjector stays pure retrieval** -- no LLM coupling, testable, deterministic

### Design Constraints

| Constraint | Value | Rationale |
|------------|-------|-----------|
| Max queries | 2-3 (configurable via `cartridge-derive-count`) | More queries = diminishing returns |
| Query length | Under 15 words | Short, specific queries produce better resonance |
| Dedup | By source_id against Tiers 2/3 | Prevents redundant passages |
| Fallback | If no derived queries supplied, Tier 4 is empty | Budget shifts to Tier 3 |

---

## Mode-Aware Gating

The `GatedProjector` from `projector.py` decides whether to inject dynamic context.
But it was designed for the "7B problem" -- suppressing broad context when the model
already knows the answer. This logic must be **mode-aware** for skill injection:

| Mode | Gating behaviour | Rationale |
|------|-----------------|-----------|
| `augment` | **Full gate**: suppress if energy low or novelty low | Standard -- don't inject what the model already knows |
| `knowledge` | **Soft gate**: suppress only if energy very low (threshold 0.15 vs default 0.3) | The skill's context is the primary knowledge source; suppress only when the cartridge truly has nothing |
| `constrain` | **No gate**: always inject | Zero-hallucination mode. The whole point is to constrain Claude to these sources. Suppressing defeats the purpose. Use `GroundingProjector` path instead of `GatedProjector`. |

Implementation note: `projector.py` already has both `GatedProjector` (broad context)
and `GroundingProjector` (citation-only). The `SkillProjector` should select between
them based on `cartridge-mode`:

```python
if skill.cartridge_mode == "constrain":
    # Always inject -- use GroundingProjector, no gating
    projector = GroundingProjector(top_k=budget_sources)
elif skill.cartridge_mode == "knowledge":
    # Soft gate
    projector = GatedProjector(base, passage_phases, gate_threshold=0.15)
else:  # augment
    # Full gate
    projector = GatedProjector(base, passage_phases, gate_threshold=0.3)
```

---

## The SkillRuntime

Codex review correctly identified that "thin orchestration" understates the runtime
glue. `auto_route_query()` takes an encoded query phase and dense fields, not skill
names. Composition requires encoder compatibility across all cartridges.

Phase 2 introduces `SkillRuntime` -- a stateful object that owns the runtime lifecycle:

```python
class SkillRuntime:
    """Owns skill discovery, cartridge resolution, encoder management, and caching.

    This is the single object that bridges skill frontmatter to RL internals.
    Without it, every CLI command would duplicate: parse frontmatter, resolve paths,
    check encoder compatibility, load lattices, encode query, route, compose.
    """

    def __init__(self, skills_root: Path, project_root: Path):
        self._skills_root = skills_root
        self._project_root = project_root
        self._skills: dict[str, SkillConfig] = {}
        self._lattices: dict[str, Lattice] = {}   # path -> loaded lattice
        self._fields: dict[str, DenseField] = {}   # path -> field-only (for routing)
        self._encoder: Encoder | None = None        # shared encoder

    def discover(self) -> list[SkillConfig]:
        """Discover and cache all cartridge-backed skills."""

    def resolve_cartridges(self, skill: SkillConfig) -> list[Path]:
        """Resolve cartridge paths, validate existence, check encoder compatibility."""

    def load_field(self, cartridge_path: Path) -> DenseField:
        """Load field-only (no store) for fast routing. Cached."""

    def load_lattice(self, cartridge_path: Path) -> Lattice:
        """Load full lattice (field + store). Cached with LRU eviction."""

    def encode_query(self, text: str) -> PhaseSpectrum:
        """Encode once, reuse across all tiers/cartridges."""

    def route(self, query_phase: PhaseSpectrum, top_n: int = 3) -> list[SkillMatch]:
        """Rank all skills by resonance energy. Uses cached fields."""

    def check_encoder_compatibility(self, paths: list[Path]) -> None:
        """Raise if cartridges have incompatible encoders (different backbone/dim/bands)."""
```

### Encoder Compatibility

All cartridges in a single skill invocation **must** share the same encoder architecture
(backbone, bands, dim). This is already enforced by `ComposedCartridge` at the field algebra
level (mismatched dimensions raise `ValueError`). The skill layer adds an earlier check:

- At `rlat skill build` time: encoder fingerprint is stored in the cartridge manifest.
- At `SkillRuntime.resolve_cartridges()` time: fingerprints are compared before loading.
- If mismatched: hard error with a clear message naming the incompatible cartridges.

### Why Not In MCP?

Skill routing runs **outside** the MCP server, in the Claude-side integration layer.
The MCP server stays cartridge-centric: it loads one primary cartridge, maintains a small
composition cache, and handles individual cartridge operations.

Rationale:
- The MCP server is a stdio process tied to one cartridge path at startup.
- Skill routing requires scanning `.claude/skills/`, parsing frontmatter, and loading
  multiple fields -- concerns that belong to the orchestration layer, not the MCP transport.
- Claude already has skill discovery and selection logic. RL's contribution is the
  *scoring signal* (resonance energy), not replacing the routing infrastructure.

Future: A `rlat skill route` CLI command provides the energy scores. Claude's skill
system can call this to augment its description-based matching. The MCP server doesn't
need to change.

---

## Skill Format Extension

### Frontmatter Schema (Backwards-Compatible)

```yaml
---
name: fabric-notebook-ingest
description: Create Fabric notebooks for data ingestion...

# Cartridge integration (all optional, all new)
cartridges:                                    # which rlats this skill draws from
  - fabric-docs.rlat                           # relative to project root or absolute
  - pyspark-docs.rlat
cartridge-queries:                             # foundational queries -- Tier 2
  - "How do you create a notebook in Fabric through the API"
  - "What are pyspark best practices for data ingestion"
  - "Fabric workspace authentication and authorization patterns"
cartridge-sources:                             # dirs to build a skill-local cartridge from
  - references/
cartridge-mode: augment                        # injection mode: augment | constrain | knowledge
cartridge-budget: 2000                         # max dynamic injection tokens (Tier 2+3+4)
cartridge-rebuild: on-change                   # rebuild policy: none | on-change | daily
cartridge-derive: true                         # enable Tier 4 derived queries
cartridge-derive-count: 3                      # max derived queries
---
```

### Field Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cartridges` | list[str] | `[]` | Paths to .rlat files this skill queries |
| `cartridge-queries` | list[str] | `[]` | Foundational queries (Tier 2), run every trigger |
| `cartridge-sources` | list[str] | `["references/"]` | Dirs to build skill-local cartridge from |
| `cartridge-mode` | str | `"augment"` | Injection mode: augment, constrain, knowledge |
| `cartridge-budget` | int | `2000` | Max tokens for dynamic injection (Tiers 2+3+4) |
| `cartridge-rebuild` | str | `"none"` | Rebuild policy: none, on-change, daily |
| `cartridge-derive` | bool | `true` | Enable Tier 4 derived queries |
| `cartridge-derive-count` | int | `3` | Max number of derived queries |

### Injection Modes

| Mode | Gating | System prompt framing | Use case |
|------|--------|----------------------|----------|
| `augment` | Full gate (threshold 0.3) | "Use your own knowledge, but add detail and citations from these sources" | Most skills -- supplement Claude's knowledge |
| `knowledge` | Soft gate (threshold 0.15) | "Base your answer primarily on this context" | Domain-specific skills where training data may be outdated |
| `constrain` | No gate (always inject) | "Answer ONLY from these sources. If the answer isn't here, say so" | Compliance, regulatory, legal -- no improvisation |

### Backwards Compatibility

Skills without any `cartridge-*` fields work exactly as they do today. A skill can adopt incrementally:

1. Add `cartridges:` only -> enables Tier 3 (user query search)
2. Add `cartridge-queries:` -> enables Tier 2 (foundational context)
3. Leave `cartridge-derive: true` (default) -> enables Tier 4 (derived queries)

---

## Cartridge Patterns

### Pattern A: External Cartridges (Most Common)

```yaml
cartridges:
  - .rlat/fabric-docs.rlat
  - .rlat/pyspark-docs.rlat
```

No build step needed. Multiple skills can reference the same cartridge.

### Pattern B: Skill-Local Cartridge

```yaml
cartridge-sources:
  - references/
```

`rlat skill build` indexes `references/` into `cartridge/<name>.rlat`.

### Pattern C: Both

```yaml
cartridges:
  - .rlat/fabric-docs.rlat
  - cartridge/my-skill.rlat
cartridge-sources:
  - references/
```

### Directory Convention

```
.claude/skills/fabric-notebook-ingest/
  SKILL.md                      # frontmatter + standing instructions (Tier 1)
  references/                   # bundled docs (existing Claude Code convention)
  cartridge/                    # skill-local cartridge artifacts (Pattern B/C only)
    my-skill.rlat               # built from references/
    primer.md                   # compressed session primer
  scripts/                      # existing convention
```

---

## The Injection Pipeline

```
User query arrives, skill triggers
    |
    v
[1. Load Tier 1: SKILL.md header]       <- always (standing instructions, templates)
    |
    v
[2. Resolve cartridges via SkillRuntime] <- validate paths, check encoder compat
    |
    v
[3. Encode query once]                  <- shared encoder, reused for all tiers
    |
    v
[4. Run Tier 2: foundational queries]   <- cartridge-queries resonate against cartridges
    |                                      each query routes to best-matching cartridge
    |                                      via auto_route_query() on cached fields
    v
[5. Run Tier 3: user query]             <- user's request resonates against all cartridges
    |
    v
[6. Run Tier 4: derived queries]         <- caller-supplied via --derived
    |                                      dedup against Tier 2/3 by source_id
    v
[7. Merge + dedup all tiers]            <- same passage from multiple queries = count once
    |
    v
[8. Budget cap (tokens)]               <- enforce cartridge-budget via _truncate_to_tokens
    |                                      Tier 2: 40%, Tier 3: 30%, Tier 4: 30%
    v
[9. Mode-aware gate]                   <- augment: full gate | knowledge: soft | constrain: skip
    |
    v
[10. Format with injection mode]        <- augment/constrain/knowledge framing
    |
    v
[11. Return: Tier 1 + dynamic body]     <- complete context for Claude
```

### Query Routing Within a Multi-Cartridge Skill

Each query (Tier 2, 3, or 4) resonates against ALL declared cartridges via
`auto_route_query()` on cached `DenseField` objects. Results come from whichever
cartridge has the strongest resonance energy:

```
Q: "How do you create a notebook in Fabric through the API"
  -> fabric-docs.rlat: energy=342  <- winner
  -> pyspark-docs.rlat: energy=28

Q: "What are pyspark best practices"
  -> fabric-docs.rlat: energy=41
  -> pyspark-docs.rlat: energy=287  <- winner

Q (user): "create a notebook for ingesting tables from REST API"
  -> fabric-docs.rlat: energy=195  <- relevant
  -> pyspark-docs.rlat: energy=163  <- also relevant
  -> Pull from BOTH, merge

Q (derived): "pyspark REST API pagination retry patterns"
  -> fabric-docs.rlat: energy=12
  -> pyspark-docs.rlat: energy=298  <- winner
```

---

## CLI Commands

### Shipped (Phase 1)

| Command | Status |
|---------|--------|
| `rlat skill build <skill-dir>` | **Shipped** |
| `rlat skill sync <skill-dir>` | **Shipped** |
| `rlat skill search <name> "query"` | **Shipped** |
| `rlat skill info [name]` | **Shipped** |

### Planned (Phase 2-4)

| Command | Phase | Delegates to |
|---------|-------|-------------|
| `rlat skill inject <name> "query"` | 2 | `SkillProjector` (four-tier pipeline) |
| `rlat skill route "query"` | 3 | `SkillRuntime.route()` |
| `rlat skill compose "a + b" "query"` | 3 | `ComposedCartridge.merge()` |
| `rlat skill catalog [dir]` | 3 | `SkillRuntime.discover()` |
| `rlat skill profile <name>` | 4 | `rlat profile` + `rlat xray` |
| `rlat skill gaps <name>` | 4 | `rlat probe gaps` |
| `rlat skill compare <a> <b>` | 4 | `rlat compare` |
| `rlat skill freshness [name]` | 4 | `check_freshness()` |

---

## New Modules (Phase 2)

### `src/resonance_lattice/skill_runtime.py`

```python
class SkillRuntime:
    """Stateful object owning skill discovery, resolution, caching, encoding."""

    def __init__(self, skills_root: Path, project_root: Path): ...
    def discover(self) -> list[SkillConfig]: ...
    def resolve_cartridges(self, skill: SkillConfig) -> list[Path]: ...
    def load_field(self, path: Path) -> DenseField: ...
    def load_lattice(self, path: Path) -> Lattice: ...
    def encode_query(self, text: str) -> PhaseSpectrum: ...
    def route(self, query_phase: PhaseSpectrum, top_n: int = 3) -> list[SkillMatch]: ...
    def check_encoder_compatibility(self, paths: list[Path]) -> None: ...
```

### `src/resonance_lattice/skill_projector.py`

```python
@dataclass
class SkillInjection:
    header: str                    # Tier 1
    body: str                      # Tiers 2+3+4
    coverage: dict                 # per-cartridge coverage signals
    gate: GateDecision | None      # None if constrain mode (no gate)
    mode: str                      # augment | constrain | knowledge
    queries_used: list[str]
    cartridge_hits: dict[str, int]
    tier_breakdown: dict[str, int] # tokens per tier
    total_tokens: int

class SkillProjector:
    def __init__(self, runtime: SkillRuntime): ...
    def project(self, skill: SkillConfig, user_query: str) -> SkillInjection: ...
    def _derive_queries(self, user_query: str, skill: SkillConfig, existing: list[str]) -> list[str]: ...
```

Reuses: `GatedProjector` and `GroundingProjector` (projector.py), `auto_route_query()` (discover.py), `adaptive_weights()` (adaptive.py), `ComposedCartridge` (composition/composed.py), `_estimate_tokens` / `_truncate_to_tokens` (materialiser.py).

---

## Implementation Phases

### Phase 1: Foundation -- SHIPPED

**Modules**: `skill.py`, `cli.py` skill subcommands
**Commands**: `build`, `sync`, `search`, `info`
**Verified**: Build from references, incremental sync, enriched search, info display

### Phase 2: Four-Tier Injection + SkillRuntime

**Scope**: The full injection pipeline with mode-aware gating.

**Create**:
- `src/resonance_lattice/skill_runtime.py` -- `SkillRuntime`
- `src/resonance_lattice/skill_projector.py` -- `SkillProjector`, `SkillInjection`

**Modify**:
- `src/resonance_lattice/cli.py` -- add `skill inject` subcommand
- `src/resonance_lattice/skill.py` -- switch to PyYAML if available (see open item)

**Verify**:
```bash
rlat skill inject fabric-notebook-ingest "create a notebook for ingesting from REST API"
# Should return: Tier 1 header + Tier 2 foundational + Tier 3 specific + Tier 4 derived
# With: token counts per tier, coverage signals, gate decision
```

### Phase 3: Routing + Composition

**Scope**: Semantic routing across all skills. Skill manifest.

**Modify**:
- `src/resonance_lattice/skill_runtime.py` -- `route()`, `discover()` with caching
- `src/resonance_lattice/cli.py` -- add `skill route`, `skill catalog`, `skill compose`

**Verify**:
```bash
rlat skill route "how do I train an encoder on Kaggle GPU?"
# Should rank kaggle-gpu above rlat (if both have cartridges)
```

### Phase 4: Lifecycle

**Scope**: Freshness tracking, quality metrics, gap detection.

**Modify**:
- `src/resonance_lattice/cli.py` -- add `skill profile`, `skill freshness`, `skill gaps`, `skill compare`

**Verify**:
```bash
rlat skill profile rlat
rlat skill freshness
```

---

## Edge Cases

| Scenario | Behaviour |
|----------|----------|
| Skill with no cartridge fields | Works exactly as today. Zero change. |
| `cartridges:` but no `cartridge-queries:` | Tier 2 empty, Tiers 1+3+4 only. |
| `cartridge-queries:` but no `cartridges:` | Queries target skill-local cartridge. |
| Missing cartridge file at runtime | Warning. Graceful fallback to Tier 1 only. |
| Stale cartridge | Still usable. `freshness` reports it. `on-change` auto-rebuilds. |
| No derived queries supplied | Tier 4 empty. Budget shifts to Tier 3 (40/60 split). |
| Derived query returns same passages as Tier 2/3 | Deduplicated by source_id. |
| `cartridge-derive: false` | Tier 4 disabled. Budget splits 40/60 between T2/T3. |
| Budget exceeded | Enforced via `_truncate_to_tokens`. Per-tier allocation. |
| `constrain` mode + low energy | **Always inject** -- no gating. Use `GroundingProjector`. |
| `augment` mode + low novelty | **Suppress** -- Tier 1 only. Model already knows this. |
| Incompatible encoders across cartridges | Hard error at resolve time. Message names the mismatched cartridges. |
| Multiple skills reference same cartridge | `SkillRuntime` caches the load. Single copy in memory. |

---

## Design Decisions and Trade-offs

### Mode-aware gating vs uniform gating

The `GatedProjector` was designed to suppress broad context when the model already
knows the topic. But `constrain` mode is the opposite -- the skill author explicitly
chose "answer only from these sources." Gating that away defeats the purpose.
Solution: `constrain` skips the gate entirely and uses `GroundingProjector`.

### Token budgets vs line budgets

RL already budgets in tokens (`MaterialiserConfig.token_budget`, projector char limits).
Using lines would be inconsistent and unpredictable (a code block is 1 line but could be
200 tokens). The frontmatter uses `cartridge-budget` in tokens. Internally, enforcement
uses `_estimate_tokens()` and `_truncate_to_tokens()` from the materialiser.

### SkillRuntime as a dedicated object vs CLI plumbing

`auto_route_query()` takes `(query_phase, cartridge_fields, top_n)` -- it needs an
encoded phase and loaded dense fields, not skill names or paths. Composing requires
encoder-compatible lattices. This means every skill operation needs: parse frontmatter,
resolve paths, check compatibility, load fields/lattices, encode once. Without a runtime
object, each command duplicates this. `SkillRuntime` owns it.

### MCP stays cartridge-centric

The MCP server is a stdio process tied to one cartridge. Skill routing requires scanning
`.claude/skills/`, parsing frontmatter, and loading multiple fields. These are orchestration
concerns. Claude's skill selection already runs outside MCP. RL contributes a scoring
signal via `rlat skill route`, not a replacement for the routing infrastructure.

### One cartridge per skill-local build, multiple external references

A skill builds at most one cartridge from its own `references/` but can reference many
external cartridges. Keeps the build boundary clean while allowing composition.

---

## Open: YAML Parser

The current parser (`skill.py:_parse_yaml_block`) handles the current schema but is
fragile. Before expanding the schema:

**Option A**: Add `pyyaml` as an optional dependency. Use it when available, fall back
to the minimal parser.

**Option B**: Vendor a minimal YAML subset parser with proper error messages.

**Recommendation**: Option A. `pyyaml` is ubiquitous in Python environments. The
fallback keeps zero-dependency installs working.

---

## Future Considerations

### Cached Tier 2 Results

Since Tier 2 queries are deterministic (same every trigger), their results could be
pre-computed at build time and cached alongside the cartridge. This eliminates Tier 2
search latency entirely. Cache invalidation: rebuild when cartridge changes.

### Cross-Project Skill Federation

A workspace with multiple projects could federate their skill manifests. A query
would route across all projects' skills. Requires workspace-level manifest aggregation.

### Skill Quality Benchmarks

`rlat skill profile` provides structural metrics (energy, rank, entropy). True quality
requires task-level evaluation: does four-tier injection produce better outputs than
static loading? The `skill-creator` eval framework could A/B test this.
