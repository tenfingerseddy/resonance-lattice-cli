# Resonance Lattice CLI Reference

The `rlat` command-line interface is the primary way to build, query, inspect, and manage Resonance Lattice knowledge models. This document is the human-authored companion to the mechanically-regenerated [CLI_REFERENCE.md](CLI_REFERENCE.md) — it explains concepts, workflows, and the *why* behind each command. For the exact flag surface of every command, see CLI_REFERENCE.md.

For related documentation see [RQL_REFERENCE.md](RQL_REFERENCE.md) (programmable query language) and [ENCODERS.md](ENCODERS.md) (encoder presets and benchmarks).

**Version**: 0.11.0 (heading to v1.0.0 2026-06-08). Regenerate the auto-generated flag reference with `python scripts/regenerate_cli_docs.py`.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Key Concepts](#key-concepts)
4. [Output Formats](#output-formats)
5. [Injection Modes](#injection-modes)
6. [Encoder System](#encoder-system)
7. [Build and Ingestion](#build-and-ingestion)
8. [Warm Worker System](#warm-worker-system)
9. [Common Workflows](#common-workflows)
10. [Environment Variables](#environment-variables)
11. [Command Reference](#command-reference)
    - [Primary](#primary-commands): search, ask, profile, compare, ls, info
    - [Build](#build-commands): build, add, sync, refresh, freshness, repoint, init, ingest, init-project, setup
    - [Serve and Context](#serve-and-context-commands): serve, summary, mcp
    - [Query and Analysis](#query-and-analysis-commands): query, resonate, ask, compose, contradictions, topology, xray, locate, probe
    - [Algebra](#algebra-commands): merge, forget, diff
    - [Export](#export-commands): export
    - [Encoder](#encoder-commands): encoders
    - [Skill](#skill-commands): skill build, skill sync, skill search, skill info, skill inject, skill route, skill profile, skill freshness, skill gaps, skill compare
    - [Memory](#memory-commands): memory init, memory write, memory recall, memory primer, memory consolidate, memory gc, memory profile, memory stats
12. [Choosing the Right Command](#choosing-the-right-command)

---

## Installation

```bash
pip install resonance-lattice
```

Verify the installation:

```bash
rlat --help
```

**Optional dependencies** for binary file ingestion:

| Format | Package |
|--------|---------|
| `.pdf` | `pdfplumber` or `PyPDF2` |
| `.docx` | `python-docx` |
| `.xlsx` / `.xls` | `openpyxl` |

Install individually as needed:

```bash
pip install pdfplumber python-docx openpyxl
```

---

## Quick Start

### 1. Build a knowledge model from your project

```bash
rlat build ./docs ./src -o project.rlat
```

This encodes all supported files into a single `.rlat` knowledge model containing a semantic field, source registry, and evidence store.

### 2. Search it

```bash
rlat search project.rlat "how does authentication work?"
```

Returns ranked passages with scores, source file paths, coverage profile, and topic clustering.

### 3. Inspect the model

```bash
rlat profile project.rlat
```

Shows the knowledge model's semantic shape: per-band energy distribution, source count, effective rank, and coverage summary.

### 4. Generate an assistant primer

```bash
rlat summary project.rlat -o .claude/resonance-context.md
```

Produces a compressed context document suitable for inclusion in CLAUDE.md or other assistant system prompts. Bootstraps itself by running internal queries to sample the knowledge model's knowledge.

---

## Key Concepts

### Knowledge Models (.rlat files)

A knowledge model is a single `.rlat` file that packages three layers into one portable artifact:

| Layer | Contents | Scaling |
|-------|----------|---------|
| **Field** | Semantic tensor (B x D x D) | Fixed-size |
| **Registry** | Source coordinates, LSH buckets, phase vectors | Scales with source count |
| **Store** | Evidence text, metadata, chunk content | Scales with corpus size |

Knowledge Models are portable by default. For a fully self-contained artifact (copy-and-go, no source tree, no network), build with `--store-mode bundled`. The default `--store-mode local` keeps the knowledge model thin and reads source files on query via `--source-root`; `--store-mode remote` pins to an upstream GitHub repo and serves from a SHA-pinned local cache. See [STORAGE_MODES.md](STORAGE_MODES.md) for the full comparison.

### Three Layers

- **Field**: The latent semantic model. A multi-band interference tensor where each band captures a different abstraction level. Fixed-size regardless of source count.
- **Registry**: Maps resonance hits back to source coordinates. Stores phase vectors and LSH buckets for fast lookup.
- **Store**: The lossless store — reads raw source files and returns passages + metadata. Ships in three serving topologies: `local` (default; files on disk, resolved at query time), `bundled` (self-contained; files packed inside the `.rlat` as zstd frames), and `remote` (SHA-pinned upstream GitHub repo with local cache). Legacy `embedded` mode (pre-chunked SQLite) is deprecated and will be removed in v2.0.0.

### Encoder

The encoder converts text into phase vectors that the field can store and query. Three encoders are well-supported and measured on BEIR-5:

- **`bge-large-en-v1.5`** — CLI default since 2026-04-20 (commit `3e0642f`). 335M params; CPU / Intel Arc iGPU friendly; best ecosystem; wins 4/5 corpora on the 2026-04-22 5-BEIR rebench.
- **`e5-large-v2`** — 335M params, opt-in; wins on counter-argument / debate retrieval (ArguAna-class corpora).
- **`qwen3-8b`** — 8B params, opt-in; frontier quality on dense-only (0.500 BEIR-5 avg vs 0.445 BGE); needs a 16 GB GPU.

See [ENCODER_CHOICE.md](ENCODER_CHOICE.md) for the per-workload decision guide. All three use random projection heads over the pretrained backbone — no trained weights ship.

### Bands

The field is divided into 5 bands by default, each capturing a different level of semantic abstraction:

| Band | Label | Captures |
|------|-------|----------|
| 0 | domain | Broad subject area |
| 1 | topic | Specific topic within a domain |
| 2 | relations | Relationships, connections, dependencies |
| 3 | entity | Named entities, identifiers, symbols |
| 4 | verbatim | Close lexical matches |

These are working labels. The bands emerge from the projection heads and are not guaranteed ontology categories.

---

## Output Formats

Most query commands accept `--format` with one of four values:

### `text` (default for most commands)

Human-readable terminal output with ANSI color. Shows coverage bars, ranked passages with scores, topic clusters, source file paths, and timing.

```
  how does authentication work?

  85ms (warm) | 10 results | confidence 82%
  5 sources · focused match
  topics: Auth(4)  Middleware(2)  Config(1)

  1. 0.892  auth.py / verify_token
      Validates the JWT token against the signing key and checks expiry...
      src/auth.py

  2. 0.845  middleware.py / AuthMiddleware
      The authentication middleware intercepts every request...
      src/middleware.py
```

### `json`

Complete structured output for scripts and programmatic consumption. Includes all metadata: `latency_ms`, `warm`, `results` array with `score`, `raw_score`, `band_scores`, `passage`, `source_id`, `file_path`, plus `coverage`, `related_topics`, and `contradictions`.

```bash
rlat search project.rlat "auth" --format json | jq '.results[:3]'
```

### `context`

Compressed line-per-passage format optimized for LLM context injection. Default format for `resonate` and `summary`. Minimal overhead, maximum information density.

```
- [0.89] Validates the JWT token against the signing key and checks expiry...
- [0.85] The authentication middleware intercepts every request before...
- [0.78] Configuration for auth providers is loaded from config.yaml...
```

### `prompt`

Rich markdown with coverage bars, full passages with band scores, related topics from cascade, and contradiction highlights. Designed for copy-paste into LLM prompts or rendering in markdown-capable tools.

```markdown
## Resonance Results (confidence: 82%)

  content      ████████░░░░  45.2
  entities     █████░░░░░░░  23.1

### Passages
[1] (0.892) auth.py / verify_token
    Validates the JWT token against the signing key...

### Related Topics
  - (hop 1, 0.756) middleware_config: Auth provider setup...
```

---

## Injection Modes

When using `--format context` or `--format prompt`, you can frame the output with a system prompt using `--mode`. Available on `search`, `query`, and `resonate`.

### `augment`

> You are a knowledgeable expert. Answer from your own knowledge.
> Below are reference passages from an authoritative knowledge base. Use them
> to add specific details, correct any uncertainties, and cite sources [1], [2]
> where applicable. Do NOT limit your answer to only what the sources contain.

Best for general-purpose assistants that should use their own knowledge alongside the knowledge model evidence.

### `constrain`

> You are an AI assistant. Answer ONLY using the provided reference passages.
> Do NOT use your own knowledge or make claims beyond what the sources contain.
> Cite every claim with [1], [2] etc. If the context doesn't cover something,
> explicitly state: 'The provided sources do not cover this.'

Best for high-stakes or compliance scenarios where hallucination must be minimized.

### `knowledge`

> You are an AI assistant. You may not have training data on this topic.
> Below are authoritative reference passages. Base your answer primarily
> on this context, and be transparent about what the context doesn't cover.

Best for domain-specific or proprietary content the model has never seen.

### `custom`

Supply your own system prompt:

```bash
rlat resonate project.rlat "query" --mode custom --custom-prompt "You are a code reviewer..."
```

---

## Encoder System

For the full encoder guide — available presets, comparison table, choosing an encoder for code vs prose vs multilingual — see [ENCODERS.md](ENCODERS.md).

### Resolution Order

When loading an encoder, the CLI resolves in this order:

1. **`--encoder`** flag — uses the specified preset (`bge-large-en-v1.5`, `e5-large-v2`, `qwen3-8b`, `random`) or a HuggingFace model name
2. **Stored encoder** — restores the encoder saved inside the knowledge model at build time
3. **Default** — downloads `BAAI/bge-large-en-v1.5` on first use

In practice, most users pick one encoder at build time and never think about it again — the knowledge model stores its encoder, and queries restore it automatically. `--encoder` on `search` / `ask` is only useful when you're explicitly overriding (rare; mismatched encoders are blocked by `_check_encoder_consistency`). Use `--encoder random` for fast testing without downloading a model.

### ONNX Acceleration

For faster CPU encoding, export the backbone to ONNX format. The CLI auto-detects an ONNX directory alongside the knowledge model:

- `project_onnx/` (same stem as knowledge model)
- `onnx_backbone/` (generic name)

Or specify explicitly:

```bash
rlat search project.rlat "query" --onnx ./onnx_backbone/
```

ONNX provides 2-5x CPU encoding speedup.

### Encoder Consistency

Knowledge Models track the encoder fingerprint (backbone name, bands, dim). When using `add` or `sync`, a mismatch between the current encoder and the knowledge model's encoder triggers a warning. This prevents mixing incompatible embeddings in one knowledge model.

---

## Build and Ingestion

### Supported File Types

**Text and markup:** `.txt`, `.md`, `.rst`, `.html`, `.htm`, `.xml`

**Code:** `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.java`, `.c`, `.cpp`, `.h`, `.hpp`, `.go`, `.rb`, `.rs`, `.swift`, `.kt`, `.lua`, `.r`, `.php`, `.sh`, `.sql`, `.css`, `.scss`

**Data and config:** `.json`, `.yaml`, `.yml`, `.toml`, `.csv`, `.tsv`

**Binary (requires optional deps):** `.docx`, `.pdf`, `.xlsx`, `.xls`

Explicitly named files are always included regardless of extension. Extension filtering only applies to recursive directory discovery.

### Skipped Directories

During recursive file discovery, these directories are automatically skipped:

`__pycache__`, `.git`, `.hg`, `.svn`, `node_modules`, `dist`, `build`, `.venv`, `venv`, `.env`, `.tox`, `.mypy_cache`, `.pytest_cache`, `.egg-info`, `.eggs`, `htmlcov`, `.ipynb_checkpoints`

### Incremental Builds

Two commands support incremental updates without full rebuilds:

- **`add`** — appends new files to an existing knowledge model. Skips files already present (by content hash).
- **`sync`** — bidirectional sync with source directories. Adds new files, re-encodes modified files, and forgets deleted files.

Both commands use manifest tracking to detect changes.

### Manifest Tracking

The build system maintains a `__manifest__` entry inside each knowledge model. For every ingested file, it stores:

- **Content hash** (MD5) for change detection
- **Chunk IDs** produced, enabling exact removal via `forget`
- **Encoder fingerprint** to enforce consistency

Legacy knowledge models built before manifest support will print a warning on first `add`/`sync` and begin tracking from that point.

### Store Modes

Three canonical modes, one lossless-store abstraction. See
[STORAGE_MODES.md](STORAGE_MODES.md) for the full reference.

| Mode | Flag | Behavior |
|------|------|----------|
| **local** (default; historical alias `external`) | `--store-mode local` or `--store-mode external` | Field + registry + manifest inside the `.rlat`; source files resolved from disk at query time via `--source-root` |
| **bundled** | `--store-mode bundled` | Lossless + self-contained — raw source files packed inside the `.rlat` as zstd frames. Re-chunking, window widening, drift detection all still work |
| **remote** | `--store-mode remote` (auto-selected for GitHub URL inputs) | Lossless + HTTP-backed — knowledge model pins to a commit SHA on an upstream repo; queries serve from a SHA-pinned local cache under `~/.cache/rlat/remote/` |
| **embedded** (deprecated) | `--store-mode embedded` | Legacy pre-chunked SQLite store; v2.0.0 removal target. Prefer `bundled` for the self-contained use case |

Choose by use case:
- `local` — developing against a working copy; large corpora where the knowledge model should stay thin.
- `bundled` — shipping a single-file artifact (HF Hub demos, offline use, CI).
- `remote` — pointing at an upstream repo you don't own (docs, public codebases). Manage freshness with `rlat freshness` (read-only) and `rlat sync` (pull upstream diff).

### Registry Quantization

Compress registry phase vectors at build time:

| Flag | Compression | Use case |
|------|------------|----------|
| `--quantize-registry 0` | None (default) | Full precision |
| `--quantize-registry 8` | ~50% reduction | Recommended for most use cases, higher quality |
| `--quantize-registry 4` | ~87% reduction | Aggressive, slight quality loss |

Quantization is data-oblivious (no codebook training needed).

### Compression

The serialized knowledge model can be compressed:

| Flag | Notes |
|------|-------|
| `--compression none` | Default. No compression. |
| `--compression zstd` | Best ratio. Requires `zstd` library. |
| `--compression lz4` | Fastest decompression. |

---

## Warm Worker System

The `search` command can spawn a background HTTP worker that keeps the knowledge model loaded in memory. This eliminates load time on subsequent queries.

### How It Works

1. First `rlat search` runs the query in-process (cold path, includes load time).
2. After returning results, it spawns a background worker process with the knowledge model pre-loaded.
3. Subsequent `rlat search` calls on the same knowledge model hit the warm worker (near-zero load time).

The worker key includes knowledge model path, modification time, encoder, and source root. A changed knowledge model automatically invalidates the worker.

### Cold vs Warm

| Metric | Cold | Warm |
|--------|------|------|
| Load time | 50-500ms (depends on knowledge model size) | ~0ms |
| JSON output `"warm"` field | `false` | `true` |
| First query | Always cold | Always warm |

### Disabling the Worker

```bash
# Per-command
rlat search project.rlat "query" --no-worker

# Environment variable (all commands)
export RLAT_NO_WORKER=1
```

### Idle Timeout

Workers shut down after 30 minutes of inactivity. Stale worker state files are cleaned up after 24 hours. If a worker dies, the CLI silently falls back to the cold path.

---

## Common Workflows

### New Project Setup

```bash
rlat init-project
```

Auto-detects `docs/`, `src/`, `lib/`, `README.md`, `CLAUDE.md`, and `AGENTS.md` in the current directory. Builds a knowledge model, generates a summary, and prints integration hints. Equivalent to:

```bash
rlat build ./docs ./src -o .rlat/project.rlat
rlat summary .rlat/project.rlat -o .claude/resonance-context.md
```

### Incremental Update Cycle

After changing source files:

```bash
rlat sync project.rlat ./docs ./src
```

Reports added, updated, removed, and unchanged files.

### Comparing Knowledge Model Versions

```bash
rlat compare old.rlat new.rlat
```

Shows overlap, unique coverage per knowledge model, and per-band energy differences.

For an algebraic delta:

```bash
rlat diff old.rlat new.rlat -o delta.rlat
```

### Merging Team Knowledge

```bash
rlat merge team-a.rlat team-b.rlat -o combined.rlat
```

Merge is commutative: `merge(A, B)` equals `merge(B, A)`.

### Privacy-Preserving Export

Share the semantic model without source text:

```bash
rlat export project.rlat -o public.rlat --field-only
```

The exported knowledge model contains field + registry but no evidence store.

### Claude Code / Cursor Integration

Generate a primer and include it in your project configuration:

```bash
rlat summary project.rlat -o .claude/resonance-context.md
```

Then reference it from `CLAUDE.md`:

```markdown
@.claude/resonance-context.md
```

### MCP Server

Start an MCP server for Claude Code integration:

```bash
rlat mcp project.rlat
```

Add to your Claude Code settings:

```json
{
  "mcpServers": {
    "resonance": {
      "command": "rlat",
      "args": ["mcp", "project.rlat"]
    }
  }
}
```

---

## Environment Variables

| Variable | Effect |
|----------|--------|
| `RLAT_VERBOSE=1` | Show encoder loading details, timing breakdown, and warnings |
| `RLAT_NO_WORKER=1` | Disable the background warm-search worker |
| `NO_COLOR` | Disable ANSI color output (follows [no-color.org](https://no-color.org)) |
| `FORCE_COLOR` | Force ANSI color output even in non-TTY environments |

---

## Command Reference

### Primary Commands

---

#### `rlat search`

Primary enriched semantic query. Returns ranked passages with coverage profile, topic clustering, source file paths, and optional cascade/contradiction enrichment.

```
rlat search <lattice> <query> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |
| `query` | Yes | Query text |

**Core options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--top-k` | int | 10 | Number of results |
| `--format` | choice | text | Output format: `text`, `json`, `context`, `prompt` |
| `-v, --verbose` | flag | off | Show raw scores and detailed timings |

**Enrichment options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cascade` | flag | off | Enable related topics cascade |
| `--no-cascade` | flag | on | Disable related topics cascade (default) |
| `--cascade-depth` | int | 2 | Cascade hop depth |
| `--with-contradictions` | flag | off | Enable contradiction detection |
| `--no-contradictions` | flag | on | Disable contradiction detection (default) |
| `--contradiction-threshold` | float | None | Destructive interference threshold (defaults to -0.3 when contradictions are enabled) |
| `--subgraph` | flag | off | Expand results with spectral neighbours |
| `--subgraph-k` | int | 3 | Neighbours per result for subgraph expansion |

**Encoder options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--encoder` | str | None | Encoder override (`e5-large-v2`, `random`) |
| `--source-root` | str | None | External source root for file-backed store |
| `--onnx` | str | None | ONNX backbone directory (auto-detects `<stem>_onnx/`) |

**Performance options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--no-worker` | flag | off | Disable background warm worker (env: `RLAT_NO_WORKER=1`) |

**Reranking options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--rerank` | choice | auto | Reranking mode: `auto` skips when dense is confident, `true` forces reranking, `false` disables |

**Injection options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mode` | choice | None | Injection framing: `augment`, `constrain`, `knowledge`, `custom` |
| `--custom-prompt` | str | None | Custom system prompt (requires `--mode custom`) |

**Composition options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--with` | str | [] | Additional `.rlat` knowledge model(s) to compose with (repeatable). Merges fields, dispatches searches to each registry independently |
| `--through` | str | None | Project primary knowledge model through this `.rlat`'s perspective (semantic projection) |
| `--diff` | str | None | Show what's new in the primary knowledge model vs this baseline `.rlat` (queryable semantic diff) |
| `--explain` | flag | off | Show composition diagnostics before searching (overlap, novelty, contradiction ratio) |

**Topic control options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--boost` | str | [] | Boost a semantic topic during search (repeatable). Amplifies the topic direction in the field |
| `--suppress` | str | [] | Suppress a semantic topic during search (repeatable). Attenuates the topic direction |
| `--boost-strength` | float | 0.5 | Strength of topic boosting |
| `--suppress-strength` | float | 0.3 | Strength of topic suppression |

**EML corpus transforms (v0.11.0+):**

These apply nonlinear spectral transforms to the corpus field *before* retrieval. Unlike `--boost`/`--suppress` which target specific topics, EML transforms reshape the field's entire eigenvalue spectrum — they're unsupervised and don't require you to name topics. All are scale-invariant (normalise eigenvalues before filtering, rescale after).

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--sharpen` | float | None | Sharpen the corpus for more precise retrieval (try 0.5–2.0). Self-EML (`expm(F) - logm(F)`): exponentially amplifies dominant topics, compresses noise. Use when results feel too fuzzy |
| `--soften` | float | None | Soften the corpus for broader exploration (try 0.5–1.5). Logarithmic flattening: surfaces buried topics under dominant ones. Use when results feel too narrow |
| `--contrast` | str | None | Path to a background `.rlat` knowledge model for asymmetric REML contrast (`expm(primary) - logm(background)`). Returns results distinctive to the primary. Use for "what does my docs know that vendor docs don't?" |
| `--tune` | choice | None | Retrieval mode preset: `focus` (precision for specific questions), `explore` (breadth for research), `denoise` (clean noisy corpora). One word, task-matched |

**Cascade routing options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cascade-through` | str[] | None | Cross-knowledge model cascade: follow semantic links through a route of `.rlat` files |

**Access and lens options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--access` | str | None | Apply an access policy (`.rlens` file). Restricts visible knowledge to the policy's allowed subspace |
| `--lens` | str | None | Apply a knowledge lens: a `.rlens` file path or a built-in name (`sharpen`, `flatten`, `denoise`) |

**Conversation memory filters:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--session` | str | None | Filter results by session ID |
| `--after` | str | None | Filter results after ISO timestamp |
| `--before` | str | None | Filter results before ISO timestamp |
| `--speaker` | choice | None | Filter results by speaker role: `human`, `assistant`, `system`, `qa_pair` |
| `--recency-weight` | float | 0.0 | Blend recency into ranking (0.0=off, 0.3=moderate, 1.0=recency-only) |

**Examples:**

```bash
# Basic search
rlat search project.rlat "how does auth work?"

# JSON output with all enrichment
rlat search project.rlat "auth" --format json --cascade --with-contradictions

# LLM-ready output with constrained framing
rlat search project.rlat "auth" --format prompt --mode constrain

# Fast search with ONNX and no worker
rlat search project.rlat "auth" --onnx ./onnx_backbone/ --no-worker

# Multi-knowledge model composition
rlat search frontend.rlat "auth" --with backend.rlat --explain

# Semantic projection through another knowledge model's lens
rlat search current.rlat "auth" --through security-policy.rlat

# Queryable diff against a baseline
rlat search v2.rlat "what changed in auth?" --diff v1.rlat

# Topic boosting and suppression
rlat search project.rlat "deployment" --boost "kubernetes" --suppress "legacy"

# EML corpus transforms — task-matched retrieval modes
rlat search project.rlat "exact error code for missing workspace" --tune focus
rlat search project.rlat "design trade-offs of capacity autoscaling" --tune explore
rlat search project.rlat "what's the schema for this table" --sharpen 1.5

# EML contrast — what's unique to my docs vs a generic baseline?
rlat search internal-api.rlat "authentication flow" --contrast vendor-docs.rlat

# Conversation memory: recent assistant messages only
rlat search memory.rlat "what did we discuss?" --session chat-42 --speaker assistant --recency-weight 0.3
```

---

#### `rlat profile`

Semantic profile of a knowledge model. Shows per-band energy distribution, source count, effective rank, coverage patterns, and overall model health.

```
rlat profile <lattice> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | Output format: `text`, `json` |

**Example:**

```bash
rlat profile project.rlat --format json
```

---

#### `rlat compare`

Compare two knowledge models. Reports overlap, unique coverage per knowledge model, per-band energy differences, and structural similarity.

```
rlat compare <lattice_a> <lattice_b> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice_a` | Yes | First `.rlat` file |
| `lattice_b` | Yes | Second `.rlat` file |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | Output format: `text`, `json` |

**Example:**

```bash
rlat compare baseline.rlat current.rlat --format json
```

---

#### `rlat ls`

List sources in a knowledge model. Shows source IDs, file paths, and optional summaries.

```
rlat ls <lattice> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | Output format: `text`, `json` |
| `-v, --verbose` | flag | off | Show summaries for each source |
| `--head` | int | None | Show only the first N sources |
| `--grep` | str | None | Filter sources by substring match |

**Examples:**

```bash
# List all sources
rlat ls project.rlat

# Filter and limit
rlat ls project.rlat --grep "auth" --head 5 -v
```

---

#### `rlat info`

Display lattice metadata: source count, field type, dimensions, bands, compression, and encoder info.

```
rlat info <lattice>
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |

**Example:**

```bash
rlat info project.rlat
```

---

### Build Commands

---

#### `rlat build`

Build a knowledge model from source files. Discovers files recursively, chunks them, encodes phase vectors, and superposes them into the field.

```
rlat build <inputs...> -o <output> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `inputs` | Yes | One or more files or directories |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output, -o` | str | **required** | Output `.rlat` file |
| `--bands` | int | 5 | Number of frequency bands |
| `--dim` | int | 2048 | Field dimension per band |
| `--field-type` | choice | dense | Field backend: `dense`, `factored`, `pq` |
| `--precision` | choice | f32 | Numeric precision: `f16`, `bf16`, `f32` |
| `--compression` | choice | none | Serialization compression: `none`, `zstd`, `lz4` |
| `--encoder` | str | None | Encoder choice (`e5-large-v2`, `random`) |
| `--quantize-registry` | int | 0 | Quantize registry phases (0=off, 8=~50% compression, 4=~87% compression) |
| `--store-mode` | choice | local | One of `bundled`, `local` (= `external`), `remote`, `embedded` (deprecated). See [STORAGE_MODES.md](STORAGE_MODES.md) |
| `--path` | str | None | Remote builds only: scope to a subdirectory of the repo (e.g. `--path Doc` for CPython docs). Persisted into `__remote_origin__` so `rlat freshness` and `rlat sync` also respect it. Ignored for local builds — pass specific paths as inputs instead |
| `--progress` | flag | off | Show JSON progress events on stderr |
| `--onnx` | str | None | ONNX backbone directory for faster encoding |
| `--input-format` | choice | (auto) | Force input format: `conversation` for chat logs |
| `--session` | str | None | Session ID to tag all chunks with (conversation memory) |
| `--timestamp` | str | None | ISO timestamp to tag all chunks with (conversation memory) |

**Examples:**

```bash
# Standard build (local mode — default)
rlat build ./docs ./src -o project.rlat

# Compressed build with quantized registry
rlat build ./docs -o project.rlat --compression zstd --quantize-registry 8

# Self-contained bundled knowledge model (lossless — raw files packed inside)
rlat build ./docs -o project.rlat --store-mode bundled

# Remote build from a GitHub repo — pins to the default-branch HEAD SHA
rlat build https://github.com/MicrosoftDocs/fabric-docs -o fabric-docs.rlat

# Remote build pinned to a specific branch, tag, or commit
rlat build https://github.com/MicrosoftDocs/fabric-docs#release-branch -o fabric-docs.rlat
rlat build https://github.com/MicrosoftDocs/fabric-docs@abc1234 -o fabric-docs.rlat

# Remote build scoped to a subdirectory (essential for monorepos)
#   --path scope is persisted into __remote_origin__ — freshness and
#   sync only track drift under that prefix
rlat build https://github.com/python/cpython --path Doc -o python-stdlib.rlat
rlat build https://github.com/pytorch/pytorch --path docs/source -o pytorch-docs.rlat
```

---

#### `rlat add`

Incrementally add files to an existing knowledge model. Skips files already present (by content hash). Uses manifest tracking to avoid duplicates.

```
rlat add <lattice> <inputs...> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |
| `inputs` | Yes | Files or directories to add |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--encoder` | str | None | Encoder override |
| `--onnx` | str | None | ONNX backbone directory for faster encoding |
| `--input-format` | choice | (auto) | Force input format: `conversation` for chat logs |
| `--session` | str | None | Session ID to tag all chunks with (conversation memory) |
| `--timestamp` | str | None | ISO timestamp to tag all chunks with (conversation memory) |

**Example:**

```bash
rlat add project.rlat ./new_docs
```

---

#### `rlat sync`

Sync a knowledge model with its source of truth. Dispatches on store mode:

- **Local (external) knowledge models** — detect new / modified / deleted files under the supplied source directories, add/update/remove chunks accordingly.
- **Remote knowledge models** — pull the upstream GitHub diff. Omit `inputs` to invoke this path; the pinned `__remote_origin__` drives the sync.

```
rlat sync <lattice> [inputs...] [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |
| `inputs` | Local mode only | Source directories to sync from. Omit for remote-mode knowledge models |

**Options (common):**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--encoder` | str | None | Encoder override |
| `--output, -o` | str | in-place | (Remote) where to write the synced knowledge model |
| `--progress` | flag | off | (Local) show JSON progress events on stderr |
| `--onnx` | str | None | ONNX backbone directory for faster encoding |
| `--input-format` | choice | (auto) | (Local) force input format: `conversation` for chat logs |
| `--session` | str | None | (Local) session ID tag for chunks (conversation memory) |
| `--timestamp` | str | None | (Local) ISO timestamp tag for chunks (conversation memory) |

**Examples:**

```bash
# Local: re-ingest a working copy
rlat sync project.rlat ./docs ./src

# Remote: pull upstream diff (lockfile-style upgrade)
rlat sync fabric-docs.rlat
```

---

#### `rlat freshness`

Read-only upstream drift check for a remote-mode knowledge model. One GitHub API call; never mutates the knowledge model or touches the disk cache. Exit code `0` when up-to-date, `1` when drift is detected — convenient for CI gates.

```
rlat freshness <lattice>
```

**Example:**

```bash
rlat freshness fabric-docs.rlat
# pinned at  abc1234aa
# upstream   def5678bb
# diff       +3 added, ~12 modified, -1 removed
# run `rlat sync <cart>` to apply.
```

Raises if the knowledge model isn't remote-mode. Use `rlat refresh` for local-mode drift detection.

---

#### `rlat refresh`

Re-index drifted files in a local-mode knowledge model. Preserves the field tensor where chunk hashes still match — only drifted / missing / newly-added chunks trigger forget+superpose cycles. Unchanged chunks stay byte-identical in the field, so `refresh` is the fast incremental update path for a working-copy knowledge model.

```
rlat refresh <lattice> --source-root <dir> [options]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `lattice` | path | **required** | Path to the `.rlat` file to refresh |
| `--source-root` | path | **required** | Directory manifest paths resolve under. The manifest's `__source_root_hint__` is advisory; pass the current root explicitly. |
| `--output, -o` | path | overwrite input | Where to write the refreshed knowledge model. Defaults to overwriting in place. |
| `--encoder` | str | stored | Encoder preset or HuggingFace model ID. Defaults to the one embedded in the knowledge model at build time (guarantees build/query parity). |
| `--onnx` | path | — | ONNX backbone directory. |
| `--openvino` | path | — | OpenVINO IR directory. |
| `--openvino-device` | choice | — | OpenVINO target: `CPU`, `GPU`, `NPU`, `AUTO`. |
| `--openvino-static-seq-len` | int | — | Fixed sequence length for OpenVINO NPU. |

**What it does (the refresh algorithm):**

1. Load the knowledge model with `--source-root` so a `LocalStore` is attached.
2. Group manifest entries by `source_file`.
3. For each file:
   - **Missing on disk** → forget every chunk bound to that file (file was deleted).
   - **Exists** → re-chunk and compare `content_hash` per chunk:
     - **Match** → skip (field preserved byte-identical).
     - **Mismatch** → `update(old_sid, new_phase)` — atomic forget+superpose that keeps the `source_id` stable so downstream references survive.
     - **New chunk** (file grew) → superpose with a predictable `source_id` derived from the file's existing id prefix.
     - **Stale chunk** (file shrunk) → remove.
4. Save the knowledge model back in local mode with the updated manifest.

**Example:**

```bash
# Detect drift and re-index in place
rlat refresh project.rlat --source-root .

# Write refreshed copy alongside the original
rlat refresh project.rlat --source-root . -o project-refreshed.rlat
```

Use `rlat refresh` for local-mode drift handling, `rlat freshness` + `rlat sync` for remote-mode drift handling, and `rlat sync` for add-and-remove reconciliation in local mode. `refresh` is the fast path when files mutated in place but the set of files didn't change.

---

#### `rlat init`

Create a new empty lattice with specified parameters.

```
rlat init -o <output> [options]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output, -o` | str | **required** | Output `.rlat` file |
| `--bands` | int | 5 | Number of frequency bands |
| `--dim` | int | 2048 | Field dimension per band |
| `--field-type` | choice | dense | Field backend: `dense`, `factored`, `pq` |
| `--pq-subspaces` | int | 8 | PQ subspaces (only for `pq` field type) |
| `--pq-codebook-size` | int | 1024 | PQ codebook size (only for `pq` field type) |
| `--svd-rank` | int | 512 | SVD rank (only for `factored` field type) |
| `--precision` | choice | f32 | Numeric precision: `f16`, `bf16`, `f32` |
| `--compression` | choice | none | Serialization compression: `none`, `zstd`, `lz4` |

**Example:**

```bash
rlat init -o empty.rlat --field-type factored --svd-rank 256
```

---

#### `rlat ingest`

Ingest documents into an existing lattice. Lower-level than `add` — operates on a single file or directory.

```
rlat ingest <lattice> <input> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |
| `input` | Yes | File or directory to ingest |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--encoder` | str | None | Encoder override (`e5-large-v2`, `random`) |

**Example:**

```bash
rlat ingest project.rlat ./new_document.md
```

---

#### `rlat init-project`

One-command project setup. Auto-detects source directories, builds a knowledge model, generates a summary, and prints integration instructions.

```
rlat init-project [inputs...] [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `inputs` | No | Files or directories (auto-detects `docs/`, `src/`, `lib/`, `README.md`, `CLAUDE.md`, `AGENTS.md` if omitted) |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output, -o` | str | `.claude/resonance-context.md` | Summary output path |
| `--encoder` | str | None | Encoder choice (`e5-large-v2`, `random`) |
| `--onnx` | str | None | ONNX backbone directory for faster encoding |
| `--auto-integrate` | flag | off | Automatically update `.mcp.json`, inject CLAUDE.md knowledge model section, and create `.rlat/manifest.json` |

**Examples:**

```bash
# Auto-detect everything
rlat init-project

# Specify sources with full integration
rlat init-project ./docs ./src --auto-integrate
```

---

#### `rlat setup`

Interactive setup wizard. Guides users through knowledge model building, encoder selection, skill-knowledge model wiring, layered memory, and tool integration. Persists configuration to `.rlat/setup.toml` for re-runnability.

```
rlat setup [options]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--non-interactive, --ni` | flag | off | Run with defaults, no interactive prompts |
| `--config` | str | None | Load setup config from TOML file |
| `--reconfigure` | flag | off | Force re-prompt even with existing setup |
| `--encoder` | str | None | Override encoder preset |
| `--no-memory` | flag | off | Disable layered memory setup |
| `--precision` | str | None | Override field precision (`f16`, `bf16`, `f32`) |
| `--compression` | str | None | Override compression (`none`, `zstd`, `lz4`) |

**Examples:**

```bash
# Interactive wizard (recommended for first time)
rlat setup

# Non-interactive with all defaults
rlat setup --ni

# Override encoder and disable memory
rlat setup --ni --encoder arctic-embed-2 --no-memory

# Replay saved config
rlat setup --config .rlat/setup.toml --ni

# Modify existing setup
rlat setup --reconfigure
```

See [SETUP_WIZARD.md](SETUP_WIZARD.md) for full documentation including the step-by-step flow, configuration file format, and architecture.

---

### Serve and Context Commands

---

#### `rlat serve`

Start an HTTP server exposing the knowledge model for network queries. Endpoints: `GET /health`, `GET /info`, `POST /query`, `POST /search`, `POST /add`, `POST /remove`.

```
rlat serve <lattice> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--port` | int | 8080 | Port number |
| `--host` | str | 0.0.0.0 | Bind address |
| `--encoder` | str | None | Encoder override |

**Example:**

```bash
rlat serve project.rlat --port 9090
```

---

#### `rlat summary`

Generate a pre-injection context primer. Bootstraps by running internal queries to sample the knowledge model's knowledge, then produces a structured summary.

```
rlat summary <lattice> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output, -o` | str | stdout | Output file path |
| `--format` | choice | context | `context` (rich primer) or `stats` (field metadata only) |
| `--queries` | str | None | Custom bootstrap queries separated by semicolons |
| `--top-k` | int | 20 | Results per bootstrap query |
| `--encoder` | str | None | Encoder override |
| `--source-root` | str | None | External source root for file-backed store |
| `--budget` | int | 2500 | Target token budget for output |
| `--sections` | str | None | Comma-separated section names to include in the primer |

**Examples:**

```bash
# Default primer
rlat summary project.rlat -o .claude/resonance-context.md

# Custom bootstrap queries
rlat summary project.rlat --queries "architecture;testing;deployment" --top-k 30

# Metadata-only stats
rlat summary project.rlat --format stats
```

---

#### `rlat mcp`

Start an MCP (Model Context Protocol) server over stdio transport. Enables Claude Code and other MCP-compatible tools to query the knowledge model directly.

```
rlat mcp <knowledge model> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `knowledge model` | Yes | Path to `.rlat` knowledge model |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--source-root` | str | None | External source root for file-backed store |
| `--onnx` | str | None | ONNX backbone directory for faster encoding |

**Example:**

```bash
rlat mcp project.rlat --source-root .
```

---

### Query and Analysis Commands

---

#### `rlat ask`

Smart query dispatcher: auto-selects the best retrieval lens (search, locate, profile, compare, negotiate, compose) based on the question's intent. Use this when you're not sure which command to run — `ask` figures it out.

```
rlat ask <lattice> <query> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |
| `query` | Yes | Your question |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--explain` | flag | off | Show which lens was selected and why, without running the query |
| `--with` | str | None | Additional knowledge model(s) for multi-knowledge model queries (repeatable) |
| `--background` | str | None | Background knowledge model for contrast queries |
| `--top-k` | int | 10 | Number of results |
| `--format` | choice | text | Output format: `text`, `json`, `context` |
| `--encoder` | str | None | Encoder override |

**Lens selection:**

| Question type | Lens selected | Example |
|---------------|---------------|---------|
| Factoid / specific | `search --tune focus` | "what is a lakehouse?" |
| Exploratory / broad | `search --tune explore` | "what approaches exist for caching?" |
| Coverage / gaps | `locate` | "what does this knowledge model cover?" |
| Overview / summarize | `profile` | "give me a high-level summary" |
| Compare two knowledge models | `compare` or `negotiate` | "how do these differ?" |
| Contrast (with background) | `search --contrast` | "what does X know that Y doesn't?" |

**Examples:**

```bash
# Auto-selects the right lens
rlat ask project.rlat "how does authentication work?"

# See which lens would be chosen without running it
rlat ask project.rlat "what are the coverage gaps?" --explain

# Multi-knowledge model query
rlat ask frontend.rlat "auth flow" --with backend.rlat
```

---

#### `rlat query`

Basic retrieval: ranked passages without enrichment. Lighter than `search` — no coverage, cascade, or contradiction analysis. Useful for scripts and debugging.

```
rlat query <lattice> <query> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |
| `query` | Yes | Query text |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--top-k` | int | 10 | Number of results |
| `--format` | choice | text | Output format: `text`, `json`, `context`, `prompt` |
| `--encoder` | str | None | Encoder override |
| `--mode` | choice | None | Injection framing: `augment`, `constrain`, `knowledge`, `custom` |
| `--custom-prompt` | str | None | Custom system prompt (requires `--mode custom`) |

**Example:**

```bash
rlat query project.rlat "storage architecture" --format json
```

---

#### `rlat resonate`

AI context output optimized for LLM injection. Same retrieval as `query` but defaults to `context` format and supports injection modes and external source resolution.

```
rlat resonate <lattice> <query> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |
| `query` | Yes | Query text |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--top-k` | int | 10 | Number of results |
| `--format` | choice | context | Output format: `text`, `json`, `context`, `prompt` |
| `--encoder` | str | None | Encoder override |
| `--mode` | choice | None | Injection framing: `augment`, `constrain`, `knowledge`, `custom` |
| `--custom-prompt` | str | None | Custom system prompt (requires `--mode custom`) |
| `--source-root` | str | None | Resolve content from local files instead of embedded store |
| `--onnx` | str | None | ONNX backbone directory for faster encoding |
| `-v, --verbose` | flag | off | Show raw scores and detailed timings |

**Examples:**

```bash
# Default context format for LLM injection
rlat resonate project.rlat "how does auth work?" --mode constrain

# With external source resolution
rlat resonate project.rlat "auth" --source-root ./docs --format prompt
```

---

#### `rlat compose`

Algebraic composition expressions. Accepts either an inline expression or a `.rctx` file defining a composition pipeline (merge, project, diff, contradict) across multiple knowledge models.

```
rlat compose <expression> <query> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `expression` | Yes | Composition expression or path to `.rctx` file |
| `query` | Yes | Search query |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--top-k` | int | 10 | Number of results |
| `--format` | choice | text | Output format: `text`, `json` |
| `--explain` | flag | off | Show composition diagnostics |
| `--encoder` | str | None | Encoder override |
| `--onnx` | str | None | ONNX backbone directory |

**Examples:**

```bash
# Merge two knowledge models and search
rlat compose "merge(frontend.rlat, backend.rlat)" "auth flow" --explain

# Use a .rctx composition file
rlat compose pipeline.rctx "deployment strategy"
```

---

#### `rlat contradictions`

Find contradicting sources for a given query. Uses destructive interference detection between top-k results to identify conflicting information.

```
rlat contradictions <lattice> <query> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |
| `query` | Yes | Query text to check for contradictions |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--band` | int | 2 | Band to check (default: relations) |
| `--threshold` | float | 0.7 | Anti-correlation threshold |
| `--top-k` | int | 20 | Number of candidates to check |
| `--encoder` | str | None | Encoder override |

**Example:**

```bash
rlat contradictions project.rlat "authentication requirements" --threshold 0.5
```

---

#### `rlat topology`

Knowledge topology analysis via eigendecomposition. Identifies knowledge clusters, connectivity patterns, and structural properties of the field.

```
rlat topology <lattice> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--band` | int | 0 | Band to analyze |
| `--top-k` | int | 20 | Number of top eigenvalues |
| `--output, -o` | str | None | Output JSON file |

**Example:**

```bash
rlat topology project.rlat --band 0 -o topology.json
```

---

#### `rlat xray`

Field X-Ray: corpus-level semantic diagnostics. Reports per-band health (effective rank, entropy, spectral gap, SNR, condition number, purity, signal eigenvalue count), overall saturation, band correlation matrix, and actionable diagnostics. Does not require an encoder — operates directly on the field matrix.

With `--deep`, adds topological analysis (community detection, Betti numbers) on top of the base report.

Requires a `DenseField` backend.

```
rlat xray <lattice> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | Output format: `text`, `json`, `prompt` |
| `--deep` | flag | off | Add topological analysis and community detection |

**Examples:**

```bash
rlat xray project.rlat
rlat xray project.rlat --deep --format json
```

---

#### `rlat locate`

Query positioning: structural analysis of where a question sits within the field's geometry. Not search (does not return passages) — reports per-band energy distribution, band focus, anti-resonance gap ratio (how much the field lacks), Mahalanobis distance from corpus center, per-band uncertainty and Fisher information, and an expansion hint (nearest richer query via steepest ascent).

Coverage labels: `strong`, `partial`, `edge`, `gap`. Requires an encoder and `DenseField` backend.

```
rlat locate <lattice> <query> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |
| `query` | Yes | Query text |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | Output format: `text`, `json`, `prompt` |
| `--encoder` | str | None | Encoder override |

**Example:**

```bash
rlat locate project.rlat "how do we handle rate limiting?"
```

---

#### `rlat probe`

RQL quick insight recipes. Each recipe composes 2-5 RQL operations into a named analysis that answers a specific question about the field or a query's position in it.

```
rlat probe <lattice> <recipe> [query] [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |
| `recipe` | Yes | Recipe name (see table below) |
| `query` | Depends | Query text (required for `novelty` and `anti`) |

**Recipes:**

| Recipe | Query required | What it reports |
|--------|----------------|-----------------|
| `health` | No | Marchenko-Pastur signal/noise split, SNR, effective rank per band |
| `novelty` | Yes | How novel the query content is relative to the corpus. Scores 0-1 with ADD/OPTIONAL/SKIP recommendation |
| `saturation` | No | Field capacity usage per band, estimated remaining source capacity |
| `band-flow` | No | Inter-band mutual information matrix, strongest and weakest band couplings |
| `anti` | Yes | What the field does NOT know about the query. Per-band gap analysis |
| `gaps` | No | Topological knowledge gap analysis: clusters, loops, robustness per band |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | Output format: `text`, `json`, `prompt` |
| `--encoder` | str | None | Encoder override |

**Examples:**

```bash
# Field health check
rlat probe project.rlat health

# Check if new content would add value
rlat probe project.rlat novelty "quantum computing concepts"

# Capacity analysis
rlat probe project.rlat saturation --format json

# Knowledge gaps for a query
rlat probe project.rlat anti "distributed consensus" --format prompt
```

---

### Algebra Commands

---

#### `rlat merge`

Merge two knowledge models into one. Combines fields, registries, and stores. Merge is commutative: `merge(A, B) = merge(B, A)`.

```
rlat merge <lattice_a> <lattice_b> -o <output>
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice_a` | Yes | First `.rlat` file |
| `lattice_b` | Yes | Second `.rlat` file |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output, -o` | str | **required** | Output merged `.rlat` file |

**Example:**

```bash
rlat merge frontend.rlat backend.rlat -o fullstack.rlat
```

---

#### `rlat forget`

Remove a source from a knowledge model. Performs algebraically exact rank-1 subtraction from the field.

```
rlat forget <lattice> --source <source_id>
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--source` | str | **required** | Source ID to remove |

Use `rlat ls` to find source IDs.

**Example:**

```bash
rlat forget project.rlat --source "deprecated_auth_md"
```

---

#### `rlat diff`

Compute the algebraic difference between two knowledge models. The result represents what is in `lattice_a` but not in `lattice_b`. Self-diff produces near-zero energy.

```
rlat diff <lattice_a> <lattice_b> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice_a` | Yes | First `.rlat` file |
| `lattice_b` | Yes | Second `.rlat` file |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output, -o` | str | None | Output delta `.rlat` file |

**Example:**

```bash
rlat diff v2.rlat v1.rlat -o whats_new.rlat
```

---

### Export Commands

---

#### `rlat export`

Export a knowledge model. Supports field-only mode for privacy-preserving sharing.

```
rlat export <knowledge model> -o <output> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `knowledge model` | Yes | Source `.rlat` knowledge model |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output, -o` | str | **required** | Output path |
| `--field-only` | flag | off | Export field+registry without source store |

**Example:**

```bash
rlat export project.rlat -o shared.rlat --field-only
```

---

### Encoder Commands

---

#### `rlat encoders`

List all available encoder presets with their backbone model, parameter count, context window, and recommended use case.

```
rlat encoders
```

No arguments or options. See [ENCODERS.md](ENCODERS.md) for the full comparison guide.

---

### Skill Commands

The `skill` subcommand family manages skill-backed knowledge models — semantic knowledge bases built from skill reference materials and injected as adaptive context.

---

#### `rlat skill build`

Build a knowledge model from a skill's reference materials (the `references/` directory inside the skill).

```
rlat skill build <skill_name> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `skill_name` | Yes | Skill name or path to skill directory |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--encoder` | str | None | Encoder preset or HuggingFace model ID |
| `--onnx` | str | None | ONNX backbone directory |

---

#### `rlat skill sync`

Incrementally sync a skill's knowledge model with its reference materials (add new, update changed, remove deleted).

```
rlat skill sync <skill_name> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `skill_name` | Yes | Skill name or path to skill directory |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--encoder` | str | None | Encoder preset or HuggingFace model ID |
| `--onnx` | str | None | ONNX backbone directory |

---

#### `rlat skill search`

Search a skill's knowledge models with enriched results.

```
rlat skill search <skill_name> <query> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `skill_name` | Yes | Skill name or path to skill directory |
| `query` | Yes | Search query |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--top-k` | int | 10 | Number of results |
| `--format` | choice | text | Output format: `text`, `json`, `context`, `prompt` |
| `--encoder` | str | None | Encoder preset or HuggingFace model ID |
| `--source-root` | str | None | External source root |
| `--onnx` | str | None | ONNX backbone directory |

---

#### `rlat skill info`

Show skill knowledge model configuration and status. Omit the skill name to list all discovered skills.

```
rlat skill info [skill_name]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `skill_name` | No | Skill name or path (omit to list all) |

---

#### `rlat skill inject`

Four-tier adaptive context injection. Combines static skill header (Tier 1), foundational queries (Tier 2), user query (Tier 3), and optional LLM-derived queries (Tier 4) into a single injectable context block.

```
rlat skill inject <skill_name> <query> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `skill_name` | Yes | Skill name or path to skill directory |
| `query` | Yes | User query to inject context for |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | Output format: `text` (diagnostic), `json` (structured), `context` (injectable body) |
| `--derived` | str | None | Tier 4 derived queries (semicolon-separated). Omit to skip Tier 4 |
| `--encoder` | str | None | Encoder preset or HuggingFace model ID |
| `--source-root` | str | None | External source root |
| `--onnx` | str | None | ONNX backbone directory |

---

#### `rlat skill route`

Rank discovered skills by relevance to a query. Returns coverage scores and suggestions for which skill(s) to use.

```
rlat skill route <query> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `query` | Yes | Query to route |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--top-n` | int | 5 | Max skills to return |
| `--format` | choice | text | Output format: `text`, `json` |

---

#### `rlat skill profile`

Semantic profile of a skill's knowledge model. Same output as `rlat profile` but scoped to a skill.

```
rlat skill profile <skill_name> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `skill_name` | Yes | Skill name or path to skill directory |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | Output format: `text`, `json` |

---

#### `rlat skill freshness`

Check whether a skill's knowledge model is stale relative to its reference materials. Omit the skill name to check all discovered skills.

```
rlat skill freshness [skill_name] [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `skill_name` | No | Skill name or path (omit for all) |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | Output format: `text`, `json` |

---

#### `rlat skill gaps`

Detect knowledge gaps in a skill's knowledge model. Reports uncovered areas based on the skill's frontmatter queries and reference structure.

```
rlat skill gaps <skill_name> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `skill_name` | Yes | Skill name or path to skill directory |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | Output format: `text`, `json`, `prompt` |
| `--encoder` | str | None | Encoder preset or HuggingFace model ID |

---

#### `rlat skill compare`

Compare two skills' knowledge models. Reports overlap, unique coverage, and semantic distance.

```
rlat skill compare <skill_a> <skill_b> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `skill_a` | Yes | First skill name or path |
| `skill_b` | Yes | Second skill name or path |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | choice | text | Output format: `text`, `json` |

---

### Memory Commands

The `rlat memory` family manages layered conversation memory — a 3-tier system (working / episodic / semantic) where each tier is a full `.rlat` knowledge model with its own retention policy.

---

#### `rlat memory init`

Create a new layered memory root with empty tier knowledge models.

```
rlat memory init <memory_root>
```

**Example:**

```bash
rlat memory init ./memory/
# Creates: memory/working.rlat, memory/episodic.rlat, memory/semantic.rlat
```

---

#### `rlat memory write`

Write conversation transcripts or text to a memory tier.

```
rlat memory write <memory_root> [options]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input-file` | str | None | Input file path (e.g., Claude Code session transcript) |
| `--input-format` | str | None | `claude_transcript`, `conversation`, or auto-detected |
| `--session` | str | None | Session ID to tag chunks with |
| `--tier` | str | working | Target tier: `working`, `episodic`, or `semantic` |
| `--onnx` | str | None | ONNX backbone directory for faster encoding |

**Example:**

```bash
rlat memory write ./memory/ --input-file transcript.jsonl \
    --input-format claude_transcript --session s_042 --tier working
```

---

#### `rlat memory recall`

Query across all memory tiers with weighted fusion.

```
rlat memory recall <memory_root> <query> [options]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--top-k` | int | 10 | Number of results |
| `--tier-weights` | str | None | Custom tier weights (e.g., `working=0.5,episodic=0.3,semantic=0.2`) |
| `--tiers` | str | None | Subset of tiers to query (e.g., `working,episodic`) |

**Example:**

```bash
rlat memory recall ./memory/ "how do we handle authentication?" --top-k 5
```

---

#### `rlat memory primer`

Generate a conversation-memory primer for CLAUDE.md inclusion. Complements the code primer (`rlat summary`) with a second document that surfaces project axioms, decision trails, active context, and recent sessions from memory.

```
rlat memory primer <memory_root> [options]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output, -o` | str | `.claude/memory-primer.md` | Output file path |
| `--code-knowledge model` | str | None | Code knowledge model for cross-primer novelty filtering |
| `--budget` | int | 2500 | Target token budget |
| `--novelty-threshold` | float | 0.3 | Suppress passages with novelty below this vs code knowledge model |
| `--encoder` | str | None | Encoder override |
| `--onnx` | str | None | ONNX backbone directory |

**Primer sections (temporal gradient):**

| Section | Source tier | Purpose |
|---------|------------|---------|
| **Project Axioms** | semantic | Settled decisions and facts, reinforced across many sessions |
| **Decision Trail** | semantic + episodic | How key decisions evolved, with contradiction/reversal detection |
| **Active Context** | episodic | Ongoing threads from recent sessions |
| **Recent Sessions** | working | What happened in the last few conversations |

**Key features:**
- **Cross-primer novelty filtering** — suppresses passages already in the code primer (zero redundancy)
- **Contradiction detection** — marks reversed decisions, most recent wins
- **Confidence badges** — `[reinforced 5x, high confidence]` or `[reversed]`

**Examples:**

```bash
# Basic: generate from memory root
rlat memory primer ./memory/ -o .claude/memory-primer.md

# With cross-primer novelty filtering
rlat memory primer ./memory/ --code-knowledge model project.rlat

# Custom budget
rlat memory primer ./memory/ --budget 3500 --code-knowledge model project.rlat
```

Then reference from CLAUDE.md alongside the code primer:
```
@.claude/resonance-context.md
@.claude/memory-primer.md
```

---

#### `rlat memory consolidate`

Promote sources between memory tiers.

```
rlat memory consolidate <memory_root> [options]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--source-tier` | str | working | Tier to promote from |
| `--target-tier` | str | episodic | Tier to promote to |
| `--session` | str | None | Only promote sources from this session |

---

#### `rlat memory gc`

Enforce TTL and capacity limits — removes expired sources.

```
rlat memory gc <memory_root>
```

---

#### `rlat memory profile`

Show per-tier memory profile: source counts, field sizes, band energies.

```
rlat memory profile <memory_root>
```

---

#### `rlat memory stats`

Quick statistics on the memory knowledge model.

```
rlat memory stats
```

---

## Choosing the Right Command

### search vs query vs resonate

| | `search` | `query` | `resonate` |
|---|---------|---------|------------|
| **Purpose** | Full enriched retrieval | Basic ranked passages | LLM-ready context |
| **Default format** | text | text | context |
| **Coverage profile** | Yes | No | No |
| **Topic cascade** | Optional (`--cascade`) | No | No |
| **Contradictions** | Optional (`--with-contradictions`) | No | No |
| **Subgraph expansion** | Optional (`--subgraph`) | No | No |
| **Warm worker** | Yes | No | No |
| **Best for** | Interactive exploration | Scripts, debugging | Feeding an LLM |

### Which command for my use case?

| Use case | Command |
|----------|---------|
| Find information in a knowledge model | `search` |
| Generate context for an LLM | `resonate` or `search --format context` |
| Understand what a knowledge model covers | `profile` |
| Compare two versions of a codebase | `compare` |
| Check field health and signal quality | `probe health` |
| See if new content would add value | `probe novelty "content"` |
| Find conflicting information | `contradictions` or `search --with-contradictions` |
| Understand knowledge topology | `topology` or `xray --deep` |
| See where a question sits in the field | `locate` |
| List what's in the knowledge model | `ls` |
| Get knowledge model metadata | `info` |
| Build from scratch | `build` |
| Update after file changes | `sync` |
| Set up a new project quickly | `init-project` |
| Generate a code primer | `summary` |
| Generate a memory primer | `memory primer` |
| Smart query (auto-select lens) | `ask` |
| Initialize layered memory | `memory init` |
| Recall from conversation memory | `memory recall` |
| Share without source text | `export --field-only` |
| Serve over HTTP | `serve` |
| Integrate with Claude Code | `mcp` |
| Combine team knowledge | `merge` |
| Remove outdated content | `forget` |
| See what changed between versions | `diff` |
| Compose multiple knowledge models in a pipeline | `compose` |
| List available encoder presets | `encoders` |
| Build/sync a skill's knowledge model | `skill build` / `skill sync` |
| Search within a skill's knowledge | `skill search` |
| Inject adaptive context from a skill | `skill inject` |
| Route a query to the best skill | `skill route` |
| Check if skill knowledge models are stale | `skill freshness` |
| Find gaps in a skill's coverage | `skill gaps` |
