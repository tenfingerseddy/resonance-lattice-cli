# Resonance Lattice CLI Reference

The `rlat` command-line interface is the primary way to build, query, inspect, and manage Resonance Lattice cartridges. This document is the single source of truth for all 25 CLI commands.

For related documentation see [TECHNICAL.md](TECHNICAL.md) (internals), [RQL_REFERENCE.md](RQL_REFERENCE.md) (programmable query language), and the [HTTP API reference](../website/src/content/docs/api-reference.md).

**Version**: 0.9.0

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
    - [Primary](#primary-commands): search, profile, compare, ls, info
    - [Build](#build-commands): build, add, sync, init, ingest, init-project
    - [Serve and Context](#serve-and-context-commands): serve, summary, mcp
    - [Query and Analysis](#query-and-analysis-commands): query, resonate, contradictions, topology, xray, locate, probe
    - [Algebra](#algebra-commands): merge, forget, diff
    - [Export](#export-commands): export
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

### 1. Build a cartridge from your project

```bash
rlat build ./docs ./src -o project.rlat
```

This encodes all supported files into a single `.rlat` cartridge containing a semantic field, source registry, and evidence store.

### 2. Search it

```bash
rlat search project.rlat "how does authentication work?"
```

Returns ranked passages with scores, source file paths, coverage profile, and topic clustering.

### 3. Inspect the model

```bash
rlat profile project.rlat
```

Shows the cartridge's semantic shape: per-band energy distribution, source count, effective rank, and coverage summary.

### 4. Generate an assistant primer

```bash
rlat summary project.rlat -o .claude/resonance-context.md
```

Produces a compressed context document suitable for inclusion in CLAUDE.md or other assistant system prompts. Bootstraps itself by running internal queries to sample the cartridge's knowledge.

---

## Key Concepts

### Cartridges (.rlat files)

A cartridge is a single `.rlat` file that packages three layers into one portable artifact:

| Layer | Contents | Scaling |
|-------|----------|---------|
| **Field** | Semantic tensor (B x D x D) | Fixed-size |
| **Registry** | Source coordinates, LSH buckets, phase vectors | Scales with source count |
| **Store** | Evidence text, metadata, chunk content | Scales with corpus size |

Cartridges are self-contained. You can copy, share, merge, diff, and query them without access to the original source files (unless built with `--store-mode external`).

### Three Layers

- **Field**: The latent semantic model. A multi-band interference tensor where each band captures a different abstraction level. Fixed-size regardless of source count.
- **Registry**: Maps resonance hits back to source coordinates. Stores phase vectors and LSH buckets for fast lookup.
- **Store**: Holds the actual evidence text and file metadata. Backed by SQLite when embedded in the cartridge.

### Encoder

The encoder converts text into phase vectors that the field can store and query. The default and promoted encoder is `intfloat/e5-large-v2` with random projection heads.

Trained-head checkpoints exist experimentally but are **not recommended** — they failed the retrieval gate (dense R@5 dropped from 0.88 to 0.05 due to build/query parity violation). The `--checkpoint` flag remains available for research use only.

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

Best for general-purpose assistants that should use their own knowledge alongside the cartridge evidence.

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

1. **`--encoder`** flag — uses the specified backbone (`e5-large-v2`, `random`, or a HuggingFace model name)
2. **Stored encoder** — restores the encoder saved inside the cartridge at build time
3. **Default** — downloads `intfloat/e5-large-v2` on first use

In practice, most users never need `--encoder`. The cartridge stores its encoder at build time, and queries restore it automatically. Use `--encoder random` for fast testing without downloading a model.

> The `--checkpoint` flag exists for experimental trained-head loading but is **not recommended** for production use. See [ENCODER_QUALITY.md](ENCODER_QUALITY.md) for details.

### ONNX Acceleration

For faster CPU encoding, export the backbone to ONNX format. The CLI auto-detects an ONNX directory alongside the cartridge:

- `project_onnx/` (same stem as cartridge)
- `onnx_backbone/` (generic name)

Or specify explicitly:

```bash
rlat search project.rlat "query" --onnx ./onnx_backbone/
```

ONNX provides 2-5x CPU encoding speedup.

### Encoder Consistency

Cartridges track the encoder fingerprint (backbone name, bands, dim). When using `add` or `sync`, a mismatch between the current encoder and the cartridge's encoder triggers a warning. This prevents mixing incompatible embeddings in one cartridge.

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

- **`add`** — appends new files to an existing cartridge. Skips files already present (by content hash).
- **`sync`** — bidirectional sync with source directories. Adds new files, re-encodes modified files, and forgets deleted files.

Both commands use manifest tracking to detect changes.

### Manifest Tracking

The build system maintains a `__manifest__` entry inside each cartridge. For every ingested file, it stores:

- **Content hash** (MD5) for change detection
- **Chunk IDs** produced, enabling exact removal via `forget`
- **Encoder fingerprint** to enforce consistency

Legacy cartridges built before manifest support will print a warning on first `add`/`sync` and begin tracking from that point.

### Store Modes

| Mode | Flag | Behavior |
|------|------|----------|
| **embedded** (default) | `--store-mode embedded` | Full store (text + metadata) packaged inside the `.rlat` file |
| **external** | `--store-mode external` | Only field + registry in the `.rlat` file. At query time, pass `--source-root` to resolve content from local files |

External mode is useful for:
- Reducing cartridge size
- Avoiding embedded PII
- Sharing the semantic model without source text

### Registry Quantization

Compress registry phase vectors at build time:

| Flag | Compression | Use case |
|------|------------|----------|
| `--quantize-registry 0` | None (default) | Full precision |
| `--quantize-registry 8` | ~50% reduction | Recommended for most use cases, higher quality |
| `--quantize-registry 4` | ~87% reduction | Aggressive, slight quality loss |

Quantization is data-oblivious (no codebook training needed).

### Compression

The serialized cartridge can be compressed:

| Flag | Notes |
|------|-------|
| `--compression none` | Default. No compression. |
| `--compression zstd` | Best ratio. Requires `zstd` library. |
| `--compression lz4` | Fastest decompression. |

---

## Warm Worker System

The `search` command can spawn a background HTTP worker that keeps the cartridge loaded in memory. This eliminates load time on subsequent queries.

### How It Works

1. First `rlat search` runs the query in-process (cold path, includes load time).
2. After returning results, it spawns a background worker process with the cartridge pre-loaded.
3. Subsequent `rlat search` calls on the same cartridge hit the warm worker (near-zero load time).

The worker key includes cartridge path, modification time, encoder, and source root. A changed cartridge automatically invalidates the worker.

### Cold vs Warm

| Metric | Cold | Warm |
|--------|------|------|
| Load time | 50-500ms (depends on cartridge size) | ~0ms |
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

Auto-detects `docs/`, `src/`, `lib/`, `README.md`, `CLAUDE.md`, and `AGENTS.md` in the current directory. Builds a cartridge, generates a summary, and prints integration hints. Equivalent to:

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

### Comparing Cartridge Versions

```bash
rlat compare old.rlat new.rlat
```

Shows overlap, unique coverage per cartridge, and per-band energy differences.

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

The exported cartridge contains field + registry but no evidence store.

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
| `--checkpoint` | str | None | Experimental trained heads checkpoint (not recommended) |
| `--source-root` | str | None | External source root for file-backed store |
| `--onnx` | str | None | ONNX backbone directory (auto-detects `<stem>_onnx/`) |

**Performance options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--no-worker` | flag | off | Disable background warm worker (env: `RLAT_NO_WORKER=1`) |

**Injection options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mode` | choice | None | Injection framing: `augment`, `constrain`, `knowledge`, `custom` |
| `--custom-prompt` | str | None | Custom system prompt (requires `--mode custom`) |

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
```

---

#### `rlat profile`

Semantic profile of a cartridge. Shows per-band energy distribution, source count, effective rank, coverage patterns, and overall model health.

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

Compare two cartridges. Reports overlap, unique coverage per cartridge, per-band energy differences, and structural similarity.

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

List sources in a cartridge. Shows source IDs, file paths, and optional summaries.

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

Build a cartridge from source files. Discovers files recursively, chunks them, encodes phase vectors, and superposes them into the field.

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
| `--checkpoint` | str | None | Experimental trained heads checkpoint (not recommended) |
| `--quantize-registry` | int | 0 | Quantize registry phases (0=off, 8=~50% compression, 4=~87% compression) |
| `--store-mode` | choice | embedded | `embedded`: full store in `.rlat`. `external`: field+registry only |

**Examples:**

```bash
# Standard build
rlat build ./docs ./src -o project.rlat

# Compressed build with quantized registry
rlat build ./docs -o project.rlat --compression zstd --quantize-registry 8

# External store mode (no embedded text)
rlat build ./docs -o project.rlat --store-mode external
```

---

#### `rlat add`

Incrementally add files to an existing cartridge. Skips files already present (by content hash). Uses manifest tracking to avoid duplicates.

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
| `--checkpoint` | str | None | Experimental trained heads checkpoint (not recommended) |

**Example:**

```bash
rlat add project.rlat ./new_docs
```

---

#### `rlat sync`

Sync a cartridge with source directories. Detects new, modified, and deleted files. Adds new files, re-encodes changed files, and forgets removed files.

```
rlat sync <lattice> <inputs...> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `lattice` | Yes | Path to `.rlat` file |
| `inputs` | Yes | Source directories to sync from |

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--encoder` | str | None | Encoder override |
| `--checkpoint` | str | None | Experimental trained heads checkpoint (not recommended) |

**Example:**

```bash
rlat sync project.rlat ./docs ./src
```

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
| `--checkpoint` | str | None | Experimental trained heads checkpoint (not recommended) |

**Example:**

```bash
rlat ingest project.rlat ./new_document.md
```

---

#### `rlat init-project`

One-command project setup. Auto-detects source directories, builds a cartridge, generates a summary, and prints integration instructions.

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
| `--checkpoint` | str | None | Experimental trained heads checkpoint (not recommended) |

**Examples:**

```bash
# Auto-detect everything
rlat init-project

# Specify sources
rlat init-project ./docs ./src --encoder random
```

---

### Serve and Context Commands

---

#### `rlat serve`

Start an HTTP server exposing the cartridge for network queries. Endpoints: `GET /health`, `GET /info`, `POST /query`, `POST /search`, `POST /add`, `POST /remove`.

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
| `--checkpoint` | str | None | Experimental trained heads checkpoint (not recommended) |

**Example:**

```bash
rlat serve project.rlat --port 9090
```

---

#### `rlat summary`

Generate a pre-injection context primer. Bootstraps by running internal queries to sample the cartridge's knowledge, then produces a structured summary.

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
| `--checkpoint` | str | None | Experimental trained heads checkpoint (not recommended) |
| `--source-root` | str | None | External source root for file-backed store |
| `--budget` | int | 2500 | Target token budget for output |

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

Start an MCP (Model Context Protocol) server over stdio transport. Enables Claude Code and other MCP-compatible tools to query the cartridge directly.

```
rlat mcp <cartridge> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `cartridge` | Yes | Path to `.rlat` cartridge |

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
| `--checkpoint` | str | None | Experimental trained heads checkpoint (not recommended) |
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
| `--checkpoint` | str | None | Experimental trained heads checkpoint (not recommended) |
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
| `--checkpoint` | str | None | Experimental trained heads checkpoint (not recommended) |

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
| `--checkpoint` | str | None | Experimental trained heads checkpoint (not recommended) |

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
| `--checkpoint` | str | None | Experimental trained heads checkpoint (not recommended) |

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

Merge two cartridges into one. Combines fields, registries, and stores. Merge is commutative: `merge(A, B) = merge(B, A)`.

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

Remove a source from a cartridge. Performs algebraically exact rank-1 subtraction from the field.

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

Compute the algebraic difference between two cartridges. The result represents what is in `lattice_a` but not in `lattice_b`. Self-diff produces near-zero energy.

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

Export a cartridge. Supports field-only mode for privacy-preserving sharing.

```
rlat export <cartridge> -o <output> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `cartridge` | Yes | Source `.rlat` cartridge |

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
| Find information in a cartridge | `search` |
| Generate context for an LLM | `resonate` or `search --format context` |
| Understand what a cartridge covers | `profile` |
| Compare two versions of a codebase | `compare` |
| Check field health and signal quality | `probe health` |
| See if new content would add value | `probe novelty "content"` |
| Find conflicting information | `contradictions` or `search --with-contradictions` |
| Understand knowledge topology | `topology` or `xray --deep` |
| See where a question sits in the field | `locate` |
| List what's in the cartridge | `ls` |
| Get cartridge metadata | `info` |
| Build from scratch | `build` |
| Update after file changes | `sync` |
| Set up a new project quickly | `init-project` |
| Generate an assistant primer | `summary` |
| Share without source text | `export --field-only` |
| Serve over HTTP | `serve` |
| Integrate with Claude Code | `mcp` |
| Combine team knowledge | `merge` |
| Remove outdated content | `forget` |
| See what changed between versions | `diff` |
