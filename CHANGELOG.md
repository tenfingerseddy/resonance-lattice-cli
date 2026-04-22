# Changelog

All notable changes to Resonance Lattice. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely; dates are ISO. Dates are the day the work landed on `main`.

---

## [Unreleased] — target v1.0.0 (2026-06-08)

### Added

- **Three-mode lossless store.** Knowledge models share one `LosslessStore` abstraction with three file-read backends, chosen via `--store-mode {bundled, local, remote}`:
  - `bundled` — raw source files packed inside the `.rlat` as zstd frames (lossless, self-contained).
  - `local` (default, historical alias `external`) — thin knowledge model + `--source-root` at query time.
  - `remote` — HTTP-backed; knowledge model pins to a commit SHA on a public GitHub repo with a SHA-pinned local cache under `~/.cache/rlat/remote/`.
- **Format v3** with extended `store_mode` enum (`0=embedded`, `1=local/external`, `2=remote`, `3=bundled`). v1/v2 knowledge models keep loading unchanged.
- **`rlat build <github-url>`** — detects GitHub URLs, pins to a commit SHA at build time, fetches the tree via the GitHub API, stages into a temp directory, and runs the normal build pipeline.
- **`rlat freshness <cart>`** — read-only upstream drift check for remote-mode knowledge models. One API call; exit `0` if up-to-date, `1` if drift (CI-friendly).
- **`rlat sync <cart>`** — for remote mode, pulls the upstream diff via the GitHub compare API, applies changes via the same chunk-reconciliation pipeline `rlat refresh` uses, and atomically bumps `__remote_origin__.commit_sha`.
- **`rlat repoint <cart> --to {local,remote,bundled}`** — switch a knowledge model's storage mode without re-encoding. Uses the shared manifest schema across modes; validates path overlap before writing (zero overlap = hard fail, <80% = warning). Supports `local ↔ remote`, `local → bundled`, and `remote → bundled`. Bundled → anything requires a rebuild.
- **Three-layer retrieval** (the Semantic-Layer architecture). `rlat search` routes through the field; `rlat ask` adds adaptive expansion + optional lexical (ripgrep) hybrid pass + grounded reader synthesis with verified citations.
- **Reader layer.** `rlat ask --reader llm` (OpenVINO local or Anthropic/OpenAI API) or `rlat ask --reader context` (evidence-only, deterministic, LLM-free). Citations structurally grounded — fabricated indices dropped; quote verification via ✓/○ markers.
- **`docs/STORAGE_MODES.md`** — central reference documenting all three modes, their format layouts, and when to pick each.
- **`docs/KNOWLEDGE_MODEL_ARCHITECTURE.md`** (renamed from `CARTRIDGE_ARCHITECTURE.md`), `docs/HONEST_CLAIMS.md` (calibration document), `docs/CORE_FEATURES.md` (seven v1.0 use cases), `docs/LICENSE_FAQ.md` (plain-language BSL reference).
- **Governance files.** `SECURITY.md` (private disclosure via GitHub Security Advisories + email fallback), `CONTRIBUTING.md` (DCO via `git commit -s`), `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `NOTICE` (third-party attributions).
- **SPDX `BUSL-1.1` headers** on every `src/resonance_lattice/*.py` file (110 files).
- **`GithubFetcher`** and **`DiskCache`** in `resonance_lattice.remote` — stdlib-`urllib` client with zero new runtime dependencies, two-tier (memory + disk) cache, SHA-pinned immutable keys, atime-LRU eviction bounded by `budget_bytes` (default 500 MB).
- **`resonance_lattice.bundled.pack`** / **`BlobReader`** — zstd-framed blob store with per-file random-access decompression.
- **Tiered encoder presets** with measured 5-BEIR numbers in [docs/HONEST_CLAIMS.md](docs/HONEST_CLAIMS.md):
  - `BAAI/bge-large-en-v1.5` — default portable encoder, 0.445 5-BEIR best-mode avg (mid-tier).
  - `intfloat/e5-large-v2` — opt-in via `--encoder e5-large-v2`, 0.455 avg; wins on counter-argument retrieval (ArguAna).
  - `BAAI/bge-m3`, `Alibaba-NLP/gte-large-en-v1.5`, `jina-embeddings-v3` — additional portable encoders, measured on the same harness.
  - `Qwen/Qwen3-Embedding-8B` — opt-in via `--encoder qwen3-8b`, 0.500 avg field_only, high-end tier (needs 16 GB GPU). Required a new `"last"` token pooling mode on the encoder — previous mean/CLS pooling collapsed Qwen3 embeddings.
- **`docs/ENCODER_CHOICE.md`** — decision guide for picking the encoder per workload.

### Changed

- **Terminology: "cartridge" → "knowledge model"** across all user-facing surfaces (README, docs, CLI help, MCP tool descriptions, error messages). Classes, function names, test file names, and the `.rlat` extension stay as-is for v1.0.0; internal rename tracked for v1.1.0.
- **`ExternalStore` → `LocalStore`** with a module-level alias preserved for backward compatibility. All 21 call sites work unchanged.
- **`--store-mode` default: `embedded` → `local`** (was `external`, renamed). Deprecation message on `embedded` now points users at `--store-mode bundled` as the self-contained replacement.
- **Default encoder: `intfloat/e5-large-v2` → `BAAI/bge-large-en-v1.5`** (CLI default only; `EncoderConfig` dataclass default stays on E5 for legacy-knowledge-model back-compat). Pass `--encoder e5-large-v2` to reproduce the prior default.
- **`LosslessStore.store` / `.remove`** no longer hard no-ops: they route `__`-prefixed reserved ids (encoder config, source manifest, remote origin, profile, retrieval config) to the attached `meta_store`. Chunk-text writes still drop.
- **`Lattice.load` dispatch** generalised from `isinstance(..., ExternalStore)` to `isinstance(..., LosslessStore)` so all three modes get encoder restoration.
- **`refresh_cartridge`**'s per-file reconciliation loop extracted to `_reconcile_file_chunks`, shared with the new `sync_remote_cartridge`. Local refresh behavior is byte-for-byte identical.
- **`__all__` in `resonance_lattice/__init__.py`** is now the single source of truth for the public API. Mutations require a `CHANGELOG.md` entry. Removed 3 unused compiler chain presets (`precision_chain`, `exploration_chain`, `focused_chain`) — still importable from `resonance_lattice.compiler`, just not from the top-level package.
- **Primer system rewritten** to be self-maintaining — commit-seeded topic discovery, lens-diverse retrieval, memory amplification, per-query reranking.

### Deprecated

- **Legacy `embedded` mode** (pre-chunked SQLite `SourceStore`). Will stop working in v2.0.0. Migration path: `rlat build ... --store-mode bundled`.
- **`CartridgeEntry`, `ComposedCartridge`, `CartridgeRef`, `CartridgeFreshness`** class names are preserved for v1.0.0 compatibility but will be renamed in v1.1.0 (Phase 2 internal rename).

### Benchmarks (negative results, 2026-04-22)

- **v15 LongMemEval rebench — v14 holds.** Two orthogonal experiments on LongMemEval_S (500 instances, 800/50 chunking, adaptive routing):
  - **Q1 (LayeredMemory architecture)** — E5-large-v2 + per-instance `LayeredMemory.recall_enriched` with per-tier `enriched_query` fan-out + max-weighted fusion. Stratified 50-slice results in `benchmarks/results/longmemeval/kaggle_v15_phase1_e5/`. Three cells:
    - E5-control (cartridge baseline): R@5 0.939, MRR 0.946.
    - P1-null (all-working tier): R@5 0.957 (+0.018, sub-noise — 50-slice SE ±0.075). Neutral overall; small reordering from fusion tie-breaking.
    - P1-tier (recency policy: ≤30d→working, ≤180d→episodic, else→semantic, weights 0.5/0.3/0.2): R@5 0.935. **knowledge-update R@5 drops 0.900→0.800 (−0.100)**. Root cause: `adaptive_memory_config` already sets `prefer_recent=True` for KU queries as post-processing; the tier-weight recency signal double-counts, pushing older-but-correct answers below recency-boosted distractors.
  - **Q2 (BGE encoder flip on LME)** — BGE-large-en-v1.5 + v14's cartridge pipeline, full 500 on RunPod A100. Results in `benchmarks/results/longmemeval/runpod_v15_bge/`. Aggregate R@5 **0.916** vs v14 E5 **0.924** (**−0.008**). Three categories drop ≥0.01 vs v14: multi-session −0.017, single-session-user −0.016, temporal-reasoning −0.031. BGE wins single-session-preference (+0.067) and knowledge-update (+0.014). Pattern confirms the ArguAna-style BGE regression noted in `memory/project_arguana_bge_regression.md` — BGE hurts cross-session and counter-argument categories at full-500 scale.
  - **Publish bar both cases: FAIL.** Requires R@5 improvement ≥0.005 AND no category drop ≥0.01. Q1 shows no positive signal and a category cliff; Q2 aggregates negatively and has three category cliffs. **v14 (E5-large-v2, cartridge, R@5 0.9244) remains the shipped LongMemEval baseline.**
  - **`LayeredMemory.recall_enriched`** ships as API (fixes the pure-dense gap in `recall`/`recall_text`); it is the correct primitive for memory-architecture experiments in the future but does not improve the LME benchmark. Unit test coverage in `tests/test_layered_memory.py` (3 fusion tests).

### License

- **Business Source License 1.1** with broad permitted use for internal development, CI/CD, coding assistants, and downstream products. Each release converts to MPL 2.0 four years after its first public release (per-version dynamic formula — no manual re-anchoring). See [LICENSE.md](LICENSE.md) and [docs/LICENSE_FAQ.md](docs/LICENSE_FAQ.md).

---

## [0.11.0] — 2026-04-15 — EML corpus intelligence

### Added

- **EML corpus transforms.** Nonlinear spectral transforms that reshape the whole field before search, unsupervised: `--sharpen`, `--soften`, `--contrast`, `--tune {focus,explore}` presets.
- **`MultiVectorField`** — per-source vector sets with soft-MaxSim retrieval (research lane; not default).
- **Contextual chunking — "natural" format** as the default (+1.4% nDCG@10 over prior chunker defaults).
- **A3 / A4 trainable circuits** research code + ternary cleanup in `rql/eml_circuits.py`.

### Changed

- **EML Phase A:** linear-calibrated emit fix + reranker compatibility.
- **EML Stage 2:** registry per-band nonlinear scoring + SciFact benchmark coverage.
- **Quality strategy** updated — closed BGE-M3 Tier 4 (SPLADE via multi-output retired after failing to beat dense+rerank).

---

## [0.10.1] — 2026-04-15 — `--compact` build flag

### Added

- **`rlat build --compact`** — produces 16× smaller knowledge models by trimming non-essential metadata from the `.rlat` payload while preserving the field + registry.

### Changed

- **IDF reranking** updated in `bench_beir_pipeline.py` methodology.
- **Asymmetric field v2** results recorded in `benchmarks/results/beir/` (asymmetric lane closed — did not beat symmetric).

---

## [0.10.0] — 2026-04-15 — threshold default + encoder benchmarks

### Added

- **Threshold sparsification** as a default for key-head encoding — FLOPS-regularised, sparsity-aware.
- **LongMemEval benchmark** harness in `benchmarks/` — R@5 / MRR across question categories.
- **Asymmetric key/value field** (star-schema separation for matching vs retrieval) — research implementation; ultimately retired after BEIR regression analysis.
- **Knowledge Physics** features across CLI, MCP, and skill integration — algebra-powered operations (merge, forget, diff, project, contradict).
- **Obsidian plugin** explorer view, graph visualization, and advanced search controls.
- **Lazy imports in `__init__.py`** — 15× faster warm CLI queries for the primary search path.
- **Website docs overhaul** — serves from repo markdown, removes static duplicates.
- **Batch build pipeline** covering encoding, field ops, and SQLite commits.

### Changed

- **README restructured** around the reader journey.
- **MCP startup** simplified — ONNX attach moved into deferred loader.
- **Primer Tier 3** quality improvements: noise filtering, README promotion, echo beats README.

---

## [0.9.0] — 2026-04-12 — Public PyPI-ready release

Initial PyPI-targeted release. Documents, benchmarks, Obsidian plugin, Claude Code skill, knowledge physics, and the cartridge-algebra composition model. See the tag commit for the full state snapshot.

---

## Prior history

Pre-0.9.0 commits live in `git log` — phase-numbered branches covered the encoder training pipeline, RAG quality evaluation, PQ batch superpose, LLM judge harness, and the nature-inspired correlated-activation experiments. That era's milestones are captured in `docs/direction/PROJECT_HISTORY.md`.
