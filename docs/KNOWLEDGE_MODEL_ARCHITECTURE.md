---
title: Knowledge Model Architecture
slug: knowledge-model-architecture
description: The knowledge model, the three-layer semantic router, storage modes, lifecycle, and compatibility rules.
nav_group: Foundations
nav_order: 30
aliases:
  - technical
---

# Knowledge Model Architecture

## What Is A Knowledge Model?

A knowledge model is the portable unit of knowledge in Resonance Lattice. It is a `.rlat` file that packages the semantic model and the lookup material required to turn a query back into evidence.

## Why Should I Use It?

The knowledge model gives you a stable artifact that can be moved across tools and machines without rebuilding a retrieval service. It also makes versioning and comparison practical because the knowledge layer is materialized as a file.

## How Does It Work?

Resonance Lattice is a **three-layer semantic router** (the 2026-04-19 pivot):

1. **Field routes** — a fixed-size latent tensor (~80 MB) models semantic structure. Given a query, the field returns the handful of chunks most likely to be relevant. The field does **not** embed document content; it encodes topic and entity structure for routing.
2. **Lossless store serves** — those chunks are resolved through a lossless store that reads the actual source file. The store is authoritative; the field is a fast router over it.
3. **Reader synthesizes** — an LLM (Claude, etc.) composes the final answer from the served passages, if you want synthesis. The router works fine without one.

### Field

The field is a latent tensor that routes queries to candidate chunks. It is a fixed-size mathematical structure that supports resonance and structural diagnostics — profile, xray, locate, compare, and the full RQL algebra all operate on it. The field is a **router**, not an embedding store: the raw corpus remains the source of truth.

### Registry

The registry maps field hits back to source-level coordinates, phase vectors, and lookup structures.

### Store

The store returns passages, metadata, and evidence text. It comes in three serving topologies — one abstraction, three backends — so the same `.rlat` format supports self-contained artifacts, disk-backed working copies, and SHA-pinned upstream mirrors.

## How Do I Use It?

### Build And Update Lifecycle

- `build` creates a new knowledge model from source material
- `add` appends new files without a full rebuild
- `sync` reconciles added, changed, and removed files (also: `rlat sync` for remote mode pulls upstream diff)
- `refresh` re-indexes drifted files in local mode, preserving the field tensor where chunk hashes still match
- `freshness` checks upstream drift for a remote-mode knowledge model (read-only)
- `repoint` switches a knowledge model's storage mode without re-encoding (local ↔ remote ↔ bundled)
- `forget` removes a source contribution
- `export` produces a field-only or shareable derivative

### Storage Modes

The canonical CLI flag is `--store-mode {bundled,local,remote}`:

| Mode | What's in the `.rlat` | Pick when |
|------|------------------------|-----------|
| **`local`** (default) | Field + registry; source files resolved from disk via `--source-root` at query time | Developing against a working copy; large corpora where you don't want to bundle source |
| **`bundled`** | Field + registry + raw source files packed inside (zstd frames) | Shipping a self-contained artifact — HF Hub, CI, offline, release bundles with provenance |
| **`remote`** | Field + registry + HTTP pointer to a SHA-pinned upstream GitHub repo + local cache | Pointing at an upstream repo you don't own; query/load never touches the network |

`external` is the historical name for `local` and is still accepted as a synonym. Legacy **`embedded`** mode (pre-chunked SQLite store) is **deprecated** and will be removed in v2.0.0. It is *not* the same as `bundled` — `bundled` stores whole files and preserves every retrieval feature, while `embedded` stored pre-chunked text and lost the whole-file view. Rebuild old `embedded` knowledge models with `--store-mode bundled` (if you want self-contained) or `--store-mode local` (thinner model, source files on disk).

See [Storage Modes](/docs/storage-modes) for format details, the freshness model, and the full bundled-vs-embedded comparison.

### Encoder Contract And Compatibility

The encoder fingerprint is part of the knowledge model contract. The default encoder is `BAAI/bge-large-en-v1.5` (flipped from E5-large-v2 on 2026-04-20). The encoder is stamped at build time; querying automatically restores the same preset (pooling, prefixes, max-length), so you cannot accidentally query a BGE knowledge model with E5. Query and incremental-update workflows validate the encoder fingerprint and surface mismatches rather than silently mixing incompatible embeddings.

All three well-supported encoders (BGE-large-en-v1.5, E5-large-v2, Qwen3-Embedding-8B) use pretrained weights with random projection heads. Trained heads were tested and rejected because they broke build/query parity. See [Encoder Choice](/docs/encoder-choice) for the per-workload decision guide.

## Technical Guide

### Build-Time Guarantees

- the field and registry are updated deterministically from source material
- manifest tracking supports incremental sync, refresh, and freshness logic across all three store modes
- encoder mismatch is surfaced instead of silently mixing incompatible embeddings
- because the three modes share a manifest schema (posix-relative `source_file` per chunk), you can flip between them via `rlat repoint` without rebuilding the field or re-encoding any text

### Practical Implications

- knowledge model portability matters more than any single interface
- the field alone is not enough; the registry and lossless store are what make evidence retrieval possible — the field is a fast router, the store is the source of truth
- external integrations should treat the knowledge model as the primary runtime dependency

---

**Version:** 0.11.0 Beta · v1.0.0 target: 2026-06-08
