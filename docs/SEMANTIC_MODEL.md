---
title: Semantic Model
slug: semantic-model
description: How retrieval works, what enriched search surfaces, and how to think about warm paths, limits, and model behavior.
nav_group: Foundations
nav_order: 40
aliases:
---

# Semantic Model

## What Is The Semantic Model?

The semantic model is the queryable representation inside a knowledge model. It is not just a vector index and it is not an LLM. It is the field-plus-registry system that lets Resonance Lattice search, inspect, compare, and compose corpora.

## Why Should I Use It?

The semantic model is what makes the product more than a search wrapper. It gives you:

- evidence-backed retrieval
- structural diagnostics
- composition and algebra over knowledge states
- a path to skill injection, memory, and programmable field operations

## How Does It Work?

Resonance Lattice is a **three-layer semantic router** (the 2026-04-19 pivot):

1. **Field routes** — a fixed-size latent tensor encodes topic and entity structure. Given a query, the field returns the handful of chunks most likely to be relevant. The field does **not** embed document content; it is a router, not an embedding store.
2. **Lossless store serves** — those chunks are resolved through a lossless store that reads the actual source file. The store is authoritative; the field is a fast router over it.
3. **Reader synthesizes** — an LLM composes the final answer from the served passages, if you want synthesis. The router works fine without one.

### Retrieval Path

At query time the question is encoded into the same space as the knowledge model. The field resonates against the query and routes it to candidate chunks; the registry resolves those hits to source coordinates; the lossless store returns the actual passages and metadata from the raw source file. Quality comes from the full pipeline (lexical injection + reranking where applicable), not from the field in isolation — the field alone under-performs flat cosine; the win is in the router-plus-store-plus-reranker stack.

### Enriched Search

The enriched search path can surface:

- ranked passages
- coverage profiles
- related topics and cascades
- contradictions
- subgraph expansion

### Warm Path

The warm path keeps the knowledge model and encoder loaded between requests. That matters for MCP workflows, repeated CLI queries, and app integrations where startup cost would otherwise dominate latency.

## How Do I Use It?

Use `search` when you want the full enriched retrieval path. Use `query` or `resonate` when you want lighter or LLM-focused output. Use `profile`, `locate`, `xray`, and `probe` when you need to inspect the semantic behavior instead of only retrieving evidence.

## Technical Guide

### What The Model Is Good At

- grounded evidence lookup across technical corpora
- semantic comparison between versions or domains
- cross-surface reuse of the same knowledge layer

### What Still Depends On Careful Evaluation

- encoder choice and prompt protocol
- benchmark fit to a specific corpus
- experimental paths such as newer memory projection ideas or fine-tuned backbones

### Mental Model

Treat the field as a **router**, not an embedding store. The raw corpus is the source of truth; the lossless store serves bytes from it. The field is a fast index over topic and entity structure that tells the store which chunks to pull — it is small, disposable, and can be re-derived from the raw corpus by re-encoding. Retrieval quality depends on the encoder contract and the query pipeline (routing + lexical injection + reranking), not on the field in isolation.

---

**Version:** 0.11.0 Beta · v1.0.0 target: 2026-06-08
