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

### Retrieval Path

At query time the question is encoded into the same space as the knowledge model. The field produces resonance signals, the registry identifies the strongest sources, and the store returns passages and metadata.

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

Treat the field as a structured semantic model and the store as the source of truth. Retrieval quality depends on both the encoder contract and the query pipeline, not on the field in isolation.
