---
title: Knowledge Model Architecture
slug: knowledge-model-architecture
description: The knowledge model, its three layers, lifecycle, and compatibility rules.
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

### Field

The field stores the semantic model of the corpus. It is the fixed-size mathematical structure that supports resonance and structural diagnostics.

### Registry

The registry maps field hits back to source-level coordinates, phase vectors, and lookup structures.

### Store

The store returns passages, metadata, and evidence text. In embedded mode it lives inside the knowledge model; in external mode the knowledge model reads evidence from local files at query time.

## How Do I Use It?

### Build And Update Lifecycle

- `build` creates a new knowledge model from source material
- `add` appends new files without a full rebuild
- `sync` reconciles added, changed, and removed files
- `forget` removes a source contribution
- `export` produces a field-only or shareable derivative

### Embedded vs External Store

Use embedded mode when you want portability and self-contained sharing. Use external mode when you want smaller artifacts or do not want source text embedded.

### Encoder Contract And Compatibility

The encoder fingerprint is part of the knowledge model contract. Query and incremental update workflows are safest when they restore or validate the same encoder settings used at build time.

## Technical Guide

### Build-Time Guarantees

- the field and registry are updated deterministically from source material
- manifest tracking supports incremental sync and freshness logic
- encoder mismatch is surfaced instead of silently mixing incompatible embeddings

### Practical Implications

- knowledge model portability matters more than any single interface
- the field alone is not enough; the registry and store make evidence retrieval possible
- external integrations should treat the knowledge model as the primary runtime dependency
