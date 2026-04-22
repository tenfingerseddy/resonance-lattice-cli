---
title: Status and Boundaries
slug: status-and-boundaries
description: Current maturity level, shipped versus experimental surfaces, and the practical limits that matter when adopting Resonance Lattice.
nav_group: Reference and Support
nav_order: 30
aliases:
---

# Status and Boundaries

## What Is The Current Status?

Resonance Lattice is currently an alpha-stage project. The core knowledge model, CLI, MCP server, HTTP server, skill surface, and benchmark tooling are real and usable, but not every research-facing path should be treated as production-ready. There is no hosted SaaS layer today.

## Why Does This Page Matter?

The repo contains both shipped functionality and experimental work. This page is the contract boundary between the two.

## How Should I Think About Stability?

### Shipped And Primary

- knowledge model build/query/profile/compare flows
- CLI and local server surfaces
- MCP tooling
- manifest, discovery, and knowledge-model-backed skill foundations
- benchmark-backed production encoder guidance

### Experimental Or Research-Weighted

- trained-head promotion paths that have not cleared benchmarks
- newer memory projection ideas
- advanced research modules that are present in the repo but not yet part of the main product contract
- planned cloud or hosted control-plane work

## How Do I Adopt It Safely?

- start with the documented production encoder path
- validate on your own corpus
- keep knowledge models and encoder settings aligned
- treat benchmark-backed behaviors differently from repo experiments

## Technical Guide

The safest adoption pattern is to treat knowledge models, the query surfaces, and the benchmark story as the stable center, then pull in advanced composition, training, or memory features only when the documentation marks them as ready.
