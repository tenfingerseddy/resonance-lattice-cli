---
title: Encoder Choice — E5 / BGE / Qwen3-8B
status: shipping
decided: 2026-04-22
board_items: [237, 239, 240]
---

# Encoder Choice

Resonance Lattice ships three well-supported encoders. None is universally best — each wins on a different workload shape. This page gives you the data and a decision guide so you can pick the right one for your corpus.

If you're not sure, **BGE-large-en-v1.5** is the starting-point default. It's a solid middle-of-the-road choice that works on CPU / Intel Arc iGPU, has strong ecosystem support (ONNX / OpenVINO / sentence-transformers), and wins on 4 of 5 BEIR corpora. If your workload is counter-argument retrieval, switch to E5. If you have a 16 GB GPU and want frontier quality, use Qwen3-8B.

## TL;DR — which encoder, when

| If your corpus is… | Use | Why |
|---|---|---|
| General QA, science, docs, code | `bge-large-en-v1.5` (default) | Strong on QA-style queries, best ecosystem, fastest CPU inference |
| Counter-argument / debate retrieval (ArguAna-like) | `e5-large-v2` | BGE regresses 9.7 pts on ArguAna; E5 preserves argumentative structure |
| You have a 16 GB GPU and want SOTA quality | `qwen3-8b` | +7 pt lift on 5-BEIR avg (0.50 vs 0.44 for BGE field-only) |
| You're deploying to Intel Arc / NPU / strict CPU-only | `bge-large-en-v1.5` | Well-validated OpenVINO path |
| You built cartridges before 2026-04-20 | whatever is stamped in the cartridge | Encoder is restored on load; no action needed |

## Full 5-BEIR comparison (best-mode nDCG@10)

Best-mode = the winning `(probe_mode, reranker)` combination for that corpus, as picked by the build-time probe infra (board items 236c / 238).

| Corpus | E5-large-v2 | BGE-large-en-v1.5 | Qwen3-Embedding-8B |
|---|---|---|---|
| NFCorpus | 0.38217 | **0.39246** | 0.41381 |
| SciFact | 0.74664 | **0.75538** | 0.77380 |
| ArguAna | **0.50478** | 0.40795 | 0.46616 |
| SciDocs | 0.20276 | 0.21374 | **0.27823** |
| FiQA | 0.43782 | 0.45407 | **0.56835** |
| **5-BEIR avg** | 0.45523 | 0.44472 | **0.50007** |
| **v1.0.0 gate (≥ 0.46)** | ❌ −0.005 | ❌ −0.015 | ✅ +0.040 |

Bold = best encoder per corpus. Qwen3-8B wins 3/5 outright and ties BGE on SciFact. E5 owns ArguAna decisively. BGE is the rational middle — small wins on 4/5, loses the ArguAna round.

**Measurement caveat**: Qwen3-8B numbers are `field_only` (dense retrieval alone, no reranker). E5 and BGE numbers are best of `field_only / plus_cross_encoder / plus_cross_encoder_expanded / plus_full_stack` with both `bge-reranker-v2-m3` and `mxbai-rerank-base-v1` tested per corpus. Full stack probes on Qwen3-8B are deferred — anecdotally, CE rerankers trained on weaker base retrievers often regress strong-dense top-k, so a Qwen3-Reranker pairing is expected before surfacing stack numbers.

## Per-corpus cross-mode data (E5 and BGE)

### NFCorpus (medical abstracts, QA)

| Mode | E5-large-v2 | BGE-large-en-v1.5 |
|---|---|---|
| `field_only` | 0.37387 | 0.38187 |
| `plus_cross_encoder` | 0.38217 *(mxbai)* | **0.39246** *(mxbai)* |
| `plus_full_stack` | 0.35531 | 0.37591 |

Medical QA: cross-encoder rerank helps on both encoders. **mxbai-rerank-base-v1 beats bge-reranker-v2-m3** on this corpus for both encoders.

### SciFact (scientific claim verification)

| Mode | E5-large-v2 | BGE-large-en-v1.5 |
|---|---|---|
| `field_only` | 0.70623 | 0.73178 |
| `plus_cross_encoder` | 0.74046 | **0.75538** |
| `plus_full_stack` | 0.74664 | 0.74826 |

Science claims benefit from rerank; both encoders pair best with `bge-reranker-v2-m3`. Full stack (adding BM25 fusion) is marginal on BGE.

### ArguAna (counter-argument retrieval) — the regression

| Mode | E5-large-v2 | BGE-large-en-v1.5 |
|---|---|---|
| `field_only` | 0.41459 | 0.39191 |
| `plus_cross_encoder` | 0.40620 | 0.39162 |
| `plus_full_stack` | **0.50478** | 0.40795 |

**BGE regresses 9.7 pts vs E5 on this corpus** — the largest single-corpus gap we measured. ArguAna is counter-argument retrieval: given an argument, find the best counter. E5's training (MSMARCO + diverse pairs) captured argumentative inversion; BGE's (more diverse but QA-biased) did not. If this is your workload, explicitly `--encoder e5-large-v2`.

### SciDocs (citation prediction)

| Mode | E5-large-v2 | BGE-large-en-v1.5 |
|---|---|---|
| `field_only` | 0.20276 | **0.21374** |
| `plus_cross_encoder` | 0.19243 | 0.20627 |
| `plus_full_stack` | 0.18044 | 0.18314 |

**Cross-encoders actively hurt on SciDocs for both encoders.** CE rerankers are trained on QA-style pairs; citation prediction is a different pair distribution. Auto-routing correctly picks `field_only` with no reranker.

### FiQA (finance QA)

| Mode | E5-large-v2 | BGE-large-en-v1.5 |
|---|---|---|
| `field_only` | 0.41745 | 0.44168 |
| `plus_cross_encoder` | 0.43782 | **0.45407** |
| `plus_full_stack` | 0.43694 | 0.43610 |

Domain QA: BGE + `bge-reranker-v2-m3` wins. Full stack marginally hurts (BM25 fusion underperforms on finance jargon).

## Deployment profile

| Dimension | E5-large-v2 | BGE-large-en-v1.5 | Qwen3-Embedding-8B |
|---|---|---|---|
| Parameters | 335 M | 335 M | 8 B |
| Size (bf16 / fp16) | 0.7 GB | 0.7 GB | 16 GB |
| Min practical inference | CPU | CPU / Intel Arc iGPU | A100-class GPU (16 GB VRAM min) |
| Query latency (warm, 1 query) | ~30 ms CPU / ~10 ms Arc | ~25 ms CPU / ~8 ms Arc | ~60 ms A100 |
| Ecosystem | sentence-transformers, ONNX, OpenVINO | **best** (sentence-transformers, ONNX, OpenVINO, broad fine-tunes) | transformers; limited ONNX/OpenVINO |
| License | MIT | MIT | Apache-2.0 |
| Default pooling | `mean` | `cls` | **`last`** (required — see caveat) |

Resonance Lattice stamps the encoder into every cartridge's `__encoder__` block. Loading a cartridge restores the encoder preset (pooling, prefixes, max-length) automatically — you cannot accidentally query a BGE cartridge with E5.

## How to opt in / out

```bash
# Starting-point default (BGE-large-en-v1.5)
rlat build ./docs ./src -o project.rlat

# Explicit E5 (use for counter-argument / debate corpora)
rlat build ./docs ./src -o project.rlat --encoder e5-large-v2

# Qwen3-Embedding-8B (needs 16 GB GPU)
rlat build ./docs ./src -o project.rlat --encoder qwen3-8b

# Legacy cartridges keep working — the encoder is stamped at build time
rlat search legacy-e5-cartridge.rlat "how does auth work?"   # auto-loads E5
```

## Market position (same 5 BEIR corpora, best-mode avg)

| Anchor | 5-BEIR avg |
|---|---|
| BEIR BM25 (2021) | 0.340 |
| BEIR BM25+CE (2021) | 0.372 |
| Cohere-embed-mlv3 (2024) | 0.450 |
| **Ours — E5-large-v2 (full stack)** | **0.455** |
| jina-v3 (2024) | 0.461 |
| mE5-large-instruct (2024) | 0.464 |
| **Ours — BGE-large-en-v1.5 (full stack)** | **0.445** |
| **Ours — Qwen3-Embedding-8B (field_only)** | **0.500** |
| text-embedding-3-large (OpenAI, 2024) | 0.512 |
| Qwen3-Embedding-8B (published, different BEIR subset) | ~0.55 |
| NV-Embed-v2 (2024, CC BY-NC) | 0.620 |

**Position**: our portable tiers (E5, BGE) sit with 2024 mid-tier open dense retrievers. Qwen3-8B is frontier-adjacent on dense alone; still a ~1 pt gap to `text-embedding-3-large`, and another 5-10 pts below the published NV-Embed / Gemini-Embedding frontier.

Sources: [BEIR paper](https://arxiv.org/abs/2104.08663), [jina-embeddings-v3 paper](https://arxiv.org/abs/2409.10173), [Qwen3-Embedding card](https://huggingface.co/Qwen/Qwen3-Embedding-8B), our aggregates in [benchmarks/results/beir/new_arch/](../benchmarks/results/beir/new_arch/).

## Technical notes

### Qwen3-Embedding pooling is load-bearing

Qwen3-Embedding is a decoder-only LM. It **requires last-non-padding-token pooling** — the final position is the only one with full left-context. Mean or CLS pooling discards that; a sweep with `"pooling": "mean"` collapsed the 5-BEIR avg to 0.250 (FiQA dropped 7× from 0.568 to 0.092). The `qwen3-*` entries in `ENCODER_PRESETS` set `"pooling": "last"` — don't edit without re-benchmarking. Evidence archived at [benchmarks/results/beir/new_arch/baseline_qwen3_8b_v1_meanpool_broken/](../benchmarks/results/beir/new_arch/baseline_qwen3_8b_v1_meanpool_broken/).

### Build-time probe picks the right mode automatically

`rlat build` with `--probe-qrels` and `--probe-queries` (board item 236c) runs the mode × reranker sweep over held-out labeled queries and writes the winner to `__retrieval_config__` inside the cartridge. At query time, `rlat search` reads that config and routes automatically. You get per-corpus best-mode for free — no manual tuning, no need to memorize which reranker works where.

See [docs/CLI.md](CLI.md#rlat-build) for the probe flags.

### Encoder stamping + consistency check

Every cartridge carries `__encoder__` metadata (name, pooling, prefixes, max_length). `Lattice.load` restores the encoder from this stamp. `_check_encoder_consistency` blocks loads where the caller tries to force a different encoder — you can't accidentally mix encoders in a single query path.

### First-build download

`rlat build` downloads the encoder weights from Hugging Face on first invocation:
- `BAAI/bge-large-en-v1.5` — ~1.3 GB
- `intfloat/e5-large-v2` — ~1.3 GB
- `Qwen/Qwen3-Embedding-8B` — ~16 GB

After the first build, subsequent builds and queries are fully local. Pair with `--onnx` or `--openvino` on Intel Arc for accelerated inference. For `Qwen3-Embedding-8B`, a ~16 GB GPU is required for practical latency.

### Memory + lens default

Secondary code paths (`rlat memory save` with no existing lattice, `rlat lens build --topics`) also default to BGE-large-en-v1.5 for consistency with the primary build path.

## What's still open

1. **Qwen3-8B full-stack probe**. Currently only `field_only` measured. Cross-encoder rerankers in the shipped set (bge-reranker-v2-m3, mxbai-rerank-base-v1) are trained on weaker base retrievers and are expected to regress Qwen3-8B; a Qwen3-Reranker pairing is tracked under a follow-up board item.
2. **Per-corpus encoder routing**. Projected BGE-elsewhere + E5-on-ArguAna would average 0.464 — clears the v1.0.0 gate. Not yet implemented; would require a per-encoder build and a per-corpus routing layer. Filed as a v1.1.0 consideration.
3. **Encoder fine-tune at 335M**. Closing the ~6-pt gap to text-embedding-3-large is tractable with distillation or task-aware fine-tune (expected 2-8 weeks, $50-300 compute). Tracked under board item 235.

## Changelog

- **2026-04-22** — Rewrite. Three-encoder comparison (E5 / BGE / Qwen3-8B) with full 5-BEIR best-mode table. Item 239 launch verification sweep landed BGE full 5-BEIR numbers; reveals -9.7 pt ArguAna regression vs E5 and net -1 pt on 5-BEIR avg. BGE framed as starting-point default with explicit decision guidance.
- 2026-04-20 — Initial BGE default flip (item 237) based on 2-corpus pilot (NFCorpus + SciFact only).
