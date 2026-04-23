# SPDX-License-Identifier: BUSL-1.1
"""Lexical band — BM25 posting list folded into the field (WS3 #291).

Builds a single extra phase-vector band per source from BM25-weighted
term vectors, feature-hashed to the semantic band's dimensionality D.
The band plugs straight into the existing `DenseField.superpose` /
`Lattice.resonate` paths — no new resonance code, no new field backend.

Why hashing, not a vocabulary sidecar:
  * corpus-local: nothing to persist alongside the cartridge except the
    seed used at build time.
  * composition-aligned: same seed across cartridges means `--with` on
    lexical bands lines up without a shared-vocab contract.

Why BM25 weights at build, unit weights at query:
  * queries are short; IDF weighting on a 3-5 token query is noise.
  * the retrieval signal we want is "does the doc contain the query's
    terms at all?"; unit-weighted query vectors hit the hashed positions
    the doc's BM25 accumulator also touches.

Tokenisation mirrors `retrieval/lexical.py` verbatim so the flag-gated
rollout (`--lexical-impl {band,rerank}`) compares apples-to-apples.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Fixed hash seed. Baked in so every cartridge hashes the same tokens to
# the same dimensions — which is what lets `merge(A, B)` align lexical
# bands across cartridges without carrying a vocab.
DEFAULT_SEED: int = 0xB10B

# BM25 Okapi defaults (Robertson et al. 1994; same as bm25s defaults).
DEFAULT_K1: float = 1.5
DEFAULT_B: float = 0.75

# Default blend weight for the lexical band (mirrors retrieval.lexical.DEFAULT_WEIGHT
# so the --hybrid auto behaviour matches today's ripgrep-backed pass).
DEFAULT_WEIGHT: float = 0.3

# Mirrored from retrieval/lexical.py for build/query parity during the
# flag-gated rollout. Any drift here silently breaks the parity gate.
_MIN_TOKEN_LEN = 3
_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "had", "her", "was", "one", "our", "out", "day", "get", "has", "him",
    "his", "how", "man", "new", "now", "old", "see", "two", "way", "who",
    "boy", "did", "its", "let", "put", "say", "she", "too", "use", "with",
    "this", "that", "from", "what", "when", "where", "which", "there",
    "been", "have", "they", "were", "will", "your", "about", "would",
})
_TOKEN_RE = re.compile(rf"[A-Za-z_][A-Za-z0-9_]{{{_MIN_TOKEN_LEN - 1},}}")


def _tokenize(text: str) -> list[str]:
    """Extract salient tokens from text. Mirrors lexical.py:199 exactly.

    Lowercased, stopword-filtered, min length 3. Duplicates are *kept*
    (build path needs term frequencies for BM25). Callers who need a
    set can wrap in `set()`.
    """
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0).lower()
        if len(tok) < _MIN_TOKEN_LEN:
            continue
        if tok in _STOPWORDS:
            continue
        out.append(tok)
    return out


def _hash_to_dim_and_sign(token: str, dim: int, seed: int) -> tuple[int, int]:
    """Deterministic (index, sign) for a token under blake2b.

    blake2b is fast, cryptographically strong, and platform-stable —
    unlike Python's built-in `hash()` which is salted per interpreter
    start (PYTHONHASHSEED). The platform-stability matters because a
    cartridge built on Linux must resolve to the same band on Windows.
    """
    key = f"{seed:x}:{token}".encode("utf-8")
    h = hashlib.blake2b(key, digest_size=8).digest()
    idx = int.from_bytes(h[:4], "little") % dim
    sign = 1 if (h[4] & 1) else -1
    return idx, sign


def encode_query_lexical(
    query: str,
    dim: int,
    seed: int = DEFAULT_SEED,
) -> NDArray[np.float32]:
    """Project a query string into a D-dim lexical vector.

    Unit-weighted feature hashing: each distinct query token contributes
    ±1 at its hashed dimension. Empty or all-stopword queries return a
    zero vector (the lexical band then contributes 0 to the resonance
    sum — correct "no signal" behaviour).
    """
    tokens = _tokenize(query)
    # Deduplicate — a token repeated in a query shouldn't amplify the
    # signal at its hashed position; it's still one concept.
    seen_tokens = list(dict.fromkeys(tokens))

    v = np.zeros(dim, dtype=np.float32)
    if not seen_tokens:
        return v
    for tok in seen_tokens:
        idx, sign = _hash_to_dim_and_sign(tok, dim, seed)
        v[idx] += sign
    norm = float(np.linalg.norm(v))
    if norm > 0:
        v /= norm
    return v


def build_lexical_vectors(
    docs: Sequence[str] | Iterable[str],
    dim: int,
    seed: int = DEFAULT_SEED,
    *,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> NDArray[np.float32]:
    """BM25-weighted feature-hashed per-doc vectors of shape (N, D).

    Args:
        docs: Corpus of N source texts (same N and order as the caller
            passes to ``Lattice.superpose_text_batch``).
        dim: Output dimensionality per vector (matches the semantic
            bands' D).
        seed: Hash seed. Default is `DEFAULT_SEED`; override only if
            coordinating a format migration.
        k1, b: BM25 Okapi hyperparameters. Defaults match `bm25s` and
            the wider IR literature.

    Returns:
        An (N, D) float32 array. Rows are L2-normalised; empty docs
        return a zero row (the lexical band then adds zero energy —
        the same "no signal" contract as `encode_query_lexical`).

    Complexity: O(sum of doc lengths). Tokenisation is the bottleneck;
    the hash step is O(1) per token.
    """
    doc_list = list(docs)
    n_docs = len(doc_list)

    tokenized: list[list[str]] = [_tokenize(d) for d in doc_list]
    doc_lens = np.array([len(toks) for toks in tokenized], dtype=np.float32)
    # Guard both "zero docs" and "all docs empty" — avg_dl=0 would
    # blow up the BM25 denominator. Clamp to 1.0 so the formula stays
    # numerically sane; empty-doc rows stay zero anyway.
    avg_dl = float(doc_lens.mean()) if n_docs > 0 and doc_lens.mean() > 0 else 1.0

    # Document frequency per term
    df: dict[str, int] = {}
    for toks in tokenized:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    # Robertson-Spärck-Jones IDF with the +1 inside the log (lucene
    # convention) to stay non-negative when df > N/2.
    idf_cache: dict[str, float] = {
        t: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
        for t, freq in df.items()
    }

    vecs = np.zeros((n_docs, dim), dtype=np.float32)
    for i, toks in enumerate(tokenized):
        if not toks:
            continue
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        dl = float(doc_lens[i])
        for t, freq in tf.items():
            w_idf = idf_cache.get(t, 0.0)
            if w_idf <= 0.0:
                continue
            denom = freq + k1 * (1.0 - b + b * dl / avg_dl)
            w = w_idf * (freq * (k1 + 1.0)) / denom if denom > 0 else 0.0
            idx, sign = _hash_to_dim_and_sign(t, dim, seed)
            vecs[i, idx] += sign * w

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs /= norms
    return vecs


# ─────────────────────────────────────────────────────────────────────
# Drop-in rerank API — parity with retrieval.lexical.lexical_rerank
# ─────────────────────────────────────────────────────────────────────
#
# Why here (and not as a separate module): this function is the V1
# integration path — it computes a lexical signal *algebraically
# equivalent* to the ripgrep post-pass but without touching disk or
# shelling out. V2 (follow-up) folds the same projection into a proper
# 6th field band for composition under merge / forget. Both share the
# build + encode primitives above, so they live together.


def lexical_band_rerank(
    results: Sequence[Any],
    query: str,
    *,
    get_text: Callable[[Any], str | None],
    dim: int,
    seed: int = DEFAULT_SEED,
    weight: float = DEFAULT_WEIGHT,
) -> list[Any]:
    """Drop-in replacement for ``retrieval.lexical.lexical_rerank``.

    Scores each candidate against a feature-hashed BM25-weighted
    per-document vector, then blends with the dense score via
    ``combined = (1 - w) * norm_dense + w * norm_lexical`` — the same
    blend contract as the ripgrep pass (lexical module's
    ``_combine_scores``). The only behavioural differences:

    * **Works across all store modes.** No subprocess, no file reads —
      `get_text` abstracts content retrieval, so bundled / local /
      remote cartridges all get the same treatment.
    * **Deterministic.** Same inputs → same outputs, across platforms.
      No "byte ≈ char" caveat; tokenisation is the same at build and
      query.
    * **Not fail-soft.** If the caller can give us texts, we give them
      a blend. If they can't, they filter the list before calling us.

    Args:
        results: A sequence of hits. Each element is treated opaquely;
            the only thing we read from it is the score via attribute
            or dict access (see ``_get_score``). Mutations happen on
            copies, so the input is not modified.
        query: The original query string. Tokenised here.
        get_text: ``(result) -> str | None``. Callers supply this to
            resolve a hit's text. Returning None is safe — that row
            gets a zero lexical contribution (dense-only signal).
        dim: Projection dimensionality. Use the encoder's D for the
            eventual band fold; any D works for the rerank path.
        seed: Hash seed. Must match what was used at build time if
            and when build-time caching is introduced.
        weight: Blend weight in [0, 1]. 0 returns the input order
            unchanged; 1 is pure-lexical. Default mirrors
            ``retrieval.lexical.DEFAULT_WEIGHT``.

    Returns:
        A new list (shallow copies of input hits) ordered by blended
        score, with ``score`` / ``raw_score`` / ``provenance`` updated.
    """
    import copy as _copy
    import dataclasses as _dc

    if not results or weight <= 0.0:
        return list(results)

    q_vec = encode_query_lexical(query, dim=dim, seed=seed)
    if float(np.linalg.norm(q_vec)) == 0.0:
        return list(results)

    # Resolve texts; missing text → empty string → zero row → zero lex score.
    texts: list[str] = []
    for r in results:
        try:
            t = get_text(r)
        except Exception:
            t = None
        texts.append(t or "")

    doc_vecs = build_lexical_vectors(texts, dim=dim, seed=seed)
    lex_scores = doc_vecs @ q_vec  # (N,)

    # Clip negatives from the random-sign hashing so the blend can't
    # *subtract* signal from a dense match. Parity with the ripgrep
    # pass which saturates at [0, 1].
    lex_scores = np.clip(lex_scores, 0.0, None)

    dense_scores = np.array(
        [float(_get_score(r)) for r in results], dtype=np.float32
    )
    d_max = float(dense_scores.max()) if dense_scores.size else 0.0
    norm_dense = dense_scores / d_max if d_max > 0 else dense_scores

    l_max = float(lex_scores.max()) if lex_scores.size else 0.0
    norm_lex = lex_scores / l_max if l_max > 0 else lex_scores

    combined = (1.0 - weight) * norm_dense + weight * norm_lex

    out: list[Any] = []
    for r, new_score, lex_contrib in zip(results, combined, norm_lex):
        prev_score = float(_get_score(r))
        prev_raw = _get_attr(r, "raw_score")
        prev_prov = _get_attr(r, "provenance") or "dense"
        updates: dict[str, Any] = {"score": float(new_score)}
        if prev_raw is None:
            updates["raw_score"] = prev_score
        if lex_contrib > 0 and "lexical" not in str(prev_prov):
            updates["provenance"] = f"{prev_prov}+lexical"

        out.append(_with_updates(r, updates))

    out.sort(key=lambda r: float(_get_score(r)), reverse=True)
    return out


def _get_score(r: Any) -> float:
    """Read `score` from an object or dict-like hit."""
    if isinstance(r, dict):
        return float(r.get("score") or 0.0)
    return float(getattr(r, "score", 0.0) or 0.0)


def _get_attr(r: Any, name: str) -> Any:
    if isinstance(r, dict):
        return r.get(name)
    return getattr(r, name, None)


def _with_updates(r: Any, updates: dict[str, Any]) -> Any:
    """Return a new hit with `updates` applied, respecting frozen dataclasses.

    Three cases:
      * dict           → shallow-copy, update keys.
      * frozen dataclass → use dataclasses.replace (accepts only declared fields).
      * mutable object → copy.copy + setattr.
    """
    import copy as _copy
    import dataclasses as _dc

    if isinstance(r, dict):
        out = dict(r)
        out.update(updates)
        return out

    if _dc.is_dataclass(r):
        # dataclasses.replace only accepts declared fields — silently
        # drop updates the dataclass doesn't model (e.g. `provenance`
        # on the frozen ScoredHit). The blended score + raw_score are
        # what the caller's aggregator actually reads.
        declared = {f.name for f in _dc.fields(r)}
        safe_updates = {k: v for k, v in updates.items() if k in declared}
        return _dc.replace(r, **safe_updates)

    copy_r = _copy.copy(r)
    for k, v in updates.items():
        try:
            setattr(copy_r, k, v)
        except (AttributeError, TypeError):
            # Read-only attribute / slot without the name. Skip silently
            # — the sort key below still reflects the new score on any
            # object where `.score` could be set successfully.
            pass
    return copy_r


__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_K1",
    "DEFAULT_B",
    "DEFAULT_WEIGHT",
    "build_lexical_vectors",
    "encode_query_lexical",
    "lexical_band_rerank",
]
