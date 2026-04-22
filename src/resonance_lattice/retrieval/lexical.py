# SPDX-License-Identifier: BUSL-1.1
"""Lexical second-pass reranking (B3 of the v1.0.0 semantic-layer plan).

The field router returns chunks by dense semantic similarity. Dense
retrieval is strong on paraphrase and fuzzy intent but weak on exact
terminology — a query for `FORMAT_VERSION` or `RlatHeader.store_mode`
can return conceptually adjacent chunks that never mention the literal
identifier. `lexical_rerank` runs `ripgrep` over the field-retrieved
neighbourhood (the files the top-K hits came from) and boosts hits
whose char range contains — or sits close to — a literal query match.

Design contract:

1. **Additive, never subtractive.** The dense ranking is the floor.
   Hybrid retrieval should improve the top-K on average without ever
   demoting a dense-only hit below a weaker dense hit that also lacks
   lexical confirmation. See `_combine_scores` — both paths normalise
   into [0,1] and blend, so relative order is preserved when lexical
   is uninformative.

2. **Fail-soft.** `rg` missing, timed out, or crashing must return the
   original list unchanged. Hybrid is a best-effort enhancement layer
   — the semantic layer as a whole must keep working when the local
   machine doesn't have ripgrep installed.

3. **Pure function of hits + query + source_root.** No hidden state,
   no coupling to `Lattice` / `Store` / `Registry`. Callers pass what
   they retrieved; we re-sort it. This keeps the retrieval package
   composable and trivially testable.

4. **Neighbourhood = files referenced by `field_hits`.** We don't run
   `rg` over the whole corpus — that would be a separate lexical index,
   not a second pass. The point of the field is to narrow the search
   space; lexical exploits that narrowing.

5. **Byte ≈ char approximation.** `rg --json` emits byte offsets; hits
   carry char offsets (from the A3 chunker work). For ASCII-dominant
   corpora (code, English docs) this is exact; for heavy multi-byte
   content proximity is approximate but never wildly wrong. Proper
   byte↔char mapping is deferred — the 1.5d B3 budget doesn't cover
   per-file UTF-8 accounting, and the common case is fine without it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

# Minimum token length. Shorter fragments (`if`, `a`) are too noisy as
# lexical signals — they match everywhere and drown out real evidence.
_MIN_TOKEN_LEN = 3

# English stopwords plus common code / CLI filler. Kept short — the
# point is to strip the obvious, not to be a linguistic resource. Add
# here when a stopword empirically shows up as noise.
_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "had", "her", "was", "one", "our", "out", "day", "get", "has", "him",
    "his", "how", "man", "new", "now", "old", "see", "two", "way", "who",
    "boy", "did", "its", "let", "put", "say", "she", "too", "use", "with",
    "this", "that", "from", "what", "when", "where", "which", "there",
    "been", "have", "they", "were", "will", "your", "about", "would",
})

# Default knobs. These are starting points; benchmarks in Block D will
# tune the weights.
DEFAULT_WEIGHT = 0.3              # lexical's share of the blended score
DEFAULT_PROXIMITY_CHARS = 500     # window around hit for "nearby" match
DEFAULT_TIMEOUT = 5.0             # seconds — rg shouldn't be slow
_MATCH_SATURATION = 3.0           # diminishing returns above N matches


@dataclass(frozen=True)
class ScoredHit:
    """Minimal hit contract for lexical reranking.

    Deliberately decoupled from `Lattice` / `Store` types — callers
    adapt their richer hit representation into this shape. Path format
    is caller's choice; we resolve to disk via `source_root` when the
    path isn't already absolute.

    Attributes:
        source_file: Path to the source file. Relative paths resolve
            against `source_root`; absolute paths are used as-is.
        char_offset: Where the hit starts in the file.
        char_length: Length of the hit span.
        text: The hit's text (not used for scoring; carried through so
            callers can display the reranked result).
        score: Dense score from the field. Higher = better.
    """
    source_file: str
    char_offset: int
    char_length: int
    text: str
    score: float


def lexical_rerank(
    field_hits: Sequence[ScoredHit],
    query: str,
    source_root: str | Path | None = None,
    *,
    weight: float = DEFAULT_WEIGHT,
    proximity_chars: int = DEFAULT_PROXIMITY_CHARS,
    rg_timeout: float = DEFAULT_TIMEOUT,
    _rg_runner: Callable[..., list[_RgMatch]] | None = None,
) -> list[ScoredHit]:
    """Rerank `field_hits` by blending in a lexical signal from ripgrep.

    Args:
        field_hits: The top-K chunks returned by dense retrieval.
        query: The original query string. Tokenised here — salient
            alphanumeric words of length ≥ 3, stopwords stripped.
        source_root: Directory that relative `source_file` paths
            resolve against. Optional if hits carry absolute paths.
        weight: Share of the blended score taken by the lexical
            signal. 0.0 = dense only (hybrid off); 1.0 = lexical only.
            Default 0.3 is a starting point; tune in Block D.
        proximity_chars: A lexical match counts toward a hit's boost
            only if it lies within this many chars of the hit's span.
        rg_timeout: Wall-clock cap on the ripgrep invocation.
        _rg_runner: Test seam — allows tests to inject deterministic
            matches without shelling out. Production callers should
            never pass this.

    Returns:
        A new list of ScoredHit, sorted descending by blended score.
        On any failure (no rg, no tokens, empty hits, subprocess error)
        returns `list(field_hits)` in input order — fail-soft is the
        core invariant.
    """
    hits = list(field_hits)
    if not hits or weight <= 0.0:
        return hits

    tokens = _extract_query_tokens(query)
    if not tokens:
        return hits

    # Resolve hit paths to absolute disk paths for rg and proximity calcs.
    # Hits with unresolvable paths are retained but get zero lexical boost.
    root = Path(source_root).resolve() if source_root else None
    resolved: dict[int, Path] = {}
    for i, hit in enumerate(hits):
        resolved_path = _resolve_path(hit.source_file, root)
        if resolved_path is not None:
            resolved[i] = resolved_path

    file_set = sorted({p for p in resolved.values()})
    if not file_set:
        return hits

    runner = _rg_runner if _rg_runner is not None else _run_ripgrep
    try:
        matches = runner(tokens, file_set, timeout=rg_timeout)
    except _RgUnavailableError:
        return hits
    except Exception:
        # Any parse / subprocess surprise → fail-soft. Rerank is
        # opportunistic; we never want a broken rg install to crash
        # the search path.
        return hits

    if not matches:
        return hits

    # Group matches by absolute file path for fast per-hit lookup.
    by_file: dict[Path, list[_RgMatch]] = {}
    for m in matches:
        by_file.setdefault(m.path, []).append(m)

    # Per-hit lexical score in [0,1]. Saturates at _MATCH_SATURATION so
    # a file that happens to use the token 50 times doesn't dominate.
    lex_scores: list[float] = []
    for i, hit in enumerate(hits):
        path = resolved.get(i)
        if path is None or path not in by_file:
            lex_scores.append(0.0)
            continue
        nearby = _count_nearby(by_file[path], hit, proximity_chars)
        lex_scores.append(min(1.0, nearby / _MATCH_SATURATION))

    return _combine_scores(hits, lex_scores, weight)


# ─────────────────────────────────────────────────────────────────────
# Query tokenization
# ─────────────────────────────────────────────────────────────────────


_TOKEN_RE = re.compile(rf"[A-Za-z_][A-Za-z0-9_]{{{_MIN_TOKEN_LEN - 1},}}")


def _extract_query_tokens(query: str) -> list[str]:
    """Salient alphanumeric tokens from `query`.

    Lowercased for case-insensitive rg matching (we pass -i to rg too).
    Deduplicated, stopwords removed, minimum length enforced. Returns
    a list — ordering matches first-occurrence in the query, which
    tests and log output appreciate.
    """
    seen: set[str] = set()
    out: list[str] = []
    for m in _TOKEN_RE.finditer(query):
        tok = m.group(0).lower()
        if len(tok) < _MIN_TOKEN_LEN:
            continue
        if tok in _STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


# ─────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────


def _resolve_path(source_file: str, root: Path | None) -> Path | None:
    """Resolve a hit's `source_file` to an absolute path on disk.

    Accepts mixed separators (Windows ↔ POSIX) — the A2 manifest
    normalisation stores forward-slash paths but tests and callers may
    pass either. Returns None when the file can't be located; the hit
    keeps its original score in that case.
    """
    if not source_file:
        return None
    p = Path(source_file)
    if p.is_absolute():
        return p if p.exists() else None
    if root is None:
        return None
    candidate = (root / source_file).resolve()
    if candidate.exists():
        return candidate
    # Try forward-slash → native conversion explicitly, for cartridges
    # built on Linux and searched on Windows or vice versa.
    candidate = (root / Path(*source_file.replace("\\", "/").split("/"))).resolve()
    return candidate if candidate.exists() else None


# ─────────────────────────────────────────────────────────────────────
# Ripgrep invocation + parsing
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _RgMatch:
    """One literal-token match from ripgrep.

    `offset` is the byte offset of the matched line's start in the
    file (what `rg --json`'s `absolute_offset` gives). Close enough to
    char offset for ASCII-dominant corpora; see module docstring.
    """
    path: Path
    line_number: int
    offset: int
    text: str


class _RgUnavailableError(RuntimeError):
    """Raised when `rg` isn't on PATH or the subprocess can't be started."""


def _run_ripgrep(
    tokens: Sequence[str],
    files: Sequence[Path],
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[_RgMatch]:
    """Run `rg --json` for `tokens` inside `files`, return parsed matches.

    Flags:
        --json         — structured output we can parse reliably.
        --word-regexp  — match whole words, so "test" doesn't match
                          "tested" and inflate the count.
        --fixed-strings — treat each token as a literal, not a regex.
        --ignore-case  — queries are usually written in natural case;
                          identifiers in code may be camel/snake. Match
                          both directions.

    Raises:
        _RgUnavailableError: rg missing on PATH or failed to start.
    """
    if shutil.which("rg") is None:
        raise _RgUnavailableError("ripgrep (rg) not found on PATH")

    cmd: list[str] = [
        "rg",
        "--json",
        "--word-regexp",
        "--fixed-strings",
        "--ignore-case",
        "--no-messages",
    ]
    for tok in tokens:
        cmd.extend(["-e", tok])
    # `--` separates flags from file arguments so a file named `-foo`
    # doesn't get parsed as a flag.
    cmd.append("--")
    cmd.extend(str(p) for p in files)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as e:
        raise _RgUnavailableError(str(e))
    except subprocess.TimeoutExpired:
        # Treat a timeout as "lexical couldn't answer in time". Hybrid
        # is opportunistic — don't let it stall a search.
        return []

    # rg exits 1 on "no matches" which is not an error for our purposes.
    # Exit 2 is a real error (bad args, decode issue); fall through to
    # parse whatever it printed, then return [] if nothing usable.
    matches: list[_RgMatch] = []
    try:
        text = proc.stdout.decode("utf-8", errors="replace")
    except Exception:
        return []

    for line in text.splitlines():
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event.get("data") or {}
        path_text = (data.get("path") or {}).get("text")
        if not path_text:
            continue
        absolute_offset = int(data.get("absolute_offset") or 0)
        line_number = int(data.get("line_number") or 0)
        submatches = data.get("submatches") or []
        line_text = (data.get("lines") or {}).get("text", "")
        if not submatches:
            continue
        for sm in submatches:
            match_text = (sm.get("match") or {}).get("text", "")
            rel_start = int(sm.get("start") or 0)
            matches.append(
                _RgMatch(
                    path=Path(path_text).resolve(),
                    line_number=line_number,
                    offset=absolute_offset + rel_start,
                    text=match_text or line_text,
                )
            )
    return matches


# ─────────────────────────────────────────────────────────────────────
# Proximity + scoring
# ─────────────────────────────────────────────────────────────────────


def _count_nearby(
    matches: Iterable[_RgMatch],
    hit: ScoredHit,
    proximity_chars: int,
) -> int:
    """How many `matches` fall within `proximity_chars` of the hit span.

    Overlap with the hit's own span counts (match inside the chunk).
    Matches before or after the chunk count if their distance to the
    nearest chunk edge is within the window. This is what "neighbourhood
    confirmation" means — a term mentioned in the enclosing function /
    section near the hit is evidence the hit is on-topic.
    """
    hit_start = hit.char_offset
    hit_end = hit.char_offset + max(0, hit.char_length)
    count = 0
    for m in matches:
        if m.offset >= hit_start - proximity_chars and m.offset <= hit_end + proximity_chars:
            count += 1
    return count


def _combine_scores(
    hits: Sequence[ScoredHit],
    lex_scores: Sequence[float],
    weight: float,
) -> list[ScoredHit]:
    """Blend dense and lexical scores and return a new sorted list.

    Dense scores are normalised to [0,1] via the batch max so the blend
    is comparable. A batch where every dense score is zero (unusual —
    no dense signal) falls back to pure lexical ordering.
    """
    if not hits:
        return []
    max_dense = max((h.score for h in hits), default=0.0)
    if max_dense <= 0.0:
        # No dense signal to normalise against — rank purely by lex.
        norm_dense = [0.0] * len(hits)
    else:
        norm_dense = [h.score / max_dense for h in hits]

    blended: list[ScoredHit] = []
    for hit, nd, ls in zip(hits, norm_dense, lex_scores):
        combined = (1.0 - weight) * nd + weight * ls
        blended.append(replace(hit, score=combined))

    # Stable sort: original order breaks ties so repeated runs are
    # deterministic. Python's sort is already stable.
    blended.sort(key=lambda h: h.score, reverse=True)
    return blended
