# SPDX-License-Identifier: BUSL-1.1
"""Query-adaptive retrieval routing for conversation memory.

Classifies a query by surface features (aggregation, temporal, knowledge-update,
entity anchor, short factoid) and returns a retrieval config tuned for that
question type plus optional post-retrieval flags (temporal_window_days,
prefer_recent). Pure function — no LLM, no encoder calls.

Derived from LongMemEval v6 analysis: 28% of questions benefit from cascade,
37% from temporal filtering, 32% from prefer-recent. See plan
"nifty-sparking-quokka.md" for the routing table.
"""

from __future__ import annotations

import re
from typing import Any

# Feature regexes. Case-insensitive, word-boundary anchored where noise matters.
_AGGREGATION_RE = re.compile(
    r"\b(how many|total|count|all|every|each|sum|average|across|overall)\b",
    re.IGNORECASE,
)
_TEMPORAL_RE = re.compile(
    r"\b(ago|months?|weeks?|days?|years?|hours?|since|before|after|"
    r"first|last|recent|earlier|later|previously|when)\b",
    re.IGNORECASE,
)
_EXPLICIT_DATE_RE = re.compile(
    r"\b(20\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_KNOWLEDGE_UPDATE_RE = re.compile(
    r"\b(current|currently|now|still|latest|updated|changed|these days|at present)\b",
    re.IGNORECASE,
)
_MY_NOUN_RE = re.compile(r"\bmy\s+([a-z][a-z\-]{2,})\b", re.IGNORECASE)
_TITLE_CASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


def _has_aggregation(q: str) -> bool:
    return bool(_AGGREGATION_RE.search(q))


def _has_temporal(q: str) -> bool:
    return bool(_TEMPORAL_RE.search(q) or _EXPLICIT_DATE_RE.search(q))


def _has_knowledge_update(q: str) -> bool:
    return bool(_KNOWLEDGE_UPDATE_RE.search(q))


def _has_entity_anchor(q: str) -> bool:
    return bool(_MY_NOUN_RE.search(q) or _TITLE_CASE_RE.search(q))


def _is_short_factoid(q: str) -> bool:
    words = [w for w in re.findall(r"\w+", q) if w]
    return len(words) <= 5


def detect_features(query: str) -> dict[str, bool]:
    """Return a dict of feature flags for introspection and tests."""
    return {
        "aggregation": _has_aggregation(query),
        "temporal": _has_temporal(query),
        "knowledge_update": _has_knowledge_update(query),
        "entity_anchor": _has_entity_anchor(query),
        "short_factoid": _is_short_factoid(query),
    }


def adaptive_memory_config(
    query: str,
    question_date: str | None = None,
) -> dict[str, Any]:
    """Return a retrieval config tuned for the query's detected type.

    Returns a dict with:
      - kwargs for ``Lattice.enriched_query`` (``enable_lexical``, ``enable_rerank``,
        ``enable_cascade``, ``cascade_depth``, ``enable_contradictions``,
        ``lexical_weight``)
      - post-retrieval flags: ``temporal_window_days`` (int or None),
        ``prefer_recent`` (bool)
      - ``features`` dict for debugging / benchmark reporting

    ``question_date`` is the ISO timestamp of the query turn. It is passed
    through unchanged so callers can use it for temporal filtering.
    """
    features = detect_features(query)

    # Defaults — tuned for conversation memory per v6/v7 findings.
    # Session diversity is ON by default: per-query analysis on v7 showed
    # the only multi-session ceiling lift (R@5 0.65 -> 0.94) comes from
    # session-level deduplication of top-20 results. Same mechanism fixes
    # the knowledge-update regression (lexical duplicates dominant session).
    # It is a no-op when every result is already from a distinct session.
    cfg: dict[str, Any] = {
        "enable_lexical": True,
        "enable_rerank": "auto",
        "enable_cross_encoder": False,
        "enable_cascade": False,
        "cascade_depth": 2,
        "enable_contradictions": False,
        "asymmetric": False,
        "lexical_weight": 0.3,
        "temporal_window_days": None,
        "temporal_window_past_days": None,   # asymmetric backward-only window
        "temporal_window_future_days": None,
        "prefer_recent": False,
        "diversify_by_session": True,
    }

    if features["aggregation"]:
        # Aggregation needs coverage across many sessions.
        cfg["enable_cascade"] = True
        cfg["cascade_depth"] = 3

    if features["temporal"]:
        # v7 analysis: 10/10 answer sessions sit 0–154 days BEFORE question_date.
        # Symmetric ±60d misses the tail (154d, 148d). Asymmetric backward
        # window (180d back, 30d forward grace) covers 10/10.
        cfg["temporal_window_past_days"] = 180
        cfg["temporal_window_future_days"] = 30
        cfg["enable_cascade"] = True

    if features["knowledge_update"]:
        # v7 finding: regression was session-diversity collapse, not stale
        # content. Diversity default + prefer-recent tiebreak solves it
        # without killing lexical retrieval.
        cfg["prefer_recent"] = True
        cfg["enable_contradictions"] = True

    if features["entity_anchor"] and not features["knowledge_update"]:
        cfg["enable_lexical"] = True
        cfg["lexical_weight"] = 0.7

    if features["short_factoid"] and not (
        features["aggregation"] or features["temporal"] or features["knowledge_update"]
    ):
        # Short factoids: trust dense, skip rerank churn
        cfg["enable_rerank"] = False
        cfg["enable_lexical"] = False

    cfg["features"] = features
    cfg["question_date"] = question_date
    return cfg
