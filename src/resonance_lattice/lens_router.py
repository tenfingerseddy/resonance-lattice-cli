# SPDX-License-Identifier: BUSL-1.1
"""Lens router: deterministic query-to-lens dispatch for smart retrieval.

Classifies a query by surface features and knowledge model context to select
the best retrieval lens (search, locate, profile, compare, compose_search)
and optional EML transform (--tune focus/explore/denoise, --contrast).

Pure function — no LLM, no encoder calls. Claude sits above and can
override. The router's job is to make the default choice good enough
that override is rare.

See docs/SKILL_INTEGRATION.md and the lens catalog in the launch plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── Feature detectors ───────────────────────────────────────────────

_COMPARE_RE = re.compile(
    r"\b(compare|differ(?:s|ent|ence)?|disagree|overlap|contradict|vs\.?|versus|"
    r"how (?:do|does|is|are) .{1,40} different|"
    r"(?:the |what.{1,10})?difference(?:s)? between)\b",
    re.IGNORECASE,
)
_LOCATE_RE = re.compile(
    r"\b(what (?:does|do) (?:this|the|it) .{1,30} (?:know|cover|contain)|"
    r"where (?:is|are) (?:the|my) gaps?|coverage|what.{1,20}missing|"
    r"(?:the |my )?gaps? in|"
    r"does (?:this |the |it )?(?:knowledge model )?.{0,20}cover)\b",
    re.IGNORECASE,
)
_PROFILE_RE = re.compile(
    r"\b(overview|summarize|characterize|describe .{1,20} corpus|"
    r"what is .{1,20} about|high.?level|shape of)\b",
    re.IGNORECASE,
)
_COMPOSE_RE = re.compile(
    r"\b(across (?:these|both|all|multiple)|through the lens of|"
    r"combining|merged|from (?:both|all) knowledge models)\b",
    re.IGNORECASE,
)
_FACTOID_RE = re.compile(
    r"\b(what is|define|definition of|how do you|"
    r"what (?:is|are) the .{1,20}(?:for|of|to))\b",
    re.IGNORECASE,
)
_EXPLORE_RE = re.compile(
    r"\b(explore|brainstorm|broad|related|adjacent|"
    r"what else|anything (?:about|related)|expand)\b",
    re.IGNORECASE,
)
_CONTRAST_RE = re.compile(
    r"\b(what (?:does|do) .{1,30} know that .{1,30} (?:doesn.?t|don.?t|lacks?)|"
    r"unique to|exclusive to|not in|novel in)\b",
    re.IGNORECASE,
)


def detect_features(query: str) -> dict[str, bool]:
    """Detect query intent features for lens routing."""
    return {
        "compare": bool(_COMPARE_RE.search(query)),
        "locate": bool(_LOCATE_RE.search(query)),
        "profile": bool(_PROFILE_RE.search(query)),
        "compose": bool(_COMPOSE_RE.search(query)),
        "factoid": bool(_FACTOID_RE.search(query)),
        "explore": bool(_EXPLORE_RE.search(query)),
        "contrast": bool(_CONTRAST_RE.search(query)),
    }


# ── Lens choice ─────────────────────────────────────────────────────

@dataclass
class LensChoice:
    """Selected retrieval lens with rationale."""
    lens: str           # search, locate, profile, compare, negotiate, compose_search
    args: dict          # additional flags: --tune, --contrast, --with-cartridges, etc.
    rationale: str      # human-readable explanation for --explain mode


def route_query(
    query: str,
    num_cartridges: int = 1,
    background_cartridge: str | None = None,
) -> LensChoice:
    """Select the best retrieval lens for a query.

    Args:
        query: The user's question.
        num_cartridges: Number of knowledge models available (>1 enables compose).
        background_cartridge: If set, a second knowledge model for contrast operations.

    Returns:
        LensChoice with the recommended lens, args, and rationale.
    """
    features = detect_features(query)

    # Priority ordering: most specific intent wins

    if features["contrast"] and background_cartridge:
        return LensChoice(
            lens="search",
            args={"contrast": background_cartridge},
            rationale=(
                f"Query asks what one cartridge knows that another doesn't. "
                f"Using EML contrast against {background_cartridge}."
            ),
        )

    if features["compare"] and num_cartridges >= 2:
        return LensChoice(
            lens="negotiate",
            args={},
            rationale="Query compares or asks about differences between knowledge models.",
        )

    if features["compare"]:
        return LensChoice(
            lens="compare",
            args={},
            rationale="Query asks to compare, diff, or find contradictions.",
        )

    if features["locate"]:
        return LensChoice(
            lens="locate",
            args={},
            rationale="Query asks about coverage, gaps, or what the knowledge model knows.",
        )

    if features["profile"]:
        return LensChoice(
            lens="profile",
            args={},
            rationale="Query asks for an overview or characterization of the corpus.",
        )

    if features["compose"] and num_cartridges >= 2:
        return LensChoice(
            lens="compose_search",
            args={},
            rationale="Query spans multiple knowledge models — composing fields for search.",
        )

    if features["factoid"] and not features["explore"]:
        return LensChoice(
            lens="search",
            args={"tune": "focus"},
            rationale=(
                "Factoid/definition query — using EML focus preset "
                "to sharpen dominant topics."
            ),
        )

    if features["explore"]:
        return LensChoice(
            lens="search",
            args={"tune": "explore"},
            rationale=(
                "Exploratory query — using EML explore preset "
                "to surface buried topics."
            ),
        )

    # Default: plain search
    return LensChoice(
        lens="search",
        args={},
        rationale="General retrieval query — using standard search.",
    )


def format_explain(choice: LensChoice, cartridge: str, query: str) -> str:
    """Format the --explain output showing the chosen lens and the command to run."""
    parts = [f"rlat {choice.lens} {cartridge} \"{query}\""]
    for k, v in choice.args.items():
        if isinstance(v, bool) and v:
            parts.append(f"--{k}")
        elif v is not None:
            parts.append(f"--{k} {v}")
    cmd = " ".join(parts)

    return (
        f"Lens: {choice.lens}\n"
        f"Rationale: {choice.rationale}\n"
        f"Command: {cmd}\n"
    )
