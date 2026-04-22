# SPDX-License-Identifier: BUSL-1.1
"""Derive bootstrap query topics from recent git history.

Primer generation's static 5-query bootstrap cannot surface work that
post-dates the queries. This module reads `git log` and returns the
high-frequency topic prefixes (e.g. `phase6/pass4a`, `refresh_cartridge`,
`dogfood`) so `rlat summary` can add them as retrieval anchors.

Entry point: `derive_commit_topics(repo_root)`. All failure modes
fall back to an empty list — callers blend the result into their
static queries, so a missing git binary or non-repo directory just
degrades gracefully to pre-existing behaviour.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

_SPLIT_RE = re.compile(r"[\s/:()\[\]]+")
_STOP = {
    # Articles / prepositions / conjunctions
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "at",
    "from", "into", "onto", "with", "by", "as", "via", "per",
    # Generic commit verbs / types (conventional-commit prefixes)
    "add", "adds", "added", "adding",
    "remove", "removes", "removed", "removing",
    "fix", "fixes", "fixed", "fixing",
    "update", "updates", "updated", "updating",
    "refactor", "refactors", "refactored", "refactoring",
    "chore", "docs", "doc", "test", "tests", "testing",
    "bump", "bumps", "release", "releases",
    "merge", "merges", "merged", "merging", "pr", "pull", "request",
    "wip", "tmp", "temp", "draft",
    "feat", "feats", "feature", "features",
    "build", "build(s)", "ci", "style", "perf", "revert",
    # Claude-Code trailer noise
    "co-authored-by", "noreply", "anthropic", "claude", "com",
    "sonnet", "opus", "haiku", "generated", "code",
    # GitHub merge-commit shrapnel
    "tenfingerseddy", "github", "com",
}

# Lines like "Merge pull request #123 from owner/branch" contribute no topic
# signal — they bury the real scope one level down. Drop them outright.
_MERGE_SUBJECT_RE = re.compile(r"^\s*Merge\s+(pull request|branch|remote-tracking)\b", re.IGNORECASE)


def derive_commit_topics(
    repo_root: Path,
    since_days: int = 14,
    max_topics: int = 8,
    max_commits: int = 500,
) -> list[str]:
    """Return up to `max_topics` high-frequency topic phrases from recent commits.

    Looks at commit subjects from the last `since_days` days. Splits each
    subject on `/` and `:` to extract prefixes (conventional-commit scopes
    and phase/track markers), lowercases, de-dupes, and keeps the most
    frequent. Returns `[]` on any failure — git missing, not a repo,
    empty window, subprocess error.
    """
    git = shutil.which("git")
    if git is None:
        return []
    if not repo_root.exists():
        return []

    try:
        proc = subprocess.run(
            [
                git, "-C", str(repo_root), "log",
                f"--since={since_days}.days",
                "--pretty=format:%s",
                f"-n{max_commits}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"commit_topics: git log failed: {exc}", file=sys.stderr)
        return []

    if proc.returncode != 0:
        return []

    subjects = [s.strip() for s in proc.stdout.splitlines() if s.strip()]
    subjects = [s for s in subjects if not _MERGE_SUBJECT_RE.match(s)]
    if not subjects:
        return []

    topic_counts: Counter[str] = Counter()
    for subject in subjects:
        # Commit subjects come in two common shapes:
        #   "scope: message"                          → use the scope
        #   "scope/subscope: message"                 → use scope AND subscope
        #   "message with no scope"                   → use leading bigram
        head = subject.split(":", 1)[0] if ":" in subject else subject
        parts = [p for p in _SPLIT_RE.split(head) if p]
        for p in parts:
            tok = p.lower().strip("-._#")
            if len(tok) < 3 or tok in _STOP or tok.isdigit():
                continue
            # Version markers (v2, v1.1, 0.11.0) aren't topics
            if re.fullmatch(r"v?\d+(\.\d+)*[a-z]?", tok):
                continue
            topic_counts[tok] += 1

    topics = [t for t, c in topic_counts.most_common() if c >= 2]
    return topics[:max_topics]


def topics_to_queries(topics: list[str], since_days: int = 14) -> list[str]:
    """Phrase raw topics as retrieval-friendly questions.

    Prepends a dated sweep query that biases retrieval toward whatever
    has been touched recently, followed by one question per topic.
    """
    out: list[str] = []
    if topics:
        out.append(
            f"What was shipped or changed in the last {since_days} days, "
            "and why did those changes happen?"
        )
    for topic in topics:
        out.append(
            f"What does the recent work on {topic} change, "
            "and what is the current state of it?"
        )
    return out
