# SPDX-License-Identifier: BUSL-1.1
"""Thin UI layer for the setup wizard.

Uses ``questionary`` when available for rich interactive prompts
(arrow-key selection, checkboxes, validation). Falls back to plain
``input()`` numbered menus when questionary is not installed.

Install the interactive backend::

    pip install resonance-lattice[wizard]
"""

from __future__ import annotations

import itertools
import sys
import threading
import time
from typing import Any

try:
    import questionary  # type: ignore[import-untyped]
    _HAS_Q = True
except ImportError:
    _HAS_Q = False


# ── Prompt wrappers ──────────────────────────────────────────────────


def select(message: str, choices: list[str], default: str | None = None) -> str:
    """Single-choice selection prompt."""
    if _HAS_Q:
        return questionary.select(message, choices=choices, default=default).ask() or choices[0]
    return _fallback_select(message, choices, default)


def checkbox(
    message: str,
    choices: list[dict[str, Any]],
) -> list[str]:
    """Multi-choice checkbox prompt.

    Each choice dict has ``name`` (display) and ``checked`` (bool default).
    Returns list of selected names.
    """
    if _HAS_Q:
        q_choices = [
            questionary.Choice(c["name"], checked=c.get("checked", False))
            for c in choices
        ]
        result = questionary.checkbox(message, choices=q_choices).ask()
        return result if result else []
    return _fallback_checkbox(message, choices)


def confirm(message: str, default: bool = True) -> bool:
    """Yes/no confirmation prompt."""
    if _HAS_Q:
        result = questionary.confirm(message, default=default).ask()
        return result if result is not None else default
    return _fallback_confirm(message, default)


def text(message: str, default: str = "") -> str:
    """Free-text input prompt."""
    if _HAS_Q:
        result = questionary.text(message, default=default).ask()
        return result if result is not None else default
    return _fallback_text(message, default)


# ── Fallback implementations ─────────────────────────────────────────


def _fallback_select(message: str, choices: list[str], default: str | None) -> str:
    print(f"\n{message}")
    for i, c in enumerate(choices, 1):
        marker = " *" if c == default else ""
        print(f"  {i}. {c}{marker}")

    default_idx = choices.index(default) + 1 if default and default in choices else 1
    raw = input(f"  Choice [{default_idx}]: ").strip()
    if not raw:
        return choices[default_idx - 1]
    try:
        idx = int(raw)
        if 1 <= idx <= len(choices):
            return choices[idx - 1]
    except ValueError:
        pass
    return choices[default_idx - 1]


def _fallback_checkbox(message: str, choices: list[dict[str, Any]]) -> list[str]:
    print(f"\n{message}")
    for i, c in enumerate(choices, 1):
        mark = "x" if c.get("checked", False) else " "
        print(f"  {i}. [{mark}] {c['name']}")

    print("  Enter numbers to toggle (comma-separated), or press Enter for defaults:")
    raw = input("  > ").strip()
    if not raw:
        return [c["name"] for c in choices if c.get("checked", False)]

    # Toggle the specified indices
    selected = {c["name"] for c in choices if c.get("checked", False)}
    for part in raw.split(","):
        part = part.strip()
        try:
            idx = int(part) - 1
            if 0 <= idx < len(choices):
                name = choices[idx]["name"]
                if name in selected:
                    selected.discard(name)
                else:
                    selected.add(name)
        except ValueError:
            pass
    return [c["name"] for c in choices if c["name"] in selected]


def _fallback_confirm(message: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"\n{message} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def _fallback_text(message: str, default: str) -> str:
    hint = f" [{default}]" if default else ""
    raw = input(f"\n{message}{hint}: ").strip()
    return raw if raw else default


# ═══════════════════════════════════════════════════════════
# Progress and status indicators
# ═══════════════════════════════════════════════════════════

# ANSI colour codes (safe on Win11 VT100 and all modern terminals)
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_RESET = "\033[0m"

_CHECK = f"{_GREEN}\u2713{_RESET}"   # green checkmark
_WARN = f"{_YELLOW}\u26a0{_RESET}"   # yellow warning
_BULLET = f"{_CYAN}\u2022{_RESET}"   # cyan bullet


def header(title: str) -> None:
    """Print a bold section header."""
    print(f"\n{_BOLD}{title}{_RESET}", file=sys.stderr)


def step_banner(step: int, total: int, label: str) -> None:
    """Print a step progress banner: [2/6] Building knowledge model..."""
    bar = _progress_bar(step, total)
    print(f"\n{bar}  {_BOLD}[{step}/{total}]{_RESET} {label}", file=sys.stderr)


def done(message: str) -> None:
    """Print a checkmark completion line."""
    print(f"  {_CHECK} {message}", file=sys.stderr)


def warn(message: str) -> None:
    """Print a warning line."""
    print(f"  {_WARN} {message}", file=sys.stderr)


def info(message: str) -> None:
    """Print an info bullet."""
    print(f"  {_BULLET} {message}", file=sys.stderr)


def skip(message: str) -> None:
    """Print a dimmed skip line."""
    print(f"  {_DIM}- {message}{_RESET}", file=sys.stderr)


def _progress_bar(current: int, total: int, width: int = 20) -> str:
    """Render a compact progress bar: [=========>          ]"""
    filled = int(width * current / total) if total > 0 else 0
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    pct = int(100 * current / total) if total > 0 else 0
    return f"{_DIM}[{bar}] {pct}%{_RESET}"


class Spinner:
    """Animated spinner for long-running operations.

    Usage::

        with Spinner("Encoding 1,247 chunks"):
            do_slow_thing()
        # prints: spinner stops, replaced by checkmark line
    """

    _FRAMES = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827",
               "\u2807", "\u280f"]  # braille spinner

    def __init__(self, message: str) -> None:
        self.message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Spinner:
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        # Clear spinner line, print done
        sys.stderr.write(f"\r\033[K  {_CHECK} {self.message}\n")
        sys.stderr.flush()

    def _spin(self) -> None:
        frames = itertools.cycle(self._FRAMES)
        while not self._stop.is_set():
            frame = next(frames)
            sys.stderr.write(f"\r  {_CYAN}{frame}{_RESET} {self.message}")
            sys.stderr.flush()
            time.sleep(0.1)


def final_summary(lines: list[str]) -> None:
    """Print the boxed final summary."""
    width = max(len(line) for line in lines) + 4
    border = "\u2550" * width
    print(f"\n{_BOLD}\u2554{border}\u2557{_RESET}", file=sys.stderr)
    for line in lines:
        padded = line.ljust(width)
        print(f"{_BOLD}\u2551{_RESET}  {padded}{_BOLD}\u2551{_RESET}", file=sys.stderr)
    print(f"{_BOLD}\u255a{border}\u255d{_RESET}", file=sys.stderr)
