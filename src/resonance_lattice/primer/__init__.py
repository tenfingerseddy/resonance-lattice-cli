# SPDX-License-Identifier: BUSL-1.1
"""Primer-generation helpers used by `rlat summary` and `rlat memory primer`.

These modules are deliberately small and dependency-light: each one takes
external signal (git history, memory-tier timestamps) and returns plain
data the primer commands can fold into their existing bootstrap +
section-rendering pipelines.
"""
