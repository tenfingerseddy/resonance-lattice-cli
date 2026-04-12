#!/usr/bin/env bash
# Search a cartridge and output context-format results for LLM injection.
# Usage: search-then-inject.sh <cartridge> <query> [mode]
#
# Defaults:
#   mode = augment
#
# Uses --no-worker since this is typically called from scripts or subagents.
# Output is context format, ready for piping into an LLM prompt.

set -euo pipefail

CARTRIDGE="${1:?Usage: search-then-inject.sh <cartridge> <query> [mode]}"
QUERY="${2:?Usage: search-then-inject.sh <cartridge> <query> [mode]}"
MODE="${3:-augment}"

rlat search "$CARTRIDGE" "$QUERY" \
    --format context \
    --mode "$MODE" \
    --no-worker
