#!/usr/bin/env bash
# Rebuild (or sync) a cartridge and regenerate the assistant primer.
# Usage: rebuild-and-prime.sh [cartridge] [sources...]
#
# Defaults:
#   cartridge = .rlat/project.rlat
#   sources   = ./docs ./src
#
# If the cartridge does not exist, runs `rlat build`.
# If it exists, runs `rlat sync` for incremental update.
# Then regenerates the summary primer.

set -euo pipefail

CARTRIDGE="${1:-.rlat/project.rlat}"
shift 2>/dev/null || true
SOURCES="${@:-./docs ./src}"

if [ ! -f "$CARTRIDGE" ]; then
    echo "Building new cartridge: $CARTRIDGE"
    mkdir -p "$(dirname "$CARTRIDGE")"
    rlat build $SOURCES -o "$CARTRIDGE"
else
    echo "Syncing cartridge: $CARTRIDGE"
    rlat sync "$CARTRIDGE" $SOURCES
fi

PRIMER=".claude/resonance-context.md"
mkdir -p "$(dirname "$PRIMER")"
echo "Generating primer: $PRIMER"
rlat summary "$CARTRIDGE" -o "$PRIMER"

echo "Done. Cartridge: $CARTRIDGE | Primer: $PRIMER"
