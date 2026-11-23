#!/usr/bin/env bash
# One command for "is this commit ready?": lint, format, tests. Same check a hook or CI would run.
#
#   scripts/check.sh          # check only
#   scripts/check.sh --fix    # apply ruff's safe fixes and formatting first, then check
#   scripts/check.sh --fast   # skip the quality gate (lint, format and tests only)
#
# Needs nothing running: the tests are offline (no GROBID, no Ollama, no downloads). The quality gate
# uses the local database, indexes and stored answers, and skips itself where they don't exist.
set -euo pipefail
cd "$(dirname "$0")/.."

fix=0
fast=0
for arg in "$@"; do
    case "$arg" in
        --fix) fix=1 ;;
        --fast) fast=1 ;;
        *) echo "unknown option $arg (use --fix, --fast)"; exit 2 ;;
    esac
done
if [[ $fix -eq 1 ]]; then
    uv run ruff check . --fix --quiet || true
    uv run ruff format . --quiet
fi

failed=0
step() {
    local name=$1
    shift
    local start=$SECONDS
    if "$@"; then
        echo "ok    $name ($((SECONDS - start))s)"
    else
        echo "FAIL  $name"
        failed=1
    fi
}

step "lint (ruff check)" uv run ruff check . --quiet
step "format (ruff format --check)" uv run ruff format --check . --quiet
step "tests (pytest)" uv run pytest -q --no-header -p no:cacheprovider
if [[ $fast -eq 0 ]]; then
    # Retrieval + guardrail replay vs eval/baseline.json, zero tolerance (~10-20 s; needs local data).
    step "quality gate (rag eval gate)" uv run rag eval gate
fi

if [[ $failed -ne 0 ]]; then
    echo "check failed"
    exit 1
fi
echo "all checks passed"
