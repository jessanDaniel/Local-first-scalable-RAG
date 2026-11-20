#!/usr/bin/env bash
# One command for "is this commit ready?": lint, format, tests. Same check a hook or CI would run.
#
#   scripts/check.sh          # check only
#   scripts/check.sh --fix    # apply ruff's safe fixes and formatting first, then check
#
# Needs nothing running: the tests are offline (no GROBID, no Ollama, no downloads).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--fix" ]]; then
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

if [[ $failed -ne 0 ]]; then
    echo "check failed"
    exit 1
fi
echo "all checks passed"
