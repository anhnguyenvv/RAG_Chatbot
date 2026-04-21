#!/usr/bin/env bash
# Thin wrapper around smoke_test.py for CI / ops use.
#
#   ./scripts/smoke.sh                            # local, http://localhost:8000
#   ./scripts/smoke.sh https://staging.example    # pass base URL as arg 1
#   SMOKE_ADMIN_KEY=xxx ./scripts/smoke.sh ...    # include admin checks
#
# Exit code 0 = all pass, 1 = at least one failure.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="${1:-${SMOKE_BASE_URL:-http://localhost:8000}}"

ARGS=(--base-url "$BASE_URL")
if [[ -n "${SMOKE_ADMIN_KEY:-}" ]]; then
  ARGS+=(--admin-key "$SMOKE_ADMIN_KEY")
fi
if [[ -n "${SMOKE_TIMEOUT:-}" ]]; then
  ARGS+=(--timeout "$SMOKE_TIMEOUT")
fi
if [[ "${SMOKE_SKIP_RATE_LIMIT:-0}" == "1" ]]; then
  ARGS+=(--skip-rate-limit)
fi
if [[ "${SMOKE_JSON:-0}" == "1" ]]; then
  ARGS+=(--json)
fi

exec python "$SCRIPT_DIR/smoke_test.py" "${ARGS[@]}"
