#!/usr/bin/env bash
set -euo pipefail

# Convenience simulation script for Milestone 6 (Requantization Epilogue & Output Buffer)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
python3 "${REPO_ROOT}/scripts/run_epilogue_test.py" "$@"
