#!/usr/bin/env bash
set -euo pipefail

# Convenience simulation script for Milestone 2 (4x4 Systolic Array)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
python3 "${REPO_ROOT}/scripts/run_array_4x4_test.py" "$@"
