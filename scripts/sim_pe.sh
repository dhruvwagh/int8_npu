#!/usr/bin/env bash
set -euo pipefail

# Quick simulation script for Milestone 1 npu_pe
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
python3 "${REPO_ROOT}/scripts/run_pe_test.py" "$@"
