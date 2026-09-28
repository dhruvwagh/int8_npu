#!/usr/bin/env bash
set -euo pipefail

# Convenience simulation script for Milestone 4 (Ping-Pong Buffers & Fetch Engine)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
python3 "${REPO_ROOT}/scripts/run_buffer_test.py" "$@"
