#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

RESULT_DIR="${1:-./work_dirs/sdgocc_r50_4/results}"
ROOT_PATH="${ROOT_PATH:-./data/nuscenes}"
SAVE_PATH="${SAVE_PATH:-./vis}"

python tools/analysis_tools/vis_occ.py "$RESULT_DIR" --root_path "$ROOT_PATH" --save_path "$SAVE_PATH"
