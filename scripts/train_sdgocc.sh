#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-projects/configs/sdgocc/sdgocc-r50_4.py}"
WORK_DIR="${WORK_DIR:-./work_dirs/sdgocc_r50_4}"

python tools/train.py --config "$CONFIG" --work-dir "$WORK_DIR" "$@"
