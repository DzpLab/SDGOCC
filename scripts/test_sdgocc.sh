#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-projects/configs/sdgocc/sdgocc-r50_4.py}"
CHECKPOINT="${CHECKPOINT:-./ckpts/best_iou_635_4.pth}"

python tools/test.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --eval mAP "$@"
