#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CHECKPOINT="${1:-checkpoints/best.pt}"
python evaluate.py --config configs/default.yaml --checkpoint "$CHECKPOINT" "${@:2}"
