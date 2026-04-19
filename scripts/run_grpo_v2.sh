#!/bin/bash
# Kick off GRPO v2 retrain with bumped config (8192 completion, 8 generations, 3 epochs).
# Preserves the v1 adapter by renaming it before overwrite.
# Run after the interaction-scaling eval has completed.
set -euo pipefail

cd "$(dirname "$0")/.."

OLD_ADAPTER="models/qwen3-8b-interaction-scaling-grpo"
BACKUP_ADAPTER="models/qwen3-8b-interaction-scaling-grpo-v1"

if [[ -d "$OLD_ADAPTER" && ! -d "$BACKUP_ADAPTER" ]]; then
    echo "[run_grpo_v2] Backing up v1 adapter: $OLD_ADAPTER -> $BACKUP_ADAPTER"
    mv "$OLD_ADAPTER" "$BACKUP_ADAPTER"
fi

mkdir -p logs
echo "[run_grpo_v2] Starting GRPO retrain; logging to logs/grpo_v2.log"
STAGE=grpo PYTHONPATH=. python3 -m src.training.train_grpo grpo > logs/grpo_v2.log 2>&1
echo "[run_grpo_v2] Done."
