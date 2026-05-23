#!/bin/bash
# Launch distributed training on multiple GPUs
# 
# Usage:
#   bash scripts/launch_distributed.sh train mt              # Train MT stage on 2 GPUs
#   bash scripts/launch_distributed.sh train mlm             # Train MLM stage on 2 GPUs
#   GPUS=4 bash scripts/launch_distributed.sh train mt       # Train on 4 GPUs
#   GPUS=1 bash scripts/launch_distributed.sh train mt       # Train on 1 GPU (fallback)

# Get the absolute directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TRAIN_SCRIPT="$REPO_ROOT/scripts/train.py"

# Configuration
GPUS=${GPUS:-2}  # Number of GPUs (default: 2 for Kaggle T4)
CONFIG=${CONFIG:-configs/part2_t4.yaml}

# Parse command-line arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <command> <stage> [other args...]"
    echo "  command: 'train' or 'eval'"
    echo "  stage: 'mlm', 'clm', or 'mt'"
    echo "  other args: passed to train.py"
    exit 1
fi

COMMAND=$1
STAGE=$2
shift 2
EXTRA_ARGS="$@"

# Make config path absolute if it's relative
if [[ ! "$CONFIG" = /* ]]; then
    CONFIG="$REPO_ROOT/$CONFIG"
fi

# Verify train script exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "[launcher] ERROR: Train script not found at $TRAIN_SCRIPT"
    echo "[launcher] SCRIPT_DIR=$SCRIPT_DIR"
    echo "[launcher] REPO_ROOT=$REPO_ROOT"
    exit 1
fi

# Change to repo root for proper imports
cd "$REPO_ROOT" || exit 1

# Build and execute the command
if [ "$GPUS" -gt 1 ]; then
    # Multi-GPU: use torchrun with DDP
    echo "[launcher] Starting distributed training on $GPUS GPUs"
    echo "[launcher] Repo root: $REPO_ROOT"
    echo "[launcher] Train script: $TRAIN_SCRIPT"
    echo "[launcher] Config: $CONFIG"
    echo "[launcher] Working directory: $(pwd)"
    echo "[launcher] Command: torchrun --nproc_per_node=$GPUS '$TRAIN_SCRIPT' --stage $STAGE --config '$CONFIG' $EXTRA_ARGS"
    torchrun --nproc_per_node="$GPUS" "$TRAIN_SCRIPT" --stage "$STAGE" --config "$CONFIG" $EXTRA_ARGS
else
    # Single GPU: fallback to standard training
    echo "[launcher] Starting single GPU training (GPUS=1)"
    echo "[launcher] Repo root: $REPO_ROOT"
    echo "[launcher] Train script: $TRAIN_SCRIPT"
    echo "[launcher] Config: $CONFIG"
    echo "[launcher] Working directory: $(pwd)"
    echo "[launcher] Command: python '$TRAIN_SCRIPT' --stage $STAGE --config '$CONFIG' $EXTRA_ARGS"
    python "$TRAIN_SCRIPT" --stage "$STAGE" --config "$CONFIG" $EXTRA_ARGS
fi
