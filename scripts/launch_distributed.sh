#!/bin/bash
# Launch distributed training on multiple GPUs
# 
# Usage:
#   bash scripts/launch_distributed.sh train mt              # Train MT stage on 2 GPUs
#   bash scripts/launch_distributed.sh train mlm             # Train MLM stage on 2 GPUs
#   GPUS=4 bash scripts/launch_distributed.sh train mt       # Train on 4 GPUs
#   GPUS=1 bash scripts/launch_distributed.sh train mt       # Train on 1 GPU (fallback)

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

# Build the command
if [ "$GPUS" -gt 1 ]; then
    # Multi-GPU: use torchrun with DDP
    echo "[launcher] Starting distributed training on $GPUS GPUs"
    echo "[launcher] Command: torchrun --nproc_per_node=$GPUS scripts/train.py --stage $STAGE --config $CONFIG $EXTRA_ARGS"
    torchrun --nproc_per_node=$GPUS scripts/train.py --stage $STAGE --config $CONFIG $EXTRA_ARGS
else
    # Single GPU: fallback to standard training
    echo "[launcher] Starting single GPU training (GPUS=1)"
    echo "[launcher] Command: python scripts/train.py --stage $STAGE --config $CONFIG $EXTRA_ARGS"
    python scripts/train.py --stage $STAGE --config $CONFIG $EXTRA_ARGS
fi
