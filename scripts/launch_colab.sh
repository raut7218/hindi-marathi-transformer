#!/bin/bash
# Colab single-T4 launcher.
#
# Usage:
#   bash scripts/launch_colab.sh train mlm
#   bash scripts/launch_colab.sh train clm
#   bash scripts/launch_colab.sh train mt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

export GPUS="${GPUS:-1}"
export CONFIG="${CONFIG:-configs/colab_t4.yaml}"

exec bash "$REPO_ROOT/scripts/launch_distributed.sh" "$@"
