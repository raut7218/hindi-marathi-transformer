#!/bin/bash
# Colab single-T4 launcher.
#
# Usage:
#   bash scripts/launch_colab.sh mlm
#   bash scripts/launch_colab.sh clm
#   bash scripts/launch_colab.sh mt
#   bash scripts/launch_colab.sh train mlm
#   bash scripts/launch_colab.sh train clm
#   bash scripts/launch_colab.sh train mt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

export GPUS="${GPUS:-1}"
export CONFIG="${CONFIG:-configs/colab_t4.yaml}"

if [ $# -eq 1 ]; then
	case "$1" in
		mlm|clm|mt)
			exec bash "$REPO_ROOT/scripts/launch_distributed.sh" train "$1"
			;;
	esac
fi

exec bash "$REPO_ROOT/scripts/launch_distributed.sh" "$@"
