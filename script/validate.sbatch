#!/usr/bin/env bash
# Portable validation launcher
#   VAL_MODEL      : model path/id (default: modelling_final)
#   VAL_DATA       : optional validation data path
#   VAL_OUTPUT     : output directory (default: validation)
#   NUM_GPUS       : world size (auto-detect)
#   DRY_RUN=1      : print command only

set -euo pipefail

if command -v conda >/dev/null 2>&1 && [[ -d "$HOME/scratch/envs/llada" ]]; then
	conda activate "$HOME/scratch/envs/llada" || true
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

: "${VAL_MODEL:=modelling_final}"
: "${VAL_DATA:=}"
: "${VAL_OUTPUT:=validation}"
mkdir -p "$VAL_OUTPUT/logs"

if [[ -z "${NUM_GPUS:-}" ]]; then
	if command -v nvidia-smi >/dev/null 2>&1; then
		NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
	else
		NUM_GPUS=1
	fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
	nvidia-smi || true
fi

CMD=(python pretraining/validate.py --model "$VAL_MODEL" --output "$VAL_OUTPUT")
if [[ -n "$VAL_DATA" ]]; then
	CMD+=( --data "$VAL_DATA" )
fi

echo "[VALIDATE] MODEL=$VAL_MODEL DATA=${VAL_DATA:-<default>} GPUS=$NUM_GPUS OUT=$VAL_OUTPUT" >&2
if [[ "${DRY_RUN:-0}" == "1" ]]; then
	printf '[DRY_RUN] '; printf '%q ' "${CMD[@]}"; echo
	exit 0
fi

"${CMD[@]}"
srun torchrun --nproc_per_node=4  --master_port=29501 pretraining/validate.py