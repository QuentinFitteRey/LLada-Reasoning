#!/usr/bin/env bash
# Portable pretraining launcher (originally SLURM). Configure via env vars:
#   PRETRAIN_OUTPUT   : output directory (default: checkpoints/pretrain_run)
#   PRETRAIN_MODEL    : base model path/id (default: modelling_final)
#   PRETRAIN_ARGS     : extra args appended to pretraining script
#   NUM_GPUS          : world size (auto-detect if unset)
#   NPROC_PER_NODE    : processes per node (defaults to NUM_GPUS)
#   DRY_RUN=1         : print command without executing

set -euo pipefail

if command -v conda >/dev/null 2>&1 && [[ -d "$HOME/scratch/envs/llada" ]]; then
	conda activate "$HOME/scratch/envs/llada" || true
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

: "${PRETRAIN_OUTPUT:=checkpoints/pretrain_run}"
: "${PRETRAIN_MODEL:=modelling_final}"
: "${PRETRAIN_ARGS:=}"

mkdir -p "$PRETRAIN_OUTPUT/logs"

if [[ -z "${NUM_GPUS:-}" ]]; then
	if command -v nvidia-smi >/dev/null 2>&1; then
		NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
	else
		NUM_GPUS=1
	fi
fi
: "${NPROC_PER_NODE:=$NUM_GPUS}"

if ! command -v torchrun >/dev/null 2>&1; then
	echo "[WARN] torchrun not found; install PyTorch to proceed." >&2
	exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
	nvidia-smi || true
fi

LAUNCH=(torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port=29501 pretraining/pretrain_llada_extended.py \
	--pretrained-model "$PRETRAIN_MODEL" \
	--output-dir "$PRETRAIN_OUTPUT" )

if [[ -n "$PRETRAIN_ARGS" ]]; then
	# shellcheck disable=SC2206
	EXTRA=( $PRETRAIN_ARGS )
	LAUNCH+=( "${EXTRA[@]}" )
fi

echo "[PRETRAIN] GPUS=$NUM_GPUS OUT=$PRETRAIN_OUTPUT MODEL=$PRETRAIN_MODEL" >&2
if [[ "${DRY_RUN:-0}" == "1" ]]; then
	printf '[DRY_RUN] '; printf '%q ' "${LAUNCH[@]}"; echo
	exit 0
fi

"${LAUNCH[@]}"
srun torchrun --nproc_per_node=4  --master_port=29501 pretraining/pretrain_llada_extended.py 