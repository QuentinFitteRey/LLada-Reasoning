#!/usr/bin/env bash
# Portable SFT (variant 2) training launcher
# Differs in batch sizing and target modules.
# Environment variables (see sft_newdataset.sbatch for descriptions) plus:
#   SFT_LORA_TARGETS default adds ff_proj up_proj ff_out

set -euo pipefail

if command -v conda >/dev/null 2>&1 && [[ -d "$HOME/scratch/envs/llada" ]]; then
    conda activate "$HOME/scratch/envs/llada" || true
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

: "${SFT_MODEL:=modelling_final}"
: "${SFT_DATA:=./filtered_conversational_dataset}"
: "${SFT_OUTPUT:=checkpoints/sft_newdataset2}"
: "${SFT_EPOCHS:=3}"
: "${SFT_LR:=1e-5}"
: "${SFT_LOCAL_BATCH:=6}"
: "${SFT_GLOBAL_BATCH:=96}"
: "${SFT_LORA_TARGETS:=q_proj v_proj k_proj attn_out ff_proj up_proj ff_out}"
: "${SFT_LORA_WEIGHTS:=}"

mkdir -p "$SFT_OUTPUT/logs"

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

if ! command -v torchrun >/dev/null 2>&1; then
    echo "[ERR] torchrun not found." >&2
    exit 1
fi

CMD=(torchrun --nproc_per_node="$NUM_GPUS" --master_port=29406 sft_training/sft_train_new_dataset.py \
    --pretrained-model "$SFT_MODEL" \
    --sft-data "$SFT_DATA" \
    --output-dir "$SFT_OUTPUT" \
    --lr "$SFT_LR" \
    --epochs "$SFT_EPOCHS" \
    --local-batch "$SFT_LOCAL_BATCH" \
    --batch-size "$SFT_GLOBAL_BATCH" \
    --warmup-ratio 0.1 \
    --lora-target-modules $SFT_LORA_TARGETS \
    --eps 1e-3 )

if [[ -n "$SFT_LORA_WEIGHTS" ]]; then
    CMD+=( --sft-lora-weights "$SFT_LORA_WEIGHTS" )
fi

echo "[SFT2] MODEL=$SFT_MODEL DATA=$SFT_DATA OUT=$SFT_OUTPUT GPUS=$NUM_GPUS" >&2
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[DRY_RUN] '; printf '%q ' "${CMD[@]}"; echo
    exit 0
fi

"${CMD[@]}"
    --resume-from-checkpoint ./checkpoints/sft_newdataset2/step-600/