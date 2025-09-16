#!/usr/bin/env bash

# Portable evaluation launcher (was SLURM .sbatch). Works on: local machine, SLURM cluster, other schedulers.
# Configuration via environment variables (override when calling, e.g. TASKS=boolq bash evaluate.sbatch):
#   MODEL_PATH   : HF hub id or local directory (default: modelling_final)
#   TASKS        : comma separated lm-eval tasks (default: gsm8k)
#   FEWSHOT      : few-shot examples (default: 5)
#   BATCH_SIZE   : batch size per process (default: 1)
#   EXTRA_ARGS   : extra model_args entries (comma separated key=value)
#   NUM_GPUS     : explicit world size (auto-detected if unset)
#   DRY_RUN=1    : print command & exit (set before script invocation or export)

set -euo pipefail

# Optional conda activation (skip if already in env)
if command -v conda >/dev/null 2>&1 && [[ -d "$HOME/scratch/envs/llada" ]]; then
  # shellcheck disable=SC1091
  conda activate "$HOME/scratch/envs/llada" || true
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

mkdir -p logs/evaluate

: "${MODEL_PATH:=modelling_final}"  # tracked model implementation (ensure weights available)
: "${TASKS:=gsm8k}"
: "${FEWSHOT:=5}"
: "${BATCH_SIZE:=1}"
: "${EXTRA_ARGS:=cfg=0.0,is_check_greedy=False,mc_num=64,gen_length=256,steps=256,block_length=16,temperature=0.0}"

if [[ -z "${NUM_GPUS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  else
    NUM_GPUS=1
  fi
fi

# Detect if running under SLURM & srun available
if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  LAUNCH_PREFIX=(srun)
else
  LAUNCH_PREFIX=()
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

echo "[EVAL] MODEL_PATH=$MODEL_PATH TASKS=$TASKS FEWSHOT=$FEWSHOT BATCH_SIZE=$BATCH_SIZE GPUS=$NUM_GPUS"
echo "[EVAL] EXTRA_ARGS=$EXTRA_ARGS"

if ! command -v accelerate >/dev/null 2>&1; then
  echo "[WARN] 'accelerate' not found on PATH. Install with: pip install accelerate" >&2
  exit 1
fi

CMD=(accelerate launch --multi_gpu --num_processes "$NUM_GPUS" --mixed_precision bf16 \
  eval_llada.py --tasks "$TASKS" --num_fewshot "$FEWSHOT" --model llada_dist \
  --batch_size "$BATCH_SIZE" --model_args "model_path=$MODEL_PATH,load_lora=False,$EXTRA_ARGS")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[DRY_RUN] '; printf '%q ' "${LAUNCH_PREFIX[@]}" "${CMD[@]}"; echo
  exit 0
fi

"${LAUNCH_PREFIX[@]}" "${CMD[@]}"