#!/usr/bin/env bash

# -------------------------------------------------------------------
# submit_evals.sh
#
# Submits a list of lm‑eval tasks in sequence, with per‐task overrides.
# -------------------------------------------------------------------

set -euo pipefail
shopt -s extglob

# ----------------------
# Common sbatch options
# ----------------------
SBATCH_OPTS="
  --nodes=1
  --ntasks-per-node=1
  --gres=gpu:H200:4
  --time=2:00:00
  --mem=1024G
  --cpus-per-task=8
  --tmp=1000G
  --mail-user=jmoutahir3@gatech.edu
  --mail-type=BEGIN,END
  --output=/home/hice1/jmoutahir3/scratch/LLada-Reasoning/evaluate_logs/llada_sft/full_logs/%x.out
  --error=/home/hice1/jmoutahir3/scratch/LLada-Reasoning/evaluate_logs/llada_sft/full_logs/%x.err
"

# -------------------
# List of all tasks
# -------------------
TASKS=( \
  gsm8k \
  # mmlu \
  # mmlu_pro \
  # arc_easy \
  # arc_challenge \
  # hellaswag \
  # hendrycks_math \
  # humaneval \
  # ifeval \
  # gpqa \
  # mbpp \
)

# -----------------------------
# Default per‐task parameters
# -----------------------------
DEFAULT_LIMIT=64
DEFAULT_FEWSHOT=5
DEFAULT_BATCHSIZE=4
DEFAULT_CONFIRM_UNSAFE=""      # e.g. "--confirm_run_unsafe_code"
DEFAULT_MODEL_ARGS="model_path=/home/hice1/jmoutahir3/scratch/LLada-Reasoning/llada_local_1.5,cfg=0.,is_check_greedy=False,mc_num=128,gen_length=256,steps=256,block_length=16,temperature=0.0,generate_batch_size=${DEFAULT_BATCHSIZE}"
# MODEL_PATH="/home/hice1/jmoutahir3/scratch/LLaDA_checkpoints/merged_pretrained_model/merged_model_good_base"
# ADAPTER_PATH="/home/hice1/jmoutahir3/scratch/LLaDA_checkpoints/sft/exp_quentin_1807/new_weights/step-2100/sft_adapter"
# DEFAULT_MODEL_ARGS="model_path=${MODEL_PATH},adapter_path=${ADAPTER_PATH},load_lora=True,cfg=0.5,is_check_greedy=False,mc_num=5,gen_length=2048,steps=2048,block_length=128,temperature=0.0"

# -------------------------------
# Overrides for specific tasks
# -------------------------------
# How many examples of few‑shot
declare -A OVERRIDE_FEWSHOT=(
  [mbpp]=3
)

# Limit (number of docs) to run—e.g. you may want fewer examples for slow tasks
declare -A OVERRIDE_LIMIT=(
  [mmlu_pro]=8
  [mbpp]=8
  [gpqa]=8
)

# Tasks flagged unsafe (HumanEval, MBPP)
declare -A OVERRIDE_CONFIRM=(
  [humaneval]="--confirm_run_unsafe_code"
  [mbpp]="--confirm_run_unsafe_code"
)

# -------------------------------------
# Environment setup for code‐eval tasks
# -------------------------------------
export HF_ALLOW_CODE_EVAL=1

prev_jobid=""

for T in "${TASKS[@]}"; do
  echo "→ Submitting evaluation for task: $T"

  # fetch overrides or fall back to defaults
  LIMIT=${OVERRIDE_LIMIT[$T]:-$DEFAULT_LIMIT}
  FEWSHOT=${OVERRIDE_FEWSHOT[$T]:-$DEFAULT_FEWSHOT}
  CONFIRM=${OVERRIDE_CONFIRM[$T]:-$DEFAULT_CONFIRM_UNSAFE}
  BATCHSIZE=$DEFAULT_BATCHSIZE

  # build the common part of the command
  MODEL_ARGS="$DEFAULT_MODEL_ARGS"

  RUN_CMD="\
module load anaconda3 && \
conda activate ~/scratch/envs/llada && \
cd /home/hice1/jmoutahir3/scratch/LLada-Reasoning && \
export PYTHONPATH=\$(pwd) && \
nvidia-smi && \
srun accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --mixed_precision bf16 \
  eval_llada.py \
    --tasks ${T} \
    --num_fewshot ${FEWSHOT} \
    --model llada_dist \
    --batch_size ${BATCHSIZE} \
    ${CONFIRM} \
    --model_args '${MODEL_ARGS}' \
  "
# Print the command for debugging
  echo "Running command: $RUN_CMD"

  # submit with or without dependency
  if [ -z "$prev_jobid" ]; then
    sbatch $SBATCH_OPTS \
      --job-name=evaluate_${T} \
      --wrap="$RUN_CMD"
  else
    sbatch $SBATCH_OPTS \
      --dependency=afterany:${prev_jobid} \
      --job-name=evaluate_${T} \
      --wrap="$RUN_CMD"
  fi

  # capture the job ID of the last submission
  # (safer than grepping squeue; grabs sbatch's stdout)
  sbatch_out=$(squeue --me --noheader --format="%i %j" | grep "evaluate_${T}" | tail -n1)
  prev_jobid=${sbatch_out%% *}

done

echo "All evaluation jobs submitted; last job ID = $prev_jobid"