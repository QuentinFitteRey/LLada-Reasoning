#!/bin/bash

CHECKPOINT_DIR="/home/hice1/qfitterey3/scratch/LLada-Reasoning/checkpoints/checkpoints_llada_nemotron_15_4_goodlora"
JOB_SCRIPT_PATH="./script/sft_final_base4_goodlora.sbatch" # Path to the sbatch script from step 1

# --- Find the latest checkpoint ---
# This finds directories named "step-..." and sorts them numerically to get the last one.
LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -type d -name "step-*" | sort -V | tail -n 1)

RESUME_ARGS=""
if [[ -d "$LATEST_CHECKPOINT" ]]; then
    echo "✅ Found latest checkpoint: $LATEST_CHECKPOINT"
    # Set the arguments needed to resume training
    RESUME_ARGS="--resume-from-checkpoint $LATEST_CHECKPOINT --sft-lora-weights $LATEST_CHECKPOINT/sft_adapter"
else
    echo "👍 No checkpoint found. Starting a new training run."
fi

# --- Submit the job ---
# We pass the resume arguments directly to the sbatch command.
# The `sbatch` command outputs "Submitted batch job <ID>", we grab the ID.
JOB_ID=$(sbatch --parsable "$JOB_SCRIPT_PATH" $RESUME_ARGS)

echo "Submitted job $JOB_ID with args: $RESUME_ARGS"

# --- Chain the next job ---
# Submit this script to run again after the job we just submitted finishes for any reason.
# The next run will find the checkpoint created by the job we just launched.
if [[ -n "$JOB_ID" ]]; then
    sbatch --dependency=afterany:$JOB_ID "$0"
    echo "Chained next job to run after job $JOB_ID finishes."
else
    echo "❌ Failed to submit job. Stopping chain."
fi