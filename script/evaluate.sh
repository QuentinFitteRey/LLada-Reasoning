#!/usr/bin/env bash

# -------------------------------------------------------------------
# submit_evals.sh
#
# Submits GSM8K → MMLU → ARC_Challenge evaluations in sequence.
# Uses sbatch dependencies so each job waits for the prior one.
# -------------------------------------------------------------------

# Common sbatch options
SBATCH_OPTS="
  --nodes=1
  --ntasks-per-node=1
  --gres=gpu:H200:4
  --time=4:00:00
  --mem=1024G
  --cpus-per-task=8
  --tmp=1000G
  --mail-user=jmoutahir3@gatech.edu
  --mail-type=BEGIN,END
  --output=/home/hice1/jmoutahir3/scratch/LLada-Reasoning/evaluate_logs/llada_sft/full_logs/%x.out
  --error=/home/hice1/jmoutahir3/scratch/LLada-Reasoning/evaluate_logs/llada_sft/full_logs/%x.err
"

# The three tasks to run, in order:
TASKS=(gsm8k mmlu mmlu_pro arc_easy arc_challenge hellaswag hendrycks_math humaneval ifeval gpqa mbpp)

MODEL_PATH="/home/hice1/jmoutahir3/scratch/LLaDA_checkpoints/merged_pretrained_model/merged_model_good_base"
ADAPTER_PATH="/home/hice1/jmoutahir3/scratch/LLaDA_checkpoints/sft/exp_quentin_1807/new_weights/step-2100/sft_adapter"

prev_jobid=""

# Set HF_ALLOW_CODE_EVAL=1 to allow code evaluation for some tasks
export HF_ALLOW_CODE_EVAL=1

for T in "${TASKS[@]}"; do
  echo "Submitting evaluation for task: $T"

  # Build the srun command (identical for each task except --tasks)
  RUN_CMD="module load anaconda3 \
    && conda activate ~/scratch/envs/llada \
    && cd /home/hice1/jmoutahir3/scratch/LLada-Reasoning \
    && export PYTHONPATH=\$(pwd) \
    && nvidia-smi \
    && srun accelerate launch \
         --multi_gpu \
         --num_processes 4 \
         --mixed_precision bf16 \
         eval_llada.py \
           --tasks ${T} \
           --limit 32 \
           --num_fewshot 5 \
           --model llada_dist \
           --batch_size 1 \
           --model_args 'model_path=${MODEL_PATH},adapter_path=${ADAPTER_PATH},load_lora=True,cfg=0.5,is_check_greedy=False,mc_num=5,gen_length=2048,steps=2048,block_length=128,temperature=0.0'"

  if [ -z "$prev_jobid" ]; then
    # first job, no dependency
    sbatch $SBATCH_OPTS \
      --job-name=evaluate_${T} \
      --wrap="$RUN_CMD"
  else
    # subsequent jobs, wait for previous to finish successfully
    sbatch $SBATCH_OPTS \
      --dependency=afterany:${prev_jobid} \
      --job-name=evaluate_${T} \
      --wrap="$RUN_CMD"
  fi

  # capture the job ID of whatever we just submitted:
  # sbatch prints: "Submitted batch job 123456"
  new_jobid=$(squeue --me --noheader --format="%i %j" | grep "evaluate_${T}" | awk '{print $1}')
  prev_jobid=$new_jobid
done

echo "All evaluation jobs submitted. Final job will be ID $prev_jobid"