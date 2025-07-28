#!/bin/bash
#SBATCH -J evaluate_llada_reasoning
#SBATCH -o /home/hice1/qfitterey3/scratch/LLada-Reasoning/evaluate_logs/llada_sft/logs/eval4.out
#SBATCH -e /home/hice1/qfitterey3/scratch/LLada-Reasoning/evaluate_logs/llada_sft/logs/eval4.err
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:4         # GPU type (H100) and number of GPUs
#SBATCH -t4:00:00                        # Duration of the job
#SBATCH --mem=1024G
#SBATCH --cpus-per-task=8
#SBATCH --tmp=1000G

JOBID=$SLURM_JOB_ID


module load anaconda3

conda activate ~/scratch/envs/llada
cd /home/hice1/qfitterey3/scratch/LLada-Reasoning
export PYTHONPATH=$(pwd)

nvidia-smi

srun accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --mixed_precision bf16 \
  eval_llada.py \
    --tasks "gsm8k" \
    --limit 128 \
    --num_fewshot 0 \
    --model llada_dist \
    --batch_size 1 \
    --model_args 'model_path=/home/hice1/qfitterey3/scratch/LLada-Reasoning/llada_local_1.5,adapter_path=/home/hice1/qfitterey3/scratch/LLada-Reasoning/checkpoints/checkpoints_llada_nemotron_15/step-2100/sft_adapter,load_lora=True,cfg=0.0,is_check_greedy=False,mc_num=1,gen_length=8192'

