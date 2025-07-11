#!/bin/bash
#SBATCH -J evaluate_llada_reasoning
#SBATCH -o /home/hice1/jmoutahir3/scratch/LLada-Reasoning/evaluate_logs/llada_1.5/logs/eval.out
#SBATCH -e /home/hice1/jmoutahir3/scratch/LLada-Reasoning/evaluate_logs/llada_1.5/logs/eval.err
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:4         # GPU type (H100) and number of GPUs
#SBATCH -t2:00:00                        # Duration of the job
#SBATCH --mem=1024G
#SBATCH --cpus-per-task=8
#SBATCH --tmp=1000G

JOBID=$SLURM_JOB_ID

echo "SFT Training: Job $JOBID has started!" | mail -s "SLURM Job Started!" jmoutahir3@gatech.edu

module load anaconda3
conda activate ~/scratch/envs/llada
cd /home/hice1/jmoutahir3/scratch/LLada-Reasoning
export PYTHONPATH=$(pwd)

srun accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --mixed_precision bf16 \
  eval_llada.py \
    --tasks mmlu_llama \
    --limit 16 \
    --num_fewshot 5 \
    --model llada_dist \
    --batch_size 1 \
    --model_args "model_path=./llada_local_1.5,\
load_lora=False,\
cfg=0.0,is_check_greedy=False,mc_num=1,max_gen_toks=1024,"\