#!/bin/bash
#SBATCH -J evaluate_llada_reasoning
#SBATCH -o /home/hice1/jmoutahir3/scratch/LLada-Reasoning/evaluate_logs/llada_sft/logs/eval_2207.out
#SBATCH -e /home/hice1/jmoutahir3/scratch/LLada-Reasoning/evaluate_logs/llada_sft/logs/eval_2207.err
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:4         # GPU type (H100) and number of GPUs
#SBATCH -t4:00:00                        # Duration of the job
#SBATCH --mem=1024G
#SBATCH --cpus-per-task=8
#SBATCH --tmp=1000G

JOBID=$SLURM_JOB_ID

module load anaconda3

conda activate ~/scratch/envs/llada
cd /home/hice1/jmoutahir3/scratch/LLada-Reasoning
export PYTHONPATH=$(pwd)

nvidia-smi

srun accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --mixed_precision bf16 \
  eval_llada.py \
    --tasks gsm8k \
    --num_fewshot 5 \
    --model llada_dist \
    --batch_size 1 \
    --model_args 'model_path=/home/hice1/jmoutahir3/scratch/LLada-Reasoning/llada_local_1.5,load_lora=False,cfg=0.0,is_check_greedy=False,mc_num=128,gen_length=256,steps=256,block_length=16,temperature=0.0'
    # --model_args 'model_path=/home/hice1/jmoutahir3/scratch/LLaDA_checkpoints/merged_pretrained_model/merged_model_good_base,adapter_path=/home/hice1/jmoutahir3/scratch/LLaDA_checkpoints/sft/exp_quentin_1807/new_weights/step-2100/sft_adapter,load_lora=True,cfg=0.0,is_check_greedy=False,mc_num=128,gen_length=1024,steps=1024,block_length=16,temperature=0.0'