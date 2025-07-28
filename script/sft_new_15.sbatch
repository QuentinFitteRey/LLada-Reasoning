#!/bin/bash
#SBATCH -J llada_reasoning_sft
#SBATCH -o /home/hice1/qfitterey3/scratch/LLada-Reasoning/checkpoints/llada_sft_15_FINAL/logs/SlurmTraining_follow.out
#SBATCH -e /home/hice1/qfitterey3/scratch/LLada-Reasoning/checkpoints/llada_sft_15_FINAL/logs/SlurmTraining_follow.err
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:8         # GPU type (H100) and number of GPUs
#SBATCH -t2:00:00                        # Duration of the job
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --tmp=1000G

module load anaconda3
conda activate ~/scratch/envs/llada
cd /home/hice1/qfitterey3/scratch/LLada-Reasoning
export PYTHONPATH=$(pwd)
nvidia-smi
srun torchrun --nproc_per_node=8 --master_port=29405 sft_training/sft_train_15.py