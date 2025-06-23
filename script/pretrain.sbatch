#!/bin/bash
#SBATCH -J pretrain_llada_reasoning
#SBATCH -o /home/hice1/qfitterey3/scratch/LLada-Reasoning/checkpoints_llada_pretrain_8k_first/logs/SlurmTraining.out
#SBATCH -e /home/hice1/qfitterey3/scratch/LLada-Reasoning/checkpoints_llada_pretrain_8k_first/logs/SlurmTraining.err
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:4         # GPU type (H100) and number of GPUs
#SBATCH -t4:00:00                        # Duration of the job
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --tmp=1000G

module load anaconda3
conda activate ~/scratch/envs/llada
cd /home/hice1/qfitterey3/scratch/LLada-Reasoning
srun torchrun --nproc_per_node=4 pretraining/pretrain_llada_extended.py 