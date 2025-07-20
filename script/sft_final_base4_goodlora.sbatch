#!/bin/bash
#SBATCH -J llada_reasoning_sft
#SBATCH -o /home/hice1/qfitterey3/scratch/LLada-Reasoning/checkpoints/checkpoints_llada_nemotron_15_4_goodlora/logs/SlurmTraining_follow2.out
#SBATCH -e /home/hice1/qfitterey3/scratch/LLada-Reasoning/checkpoints/checkpoints_llada_nemotron_15_4_goodlora/logs/SlurmTraining_follow2.err
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
srun torchrun --nproc_per_node=8 --master_port=29405 sft_training/sft_train_new_base.py \
    --pretrained-model /home/hice1/qfitterey3/scratch/LLada-Reasoning/llada_local_1.5 \
    --output-dir ./checkpoints/checkpoints_llada_nemotron_15_4_goodlora \
    --lr 2e-5 \
    --batch-size 64 \
    --lora-target-modules q_proj v_proj k_proj attn_out \
    --eps 0.1 \
    --sft-lora-weights /home/hice1/qfitterey3/scratch/LLada-Reasoning/checkpoints/checkpoints_llada_nemotron_15_4_goodlora/step-1200/sft_adapter \
    --resume-from-checkpoint /home/hice1/qfitterey3/scratch/LLada-Reasoning/checkpoints/checkpoints_llada_nemotron_15_4_goodlora/step-1200 \
