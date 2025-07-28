#!/bin/bash
#SBATCH -J llada_sft_run
#SBATCH -o /home/hice1/qfitterey3/scratch/LLada-Reasoning/checkpoints/sft_newdataset1/logs/sft_run3.out
#SBATCH -e /home/hice1/qfitterey3/scratch/LLada-Reasoning/checkpoints/sft_newdataset1/logs/sft_run3.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H200:4
#SBATCH -t 4:00:00                # Increased time limit for a 3-epoch run
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --tmp=1000G

# --- Environment Setup ---
module load anaconda3
conda activate ~/scratch/envs/llada
cd /home/hice1/qfitterey3/scratch/LLada-Reasoning
export PYTHONPATH=$(pwd)
nvidia-smi

# --- Training Launch Command ---
srun torchrun --nproc_per_node=4 --master_port=29405 sft_training/sft_train_new_dataset.py \
    --pretrained-model /home/hice1/qfitterey3/scratch/LLada-Reasoning/llada_local_1.5 \
    --sft-data ./filtered_conversational_dataset/ \
    --output-dir ./checkpoints/sft_newdataset1 \
    --lr 1e-4 \
    --epochs 3 \
    --local-batch 8 \
    --batch-size 64 \
    --warmup-ratio 0.1 \
    --lora-target-modules q_proj v_proj k_proj attn_out \
    --eps 1e-3 \
    --sft-lora-weights ./checkpoints/sft_newdataset1/step-1000/sft_adapter \
    --resume-from-checkpoint ./checkpoints/sft_newdataset1/step-1000/