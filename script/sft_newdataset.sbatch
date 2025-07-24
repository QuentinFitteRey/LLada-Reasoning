#!/bin/bash
#SBATCH -J llada_sft_run
#SBATCH -o /home/hice1/qfitterey3/scratch/LLada-Reasoning/checkpoints/sft_newdataset/logs/sft_run.out
#SBATCH -e /home/hice1/qfitterey3/scratch/LLada-Reasoning/checkpoints/sft_newdataset/logs/sft_run.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H200:8
#SBATCH -t 2:00:00                # Increased time limit for a 3-epoch run
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
srun torchrun --nproc_per_node=8 --master_port=29405 sft_training/sft_train_new_dataset.py \
    --pretrained-model /home/hice1/qfitterey3/scratch/LLada-Reasoning/llada_local_1.5 \
    --sft-data ./filtered_conversational_dataset/ \
    --output-dir ./checkpoints/sft_newdataset \
    --lr 1e-5 \
    --epochs 3 \
    --local-batch 8 \
    --batch-size 128 \
    --warmup-ratio 0.1 \
    --lora-target-modules q_proj v_proj k_proj attn_out \
    --eps 1e-3
    
    # --- Optional flags for resuming a run ---
    # --sft-lora-weights ./checkpoints/checkpoints_llada_nemotron_15_4_goodlora/step-XXXX/ \
    # --resume-from-checkpoint ./checkpoints/checkpoints_llada_nemotron_15_4_goodlora/step-XXXX/