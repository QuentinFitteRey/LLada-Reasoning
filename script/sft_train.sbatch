#!/bin/bash
#SBATCH -J sft_llada_reasoning
#SBATCH -o ./checkpoints/checkpoints_llada_sft_train/logs/SlurmTraining.out
#SBATCH -e ./checkpoints/checkpoints_llada_sft_train/logs/SlurmTraining.err
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:4
#SBATCH -t4:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --tmp=1000G

module load anaconda3
conda activate ~/scratch/envs/llada
cd /home/hice1/jmoutahir3/scratch/LLada-Reasoning

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get the job ID
JOBID=$SLURM_JOB_ID
echo "SFT Training: Job $JOBID has started!" | mail -s "SLURM Job Started!" jmoutahir3@gatech.edu
echo "SFT Training: Job $JOBID has started!" | mail -s "SLURM Job Started!" moutahirjed@gmail.com

export PYTHONPATH=$PWD

srun torchrun --nproc_per_node=4 -m sft_training.sft_train \
  --sft-data            ./sft_training/sft_data/combined \
  --pretrained-model    ./llada_local \
  --output-dir          ./checkpoints/sft_lora_run \
  --use-lora \
  --lora-resume         /home/hice1/jmoutahir3/scratch/LLaDA_checkpoints/new_test_checkpoint \
  --batch-size          256 \
  --local-batch         2 \
  --seq-len             8192 \
  --lr                  2.5e-5 \
  --weight-decay        0.1 \
  --warmup-steps        50 \
  --epochs              3 \
  --validation-interval 100 \
  --checkpoint-interval 100 \
  --use-wandb \
  --wandb-project       llada-sft