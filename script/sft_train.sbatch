#!/bin/bash
#SBATCH -J pretrain_llada_reasoning
#SBATCH -o ./checkpoints/checkpoints_llada_sft_train/logs/SlurmTraining.out
#SBATCH -e ./checkpoints/checkpoints_llada_sft_train/logs/SlurmTraining.err
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:4
#SBATCH -t4:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --tmp=1000G
#SBATCH -A gts-yt7
#SBATCH -q inferno

module load anaconda3
conda activate ~/scratch/envs/llada
cd /storage/home/hcoda1/0/jmoutahir3/scratch/LLada-Reasoning

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get the job ID
JOBID=$SLURM_JOB_ID
echo "SFT Training: Job $JOBID has started!" | mail -s "SLURM Job Started!" jmoutahir3@gatech.edu
echo "SFT Training: Job $JOBID has started!" | mail -s "SLURM Job Started!" moutahirjed@gmail.com

export PYTHONPATH=$PWD

srun torchrun --nproc_per_node=1 -m sft_training.sft_train \
  --sft-data     ./sft_training/sft_data/combined \
  --pretrained-model ./llada_local \
  --output-dir   ./checkpoints/sft_lora_run \
  --lora-resume /home/hice1/jmoutahir3/scratch/LLaDA_checkpoints/new_test_checkpoint \
  --use-lora \
  --lora-r       16 \
  --lora-alpha   32 \
  --lora-dropout 0.05 \
  --batch-size   64 \
  --local-batch  1 \
  --seq-len      8192 \
  --lr           5e-5 \
  --epochs       3 \
  --max-steps    1000 \
  --warmup-steps 100 \
  --validation-interval 100 \
  --checkpoint-interval 100 \
  --use-wandb \
  --wandb-project llada-sft