#!/bin/bash
#SBATCH -J test_llada_reasoning
#SBATCH -o /home/hice1/jmoutahir3/scratch/LLada-Reasoning/test_logs/llada_sft/logs/test_generation.out
#SBATCH -e /home/hice1/jmoutahir3/scratch/LLada-Reasoning/test_logs/llada_sft/logs/test_generation.err
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H200:1         # GPU type (H100) and number of GPUs
#SBATCH -t2:00:00                        # Duration of the job
#SBATCH --mem=1024G
#SBATCH --cpus-per-task=8
#SBATCH --tmp=1000G

JOBID=$SLURM_JOB_ID

module load anaconda3

conda activate ~/scratch/envs/llada
cd /home/hice1/jmoutahir3/scratch/LLada-Reasoning
export PYTHONPATH=$(pwd)

nvidia-smi

srun python new_generation.py