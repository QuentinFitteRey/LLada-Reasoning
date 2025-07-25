#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3
conda activate ~/scratch/envs/llada


# === PATHS ===
BASE_MODEL_PATH="./llada_local_15"      # Your base model path (non-merged)
OUTPUT_DIR="./grpo_checkpoints_lora"      # Where to save the trained GRPO LoRA adapter

mkdir -p "$OUTPUT_DIR"

# === TRAIN CONFIG ===
TRAIN_BATCH=64
MICRO_BATCH=1
ZERO_STAGE=2
BF16="--bf16"
FLASH="--flash_attn"
MAX_LEN=4096
EPOCHS=1
LR=5e-5
BETA=0.1

# === LORA CONFIG ===
LORA_ARGS="\
--lora_rank 128 \
--lora_alpha 256 \
--lora_dropout 0.1 \
--target_modules q_proj v_proj k_proj attn_out
"

# === GRPO/GENERATION CONFIG (NEW) ===
# These are arguments for your custom generate_with_dual_cache function and the trainer
NUM_SAMPLES=4           # 'k' samples to generate per prompt
GEN_STEPS=2048           # Number of diffusion steps for generation
GEN_LENGTH=2048          # Max new tokens to generate
TEMP=0.7                # Temperature for sampling
REMASK_THRESH=0.9       # Remasking threshold for diffusion
REP_PENALTY=1.2         # Repetition penalty for generation

# === ENVIRONMENT ===
export DS_MASTER_PORT=42000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === RUN ===
deepspeed rlhf/train_grpo.py \
    --pretrain "$BASE_MODEL_PATH" \
    --dataset /home/hice1/yluo432/scratch/LLada-Reasoning/filtered_conversational_dataset_4k \
    --dataset_probs 1.0 \
    --save_path "$OUTPUT_DIR" \
    --max_len "$MAX_LEN" \
    --train_batch_size "$TRAIN_BATCH" \
    --micro_train_batch_size "$MICRO_BATCH" \
    --zero_stage "$ZERO_STAGE" \
    --learning_rate "$LR" \
    --beta "$BETA" \
    --adam_betas 0.9 0.95 \
    --l2 0.01 \
    --lr_warmup_ratio 0.003 \
    --max_epochs "$EPOCHS" \
    --logging_steps 1 \
    --save_steps -1 \
    --eval_steps -1 \
    --gradient_checkpointing \
    --use_wandb dc953a73754e73f853a4148bed458100f5ed36f7 \
    --wandb_project "LLada-Reasoning" \
    --wandb_run_name "GRPO_training" \
    $BF16 \
    $FLASH \
    $LORA_ARGS \
    --num_samples_per_prompt "$NUM_SAMPLES" \
    --gen_steps "$GEN_STEPS" \
    --gen_length "$GEN_LENGTH" \
    --temperature "$TEMP" \
    --remasking_threshold "$REMASK_THRESH" \
    --repetition_penalty "$REP_PENALTY"