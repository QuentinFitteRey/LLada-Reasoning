#!/bin/bash
set -x

# === PATHS ===
BASE_MODEL_PATH="./llada_local_1.5"  # Your base model path (non-merged)
OUTPUT_DIR="./dpo_checkpoints_lora"  # Where to save trained LoRA adapter

mkdir -p "$OUTPUT_DIR"

# === TRAIN CONFIG ===
TRAIN_BATCH=16
MICRO_BATCH=1
ZERO_STAGE=3
BF16="--bf16"
FLASH="--flash_attn"
# GRAD_CHECK="--gradient_checkpointing"
MAX_LEN=4096
EPOCHS=1
LR=5e-7
BETA=0.1

# === LORA CONFIG (match your LoRAConfig) ===
LORA_ARGS="\
--lora_rank 32 \
--lora_alpha 64 \
--lora_dropout 0.5 \
--target_modules q_proj v_proj k_proj o_proj
"

export DS_MASTER_PORT=42000

# === RUN ===
deepspeed ./train_dpo.py \
    --pretrain "$BASE_MODEL_PATH" \
    --dataset stanfordnlp/SHP \
    --dataset_probs 1.0 \
    --save_path "$OUTPUT_DIR" \
    --max_len "$MAX_LEN" \
    --train_batch_size "$TRAIN_BATCH" \
    --micro_train_batch_size "$MICRO_BATCH" \
    --zero_stage "$ZERO_STAGE" \
    --learning_rate "$LR" \
    --beta "$BETA" \
    --max_epochs "$EPOCHS" \
    --logging_steps 1 \
    --save_steps -1 \
    --eval_steps -1 \
    $BF16 \
    $FLASH \
    $GRAD_CHECK \
    $LORA_ARGS
    # --apply_chat_template \
