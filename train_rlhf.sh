#!/bin/bash
set -x

# === PATHS ===
BASE_MODEL_PATH="./llada_local_1.5"  # Your base model path (non-merged)
OUTPUT_DIR="./dpo_checkpoints_lora"  # Where to save trained LoRA adapter

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
BETA=0.2

# === LORA CONFIG (match your LoRAConfig) ===
LORA_ARGS="\
--lora_rank 32 \
--lora_alpha 64 \
--lora_dropout 0.1 \
--target_modules q_proj v_proj k_proj attn_out
"

export DS_MASTER_PORT=42000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# === RUN ===
deepspeed rlhf/train_dpo.py \
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
    --adam_betas 0.9 0.95 \
    --l2 0.01 \
    --lr_warmup_ratio 0.003 \
    --max_epochs "$EPOCHS" \
    --logging_steps 1 \
    --save_steps -1 \
    --eval_steps -1 \
    --use_wandb dc953a73754e73f853a4148bed458100f5ed36f7\
    --wandb_project "LLada-Reasoning" \
    --wandb_run_name "VRPO_lora" \
    --gradient_checkpointing \
    $BF16 \
    $FLASH \
    $GRAD_CHECK \
    $LORA_ARGS
    # --apply_chat_template \
