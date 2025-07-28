# LLada-Reasoning Quick Reference Guide

## Environment Setup
```bash
# Setup
git clone https://github.com/QuentinFitteRey/LLada-Reasoning
cd LLada-Reasoning/
conda create -n llada python=3.11
conda activate llada
pip install -r requirements.txt

# Download base model
# Edit download.py to choose between base model or 1.5 version
python download.py

# Copy modeling files for advanced features (YARN, KC cache, etc.)
# Replace /path/to/model/weights/ with your actual model directory
cp modelling_final/* /path/to/model/weights/directory/

# Optional environment variables
export HF_HOME=./cache
export PYTHONPATH=$(pwd)
```

## Model Setup Notes

**Important**: The `modelling_final/` directory contains updated model files that enable:
- **YARN scaling**: Better long context handling with YaRN (Yet another RoPE extensioN)
- **KC cache**: Efficient key-value caching mechanisms  
- **Advanced features**: Latest model architecture improvements

**Usage**: Copy these files to your model weights directory to replace the default model files and enable these advanced features.

**Model Version Notes**: 
- **download.py**: Edit this file to choose between downloading the base model or 1.5 version
- **Adapter Compatibility**: Adapters are tied to the specific model version they were trained on:
  - Adapters trained on base model → Use with `./llada_local`
  - Adapters trained on 1.5 model → Use with `./llada_local_1.5`
- **Training Paths**: Ensure you use the correct base model path when training adapters

## Training Commands (Multi-GPU)

### Pretraining
```bash
# Multi-GPU pretraining (4 GPUs)
torchrun --nproc_per_node=4 --master_port=29501 pretraining/pretrain_llada_extended.py \
    --train-data ./data_pretrain_stratified/train_new.txt \
    --val-data ./data_pretrain_stratified/val_new.txt \
    --output-dir ./checkpoints/pretrain_experiment \
    --pretrained-model ./llada_local \
    --batch-size 64 \
    --local-batch 8 \
    --seq-len 8192 \
    --lr 1e-5 \
    --max-train-steps 500 \
    --use-lora \
    --lora-r 128 \
    --lora-alpha 256
```

### SFT Training
```bash
# Multi-GPU SFT (8 GPUs)
torchrun --nproc_per_node=8 --master_port=29405 sft_training/sft_train_new_dataset.py \
    --pretrained-model ./llada_local_1.5 \
    --sft-data ./filtered_conversational_dataset/ \
    --output-dir ./checkpoints/sft_experiment \
    --lr 1e-5 \
    --epochs 3 \
    --local-batch 8 \
    --batch-size 128 \
    --warmup-ratio 0.1 \
    --lora-target-modules q_proj v_proj k_proj attn_out
```

### RLHF Training

#### DPO (Direct Preference Optimization)
```bash
python rlhf/train_dpo.py \
    --pretrain ./llada_local_1.5 \
    --sft_adapter ./checkpoints/sft_adapter \
    --output_dir ./dpo_checkpoints_lora \
    --train_batch 64 \
    --micro_batch 2 \
    --zero_stage 2 \
    --max_len 4096 \
    --epochs 1 \
    --lr 5e-5 \
    --beta 0.2 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --target_modules q_proj v_proj k_proj attn_out
```

#### GRPO (Group Relative Policy Optimization)

**Prerequisites**: Start Ollama services for GRPO training:
```bash
# Terminal 1: Start Ollama server
bash script/server_ollama.sh

# Terminal 2: Start Ollama judge  
bash script/judge_ollama.sh

# Note: GRPO uses a Large Language Model as a judge to provide rewards
# for preference optimization. Ollama serves this judge model.
# You can change the model in the scripts to any Ollama model,
# but the current configuration provides the best performance
```

**Training**:
```bash
python rlhf/train_grpo.py \
    --pretrain ./llada_local_1.5 \
    --sft_adapter ./checkpoints/sft_adapter \
    --output_dir ./grpo_checkpoints_lora \
    --train_batch 32 \
    --micro_batch 1 \
    --zero_stage 2 \
    --max_len 4096 \
    --epochs 1 \
    --lr 3e-5 \
    --group_size 4 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --target_modules q_proj v_proj k_proj attn_out
```

## Model Operations

### Model Merging
```bash
# Merge LoRA adapter with base model
python merge_model.py \
    --base_model ./llada_local_1.5 \
    --adapter_path ./checkpoints/sft_adapter \
    --output_path ./merged_model
```

### Evaluation
```bash
# Run evaluation
python eval_llada.py \
    --model_path ./merged_model \
    --eval_tasks gsm8k,hellaswag \
    --output_dir ./eval_results
```

### Text Generation
```bash
# Generate text (edit prompts directly in the generation.py file)
python generation.py --model_path ./merged_model

# Note: Prompts are configured inside the generation.py script, 
# not passed as command line arguments
```



### Training Progress
```bash
# View training logs
tail -f ./checkpoints/experiment_name/logs/train.log

# Monitor wandb (if configured)
wandb login
```

## Common Parameters

### LoRA Configuration
- Rank: 128
- Alpha: 256
- Dropout: 0.05-0.1
- Target modules: q_proj, v_proj, k_proj, attn_out

### Training Configuration
- Learning rate: 1e-5 (pretraining/SFT), 5e-5 (DPO), 3e-5 (GRPO)
- Batch sizes: 64-128 (global), 8 (local)
- Sequence length: 8192 (pretraining), 4096 (SFT/RLHF)
- Warmup: 10-20 steps or 0.1 ratio

### RLHF-Specific Parameters
- **DPO Beta**: 0.2 (controls strength of preference optimization)
- **GRPO Group Size**: 4 (number of responses per group)
- **Zero Stage**: 2 (DeepSpeed memory optimization level)


## Wandb Setup
```bash
# Login to wandb (one-time)
wandb login

# Training metrics are automatically logged
# View at: https://wandb.ai/your_username/llada-reasoning
```

## Complete Training Pipeline Example
```bash
# 1. Pretraining
torchrun --nproc_per_node=4 pretraining/pretrain_llada_extended.py \
    --output-dir ./checkpoints/pretrain

# 2. Merge pretrained adapter
python merge_model.py \
    --base_model ./llada_local \
    --adapter_path ./checkpoints/pretrain \
    --output_path ./pretrain_merged_model

# 3. SFT Training  
torchrun --nproc_per_node=8 sft_training/sft_train_new_dataset.py \
    --pretrained-model ./pretrain_merged_model \
    --output-dir ./checkpoints/sft

# 4. Merge SFT adapter
python merge_model.py \
    --base_model ./pretrain_merged_model \
    --adapter_path ./checkpoints/sft \
    --output_path ./sft_merged_model

# 5. RLHF Training (DPO)
python rlhf/train_dpo.py \
    --pretrain ./sft_merged_model \
    --sft_adapter ./checkpoints/sft \
    --output_dir ./checkpoints/dpo

# 6. Final Model Merging
python merge_model.py \
    --base_model ./sft_merged_model \
    --adapter_path ./checkpoints/dpo \
    --output_path ./final_model

# 7. Evaluation
python eval_llada.py --model_path ./final_model
```

This quick reference provides the essential commands and configurations for training LLada-Reasoning models efficiently.
