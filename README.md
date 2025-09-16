# LLada-Reasoning

A comprehensive training framework for large language diffusion models with reasoning capabilities, featuring pretraining, supervised fine-tuning (SFT), and reinforcement learning from human feedback (RLHF).

## Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/QuentinFitteRey/LLada-Reasoning
cd LLada-Reasoning/
conda create -n llada python=3.11 -y
conda activate llada
pip install -r requirements.txt  # full stack (heavy)

# (Optional) Minimal core install (create later):
# pip install -r requirements-core.txt
```
### 2. Download Base Model
```bash
# Download the base model weights
# Edit download.py to choose between base model or 1.5 version
python download.py

# The repository ships only one tracked modeling implementation: modelling_final/
# These files already represent the final architecture (YARN, KV cache optimizations, etc.)
# If you overwrite a downloaded model directory with them:
cp modelling_final/* /path/to/your/model/weights/directory/

# Note: Adapter compatibility depends on which base model you use:
# - Adapters trained on base model work with ./llada_local
# - Adapters trained on 1.5 model work with ./llada_local_1.5
```

### 3. Train Your Model

#### Pretraining (Extended Context)
```bash
torchrun --nproc_per_node=4 pretraining/pretrain_llada_extended.py \
    --output-dir ./checkpoints/pretrain_run
```

#### Supervised Fine-Tuning
```bash
torchrun --nproc_per_node=8 sft_training/sft_train_new_dataset.py \
    --pretrained-model ./pretrain_merged_model \
    --output-dir ./checkpoints/sft_run
```

#### RLHF Training (DPO or GRPO)
```bash
# DPO
python rlhf/train_dpo.py --output_dir ./dpo_checkpoints

# GRPO (requires Ollama services)
bash script/server_ollama.sh  # Terminal 1
bash script/judge_ollama.sh   # Terminal 2
python rlhf/train_grpo.py --output_dir ./grpo_checkpoints
```

### 4. Use Your Model

#### Merge and Generate
```bash
# Merge LoRA adapters
python merge_model.py --base_model ./base_model --adapter_path ./checkpoints/adapter

# Generate text (edit prompts and model paths in generation.py)
python generation.py

# Evaluate

# Make sure to add --confirm_run_unsafe_code if the benchmarks task is flagged as unsafe

# Model with no adaptater
accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --mixed_precision bf16 \
  eval_llada.py \
    --tasks [TASK_NAME (eg. gsm8k)] \
    --num_fewshot [number allowed by the benchmark (eg. 5)] \
    --model llada_dist \
    --batch_size 1 \
    --model_args 'model_path=/path/to/model,load_lora=False,cfg=0.0,is_check_greedy=False,mc_num=128,gen_length=256,steps=256,block_length=16,temperature=0.0'

# Model with adapater
accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --mixed_precision bf16 \
  eval_llada.py \
    --tasks [TASK_NAME (eg. gsm8k)] \
    --num_fewshot [number allowed by the benchmark (eg. 5)] \
    --model llada_dist \
    --batch_size 1 \
    --model_args 'model_path=/path/to/model,adapter_path=/path/to/adaptater,load_lora=True,cfg=0.0,is_check_greedy=False,mc_num=128,gen_length=1024,steps=1024,block_length=16,temperature=0.0' 
```

## Key Features

- **Multi-GPU Training**: Distributed training across multiple GPUs
- **Parameter-Efficient**: LoRA fine-tuning for memory efficiency
- **Extended Context**: Support for sequences up to 8192 tokens
- **Multiple RLHF Methods**: Both DPO and GRPO implementations
- **Easy Evaluation**: Built-in benchmarking and generation tools

## Complete Documentation

📖 **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Complete commands, parameters, and troubleshooting


## Monitoring

Training automatically logs to Weights & Biases:
```bash
wandb login  # One-time setup
```

## Directory Overview

| Path | Tracked | Purpose |
|------|---------|---------|
| modelling_final/ | yes | Final model config + modeling code to copy into weight dirs |
| pretraining/ | yes | Extended-context masking pretraining script |
| sft_training/ | yes | Supervised fine-tuning (chat style) |
| rlhf/ | yes | DPO / GRPO training components |
| generation.py | yes | Dual-cache generation utilities |
| eval_llada.py | yes | Evaluation harness (experimental) |
| checkpoints/, wandb/, filtered_conversational_dataset/ | ignored | Local artifacts & data |
| llada_local*, merged_model/ | ignored | Downloaded or merged model weights |

## Minimal vs Full Installation

If you only need inference + basic fine-tuning (LoRA): create (optional) a requirements-core.txt with roughly:
```
torch
transformers
accelerate
datasets
peft
bitsandbytes
wandb
lm_eval
safetensors
einops
```
The full `requirements.txt` adds: RLHF tooling (openrlhf, deepspeed), serving (vllm, fastapi), optimization (triton, xformers), experimentation (ray), geometry / vision (opencv, trimesh, cadquery), etc.

## Notes

- Evaluation harness still evolving; parameter names may change.
- Ensure you run download.py before training so model files exist at ./llada_local (or adjust paths).
```
