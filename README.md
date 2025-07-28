# LLada-Reasoning

A comprehensive training framework for large language models with reasoning capabilities, featuring pretraining, supervised fine-tuning (SFT), and reinforcement learning from human feedback (RLHF).

## Quick Start

### 1. Setup
```bash
git clone https://github.com/QuentinFitteRey/LLada-Reasoning
cd LLada-Reasoning/
conda create -n llada python=3.11
conda activate llada
pip install -r requirements.txt
```

### 2. Download Base Model
```bash
# Download the base model weights
# Edit download.py to choose between base model or 1.5 version
python download.py

# Copy modeling files for advanced features (YARN, KC cache, etc.)
# Copy all files from modelling_final/ to your model weights directory
cp modelling_final/* /path/to/your/model/weights/directory/

# Note: Adapter compatibility depends on which base model you use:
# - Adapters trained on base model work with ./llada_local
# - Adapters trained on 1.5 model work with ./llada_local_1.5
```

### 4. Train Your Model

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

### 5. Use Your Model

#### Merge and Generate
```bash
# Merge LoRA adapters
python merge_model.py --base_model ./base_model --adapter_path ./checkpoints/adapter

# Generate text (edit prompts in generation.py)
python generation.py --model_path ./merged_model

# Evaluate
python eval_llada.py --model_path ./merged_model
```

## Key Features

- **Multi-GPU Training**: Distributed training across multiple GPUs
- **Parameter-Efficient**: LoRA fine-tuning for memory efficiency
- **Extended Context**: Support for sequences up to 8192 tokens
- **Multiple RLHF Methods**: Both DPO and GRPO implementations
- **Easy Evaluation**: Built-in benchmarking and generation tools

## Complete Documentation

📖 **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Complete commands, parameters, and troubleshooting

## Hardware Requirements

- **GPUs**: CUDA-compatible GPUs (4-8 recommended)
- **Memory**: 64GB+ system RAM
- **Storage**: 500GB+ available space

## Monitoring

Training automatically logs to Weights & Biases:
```bash
wandb login  # One-time setup
```
