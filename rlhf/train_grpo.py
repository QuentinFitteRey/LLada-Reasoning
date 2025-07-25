import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import re
from datetime import datetime
from functools import partial

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer import get_scheduler

from rlhf.actor import Actor
from rlhf.deepspeedStrategy import DeepspeedStrategy
# Use PromptDataset for GRPO as we only need prompts
from rlhf.prompt_dataset import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from rlhf.ruler import sync_ruler_reward
from peft import LoraConfig, get_peft_model, PeftModel


# Import the new GRPO Trainer
from rlhf.grpo_trainer import GRPOTrainer


def regex_reward_fn(texts: list[str], prompt: str) -> list[float]:
    """
    A simple example reward function.
    Gives a reward of 1.0 if the text contains a specific desirable pattern, 
    and a penalty of -1.0 if it contains an undesirable one.
    """
    rewards = []
    good_pattern = re.compile(r"the final answer is \d+", re.IGNORECASE)
    bad_pattern = re.compile(r"i don't know", re.IGNORECASE)
    
    for text in texts:
        if good_pattern.search(text):
            rewards.append(1.0)
        elif bad_pattern.search(text):
            rewards.append(-1.0)
        else:
            rewards.append(0.0)
    return rewards

def init_model(args, model_path):
    """Initializes a model and tokenizer from a given path."""
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    special_tokens_to_add = {
        "additional_special_tokens": ["<|mdm_mask|>", "<|start_header_id|>", "<|end_header_id|>","<|eot_id|>","<|begin_of_thought|>","<|end_of_thought|>" "<|begin_of_solution|>", "<|end_of_solution|>"]
    }
    if tokenizer.pad_token is None:
        special_tokens_to_add["pad_token"] = "<|pad|>"
        
    tokenizer.add_special_tokens(special_tokens_to_add)
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, "/home/hice1/yluo432/scratch/LLada-Reasoning/step-1100/sft_adapter")
    model = model.merge_and_unload()
    special_tokens_to_add = {
        "additional_special_tokens": ["<|mdm_mask|>", "<|start_header_id|>", "<|end_header_id|>","<|eot_id|>","<|begin_of_thought|>","<|end_of_thought|>","<|begin_of_solution|>", "<|end_of_solution|>"]
    }
    if tokenizer.pad_token is None:
        special_tokens_to_add["pad_token"] = "<|pad|>"
        
    tokenizer.add_special_tokens(special_tokens_to_add)
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model from {model_path} loaded successfully.")
    return model, tokenizer

def train(args):
    def get_strategy(args):
        strategy = DeepspeedStrategy(
            seed=getattr(args, "seed", 42),
            full_determinism=getattr(args, "full_determinism", False),
            max_norm=getattr(args, "max_norm", 1.0),
            micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
            train_batch_size=getattr(args, "train_batch_size", 128),
            zero_stage=args.zero_stage,
            bf16=getattr(args, "bf16", True),
            args=args,
        )
        return strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # --- Initialize Models and Tokenizer ---
    base_model, tokenizer = init_model(args, args.pretrain)
    ref_model_instance, _ = init_model(args, args.ref_pretrain)

    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=args.target_modules,
    )
    
    peft_model = get_peft_model(base_model, lora_config)
    
    model = Actor(peft_model, tokenizer=tokenizer, use_flash_attention_2=args.flash_attn, bf16=args.bf16, ds_config=strategy.get_ds_train_config(is_actor=True))
    ref_model = Actor(ref_model_instance, tokenizer=tokenizer, use_flash_attention_2=args.flash_attn, bf16=args.bf16, ds_config=strategy.get_ds_eval_config(offload=args.ref_offload))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant})

    # --- Optimizer and Scheduler ---
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)
    
    # --- Data Preparation (for Prompts) ---
    train_data = blending_datasets(args.dataset, args.dataset_probs, strategy, args.seed, max_count=args.max_samples)
    
    def map_prompt(example):
        """
        Pre-formats the prompt according to the chat template and filters out long examples.
        """

        user_prompt = example["user"]

        # formatted_prompt = (
        #     f"<|begin_of_text|>"
        #     f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt.strip()}<|eot_id|>"
        #     f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        # )

        return {
            "prompt": user_prompt,
        }


    train_data = train_data.map(map_prompt, remove_columns=train_data.column_names).select(range(min(args.max_samples, len(train_data))))
    train_dataset = PromptDataset(train_data, tokenizer, args.max_len, strategy, input_template=args.input_template)
    train_dataloader = strategy.setup_dataloader(train_dataset, args.micro_train_batch_size, True, True,train_dataset.collate_fn)
    

    # (Optional) Setup eval dataloader similarly
    eval_dataloader = None # Placeholder

    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)
    
    scheduler = get_scheduler(
        args.lr_scheduler, optim, num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps, scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # --- Strategy Prepare ---
    # Note: The reward_fn is NOT prepared by the strategy
    ((model, optim, scheduler), ref_model) = strategy.prepare((model, optim, scheduler), ref_model)

    # --- Trainer Initialization ---
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_fn=sync_ruler_reward, # Pass the chosen reward function
        tokenizer=tokenizer,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        beta=args.beta,
        max_epochs=args.max_epochs,
        num_samples_per_prompt=args.num_samples_per_prompt,
        gen_steps=args.gen_steps,
        gen_length=args.gen_length,
        temperature=args.temperature,
        remasking_threshold=args.remasking_threshold,
        repetition_penalty=args.repetition_penalty
    )

    trainer.fit(args, num_update_steps_per_epoch=num_update_steps_per_epoch)
    strategy.save_model(model, tokenizer, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- Add all arguments from your dpo script ---
    # Models & Paths
    parser.add_argument("--pretrain", type=str, required=True, help="Path to the base model")
    parser.add_argument("--ref_pretrain", type=str, default=None, help="Path to the reference model, defaults to pretrain")
    parser.add_argument("--save_path", type=str, default="./ckpt_grpo")
    parser.add_argument("--ckpt_path", type=str, default="./ckpt_grpo/checkpoints")
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    # Datasets
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_probs", type=str, default="1.0")
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--input_key", type=str, default="prompt")

    # Training
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1, help="KL regularization coefficient")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    # DeepSpeed
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--micro_train_batch_size", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--ref_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    # Logging & Saving
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="llada_grpo")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # --- New GRPO/Generation Arguments ---
    parser.add_argument("--num_samples_per_prompt", type=int, default=4, help="Number of responses to generate per prompt (k)")
    parser.add_argument("--gen_steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--remasking_threshold", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)

    args = parser.parse_args()
    if args.ref_pretrain is None: args.ref_pretrain = args.pretrain
    train(args)