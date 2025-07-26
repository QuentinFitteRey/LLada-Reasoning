import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import re
from datetime import datetime
from functools import partial

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer import get_scheduler

from rlhf.actor import Actor
from rlhf.deepspeedStrategy import DeepspeedStrategy
from rlhf.prompt_dataset import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from rlhf.ruler import sync_ruler_reward
from rlhf.grpo_trainer import GRPOTrainer

def init_base_model_and_tokenizer(args):
    """
    Initializes the base model and tokenizer from a given path,
    adds the required special tokens, and resizes the model's embeddings
    to match the new tokenizer size. This provides a consistent starting
    point for both the policy and reference models.
    """
    print(f"Loading tokenizer from: {args.pretrain}")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain)

    print("Adding special tokens to tokenizer...")
    special_tokens_to_add = {
        "additional_special_tokens": [
            "<|mdm_mask|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
            "<|begin_of_thought|>", "<|end_of_thought|>" "<|begin_of_solution|>", "<|end_of_solution|>"
        ]
    }
    if tokenizer.pad_token is None:
        special_tokens_to_add["pad_token"] = "<|pad|>"
    
    tokenizer.add_special_tokens(special_tokens_to_add)

    print(f"Loading base model from: {args.pretrain}")
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrain,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    print(f"Resizing model embeddings to new tokenizer size: {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def train(args):
    # --- Strategy Setup ---
    def get_strategy(args):
        return DeepspeedStrategy(
            seed=getattr(args, "seed", 42),
            max_norm=getattr(args, "max_norm", 1.0),
            micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
            train_batch_size=getattr(args, "train_batch_size", 128),
            zero_stage=args.zero_stage,
            bf16=getattr(args, "bf16", True),
            args=args,
        )
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # --- Initialize Base Model and Tokenizer ---
    # This provides the correctly resized starting point for both models.
    base_model, tokenizer = init_base_model_and_tokenizer(args)

    # --- REFERENCE MODEL CREATION ---
    # The reference model is the base model with ONLY the SFT adapter loaded. It is completely frozen.
    strategy.print(f"Creating reference model by loading frozen SFT adapter from: {args.sft_adapter_path}")
    ref_model_peft = PeftModel.from_pretrained(
        base_model,
        args.sft_adapter_path,
        is_trainable=False,
        adapter_name="sft_adapter"  # Naming for clarity
    )
    ref_model = Actor(ref_model_peft, tokenizer=tokenizer, use_flash_attention_2=args.flash_attn, bf16=args.bf16, ds_config=strategy.get_ds_eval_config(offload=args.ref_offload))

    # --- POLICY MODEL CREATION (STACKING ADAPTERS) ---
    # We need a fresh base model instance to avoid object reference issues.
    policy_base_model, _ = init_base_model_and_tokenizer(args)

    # 1. Load the SFT adapter onto the fresh base model and keep it frozen.
    strategy.print(f"Creating policy model by loading frozen SFT adapter from: {args.sft_adapter_path}")
    policy_model_peft = PeftModel.from_pretrained(
        policy_base_model,
        args.sft_adapter_path,
        is_trainable=False,
        adapter_name="sft_adapter"
    )
    special_tokens_to_add_buggy = {
        "additional_special_tokens": [
            "<|mdm_mask|>", "<|start_header_id|>", "<|end_header_id|>","<|eot_id|>",
            "<|begin_of_thought|>","<|end_of_thought|>", "<|begin_of_solution|>", "<|end_of_solution|>"
        ]
    }
    if tokenizer.pad_token is None:
        special_tokens_to_add_buggy["pad_token"] = "<|pad|>"
    
    tokenizer.add_special_tokens(special_tokens_to_add_buggy)
    new_vocab_size = len(tokenizer)
    policy_model_peft.resize_token_embeddings(new_vocab_size)
    ref_model.model.resize_token_embeddings(new_vocab_size)

    # 2. Define the configuration for the NEW adapter for GRPO training.
    strategy.print("Defining and adding a new, trainable adapter for GRPO training...")
    grpo_lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=args.target_modules,
    )

    # 3. Add the new adapter to the model and set it as the active one for training.
    policy_model_peft.add_adapter("grpo_adapter", grpo_lora_config)
    if args.load_ckpt:
        adapter_dir = os.path.join(args.load_ckpt, "grpo_adapter")
        policy_model_peft.load_adapter(
        adapter_dir,     # path to the folder where you did `.save_pretrained()`
        adapter_name="grpo_adapter",
        is_trainable=True
    )
    policy_model_peft.set_adapter("grpo_adapter")

    strategy.print("Policy model created with stacked adapters. Trainable parameters:")
    policy_model_peft.print_trainable_parameters()

    model = Actor(policy_model_peft, tokenizer=tokenizer, use_flash_attention_2=args.flash_attn, bf16=args.bf16, ds_config=strategy.get_ds_train_config(is_actor=True))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant})

    # --- USER REQUEST: Add special tokens again with missing comma and resize ---
    strategy.print("Applying second round of special tokens (with missing comma) and resizing...")

    
    # strategy.print(f"Resizing both policy and reference models to new tokenizer size: {new_vocab_size}")
    # The `resize_token_embeddings` method should be called on the underlying PeftModel
    # strategy.print("Models resized successfully.")
    # --- END OF USER REQUEST ---

    # --- Optimizer, Data, and Scheduler ---
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)
    
    train_data = blending_datasets(args.dataset, args.dataset_probs, strategy, args.seed, max_count=args.max_samples)
    
    def map_prompt(example):
        return {"prompt": example.get("user", example.get("prompt", ""))}

    train_data = train_data.map(map_prompt, remove_columns=train_data.column_names).select(range(min(args.max_samples, len(train_data))))
    train_dataset = PromptDataset(train_data, tokenizer, args.max_len, strategy, input_template=args.input_template)
    train_dataloader = strategy.setup_dataloader(train_dataset, args.micro_train_batch_size, True, True, train_dataset.collate_fn)
    
    eval_dataloader = None

    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)
    
    scheduler = get_scheduler(
        args.lr_scheduler, optim, num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps, scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # --- Strategy Prepare & Trainer Initialization ---
    ((model, optim, scheduler), ref_model) = strategy.prepare((model, optim, scheduler), ref_model)

    consumed_samples = 0
    if args.load_ckpt:
        strategy.print(f"Loading checkpoint from: {args.load_ckpt}")
        base_dir = args.ckpt_path
        tag          = "ds_checkpoint"
        # engine.load_checkpoint returns (load_succeeded, client_state)
        loaded, client_state = strategy.engine.load_checkpoint(
            base_dir,
            tag=tag
        )
        if not loaded:
            raise RuntimeError(f"Failed to load DS checkpoint from {base_dir}/{tag}")
        if "consumed_samples" in client_state:
            consumed_samples = client_state["consumed_samples"]
        else:
            consumed_samples = 0
    trainer = GRPOTrainer(
        model=model, ref_model=ref_model, reward_fn=sync_ruler_reward,
        tokenizer=tokenizer, strategy=strategy, optim=optim,
        train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, scheduler=scheduler,
        beta=args.beta, max_epochs=args.max_epochs,
        num_samples_per_prompt=args.num_samples_per_prompt, gen_steps=args.gen_steps,
        gen_length=args.gen_length, temperature=args.temperature,
        remasking_threshold=args.remasking_threshold, repetition_penalty=args.repetition_penalty
    )

    trainer.fit(args,consumed_samples=consumed_samples, num_update_steps_per_epoch=num_update_steps_per_epoch)
    
    # This will save only the newly trained 'grpo_adapter'
    strategy.save_model(model.model, tokenizer, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Models & Paths
    parser.add_argument("--pretrain", type=str, required=True, help="Path to the base model")
    parser.add_argument("--sft_adapter_path", type=str, required=True, help="Path to the pre-trained SFT LoRA adapter.")
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
    parser.add_argument("--load_ckpt", type=str, default=None, help="Path to a checkpoint to resume training from.")

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
