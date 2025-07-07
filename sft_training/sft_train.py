#!/usr/bin/env python3
import os
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from datasets import load_from_disk, interleave_datasets
from peft import LoraConfig, get_peft_model, PeftModel
import torch.nn.functional as F
import wandb
import time
import math

# Assuming llada_local is in the python path
from llada_local.modeling_llada import LLaDAModelLM
from llada_local.configuration_llada import ActivationCheckpointingStrategy

# -------- DDP Setup --------
def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# -------- SFT Dataset & Collator (Unchanged) --------
class SFTDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset, self.tokenizer, self.max_length = dataset, tokenizer, max_length
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        ex = self.dataset[idx]
        user_ids = self.tokenizer.encode(ex['user'] + self.tokenizer.eos_token, add_special_tokens=False)
        asm_ids = self.tokenizer.encode(ex['assistant'] + self.tokenizer.eos_token, add_special_tokens=False)
        prompt_length = len(user_ids)
        input_ids = (user_ids + asm_ids)[:self.max_length]
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "prompt_length": prompt_length}

class SFTDataCollator:
    def __init__(self, tokenizer): self.pad_id = tokenizer.pad_token_id
    def __call__(self, features):
        input_ids = [ex['input_ids'] for ex in features]
        prompt_lengths = torch.tensor([ex['prompt_length'] for ex in features], dtype=torch.long)
        padded_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        return {'input_ids': padded_ids, 'prompt_lengths': prompt_lengths}

# -------- Core Functions (Unchanged) --------
def forward_process(input_ids, mask_token_id, mask_ratio):
    b, l = input_ids.shape
    p_mask = torch.full((b, l), mask_ratio, device=input_ids.device)
    masked_indices = torch.bernoulli(p_mask).bool()
    return torch.where(masked_indices, mask_token_id, input_ids), masked_indices, p_mask

@torch.no_grad()
def print_trainable_parameters(model):
    trainable, total = 0, 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad: trainable += param.numel()
    print(f"trainable params: {trainable:,} || total params: {total:,} || trainable%: {100 * trainable / total:.2f}")

def calc_sft_loss(model, batch, mask_id, pad_id, mask_ratio=0.5):
    ids, p_lens = batch["input_ids"], batch["prompt_lengths"]
    dev, b, l = ids.device, ids.shape[0], ids.shape[1]
    if l == 0: return None, 0.0
    noisy_in, masked_pos, p_mask = forward_process(ids, mask_id, mask_ratio)
    pos = torch.arange(l, device=dev).unsqueeze(0).expand(b, l)
    p_mask_bool = pos < p_lens.unsqueeze(1)
    noisy_in[p_mask_bool] = ids[p_mask_bool]
    logits = model(input_ids=noisy_in).logits
    ans_mask = ~p_mask_bool
    loss_mask = ans_mask & masked_pos & (ids != pad_id)
    if not loss_mask.any(): return None, 0.0
    loss = F.cross_entropy(logits[loss_mask], ids[loss_mask], reduction='none')
    norm_loss = loss / p_mask[loss_mask]
    total_ans_toks = torch.sum(ans_mask.int(), dim=1).sum()
    final_loss = norm_loss.sum() / total_ans_toks
    return final_loss, final_loss.item()

@torch.no_grad()
def run_validation(model, loader, dev, mask_id, pad_id, limit=None, mc_runs=1):
    model.eval()
    total_loss, count = 0.0, 0
    for i, batch in enumerate(loader):
        if limit and i >= limit: break
        batch = {k: v.to(dev) for k, v in batch.items()}
        losses = [calc_sft_loss(model, batch, mask_id, pad_id)[1] for _ in range(mc_runs)]
        valid_losses = [l for l in losses if l is not None and not math.isinf(l)]
        if valid_losses:
            total_loss += sum(valid_losses) / len(valid_losses)
            count += 1
    loss_t = torch.tensor(total_loss, device=dev); count_t = torch.tensor(count, device=dev)
    dist.all_reduce(loss_t, op=dist.ReduceOp.SUM); dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
    avg_loss = loss_t.item() / count_t.item() if count_t.item() > 0 else 0.0
    model.train()
    return avg_loss

# -------- Main --------
def main():
    parser = argparse.ArgumentParser(description='SFT for LLaDA with Multi-Stage LoRA')
    # Model and Data Paths
    parser.add_argument('--pretrained-model', type=str, default="./merged_model", help="Path to the base Hugging Face model.")
    parser.add_argument('--sft-data', type=str, default='./sft_data/combined', help='Path to SFT DatasetDict.')
    parser.add_argument('--output-dir', type=str, default='./checkpoints/llada_sft_first_from_pretrain_alldata', help='Directory to save checkpoints and logs.')
    parser.add_argument('--pretraining-lora-weights', type=str, default=None, help="Optional: Path to non-trainable LoRA weights from pre-training stage.")
    parser.add_argument('--sft-lora-weights', type=str, default=None, help="Optional: Path to trainable LoRA weights from a previous SFT run to continue training.")
    parser.add_argument('--resume-from-checkpoint', type=str, default=None, help="Path to resume trainer state (optimizer, scheduler, step).")
    # LoRA options (for creating a new adapter)
    parser.add_argument('--lora-r', type=int, default=128) 
    parser.add_argument('--lora-alpha', type=int, default=256)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--lora-target-modules', nargs='+', default=['q_proj', 'v_proj', 'k_proj', 'o_proj'])
    # Other arguments
    parser.add_argument('--torch-dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--activation-checkpointing', action='store_true', default=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--local-batch', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=50)
    parser.add_argument('--min-lr-ratio', type=float, default=0.1)
    parser.add_argument('--grad-clip-norm', type=float, default=1.0)
    parser.add_argument('--validation-interval', type=int, default=100)
    parser.add_argument('--checkpoint-interval', type=int, default=100)
    parser.add_argument('--val-limit-batches', type=int, default=75)
    parser.add_argument('--use-wandb', action='store_true', default=True)
    parser.add_argument('--wandb-project', type=str, default='llada-sft')
    args = parser.parse_args()

    setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    is_main = rank == 0

    # W&B Setup
    if is_main and args.use_wandb: wandb.init(project=args.wandb_project, config=vars(args))

    # --- Layered Model & LoRA Loading Logic ---
    dtype = getattr(torch, args.torch_dtype)
    print(dtype)
    if is_main: print(f"Loading base model: {args.pretrained_model}")
    model = LLaDAModelLM.from_pretrained(args.pretrained_model, trust_remote_code=True, torch_dtype=torch.bfloat16)

    if args.pretraining_lora_weights:
        if is_main: print(f"Applying non-trainable pre-training LoRA from: {args.pretraining_lora_weights}")
        model = PeftModel.from_pretrained(model, args.pretraining_lora_weights, is_trainable=False)

    if args.sft_lora_weights:
        if is_main: print(f"Loading trainable SFT LoRA from: {args.sft_lora_weights}")
        model.add_adapter(LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=args.lora_target_modules, lora_dropout=args.lora_dropout, bias='none'), adapter_name="sft_adapter")
        model.load_adapter(args.sft_lora_weights, adapter_name="sft_adapter")
        model.set_adapter("sft_adapter")
    else:
        if is_main: print("Creating new trainable SFT LoRA adapter.")
        lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=args.lora_target_modules, lora_dropout=args.lora_dropout, bias='none', task_type='CAUSAL_LM')
        model = get_peft_model(model, lora_config)

    if is_main: print_trainable_parameters(model)
    
    model.to(device) 
    # Tokenizer and other setup
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)
    special_tokens = {}
    if tokenizer.pad_token is None: special_tokens["pad_token"] = "<|pad|>"
    if tokenizer.convert_tokens_to_ids("<|mdm_mask|>") == tokenizer.unk_token_id: special_tokens["additional_special_tokens"] = ["<|mdm_mask|>"]
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
    mask_id, pad_id = tokenizer.convert_tokens_to_ids("<|mdm_mask|>"), tokenizer.pad_token_id

    if args.activation_checkpointing:
        if is_main: print("Enabling custom gradient checkpointing...")
        model.base_model.model.set_activation_checkpointing(strategy=ActivationCheckpointingStrategy.whole_layer)

    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    # Data Loading
    ds = load_from_disk("./sft_data/combined/")
    train_keys = [k for k in ds.keys() if k.endswith("_train")]
    val_keys = [k for k in ds.keys() if k.endswith("_validation")]
    limited_train_datasets, train_sizes = [], []
    limited_val_datasets, val_sizes = [], []
    TRAIN_LIMIT, VAL_LIMIT = 50000, 2500

    if is_main: print("=== TRAINING DATASETS (Limited) ===")
    for key in train_keys:
        original_size, limited_size = len(ds[key]), min(len(ds[key]), TRAIN_LIMIT)
        limited_train_datasets.append(ds[key].select(range(limited_size)))
        train_sizes.append(limited_size)
        if is_main: print(f"{key}: {original_size:,} -> {limited_size:,} samples")

    if is_main: print("\n=== VALIDATION DATASETS (Limited) ===")
    for key in val_keys:
        original_size, limited_size = len(ds[key]), min(len(ds[key]), VAL_LIMIT)
        limited_val_datasets.append(ds[key].select(range(limited_size)))
        val_sizes.append(limited_size)
        if is_main: print(f"{key}: {original_size:,} -> {limited_size:,} samples")

    total_working, total_working_val = sum(train_sizes), sum(val_sizes)
    train_probs = [s / total_working for s in train_sizes] if total_working > 0 else []
    val_probs = [s / total_working_val for s in val_sizes] if total_working_val > 0 else []
    
    train_ds = SFTDataset(interleave_datasets(limited_train_datasets, probabilities=train_probs, seed=42), tokenizer, args.seq_len)
    val_ds = SFTDataset(interleave_datasets(limited_val_datasets, probabilities=val_probs, seed=42), tokenizer, args.seq_len)
    collator = SFTDataCollator(tokenizer)
    train_sampler = DistributedSampler(train_ds, rank=rank, num_replicas=world_size, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=args.local_batch, sampler=train_sampler, collate_fn=collator)
    val_sampler = DistributedSampler(val_ds, rank=rank, num_replicas=world_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.local_batch, sampler=val_sampler, collate_fn=collator)

    # Optimizer, Scheduler, and Trainer State Resuming
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    accum_steps = args.batch_size // (args.local_batch * world_size)
    total_steps = (len(train_loader) // accum_steps) * args.epochs
    def lr_fn(step):
        if step < args.warmup_steps: return float(step) / float(max(1, args.warmup_steps))
        p = (step - args.warmup_steps) / (total_steps - args.warmup_steps)
        return max(args.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * p)))
    scheduler = LambdaLR(optimizer, lr_fn)
    
    start_epoch, global_step = 0, 0
    if args.resume_from_checkpoint:
        state_path = os.path.join(args.resume_from_checkpoint, "trainer_state.pt")
        if os.path.exists(state_path):
            if is_main: print(f"Resuming trainer state from {state_path}")
            state = torch.load(state_path, map_location=device)
            optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
            global_step, start_epoch = state["global_step"], state["epoch"]
            torch.set_rng_state(state["rng_state_cpu"].to("cpu")); torch.cuda.set_rng_state(state["rng_state_gpu"].to(device))
    
    # ✨ NEW: Initial Validation Step
    dist.barrier()
    if global_step == 0: # Only run validation if not resuming from a checkpoint
        if is_main: print("\nRunning initial validation before training...")
        initial_val_loss = run_validation(model, val_loader, device, mask_id, pad_id, limit=args.val_limit_batches)
        if is_main:
            print(f"Initial Validation Loss: {initial_val_loss:.4f}")
            if args.use_wandb:
                wandb.log({"validation/initial_loss": initial_val_loss, "step": 0})
    dist.barrier()

    # Training Loop
    if is_main: print("\nStarting SFT training...")
    model.train()
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, log_loss = calc_sft_loss(model, batch, mask_id, pad_id)
            if loss is not None: (loss / accum_steps).backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                global_step += 1

                if is_main:
                    print(f"E:{epoch+1} S:{global_step}/{total_steps} L:{log_loss:.4f} LR:{scheduler.get_last_lr()[0]:.2e}")
                    if args.use_wandb: wandb.log({"train/loss": log_loss, "train/lr": scheduler.get_last_lr()[0]}, step=global_step)

                if global_step > 0 and global_step % args.validation_interval == 0:
                    val_loss = run_validation(model, val_loader, device, mask_id, pad_id, args.val_limit_batches)
                    if is_main:
                        print(f"  Validation Loss @ S:{global_step}: {val_loss:.4f}")
                        if args.use_wandb: wandb.log({"validation/loss": val_loss}, step=global_step)
                    dist.barrier()

                if global_step > 0 and global_step % args.checkpoint_interval == 0:
                    dist.barrier()
                    if is_main:
                        ckpt_dir = os.path.join(args.output_dir, f"step-{global_step}")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        model.module.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "global_step": global_step, "epoch": epoch, "rng_state_cpu": torch.get_rng_state(), "rng_state_gpu": torch.cuda.get_rng_state()}
                        torch.save(state, os.path.join(ckpt_dir, "trainer_state.pt"))
                        print(f"  Checkpoint saved: {ckpt_dir}")
                    dist.barrier()

    # Final Save
    dist.barrier()
    if is_main:
        final_dir = os.path.join(args.output_dir, "final")
        model.module.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"SFT complete. Final model saved to {final_dir}")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()