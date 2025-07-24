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

        user_message = {"role": "user", "content": ex['user']}
        assistant_message = {"role": "assistant", "content": ex['assistant']}
        prompt_ids = self.tokenizer.apply_chat_template(
            [user_message],
            tokenize=True,
            add_generation_prompt=True
        )
        prompt_length = len(prompt_ids)

        input_ids = self.tokenizer.apply_chat_template(
            [user_message, assistant_message],
            tokenize=True,
            add_generation_prompt=False
        )[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "prompt_length": prompt_length,
        }

class SFTDataCollator:
    def __init__(self, tokenizer): self.pad_id = tokenizer.pad_token_id
    def __call__(self, features):
        input_ids = [ex['input_ids'] for ex in features]
        prompt_lengths = torch.tensor([ex['prompt_length'] for ex in features], dtype=torch.long)
        padded_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        return {'input_ids': padded_ids, 'prompt_lengths': prompt_lengths}

# -------- Core Functions --------
def forward_process(input_ids, mask_token_id, mask_ratio):
    b, l = input_ids.shape
    device = input_ids.device

    if isinstance(mask_ratio, torch.Tensor):
        p_mask = mask_ratio.to(device).unsqueeze(1).expand(b, l)
    else:
        p_mask = torch.full((b, l), mask_ratio, device=device)

    masked_indices = torch.bernoulli(p_mask).bool()
    return torch.where(masked_indices, mask_token_id, input_ids), masked_indices, p_mask

@torch.no_grad()
def print_trainable_parameters(model):
    trainable, total = 0, 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad: trainable += param.numel()
    print(f"trainable params: {trainable:,} || total params: {total:,} || trainable%: {100 * trainable / total:.2f}")

def calc_sft_loss(model, batch, mask_id, pad_id, eps, think_end_id, answer_weight):
    ids, p_lens = batch["input_ids"], batch["prompt_lengths"]
    dev, b, l = ids.device, ids.shape[0], ids.shape[1]
    if l == 0: return None, 0.0

    t = torch.rand(b, device=dev)
    mask_ratio_per_sequence = (1.0 - eps) * t + eps
    noisy_in, masked_pos, p_mask = forward_process(ids, mask_id, mask_ratio_per_sequence)

    pos = torch.arange(l, device=dev).unsqueeze(0).expand(b, l)
    p_mask_bool = pos < p_lens.unsqueeze(1)
    noisy_in[p_mask_bool] = ids[p_mask_bool]

    logits = model(input_ids=noisy_in).logits

    ans_mask = ~p_mask_bool
    loss_mask = ans_mask & masked_pos & (ids != pad_id)
    if not loss_mask.any(): return None, 0.0

    # is_answer_token_mask = torch.cumsum((ids == think_end_id), dim=1) > 0

    # weights = torch.ones(b, l, device=dev)
    # weights[is_answer_token_mask] = answer_weight

    loss = F.cross_entropy(logits[loss_mask], ids[loss_mask], reduction='none')
    norm_loss = loss / p_mask[loss_mask]

    total_ans_toks = torch.sum(ans_mask.int(), dim=1).sum()
    if total_ans_toks == 0:
        return None, 0.0

    final_loss = norm_loss.sum() / total_ans_toks
    return final_loss, final_loss.item()


@torch.no_grad()
def run_validation(model, loader, dev, mask_id, pad_id, eps, think_end_id, answer_weight, limit=None):
    model.eval()
    total_loss, count = 0.0, 0
    for i, batch in enumerate(loader):
        if limit and i >= limit: break
        batch = {k: v.to(dev) for k, v in batch.items()}
        loss_val = calc_sft_loss(model, batch, mask_id, pad_id, eps=eps, think_end_id=think_end_id, answer_weight=answer_weight)[1]
        if loss_val is not None and not math.isinf(loss_val):
            total_loss += loss_val
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
    parser.add_argument('--pretrained-model', type=str, default="./merged_model_good_base", help="Path to the base Hugging Face model.")
    parser.add_argument('--sft-data', type=str, default='./filtered_conversational_dataset/', help='Path to SFT DatasetDict.')
    parser.add_argument('--output-dir', type=str, default='./checkpoints/checkpoints_llada_nemotron_pretrain', help='Directory to save checkpoints and logs.')
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
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--warmup-ratio', type=float, default=0.1) # Changed to ratio
    parser.add_argument('--min-lr-ratio', type=float, default=0.1)
    parser.add_argument('--grad-clip-norm', type=float, default=1.0)
    # NEW: Argument for eps-based dynamic masking
    parser.add_argument('--eps', type=float, default=1e-3, help="Epsilon for dynamic masking, setting the minimum mask ratio.")
    parser.add_argument('--answer-weight', type=float, default=1.0, help="Weight to apply to the loss on tokens in the final answer (after </think>).")
    parser.add_argument('--validation-interval', type=int, default=100)
    parser.add_argument('--checkpoint-interval', type=int, default=100)
    parser.add_argument('--val-limit-batches', type=int, default=75)
    parser.add_argument('--use-wandb', action='store_true', default=True)
    parser.add_argument('--wandb-project', type=str, default='llada-sft')
    parser.add_argument('--wandb-run-id', type=str, default=None, help="W&B run ID to resume logging.")
    args = parser.parse_args()

    setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    is_main = rank == 0

    # --- W&B and State Resuming Logic ---
    wandb_run_id = args.wandb_run_id
    start_epoch, global_step = 0, 0
    start_dataloader_step = -1 # ADDED: Default to -1 (no skipping)

    if args.resume_from_checkpoint:
        state_path = os.path.join(args.resume_from_checkpoint, "trainer_state.pt")
        if os.path.exists(state_path):
            if is_main: print(f"Loading trainer state from {state_path}")
            state = torch.load(state_path, map_location='cpu')
            if not wandb_run_id:
                wandb_run_id = state.get("wandb_run_id")
            # ADDED: Load the dataloader step if it exists in the checkpoint
            start_dataloader_step = state.get("dataloader_step", -1)
        else:
            state = None
    else:
        state = None

    if is_main and args.use_wandb:
        if wandb_run_id:
            wandb.init(project=args.wandb_project, id=wandb_run_id, resume="must")
        else:
            wandb_run_id = wandb.util.generate_id()
            wandb.init(project=args.wandb_project, id=wandb_run_id, config=vars(args))

    # Layered Model & LoRA Loading Logic
    dtype = getattr(torch, args.torch_dtype)
    if is_main: print(f"Loading base model: {args.pretrained_model}")
    model = LLaDAModelLM.from_pretrained(
        args.pretrained_model,
        trust_remote_code=True,
        torch_dtype=dtype
    )

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)
    special_tokens_to_add = {
        "additional_special_tokens": ["<|mdm_mask|>", "<|start_header_id|>", "<|end_header_id|>","<|eot_id|>","<|begin_of_thought|>","<|end_of_thought|>" "<|begin_of_solution|>", "<|end_of_solution|>"]
    }

    if tokenizer.pad_token is None:
        special_tokens_to_add["pad_token"] = "<|pad|>"
    tokenizer.add_special_tokens(special_tokens_to_add)
    model.resize_token_embeddings(len(tokenizer))
    if is_main:
        print(f"Added special tokens and resized model embeddings. New vocabulary size: {len(tokenizer)}")
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")

    if args.sft_lora_weights:
        if is_main: print(f"Resuming SFT training from checkpoint: {args.sft_lora_weights}")
        model = PeftModel.from_pretrained(
            model,
            args.sft_lora_weights,
            local_files_only=True,
            adapter_name="sft_adapter"
        )
        if is_main: print("Loaded existing 'sft_adapter'.")
    else:
        if is_main: print("Starting a new SFT training run.")
        sft_lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout, bias='none', task_type='CAUSAL_LM'
        )
        if args.pretraining_lora_weights:
            if is_main: print(f"Applying non-trainable base LoRA from: {args.pretraining_lora_weights}")
            model = PeftModel.from_pretrained(
                model, args.pretraining_lora_weights, adapter_name="pretrained_lora", is_trainable=False
            )
            model.add_adapter(sft_lora_config, adapter_name="sft_adapter")
            if is_main: print("Added new trainable 'sft_adapter'.")
        else:
            model = get_peft_model(model, sft_lora_config, adapter_name="sft_adapter")
            if is_main: print("Created new trainable 'sft_adapter'.")

    model.set_adapter("sft_adapter")
    if is_main: print("Set 'sft_adapter' as the active adapter.")

    if is_main: print_trainable_parameters(model)
    model.to(device)

    mask_id = tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
    pad_id = tokenizer.pad_token_id

    if args.activation_checkpointing:
        if is_main: print("Enabling custom gradient checkpointing...")
        model.base_model.model.set_activation_checkpointing(strategy=ActivationCheckpointingStrategy.whole_layer)

    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    train_ds= SFTDataset(load_from_disk(args.sft_data)["train"], tokenizer=tokenizer, max_length=args.seq_len)
    val_ds = SFTDataset(load_from_disk(args.sft_data)["test"], tokenizer=tokenizer, max_length=args.seq_len)
    collator = SFTDataCollator(tokenizer)
    train_sampler = DistributedSampler(train_ds, rank=rank, num_replicas=world_size, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=args.local_batch, sampler=train_sampler, collate_fn=collator)
    val_sampler = DistributedSampler(val_ds, rank=rank, num_replicas=world_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.local_batch, sampler=val_sampler, collate_fn=collator)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),  # <-- Add this
        eps=1e-08             # <-- Add this
    )
    accum_steps = args.batch_size // (args.local_batch * world_size)
    total_steps = (len(train_loader) // accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio) 
    if is_main: print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    def lr_fn(step):
        if step < warmup_steps: return float(step) / float(max(1, warmup_steps)) # Use new variable
        p = (step - warmup_steps) / (total_steps - warmup_steps)
        return max(args.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * p)))
    scheduler = LambdaLR(optimizer, lr_fn)

    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        for opt_state in optimizer.state.values():
            for k, v in opt_state.items():
                if isinstance(v, torch.Tensor):
                    opt_state[k] = v.to(device)
        global_step = state["global_step"]
        start_epoch = state["epoch"]
        torch.set_rng_state(state["rng_state_cpu"])
        torch.cuda.set_rng_state(state["rng_state_gpu"], device=device)

    dist.barrier()
    if global_step == 0:
        if is_main: print("\nRunning initial validation before training...")
        initial_val_loss = run_validation(model, val_loader, device, mask_id, pad_id, eps=args.eps, limit=args.val_limit_batches, think_end_id=think_end_id, answer_weight=args.answer_weight)
        if is_main:
            print(f"Initial Validation Loss: {initial_val_loss:.4f}")
            if args.use_wandb: wandb.log({"validation/loss": initial_val_loss, "step": 0})
    dist.barrier()

    if is_main: print("\nStarting SFT training...")
    model.train()
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_loader):
            # ADDED: Logic to skip to the correct step on resume
            if epoch == start_epoch and step <= start_dataloader_step:
                print(step)
                continue

            batch = {k: v.to(device) for k, v in batch.items()}
            loss, log_loss = calc_sft_loss(model, batch, mask_id, pad_id, eps=args.eps, think_end_id=think_end_id, answer_weight=args.answer_weight)
            if loss is not None: (loss / accum_steps).backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                global_step += 1

                if is_main:
                    print(f"E:{epoch+1} S:{global_step}/{total_steps} L:{log_loss:.4f} LR:{scheduler.get_last_lr()[0]:.2e}")
                    if args.use_wandb: wandb.log({"train/loss": log_loss, "train/lr": scheduler.get_last_lr()[0]}, step=global_step)

                if global_step > 0 and global_step % args.validation_interval == 0:
                    val_loss = run_validation(model, val_loader, device, mask_id, pad_id, eps=args.eps, limit=args.val_limit_batches, think_end_id=think_end_id, answer_weight=args.answer_weight)
                    if is_main:
                        print(f"   Validation Loss @ S:{global_step}: {val_loss:.4f}")
                        if args.use_wandb: wandb.log({"validation/loss": val_loss}, step=global_step)
                    dist.barrier()

                if global_step > 0 and global_step % args.checkpoint_interval == 0:
                    dist.barrier()
                    if is_main:
                        ckpt_dir = os.path.join(args.output_dir, f"step-{global_step}")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        model.module.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        state = {
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "global_step": global_step,
                            "epoch": epoch,
                            "dataloader_step": step, # MODIFIED: Save the dataloader step
                            "rng_state_cpu": torch.get_rng_state(),
                            "rng_state_gpu": torch.cuda.get_rng_state(),
                            "wandb_run_id": wandb_run_id
                        }
                        torch.save(state, os.path.join(ckpt_dir, "trainer_state.pt"))
                        print(f"   Checkpoint saved: {ckpt_dir}")
                    dist.barrier()
        # MODIFIED: Reset dataloader step counter after the first epoch of a resumed run is complete
        start_dataloader_step = -1


    dist.barrier()
    if is_main:
        final_dir = os.path.join(args.output_dir, "final")
        model.module.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"SFT complete. Final model saved to {final_dir}")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()