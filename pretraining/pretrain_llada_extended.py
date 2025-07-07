# (Imports and parser remain the same)
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
import argparse
import time
import random 
import math

import wandb

from llada_local.modeling_llada import LLaDAModelLM
from peft import LoraConfig, get_peft_model, PeftModel
from llada_local.configuration_llada import ActivationCheckpointingStrategy

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence

# --- (Parser definition remains the same as your last version) ---
parser = argparse.ArgumentParser(description="Fine-tune LLaDA for extended context")
parser.add_argument("--train-data",          type=str, required=False, help="Path to the training data file", default="./data_pretrain_stratified/train_new.txt")
parser.add_argument("--val-data",            type=str, required=False, help="Path to the validation data file", default="./data_pretrain_stratified/val_new.txt")
parser.add_argument("--output-dir",          type=str, default="./checkpoints/checkpoints_llada_pretrain_8k_clip_lr15_betterval_128_good_newdata", help="Directory to save checkpoints and final model.")
parser.add_argument("--pretrained-model",    type=str, required=False, help="Path to the base model to fine-tune", default="./llada_local")
parser.add_argument("--resume-from-checkpoint", type=str, default="", help="Path to a checkpoint directory to resume training from.")
parser.add_argument("--batch-size",          type=int, default=64, help="Global batch size (for gradient accumulation)")
parser.add_argument("--local-batch",         type=int, default=8, help="Local batch size (per-device)")
parser.add_argument("--seq-len",             type=int, default=8192, help="Maximum sequence length for training (will truncate longer sequences)")
parser.add_argument("--lr",                  type=float, default=1e-5, help="Peak learning rate for fine-tuning")
parser.add_argument("--weight-decay",        type=float, default=0.0, help="Weight decay for the optimizer.")
parser.add_argument("--max-train-steps",     type=int, default=500, help="Total number of training steps for the fine-tuning run.")
# --- LoRA Parameters ---
parser.add_argument("--use-lora",            action='store_true', default=True, help="Enable LoRA for parameter-efficient fine-tuning.")
parser.add_argument("--lora-r",              type=int, default=128, help="The rank of the LoRA matrices.")
parser.add_argument("--lora-alpha",          type=int, default=256, help="The scaling factor for LoRA matrices (often 2*r).")
parser.add_argument("--lora-dropout",        type=float, default=0.05, help="Dropout probability for LoRA layers.")
parser.add_argument("--lora-target-modules", type=str, nargs='+', default=['q_proj', 'v_proj', 'o_proj', 'k_proj'], help="Modules to apply LoRA to.")
# --- Training Stability Parameters ---
parser.add_argument("--warmup-steps",        type=int, default=20, help="Number of warmup steps for the learning rate scheduler.")
parser.add_argument("--min-lr-ratio",        type=float, default=0.1, help="The learning rate will decay to this ratio of the peak LR (lr * min_lr_ratio).")
parser.add_argument("--grad-clip-norm",      type=float, default=1.0, help="Maximum norm for gradient clipping.")
parser.add_argument("--val-mc-runs",         type=int, default=1, help="Number of Monte Carlo runs for a stable validation loss.")
# --- Logging and Checkpointing ---
parser.add_argument("--validation-interval", type=int, default=32)
parser.add_argument("--checkpoint-interval", type=int, default=32)
parser.add_argument("--val-limit-batches",   type=int, default=75, help="Number of batches to run validation on.")
parser.add_argument("--epochs",              type=int, default=1)
parser.add_argument("--activation-checkpointing", action='store_true', default=True, help="Enable activation checkpointing for memory efficiency.")
parser.add_argument("--use-wandb",           action='store_true', default=True, help="Enable Weights & Biases logging.")
parser.add_argument("--wandb-project",       type=str, default="llada-pretrain", help="W&B project name.")
parser.add_argument("--wandb-entity",        type=str, default=None, help="W&B entity (username or team).")
parser.add_argument("--wandb-run-name",      type=str, default=None, help="A name for this specific W&B run.")
args = parser.parse_args()


# MODIFIED: Your previous version of the Dataset is fine.
class LineByLineTextDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(path, encoding="utf-8") as f:
            self.examples = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        line = self.examples[i]
        # Keep random length sampling commented out as per your last version
        target_seq_len = self.max_len
        tokenized = self.tokenizer(
            line,
            add_special_tokens=True,
            max_length=target_seq_len,
            padding=False,
            truncation=True
        )
        return torch.tensor(tokenized.input_ids, dtype=torch.long)

# The fixed DataCollator is not needed if not using random lengths, but we keep it for correctness
class DataCollatorWithPadding:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    def __call__(self, examples):
        # NOTE: Since random length is off, this will just pad to max_len, which is fine.
        # If you re-enable random lengths, this distributed-aware collator is required.
        padded_batch = pad_sequence(examples, batch_first=True, padding_value=self.pad_token_id)
        return {"input_ids": padded_batch}

# --- (calc_loss, forward_process, print_trainable_parameters are unchanged) ---
def calc_loss(model, batch, mask_token_id, pad_token_id, eps=1e-3):
    input_ids = batch["input_ids"]
    b, l = input_ids.shape
    device = input_ids.device
    if l == 0: return None, 0.0
    t = torch.rand(b, device=device)
    mask_ratio = (1 - eps) * t + eps
    noisy_input, masked, p_mask = forward_process(input_ids, mask_token_id, mask_ratio)
    logits = model(input_ids=noisy_input).logits
    masked &= (input_ids != pad_token_id)
    if not masked.any(): return None, 0.0
    losses = F.cross_entropy(logits[masked], input_ids[masked], reduction="none")
    weighted = losses / p_mask[masked]
    final_loss = weighted.sum() / (b * l)
    return final_loss, final_loss.item()

def forward_process(input_ids, mask_token_id, mask_ratio):
    b, l = input_ids.shape
    if isinstance(mask_ratio, torch.Tensor) and mask_ratio.ndim == 1: p_mask = mask_ratio.view(b, 1).expand(b, l)
    else: p_mask = torch.full((b, l), mask_ratio, device=input_ids.device)
    masked_indices = torch.rand_like(p_mask, dtype=torch.float) < p_mask
    noisy_input = torch.where(masked_indices, mask_token_id, input_ids)
    return noisy_input, masked_indices, p_mask

def print_trainable_parameters(model):
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad: trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")


# NEW: Revamped validation function for correct distributed evaluation
@torch.no_grad()
def run_validation(model, val_loader, device, mask_id, pad_id, limit_batches, mc_runs):
    model.eval()
    total_avg_loss = 0.0
    total_batches = 0
    
    # Each rank processes its own shard of the validation data
    for i, batch in enumerate(val_loader):
        print(f"[Rank {dist.get_rank()}] Processing validation batch {i + 1}/{len(val_loader)}")
        if i >= limit_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        mc_losses = []
        
        for _ in range(mc_runs):
            _, log_loss = calc_loss(model, batch, mask_id, pad_id)
            if log_loss is not None and not math.isinf(log_loss):
                mc_losses.append(log_loss)
        
        if mc_losses:
            batch_avg_loss = sum(mc_losses) / len(mc_losses)
            total_avg_loss += batch_avg_loss
            total_batches += 1

    # Place the calculated average loss for this rank into a tensor
    local_avg_loss = torch.tensor(total_avg_loss / total_batches if total_batches > 0 else 0.0, device=device)
    
    # Sum the losses from all ranks and count how many ranks participated
    dist.all_reduce(local_avg_loss, op=dist.ReduceOp.SUM)
    world_size = dist.get_world_size()

    # Divide by the number of ranks to get the true average loss across all data
    global_avg_val_loss = local_avg_loss / world_size
    
    model.train()
    return global_avg_val_loss.item()


def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

def main():
    setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    is_main_process = rank == 0
    
    # --- (WandB setup is unchanged) ---
    wandb_run_id = None
    if is_main_process and args.use_wandb:
        if args.resume_from_checkpoint:
            trainer_state_path = os.path.join(args.resume_from_checkpoint, "trainer_state.pt")
            if os.path.exists(trainer_state_path):
                state = torch.load(trainer_state_path, map_location="cpu")
                if "wandb_run_id" in state:
                    wandb_run_id = state["wandb_run_id"]
        
        run_name = args.wandb_run_name if args.wandb_run_name else f"llada-finetune-{int(time.time())}"

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=wandb_run_id, 
            name=run_name,
            config=args,
            resume="allow"  
        )

    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- (Model and Tokenizer setup is unchanged) ---
    accumulation_steps = args.batch_size // (args.local_batch * world_size)
    
    if not args.pretrained_model:
        raise ValueError("A --pretrained-model must be provided.")
    
    model = LLaDAModelLM.from_pretrained(
        args.pretrained_model, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    if args.use_lora:
        if args.resume_from_checkpoint:
            if is_main_process:
                print(f"Resuming training by loading LoRA adapter from: {args.resume_from_checkpoint}")
            model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
        else:
            if is_main_process: print("LoRA enabled. Preparing model for PEFT...")
            lora_config = LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules, lora_dropout=args.lora_dropout,
                bias="none", task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)

    model.to(device) 

    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "<|pad|>"
    if tokenizer.convert_tokens_to_ids("<|mdm_mask|>") == tokenizer.unk_token_id:
        special_tokens_dict["additional_special_tokens"] = ["<|mdm_mask|>"]
    
    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

    mask_id = tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
    pad_id = tokenizer.pad_token_id

    if args.activation_checkpointing:
        if is_main_process: print("Enabling custom gradient checkpointing...")
        model.base_model.model.set_activation_checkpointing(strategy=ActivationCheckpointingStrategy.whole_layer)

    if is_main_process:
        print_trainable_parameters(model)

    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    # --- (Optimizer, Scheduler, and Checkpoint Resuming are unchanged) ---
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(current_step: int):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        progress = float(current_step - args.warmup_steps) / float(max(1, args.max_train_steps - args.warmup_steps))
        progress = min(1.0, progress)
        return 1.0 - progress * (1.0 - args.min_lr_ratio)
    sch = LambdaLR(opt, lr_lambda)

    start_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint:
        trainer_state_path = os.path.join(args.resume_from_checkpoint, "trainer_state.pt")
        if os.path.exists(trainer_state_path):
            if is_main_process:
                print(f"Loading trainer state from {trainer_state_path}")
            state = torch.load(trainer_state_path, map_location="cpu")
            opt.load_state_dict(state["optimizer"])
            sch.load_state_dict(state["scheduler"])
            for opt_state in opt.state.values():
                for k, v in opt_state.items():
                    if isinstance(v, torch.Tensor):
                        opt_state[k] = v.to(device)
            global_step = state["global_step"]
            start_epoch = state["epoch"]
            torch.set_rng_state(state["rng_state_cpu"])
            torch.cuda.set_rng_state(state["rng_state_gpu"], device=device)
            if is_main_process:
                print(f"Resumed training from Step {global_step} at Epoch {start_epoch}.")
        else:
            if is_main_process:
                print(f"Warning: --resume-from-checkpoint provided but trainer_state.pt not found in {args.resume_from_checkpoint}. Starting from scratch.")
    
    train_ds = LineByLineTextDataset(args.train_data, tokenizer, args.seq_len)
    val_ds = LineByLineTextDataset(args.val_data, tokenizer, args.seq_len)
    collator = DataCollatorWithPadding(pad_token_id=pad_id)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=args.local_batch, collate_fn=collator, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=args.local_batch, collate_fn=collator, sampler=val_sampler)

    # --- CORRECTED INITIAL VALIDATION ---
    dist.barrier()
    if global_step == 0:
        # Now, all processes run validation in parallel
        initial_val_loss = run_validation(model, val_loader, device, mask_id, pad_id, args.val_limit_batches, args.val_mc_runs)
        
        # But only the main process prints and logs
        if is_main_process:
            print("Running initial validation...")
            print(f"Initial Pre-training Validation Loss: {initial_val_loss:.4f}")
            if args.use_wandb:
                wandb.log({"validation/loss": initial_val_loss, "step": 0})
    
    # Add a barrier here to prevent the deadlock
    dist.barrier()
    if is_main_process: print("Starting fine-tuning...")
    
    model.train()
    opt.zero_grad()
    
    training_is_done = False
    for epoch in range(start_epoch, args.epochs):
        if training_is_done: break
        
        train_loader.sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            loss, log_loss = calc_loss(model, batch, mask_id, pad_id)
            
            if loss is not None:
                (loss / accumulation_steps).backward()

            if (i + 1) % accumulation_steps == 0:
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

                opt.step()
                sch.step()
                opt.zero_grad()
                global_step += 1
                
                if is_main_process:
                    print(f"[Epoch {epoch+1}][Step {global_step}/{args.max_train_steps}] Train Loss = {log_loss:.4f}, LR = {sch.get_last_lr()[0]:.2e}")

                    if args.use_wandb:
                        wandb.log({
                            "train/loss": log_loss,
                            "train/learning_rate": sch.get_last_lr()[0],
                            "step": global_step,
                            "epoch": epoch + 1
                        })

                if global_step > 0 and global_step % args.validation_interval == 0:
                    # All processes run validation in parallel
                    val_loss = run_validation(model, val_loader, device, mask_id, pad_id, args.val_limit_batches, args.val_mc_runs)
                    
                    # Only the main process prints and logs the final averaged result
                    if is_main_process:
                        print(f"  Validation Loss @ Step {global_step}: {val_loss:.4f}")
                        if args.use_wandb:
                            wandb.log({
                                "validation/loss": val_loss,
                                "step": global_step
                            })
                    dist.barrier()


                if global_step > 0 and global_step % args.checkpoint_interval == 0:
                    if is_main_process:
                        ckpt_dir = os.path.join(args.output_dir, f"step-{global_step}")
                        print(f"  Saving checkpoint to {ckpt_dir}")
                        model.module.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)

                        trainer_state = {
                            "optimizer": opt.state_dict(), "scheduler": sch.state_dict(),
                            "global_step": global_step, "epoch": epoch,
                            "rng_state_cpu": torch.get_rng_state(), "rng_state_gpu": torch.cuda.get_rng_state(),
                        }
                        if args.use_wandb:
                            trainer_state["wandb_run_id"] = wandb.run.id
                        
                        torch.save(trainer_state, os.path.join(ckpt_dir, "trainer_state.pt"))
                    dist.barrier()

            if global_step >= args.max_train_steps:
                training_is_done = True
                break

    # --- (Final checkpoint saving logic is unchanged) ---
    dist.barrier()
    if is_main_process:
        print("Fine-tuning complete.")
        final_ckpt_dir = os.path.join(args.output_dir, "final_checkpoint")
        print(f"Saving final model to {final_ckpt_dir}")
        
        unwrapped_model = model.module
        if args.use_lora:
            print("Merging LoRA weights and saving the final, standalone model...")
            merged_model = unwrapped_model.merge_and_unload()
            merged_model.save_pretrained(final_ckpt_dir)
        else:
            unwrapped_model.save_pretrained(final_ckpt_dir)
        
        tokenizer.save_pretrained(final_ckpt_dir)

    if is_main_process and args.use_wandb:
        wandb.finish()

    cleanup_ddp()

if __name__ == "__main__":
    main()