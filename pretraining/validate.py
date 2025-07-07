# evaluate_model.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import argparse
import time
import math

from llada_local.modeling_llada import LLaDAModelLM
from peft import PeftModel

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence

def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

class LineByLineTextDataset(Dataset):
    """A simple dataset that reads a text file line by line."""
    def __init__(self, path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(path, encoding="utf-8") as f:
            self.examples = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        line = self.examples[i]
        tokenized = self.tokenizer(
            line,
            add_special_tokens=True,
            max_length=self.max_len,
            padding=False,
            truncation=True
        )
        return torch.tensor(tokenized.input_ids, dtype=torch.long)

class DataCollatorWithPadding:
    """Pads sequences to the length of the longest sequence in a batch."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        padded_batch = pad_sequence(examples, batch_first=True, padding_value=self.pad_token_id)
        return {"input_ids": padded_batch}

def forward_process(input_ids, mask_token_id, mask_ratio):
    b, l = input_ids.shape
    if isinstance(mask_ratio, torch.Tensor) and mask_ratio.ndim == 1: p_mask = mask_ratio.view(b, 1).expand(b, l)
    else: p_mask = torch.full((b, l), mask_ratio, device=input_ids.device)
    masked_indices = torch.rand_like(p_mask, dtype=torch.float) < p_mask
    noisy_input = torch.where(masked_indices, mask_token_id, input_ids)
    return noisy_input, masked_indices, p_mask

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

@torch.no_grad()
def run_validation(model, val_loader, device, mask_id, pad_id, limit_batches, mc_runs):
    model.eval()
    total_avg_loss = 0.0
    total_batches = 0
    is_main_process = dist.get_rank() == 0

    # Each rank processes its own shard of the validation data
    for i, batch in enumerate(val_loader):
        if is_main_process and i % 10 == 0:
             print(f"  Processing validation batch {i + 1}/{len(val_loader)}...")
        if limit_batches and i >= limit_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        mc_losses = []
        for _ in range(mc_runs):
            _, log_loss = calc_loss(model, batch, mask_id, pad_id)
            if log_loss is not None and not math.isinf(log_loss):
                mc_losses.append(log_loss)
        
        if mc_losses:
            total_avg_loss += sum(mc_losses) / len(mc_losses)
            total_batches += 1

    # Synchronize results across all GPUs
    local_stats = torch.tensor([total_avg_loss, total_batches], dtype=torch.float64, device=device)
    dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)
    
    global_total_loss, global_total_batches = local_stats.tolist()
    
    final_avg_loss = global_total_loss / global_total_batches if global_total_batches > 0 else 0.0
    return final_avg_loss

def main():
    parser = argparse.ArgumentParser(description="Evaluate a LLaDA model")
    # --- Simplified arguments for evaluation ---
    parser.add_argument("--model-path", type=str, default="./llada_local")
    parser.add_argument("--base-model-path", type=str, default=None, help="Path to the base model, required if loading a LoRA adapter.")
    parser.add_argument("--val-data", type=str, default="./data_pretrain/val.txt", help="Path to the validation data file.")
    parser.add_argument("--seq-len", type=int, default=8192, help="Maximum sequence length for evaluation (e.g., 4096 or 8192).")
    parser.add_argument("--local-batch", type=int, default=8, help="Local batch size (per-device).")
    parser.add_argument("--val-mc-runs", type=int, default=1, help="Number of Monte Carlo runs for a more stable validation loss.")
    parser.add_argument("--val-limit-batches", type=int, default=75, help="Limit validation to N batches for a quick check.")
    args = parser.parse_args()
    
    setup_ddp()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    is_main_process = rank == 0

    # --- Model and Tokenizer Loading ---
    tokenizer_path = args.base_model_path if args.base_model_path else args.model_path
    if is_main_process: print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if args.base_model_path:
        if is_main_process:
            print(f"Loading base model from: {args.base_model_path}")
            print(f"Applying LoRA adapter from: {args.model_path}")
        model = LLaDAModelLM.from_pretrained(args.base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, args.model_path, is_trainable=True)
    else:
        if is_main_process: print(f"Loading full model from: {args.model_path}")
        model = LLaDAModelLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

    model.to(device)

    # --- Handle Special Tokens ---
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
    
    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    # --- Dataset and DataLoader ---
    val_ds = LineByLineTextDataset(args.val_data, tokenizer, args.seq_len)
    collator = DataCollatorWithPadding(pad_token_id=pad_id)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.local_batch, collate_fn=collator, sampler=val_sampler)

    # --- Run Validation ---
    if is_main_process:
        print("\nStarting validation...")
        print(f"Model: {args.model_path}")
        print(f"Dataset: {args.val_data}")
        print(f"Context Length: {args.seq_len}\n")

    val_loss = run_validation(model, val_loader, device, mask_id, pad_id, args.val_limit_batches, args.val_mc_runs)

    # --- Print Final Result ---
    if is_main_process:
        print("\n" + "="*50)
        print("          Validation Complete")
        print("="*50)
        print(f"  Model Path: {args.model_path}")
        print(f"  Sequence Length: {args.seq_len}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print("="*50 + "\n")
        
    cleanup_ddp()

if __name__ == "__main__":
    main()