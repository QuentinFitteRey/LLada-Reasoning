#!/usr/bin/env python
import os
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
import argparse
from llada_local.modeling_llada import LLaDAModelLM, LLaDAConfig

# Helpers
def random_remask(masked_indices, mask_ratio):
    # randomly pick a subset of the currently masked positions to stay masked
    b, l = masked_indices.shape
    # flatten & sample
    flat = masked_indices.view(-1)
    keep = torch.rand_like(flat, dtype=torch.float) < mask_ratio
    new_flat = flat & keep
    return new_flat.view(b, l)

def low_confidence_remask(masked_indices, logits, mask_ratio):
    # among the masked positions, compute confidence = max prob
    probs = F.softmax(logits, dim=-1)
    b, l, v = probs.shape
    # get max prob at each position
    conf, _ = probs.max(dim=-1)               # [b, l]
    conf_masked = conf.masked_fill(~masked_indices, 1.0)
    # we want to remask the fraction mask_ratio of lowest-confidence tokens
    k = int((masked_indices.sum() * mask_ratio).item())
    if k == 0:
        return masked_indices
    # flatten and find the k smallest
    flat_conf = conf_masked.view(-1)
    idx = torch.topk(-flat_conf, k).indices     # indices of lowest conf
    new_flat = torch.zeros_like(flat_conf, dtype=torch.bool)
    new_flat[idx] = True
    return new_flat.view(b, l)

# 1. CLI and config parsing
parser = argparse.ArgumentParser(description="Pretrain LLaDA with extended context")
parser.add_argument("--train-data",       type=str, required=True)
parser.add_argument("--val-data",         type=str, required=True)
parser.add_argument("--output-dir",       type=str, default="./checkpoints")
parser.add_argument("--model-config",     type=str, help="Path to modified LLaDAConfig or JSON")
parser.add_argument("--pretrained-model", type=str, default=None,
                    help="HuggingFace or local checkpoint to initialize from")
parser.add_argument("--seq-len",          type=int, default=4096, help="Extended context length")
parser.add_argument("--batch-size",       type=int, default=1280)
parser.add_argument("--local-batch",      type=int, default=4)
parser.add_argument("--lr",               type=float, default=4e-4)
parser.add_argument("--min-lr",           type=float, default=1e-5)
parser.add_argument("--weight-decay",     type=float, default=0.1)
parser.add_argument("--warmup-steps",     type=int, default=2000)
parser.add_argument("--total-tokens-1",   type=float, default=1.2e12)
parser.add_argument("--total-tokens-2",   type=float, default=2.0e12)
parser.add_argument("--total-tokens-3",   type=float, default=2.3e12)
parser.add_argument("--validation-interval", type=int, default=100)
parser.add_argument("--mask-strategy",    choices=["random","low_conf"], default="low_conf")
parser.add_argument("--sampling-method",  choices=["fixed_length","semi_pad","semi_origin"], default="fixed_length")
parser.add_argument("--epochs",           type=int, default=1)
args = parser.parse_args()

# 2. Load or build config
config = LLaDAConfig(
    vocab_size=126352,
    embedding_size=126352,
    d_model=4096,
    n_layers=32,
    n_heads=32,
    rope=True,
    alibi=False,
    flash_attention=True,
    block_type='llama',
)

# 3. Dataset + Collator
class IterableTextDataset(IterableDataset):
    def __init__(self, path, tokenizer, seq_len):
        self.path = path
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buf = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                ids = self.tokenizer(line, add_special_tokens=False).input_ids
                buf.extend(ids)
                while len(buf) >= self.seq_len:
                    yield torch.tensor(buf[:self.seq_len], dtype=torch.long)
                    buf = buf[self.seq_len:]

class DataCollator:
    def __call__(self, examples):
        return {"input_ids": torch.stack(examples)}

# 4. Forward + Masking
def forward_process(input_ids, mask_token_id, mask_ratio):
    b, l = input_ids.shape
    if isinstance(mask_ratio, float):
        p_mask = torch.full((b, l), mask_ratio, device=input_ids.device)
    else:
        p_mask = mask_ratio[:, None].expand(-1, l)
    masked = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy = torch.where(masked, mask_token_id, input_ids)
    return noisy, masked, p_mask

# 5. Loss
def calc_loss(model, batch, mask_token_id, pad_token_id, eps=1e-3):
    input_ids = batch["input_ids"]
    b, l = input_ids.shape
    device = input_ids.device

    # 1% variable-length hack
    if random.random() < 0.01:
        rnd = torch.randint(1, l+1, (1,)).item()
        input_ids = input_ids[:, :rnd]

    t = torch.rand(b, device=device)
    mask_ratio = (1 - eps) * t + eps
    noisy_input, masked, p_mask = forward_process(input_ids, mask_token_id, mask_ratio)
    logits = model(input_ids=noisy_input).logits

    if args.mask_strategy == "low_conf":
        masked = low_confidence_remask(masked, logits, mask_ratio)
    else:
        masked = random_remask(masked, mask_ratio)

    # now compute loss only on masked positions
    masked &= (input_ids != pad_token_id)

    if not masked.any(): 
        return None

    logits = model(input_ids=noisy).logits
    losses = F.cross_entropy(logits[masked], input_ids[masked], reduction="none")
    weighted = losses / p_mask[masked]
    return weighted.sum() / (b * l)

# 6. Scheduler
def get_scheduler(opt):
    total_steps1 = int(args.total_tokens_1 / (args.batch_size * args.seq_len))
    total_steps2 = int(args.total_tokens_2 / (args.batch_size * args.seq_len))
    total_steps3 = int(args.total_tokens_3 / (args.batch_size * args.seq_len))
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        if step < total_steps1:
            return 1.0
        if step < total_steps2:
            return 0.25
        # linear decay
        progress = (step - total_steps2) / max(1, (total_steps3 - total_steps2))
        return (0.25 - progress * (0.25 - args.min_lr/args.lr))
    return LambdaLR(opt, lr_lambda)

# 7. Main
def main():
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model + Tokenizer
    model = LLaDAModelLM(config, init_params=not args.pretrained_model)
    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model), strict=False)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model or "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )
    mask_id = tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
    pad_id  = tokenizer.pad_token_id

    # Optimizer & Scheduler
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = get_scheduler(opt)

    # DataLoaders
    train_ds = IterableTextDataset(args.train_data, tokenizer, args.seq_len)
    val_ds   = IterableTextDataset(args.val_data, tokenizer, args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.local_batch, collate_fn=DataCollator())
    val_loader   = DataLoader(val_ds,   batch_size=args.local_batch, collate_fn=DataCollator())

    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            global_step += 1
            batch = {k: v.to(device) for k,v in batch.items()}
            opt.zero_grad()
            loss = calc_loss(model, batch, mask_id, pad_id)
            if loss is not None:
                loss.backward()
                opt.step()
                sch.step()

            if global_step % args.validation_interval == 0:
                # (you can slot in your validation function here)
                print(f"[Step {global_step}] loss = {loss:.4f}, lr = {sch.get_last_lr()[0]:.2e}")

            if global_step % 10000 == 0:
                ckpt = os.path.join(args.output_dir, f"step-{global_step}")
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)

    print("Pre-training complete.")

if __name__ == "__main__":
    main()