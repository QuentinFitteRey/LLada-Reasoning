import os
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import argparse
from llada_local.modeling_llada import LLaDAModelLM, LLaDAConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from llada_local.configuration_llada import ActivationCheckpointingStrategy


parser = argparse.ArgumentParser(description="Fine-tune LLaDA for extended context")
parser.add_argument("--train-data",       type=str, required=False, help="Path to the training data file (e.g., ./data_long_context/train.txt)",default="./data/train.txt")
parser.add_argument("--val-data",         type=str, required=False, help="Path to the validation data file (e.g., ./data_long_context/val.txt)",default="./data/val.txt")
parser.add_argument("--output-dir",       type=str, default="./checkpoints_finetuned")
parser.add_argument("--pretrained-model", type=str, required=False, help="Path to the base model to fine-tune",default="./llada_local")
parser.add_argument("--batch-size",       type=int, default=64, help="Global batch size (for gradient accumulation)")
parser.add_argument("--local-batch",      type=int, default=8, help="Local batch size (per-device)")
parser.add_argument("--seq-len",         type=int, default=8192, help="Sequence length for training (must match model's context size)")
parser.add_argument("--lr",               type=float, default=2e-5, help="Peak learning rate for fine-tuning")
parser.add_argument("--weight-decay",     type=float, default=0.1)
parser.add_argument("--max-train-steps",  type=int, default=500, help="Total number of training steps for the fine-tuning run.")
# --- LoRA Parameters ---
parser.add_argument("--use-lora",         action='store_true',default=True, help="Enable LoRA for parameter-efficient fine-tuning.")
parser.add_argument("--lora-r",           type=int, default=16, help="The rank of the LoRA matrices.")
parser.add_argument("--lora-alpha",       type=int, default=32, help="The scaling factor for LoRA matrices (often 2*r).")
parser.add_argument("--lora-dropout",     type=float, default=0.05, help="Dropout probability for LoRA layers.")
parser.add_argument("--lora-target-modules", type=str, nargs='+', default=['q_proj', 'v_proj'], help="Modules to apply LoRA to.")
# --- Training Parameters ---
parser.add_argument("--warmup-steps",     type=int, default=50, help="Number of warmup steps for the learning rate scheduler.")
parser.add_argument("--validation-interval", type=int, default=50)
parser.add_argument("--checkpoint-interval", type=int, default=100)
parser.add_argument("--epochs",           type=int, default=1)
parser.add_argument("--activation-checkpointing", action='store_true', default=True, help="Enable activation checkpointing for memory efficiency.")
args = parser.parse_args()


# 2. Dataset + Collator
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

def calc_loss(model, batch, mask_token_id, pad_token_id, eps=1e-3):
    input_ids = batch["input_ids"]
    b, l = input_ids.shape
    device = input_ids.device
    if l == 0: return None, 0.0

    t = torch.rand(b, device=device)
    mask_ratio = (1 - eps) * t + eps
    
    noisy_input, masked, p_mask = forward_process(input_ids, mask_token_id, mask_ratio)
    
    # A single, efficient forward pass.
    logits = model(input_ids=noisy_input).logits

    masked &= (input_ids != pad_token_id)

    if not masked.any():
        return None, 0.0

    losses = F.cross_entropy(logits[masked], input_ids[masked], reduction="none")
    weighted = losses / p_mask[masked]
    final_loss = weighted.sum() / (b * l)

    return final_loss, final_loss.item()

def forward_process(input_ids, mask_token_id, mask_ratio):
    b, l = input_ids.shape
    if isinstance(mask_ratio, torch.Tensor) and mask_ratio.ndim == 1:
        p_mask = mask_ratio.view(b, 1).expand(b, l)
    else:
        p_mask = torch.full((b, l), mask_ratio, device=input_ids.device)

    masked_indices = torch.rand_like(p_mask, dtype=torch.float) < p_mask
    noisy_input = torch.where(masked_indices, mask_token_id, input_ids)
    return noisy_input, masked_indices, p_mask


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def run_validation(model, val_loader, device, mask_id, pad_id, limit_batches=50):
    model.eval()
    val_loss_accum = 0.0
    val_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _, log_loss = calc_loss(model, batch, mask_id, pad_id)
            val_loss_accum += log_loss
            val_batches += 1
            if val_batches >= limit_batches:
                break
    avg_val_loss = val_loss_accum / val_batches if val_batches > 0 else 0.0
    model.train()
    return avg_val_loss

def main():
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    assert args.batch_size % args.local_batch == 0, "Global batch size must be divisible by local batch size."
    accumulation_steps = args.batch_size // args.local_batch

    if not args.pretrained_model:
        raise ValueError("A --pretrained-model must be provided.")
    
    model = LLaDAModelLM.from_pretrained(args.pretrained_model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
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

    if args.use_lora:
        print("LoRA enabled. Preparing model for PEFT...")
        # prepare_model_for_kbit_training is good practice for PEFT, even if not using k-bit quantization
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM" ,
        )
        model = get_peft_model(model, lora_config)
        print("Model converted to PEFT model with LoRA adapters.")

    if args.activation_checkpointing:
        print("Enabling custom gradient checkpointing...")
        model.base_model.model.set_activation_checkpointing(strategy=ActivationCheckpointingStrategy.whole_layer)

    print_trainable_parameters(model)

    # Verify model dtype for FlashAttention compatibility
    print(f"Model dtype: {model.dtype}")
    if model.dtype not in [torch.float16, torch.bfloat16]:
        print("Warning: Model is not in fp16/bf16, converting to bfloat16 for FlashAttention compatibility")
        model = model.to(torch.bfloat16)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print(f"Setting up cosine scheduler for {args.max_train_steps} steps with {args.warmup_steps} warmup steps.")
    sch = get_cosine_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps
    )

    train_ds = IterableTextDataset(args.train_data, tokenizer, args.seq_len)
    val_ds   = IterableTextDataset(args.val_data, tokenizer, args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.local_batch, collate_fn=DataCollator())
    val_loader   = DataLoader(val_ds,   batch_size=args.local_batch, collate_fn=DataCollator())

    print("Running initial validation on the extended-context model before training...")
    initial_val_loss = run_validation(model, val_loader, device, mask_id, pad_id)
    print(f"Initial Pre-training Validation Loss: {initial_val_loss:.4f}")
    
    print("Starting fine-tuning...")
    global_step = 0
    model.train()
    opt.zero_grad()
    
    training_is_done = False
    for epoch in range(args.epochs):
        if training_is_done:
            break
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            loss, log_loss = calc_loss(model, batch, mask_id, pad_id)
            
            if loss is not None:
                loss = loss / accumulation_steps
                loss.backward()

            if (i + 1) % accumulation_steps == 0:
                opt.step()
                sch.step()
                opt.zero_grad()
                global_step += 1
                
                print(f"[Step {global_step}/{args.max_train_steps}] Train Loss = {log_loss:.4f}, LR = {sch.get_last_lr()[0]:.2e}")

                if global_step % args.validation_interval == 0:
                    val_loss = run_validation(model, val_loader, device, mask_id, pad_id)
                    print(f"  Validation Loss @ Step {global_step}: {val_loss:.4f}")

                if global_step % args.checkpoint_interval == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"step-{global_step}")
                    print(f"  Saving checkpoint to {ckpt_dir}")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
            
            if global_step >= args.max_train_steps:
                training_is_done = True
                break

    print("Fine-tuning complete.")
    
    final_ckpt_dir = os.path.join(args.output_dir, "final_checkpoint")
    print(f"Saving final model to {final_ckpt_dir}")
    model.save_pretrained(final_ckpt_dir)
    tokenizer.save_pretrained(final_ckpt_dir)


if __name__ == "__main__":
    main()