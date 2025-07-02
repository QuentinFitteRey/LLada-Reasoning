#!/usr/bin/env python3
import os
import argparse
import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk, DatasetDict, concatenate_datasets
from peft import LoraConfig, get_peft_model, PeftModel
import torch.nn.functional as F
import wandb
from llada_local.configuration_llada import ActivationCheckpointingStrategy

# -------- DDP Setup --------
def setup_ddp():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', rank)))

# -------- Data Collator --------
class SFTDataCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        input_ids_list, labels_list = [], []
        for ex in features:
            user_ids = self.tokenizer.encode(ex['user'] + self.tokenizer.eos_token, add_special_tokens=False)
            asm_ids  = self.tokenizer.encode(ex['assistant'] + self.tokenizer.eos_token, add_special_tokens=False)
            seq = (user_ids + asm_ids)[-self.max_length:]
            labels = [-100] * len(user_ids) + asm_ids
            labels = labels[-self.max_length:]
            input_ids_list.append(torch.tensor(seq, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_id)
        labels    = pad_sequence(labels_list,    batch_first=True, padding_value=-100)
        return {'input_ids': input_ids, 'labels': labels}

# -------- Validation --------
@torch.no_grad()
def run_validation(model, loader, device, limit_batches=None, mc_runs=1):
    model.eval()
    total_loss = 0.0
    count = 0
    for idx, batch in enumerate(loader):
        if limit_batches and idx >= limit_batches:
            break
        batch = {k:v.to(device) for k,v in batch.items()}
        losses = []
        for _ in range(mc_runs):
            outputs = model(**batch, use_cache=False)
            logits = outputs.logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch['labels'].view(-1), ignore_index=-100
            )
            losses.append(loss.item())
        total_loss += sum(losses) / len(losses)
        count += 1
    model.train()
    return total_loss / count if count else 0.0

# -------- Main --------
def main():
    parser = argparse.ArgumentParser(description='Supervised Fine-Tuning (SFT) with LoRA/DDP')
    # paths
    parser.add_argument('--sft-data',        type=str,  required=True,
                        help='Path to DatasetDict saved via save_to_disk')
    parser.add_argument('--pretrained-model',type=str,  required=True,
                        help='Path to pretrained extended-context model')
    parser.add_argument('--output-dir',      type=str,  default='./checkpoints/sft_loRA')
    # LoRA
    parser.add_argument('--use-lora',        action='store_true',
                        help='Enable LoRA for SFT')
    parser.add_argument('--lora-r',          type=int, default=16)
    parser.add_argument('--lora-alpha',      type=int, default=32)
    parser.add_argument('--lora-dropout',    type=float, default=0.05)
    parser.add_argument('--lora-target-modules', nargs='+', default=['q_proj','v_proj','k_proj','o_proj'])
    parser.add_argument('--lora-resume',     type=str, default=None,
                        help='Path to existing LoRA adapter')
    # precision & checkpointing
    parser.add_argument('--torch-dtype',     type=str, default='bfloat16',
                        choices=['float32','bfloat16','float16'],
                        help='torch.dtype for model weights')
    parser.add_argument('--activation-checkpointing', action='store_true', default=True,
                        help='Enable activation checkpointing')
    # training
    parser.add_argument('--batch-size',      type=int, default=64,
                        help='Global batch size')
    parser.add_argument('--local-batch',     type=int, default=1,
                        help='Per-device batch size')
    parser.add_argument('--seq-len',         type=int, default=8192)
    parser.add_argument('--max-steps',       type=int, default=1000)
    parser.add_argument('--epochs',          type=int, default=3)
    parser.add_argument('--lr',              type=float, default=5e-5)
    parser.add_argument('--weight-decay',    type=float, default=0.0)
    parser.add_argument('--warmup-steps',     type=int, default=100)
    parser.add_argument('--min-lr-ratio',    type=float, default=0.1)
    parser.add_argument('--grad-clip-norm',  type=float, default=1.0)
    # validation & checkpointing
    parser.add_argument('--validation-interval', type=int, default=100)
    parser.add_argument('--checkpoint-interval', type=int, default=100)
    parser.add_argument('--val-limit-batches',  type=int, default=50)
    parser.add_argument('--val-mc-runs',        type=int, default=1)
    # logging
    parser.add_argument('--use-wandb',       action='store_true')
    parser.add_argument('--wandb-project',   type=str, default='llada-sft')
    parser.add_argument('--wandb-entity',    type=str, default=None)
    parser.add_argument('--wandb-run-name',  type=str, default=None)
    args = parser.parse_args()

    setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{os.environ.get("LOCAL_RANK", rank)}')
    is_main = (rank == 0)

    if is_main and args.use_wandb:
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity,
                   name=args.wandb_run_name,
                   config=vars(args))

    # load model & tokenizer
    dtype = getattr(torch, args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        use_fast=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))

    # activation checkpointing
    if args.activation_checkpointing:
        model.set_activation_checkpointing(strategy=ActivationCheckpointingStrategy.whole_layer)

    # LoRA
    if args.use_lora:
        if args.lora_resume:
            # 1) load adapter (this will by default pull in inference_mode=True)
            model = PeftModel.from_pretrained(
                model,
                args.lora_resume,
                is_trainable=True,
                local_files_only=True,
                torch_dtype=dtype
            )
        else:
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules,
                lora_dropout=args.lora_dropout,
                bias='none',
                task_type='CAUSAL_LM'
            )
            model = get_peft_model(model, lora_config)
        if is_main:
            print(f'LoRA enabled: r={args.lora_r}, alpha={args.lora_alpha}')

    model.to(device)
    model = DDP(model, device_ids=[int(os.environ.get('LOCAL_RANK', rank))])

    # load dataset
    ds = load_from_disk(args.sft_data)
    if isinstance(ds, DatasetDict):
        train_ds = ds['train'] if 'train' in ds else concatenate_datasets(list(ds.values()))
        val_ds   = ds.get('validation', None)
    else:
        train_ds = ds
        val_ds   = None

    collator = SFTDataCollator(tokenizer, max_length=args.seq_len)
    train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
    train_loader  = DataLoader(train_ds, batch_size=args.local_batch,
                               sampler=train_sampler, collate_fn=collator)
    if val_ds:
        val_sampler = DistributedSampler(val_ds, world_size, rank, shuffle=False)
        val_loader  = DataLoader(val_ds, batch_size=args.local-batch,
                                  sampler=val_sampler, collate_fn=collator)

    # optimizer & scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        prog = float(step - args.warmup_steps) / float(max(1, args.max_steps - args.warmup_steps))
        prog = min(1.0, prog)
        return max(args.min_lr_ratio, 1.0 - prog * (1.0 - args.min_lr_ratio))
    scheduler = LambdaLR(optimizer, lr_lambda)

    accum_steps = args.batch_size // (args.local_batch * world_size)
    global_step = 0

    dist.barrier()
    if is_main and val_ds:
        init_loss = run_validation(model, val_loader, device,
                                   limit_batches=args.val_limit_batches,
                                   mc_runs=args.val_mc_runs)
        print(f'Initial val loss: {init_loss:.4f}')
        if args.use_wandb: wandb.log({'validation/initial_loss': init_loss, 'step': 0})
    dist.barrier()
    model.train()

    # training loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, use_cache=False)
            logits  = outputs.logits
            loss    = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch['labels'].view(-1), ignore_index=-100
            )
            loss = loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                global_step += 1

                if is_main:
                    print(f'[Epoch {epoch+1}] Step {global_step}/{args.max_steps} - loss: {loss.item() * accum_steps:.4f}')
                    if args.use_wandb: wandb.log({'train/loss': loss.item() * accum_steps, 'step': global_step})

                if val_ds and global_step % args.validation_interval == 0:
                    val_loss = run_validation(model, val_loader, device,
                                              limit_batches=args.val_limit_batches,
                                              mc_runs=args.val_mc_runs)
                    if is_main:
                        print(f'Val loss @ step {global_step}: {val_loss:.4f}')
                        if args.use_wandb: wandb.log({'validation/loss': val_loss, 'step': global_step})
                    dist.barrier()

                if global_step % args.checkpoint_interval == 0:
                    if is_main:
                        ckpt_dir = os.path.join(args.output_dir, f'step-{global_step}')
                        os.makedirs(ckpt_dir, exist_ok=True)
                        model.module.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        print(f'Checkpoint saved: {ckpt_dir}')
                    dist.barrier()

                if global_step >= args.max_steps:
                    break
        if global_step >= args.max_steps:
            break

    if is_main:
        final_dir = os.path.join(args.output_dir, 'final')
        os.makedirs(final_dir, exist_ok=True)
        model.module.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f'SFT complete, saved to {final_dir}')
        if args.use_wandb: wandb.finish()

    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()