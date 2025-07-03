#!/usr/bin/env python3
import os
import argparse
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
            user_ids = self.tokenizer.encode(
                ex['user'] + self.tokenizer.eos_token,
                add_special_tokens=False
            )
            asm_ids = self.tokenizer.encode(
                ex['assistant'] + self.tokenizer.eos_token,
                add_special_tokens=False
            )
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
    total_loss, count = 0.0, 0
    for idx, batch in enumerate(loader):
        if limit_batches and idx >= limit_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        losses = []
        for _ in range(mc_runs):
            outputs = model(**batch, use_cache=False)
            logits  = outputs.logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch['labels'].view(-1),
                ignore_index=-100
            )
            losses.append(loss.item())
        total_loss += sum(losses) / len(losses)
        count += 1
    model.train()
    return total_loss / count if count else 0.0

# -------- Main --------
def main():
    parser = argparse.ArgumentParser(description='SFT with LoRA & DDP')
    parser.add_argument('--sft-data',         type=str, required=True,
                        help='Path to combined DatasetDict')
    parser.add_argument('--pretrained-model', type=str, required=True)
    parser.add_argument('--output-dir',       type=str, default='./checkpoints/sft_loRA')
    # LoRA options
    parser.add_argument('--use-lora',         action='store_true')
    parser.add_argument('--lora-r',           type=int, default=16)
    parser.add_argument('--lora-alpha',       type=int, default=32)
    parser.add_argument('--lora-dropout',     type=float, default=0.05)
    parser.add_argument('--lora-target-modules', nargs='+',
                        default=['q_proj','v_proj','k_proj','o_proj'])
    parser.add_argument('--lora-resume',      type=str, default=None)
    # precision & checkpointing
    parser.add_argument('--torch-dtype',      type=str, default='bfloat16',
                        choices=['float32','bfloat16','float16'])
    parser.add_argument('--activation-checkpointing',
                        action='store_true', default=True)
    # training hyperparams
    parser.add_argument('--batch-size',       type=int,   default=64,
                        help='Global batch size')
    parser.add_argument('--local-batch',      type=int,   default=1,
                        help='Per-GPU batch size')
    parser.add_argument('--seq-len',          type=int,   default=8192)
    parser.add_argument('--max-steps',        type=int,   default=1000)
    parser.add_argument('--epochs',           type=int,   default=3)
    parser.add_argument('--lr',               type=float, default=5e-5)
    parser.add_argument('--weight-decay',     type=float, default=0.0)
    parser.add_argument('--warmup-steps',     type=int,   default=100)
    parser.add_argument('--min-lr-ratio',     type=float, default=0.1)
    parser.add_argument('--grad-clip-norm',   type=float, default=1.0)
    # validation & checkpointing
    parser.add_argument('--validation-interval', type=int, default=100)
    parser.add_argument('--checkpoint-interval', type=int, default=100)
    parser.add_argument('--val-limit-batches',   type=int, default=50)
    parser.add_argument('--val-mc-runs',         type=int, default=1)
    # logging
    parser.add_argument('--use-wandb',        action='store_true')
    parser.add_argument('--wandb-project',    type=str, default='llada-sft')
    parser.add_argument('--wandb-entity',     type=str, default=None)
    parser.add_argument('--wandb-run-name',   type=str, default=None)
    args = parser.parse_args()

    setup_ddp()
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = torch.device(f'cuda:{os.environ.get("LOCAL_RANK", rank)}')
    is_main    = (rank == 0)

    # optional W&B
    if is_main and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args)
        )

    # load model & tokenizer
    dtype = getattr(torch, args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        use_fast=True,
        local_files_only=True
    )
    # ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))

    # activation checkpointing
    if args.activation_checkpointing:
        model.set_activation_checkpointing(
            strategy=ActivationCheckpointingStrategy.whole_layer
        )

    # LoRA
    if args.use_lora:
        if args.lora_resume:
            model = PeftModel.from_pretrained(
                model,
                args.lora_resume,
                is_trainable=True,
                torch_dtype=dtype,
                local_files_only=True
            )
        else:
            lora_cfg = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules,
                lora_dropout=args.lora_dropout,
                bias='none',
                task_type='CAUSAL_LM'
            )
            model = get_peft_model(model, lora_cfg)
        if is_main:
            print(f"LoRA enabled (r={args.lora_r}, α={args.lora_alpha})")

    model.to(device)
    model = DDP(model, device_ids=[int(os.environ.get("LOCAL_RANK", rank))])

    # load combined SFT data
    ds = load_from_disk(args.sft_data)
    if not isinstance(ds, DatasetDict):
        raise ValueError("`--sft-data` must point at a DatasetDict with *_train and *_validation splits")

    # collect all train / validation splits
    train_keys = [k for k in ds if k.endswith("_train")]
    val_keys   = [k for k in ds if k.endswith("_validation")]
    train_ds   = concatenate_datasets([ds[k] for k in train_keys])
    val_ds     = concatenate_datasets([ds[k] for k in val_keys]) if val_keys else None

    # data loaders
    collator       = SFTDataCollator(tokenizer, max_length=args.seq_len)
    train_sampler  = DistributedSampler(train_ds, world_size, rank, shuffle=True)
    train_loader   = DataLoader(
        train_ds,
        batch_size=args.local_batch,
        sampler=train_sampler,
        collate_fn=collator,
    )
    if val_ds is not None:
        val_sampler = DistributedSampler(val_ds, world_size, rank, shuffle=False)
        val_loader  = DataLoader(
            val_ds,
            batch_size=args.local_batch,
            sampler=val_sampler,
            collate_fn=collator,
        )

    # optimizer & LR scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    def lr_fn(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return max(args.min_lr_ratio, 1.0 - prog * (1.0 - args.min_lr_ratio))
    scheduler = LambdaLR(optimizer, lr_fn)

    accum_steps = args.batch_size // (args.local_batch * world_size)
    
    # how many optimizer steps per epoch?
    num_update_steps_per_epoch = len(train_loader) // accum_steps
    total_training_steps = num_update_steps_per_epoch * args.epochs

    # warmup_steps now from args (default=50)
    warmup_steps = args.warmup_steps

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # constant until last 10%
        if step < 0.9 * total_training_steps:
            return 1.0
        # linear decay to min_lr_ratio over final 10%
        prog = (step - 0.9 * total_training_steps) / (0.1 * total_training_steps)
        return max(args.min_lr_ratio, 1.0 - prog * (1.0 - args.min_lr_ratio))

    scheduler = LambdaLR(optimizer, lr_fn)

    global_step = 0

    # initial validation
    dist.barrier()
    if is_main and val_ds is not None:
        init_loss = run_validation(
            model, val_loader, device,
            limit_batches=args.val_limit_batches,
            mc_runs=args.val_mc_runs
        )
        print(f"Initial validation loss: {init_loss:.4f}")
        if args.use_wandb:
            wandb.log({"validation/initial_loss": init_loss, "step": 0})
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
                batch["labels"].view(-1),
                ignore_index=-100
            ) / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                global_step += 1

                if is_main:
                    train_loss = (loss.item() * accum_steps)
                    print(f"[Epoch {epoch+1}] Step {global_step}/{args.max_steps} — train_loss: {train_loss:.4f}")
                    if args.use_wandb:
                        wandb.log({"train/loss": train_loss, "step": global_step})

                # validation interval
                if val_ds is not None and (global_step % args.validation_interval == 0):
                    val_loss = run_validation(
                        model, val_loader, device,
                        limit_batches=args.val_limit_batches,
                        mc_runs=args.val_mc_runs
                    )
                    if is_main:
                        print(f"[Step {global_step}] validation_loss: {val_loss:.4f}")
                        if args.use_wandb:
                            wandb.log({"validation/loss": val_loss, "step": global_step})
                    dist.barrier()

                # checkpoint interval
                if global_step % args.checkpoint_interval == 0 and is_main:
                    ckpt_dir = os.path.join(args.output_dir, f"step-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model.module.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"Checkpoint saved to {ckpt_dir}")
                    dist.barrier()

        #         if global_step >= args.max_steps:
        #             break
        # if global_step >= args.max_steps:
        #     break

    # final save
    if is_main:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        model.module.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"SFT complete, final weights in {final_dir}")
        if args.use_wandb:
            wandb.finish()

    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()