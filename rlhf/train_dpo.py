import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
from datetime import datetime

from transformers.trainer import get_scheduler
from rlhf.actor import Actor

from openrlhf.datasets import RewardDataset
from openrlhf.datasets.utils import blending_datasets
from rlhf.dpo_trainer import DPOTrainer
from rlhf.deepspeedStrategy import DeepspeedStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict, get_peft_model
import os
os.environ["MASTER_PORT"] = "42000"

def init_model(args):
    """
    Initializes and returns the base model and tokenizer.
    """
    local_model_path = args.pretrain

    print(f"Loading tokenizer from: {local_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    print(f"Loading base model from: {local_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    print("Base model loaded successfully.")
    return model, tokenizer

def train(args):
    # configure strategy
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

    # configure model
    # load huggingface model
    base_model, tokenizer = init_model(args)

    # 2. Create the LoRA config from args
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    # 3. Apply LoRA adapters to create the PEFT model
    peft_model = get_peft_model(base_model, lora_config)
    
    # 4. Wrap the PEFT model in your Actor class
    model = Actor(
        peft_model,
        tokenizer=tokenizer,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )

    # strategy.print("Trainable parameters:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         strategy.print(name)
    # --- REFERENCE MODEL CREATION ---
    # 1. Load a FRESH, clean base model for the reference model
    ref_base_model, _ = init_model(args)
    
    # 2. Wrap the BASE model in the Actor class. NO LoRA.
    ref_model = Actor(
        ref_base_model,
        tokenizer=tokenizer,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        ds_config=strategy.get_ds_eval_config(offload=args.ref_offload),
    )
    if args.ref_offload:
        ref_model._offload = True

    # gradient_checkpointing
    if args.gradient_checkpointing:
        strategy.print("\n\n\n\nEnable gradient checkpointing\n\n\n\n")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    # prepare for data and dataset
    train_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        dataset_split=args.dataset_split,
    )
    def map_shp_preformatted_prompt(example):
        """
        Pre-formats the prompt according to the chat template, leaving the
        responses ready for simple concatenation.
        """
        user_prompt = example["history"]

        # The prompt is formatted with all tokens leading up to the assistant's turn
        formatted_prompt = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt.strip()}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        # Determine which response is chosen and which is rejected
        if example["labels"] == 1:
            chosen_response = example["human_ref_B"]
            rejected_response = example["human_ref_A"]
        else:
            chosen_response = example["human_ref_A"]
            rejected_response = example["human_ref_B"]

        # The response strings just need the end-of-turn token appended
        return {
            "prompt": formatted_prompt,
            "chosen": f"{chosen_response.strip()}<|eot_id|>",
            "rejected": f"{rejected_response.strip()}<|eot_id|>"
        }

    train_data = train_data.map(map_shp_preformatted_prompt)

    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    train_dataset = RewardDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        input_template=args.input_template,
        is_dpo=True,
    )

    # prepare dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.collate_fn,
    )

    eval_dataset = None
    eval_dataloader = None
    if getattr(args, "eval_dataset", None):
        eval_data = blending_datasets(
            args.eval_dataset,
            None,  # No probability sampling for eval datasets
            strategy,
            dataset_split=args.eval_split,
        )
        eval_dataset = RewardDataset(
            eval_data,
            tokenizer,
            args.max_len,
            strategy,
            input_template=args.input_template,
            is_dpo=True,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            args.micro_train_batch_size,
            True,
            False,
            eval_dataset.collate_fn,
        )

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )
    # strategy prepare
    ((model, optim, scheduler), ref_model) = strategy.prepare((model, optim, scheduler), ref_model)

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model.model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is expected to be C(k,2), k means # response of each prompt
    # be limited with the format of dataset 'Dahoas/rm-static', we'd better use batch_size as 1
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        beta=args.beta,
        max_epochs=args.max_epochs,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoints
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_dpo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--use_ds_universal_ckpt", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepcompile", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--ref_offload", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--use_liger_kernel", action="store_true", default=False, help="Enable Liger Kernel")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--ds_tensor_parallel_size", type=int, default=1, help="DeepSpeed Tensor parallel size")

    # DPO
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--ipo", action="store_true", default=False)  # IPO https://arxiv.org/pdf/2310.12036v2.pdf
    parser.add_argument("--label_smoothing", type=float, default=0.0)  # cDPO https://arxiv.org/pdf/2305.18290.pdf
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument(
        "--nll_loss_coef", type=float, default=0, help="Regularization with NLL loss, see LLama 3.1 tech report."
    )
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # Context Parallel
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # Custom dataset
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--ref_pretrain", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, help="Path to the training dataset")
    parser.add_argument("--dataset_probs", type=str, default=None, help="Sampling probabilities for training datasets")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Path to the evaluation dataset")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=1000000, help="Maximum number of samples to use")

    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--max_len", type=int, default=512)

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_dpo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="exp_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    if args.ref_pretrain is None or args.ref_pretrain == "":
        args.ref_pretrain = args.pretrain

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    train(args)
