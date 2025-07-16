from typing import Optional, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig
import torch.nn.functional as F

from openrlhf.models.ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor
from openrlhf.models.utils import compute_entropy


class LLadaActor(nn.Module):
    """
    Base class for Actor models in reinforcement learning.

    This class serves as a foundation for implementing various actor models, which are responsible for selecting actions based on the policy learned from the environment.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        use_flash_attention_2 (bool, optional): Whether to utilize Flash Attention 2.0 for improved performance. Defaults to False.
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        target_modules (list, optional): List of target modules for applying LoRA. Defaults to None.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.
        temperature (float, optional): Temperature for action selection. Defaults to 1.0.
        use_liger_kernel (bool, optional): Whether to use Liger Kernel for the model. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        temperature=1.0,
        use_liger_kernel=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature

        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            if use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM

                model_class = AutoLigerKernelForCausalLM
            else:
                model_class = AutoModelForCausalLM

            self.model = model_class.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False
        else:
            self.model = pretrain_or_model

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        return_logprobs=False,
        mode: Literal["monte_carlo", "loss", "fall_through"] = "loss",
        return_entropy=False,
    ) -> torch.Tensor:
        """Returns action log probs"""
        batch, seqlen = sequences.size()
        if mode == "monte_carlo":
            output = self.monte_carlo_forward(sequences, attention_mask)
            return output if not return_output else (output, {"logits": None})  # logits not used in MC
        elif mode == "fall_through":
            return self.model(sequences, attention_mask)
        else:
            """
            Really not sure here what correspond to the original autoregressive terms. Just returning some analogous
            to try to not break the interace.
            """
            l_pi, logits = self.calc_loss(sequences, attention_mask, t=torch.rand(1).item())
            if return_entropy:
                assert return_output
                entropy = compute_entropy(l_pi)
                return (entropy[:, :-1], {"logits": l_pi})  # optional
            
            return_action_log_probs = action_mask is not None 
            if not return_action_log_probs and not return_logprobs:
                assert return_output
                return {"logits": l_pi}

            log_probs = F.log_softmax(logits / self.temperature, dim=-1)

            if return_logprobs and not return_action_log_probs:
                return (log_probs, {"logits": logits}) if return_output else log_probs

            # You are using a masked token loss format, so use mask * log p_theta
            log_p_at_labels = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
            action_log_probs = log_p_at_labels * action_mask.float()  # [B, T]

            return (action_log_probs, {"logits": logits}) if return_output else action_log_probs
    
    def l_pi_from_logits(self, logits, labels, mask, t: float = 1.0) -> torch.Tensor:
        """
        logits -> l_pi by averaging the log probs of predicting ground truth on masked tokens
        """
        log_probs = F.log_softmax(logits/self.temperature, dim=-1)  # log p_theta [B, T, V]
        log_p_theta_at_targets = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
        masked_log_probs = log_p_theta_at_targets * mask.float()  # [B, T]
        l_pi = masked_log_probs.sum(dim=1) / (t + 1e-8)  # [B]
        return l_pi
    
    def monte_carlo_forward(self, input_ids, input_mask, n_t=8, n_yt=1, eps=1e-6):
        """
        Compute the ELBO estimator with double monte carlo
        """
        b = input_ids.size(0)
        total_loss = torch.zeros(b, device=input_ids.device)
        count = 0

        for _ in range(n_t):
            t = (1 - eps) * torch.rand(1).item() + eps
            for _ in range(n_yt):
                l_pi, _ = self.calc_loss(input_ids,input_mask, t)
                if l_pi is not None:
                    total_loss += l_pi  # loss is shape (B,)
                    count += 1

        if count == 0:
            return torch.zeros(b, device=input_ids.device)
        return total_loss / count  # shape: (B,)

    def calc_loss(self, input_ids, attention_mask, t):
        """
        compute l_pi(y_t, t, y|x), the token-level per-timestep ELBO loss, from a input_ids = x||y
        """
        b, l = input_ids.shape
        if l == 0: raise Exception("length is zero.")
        noisy_input, masked, p_mask = self.forward_process(input_ids, t)
        logits = self.model(noisy_input, attention_mask=attention_mask).logits
        l_pi = self.l_pi_from_logits(logits, input_ids, masked, t)
        return l_pi, logits

    def forward_process(self, input_ids, mask_ratio):
        b, l = input_ids.shape
        if isinstance(mask_ratio, torch.Tensor) and mask_ratio.ndim == 1: 
            p_mask = mask_ratio.view(b, 1).expand(b, l)
        else: 
            p_mask = torch.full((b, l), mask_ratio, device=input_ids.device)
        masked_indices = torch.rand_like(p_mask, dtype=torch.float) < p_mask
        noisy_input = torch.where(masked_indices, self.mask_id, input_ids)
        return noisy_input, masked_indices, p_mask

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
