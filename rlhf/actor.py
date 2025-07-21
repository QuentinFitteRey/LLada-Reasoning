from typing import Optional, Literal
from configuration_llada import ActivationCheckpointingStrategy
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

class Actor(nn.Module):
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
        tokenizer=None,
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
        if tokenizer is not None:
            self.tokenizer = tokenizer
        self.model = pretrain_or_model
        self.mask_id = self.tokenizer.convert_tokens_to_ids("<|mdm_mask|>")

    def forward(
        self,
        sequences: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        masked: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] | float = None,
        mode: Literal["loss", "fall_through"] = "loss",
    ) -> torch.Tensor:
        """Returns action log probs"""
        if mode == "loss":
            loss, _ =  self.calc_loss(sequences, attention_mask, masked, t)
            return loss
        elif mode == "fall_through":
            return self.model(sequences, attention_mask)
    
    def l_pi_from_logits(self, logits, labels, mask, t: float = 1.0) -> torch.Tensor:
        """
        logits -> l_pi by averaging the log probs of predicting ground truth on masked tokens
        """
        log_probs = F.log_softmax(logits/self.temperature, dim=-1)  # log p_theta [B, T, V]
        log_p_theta_at_targets = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
        masked_log_probs = log_p_theta_at_targets * mask.float()  # [B, T]
        l_pi = masked_log_probs.sum(dim=1) / (t + 1e-8)  # [B]
        return l_pi
    
    def calc_loss(self, input_ids, attention_mask, masked, t):
        """
        compute l_pi(y_t, t, y|x), the token-level per-timestep ELBO loss, from a input_ids = x||y
        """
        b, l = input_ids.shape
        if l == 0: raise Exception("length is zero.")
        noisy_input = torch.where(masked, self.mask_id, input_ids)
        logits = self.model(noisy_input, attention_mask=attention_mask).logits
        l_pi = self.l_pi_from_logits(logits, input_ids, masked, t)
        return l_pi, logits

    # def monte_carlo_forward(self, input_ids, input_mask, shared_mask=None, n_t=8, n_yt=1, eps=0.1):
    #     """
    #     Compute the ELBO estimator with double monte carlo
    #     """
    #     # print("This is one run")
    #     b = input_ids.size(0)
    #     total_loss = torch.zeros(b, device=input_ids.device)
    #     if shared_mask is None:
    #         count = 0
    #         shared_mask = []
    #         for _ in range(n_t):
    #             t = (1 - eps) * torch.rand(1).item() + eps 
    #             for _ in range(n_yt):
    #                 noisy_input, masked, p_mask = self.forward_process(input_ids, t)
    #                 shared_mask.append((t, masked))
    #                 l_pi, _ = self.calc_loss(input_ids,input_mask, noisy_input, masked, t)
    #                 if l_pi is not None:
    #                     total_loss  = total_loss + l_pi  # loss is shape (B,)
    #                     count += 1
    #                 # print(f"Random sampling: t:{t}; mask:{masked.sum()}; l_pi:{l_pi}")
    #         if count == 0:
    #             return torch.zeros(b, device=input_ids.device)
    #         return total_loss / count, shared_mask  # shape: (B,)
    #     else:
    #         for t, masked in shared_mask:
    #             noisy_input = torch.where(masked, self.mask_id, input_ids)
    #             l_pi, _ = self.calc_loss(input_ids, input_mask, noisy_input, masked, t)
    #             if l_pi is not None:
    #                 total_loss = total_loss + l_pi
    #             # print(f"shared mask: t:{t}; mask:{masked.sum()}; l_pi:{l_pi}")
    #         return total_loss / len(shared_mask), shared_mask  # shape: (B,)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.set_activation_checkpointing(ActivationCheckpointingStrategy.fine_grained)
        self.model.config.use_cache = False
        self.model.enable_input_require_grads()

    def gradient_checkpointing_disable(self):
        self.model.set_activation_checkpointing(None)

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
