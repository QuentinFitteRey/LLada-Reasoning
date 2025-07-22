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
        self.pad_id = self.tokenizer.pad_token_id

    def forward(
        self,
        sequences: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        masked: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] | float = None,
        prompt_id_lens: Optional[torch.Tensor] = None,
        mode: Literal["loss", "fall_through"] = "loss",
    ) -> torch.Tensor:
        """Returns action log probs"""
        if mode == "loss":
            loss, _ = self.calc_loss(sequences, attention_mask, masked, prompt_id_lens, t)
            return loss
        elif mode == "fall_through":
            return self.model(sequences, attention_mask)
    
    def l_pi_from_logits(self, logits, labels, mask, prompt_id_lens, t: float = 1.0) -> torch.Tensor:
        """
        Calculates log prob by averaging over masked tokens IN THE ANSWER ONLY.
        """
        b, l = labels.shape
        dev = labels.device

        # --- MODIFIED: Create a mask for the answer part only ---
        pos = torch.arange(l, device=dev).unsqueeze(0)
        ans_mask = pos >= prompt_id_lens.unsqueeze(1)
        
        # The final loss mask includes only tokens that are:
        # 1. In the answer.
        # 2. Were randomly selected to be masked.
        # 3. Are not padding tokens.
        final_loss_mask = ans_mask & mask & (labels != self.pad_id)
        # --- END MODIFICATION ---

        if not final_loss_mask.any():
            return (logits.sum() * 0.0).expand(b)

        log_probs = F.log_softmax(logits / self.temperature, dim=-1)
        log_p_theta_at_targets = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        # Apply the final, precise mask
        masked_log_probs = log_p_theta_at_targets * final_loss_mask.float()
        
        # Sum the log-probs for the sequence. Normalization by `t` is specific to the Llada objective.
        l_pi = masked_log_probs.sum(dim=1) / (t + 1e-8)
        return l_pi

    def calc_loss(self, input_ids, attention_mask, masked_indices, prompt_id_lens, t):
        """
        Computes the sequence log-probability, ensuring the prompt is not masked.
        """
        b, l = input_ids.shape
        dev = input_ids.device
        if l == 0:
            return torch.zeros(b, device=dev), None
        prompt_id_lens = prompt_id_lens.to(dev)
        # --- MODIFIED: Un-mask the prompt ---
        # 1. Create the noisy input from the boolean mask
        noisy_input = torch.where(masked_indices, self.mask_id, input_ids)
        
        # 2. Create a mask for the prompt section
        pos = torch.arange(l, device=dev).unsqueeze(0)
        prompt_mask_bool = pos < prompt_id_lens.unsqueeze(1)

        # 3. Restore the original prompt in the noisy input
        noisy_input[prompt_mask_bool] = input_ids[prompt_mask_bool]
        # --- END MODIFICATION ---

        logits = self.model(noisy_input, attention_mask=attention_mask).logits
        l_pi = self.l_pi_from_logits(logits, input_ids, masked_indices, prompt_id_lens, t)
        return l_pi, logits

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.set_activation_checkpointing(ActivationCheckpointingStrategy.fine_grained)
        self.model.config.use_cache = False
        self.model.enable_input_require_grads()

    def gradient_checkpointing_disable(self):
        self.model.set_activation_checkpointing(None)

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
