# Place this in rlhf/grpo_loss.py
import torch
import torch.nn.functional as F

class GRPOLoss:
    def __init__(self, beta: float, epsilon: float = 0.2):
        """
        Initializes the GRPOLoss function based on the PPO-clip objective.
        
        Args:
            beta (float): The KL divergence regularization coefficient.
            epsilon (float): The clipping parameter for the PPO objective.
        """
        self.beta = beta
        self.epsilon = epsilon

    def __call__(self,
                 policy_logprobs: torch.Tensor,
                 old_policy_logprobs: torch.Tensor,
                 ref_logprobs: torch.Tensor,
                 rewards: torch.Tensor,
                 num_samples_per_prompt: int):
        """Calculates the GRPO loss using PPO-style clipping."""
        batch_size = rewards.shape[0] // num_samples_per_prompt
        k = num_samples_per_prompt

        rewards = rewards.view(batch_size, k)
        policy_logprobs = policy_logprobs.view(batch_size, k)
        old_policy_logprobs = old_policy_logprobs.view(batch_size, k)
        ref_logprobs = ref_logprobs.view(batch_size, k)

        advantages = rewards.detach() - rewards.mean(dim=1, keepdim=True).detach()
        
        # --- Probability Ratio Calculation ---
        log_ratio = policy_logprobs - old_policy_logprobs
        ratio = torch.exp(log_ratio)
        # ---
        
        unclipped_loss = ratio * advantages
        clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        clipped_loss = clipped_ratio * advantages
        
        policy_loss = -torch.min(unclipped_loss, clipped_loss)
        kl_div = policy_logprobs - ref_logprobs
        
        loss_per_sample = policy_loss + self.beta * kl_div
        loss = loss_per_sample.mean()

        return loss, rewards.mean(), kl_div.mean()