import os
import torch
from tqdm import tqdm
from openrlhf.utils.distributed_sampler import DistributedSampler
import torch.nn.functional as F
# --- Core Imports ---
from rlhf.grpo_loss import GRPOLoss
from fixed_generation_dual_cache import generate_with_dual_cache

SAVE_EVERY = 2

class GRPOTrainer:

    def __init__(
        self,
        model,
        ref_model,
        reward_fn,
        strategy,
        tokenizer,
        optim: torch.optim.Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        beta: float = 0.1,
        epsilon: float = 0.2,
        max_epochs: int = 1,
        # Generation parameters
        num_samples_per_prompt: int = 4,
        gen_steps: int = 256,
        gen_length: int = 256,
        block_length: int = 16,
        temperature: float = 0.0,
        remasking_threshold: float = 0.9,
        repetition_penalty: float = 1.2,
        # LLADA-specific ELBO estimation parameters
        n_t: int = 4,
        eps: float = 0.1,
    ) -> None:
        self.strategy = strategy
        self.epochs = max_epochs
        self.model = model
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        
        self.num_samples_per_prompt = num_samples_per_prompt
        self.gen_steps = gen_steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.temperature = temperature
        self.remasking_threshold = remasking_threshold
        self.repetition_penalty = repetition_penalty
        self.n_t = n_t
        self.eps = eps
        
        self.mask_id = self.tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
        self.pad_id = self.tokenizer.pad_token_id
        
        self.loss_fn = GRPOLoss(beta, epsilon)
        
        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb
            self._wandb = wandb
            wandb.init(
                entity=strategy.args.wandb_org, project=strategy.args.wandb_project,
                group=strategy.args.wandb_group, name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__, reinit=True
            )
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def forward_process(self, input_ids, mask_ratio):
        b, l = input_ids.shape
        p_mask = torch.full((b, l), mask_ratio, device=input_ids.device)
        masked_indices = torch.rand_like(p_mask, dtype=torch.float) < p_mask
        noisy_input = torch.where(masked_indices, self.mask_id, input_ids)
        return noisy_input, masked_indices, p_mask

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        if args.eval_steps <= 0: args.eval_steps = float("inf")
        if args.save_steps <= 0: args.save_steps = float("inf")

        grad_accum_steps = self.strategy.accumulated_gradient
        ppo_epochs = 4        
        start_epoch = consumed_samples // (num_update_steps_per_epoch * args.train_batch_size)
        global_step = consumed_samples // args.train_batch_size
        
        experience_buffer = []

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())

        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples)

            step_bar = tqdm(range(len(self.train_dataloader)), desc=f"Epoch {epoch} steps", disable=not self.strategy.is_rank_0())

            for step_idx, batch in enumerate(self.train_dataloader):
                print(f"Processing batch {step_idx + 1}/{len(self.train_dataloader)}")
                with torch.no_grad():
                    generated_sequences, rewards, prompt_lens_tensor, _ = self._generate_and_score(batch)
                    experience_buffer.append({
                        'sequences': generated_sequences.cpu(),
                        'rewards': rewards.cpu(),
                        'prompt_lens': prompt_lens_tensor.repeat_interleave(self.num_samples_per_prompt).cpu()
                    })
                if len(experience_buffer) == grad_accum_steps:
                    global_step += 1
                    device = self.model.model.device

                    # Pad the generations
                    pad_id = self.tokenizer.pad_token_id
                    seqs = [exp["sequences"] for exp in experience_buffer]
                    L_max = max(s.size(1) for s in seqs)
                    padded_seqs = []
                    for s in seqs:
                        pad_len = L_max - s.size(1)
                        if pad_len > 0:
                            s = F.pad(s, (0, pad_len), value=pad_id)
                        padded_seqs.append(s)

                    full_sequences = torch.cat(padded_seqs, dim=0).to(device)
                    full_rewards = torch.cat([exp['rewards'] for exp in experience_buffer], dim=0).to(device)
                    full_prompt_lens = torch.cat([exp['prompt_lens'] for exp in experience_buffer], dim=0).to(device)
                    full_attention_mask = (full_sequences != self.pad_id).long()
                    
                    # --- Run Multi-Epoch Update on the Full Batch ---
                    for ppo_epoch in range(ppo_epochs):
                        self.model.train()
                        epoch_loss_sum   = 0.0
                        epoch_reward_sum = 0.0

                        # Pre-calculate the fixed 'old' logprobs for the entire accumulated batch
                        with torch.no_grad():
                            t_fixed = (1 - self.eps) * torch.rand(1).item() + self.eps
                            _, masked_sequences_fixed, _ = self.forward_process(full_sequences, t_fixed)
                            old_policy_logprobs_t_fixed = self.model(full_sequences, full_attention_mask, masked_sequences_fixed, t_fixed, full_prompt_lens)
                            ref_logprobs_t_fixed = self.ref_model(full_sequences, full_attention_mask, masked_sequences_fixed, t_fixed, full_prompt_lens)

                        # Accumulate gradients over the n_t Monte Carlo samples
                        self.strategy.zero_grad()
                        for _ in range(self.n_t):
                            policy_logprobs_t = self.model(full_sequences, full_attention_mask, masked_sequences_fixed, t_fixed, full_prompt_lens)
                            loss_t, mean_reward_t, _ = self.loss_fn(
                                policy_logprobs=policy_logprobs_t,
                                old_policy_logprobs=old_policy_logprobs_t_fixed,
                                ref_logprobs=ref_logprobs_t_fixed,
                                rewards=full_rewards,
                                num_samples_per_prompt=self.num_samples_per_prompt
                            )
                            epoch_loss_sum   += loss_t.item()
                            epoch_reward_sum += mean_reward_t.item()
                            scaled_loss = loss_t / self.n_t
                            print(f"  loss_t type={loss_t}, requires_grad={loss_t.requires_grad}")
                            self.strategy.backward(scaled_loss, self.model, self.optimizer)
                            total_norm = 0.0
                            for p in self.model.parameters():
                                if p.grad is not None:
                                    total_norm += p.grad.data.norm().item()**2
                            total_norm = total_norm**0.5
                            print(f"  [DEBUG] grad norm={total_norm:.6f}")
                            
                        # Update the model weights after each PPO epoch
                        self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                        avg_epoch_loss   = epoch_loss_sum   / self.n_t
                        avg_epoch_reward = epoch_reward_sum / self.n_t
                        # --- Clear Buffer, Log, and Save Checkpoint ---
                        experience_buffer.clear()
                        logs_dict = {"loss": avg_epoch_loss, "reward": avg_epoch_reward, "lr": self.scheduler.get_last_lr()[0]}
                        logs_dict = self.strategy.all_reduce(logs_dict)
                        step_bar.set_postfix(logs_dict)
                        
                        client_states = {"consumed_samples": global_step * args.train_batch_size}
                        self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)
                        # if global_step % SAVE_EVERY == 0:
                        #     ckpt_path = os.path.join(self.strategy.args.save_path, f"_GRPO_lora_ckpt_{global_step}")
                        #     os.makedirs(ckpt_path, exist_ok=True)
                            # self.model.model.save_pretrained(ckpt_path, only_save_adapter=True)
                            # print(f"Saved LoRA checkpoint at step {global_step} to {ckpt_path}", flush=True)
                            # tag = "ds_checkpoint"
                            # if self.strategy.is_rank_0():
                            #     print(f"Saving full DS checkpoint to {os.path.join(ckpt_path, tag)}")
                            # self.strategy.engine.save_checkpoint(
                            #     ckpt_path,   # base dir
                            #     client_state=client_states,  # your consumed_samples, etc.
                            #     tag=tag
                            # )
                
                step_bar.update(1)
            epoch_bar.update(1)

    def _generate_and_score(self, batch):
        prompt = batch["prompt_texts"]
        prompt_ids = batch["prompt_ids"].to(torch.cuda.current_device())
        prompt_lens = batch["prompt_lens"].to(torch.cuda.current_device())
        prompt_tensor_len = prompt_ids.shape[1]
        
        expanded_prompt_ids = prompt_ids.repeat_interleave(self.num_samples_per_prompt, dim=0)
        # print("Generating responses for prompt: " , prompt)
        with torch.no_grad():
            generated_sequences, _ = generate_with_dual_cache(
                model=self.model.model, prompt=expanded_prompt_ids, steps=self.gen_steps, 
                gen_length=self.gen_length, block_length=self.block_length,
                temperature=self.temperature, remasking='low_confidence', 
                threshold=self.remasking_threshold, repetition_penalty=self.repetition_penalty,
                mask_id=self.mask_id
            )
            response_tokens = generated_sequences[:, prompt_tensor_len:]
            decoded_responses = self.tokenizer.batch_decode(response_tokens, skip_special_tokens=True)
            # print("Fetching rewards for generated responses...")
            rewards = torch.tensor(self.reward_fn(decoded_responses, prompt), dtype=torch.float32, device=self.model.model.device)
            # assert len(rewards) == generated_sequences.size(0), \
            #     f"Expected {generated_sequences.size(0)} scores, got {len(rewards)}"
            # print("\n\n\n  === DEBUG rewards ===")
            # for i, (resp, score) in enumerate(zip(decoded_responses[:4], rewards[:4])):
            #     print(f"\n\n[{i}] score={score:.1f}  text={resp!r}")
            # print(f"  rewards tensor: shape={rewards.shape},  min={rewards.min():.1f}, max={rewards.max():.1f}")
        
        return generated_sequences, rewards, prompt_lens, prompt_tensor_len
        
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            if self._wandb is not None and self.strategy.is_rank_0():
                self._wandb.log({"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()})

        if self.eval_dataloader is not None and global_step % args.eval_steps == 0 and len(self.eval_dataloader) > 0:
            self.evaluate(global_step, args)

        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(self.model.model, args.ckpt_path, tag, client_state=client_states)
    
    def evaluate(self, global_step, args):
        self.model.eval()
        self.ref_model.eval()
        
        eval_bar = tqdm(range(len(self.eval_dataloader)), desc=f"Eval stage of global_step {global_step}", disable=not self.strategy.is_rank_0())
        total_loss, total_reward = 0, 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                generated_sequences, rewards, prompt_lens_tensor, _ = self._generate_and_score(batch)
                attention_mask = (generated_sequences != self.pad_id).long()
                expanded_prompt_lens = prompt_lens_tensor.repeat_interleave(self.num_samples_per_prompt)

                t = (1 - self.eps) * torch.rand(1).item() + self.eps
                _, masked_sequences, _ = self.forward_process(generated_sequences, t)

                old_policy_logprobs_t = self.model(generated_sequences, attention_mask, masked_sequences, t, expanded_prompt_lens)
                ref_logprobs_t = self.ref_model(generated_sequences, attention_mask, masked_sequences, t, expanded_prompt_lens)
                policy_logprobs_t = old_policy_logprobs_t
                
                loss, mean_reward, _ = self.loss_fn(
                    policy_logprobs_t, old_policy_logprobs_t, ref_logprobs_t, rewards, self.num_samples_per_prompt
                )
                total_loss += loss.item()
                total_reward += mean_reward.item()
                eval_bar.update()

        logs = {"eval_loss": total_loss / len(self.eval_dataloader), "eval_reward": total_reward / len(self.eval_dataloader)}
        logs = self.strategy.all_reduce(logs)
        eval_bar.set_postfix(logs)

        if self.strategy.is_rank_0() and self._wandb is not None:
            self._wandb.log({"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()})
        
        self.model.train()