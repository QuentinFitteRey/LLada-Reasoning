import os
from abc import ABC

from click import prompt
import torch
from torch.optim import Optimizer
from tqdm import tqdm
import torch.nn.functional as F
from openrlhf.models.loss import DPOLoss
from openrlhf.utils.distributed_sampler import DistributedSampler


class DPOTrainer(ABC):
    """
    Trainer for Direct Preference Optimization (DPO) training.

    Args:
        model (torch.nn.Module): The primary model to be trained.
        ref_model (torch.nn.Module): The reference model for comparing and guiding preference.
        strategy (Strategy): The strategy to use for training.
        tokenizer (Tokenizer): The tokenizer for processing input data.
        optim (Optimizer): The optimizer for training the model.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler to control learning rate during training.
        max_norm (float, defaults to 0.5): Maximum gradient norm for gradient clipping.
        beta (float, defaults to 0.01): Coefficient for regularizing the preference loss.
        max_epochs (int, defaults to 2): Maximum number of training epochs.
        save_hf_ckpt (bool): Whether to save huggingface-format model weight.
        disable_ds_ckpt (bool): Whether not to save deepspeed-format model weight. (Deepspeed model weight is used for training recovery)
    """

    def __init__(
        self,
        model,
        ref_model,
        strategy,
        tokenizer,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm=0.5,
        beta=0.01,
        max_epochs: int = 2,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.ref_model = ref_model
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt

        self.beta = beta
        self.loss_fn = DPOLoss(self.beta, self.args.label_smoothing, self.args.ipo)

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # NLL loss
        self.nll_loss = self.args.nll_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        special_tokens_dict = {}
        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = "<|pad|>"
        if self.tokenizer.convert_tokens_to_ids("<|mdm_mask|>") == self.tokenizer.unk_token_id:
            special_tokens_dict["additional_special_tokens"] = ["<|mdm_mask|>"]
        
        if special_tokens_dict:
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.mask_id = self.tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
        self.pad_id = self.tokenizer.pad_token_id
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def forward_process(self, input_ids, mask_ratio):
        b, l = input_ids.shape
        if isinstance(mask_ratio, torch.Tensor) and mask_ratio.ndim == 1: 
            p_mask = mask_ratio.view(b, 1).expand(b, l)
        else: 
            p_mask = torch.full((b, l), mask_ratio, device=input_ids.device)
        masked_indices = torch.rand_like(p_mask, dtype=torch.float) < p_mask
        noisy_input = torch.where(masked_indices, self.mask_id, input_ids)
        return noisy_input, masked_indices, p_mask

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Define constants for the new loop
        n_t = 8  # Your number of Monte Carlo samples
        eps = 0.1 # Your epsilon for mask ratio sampling

        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )

        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            self.ref_model.eval()

            for step_idx, data in enumerate(self.train_dataloader):
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                prompt_id_lens = torch.tensor(prompt_id_lens, device=chosen_ids.device)
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                print(f"chose answer{self.tokenizer.batch_decode(chosen_ids, skip_special_tokens=True)}")
                print(f"reject answer{self.tokenizer.batch_decode(reject_ids, skip_special_tokens=True)}")
                #print prompt only based on length
                print(f"prompt length: {prompt_id_lens}")
                print(f"prompt: {self.tokenizer.batch_decode(chosen_ids[:, :prompt_id_lens[0]], skip_special_tokens=True)}")   

                # Accumulators for logging average values
                batch_loss_sum = 0
                batch_chosen_reward_sum = 0
                batch_reject_reward_sum = 0

                # --- START: New Monte Carlo & Gradient Accumulation Loop ---
                for _ in range(n_t):
                    # 1. Generate ONE random mask ratio `t` for a fair comparison
                    t = (1 - eps) * torch.rand(1).item() + eps
                    
                    # 2. Process CHOSEN sequence with this `t`
                    _, masked_chosen, _ = self.forward_process(chosen_ids, t)
                    _, masked_rejected, _ = self.forward_process(reject_ids, t)
                    # 3. Process REJECTED sequence with the SAME `t`
                    policy_chosen_logps = self.model(chosen_ids, c_mask, masked_chosen, prompt_id_lens, t)
                    with torch.no_grad():
                        ref_chosen_logps = self.ref_model(chosen_ids, c_mask, masked_chosen, prompt_id_lens, t)
                    
                    policy_rejected_logps = self.model(reject_ids, r_mask, masked_rejected, prompt_id_lens, t)
                    with torch.no_grad():
                        ref_rejected_logps = self.ref_model(reject_ids, r_mask, masked_rejected, prompt_id_lens, t)
 
                    
                    # 4. Calculate DPO loss for this SINGLE, FAIR comparison
                    loss, chosen_reward, reject_reward = self.loss_fn(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        ref_chosen_logps,
                        ref_rejected_logps
                    )
                    
                    # 5. Scale loss for averaging and backpropagate.
                    # This accumulates gradients correctly without memory spikes.
                    scaled_loss = loss / n_t
                    self.strategy.backward(scaled_loss, self.model, self.optimizer)

                    # Accumulate stats for logging
                    batch_loss_sum += loss.item()
                    batch_chosen_reward_sum += chosen_reward.mean().item()
                    batch_reject_reward_sum += reject_reward.mean().item()
                # --- END: New Monte Carlo & Gradient Accumulation Loop ---

                # 6. Step the optimizer AFTER all gradients are accumulated
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                # Update logs with the averaged values from the MC samples
                avg_loss = batch_loss_sum / n_t
                avg_chosen_reward = batch_chosen_reward_sum / n_t
                avg_reject_reward = batch_reject_reward_sum / n_t
                acc = (avg_chosen_reward > avg_reject_reward)

                logs_dict = {
                    "loss": avg_loss,
                    "acc": acc,
                    "chosen_reward": avg_chosen_reward,
                    "reject_reward": avg_reject_reward,
                    "reward_diff": avg_chosen_reward - avg_reject_reward,
                    "lr": self.scheduler.get_last_lr()[0],
                }

                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # Global step for logging and checkpointing
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    torch.cuda.empty_cache()
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1
            
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0 and self.eval_dataloader is not None:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(
                    self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
                )
            if self.save_hf_ckpt:
                save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)

    def evaluate(self, eval_dataloader, steps=0):
        self.model.eval()
        with torch.no_grad():
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of global_step %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
            acc_sum = 0
            loss_sum = 0
            times = 0
            for data in eval_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                prompt_id_lens = torch.tensor(prompt_id_lens, device=chosen_ids.device)
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                _, masked_chosen, _ = self.forward_process(chosen_ids, t)
                _, masked_rejected, _ = self.forward_process(reject_ids, t)
                # 3. Process REJECTED sequence with the SAME `t`
                policy_chosen_logps = self.model(chosen_ids, c_mask, masked_chosen, prompt_id_lens, t)
                ref_chosen_logps = self.ref_model(chosen_ids, c_mask, masked_chosen, prompt_id_lens, t)
                
                policy_rejected_logps = self.model(reject_ids, r_mask, masked_rejected, prompt_id_lens, t)
                ref_rejected_logps = self.ref_model(reject_ids, r_mask, masked_rejected, prompt_id_lens, t)
                loss, chosen_reward, reject_reward = self.loss_fn(
                    policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
                )
                acc_sum += (chosen_reward > reject_reward).float().mean().item()
                loss_sum += loss.item()
                times += 1
                step_bar.update()

            logs = {
                "eval_loss": loss_sum / times,
                "acc_mean": acc_sum / times,
            }
            logs = self.strategy.all_reduce(logs)
            step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state

    def _get_batch_logps(
        self,
        per_token_logps: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            per_token_logps: Per token log probabilities. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert average_log_prob == False

        loss_masks = attention_mask.clone().bool()
        # mask prompts
        for mask, source_len in zip(loss_masks, prompt_id_lens):
            mask[:source_len] = False
        loss_masks = loss_masks[:, 1:]

        logprobs_sums = (per_token_logps * loss_masks).sum(-1)
        logprobs_means = (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)
        return logprobs_sums, logprobs_means
