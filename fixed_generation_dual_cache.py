import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional


def add_gumbel_noise(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature == 0:
        return logits
    noise = torch.rand_like(logits)
    # classic Gumbel noise
    gumbel_noise = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
    # apply temperature
    return (logits / temperature) + gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    # distribute mask counts evenly over steps
    mask_count = mask_index.sum(dim=1, keepdim=True)
    base = mask_count // steps
    rem = mask_count % steps
    counts = base.repeat(1, steps)
    for i in range(counts.size(0)):
        counts[i, :rem[i]] += 1
    return counts


@torch.no_grad()
def generate_with_dual_cache(
    model,
    prompt: torch.LongTensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = 'low_confidence',
    mask_id: int = 126336,
    threshold: Optional[torch.Tensor] = None,
    repetition_penalty: float = 1.0,
    stop_token_id: Optional[int] = None,
):
    """
    Dual-cache generation: block-wise sampling with classifier-free guidance and repetition penalty.
    """
    prompt_len = prompt.shape[1]
    total_len = prompt_len + gen_length
    x = torch.full((1, total_len), mask_id, dtype=torch.long, device=prompt.device)
    x[:, :prompt_len] = prompt

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, "steps must be divisible by number of blocks"
    sub_steps = steps // num_blocks

    nfe = 0
    for block_idx in range(num_blocks):
        start = prompt_len + block_idx * block_length
        end = start + block_length

        # —— PRIME CACHE FOR THIS BLOCK —— #
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        logits = output.logits
        nfe += 1

        # compute how many tokens to transfer per sub-step
        mask_block = (x[:, start:end] == mask_id)
        num_trans = get_num_transfer_tokens(mask_block, sub_steps)

        # ---- INITIAL FILL FOR THE BLOCK ---- #
        full_mask = (x == mask_id)
        full_mask[:, end:] = False
        x0, transfer_idx = get_transfer_index(
            logits,
            temperature,
            remasking,
            full_mask,
            x,
            num_trans[:, 0] if threshold is None else None,
            threshold,
            repetition_penalty,
            penalty_context=x,
            prompt_len=prompt_len,
            mask_id=mask_id,
        )
        x[transfer_idx] = x0[transfer_idx]

        # ---- ITERATIVE REFINEMENT ---- #
        replace_mask = torch.zeros_like(x, dtype=torch.bool)
        replace_mask[:, start:end] = True
        for step in range(1, sub_steps):
            mask_slice = (x[:, start:end] == mask_id)
            if not mask_slice.any():
                break
            nfe += 1

            # forward only the block slice with cache replacement
            out = model(
                x[:, start:end],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_mask,
            )
            logits = out.logits
            past_key_values = out.past_key_values

            x0_blk, transfer_blk = get_transfer_index(
                logits,
                temperature,
                remasking,
                mask_slice,
                x[:, start:end],
                num_trans[:, step] if threshold is None else None,
                threshold,
                repetition_penalty,
                penalty_context=x,
                prompt_len=prompt_len,
                mask_id=mask_id,
            )
            x[:, start:end][transfer_blk] = x0_blk[transfer_blk]

        # early stop if stop_token encountered in this block
        if stop_token_id is not None and (x[:, start:end] == stop_token_id).any():
            break

    return x, nfe


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer: Optional[torch.Tensor],
    threshold: Optional[torch.Tensor] = None,
    repetition_penalty: float = 1.0,
    penalty_context: Optional[torch.Tensor] = None,
    prompt_len: int = 0,
    mask_id: Optional[int] = None,
):
    # apply repetition penalty
    if repetition_penalty > 1.0 and penalty_context is not None:
        # penalize logits of repeated tokens
        for i in range(logits.size(0)):
            gen_slice = penalty_context[i, prompt_len:]  # only generated part
            if mask_id is not None:
                gen_slice = gen_slice[gen_slice != mask_id]
            if gen_slice.numel() > 0:
                unique_toks = gen_slice.unique()
                logits[i, :, unique_toks] /= repetition_penalty

    # sample via Gumbel max
    logits_noise = add_gumbel_noise(logits, temperature)
    x0 = torch.argmax(logits_noise, dim=-1)

    # compute confidence for remasking
    if remasking == 'low_confidence':
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        conf = torch.squeeze(probs.gather(-1, x0.unsqueeze(-1)), -1)
    elif remasking == 'random':
        conf = torch.rand_like(x0, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    # mask out non-mask positions
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, conf, -np.inf)

    # select top-k positions to update
    transfer = torch.zeros_like(x0, dtype=torch.bool)
    if threshold is not None:
        num_transfer = mask_index.sum(dim=1, keepdim=True)
    for i in range(logits.size(0)):
        k = num_transfer[i] if num_transfer is not None else 0
        _, idxs = torch.topk(confidence[i], k=k)
        transfer[i, idxs] = True
        if threshold is not None:
            for j in range(1, k):
                if confidence[i, idxs[j]] < threshold:
                    transfer[i, idxs[j]] = False
    return x0, transfer
