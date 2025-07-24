import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional
from init_model import init_model

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
    batch_size, prompt_len = prompt.shape
    total_len = prompt_len + gen_length

    # <-- change here to use batch_size -->
    x = torch.full(
        (batch_size, total_len),
        mask_id,
        dtype=torch.long,
        device=prompt.device
    )
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
        x0, transfer_idx = get_transfer_index_vectorized(
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

            x0_blk, transfer_blk = get_transfer_index_vectorized(
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

def get_transfer_index_vectorized(
    logits: torch.Tensor,               # (B, S, V)
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,           # (B, S)
    x: torch.Tensor,                    # (B, total_len)
    num_transfer: Optional[torch.Tensor],
    threshold: Optional[float] = None,
    repetition_penalty: float = 1.0,
    penalty_context: Optional[torch.Tensor] = None,
    prompt_len: int = 0,
    mask_id: Optional[int] = None,
):
    B, S, V = logits.shape
    device = logits.device

    # ---- 1) Vectorized repetition penalty ----
    if repetition_penalty > 1.0 and penalty_context is not None:
        # get generated tokens (B, gen_len)
        gen = penalty_context[:, prompt_len:]
        valid = gen != mask_id
        # build a (B, V) mask of “seen” tokens
        counts = torch.zeros((B, V), device=device, dtype=torch.long)
        counts = counts.scatter_add(
            1,
            gen.masked_fill(~valid, 0),
            valid.to(torch.long)
        )
        seen = counts>0           # (B, V) bool
        # apply penalty only on those token‐channels
        logits = torch.where(
            seen[:, None, :],
            logits / repetition_penalty,
            logits
        )

    # ---- 2) Gumbel‐max sampling ----
    noise = add_gumbel_noise(logits, temperature)  # (B,S,V)
    x0    = noise.argmax(dim=-1)                   # (B,S)

    # ---- 3) Confidence scores ----
    if remasking == 'low_confidence':
        probs = F.softmax(logits.to(torch.float64), dim=-1)            # (B,S,V)
        conf  = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)         # (B,S)
    elif remasking == 'random':
        conf = torch.rand((B,S), device=device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    # mask out unchanged positions
    x0 = torch.where(mask_index, x0, x)
    conf = torch.where(mask_index, conf, -float('inf'))

    # ---- 4) Vectorized top‐k / threshold selection ----
    # sort confidences desc
    conf_vals, conf_idxs = conf.sort(dim=1, descending=True)  # both (B,S)
    # how many to take?
    if threshold is None:
        k = num_transfer if num_transfer.dim() == 1 else num_transfer.squeeze(1) # (B,)
        above_thr = torch.ones_like(conf_vals, dtype=torch.bool)
    else:
        # if threshold is set, we default k = full mask count, then drop below‐thr
        k = mask_index.sum(dim=1)          # (B,)
        above_thr = conf_vals >= threshold

    # build a mask on the *sorted* positions: take top‐k in each row
    ar = torch.arange(S, device=device).unsqueeze(0).expand(B, S)  # (B,S)
    sorted_mask = (ar < k.unsqueeze(1)) & above_thr               # (B,S)

    # scatter back into original order
    transfer = torch.zeros((B, S), dtype=torch.bool, device=device)
    transfer = transfer.scatter(1, conf_idxs, sorted_mask)

    return x0, transfer


# def main():
#     device = 'cuda'
#     model, tokenizer = init_model(
#         model_path    = "/home/hice1/jmoutahir3/scratch/LLada-Reasoning/llada_local_1.5",
#         adapter_path  = "/home/hice1/jmoutahir3/scratch/LLaDA_checkpoints/sft/final/final/sft_adapter",
#         load_lora     = True,
#         device        = device,
#     )
#     model = model.to(device)

#     # 1) Define your four prompts
#     prompts = [
#         "A number consists of two digits. The digit in the tens place is three times the digit in the units place. If you reverse the digits, the new number is 36 less than the original number. What is the original number?",
#         "Explain how a Kalman filter works in the context of object tracking.",
#         "Describe the process of fine‑tuning a CLIP model with LoRA on a small dataset.",
#         "What are the key benefits of using Yarn rotary embeddings over vanilla RoPE?",
#     ]

#     # Artificially make the batch size 8
#     prompts = prompts * 2  # Repeat to make batch size 8

#     # 2) Apply chat template to each, then batch‑tokenize (with padding)
#     wrapped = []
#     for p in prompts:
#         msg = [{"role":"user","content": p}]
#         wrapped.append(
#             tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
#         )
#     batch = tokenizer(
#         wrapped,
#         padding=True,
#         return_tensors="pt",
#     )
#     input_ids = batch["input_ids"].to(device)

#     # 3) Measure generation time
#     t0 = torch.cuda.Event(enable_timing=True)
#     t1 = torch.cuda.Event(enable_timing=True)
#     t0.record()
#     out, nfe = generate_with_dual_cache(
#         model,
#         input_ids,
#         steps=256,
#         gen_length=256,
#         block_length=16,
#         temperature=0.0,
#         remasking='low_confidence',
#         threshold=0.9,
#         repetition_penalty=1.2,
#     )
#     t1.record()
#     torch.cuda.synchronize()
#     elapsed = t0.elapsed_time(t1) / 1000.0
#     print(f"Generation time: {elapsed:.2f}s, NFE: {nfe}")

#     # 4) Decode and print each result
#     gens = tokenizer.batch_decode(
#         out[:, input_ids.shape[1]:],
#         skip_special_tokens=False
#     )
#     for i, text in enumerate(gens, start=1):
#         print(f"\n=== Prompt {i} ===\n{prompts[i-1]}\n\n→ Generation:\n{text}")

def main():
    device = 'cuda'
    model, tokenizer = init_model(
        model_path    = "/home/hice1/jmoutahir3/scratch/LLada-Reasoning/llada_local_1.5",
        adapter_path  = "/home/hice1/jmoutahir3/scratch/LLaDA_checkpoints/sft/final/final/sft_adapter",
        load_lora     = True,
        device        = device,
    )
    model = model.to(device)

    # single “template” prompt for warm‑up + experiments
    single = "A number consists of two digits. The digit in the tens place is three times the digit in the units place. If you reverse the digits, the new number is 36 less than the original number. What is the original number?"

    # --- 0) WARM‑UP PASS (cache priming only) ---
    warm = tokenizer.apply_chat_template(
        [{"role":"user","content": single}],
        add_generation_prompt=True,
        tokenize=False
    )
    warm_ids = tokenizer([warm], return_tensors="pt", padding=True)["input_ids"].to(device)
    # one quick call to populate all caches
    _ , _ = generate_with_dual_cache(
        model,
        warm_ids,
        steps=256,
        gen_length=256,
        block_length=16,
        temperature=0.0,
        remasking='low_confidence',
        threshold=0.9,
        repetition_penalty=1.2,
    )

    # --- 1) Now run the real timing loop ---
    batch_sizes = [1]#, 2, 4, 8, 16, 32, 64]
    print(f"{'batch':>5}  {'time (s)':>8}  {'NFE':>4}")
    print("-" * 24)

    for bs in batch_sizes:
        # build batch of size bs
        prompts = [single] * bs
        wrapped = [
            tokenizer.apply_chat_template(
                [{"role":"user","content":p}],
                add_generation_prompt=True,
                tokenize=False
            )
            for p in prompts
        ]
        enc = tokenizer(wrapped, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        # time it
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0.record()
        out, nfe = generate_with_dual_cache(
            model,
            input_ids,
            steps=256,
            gen_length=256,
            block_length=16,
            temperature=0.0,
            remasking='low_confidence',
            # threshold=0.01,
            repetition_penalty=1.2,
        )
        t1.record()
        torch.cuda.synchronize()
        elapsed = t0.elapsed_time(t1) / 1000.0

        # decode and print
        gens = tokenizer.batch_decode(
            out[:, input_ids.shape[1]:],
            skip_special_tokens=False
        )
        for i, text in enumerate(gens, start=1):
            print(f"\n=== Prompt {i} ===\n{prompts[i-1]}\n\n→ Generation:\n{text}")

        print(f"{bs:5d}  {elapsed:8.3f}  {nfe:4d}")


if __name__ == '__main__':
    main()