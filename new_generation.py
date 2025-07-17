# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
from typing import Literal
from transformers import AutoTokenizer, AutoModel
from init_model import init_model


def add_gumbel_noise(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature == 0:
        return logits
    noise = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
    return (logits / temperature) + gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

@torch.no_grad()
def generate_with_dual_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                              remasking='low_confidence', mask_id=126336, threshold=0.8):
    """
    Dual-cache generation: initial full-context priming of the cache, then iterative block-wise sampling
    with per-block key/value replacement.
    """
    # Create the full sequence buffer and compute constants
    total_len = prompt.shape[1] + gen_length
    x = torch.full((1, total_len), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, "steps must be divisible by number of blocks"
    sub_steps = steps // num_blocks

    nfe = 0
    # 1) Prime cache on full sequence once
    output = model(x, use_cache=True)
    past_key_values = output.past_key_values

    for block_idx in range(num_blocks):
        # Define global positions for this block
        start = prompt.shape[1] + block_idx * block_length
        end = start + block_length

        # Pre-compute transfer counts for this block
        mask_block = (x[:, start:end] == mask_id)
        num_trans = get_num_transfer_tokens(mask_block, sub_steps)

        # Initial fill for this block (first sub-step)
        mask_full = (x == mask_id)
        mask_full[:, end:] = False
        x0, transfer_idx = get_transfer_index(
            output.logits, temperature, remasking,
            mask_full, x,
            num_trans[:, 0] if threshold is None else None,
            threshold
        )
        x[transfer_idx] = x0[transfer_idx]
        nfe += 1

        # Iterative refinement within the block
        for step in range(1, sub_steps):
            nfe += 1
            # Extract only the block tokens for input
            block_x = x[:, start:end]
            block_mask = (block_x == mask_id)

            # Build full-length replace mask for cache: replace the entire block slice
            replace_mask = torch.zeros((1, total_len), dtype=torch.bool, device=model.device)
            replace_mask[:, start:end] = True

            # Forward slice through model with cache and replacement
            out = model(
                block_x,
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_mask
            )
            logits = out.logits
            past_key_values = out.past_key_values  # update cache

            # Select tokens and update x
            x0_blk, transfer_blk = get_transfer_index(
                logits, temperature, remasking,
                block_mask, block_x,
                num_trans[:, step] if threshold is None else None,
                threshold
            )
            # write back into global x
            x[:, start:end][transfer_blk] = x0_blk[transfer_blk]

            # If no masks left, break early
            if not block_mask.any():
                break

    return x, nfe


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(torch.gather(p, -1, x0.unsqueeze(-1)), -1)
    elif remasking == 'random':
        x0_p = torch.rand(x0.shape, device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, sel = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, sel] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, sel[k]] < threshold:
                    transfer_index[j, sel[k]] = False
    return x0, transfer_index


def main():
    device = 'cuda'
    model, tokenizer = init_model(
        model_path = "/home/hice1/jmoutahir3/scratch/LLaDA_checkpoints/merged_pretrained_model/merged_model_good_base",
        adapter_path = "/home/hice1/jmoutahir3/scratch/LLaDA_checkpoints/sft/new_weights/step-1400",
        load_lora = True,
        device  = "cuda"
    )
    model = model.to(device)

    use_thinking = True

    # Example reasoning question
    question = "You have a 3-gallon jug and a 5-gallon jug, neither of which has any measurement markings, and an unlimited water supply. Using only these two jugs, measure out exactly 4 gallons of water."

    thinking_mode = """You must think step by step and provide detailed thinking on the problem before giving the final answer.
        You must put your thinking process between <think> and </think> tags and then output the final answer with a summary of your thinking process.
        In your thinking process, this requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop a well-considered thinking process.
        """
    not_thinking_mode = """You are not required to have detailed thinking on the problem between <think> and </think> tags.
    You can provide a direct answer to the question without detailed thinking.
    You can still take steps to solve the problem, but you do not need to provide detailed thinking on the problem.
    """

    if use_thinking:
        prompt = f"<BOS><start_id>user<end_id>\n{thinking_mode}\n{question}<eot_id><start_id>assistant<end_id>\n"
    else:
        prompt = f"<BOS><start_id>user<end_id>\n{not_thinking_mode}\n{question}<eot_id><start_id>assistant<end_id>\n"

    input_ids = torch.tensor(tokenizer(prompt)['input_ids']).to(device).unsqueeze(0)

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    out, _ = generate_with_dual_cache(
        model, input_ids,
        steps=1024, gen_length=1024, block_length=64,
        temperature=0.0, remasking='low_confidence'
    )
    t1.record(); torch.cuda.synchronize()
    print(f"Generation time: {t0.elapsed_time(t1)/1000:.2f}s")

    text = tokenizer.batch_decode(
        out[:, input_ids.shape[1]:], skip_special_tokens=False
    )[0]
    print(text)

if __name__ == '__main__':
    main()