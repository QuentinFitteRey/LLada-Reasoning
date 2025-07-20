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
                             remasking='low_confidence', mask_id=126336, threshold=None, repetition_penalty=1.0, stop_token_id=None):
    """
    Dual-cache generation: initial full-context priming of the cache, then iterative block-wise sampling
    with per-block key/value replacement. Includes repetition penalty.
    """
    # Create the full sequence buffer and compute constants
    prompt_len = prompt.shape[1]
    total_len = prompt_len + gen_length
    x = torch.full((1, total_len), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_len] = prompt.clone()

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
        start = prompt_len + block_idx * block_length
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
            threshold,
            repetition_penalty=repetition_penalty,
            penalty_context=x,
            prompt_len=prompt_len,
            mask_id=mask_id
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
                threshold,
                repetition_penalty=repetition_penalty,
                penalty_context=x, # Use the full 'x' for penalty context
                prompt_len=prompt_len,
                mask_id=mask_id
            )
            # write back into global x
            x[:, start:end][transfer_blk] = x0_blk[transfer_blk]

            # If no masks left, break early
            if not block_mask.any():
                break
            
        if stop_token_id is not None and (x[:, start:end] == stop_token_id).any():
            print(f"Stop token {stop_token_id} encountered, stopping early.")
            pos = (x[:, start:end] == stop_token_id).nonzero(as_tuple=True)[1]
            x[:, ]
            break

    return x, nfe


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None,
                       repetition_penalty=1.0, penalty_context=None, prompt_len=0, mask_id=None):
    """
    Applies repetition penalty, selects new tokens, and determines which indices to update.
    """
    # Apply repetition penalty to logits
    if repetition_penalty > 1.0 and penalty_context is not None:
        # Use the provided penalty_context (the full sequence) to find repeated tokens
        for i in range(logits.shape[0]): # Iterate over batch
            # Slice the context to get only generated tokens (after the prompt)
            generated_slice = penalty_context[i, prompt_len:]
            
            # Exclude the mask token from being penalized
            if mask_id is not None:
                generated_slice = generated_slice[generated_slice != mask_id]
            
            unique_generated_tokens = torch.unique(generated_slice)
            
            if unique_generated_tokens.numel() > 0:
                # Apply penalty by dividing the logits of repeated tokens.
                # This makes them less likely to be chosen again.
                logits[i, :, unique_generated_tokens] /= repetition_penalty

    # Sample new tokens
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    # Determine confidence for re-masking
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(torch.gather(p, -1, x0.unsqueeze(-1)), -1)
    elif remasking == 'random':
        x0_p = torch.rand(x0.shape, device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    # Select which tokens to transfer based on confidence
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
    model, tokenizer = init_model(lora=True)
    model = model.to(device)
    think = False

    stop_token_id= tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if think:
        prompt= "You must perform a detailed, step-by-step thinking process to solve the problem. Your thinking should be a comprehensive cycle of analysis, exploration, and self-correction. Engage in reflection, back-tracing to refine errors, and iteration to develop a well-considered path to the solution. Put this entire process between <think> and </think> tags. \nAfter the closing </think> tag, present your final answer. Your answer should begin with the conclusion, followed by a brief summary that explains how you arrived at it by referencing the key steps from your thinking process.\n"

    else:
        prompt = "You are not required to have detailed thinking on the problem between <think> and </think> tags. \nYou can provide a direct answer to the question without detailed thinking. \nYou can still take steps to solve the problem, but you do not need to provide detailed thinking on the problem.\n"
#     prompt += """
#         Here's a good prompt for the Zebra Puzzle, including a suggested method for solving it:

# "I need help solving a classic logic puzzle, often called the 'Zebra Puzzle' or 'Einstein's Riddle'. The goal is to figure out who owns the zebra and who drinks water.

# Here are the clues:

# There are five houses in a row, each with a different color.

# In each house lives a person of a different nationality.

# Each person drinks a different beverage, smokes a different brand of cigarettes, and keeps a different pet.

# The Brit lives in the red house.

# The Swede keeps dogs as pets.

# The Dane drinks tea.

# The green house is immediately to the left of the white house.

# The green house's owner drinks coffee.

# The person who smokes Pall Mall keeps birds.

# The owner of the yellow house smokes Dunhill.

# The man living in the center house drinks milk.

# The Norwegian lives in the first house.

# The man who smokes Blend lives next to the man with cats.

# The man who keeps horses lives next to the man who smokes Dunhill.

# The owner who smokes Blue Master drinks beer.

# The German smokes Prince.

# The Norwegian lives next to the blue house.

# The man who smokes Blend has a neighbor who drinks water.

# Suggested Method for Solving:

# To solve this, I recommend using a grid-based approach. Create a table with 5 rows (for the houses 1 through 5) and columns for each attribute: House Number, Color, Nationality, Drink, Cigarettes, and Pet.

# Start by placing the most definitive clues first (e.g., "The Norwegian lives in the first house" or "The man living in the center house drinks milk"). Then, use relative clues (e.g., "The green house is immediately to the left of the white house" or "lives next to") to deduce further placements. Systematically go through each clue, updating your grid and identifying new connections. It's often helpful to list out all possibilities for an attribute and then eliminate them as you gain more information.

# Please provide a detailed, step-by-step deduction process, showing how each clue helps fill in the grid. Finally, clearly state who owns the zebra and who drinks water.
#     """
#     prompt += """
#     You have two unmarked jugs—one holds 3 gal, the other 5 gal—and an unlimited water supply. Your goal is to end up with exactly 4 gal in one of the jugs using **exactly six moves**. A move is defined as one of:

#   1. Fill a jug completely from the supply.  
#   2. Empty a jug completely onto the ground.  
#   3. Pour from one jug to the other until **either** the source is empty **or** the destination is full.

#     You may **not** partially empty a jug, nor exceed a jug's capacity. If your sequence exceeds six moves, discard it and start over. PLEASE track your moves and the state of the jugs at each step.
#     """

    prompt += "A number consists of two digits. The digit in the tens place is three times the digit in the units place. If you reverse the digits, the new number is 36 less than the original number. What is the original number? You have to ouput the answer at the end even if you are not sure about the answer. \n"
#     prompt += """
#     Given the following question and four candidate answers (A, B, C and D), choose the best answer.
# Question: Let p = (1, 2, 5, 4)(2, 3) in S_5 . Find the index of <p> in S_5.
# A. 8
# B. 2
# C. 24
# D. 120
# Your response should end with "The best answer is [the_answer_letter]" where the [the_answer_letter] is one of A, B, C or D.
#     """
    print(f"Prompt: {prompt}")
    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    out, _ = generate_with_dual_cache(
        model, input_ids,
        steps=2048*2, gen_length=2048*2, block_length=128,
        temperature=0.0, remasking='low_confidence', threshold=0.9,
        repetition_penalty=1.2, # Set penalty > 1.0 to discourage repetition
        stop_token_id=stop_token_id
    )
    t1.record(); torch.cuda.synchronize()
    print(f"Generation time: {t0.elapsed_time(t1)/1000:.2f}s")

    text = tokenizer.batch_decode(
        out[:, input_ids.shape[1]:], skip_special_tokens=False
    )[0]
    print(text)

if __name__ == '__main__':
    main()