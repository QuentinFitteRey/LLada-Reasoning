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
    """
    Adds Gumbel noise to logits for stable and efficient categorical sampling.
    This is the corrected, numerically stable implementation of the Gumbel-Max trick.
    """
    if temperature == 0:
        return logits
    
    # G = -log(-log(U)) where U ~ Uniform(0, 1)
    noise = torch.rand_like(logits)
    # Add a small epsilon for numerical stability to prevent log(0)
    gumbel_noise = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
    
    # Scale logits by temperature and add noise. This is the correct application.
    return (logits / temperature) + gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    Precomputes the number of tokens to unmask at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    # This loop is fine for batch size 1. For larger batches, it could be vectorized.
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    """
    Selects which tokens to unmask based on confidence.
    This version is vectorized for improved performance with batching.
    """
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float32), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -torch.inf)

    # Vectorized top-k selection
    k = num_transfer_tokens.squeeze(-1)
    max_k = k.max()
    
    # Ensure we don't try to select more tokens than are masked
    num_masked = mask_index.sum(dim=-1)
    max_k = min(max_k, num_masked.min())
    k = torch.clamp(k, max=num_masked)

    if max_k == 0: # No tokens to transfer
        return x0, torch.zeros_like(x0, dtype=torch.bool)
        
    top_confidence, top_indices = torch.topk(confidence, k=max_k, dim=-1)
    
    # Create a mask for variable k values per batch item
    k_mask = torch.arange(max_k, device=x.device)[None, :] < k[:, None]
    
    # Use scatter to create the final boolean index tensor
    transfer_index = torch.zeros_like(confidence, dtype=torch.bool).scatter_(
        dim=-1, index=top_indices, src=k_mask
    )

    if threshold is not None:
        # Vectorized thresholding
        passes_threshold = top_confidence > threshold
        threshold_mask = k_mask & passes_threshold
        transfer_index.scatter_(dim=-1, index=top_indices, src=threshold_mask)

    return x0, transfer_index


# --- GENERATION FUNCTIONS ---

@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None):
    """
    Baseline semi-autoregressive generation without KV caching. Correct but inefficient.
    """
    prompt_len = prompt.shape[1]
    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt_len] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = block_start + block_length
        
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            
            mask_index[:, block_end:] = 0
            
            k = num_transfer_tokens[:, i] if threshold is None else None
            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, k, threshold)
            x[transfer_index] = x0[transfer_index]
            
            i += 1
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
    return x, nfe

@torch.no_grad()
def generate_with_dual_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                             remasking='low_confidence', mask_id=126336, threshold=None):
    """
    Efficient semi-autoregressive generation using a KV cache that is dynamically updated.
    This version includes optimizations for cache creation and token selection.
    This implementation assumes the model's forward pass is modified to handle 'replace_position'.
    """
    prompt_len = prompt.shape[1]
    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt_len] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0
    
    # --- Refinement 1: Efficient Initial Cache Creation ---
    # Initially, only compute and cache the prompt.
    output = model(prompt, use_cache=True)
    past_key_values = output.past_key_values
    nfe += 1
    
    for num_block in range(num_blocks):
        current_block_start_abs = prompt_len + num_block * block_length
        current_block_end_abs = current_block_start_abs + block_length

        # The part of 'x' we will be operating on in this block loop.
        # It contains the prompt and all previously generated blocks.
        context_and_current_block = x[:, :current_block_end_abs]

        block_mask_index = (context_and_current_block[:, current_block_start_abs:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        i = 0
        while True:
            # Check if the block is complete
            if (x[:, current_block_start_abs:current_block_end_abs] == mask_id).sum() == 0:
                break
            
            nfe += 1
            # The input to the model is now only the current block being generated
            input_ids_for_block = x[:, current_block_start_abs:current_block_end_abs]
            mask_index_in_block = (input_ids_for_block == mask_id)

            logits = model(
                input_ids_for_block,
                past_key_values=past_key_values,
                # We don't need to return a new cache from this call, just use the prefix cache
                use_cache=False 
            ).logits

            k = num_transfer_tokens[:, i] if threshold is None else None
            x0_block, transfer_index_block = get_transfer_index(
                logits, temperature, remasking, mask_index_in_block,
                input_ids_for_block, k, threshold
            )
            
            # Update the original tensor 'x' in the correct slice
            x_slice = x[:, current_block_start_abs:current_block_end_abs]
            x_slice[transfer_index_block] = x0_block[transfer_index_block]
            x[:, current_block_start_abs:current_block_end_abs] = x_slice
            
            i += 1
        
        # --- Refinement 1: Update cache after block is finalized ---
        if num_block < num_blocks - 1: # No need to update cache after the last block
            newly_finished_block = x[:, current_block_start_abs:current_block_end_abs]
            output = model(newly_finished_block, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            nfe += 1
            
    return x, nfe

def main():
    device = 'cuda'

    model, tokenizer = init_model(lora=True)
    model = model.to(device)
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. To do 8 km she is running "

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    # m = [{"role": "user", "content": prompt}, ]
    # prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    print("Running optimized generation with dual-cache strategy...")
    # time the generation process
    time_start = torch.cuda.Event(enable_timing=True)
    time_end = torch.cuda.Event(enable_timing=True)
    time_start.record()

    out, nfe = generate_with_dual_cache(
        model, input_ids, steps=128, gen_length=128,
        block_length=32, temperature=0.0, remasking='low_confidence'
    )
    time_end.record()
    torch.cuda.synchronize()  # Wait for the events to complete
    elapsed_time = time_start.elapsed_time(time_end) / 1000.0  # Convert ms to seconds
    print(f"Generation completed in {elapsed_time:.2f} seconds.")
    print(f"\nTotal Model Forward Passes (NFE): {nfe}")
    print("\nGenerated Answer:")
    generated_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    print(generated_text)


if __name__ == '__main__':
    main()